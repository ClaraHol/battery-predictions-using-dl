"""Extract three 1C reference-discharge curves for a Sony VTC5A cell.

For a cell, provide all its CU_*.mat and CYC_*.mat files. These are indexed by
check-up / cycling-interval number and joined in the sequence:
    CU000 (EFC=0), CYC01, CU001, CYC02, CU002, ...

EFC counts both charging and discharging throughput: (Q_ch+Q_dis)/(2*2.5 Ah).
We exclude the preliminary CU000 throughput from the ageing EFC origin, but
include throughput of subsequent check-ups. CU EFC is recorded at CU *start*.

Returns (curves DataFrame, metadata dict), as in the VTC6 extractor.
No parquet writing is performed here; the parent batch script handles that.
"""

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat

Q_NOM_AH = 2.5
TARGET_EFC = 50.0
REFERENCE_DISCHARGE_A = 2.5


def _file_number(path: Path, kind: str) -> int:
    """Read stage index from filename, falling back to the directory name."""
    if kind == "CU":
        pattern = r"(?:^|_)CU_(?:cyc|cal|dyn|cyclic|calendar|dynamic)_(\d+)(?:_|\.|$)"
        folder_pattern = r"^CU(\d+)$"
    else:
        pattern = r"(?:^|_)Cyc(\d+)(?:_|\.|$)"
        folder_pattern = r"^Cyc(\d+)$"
    m = re.search(pattern, path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    for part in reversed(path.parts[:-1]):
        m = re.fullmatch(folder_pattern, part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot identify {kind} interval number: {path}")


def _index_files(paths: Iterable[str | Path], kind: str) -> dict[int, Path]:
    files = {}
    for path in map(Path, paths):
        number = _file_number(path, kind)
        if number in files:
            raise ValueError(f"Duplicate {kind}{number:03d}: {files[number]} and {path}")
        files[number] = path
    return files


def _read_dataset(path: Path):
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)["Dataset"]
    except (KeyError, OSError, ValueError, NotImplementedError) as e:
        raise ValueError(f"Cannot read Dataset from {path}: {e}") from e
    return data


def _array(data, name: str) -> np.ndarray:
    if not hasattr(data, name):
        raise ValueError(f"Dataset missing field {name}")
    values = np.asarray(getattr(data, name), dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} array")
    return values


def _cycling_throughput_ah(data) -> float:
    """Sum reset-safe increments of separate charge and discharge counters."""
    q_ch = _array(data, "AhChSet")
    q_dis = _array(data, "AhDisSet")
    if len(q_ch) != len(q_dis):
        raise ValueError("AhChSet and AhDisSet have different lengths")
    throughput = (
        np.maximum(np.diff(q_ch), 0).sum()
        + np.maximum(np.diff(q_dis), 0).sum()
    )
    return float(throughput)


def _checkup_throughput_ah(data) -> float:
    """Ah is a signed, local check-up counter. Both signs count towards EFC."""
    q = _array(data, "Ah")
    return float(np.abs(np.diff(q)).sum())


def _reference_discharge(data, file_path: Path, checkup_number: int) -> pd.DataFrame:
    """Find *complete* near-2.5 A discharge steps; prefer the second one.

    The observed CU000/CU001/CU002 sequences have two full discharges at
    Line 20 and Line 27. Search by behavior, not by a hard-coded Line number,
    so other cell protocols can be diagnosed or handled consistently.
    """
    time_h = _array(data, "Time")
    voltage = _array(data, "U")
    current = _array(data, "I")
    line = np.asarray(getattr(data, "Line")).reshape(-1)
    command = np.asarray(getattr(data, "Command")).reshape(-1)
    n = len(time_h)
    if not (len(voltage) == len(current) == len(line) == len(command) == n):
        raise ValueError(f"Inconsistent CU field lengths: {file_path}")
    if np.any(np.diff(time_h) < 0):
        raise ValueError(f"Nonmonotone time in {file_path}")

    # Every contiguous Line/Command run is one candidate experimental step.
    boundaries = np.r_[
        0, np.flatnonzero((line[1:] != line[:-1]) | (command[1:] != command[:-1])) + 1, n
    ]
    candidates = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if str(command[start]).lower() != "discharge" or end - start < 75:
            continue
        v = voltage[start:end]
        i = current[start:end]
        t = time_h[start:end]
        if not (0.55 <= t[-1] - t[0] <= 1.30):
            continue
        # Raw TUM convention is negative during discharge.
        active = i < -0.8 * REFERENCE_DISCHARGE_A
        if active.mean() < 0.95:
            continue
        if abs(np.median(i[active]) + REFERENCE_DISCHARGE_A) > 0.15:
            continue
        if not (v[0] >= 4.0 and v[-1] <= 2.65):
            continue
        if (t[-1] - t[0]) * REFERENCE_DISCHARGE_A < 1.65:
            continue
        candidates.append((int(start), int(end), int(line[start])))

    if len(candidates) < 2:
        raise ValueError(
            f"CU{checkup_number:03d}: expected two full 1C discharges, "
            f"found {len(candidates)} in {file_path.name}"
        )
    # The reference protocol has two full discharges; consistently take
    # the second after the additional conditioning charge.
    chosen = candidates[1]
    start, end, line_num = chosen
    t_s = (time_h[start:end] - time_h[start]) * 3600.0
    curve = pd.DataFrame({
        "time_s": t_s,
        "voltage_V": voltage[start:end],
        "current_A": np.abs(current[start:end]),
    })
    if np.any(np.diff(curve["time_s"].to_numpy()) <= 0):
        # Occasional duplicate tester stamps need removing without sorting
        # (sorting could hide a broken time series).
        curve = curve.loc[curve["time_s"].diff().fillna(1).gt(0)].reset_index(drop=True)
    if len(curve) < 75:
        raise ValueError(f"CU{checkup_number:03d}: too few time points after cleanup")
    curve.attrs.update({
        "checkup_number": checkup_number,
        "source_file": file_path.name,
        "reference_line": line_num,
        "raw_start_idx": start,
        "raw_end_idx_exclusive": end,
        "n_eligible_discharges": len(candidates),
    })
    return curve


def extract_curve_vtc5a(
    cu_files: Iterable[str | Path],
    cyc_files: Iterable[str | Path],
    *,
    target_efc: float = TARGET_EFC,
    require_cyc_intervals: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return 3 VTC5A RPT curves plus metadata for one cell.

    Caller must pass files belonging to ONE cell (identified by parent script).
    For normally cycling cells, missing CYC intervals are errors. For calendar
    ageing without cycling intervals set require_cyc_intervals=False.

    Fails rather than extrapolating when EFC50 is unbracketed.
    """
    cu_index = _index_files(cu_files, "CU")
    cyc_index = _index_files(cyc_files, "CYC")
    if 0 not in cu_index:
        raise ValueError("Missing beginning-of-life CU000")
    if len(cu_index) < 2:
        raise ValueError("Too few check-ups")

    running_efc = 0.0
    rpt_efcs: dict[int, float] = {0: 0.0}
    # Only need to read the ageing history until the first CU after EFC50.
    for cu_num in sorted(n for n in cu_index if n > 0):
        previous_cu_num = max(n for n in rpt_efcs if n < cu_num)
        for k in range(previous_cu_num + 1, cu_num + 1):
            if k not in cyc_index:
                if require_cyc_intervals:
                    raise ValueError(f"Missing CYC{k:02d} before CU{cu_num:03d}")
            else:
                running_efc += (
                    _cycling_throughput_ah(_read_dataset(cyc_index[k]))
                    / (2 * Q_NOM_AH)
                )
            # A check-up's throughput is counted *after* assigning its EFC.
            # The intermediate check-up is required if it precedes the target.
            if k < cu_num:
                if k not in cu_index:
                    raise ValueError(f"Missing CU{k:03d} in chronology")
                running_efc += (
                    _checkup_throughput_ah(_read_dataset(cu_index[k]))
                    / (2 * Q_NOM_AH)
                )
        rpt_efcs[cu_num] = running_efc
        if running_efc > target_efc:
            break
        # Add throughput of this check-up before the next ageing interval.
        running_efc += _checkup_throughput_ah(_read_dataset(cu_index[cu_num])) / (2 * Q_NOM_AH)

    ordered = sorted(rpt_efcs)
    if any(rpt_efcs[b] <= rpt_efcs[a] for a, b in zip(ordered[:-1], ordered[1:])):
        raise ValueError("Non-increasing RPT EFC positions")
    efcs = np.array([rpt_efcs[k] for k in ordered])
    upper_pos = int(np.searchsorted(efcs, target_efc, side="right"))
    if upper_pos == 0 or upper_pos == len(efcs):
        raise ValueError(f"EFC{target_efc:g} is not strictly bracketed")
    lower_num, upper_num = ordered[upper_pos-1], ordered[upper_pos]
    low_efc, high_efc = rpt_efcs[lower_num], rpt_efcs[upper_num]
    if not (low_efc < target_efc < high_efc):
        raise ValueError(f"EFC{target_efc:g} exactly matches a check-up or is unbracketed")

    w_upper = (target_efc - low_efc) / (high_efc - low_efc)
    selection = {"efc0": 0, "efc50_lower": lower_num, "efc50_upper": upper_num}
    frames = []
    info = {}
    for label, num in selection.items():
        curve = _reference_discharge(_read_dataset(cu_index[num]), cu_index[num], num)
        info[label] = dict(curve.attrs)
        curve["curve_type"] = label
        curve["rpt_efc"] = float(rpt_efcs[num])
        frames.append(curve)

    metadata = {
        "efc0": float(rpt_efcs[0]),
        "efc50_lower": float(low_efc),
        "efc50_upper": float(high_efc),
        "weight_lower": float(1 - w_upper),
        "weight_upper": float(w_upper),
        "direction": "discharge",
        "n_rpts": len(cu_index),
        "efc_definition": "(charge_Ah+discharge_Ah)/(2*2.5Ah); excludes CU000 throughput",
    }
    for label, details in info.items():
        metadata[f"{label}_cu_number"] = details["checkup_number"]
        metadata[f"{label}_source_file"] = details["source_file"]
        metadata[f"{label}_line"] = details["reference_line"]
        metadata[f"{label}_raw_start_idx"] = details["raw_start_idx"]
        metadata[f"{label}_raw_end_idx_exclusive"] = details["raw_end_idx_exclusive"]
        metadata[f"{label}_eligible_discharges"] = details["n_eligible_discharges"]

    return pd.concat(frames, ignore_index=True), metadata
