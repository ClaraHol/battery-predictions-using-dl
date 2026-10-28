"""Robust VTC5A RPT extraction with lifetime scanning.

Key policy:
- EFC0/EFC50 feature extraction is strict: required stages must be readable and
  EFC50 must be genuinely bracketed.
- Lifetime scanning is best-effort after EFC50 is secured.  A late duplicate,
  missing stage, or nonstandard CU does not discard valid feature curves; it
  right-censors the available lifetime information and records the reason.
- CYC stages split as CycNN, CycNNb, CycNNc are accumulated in suffix order.
  Ambiguous same-suffix duplicates are never silently merged.
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
LIFETIME_SOH_THRESHOLD = 0.90
ACTIVE_CURRENT_A = 0.05


class StageAmbiguityError(ValueError):
    pass


class LifetimeScanStop(RuntimeError):
    pass


def _round_to_step(value: float, step: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.floor(value / step + 0.5) * step)


def _file_number(path: Path, kind: str) -> int:
    if kind == "CU":
        pattern = r"(?:^|_)CU_(?:cyc|cal|dyn|cyclic|calendar|dynamic)_(\d+)(?:[a-z])?(?:_|\.|$)"
        folder_pattern = r"^CU(\d+)(?:_(?:cyc|cal|dyn))?$"
    else:
        pattern = r"(?:^|_)Cyc(\d+)(?:[a-z])?(?:_|\.|$)"
        folder_pattern = r"^Cyc(\d+)(?:[a-z])?$"
    m = re.search(pattern, path.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    for part in reversed(path.parts[:-1]):
        m = re.fullmatch(folder_pattern, part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot identify {kind} interval number: {path}")


def _stage_suffix(path: Path, kind: str) -> str:
    """Return '', 'b', 'c', ... when filename explicitly encodes a stage part."""
    name = path.name
    if kind == "CYC":
        m = re.search(r"(?:^|_)Cyc\d+([a-z]?)(?:_|\.|$)", name, flags=re.IGNORECASE)
    else:
        m = re.search(
            r"(?:^|_)CU_(?:cyc|cal|dyn|cyclic|calendar|dynamic)_\d+([a-z]?)(?:_|\.|$)",
            name,
            flags=re.IGNORECASE,
        )
    return (m.group(1).lower() if m else "")


def _index_files(paths: Iterable[str | Path], kind: str) -> dict[int, list[Path]]:
    files: dict[int, list[Path]] = {}
    for path in map(Path, paths):
        number = _file_number(path, kind)
        files.setdefault(number, []).append(path)
    for number in files:
        files[number] = sorted(files[number], key=lambda p: (p.name.lower(), str(p)))
    return files


def _read_dataset(path: Path):
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)["Dataset"]
    except (KeyError, OSError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"Cannot read Dataset from {path}: {exc}") from exc


def _array(data, name: str) -> np.ndarray:
    if not hasattr(data, name):
        raise ValueError(f"Dataset missing field {name}")
    values = np.asarray(getattr(data, name), dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} array")
    return values


def _cycling_throughput_ah(data) -> float:
    q_ch = _array(data, "AhChSet")
    q_dis = _array(data, "AhDisSet")
    if len(q_ch) != len(q_dis):
        raise ValueError("AhChSet and AhDisSet have different lengths")
    return float(np.maximum(np.diff(q_ch), 0).sum() + np.maximum(np.diff(q_dis), 0).sum())


def _checkup_throughput_ah(data) -> float:
    q = _array(data, "Ah")
    return float(np.abs(np.diff(q)).sum())


def _temperature_from_path(path: Path) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*deg", path.name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def _cycling_condition_contribution(data) -> dict:
    time_h = _array(data, "Time")
    current = _array(data, "I")
    if len(time_h) != len(current):
        raise ValueError("CYC Time and I have different lengths")
    dt_h = np.diff(time_h)
    if np.any(dt_h < 0):
        raise ValueError("Nonmonotone CYC Time")
    i_mid = 0.5 * (current[:-1] + current[1:])
    valid = dt_h > 0
    charge = valid & (i_mid > ACTIVE_CURRENT_A)
    discharge = valid & (i_mid < -ACTIVE_CURRENT_A)
    return {
        "charge_Ah": float(np.sum(i_mid[charge] * dt_h[charge])),
        "charge_active_h": float(np.sum(dt_h[charge])),
        "discharge_Ah": float(np.sum(-i_mid[discharge] * dt_h[discharge])),
        "discharge_active_h": float(np.sum(dt_h[discharge])),
    }


def _reference_discharge(data, file_path: Path, checkup_number: int) -> pd.DataFrame:
    """Select the second complete ~1C discharge; use behavior as fallback.

    Some CU variants use a different Command encoding.  We first use the
    command-aware segmentation, then fall back to Line-only segmentation while
    keeping the same current/voltage/duration requirements.
    """
    time_h = _array(data, "Time")
    voltage = _array(data, "U")
    current = _array(data, "I")
    if not hasattr(data, "Line"):
        raise ValueError("Dataset missing field Line")
    line = np.asarray(getattr(data, "Line")).reshape(-1)
    command = np.asarray(getattr(data, "Command")).reshape(-1) if hasattr(data, "Command") else None
    n = len(time_h)
    if not (len(voltage) == len(current) == len(line) == n):
        raise ValueError(f"Inconsistent CU field lengths: {file_path}")
    if command is not None and len(command) != n:
        command = None
    if np.any(np.diff(time_h) < 0):
        raise ValueError(f"Nonmonotone time in {file_path}")

    def find_candidates(use_command: bool):
        change = line[1:] != line[:-1]
        if use_command and command is not None:
            change = change | (command[1:] != command[:-1])
        boundaries = np.r_[0, np.flatnonzero(change) + 1, n]
        candidates = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start < 75:
                continue
            if use_command and command is not None:
                cmd = str(command[start]).strip().lower()
                if "dis" not in cmd:
                    continue
            v = voltage[start:end]
            i = current[start:end]
            t = time_h[start:end]
            duration = t[-1] - t[0]
            if not (0.55 <= duration <= 1.30):
                continue
            active = i < -0.8 * REFERENCE_DISCHARGE_A
            if active.mean() < 0.95:
                continue
            if abs(np.median(i[active]) + REFERENCE_DISCHARGE_A) > 0.15:
                continue
            if not (v[0] >= 4.0 and v[-1] <= 2.65):
                continue
            if duration * REFERENCE_DISCHARGE_A < 1.65:
                continue
            candidates.append((int(start), int(end), int(line[start])))
        return candidates

    candidates = find_candidates(True)
    if len(candidates) < 2:
        candidates = find_candidates(False)
    if len(candidates) < 2:
        raise ValueError(
            f"CU{checkup_number:03d}: expected two full 1C discharges, "
            f"found {len(candidates)} in {file_path.name}"
        )

    start, end, line_num = candidates[1]
    t_s = (time_h[start:end] - time_h[start]) * 3600.0
    curve = pd.DataFrame({
        "time_s": t_s,
        "voltage_V": voltage[start:end],
        "current_A": np.abs(current[start:end]),
    })
    if np.any(np.diff(curve["time_s"].to_numpy()) <= 0):
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


def _curve_capacity_ah(curve: pd.DataFrame) -> float:
    t = curve["time_s"].to_numpy(float)
    i = curve["current_A"].to_numpy(float)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(i, t) / 3600.0)


def _interpolate_lifetime(prev_row: dict, curr_row: dict, threshold: float) -> float:
    e0, e1 = float(prev_row["rpt_efc"]), float(curr_row["rpt_efc"])
    s0, s1 = float(prev_row["soh"]), float(curr_row["soh"])
    if not (s0 > threshold >= s1):
        raise ValueError("Lifetime interpolation rows do not bracket SOH threshold")
    if np.isclose(s0, s1):
        return e1
    w = (threshold - s0) / (s1 - s0)
    return float(e0 + w * (e1 - e0))


def extract_curve_vtc5a(
    cu_files: Iterable[str | Path],
    cyc_files: Iterable[str | Path],
    *,
    target_efc: float = TARGET_EFC,
    lifetime_soh_threshold: float = LIFETIME_SOH_THRESHOLD,
    require_cyc_intervals: bool = True,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if not 0 < lifetime_soh_threshold < 1:
        raise ValueError("lifetime_soh_threshold must lie in (0, 1)")
    if not require_cyc_intervals:
        raise ValueError("EFC50 feature extraction is not applicable to calendar/CU-only cells")

    cu_index = _index_files(cu_files, "CU")
    cyc_index = _index_files(cyc_files, "CYC")
    if 0 not in cu_index:
        raise ValueError("Missing beginning-of-life CU000")

    cu_choice: dict[int, Path] = {}
    cu_data_cache = {}
    curve_cache: dict[int, pd.DataFrame] = {}

    def choose_cu(num: int, strict: bool = True):
        if num in cu_choice:
            return cu_choice[num]
        candidates = cu_index.get(num, [])
        if not candidates:
            raise ValueError(f"Missing CU{num:03d}")
        valid = []
        errors = []
        for path in candidates:
            try:
                data = _read_dataset(path)
                curve = _reference_discharge(data, path, num)
                valid.append((path, data, curve))
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
        if len(valid) == 1:
            path, data, curve = valid[0]
            cu_choice[num] = path
            cu_data_cache[num] = data
            curve_cache[num] = curve
            return path
        if len(valid) == 0:
            msg = f"CU{num:03d}: no valid reference-discharge file among {len(candidates)} candidate(s)"
            if errors:
                msg += "; " + " | ".join(errors[:3])
            raise ValueError(msg)
        names = ", ".join(v[0].name for v in valid)
        raise StageAmbiguityError(f"Ambiguous CU{num:03d}: multiple valid files: {names}")

    def get_cu(num: int):
        choose_cu(num)
        return cu_data_cache[num]

    def get_reference(num: int):
        choose_cu(num)
        return curve_cache[num]

    def get_cyc_parts(num: int):
        paths = cyc_index.get(num, [])
        if not paths:
            raise ValueError(f"Missing CYC{num:02d}")
        if len(paths) == 1:
            return paths
        suffixes = [_stage_suffix(p, "CYC") for p in paths]
        # A base segment plus b/c/... segments is a recognized split interval.
        if len(set(suffixes)) == len(suffixes) and any(s for s in suffixes):
            def split_order(path):
                suffix = _stage_suffix(path, "CYC")
                return 0 if suffix == "" else ord(suffix[0]) - 96
            return sorted(paths, key=split_order)
        raise StageAmbiguityError(
            f"Ambiguous CYC{num:03d}: " + ", ".join(p.name for p in paths)
        )

    curve0 = get_reference(0)
    initial_capacity = _curve_capacity_ah(curve0)
    if not 1.5 <= initial_capacity <= 3.1:
        raise ValueError(f"Implausible CU000 reference capacity: {initial_capacity:.3f} Ah")

    rpt_rows = [{
        "cu_number": 0,
        "rpt_efc": 0.0,
        "capacity_Ah": initial_capacity,
        "soh": 1.0,
        "source_file": cu_choice[0].name,
        "reference_line": curve0.attrs["reference_line"],
        "n_eligible_discharges": curve0.attrs["n_eligible_discharges"],
    }]

    running_efc = 0.0
    efc50_lower_num = efc50_upper_num = None
    lifetime_lower_num = lifetime_upper_num = None
    cycle_life_efc = np.nan
    lifetime_status = "scanning"
    lifetime_censor_reason = ""

    condition_charge_Ah = condition_charge_h = 0.0
    condition_discharge_Ah = condition_discharge_h = 0.0
    condition_temperatures = []
    condition_cyc_numbers = []

    max_stage = max(max(cu_index, default=0), max(cyc_index, default=0))
    stage = 1
    while stage <= max_stage:
        # Once EFC50 is available, any later data-format problem only censors lifetime.
        post_feature = efc50_upper_num is not None
        try:
            # Add throughput of the previous check-up before the next ageing stage.
            if stage > 1:
                running_efc += _checkup_throughput_ah(get_cu(stage - 1)) / (2 * Q_NOM_AH)

            cyc_paths = get_cyc_parts(stage)
            for cyc_path in cyc_paths:
                cyc_data = _read_dataset(cyc_path)
                if running_efc <= target_efc:
                    contribution = _cycling_condition_contribution(cyc_data)
                    condition_charge_Ah += contribution["charge_Ah"]
                    condition_charge_h += contribution["charge_active_h"]
                    condition_discharge_Ah += contribution["discharge_Ah"]
                    condition_discharge_h += contribution["discharge_active_h"]
                    temp = _temperature_from_path(cyc_path)
                    if np.isfinite(temp):
                        condition_temperatures.append(float(temp))
                    condition_cyc_numbers.append(stage)
                running_efc += _cycling_throughput_ah(cyc_data) / (2 * Q_NOM_AH)

            curve = get_reference(stage)
            capacity = _curve_capacity_ah(curve)
            soh = capacity / initial_capacity
            row = {
                "cu_number": stage,
                "rpt_efc": float(running_efc),
                "capacity_Ah": float(capacity),
                "soh": float(soh),
                "source_file": cu_choice[stage].name,
                "reference_line": curve.attrs["reference_line"],
                "n_eligible_discharges": curve.attrs["n_eligible_discharges"],
            }
            previous_row = rpt_rows[-1]
            if row["rpt_efc"] <= previous_row["rpt_efc"]:
                raise ValueError(
                    f"Non-increasing RPT EFC positions: CU{previous_row['cu_number']:03d}="
                    f"{previous_row['rpt_efc']:.3f}, CU{stage:03d}={row['rpt_efc']:.3f}"
                )
            rpt_rows.append(row)

            if efc50_upper_num is None and previous_row["rpt_efc"] < target_efc < row["rpt_efc"]:
                efc50_lower_num = int(previous_row["cu_number"])
                efc50_upper_num = int(stage)

            if lifetime_upper_num is None and previous_row["soh"] > lifetime_soh_threshold >= row["soh"]:
                lifetime_lower_num = int(previous_row["cu_number"])
                lifetime_upper_num = int(stage)
                cycle_life_efc = _interpolate_lifetime(previous_row, row, lifetime_soh_threshold)
                lifetime_status = "observed_crossing"

            if efc50_upper_num is not None and lifetime_upper_num is not None:
                break

        except (ValueError, StageAmbiguityError) as exc:
            if not post_feature:
                raise
            lifetime_status = "censored_data_gap"
            lifetime_censor_reason = str(exc)
            break

        stage += 1

    if efc50_upper_num is None:
        raise ValueError(f"EFC{target_efc:g} is not strictly bracketed")

    rpt_summary = pd.DataFrame(rpt_rows)
    if lifetime_upper_num is None:
        cycle_life_censored = True
        cycle_life_efc = float(rpt_summary.iloc[-1]["rpt_efc"])
        if lifetime_status == "scanning":
            lifetime_status = "censored_end_of_data"
            lifetime_censor_reason = "SOH threshold not reached in available valid RPTs"
    else:
        cycle_life_censored = False

    low_row = rpt_summary.loc[rpt_summary["cu_number"].eq(efc50_lower_num)].iloc[0]
    high_row = rpt_summary.loc[rpt_summary["cu_number"].eq(efc50_upper_num)].iloc[0]
    low_efc, high_efc = float(low_row["rpt_efc"]), float(high_row["rpt_efc"])
    w_upper = (target_efc - low_efc) / (high_efc - low_efc)

    selection = {"efc0": 0, "efc50_lower": efc50_lower_num, "efc50_upper": efc50_upper_num}
    frames, info = [], {}
    for label, num in selection.items():
        curve = get_reference(num).copy()
        info[label] = dict(curve.attrs)
        curve["curve_type"] = label
        curve["rpt_efc"] = float(rpt_summary.loc[rpt_summary["cu_number"].eq(num), "rpt_efc"].iloc[0])
        frames.append(curve)

    if condition_temperatures:
        unique_temperatures = np.unique(np.asarray(condition_temperatures, float))
        temperature_raw = float(np.mean(unique_temperatures))
        temperature_rounded = _round_to_step(temperature_raw, 5.0)
        temperature_values = ",".join(f"{x:g}" for x in unique_temperatures)
    else:
        temperature_raw = temperature_rounded = np.nan
        temperature_values = ""

    charge_current_A = condition_charge_Ah / condition_charge_h if condition_charge_h > 0 else np.nan
    discharge_current_A = condition_discharge_Ah / condition_discharge_h if condition_discharge_h > 0 else np.nan
    charge_c_rate_raw = charge_current_A / Q_NOM_AH if np.isfinite(charge_current_A) else np.nan
    discharge_c_rate_raw = discharge_current_A / Q_NOM_AH if np.isfinite(discharge_current_A) else np.nan

    last_rpt = rpt_summary.iloc[-1]
    metadata = {
        "efc0": 0.0,
        "efc50_lower": low_efc,
        "efc50_upper": high_efc,
        "weight_lower": float(1 - w_upper),
        "weight_upper": float(w_upper),
        "direction": "discharge",
        "n_rpts_available": int(sum(len(v) > 0 for v in cu_index.values())),
        "n_rpts_scanned": len(rpt_summary),
        "efc_definition": "(charge_Ah+discharge_Ah)/(2*2.5Ah); excludes CU000 throughput",
        "initial_rpt_capacity_Ah": float(initial_capacity),
        "last_rpt_capacity_Ah": float(last_rpt["capacity_Ah"]),
        "last_rpt_soh": float(last_rpt["soh"]),
        "last_rpt_efc": float(last_rpt["rpt_efc"]),
        "cycle_life_efc": float(cycle_life_efc),
        "cycle_life_censored": bool(cycle_life_censored),
        "cycle_life_soh_threshold": float(lifetime_soh_threshold),
        "cycle_life_lower_cu_number": float(lifetime_lower_num) if lifetime_lower_num is not None else np.nan,
        "cycle_life_upper_cu_number": float(lifetime_upper_num) if lifetime_upper_num is not None else np.nan,
        "lifetime_status": lifetime_status,
        "lifetime_censor_reason": lifetime_censor_reason,
        "temperature_C_raw": temperature_raw,
        "temperature_C": temperature_rounded,
        "temperature_values_C": temperature_values,
        "temperature_method": "mean of distinct CYC filename '*deg' set-points through EFC50; rounded 5C",
        "charge_c_rate_raw": float(charge_c_rate_raw),
        "charge_c_rate": _round_to_step(charge_c_rate_raw, 0.5),
        "discharge_c_rate_raw": float(discharge_c_rate_raw),
        "discharge_c_rate": _round_to_step(discharge_c_rate_raw, 0.5),
        "c_rate_method": "active-time mean current in CYC intervals through EFC50 / 2.5Ah; rounded 0.5C",
        "condition_n_cyc_files": len(set(condition_cyc_numbers)),
    }
    for label, details in info.items():
        metadata[f"{label}_cu_number"] = details["checkup_number"]
        metadata[f"{label}_source_file"] = details["source_file"]
        metadata[f"{label}_line"] = details["reference_line"]
        metadata[f"{label}_raw_start_idx"] = details["raw_start_idx"]
        metadata[f"{label}_raw_end_idx_exclusive"] = details["raw_end_idx_exclusive"]
        metadata[f"{label}_eligible_discharges"] = details["n_eligible_discharges"]

    return pd.concat(frames, ignore_index=True), metadata, rpt_summary
