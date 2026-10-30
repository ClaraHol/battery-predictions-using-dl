"""Extract EFC0/EFC50 discharge curves and metadata from Zhu2022 Dataset_2_NCM (MJ1).

This file intentionally targets ``Dataset_2_NCM_battery`` only.  In the
source publication this is the 3.5 Ah NCM cell set (MJ1), not the 2.5 Ah
Samsung 25R NCM+NCA set in Dataset_3.

Each CSV contains the complete cycling history of one cell.  Every cycle has
CC-CV charge, a post-charge relaxation, CC discharge, and a post-discharge
relaxation.  The CC discharge is used as the model observation.

Common output schema:
    curves:   time_s, voltage_V, current_A, curve_type, rpt_efc, cycle_number
    metadata: EFC50 interpolation, 90% SOH lifetime, temperature and C-rates

EFC is defined as bidirectional throughput:
    (cumulative_charge_Ah + cumulative_discharge_Ah) / (2 * Q_nom_Ah)
with each cycle's observation assigned the EFC at the beginning of that cycle.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


Q_NOM_AH = 3.5
TARGET_EFC = 50.0
LIFETIME_SOH_THRESHOLD = 0.90
REFERENCE_DISCHARGE_C = 1.0

REQUIRED_COLUMNS = {
    "time/s",
    "Ecell/V",
    "<I>/mA",
    "Q discharge/mA.h",
    "Q charge/mA.h",
    "cycle number",
}

FILE_RE = re.compile(
    r"^CY(?P<temperature>\d+)-(?P<charge>[0-9]+)_(?P<discharge>[0-9]+)-#(?P<tag>\d+)\.csv$",
    re.IGNORECASE,
)


def _decode_rate(token: str) -> float:
    """Decode filename C-rate tokens: 025 -> 0.25, 05 -> 0.5, 1 -> 1."""
    token = str(token)
    if len(token) > 1 and token.startswith("0"):
        return int(token) / (10 ** (len(token) - 1))
    return float(token)


def _round_to_5(value: float) -> float:
    """Round half-up to the nearest 5 degC, matching paper preprocessing."""
    return float(np.floor(float(value) / 5.0 + 0.5) * 5.0)


def parse_protocol(file_path: str | Path) -> dict:
    """Parse temperature and charge/discharge C-rates from the CSV filename."""
    path = Path(file_path)
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Unexpected Samsung-MJ1 filename {path.name}; expected e.g. CY25-05_1-#1.csv"
        )

    temperature_raw = float(match.group("temperature"))
    charge_c = _decode_rate(match.group("charge"))
    discharge_c = _decode_rate(match.group("discharge"))

    return {
        "temperature_C_raw": temperature_raw,
        "temperature_C": _round_to_5(temperature_raw),
        "charge_c_rate": float(charge_c),
        "discharge_c_rate": float(discharge_c),
        "cell_tag": int(match.group("tag")),
        "condition": path.stem.rsplit("-#", 1)[0],
    }


def _read_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} missing columns: {sorted(missing)}")

    numeric = list(REQUIRED_COLUMNS)
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[numeric].isna().any().any():
        bad = df[numeric].columns[df[numeric].isna().any()].tolist()
        raise ValueError(f"{file_path.name} contains non-numeric/NaN values in {bad}")

    cycle = df["cycle number"].to_numpy(dtype=float)
    if not np.allclose(cycle, np.round(cycle), atol=1e-9):
        raise ValueError("cycle number contains non-integer values")
    if np.any(np.diff(cycle) < 0):
        raise ValueError("cycle number decreases inside file")

    time = df["time/s"].to_numpy(dtype=float)
    if np.any(np.diff(time) <= 0):
        raise ValueError("time/s is not strictly increasing")

    observed_cycles = np.unique(cycle.astype(int))
    expected_cycles = np.arange(observed_cycles.min(), observed_cycles.max() + 1)
    if not np.array_equal(observed_cycles, expected_cycles):
        missing_cycles = sorted(set(expected_cycles) - set(observed_cycles))
        raise ValueError(f"cycle sequence has gaps: {missing_cycles[:20]}")

    return df


def _counter_capacity_ah(cycle_df: pd.DataFrame, column: str) -> float:
    values = cycle_df[column].to_numpy(dtype=float)
    capacity = (np.nanmax(values) - np.nanmin(values)) / 1000.0
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"Invalid {column} capacity")
    return float(capacity)


def _protocol_current_median(
    cycle_df: pd.DataFrame,
    expected_current_a: float,
    *,
    charge: bool,
) -> float:
    current = cycle_df["<I>/mA"].to_numpy(dtype=float) / 1000.0
    if charge:
        mask = (current > 0.8 * expected_current_a) & (current < 1.2 * expected_current_a)
        values = current[mask]
    else:
        mask = (current < -0.8 * expected_current_a) & (current > -1.2 * expected_current_a)
        values = -current[mask]
    return float(np.median(values)) if len(values) else np.nan


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return indices
    parts = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return max(parts, key=len)


def _reference_discharge(
    cycle_df: pd.DataFrame,
    cycle_number: int,
    expected_current_a: float,
) -> pd.DataFrame:
    """Extract the longest constant-current discharge in one cycle."""
    current_a = cycle_df["<I>/mA"].to_numpy(dtype=float) / 1000.0
    mask = (
        (current_a < -0.8 * expected_current_a)
        & (current_a > -1.2 * expected_current_a)
    )
    run = _longest_true_run(mask)
    if len(run) < 100:
        raise ValueError(
            f"Cycle {cycle_number}: no sufficiently long {expected_current_a:.3f} A discharge"
        )

    selected = cycle_df.iloc[run].copy()
    time = selected["time/s"].to_numpy(dtype=float)
    voltage = selected["Ecell/V"].to_numpy(dtype=float)
    current = -selected["<I>/mA"].to_numpy(dtype=float) / 1000.0
    time = time - time[0]

    if np.any(np.diff(time) <= 0):
        raise ValueError(f"Cycle {cycle_number}: non-increasing discharge time")

    duration_h = (time[-1] - time[0]) / 3600.0
    median_current = float(np.median(current))
    if not 0.45 <= duration_h <= 1.25:
        raise ValueError(
            f"Cycle {cycle_number}: unexpected discharge duration {duration_h:.3f} h"
        )
    if voltage[0] < 3.8:
        raise ValueError(
            f"Cycle {cycle_number}: discharge starts too low at {voltage[0]:.3f} V"
        )
    if voltage[-1] > 2.60:
        raise ValueError(
            f"Cycle {cycle_number}: discharge ends too high at {voltage[-1]:.3f} V"
        )
    if abs(median_current - expected_current_a) > max(0.15, 0.06 * expected_current_a):
        raise ValueError(
            f"Cycle {cycle_number}: unexpected discharge current {median_current:.3f} A"
        )

    curve = pd.DataFrame(
        {
            "time_s": time,
            "voltage_V": voltage,
            "current_A": current,
        }
    )
    curve.attrs.update(
        {
            "cycle_number": int(cycle_number),
            "duration_h": float(duration_h),
            "median_current_A": median_current,
        }
    )
    return curve


def _cycle_summary(df: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    expected_charge_a = protocol["charge_c_rate"] * Q_NOM_AH
    expected_discharge_a = protocol["discharge_c_rate"] * Q_NOM_AH

    rows = []
    cumulative_efc = 0.0
    for cycle_number, cycle_df in df.groupby("cycle number", sort=True):
        cycle_number = int(round(float(cycle_number)))
        charge_ah = _counter_capacity_ah(cycle_df, "Q charge/mA.h")
        discharge_ah = _counter_capacity_ah(cycle_df, "Q discharge/mA.h")

        rows.append(
            {
                "cycle_number": cycle_number,
                "rpt_efc": float(cumulative_efc),
                "charge_capacity_Ah": charge_ah,
                "capacity_Ah": discharge_ah,
                "charge_current_median_A": _protocol_current_median(
                    cycle_df, expected_charge_a, charge=True
                ),
                "discharge_current_median_A": _protocol_current_median(
                    cycle_df, expected_discharge_a, charge=False
                ),
            }
        )
        cumulative_efc += (charge_ah + discharge_ah) / (2.0 * Q_NOM_AH)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No cycles found")
    if summary["rpt_efc"].duplicated().any() or np.any(np.diff(summary["rpt_efc"]) <= 0):
        raise ValueError("Cycle EFC positions are not strictly increasing")

    initial_capacity = float(summary.loc[0, "capacity_Ah"])
    if not 2.0 <= initial_capacity <= 4.0:
        raise ValueError(f"Implausible initial discharge capacity {initial_capacity:.3f} Ah")
    summary["soh"] = summary["capacity_Ah"] / initial_capacity
    return summary


def _lifetime_from_summary(summary: pd.DataFrame) -> dict:
    threshold = LIFETIME_SOH_THRESHOLD
    crossing = np.flatnonzero(summary["soh"].to_numpy(dtype=float) <= threshold)

    if len(crossing) == 0:
        last = summary.iloc[-1]
        return {
            "cycle_life_efc": float(last["rpt_efc"]),
            "cycle_life_censored": True,
            "cycle_life_soh_threshold": threshold,
            "lifetime_status": "censored_end_of_file",
            "lifetime_lower_cycle": int(last["cycle_number"]),
            "lifetime_upper_cycle": np.nan,
            "last_observed_soh": float(last["soh"]),
        }

    upper_idx = int(crossing[0])
    if upper_idx == 0:
        raise ValueError("Initial cycle is already at/below the SOH threshold")

    lower = summary.iloc[upper_idx - 1]
    upper = summary.iloc[upper_idx]
    s0, s1 = float(lower["soh"]), float(upper["soh"])
    e0, e1 = float(lower["rpt_efc"]), float(upper["rpt_efc"])

    if not (s0 > threshold >= s1):
        raise ValueError("Invalid SOH threshold bracket")
    if np.isclose(s1, s0):
        lifetime = e1
    else:
        weight = (threshold - s0) / (s1 - s0)
        lifetime = e0 + weight * (e1 - e0)

    return {
        "cycle_life_efc": float(lifetime),
        "cycle_life_censored": False,
        "cycle_life_soh_threshold": threshold,
        "lifetime_status": "observed",
        "lifetime_lower_cycle": int(lower["cycle_number"]),
        "lifetime_upper_cycle": int(upper["cycle_number"]),
        "last_observed_soh": float(summary.iloc[-1]["soh"]),
    }


def extract_curve_samsung_mj1(
    file_path: str | Path,
    *,
    target_efc: float = TARGET_EFC,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return three discharge curves, cell metadata, and per-cycle SOH summary."""
    file_path = Path(file_path)
    protocol = parse_protocol(file_path)
    df = _read_csv(file_path)
    summary = _cycle_summary(df, protocol)

    efcs = summary["rpt_efc"].to_numpy(dtype=float)
    lower_candidates = np.flatnonzero(efcs < target_efc)
    upper_candidates = np.flatnonzero(efcs > target_efc)
    if not len(lower_candidates) or not len(upper_candidates):
        raise ValueError(f"EFC{target_efc:g} is not strictly bracketed")

    lower_pos = int(lower_candidates[-1])
    upper_pos = int(upper_candidates[0])
    lower = summary.iloc[lower_pos]
    upper = summary.iloc[upper_pos]
    lower_efc = float(lower["rpt_efc"])
    upper_efc = float(upper["rpt_efc"])
    weight_upper = (target_efc - lower_efc) / (upper_efc - lower_efc)

    selection = {
        "efc0": int(summary.iloc[0]["cycle_number"]),
        "efc50_lower": int(lower["cycle_number"]),
        "efc50_upper": int(upper["cycle_number"]),
    }
    efc_lookup = dict(zip(summary["cycle_number"], summary["rpt_efc"]))
    expected_discharge_a = protocol["discharge_c_rate"] * Q_NOM_AH

    frames = []
    for curve_type, cycle_number in selection.items():
        cycle_df = df.loc[df["cycle number"].eq(float(cycle_number))]
        curve = _reference_discharge(cycle_df, cycle_number, expected_discharge_a)
        curve["curve_type"] = curve_type
        curve["rpt_efc"] = float(efc_lookup[cycle_number])
        curve["cycle_number"] = int(cycle_number)
        frames.append(curve)

    lifetime = _lifetime_from_summary(summary)
    initial = summary.iloc[0]

    metadata = {
        "efc0": 0.0,
        "efc50_lower": lower_efc,
        "efc50_upper": upper_efc,
        "weight_lower": float(1.0 - weight_upper),
        "weight_upper": float(weight_upper),
        "direction": "discharge",
        "efc0_cycle_number": selection["efc0"],
        "efc50_lower_cycle_number": selection["efc50_lower"],
        "efc50_upper_cycle_number": selection["efc50_upper"],
        "n_cycles": int(len(summary)),
        "initial_capacity_Ah": float(initial["capacity_Ah"]),
        "temperature_C_raw": protocol["temperature_C_raw"],
        "temperature_C": protocol["temperature_C"],
        "charge_c_rate": protocol["charge_c_rate"],
        "discharge_c_rate": protocol["discharge_c_rate"],
        "charge_current_expected_A": float(protocol["charge_c_rate"] * Q_NOM_AH),
        "discharge_current_expected_A": float(protocol["discharge_c_rate"] * Q_NOM_AH),
        "condition": protocol["condition"],
        "cell_tag": protocol["cell_tag"],
        "source_file": file_path.name,
        "source_dataset": "Dataset_2_NCM_battery",
        "dataset_provenance": "Zhu2022",
        "source_family": "samsung_mj1",
        "efc_definition": "(charge_Ah+discharge_Ah)/(2*3.5Ah); observation EFC at cycle start",
        **lifetime,
    }

    return pd.concat(frames, ignore_index=True), metadata, summary
