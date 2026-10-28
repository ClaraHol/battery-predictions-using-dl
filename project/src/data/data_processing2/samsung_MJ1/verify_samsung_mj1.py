"""Verify processed Zhu2022 Dataset_2_NCM_battery (MJ1) curves and metadata."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/samsung_mj1")
EXPECTED_CURVE_TYPES = {"efc0", "efc50_lower", "efc50_upper"}
Q_NOM_AH = 3.5
SOH_THRESHOLD = 0.90


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _is_multiple(value, step, atol=1e-8):
    return np.isclose(value / step, round(value / step), atol=atol)


def _recompute_lifetime(summary):
    summary = summary.sort_values("cycle_number")
    soh = summary["soh"].to_numpy(dtype=float)
    efc = summary["rpt_efc"].to_numpy(dtype=float)
    crossing = np.flatnonzero(soh <= SOH_THRESHOLD)
    if not len(crossing):
        return float(efc[-1]), True
    j = int(crossing[0])
    if j == 0:
        return float(efc[0]), False
    s0, s1 = soh[j - 1], soh[j]
    e0, e1 = efc[j - 1], efc[j]
    if np.isclose(s1, s0):
        return float(e1), False
    w = (SOH_THRESHOLD - s0) / (s1 - s0)
    return float(e0 + w * (e1 - e0)), False


def main():
    root = DATA_DIR.expanduser()
    curves = pd.read_parquet(root / "samsung_mj1_all_curves.parquet")
    metadata = pd.read_parquet(root / "samsung_mj1_all_metadata.parquet")
    cycles = pd.read_parquet(root / "samsung_mj1_cycle_summary.parquet")

    print("Metadata header:")
    print(metadata.head())

    problems = []
    warnings = []

    if metadata["cell_id"].duplicated().any():
        problems.append("Duplicate cell IDs in metadata")
    if set(curves["cell_id"]) != set(metadata["cell_id"]):
        problems.append("Cell IDs differ between curves and metadata")
    if set(cycles["cell_id"]) != set(metadata["cell_id"]):
        problems.append("Cell IDs differ between cycle summary and metadata")
    if (metadata["chemistry"] != "MJ1").any() or (curves["chemistry"] != "MJ1").any():
        problems.append("Incorrect chemistry label")
    if (metadata["source_dataset"] != "Dataset_2_NCM_battery").any():
        problems.append("Unexpected source_dataset")
    if "dataset_provenance" in metadata and (metadata["dataset_provenance"] != "Zhu2022").any():
        problems.append("Unexpected dataset_provenance")
    if "source_family" in metadata and (metadata["source_family"] != "samsung_mj1").any():
        problems.append("Unexpected source_family")
    if (metadata["direction"] != "discharge").any():
        problems.append("Unexpected curve direction")

    # The full published Dataset_2 contains 55 NCM cells: 23/4/28 at 25/35/45 C.
    if len(metadata) != 55:
        warnings.append(f"Processed {len(metadata)} cells; full Dataset_2 is expected to contain 55")

    expected_temp_counts = {25.0: 23, 35.0: 4, 45.0: 28}
    observed_temp_counts = metadata["temperature_C"].value_counts().to_dict()
    if len(metadata) == 55 and observed_temp_counts != expected_temp_counts:
        warnings.append(
            f"Temperature counts differ from published Dataset_2 counts: {observed_temp_counts}"
        )

    meta = metadata.set_index("cell_id")

    for row in metadata.itertuples(index=False):
        cell_id = row.cell_id

        if not np.isclose(row.efc0, 0.0, atol=1e-9):
            problems.append((cell_id, f"efc0={row.efc0} is not zero"))
        if not row.efc50_lower < 50.0 < row.efc50_upper:
            problems.append((cell_id, "EFC50 is not strictly bracketed"))
        if not np.isclose(row.weight_lower + row.weight_upper, 1.0, atol=1e-8):
            problems.append((cell_id, "EFC50 weights do not sum to 1"))
        expected_upper = (50.0 - row.efc50_lower) / (row.efc50_upper - row.efc50_lower)
        if not np.isclose(row.weight_upper, expected_upper, atol=1e-8):
            problems.append((cell_id, "Incorrect EFC50 interpolation weight"))

        if not np.isclose(row.cycle_life_soh_threshold, SOH_THRESHOLD, atol=1e-12):
            problems.append((cell_id, "Unexpected SOH lifetime threshold"))
        if not _is_multiple(float(row.temperature_C), 5.0):
            problems.append((cell_id, f"temperature_C={row.temperature_C} is not rounded to 5 C"))
        if float(row.temperature_C) not in {25.0, 35.0, 45.0}:
            warnings.append((cell_id, f"Unexpected Dataset_2 temperature {row.temperature_C} C"))
        if not np.isclose(row.charge_c_rate, 0.5, atol=1e-9):
            warnings.append((cell_id, f"Unexpected Samsung-MJ1 charge C-rate {row.charge_c_rate}"))
        if not np.isclose(row.discharge_c_rate, 1.0, atol=1e-9):
            warnings.append((cell_id, f"Unexpected Samsung-MJ1 discharge C-rate {row.discharge_c_rate}"))

        cell_cycles = cycles.loc[cycles["cell_id"].eq(cell_id)].sort_values("cycle_number")
        if len(cell_cycles) != int(row.n_cycles):
            problems.append((cell_id, f"n_cycles={row.n_cycles} but summary has {len(cell_cycles)} rows"))
        if cell_cycles.empty:
            continue
        if np.any(np.diff(cell_cycles["rpt_efc"].to_numpy(dtype=float)) <= 0):
            problems.append((cell_id, "Non-increasing cycle EFC positions"))

        initial_capacity = float(cell_cycles.iloc[0]["capacity_Ah"])
        if not np.isclose(row.initial_capacity_Ah, initial_capacity, rtol=1e-8, atol=1e-10):
            problems.append((cell_id, "initial_capacity_Ah differs from cycle summary"))
        expected_soh = cell_cycles["capacity_Ah"].to_numpy(dtype=float) / initial_capacity
        if not np.allclose(cell_cycles["soh"], expected_soh, rtol=1e-8, atol=1e-10):
            problems.append((cell_id, "SOH does not equal capacity / initial capacity"))

        life, censored = _recompute_lifetime(cell_cycles)
        if bool(row.cycle_life_censored) != censored:
            problems.append((cell_id, "Stored lifetime censoring disagrees with cycle summary"))
        if not np.isclose(row.cycle_life_efc, life, rtol=1e-8, atol=1e-6):
            problems.append((cell_id, f"Stored lifetime {row.cycle_life_efc:.3f} != recomputed {life:.3f}"))
        if censored:
            warnings.append((cell_id, f"90% SOH not reached; lifetime >= {life:.1f} EFC"))

        # Protocol-current cross-check from the cycle summaries.
        ch = cell_cycles["charge_current_median_A"].dropna().to_numpy(dtype=float)
        dis = cell_cycles["discharge_current_median_A"].dropna().to_numpy(dtype=float)
        if len(ch) and abs(np.median(ch) - row.charge_current_expected_A) > 0.15:
            problems.append((cell_id, "Observed CC charge current disagrees with filename protocol"))
        if len(dis) and abs(np.median(dis) - row.discharge_current_expected_A) > 0.15:
            problems.append((cell_id, "Observed discharge current disagrees with filename protocol"))

    # Curve-level checks.
    for cell_id, cell in curves.groupby("cell_id", sort=False):
        observed = set(cell["curve_type"].unique())
        if observed != EXPECTED_CURVE_TYPES:
            problems.append((cell_id, f"Expected 3 curve types, got {sorted(observed)}"))
            continue

        m = meta.loc[cell_id]
        expected_efc = {
            "efc0": m["efc0"],
            "efc50_lower": m["efc50_lower"],
            "efc50_upper": m["efc50_upper"],
        }
        expected_cycle = {
            "efc0": int(m["efc0_cycle_number"]),
            "efc50_lower": int(m["efc50_lower_cycle_number"]),
            "efc50_upper": int(m["efc50_upper_cycle_number"]),
        }

        for curve_type, frame in cell.groupby("curve_type", sort=False):
            t = frame["time_s"].to_numpy(dtype=float)
            v = frame["voltage_V"].to_numpy(dtype=float)
            i = frame["current_A"].to_numpy(dtype=float)
            issues = []

            if len(t) < 100:
                issues.append("Too few samples")
            if not (np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(i).all()):
                issues.append("Non-finite values")
            if abs(t[0]) > 1e-6:
                issues.append("Time does not start at zero")
            if np.any(np.diff(t) <= 0):
                issues.append("Non-increasing time")
            if np.any(np.diff(t) > 120):
                issues.append("Large time gap")

            duration_h = (t[-1] - t[0]) / 3600.0
            capacity_ah = _trapz(i, t) / 3600.0
            current_med = float(np.median(i))
            if not 0.45 <= duration_h <= 1.25:
                issues.append("Unusual duration")
            if v[0] < 3.8:
                issues.append("Low starting voltage")
            if v[-1] > 2.60:
                issues.append("High ending voltage")
            if (i <= 0).any():
                issues.append("Discharge current not stored as positive magnitude")
            if abs(current_med - 3.5) > 0.20:
                issues.append("Unexpected discharge current")

            if not np.allclose(frame["rpt_efc"], expected_efc[curve_type], atol=1e-8):
                issues.append("Curve EFC differs from metadata")
            if frame["cycle_number"].nunique() != 1 or int(frame["cycle_number"].iloc[0]) != expected_cycle[curve_type]:
                issues.append("Curve cycle number differs from metadata")

            cycle_row = cycles.loc[
                cycles["cell_id"].eq(cell_id)
                & cycles["cycle_number"].eq(expected_cycle[curve_type])
            ]
            if len(cycle_row) == 1:
                counter_capacity = float(cycle_row.iloc[0]["capacity_Ah"])
                rel = abs(capacity_ah - counter_capacity) / counter_capacity
                if rel > 0.02:
                    issues.append(f"Integrated/counter capacity differs by {100*rel:.1f}%")

            status = "OK" if not issues else "CHECK"
            print(
                f"{cell_id:28} {curve_type:13} N={len(t):4d} "
                f"T={duration_h:5.3f}h V={v[0]:.3f}->{v[-1]:.3f} "
                f"I={current_med:.3f}A Q={capacity_ah:.3f}Ah {status}"
            )
            if issues:
                problems.append((cell_id, f"{curve_type}: {', '.join(issues)}"))

    print("\n========== SUMMARY ==========")
    print(f"Cells in curves:       {curves['cell_id'].nunique()}")
    print(f"Cells in metadata:     {len(metadata)}")
    print(f"Cells in cycle summary:{cycles['cell_id'].nunique():6d}")
    print(f"Extracted curves:      {curves.groupby(['cell_id', 'curve_type']).ngroups}")
    print(f"Expected curves:       {len(metadata) * 3}")
    print(f"Cycle rows:            {len(cycles)}")
    print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
    print(f"Flagged problems:      {len(problems)}")
    print(f"Warnings:              {len(warnings)}")

    if problems:
        print("\nProblems:")
        for problem in problems:
            if isinstance(problem, tuple):
                print(f"  {problem[0]}: {problem[1]}")
            else:
                print(f"  {problem}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            if isinstance(warning, tuple):
                print(f"  {warning[0]}: {warning[1]}")
            else:
                print(f"  {warning}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
