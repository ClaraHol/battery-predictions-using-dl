"""Verify processed Zhu2022 Dataset_3_NCM_NCA_battery Samsung 25R data."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/samsung_25r")
EXPECTED_CURVE_TYPES = {"efc0", "efc50_lower", "efc50_upper"}
Q_NOM_AH = 2.5
SOH_THRESHOLD = 0.90
EXPECTED_CHARGE_A = 1.25


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
    curves = pd.read_parquet(root / "samsung_25r_all_curves.parquet")
    metadata = pd.read_parquet(root / "samsung_25r_all_metadata.parquet")
    cycles = pd.read_parquet(root / "samsung_25r_cycle_summary.parquet")

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
    if (metadata["chemistry"] != "25R").any() or (curves["chemistry"] != "25R").any():
        problems.append("Incorrect chemistry label")
    if (metadata["source_dataset"] != "Dataset_3_NCM_NCA_battery").any():
        problems.append("Unexpected source_dataset")
    if "dataset_provenance" in metadata and (metadata["dataset_provenance"] != "Zhu2022").any():
        problems.append("Unexpected dataset_provenance")
    if "source_family" in metadata and (metadata["source_family"] != "samsung_25r").any():
        problems.append("Unexpected source_family")
    if (metadata["direction"] != "charge").any():
        problems.append("Unexpected curve direction; Samsung-25R should use CC charge profiles")

    # Dataset 3 contains 9 cells: 3 each at 1C, 2C and 4C discharge.
    if len(metadata) != 9:
        warnings.append(f"Processed {len(metadata)} cells; full Dataset_3 is expected to contain 9")

    if len(metadata) == 9:
        expected_discharge_counts = {1.0: 3, 2.0: 3, 4.0: 3}
        observed = metadata["discharge_c_rate"].value_counts().sort_index().to_dict()
        if observed != expected_discharge_counts:
            warnings.append(
                f"Discharge-rate counts differ from published Dataset_3 counts: {observed}"
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
        if not np.isclose(row.temperature_C, 25.0, atol=1e-9):
            problems.append((cell_id, f"Dataset_3 expected 25 C, got {row.temperature_C} C"))
        if not np.isclose(row.charge_c_rate, 0.5, atol=1e-9):
            problems.append((cell_id, f"Dataset_3 expected 0.5C charge, got {row.charge_c_rate}C"))
        if float(row.discharge_c_rate) not in {1.0, 2.0, 4.0}:
            problems.append((cell_id, f"Unexpected Dataset_3 discharge rate {row.discharge_c_rate}C"))

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

        # Cross-check filename protocol against measured current.
        ch = cell_cycles["charge_current_median_A"].dropna().to_numpy(dtype=float)
        dis = cell_cycles["discharge_current_median_A"].dropna().to_numpy(dtype=float)
        if len(ch) and abs(np.median(ch) - row.charge_current_expected_A) > 0.12:
            problems.append((cell_id, "Observed CC charge current disagrees with filename protocol"))
        discharge_tol = max(0.20, 0.05 * float(row.discharge_current_expected_A))
        if len(dis) and abs(np.median(dis) - row.discharge_current_expected_A) > discharge_tol:
            problems.append((cell_id, "Observed discharge current disagrees with filename protocol"))

    # Curve-level checks: these are CC charge curves, not discharge curves.
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

            if len(t) < 75:
                issues.append("Too few samples")
            if not (np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(i).all()):
                issues.append("Non-finite values")
            if abs(t[0]) > 1e-6:
                issues.append("Time does not start at zero")
            if np.any(np.diff(t) <= 0):
                issues.append("Non-increasing time")
            if np.any(np.diff(t) > 180):
                issues.append("Large time gap")

            duration_h = (t[-1] - t[0]) / 3600.0
            cc_charge_ah = _trapz(i, t) / 3600.0
            current_med = float(np.median(i))
            if not 0.40 <= duration_h <= 2.50:
                issues.append("Unusual CC-charge duration")
            if v[0] > 3.75:
                issues.append("High starting voltage for CC charge")
            if v[-1] < 4.05:
                issues.append("Low ending voltage for CC charge")
            if (i <= 0).any():
                issues.append("Charge current not stored as positive magnitude")
            if abs(current_med - EXPECTED_CHARGE_A) > 0.12:
                issues.append("Unexpected CC-charge current")

            if not np.allclose(frame["rpt_efc"], expected_efc[curve_type], atol=1e-8):
                issues.append("Curve EFC differs from metadata")
            if frame["cycle_number"].nunique() != 1 or int(frame["cycle_number"].iloc[0]) != expected_cycle[curve_type]:
                issues.append("Curve cycle number differs from metadata")

            # The stored Q-charge counter includes the following CV charge, so
            # integrated CC capacity should be <= full cycle charge capacity.
            cycle_row = cycles.loc[
                cycles["cell_id"].eq(cell_id)
                & cycles["cycle_number"].eq(expected_cycle[curve_type])
            ]
            if len(cycle_row) == 1:
                full_charge_ah = float(cycle_row.iloc[0]["charge_capacity_Ah"])
                if cc_charge_ah > 1.05 * full_charge_ah:
                    issues.append("Integrated CC charge exceeds full charge counter")
                ratio = cc_charge_ah / full_charge_ah if full_charge_ah > 0 else np.nan
                if np.isfinite(ratio) and ratio < 0.35:
                    warnings.append(
                        (cell_id, f"{curve_type}: CC charge is only {100*ratio:.1f}% of full charge capacity")
                    )

            status = "OK" if not issues else "CHECK"
            print(
                f"{cell_id:28} {curve_type:13} N={len(t):4d} "
                f"T={duration_h:5.3f}h V={v[0]:.3f}->{v[-1]:.3f} "
                f"I={current_med:.3f}A Qcc={cc_charge_ah:.3f}Ah {status}"
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
