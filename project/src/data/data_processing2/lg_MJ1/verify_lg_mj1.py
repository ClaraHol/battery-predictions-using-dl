"""Verify processed VITO/EVERLASTING LG-MJ1 curves and metadata."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/lg_mj1")
SOH_THRESHOLD = 0.90
EXPECTED_CURVE_TYPES = {"efc0", "efc50_lower", "efc50_upper"}


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _is_multiple(value, step, atol=1e-9):
    return np.isclose(value / step, round(value / step), atol=atol)


def _recompute_lifetime(summary):
    summary = summary.sort_values("rpt_efc")
    soh = summary["soh"].to_numpy(dtype=float)
    efc = summary["rpt_efc"].to_numpy(dtype=float)
    crossing = np.flatnonzero(soh <= SOH_THRESHOLD)
    if len(crossing) == 0:
        return float(efc[-1]), True
    upper = int(crossing[0])
    if upper == 0:
        return float(efc[0]), False
    s0, s1 = soh[upper - 1], soh[upper]
    e0, e1 = efc[upper - 1], efc[upper]
    if np.isclose(s0, s1):
        return float(e1), False
    w = (SOH_THRESHOLD - s0) / (s1 - s0)
    return float(e0 + w * (e1 - e0)), False


def main():
    root = DATA_DIR.expanduser()
    curves = pd.read_parquet(root / "lg_mj1_all_curves.parquet")
    metadata = pd.read_parquet(root / "lg_mj1_all_metadata.parquet")
    rpts = pd.read_parquet(root / "lg_mj1_rpt_summary.parquet")

    print("Metadata header:")
    print(metadata.head())

    problems = []
    warnings = []

    if metadata["cell_id"].duplicated().any():
        problems.append("Duplicate cell IDs in metadata")
    if set(curves["cell_id"]) != set(metadata["cell_id"]):
        problems.append("Cell IDs differ between curves and metadata")
    if set(rpts["cell_id"]) != set(metadata["cell_id"]):
        problems.append("Cell IDs differ between RPT summary and metadata")
    if (metadata["chemistry"] != "MJ1").any() or (curves["chemistry"] != "MJ1").any():
        problems.append("Incorrect chemistry label")
    if (metadata["direction"] != "discharge").any():
        problems.append("Unexpected curve direction")
    if (metadata["source_family"] != "lg_mj1").any():
        problems.append("Unexpected source_family")
    if (metadata["source_dataset"] != "EVERLASTING_VITO_25_45C").any():
        problems.append("Unexpected source_dataset")

    # The public 25/45 C VITO release contains 17 unique cell CSVs.
    if len(metadata) != 17:
        warnings.append(f"Processed {len(metadata)} cells; the public VITO 25/45 C release has 17 unique CSV cells")

    meta = metadata.set_index("cell_id")

    for row in metadata.itertuples(index=False):
        cell_id = row.cell_id
        if not np.isclose(row.efc0, 0.0, atol=1e-9):
            problems.append((cell_id, f"efc0={row.efc0} is not zero"))
        if not row.efc50_lower < 50.0 < row.efc50_upper:
            problems.append((cell_id, "EFC50 is not strictly bracketed"))
        if not np.isclose(row.weight_lower + row.weight_upper, 1.0, atol=1e-8):
            problems.append((cell_id, "EFC50 weights do not sum to one"))
        expected_upper = (50.0 - row.efc50_lower) / (row.efc50_upper - row.efc50_lower)
        if not np.isclose(row.weight_upper, expected_upper, atol=1e-8):
            problems.append((cell_id, "Incorrect EFC50 interpolation weight"))

        if not np.isclose(row.cycle_life_soh_threshold, SOH_THRESHOLD, atol=1e-12):
            problems.append((cell_id, "Unexpected SOH lifetime threshold"))
        if not _is_multiple(float(row.temperature_C), 5.0):
            problems.append((cell_id, f"temperature_C={row.temperature_C} is not rounded to 5 C"))
        if float(row.temperature_C) not in {25.0, 45.0}:
            warnings.append((cell_id, f"Unexpected VITO release temperature {row.temperature_C} C"))
        if np.isfinite(row.charge_c_rate) and not _is_multiple(float(row.charge_c_rate), 0.5):
            problems.append((cell_id, f"charge_c_rate={row.charge_c_rate} is not rounded to 0.5 C"))
        if np.isfinite(row.discharge_c_rate) and not _is_multiple(float(row.discharge_c_rate), 0.5):
            problems.append((cell_id, f"discharge_c_rate={row.discharge_c_rate} is not rounded to 0.5 C"))

        expected_time_method = "cumulative_positive_raw_dt_with_timestamp_reset_repair"
        if getattr(row, "time_axis_method", None) != expected_time_method:
            problems.append((cell_id, f"Unexpected time_axis_method={getattr(row, 'time_axis_method', None)!r}"))
        for field in ("n_time_resets", "n_timestamp_time_repairs", "n_fallback_time_repairs"):
            value = getattr(row, field, None)
            if value is None or not np.isfinite(value) or float(value) < 0:
                problems.append((cell_id, f"Invalid {field}={value}"))
        if not np.isfinite(row.reconstructed_duration_s) or row.reconstructed_duration_s <= 0:
            problems.append((cell_id, "Invalid reconstructed_duration_s"))
        if row.n_timestamp_time_repairs > row.n_time_resets:
            problems.append((cell_id, "More timestamp repairs than detected time resets"))
        # Timer resets are expected in this dataset and are not warnings by
        # themselves. Only flag unusually frequent cadence fallbacks, where no
        # usable timestamp was available to bridge a reset/repeat boundary.
        fallback_limit = max(10, int(1e-4 * row.n_rows_scanned))
        if row.n_fallback_time_repairs > fallback_limit:
            warnings.append(
                (cell_id, f"Time reconstruction required {int(row.n_fallback_time_repairs)} cadence fallbacks")
            )

        cell_rpts = rpts.loc[rpts["cell_id"].eq(cell_id)].sort_values("rpt_efc")
        if len(cell_rpts) != int(row.n_rpts):
            problems.append((cell_id, f"n_rpts={row.n_rpts} but summary has {len(cell_rpts)} rows"))
        if cell_rpts.empty:
            continue
        if np.any(np.diff(cell_rpts["rpt_efc"].to_numpy(dtype=float)) <= 0):
            problems.append((cell_id, "Non-increasing RPT EFC positions"))

        initial_capacity = float(cell_rpts.iloc[0]["capacity_Ah"])
        if not np.isclose(row.initial_capacity_Ah, initial_capacity, rtol=1e-8, atol=1e-10):
            problems.append((cell_id, "initial_capacity_Ah differs from RPT summary"))
        expected_soh = cell_rpts["capacity_Ah"].to_numpy(dtype=float) / initial_capacity
        if not np.allclose(cell_rpts["soh"], expected_soh, rtol=1e-8, atol=1e-10):
            problems.append((cell_id, "SOH does not equal capacity / initial capacity"))

        life, censored = _recompute_lifetime(cell_rpts)
        if bool(row.cycle_life_censored) != censored:
            problems.append((cell_id, "Stored lifetime censoring disagrees with RPT summary"))
        if not np.isclose(row.cycle_life_efc, life, rtol=1e-8, atol=1e-5):
            problems.append((cell_id, f"Stored lifetime {row.cycle_life_efc:.3f} != recomputed {life:.3f}"))
        if censored:
            warnings.append((cell_id, f"90% SOH not reached; lifetime >= {life:.1f} EFC"))

        nonpair = cell_rpts.loc[cell_rpts["n_full_1c_discharges"] != 2]
        if len(nonpair):
            details = ", ".join(
                f"RPT{int(r.rpt_index)}:{int(r.n_full_1c_discharges)} cycles"
                for r in nonpair.itertuples(index=False)
            )
            warnings.append((cell_id, f"Check-up did not contain exactly two detected full 1C discharges: {details}"))

        if (cell_rpts["selection_status"] == "single_detected_cycle").any():
            warnings.append((cell_id, "At least one check-up used a single detected 1C cycle instead of documented second cycle"))

        current_med = cell_rpts["median_current_A"].to_numpy(dtype=float)
        if np.any(np.abs(current_med - 3.5) > 0.4):
            problems.append((cell_id, "RPT summary contains unexpected reference-discharge current"))

        counter = cell_rpts["counter_capacity_Ah"].to_numpy(dtype=float)
        integrated = cell_rpts["capacity_Ah"].to_numpy(dtype=float)
        valid = np.isfinite(counter) & (counter > 0)
        if np.any(valid):
            rel = np.abs(counter[valid] - integrated[valid]) / integrated[valid]
            if np.nanmax(rel) > 0.05:
                warnings.append((cell_id, f"Integrated/counter RPT capacity differs by up to {100*np.nanmax(rel):.1f}%"))

    for cell_id, cell in curves.groupby("cell_id", sort=False):
        observed = set(cell["curve_type"].unique())
        if observed != EXPECTED_CURVE_TYPES:
            problems.append((cell_id, f"Expected 3 curve types, got {sorted(observed)}"))
            continue

        m = meta.loc[cell_id]
        expected_efc = {
            "efc0": float(m["efc0"]),
            "efc50_lower": float(m["efc50_lower"]),
            "efc50_upper": float(m["efc50_upper"]),
        }
        expected_rpt = {
            "efc0": int(m["efc0_rpt_index"]),
            "efc50_lower": int(m["efc50_lower_rpt_index"]),
            "efc50_upper": int(m["efc50_upper_rpt_index"]),
        }

        for curve_type, frame in cell.groupby("curve_type", sort=False):
            t = frame["time_s"].to_numpy(dtype=float)
            v = frame["voltage_V"].to_numpy(dtype=float)
            i = frame["current_A"].to_numpy(dtype=float)
            issues = []

            if len(t) < 20:
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
            capacity_ah = _trapz(i, t) / 3600.0
            current_med = float(np.median(i))
            if not 0.65 <= duration_h <= 1.30:
                issues.append("Unusual 1C discharge duration")
            if v[0] < 4.00:
                issues.append("Low starting voltage")
            if v[-1] > 2.65:
                issues.append("High ending voltage")
            if (i <= 0).any():
                issues.append("Discharge current not stored as positive magnitude")
            if abs(current_med - 3.5) > 0.4:
                issues.append("Unexpected 1C reference current")
            if not 2.3 <= capacity_ah <= 4.0:
                issues.append("Unusual integrated capacity")
            if not np.allclose(frame["rpt_efc"], expected_efc[curve_type], atol=1e-8):
                issues.append("Curve EFC differs from metadata")
            if frame["rpt_index"].nunique() != 1 or int(frame["rpt_index"].iloc[0]) != expected_rpt[curve_type]:
                issues.append("Curve RPT index differs from metadata")

            rpt_row = rpts.loc[
                rpts["cell_id"].eq(cell_id)
                & rpts["rpt_index"].eq(expected_rpt[curve_type])
            ]
            if len(rpt_row) == 1:
                expected_capacity = float(rpt_row.iloc[0]["capacity_Ah"])
                rel = abs(capacity_ah - expected_capacity) / expected_capacity
                if rel > 0.01:
                    issues.append(f"Curve/RPT integrated capacity differs by {100*rel:.1f}%")

            status = "OK" if not issues else "CHECK"
            print(
                f"{cell_id:20} {curve_type:13} N={len(t):5d} "
                f"T={duration_h:5.3f}h V={v[0]:.3f}->{v[-1]:.3f} "
                f"I={current_med:.3f}A Q={capacity_ah:.3f}Ah {status}"
            )
            if issues:
                problems.append((cell_id, f"{curve_type}: {', '.join(issues)}"))

    print("\n========== SUMMARY ==========")
    print(f"Cells in curves:       {curves['cell_id'].nunique()}")
    print(f"Cells in metadata:     {len(metadata)}")
    print(f"Cells in RPT summary:  {rpts['cell_id'].nunique()}")
    print(f"Extracted curves:      {curves.groupby(['cell_id', 'curve_type']).ngroups}")
    print(f"Expected curves:       {len(metadata) * 3}")
    print(f"RPT rows:              {len(rpts)}")
    print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
    print(f"Raw time resets fixed: {int(metadata['n_time_resets'].sum())}")
    print(f"Timestamp repairs:     {int(metadata['n_timestamp_time_repairs'].sum())}")
    print(f"Fallback repairs:      {int(metadata['n_fallback_time_repairs'].sum())}")
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
