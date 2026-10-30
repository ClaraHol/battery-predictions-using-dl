"""Verify VTC5A curves, lifetime/SOH history, and condition metadata."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/vtc5a")
CURVE_TYPES = ("efc0", "efc50_lower", "efc50_upper")
EXPECTED_SOH_THRESHOLD = 0.90


def round_to_step(value, step):
    return float(np.floor(float(value) / step + 0.5) * step)


def curve_diagnostics(frame):
    issues = []
    frame = frame.sort_index()
    t = frame["time_s"].to_numpy(dtype=float)
    v = frame["voltage_V"].to_numpy(dtype=float)
    i = frame["current_A"].to_numpy(dtype=float)
    if len(t) < 2:
        return dict(
            n=len(t), duration_h=np.nan, v_start=np.nan, v_end=np.nan,
            i_median=np.nan, capacity_Ah=np.nan, issues="Too few samples"
        )
    if not (np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(i).all()):
        issues.append("Non-finite values")
    dt = np.diff(t)
    duration = (t[-1] - t[0]) / 3600
    median_current = float(np.median(i))
    if hasattr(np, "trapezoid"):
        capacity = float(np.trapezoid(i, t) / 3600)
    else:
        capacity = float(np.trapz(i, t) / 3600)
    if len(t) < 75:
        issues.append("Too few samples")
    if abs(t[0]) > 1e-3:
        issues.append("Time does not begin at 0")
    if np.any(dt <= 0):
        issues.append("Non-increasing time")
    if np.any(dt > 120):
        issues.append("Time gap >120 s")
    if not 0.55 <= duration <= 1.30:
        issues.append("Unusual duration")
    if v[0] < 4.0:
        issues.append("Low start voltage")
    if v[-1] > 2.65:
        issues.append("High end voltage")
    if not 2.0 <= capacity <= 3.0:
        issues.append("Unusual discharged capacity")
    if (i < 0).any():
        issues.append("Negative current: not absolute-valued")
    if abs(median_current - 2.5) > 0.15:
        issues.append("Unexpected median current")
    if np.mean(np.abs(i - 2.5) > 0.30) > 0.05:
        issues.append(">5% samples far from reference current")
    return dict(
        n=len(t), duration_h=duration, v_start=v[0], v_end=v[-1],
        i_median=median_current, capacity_Ah=capacity, issues="; ".join(issues)
    )


def recompute_lifetime(rpts, threshold):
    rpts = rpts.sort_values("cu_number").reset_index(drop=True)
    crossing = None
    for j in range(1, len(rpts)):
        s0 = float(rpts.loc[j - 1, "soh"])
        s1 = float(rpts.loc[j, "soh"])
        if s0 > threshold >= s1:
            e0 = float(rpts.loc[j - 1, "rpt_efc"])
            e1 = float(rpts.loc[j, "rpt_efc"])
            if np.isclose(s0, s1):
                life = e1
            else:
                life = e0 + (threshold - s0) / (s1 - s0) * (e1 - e0)
            crossing = (
                float(life),
                int(rpts.loc[j - 1, "cu_number"]),
                int(rpts.loc[j, "cu_number"]),
            )
            break
    return crossing


def main():
    root = DATA_DIR.expanduser()
    curves = pd.read_parquet(root / "vtc5a_all_curves.parquet")
    metadata = pd.read_parquet(root / "vtc5a_all_metadata.parquet")
    rpts = pd.read_parquet(root / "vtc5a_rpt_summary.parquet")

    print("Metadata header:")
    print(metadata.head())

    problems = []
    warnings = []
    rows = []

    required_meta = {
        "cell_id", "chemistry", "efc0", "efc50_lower", "efc50_upper",
        "weight_lower", "weight_upper", "direction",
        "initial_rpt_capacity_Ah", "last_rpt_capacity_Ah", "last_rpt_soh",
        "last_rpt_efc", "cycle_life_efc", "cycle_life_censored",
        "cycle_life_soh_threshold", "n_rpts_available", "n_rpts_scanned",
        "temperature_C_raw", "temperature_C",
        "charge_c_rate_raw", "charge_c_rate",
        "discharge_c_rate_raw", "discharge_c_rate",
    }
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        problems.append(f"Metadata missing columns: {sorted(missing_meta)}")

    if metadata["cell_id"].duplicated().any():
        problems.append("Duplicate cell IDs in metadata")
    metadata_ids = set(metadata["cell_id"])
    curves_ids = set(curves["cell_id"])
    rpt_ids = set(rpts["cell_id"])
    if curves_ids != metadata_ids:
        problems.append(
            f"Curve/metadata IDs differ: only metadata={sorted(metadata_ids-curves_ids)}, "
            f"only curves={sorted(curves_ids-metadata_ids)}"
        )
    if rpt_ids != metadata_ids:
        problems.append(
            f"RPT/metadata IDs differ: only metadata={sorted(metadata_ids-rpt_ids)}, "
            f"only RPT={sorted(rpt_ids-metadata_ids)}"
        )
    if (metadata["chemistry"] != "VTC5A").any() or (curves["chemistry"] != "VTC5A").any():
        problems.append("Incorrect chemistry label")
    if (rpts["chemistry"] != "VTC5A").any():
        problems.append("Incorrect RPT chemistry label")
    if (metadata["direction"] != "discharge").any():
        problems.append("Unexpected direction")

    meta_lookup = metadata.drop_duplicates("cell_id").set_index("cell_id")

    for cell_id in sorted(metadata_ids | curves_ids | rpt_ids):
        selected = curves.loc[curves["cell_id"] == cell_id]
        observed = set(selected["curve_type"])
        if observed != set(CURVE_TYPES):
            problems.append(f"{cell_id}: expected 3 curve types, got {sorted(observed)}")

        if cell_id in meta_lookup.index:
            m = meta_lookup.loc[cell_id]
            if not np.isfinite([
                m["efc0"], m["efc50_lower"], m["efc50_upper"],
                m["weight_lower"], m["weight_upper"],
            ]).all():
                problems.append(f"{cell_id}: non-finite EFC/weights")
            elif not (m["efc50_lower"] < 50 < m["efc50_upper"]):
                problems.append(f"{cell_id}: EFC50 unbracketed")
            else:
                expected_upper = (50 - m["efc50_lower"]) / (
                    m["efc50_upper"] - m["efc50_lower"]
                )
                if not np.isclose(m["weight_upper"], expected_upper, atol=1e-6):
                    problems.append(f"{cell_id}: incorrect upper interpolation weight")
                if not np.isclose(m["weight_lower"] + m["weight_upper"], 1, atol=1e-6):
                    problems.append(f"{cell_id}: weights do not sum to 1")
            if not np.isclose(m["efc0"], 0, atol=1e-6):
                problems.append(f"{cell_id}: EFC0 not zero")

            # Paper-style condition clustering.
            if np.isfinite(m["temperature_C_raw"]):
                expected_t = round_to_step(m["temperature_C_raw"], 5.0)
                if not np.isclose(m["temperature_C"], expected_t, atol=1e-8):
                    problems.append(f"{cell_id}: temperature not rounded to nearest 5 C")
            else:
                warnings.append(f"{cell_id}: no ageing temperature extracted from CYC filenames")

            for raw_name, rounded_name in [
                ("charge_c_rate_raw", "charge_c_rate"),
                ("discharge_c_rate_raw", "discharge_c_rate"),
            ]:
                if np.isfinite(m[raw_name]):
                    expected_rate = round_to_step(m[raw_name], 0.5)
                    if not np.isclose(m[rounded_name], expected_rate, atol=1e-8):
                        problems.append(f"{cell_id}: {rounded_name} not rounded to nearest 0.5 C")
                else:
                    warnings.append(f"{cell_id}: {raw_name} is missing")

            if (np.isfinite(m.get("charge_c_rate", np.nan)) and m["charge_c_rate"] > 10) or \
               (np.isfinite(m.get("discharge_c_rate", np.nan)) and m["discharge_c_rate"] > 10):
                warnings.append(
                    f"{cell_id}: rounded C-rate exceeds 10C; inspect condition extraction before downstream use"
                )

            cell_rpts = rpts.loc[rpts["cell_id"] == cell_id].sort_values("cu_number")
            if len(cell_rpts) != int(m["n_rpts_scanned"]):
                problems.append(
                    f"{cell_id}: n_rpts_scanned={int(m['n_rpts_scanned'])} but summary has {len(cell_rpts)}"
                )
            if int(m["n_rpts_available"]) < len(cell_rpts):
                problems.append(f"{cell_id}: scanned more RPTs than available")

            if not cell_rpts.empty:
                efcs = cell_rpts["rpt_efc"].to_numpy(dtype=float)
                capacities = cell_rpts["capacity_Ah"].to_numpy(dtype=float)
                soh = cell_rpts["soh"].to_numpy(dtype=float)
                if not (np.isfinite(efcs).all() and np.isfinite(capacities).all() and np.isfinite(soh).all()):
                    problems.append(f"{cell_id}: non-finite RPT summary values")
                if not np.isclose(efcs[0], 0, atol=1e-8):
                    problems.append(f"{cell_id}: first RPT EFC is not zero")
                if len(efcs) > 1 and np.any(np.diff(efcs) <= 0):
                    problems.append(f"{cell_id}: RPT EFC is not strictly increasing")
                if not np.isclose(soh[0], 1, atol=1e-6):
                    problems.append(f"{cell_id}: CU000 SOH is not 1")
                expected_soh = capacities / capacities[0]
                if not np.allclose(soh, expected_soh, rtol=1e-6, atol=1e-8):
                    problems.append(f"{cell_id}: RPT SOH inconsistent with capacity/CU000 capacity")
                if not np.isclose(m["initial_rpt_capacity_Ah"], capacities[0], rtol=1e-6):
                    problems.append(f"{cell_id}: initial capacity metadata mismatch")
                if not np.isclose(m["last_rpt_capacity_Ah"], capacities[-1], rtol=1e-6):
                    problems.append(f"{cell_id}: last capacity metadata mismatch")
                if not np.isclose(m["last_rpt_soh"], soh[-1], rtol=1e-6):
                    problems.append(f"{cell_id}: last SOH metadata mismatch")
                if not np.isclose(m["last_rpt_efc"], efcs[-1], rtol=1e-6):
                    problems.append(f"{cell_id}: last RPT EFC metadata mismatch")

                threshold = float(m["cycle_life_soh_threshold"])
                if not np.isclose(threshold, EXPECTED_SOH_THRESHOLD, atol=1e-12):
                    warnings.append(
                        f"{cell_id}: lifetime threshold is {threshold:.3f}, expected {EXPECTED_SOH_THRESHOLD:.3f}"
                    )
                crossing = recompute_lifetime(cell_rpts, threshold)
                censored = bool(m["cycle_life_censored"])
                if crossing is None:
                    if not censored:
                        problems.append(f"{cell_id}: uncensored lifetime but no SOH threshold crossing")
                    if not np.isclose(m["cycle_life_efc"], efcs[-1], rtol=1e-6):
                        problems.append(f"{cell_id}: censored lifetime is not last observed RPT EFC")
                    if soh[-1] <= threshold:
                        problems.append(f"{cell_id}: marked censored despite observed threshold crossing")
                    status = str(m.get("lifetime_status", ""))
                    if status == "censored_data_gap":
                        reason = str(m.get("lifetime_censor_reason", ""))
                        warnings.append(f"{cell_id}: lifetime lower bound only; later scan stopped: {reason}")
                else:
                    life, lower_cu, upper_cu = crossing
                    if censored:
                        problems.append(f"{cell_id}: marked censored despite observed threshold crossing")
                    if not np.isclose(m["cycle_life_efc"], life, rtol=1e-6, atol=1e-6):
                        problems.append(
                            f"{cell_id}: lifetime mismatch metadata={m['cycle_life_efc']:.3f}, recomputed={life:.3f}"
                        )
                    if np.isfinite(m.get("cycle_life_lower_cu_number", np.nan)):
                        if int(m["cycle_life_lower_cu_number"]) != lower_cu:
                            problems.append(f"{cell_id}: lifetime lower CU mismatch")
                    if np.isfinite(m.get("cycle_life_upper_cu_number", np.nan)):
                        if int(m["cycle_life_upper_cu_number"]) != upper_cu:
                            problems.append(f"{cell_id}: lifetime upper CU mismatch")

        for curve_type, frame in selected.groupby("curve_type", sort=False):
            result = curve_diagnostics(frame)
            if cell_id in meta_lookup.index and curve_type in CURVE_TYPES:
                column = {
                    "efc0": "efc0",
                    "efc50_lower": "efc50_lower",
                    "efc50_upper": "efc50_upper",
                }[curve_type]
                expected_efc = meta_lookup.loc[cell_id, column]
                if not np.allclose(frame["rpt_efc"].to_numpy(), expected_efc, atol=1e-6):
                    result["issues"] += (
                        ("; " if result["issues"] else "") + "Curve EFC differs from metadata"
                    )
            result.update(
                cell_id=cell_id,
                curve_type=curve_type,
                status="CHECK" if result["issues"] else "OK",
            )
            rows.append(result)
            print(
                f"{cell_id:17} {curve_type:13} N={result['n']:4} "
                f"T={result['duration_h']:5.3f}h "
                f"V={result['v_start']:.3f}->{result['v_end']:.3f} "
                f"I={result['i_median']:.3f}A Q={result['capacity_Ah']:.3f}Ah "
                f"{result['status']}"
            )
            if result["issues"]:
                print(f"   -> {result['issues']}")

    report = pd.DataFrame(rows)
    out = root / "vtc5a_quality_report.csv"
    report.to_csv(out, index=False)
    flagged = report.loc[report["status"] == "CHECK"] if not report.empty else report

    log_path = root / "vtc5a_processing_log.csv"
    skipped = None
    excluded = None
    if log_path.is_file():
        processing_log = pd.read_csv(log_path)
        skipped = processing_log.loc[processing_log["status"] == "SKIPPED"]
        excluded = processing_log.loc[processing_log["status"] == "EXCLUDED"]

    print("\n========== SUMMARY ==========")
    print(f"Cells in curves:       {curves['cell_id'].nunique()}")
    print(f"Cells in metadata:     {len(metadata)}")
    print(f"Cells in RPT summary:  {rpts['cell_id'].nunique()}")
    print(f"Extracted curves:      {len(report)}")
    print(f"Expected curves:       {3 * len(metadata)}")
    print(f"RPT rows:              {len(rpts)}")
    print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
    print(f"Flagged curves:        {len(flagged)}")
    print(f"Metadata problems:     {len(problems)}")
    print(f"Warnings:              {len(warnings)}")

    if skipped is not None:
        print(f"Cells skipped during extraction: {len(skipped)}")
        for row in skipped.itertuples(index=False):
            print(f"  SKIPPED {row.cell_id}: {row.reason}")
    if excluded is not None:
        print(f"Cells excluded as not applicable: {len(excluded)}")
    if problems:
        print("\nProblems:")
        for problem in problems:
            print(" ", problem)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(" ", warning)

    print(f"Saved detailed curve checks to: {out}")
    if problems or len(flagged):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
