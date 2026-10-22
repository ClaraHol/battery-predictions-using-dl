"""Check completeness and consistency of extracted Sony VTC5A RPT curves.

Usage: python verify_vtc5a.py --data-dir /work3/s214728/battery-predictions/processed_data/vtc5a

Checks both parquet files and writes vtc5a_quality_report.csv. A flagged
curve is a prompt for inspection, not necessarily an invalid measurement.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CURVE_TYPES = ("efc0", "efc50_lower", "efc50_upper")


def curve_diagnostics(frame):
    issues = []
    frame = frame.sort_index()  # preserve extraction order, do not reorder by time
    t = frame["time_s"].to_numpy(dtype=float)
    v = frame["voltage_V"].to_numpy(dtype=float)
    i = frame["current_A"].to_numpy(dtype=float)
    if len(t) < 2:
        return dict(n=len(t), duration_h=np.nan, v_start=np.nan, v_end=np.nan,
                    i_median=np.nan, capacity_Ah=np.nan, issues="Too few samples")
    if not (np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(i).all()):
        issues.append("Non-finite values")
    dt = np.diff(t)
    duration = (t[-1]-t[0])/3600
    median_current = float(np.median(i))
    capacity = float(np.trapezoid(i, t)/3600) if hasattr(np, 'trapezoid') else float(np.trapz(i,t)/3600)
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
    if abs(median_current-2.5) > 0.15:
        issues.append("Unexpected median current")
    if np.mean(np.abs(i-2.5)>0.30) > 0.05:
        issues.append(">5% samples far from reference current")
    return dict(n=len(t), duration_h=duration, v_start=v[0], v_end=v[-1],
                i_median=median_current, capacity_Ah=capacity, issues="; ".join(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/work3/s214728/battery-predictions/processed_data/vtc5a"))
    args = parser.parse_args()
    root = args.data_dir
    curves = pd.read_parquet(root / "vtc5a_all_curves.parquet")
    metadata = pd.read_parquet(root / "vtc5a_all_metadata.parquet")

    problems = []
    rows = []
    if metadata["cell_id"].duplicated().any():
        problems.append("Duplicate cell IDs in metadata")
    metadata_ids = set(metadata["cell_id"])
    curves_ids = set(curves["cell_id"])
    if curves_ids != metadata_ids:
        problems.append(f"Cell IDs differ: only in metadata={sorted(metadata_ids-curves_ids)}, "
                        f"only in curves={sorted(curves_ids-metadata_ids)}")
    if (metadata["chemistry"] != "VTC5A").any() or (curves["chemistry"] != "VTC5A").any():
        problems.append("Incorrect chemistry label")
    if (metadata["direction"] != "discharge").any():
        problems.append("Unexpected direction")

    meta_lookup = metadata.drop_duplicates("cell_id").set_index("cell_id")
    for cell_id in sorted(metadata_ids | curves_ids):
        selected = curves.loc[curves["cell_id"] == cell_id]
        observed = set(selected["curve_type"])
        if observed != set(CURVE_TYPES):
            problems.append(f"{cell_id}: expected 3 types, got {sorted(observed)}")
        if cell_id in meta_lookup.index:
            m = meta_lookup.loc[cell_id]
            if not (np.isfinite([m['efc0'],m['efc50_lower'],m['efc50_upper'],
                                 m['weight_lower'],m['weight_upper']]).all()):
                problems.append(f"{cell_id}: non-finite EFC/weights")
            elif not (m["efc50_lower"] < 50 < m["efc50_upper"]):
                problems.append(f"{cell_id}: EFC50 unbracketed")
            else:
                expected_upper = (50-m["efc50_lower"])/(m["efc50_upper"]-m["efc50_lower"])
                if not np.isclose(m["weight_upper"], expected_upper, atol=1e-6):
                    problems.append(f"{cell_id}: incorrect upper interpolation weight")
                if not np.isclose(m["weight_lower"]+m["weight_upper"], 1, atol=1e-6):
                    problems.append(f"{cell_id}: weights do not sum to 1")
            if not np.isclose(m["efc0"], 0, atol=1e-6):
                problems.append(f"{cell_id}: EFC0 not zero")

        for curve_type, frame in selected.groupby("curve_type", sort=False):
            r = curve_diagnostics(frame)
            if cell_id in meta_lookup.index and curve_type in CURVE_TYPES:
                expected_efc = meta_lookup.loc[cell_id, {"efc0":"efc0", "efc50_lower":"efc50_lower", "efc50_upper":"efc50_upper"}[curve_type]]
                if not np.allclose(frame["rpt_efc"].to_numpy(), expected_efc, atol=1e-6):
                    r["issues"] += ("; " if r["issues"] else "") + "Curve EFC differs from metadata"
            r.update(cell_id=cell_id, curve_type=curve_type, status="CHECK" if r["issues"] else "OK")
            rows.append(r)
            print(f"{cell_id:17} {curve_type:13} N={r['n']:4} "
                  f"T={r['duration_h']:5.3f}h V={r['v_start']:.3f}->{r['v_end']:.3f} "
                  f"I={r['i_median']:.3f}A Q={r['capacity_Ah']:.3f}Ah {r['status']}")
            if r["issues"]:
                print(f"   -> {r['issues']}")

    report = pd.DataFrame(rows)
    out = root / "vtc5a_quality_report.csv"
    report.to_csv(out, index=False)
    flagged = report.loc[report["status"] == "CHECK"] if not report.empty else report
    log_path = root / "vtc5a_processing_log.csv"
    skipped = None
    if log_path.is_file():
        processing_log = pd.read_csv(log_path)
        skipped = processing_log.loc[processing_log["status"] == "SKIPPED"]

    print("\n========== SUMMARY ==========")
    print(f"Cells in metadata: {len(metadata)}")
    print(f"Extracted curves: {len(report)} / {3*len(metadata)} expected")
    print(f"Flagged curves: {len(flagged)}")
    if skipped is not None:
        print(f"Cells skipped during extraction: {len(skipped)}")
        for r in skipped.itertuples(index=False):
            print(f"  SKIPPED {r.cell_id}: {r.reason}")
    for p in problems:
        print("  ISSUE:", p)
    print(f"Saved detailed checks to: {out}")
    if problems or len(flagged):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
