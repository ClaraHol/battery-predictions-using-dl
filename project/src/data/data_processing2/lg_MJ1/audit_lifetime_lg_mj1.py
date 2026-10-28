"""Audit LG-MJ1 lifetime/EFC definitions against the original Discovery Learning labels.

This script is intentionally an AUDIT tool. It does not modify the LG-MJ1
extractor and it does not decide which lifetime definition is correct.

It scans the VITO/EVERLASTING LG-MJ1 raw CSV files, reconstructs the RPT
summary using the current LG-MJ1 extractor, and evaluates several SOH
thresholds. The resulting lifetime candidates are compared with the original
LG-MJ1 lifetime labels supplied with Discovery Learning (data_mj1_efc0.npy).

Important:
- The original feature arrays contain 15 LG-MJ1 labels, but the raw VITO
  directory may contain more files than the 15 cells used by Discovery
  Learning. The script therefore compares distributions by default rather
  than assuming a row-by-row mapping between raw files and the original labels.
- Step is never used for chronology.
- The LG-MJ1 extractor's repaired cumulative time axis is used through
  _scan_file().
- The current EFC definition is bidirectional throughput divided by
  2 * 3.5 Ah, zeroed at the first check-up. This script sweeps SOH thresholds
  while keeping that EFC definition fixed. A separate EFC-convention audit
  should be performed if the threshold sweep cannot explain the discrepancy.

The original Discovery Learning code loads the precomputed lifetime labels
from the public feature arrays; it does not expose the source-dataset lifetime
calculation. Consequently, this script treats the original labels as the
reference distribution, not as a proof of how the labels were generated.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Editable paths / settings
# ---------------------------------------------------------------------------

# Directory containing the VITO LG-MJ1 CSV files.
RAW_DIR = Path("/work3/claho/battery_datasets/lg_mj1")

# Original Discovery Learning LG-MJ1 lifetime labels. The last column of
# data_mj1_efc0.npy is the cycle-life/EFC label.

ORIGINAL_LABELS_NPY = Path("/work3/claho/battery_datasets/pouch_cells/raw_feature_data/data_mj1_efc0.npy")

OUTPUT_CELL_AUDIT = Path("lg_mj1_lifetime_audit.csv")
OUTPUT_THRESHOLD_SUMMARY = Path("lg_mj1_lifetime_threshold_summary.csv")

# Candidate SOH definitions. The original paper defines cycle life generally
# as EFC until 90% capacity, but the VITO source protocol may use a different
# end-of-life convention. We sweep the plausible thresholds instead of
# assuming one.
SOH_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75)

# ---------------------------------------------------------------------------
# Import the chemistry-specific extractor.
# It is expected to live next to this audit script, as in the other project
# processing scripts.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_curve_lg_mj1 import (  # noqa: E402
    _group_checkups,
    _scan_file,
    _summary_from_checkups,
    parse_protocol,
)


FILE_PATTERNS = (
    "Cycl_T*_Vito_Cell*_AllData.csv",
    "DrivingAgeing_T*_Vito_Cell*_AllData.csv",
    "DrivingAgeingPower_T*_Vito_Cell*_AllData.csv",
)


def find_raw_files(raw_dir: Path) -> list[Path]:
    """Find LG-MJ1 source files without relying on directory ordering."""
    files: set[Path] = set()
    for pattern in FILE_PATTERNS:
        files.update(raw_dir.glob(pattern))
    return sorted(files)


def interpolate_lifetime(summary: pd.DataFrame, threshold: float) -> dict[str, object]:
    """Return the interpolated EFC at the first RPT crossing of an SOH threshold."""
    soh = summary["soh"].to_numpy(dtype=float)
    efc = summary["rpt_efc"].to_numpy(dtype=float)

    crossing = np.flatnonzero(soh <= threshold)
    if len(crossing) == 0:
        return {
            "cycle_life_efc": float(efc[-1]),
            "censored": True,
            "crossing_lower_rpt": int(len(summary) - 1),
            "crossing_upper_rpt": np.nan,
            "last_soh": float(soh[-1]),
        }

    upper_idx = int(crossing[0])
    if upper_idx == 0:
        return {
            "cycle_life_efc": 0.0,
            "censored": False,
            "crossing_lower_rpt": np.nan,
            "crossing_upper_rpt": 0,
            "last_soh": float(soh[-1]),
        }

    lower_idx = upper_idx - 1
    s0 = float(soh[lower_idx])
    s1 = float(soh[upper_idx])
    e0 = float(efc[lower_idx])
    e1 = float(efc[upper_idx])

    if s0 <= threshold or s1 > threshold:
        raise ValueError("Invalid SOH crossing bracket")

    if np.isclose(s1, s0):
        life = e1
    else:
        fraction = (threshold - s0) / (s1 - s0)
        life = e0 + fraction * (e1 - e0)

    return {
        "cycle_life_efc": float(life),
        "censored": False,
        "crossing_lower_rpt": int(lower_idx),
        "crossing_upper_rpt": int(upper_idx),
        "last_soh": float(soh[-1]),
    }


def summarize(values: pd.Series) -> dict[str, float]:
    """Return robust summary statistics for non-null lifetime values."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def load_original_labels(path: Path) -> np.ndarray:
    """Load the original LG-MJ1 labels from the final column of data_mj1_efc0."""
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected original feature-array shape: {data.shape}")
    labels = np.asarray(data[:, -1], dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError("Original LG-MJ1 labels contain non-finite values")
    return labels


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")
    if not ORIGINAL_LABELS_NPY.exists():
        raise FileNotFoundError(f"Original label file does not exist: {ORIGINAL_LABELS_NPY}")

    raw_files = find_raw_files(RAW_DIR)
    if not raw_files:
        raise FileNotFoundError(f"No LG-MJ1 VITO CSV files found in {RAW_DIR}")

    original_labels = load_original_labels(ORIGINAL_LABELS_NPY)
    original_summary = summarize(pd.Series(original_labels))

    print("========== LG-MJ1 LIFETIME AUDIT ==========")
    print(f"Raw CSV files found:        {len(raw_files)}")
    print(f"Original labels:            {len(original_labels)}")
    print(
        "Original label median:      "
        f"{original_summary['median']:.3f} EFC"
    )
    print()

    cell_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for file_path in raw_files:
        print(f"Scanning: {file_path.name}")
        try:
            protocol = parse_protocol(file_path)
            candidates, long_rests, stats = _scan_file(file_path)
            checkups = _group_checkups(candidates, long_rests)
            summary = _summary_from_checkups(checkups)

            row: dict[str, object] = {
                "source_file": file_path.name,
                "cell_number": protocol["cell_number"],
                "test_type": protocol["test_type"],
                "temperature_C": protocol["temperature_C"],
                "soc_lower_pct": protocol["soc_lower_pct"],
                "soc_upper_pct": protocol["soc_upper_pct"],
                "charge_c_rate_filename": protocol["charge_c_rate_filename"],
                "discharge_c_rate_filename": protocol["discharge_c_rate_filename"],
                "n_rpts": len(summary),
                "initial_capacity_Ah": float(summary.iloc[0]["capacity_Ah"]),
                "last_rpt_efc": float(summary.iloc[-1]["rpt_efc"]),
                "last_rpt_soh": float(summary.iloc[-1]["soh"]),
                "n_rows_scanned": int(stats["n_rows"]),
                "reconstructed_duration_h": float(stats["reconstructed_duration_s"] / 3600.0),
                "n_time_resets": int(stats["n_time_resets"]),
                "n_timestamp_time_repairs": int(stats["n_timestamp_time_repairs"]),
                "n_fallback_time_repairs": int(stats["n_fallback_time_repairs"]),
            }

            for threshold in SOH_THRESHOLDS:
                result = interpolate_lifetime(summary, threshold)
                tag = f"{int(round(threshold * 100)):02d}"
                row[f"life_{tag}_efc"] = result["cycle_life_efc"]
                row[f"life_{tag}_censored"] = result["censored"]
                row[f"life_{tag}_lower_rpt"] = result["crossing_lower_rpt"]
                row[f"life_{tag}_upper_rpt"] = result["crossing_upper_rpt"]

            cell_rows.append(row)
            print(
                f"  RPTs={len(summary):3d} "
                f"last EFC={summary.iloc[-1]['rpt_efc']:.2f} "
                f"last SOH={summary.iloc[-1]['soh']:.3f}"
            )

        except Exception as exc:
            failures.append((file_path.name, str(exc)))
            print(f"  FAILED: {exc}")

    audit = pd.DataFrame(cell_rows).sort_values(
        ["cell_number", "source_file"], kind="stable"
    )
    audit.to_csv(OUTPUT_CELL_AUDIT, index=False)

    summary_rows: list[dict[str, object]] = []
    for threshold in SOH_THRESHOLDS:
        tag = f"{int(round(threshold * 100)):02d}"
        values = audit[f"life_{tag}_efc"] if len(audit) else pd.Series(dtype=float)
        censored = audit[f"life_{tag}_censored"] if len(audit) else pd.Series(dtype=bool)
        stats = summarize(values)

        summary_rows.append(
            {
                "soh_threshold": threshold,
                "n_raw_cells_successfully_scanned": len(audit),
                "n_observed_crossings": int((~censored.astype(bool)).sum()) if len(audit) else 0,
                "n_censored": int(censored.astype(bool).sum()) if len(audit) else 0,
                "median_lifetime_efc": stats["median"],
                "mean_lifetime_efc": stats["mean"],
                "min_lifetime_efc": stats["min"],
                "max_lifetime_efc": stats["max"],
                "original_label_count": original_summary["n"],
                "original_label_median_efc": original_summary["median"],
                "median_difference_vs_original": (
                    stats["median"] - original_summary["median"]
                    if np.isfinite(stats["median"])
                    else np.nan
                ),
                "median_ratio_vs_original": (
                    stats["median"] / original_summary["median"]
                    if np.isfinite(stats["median"])
                    else np.nan
                ),
            }
        )

    threshold_summary = pd.DataFrame(summary_rows)
    threshold_summary.to_csv(OUTPUT_THRESHOLD_SUMMARY, index=False)

    print("\n========== THRESHOLD COMPARISON ==========")
    print(
        threshold_summary[
            [
                "soh_threshold",
                "n_observed_crossings",
                "n_censored",
                "median_lifetime_efc",
                "original_label_median_efc",
                "median_difference_vs_original",
                "median_ratio_vs_original",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\n========== FILE SUMMARY ==========")
    print(f"Successfully scanned: {len(audit)}")
    print(f"Failed:               {len(failures)}")
    print(f"Cell audit:           {OUTPUT_CELL_AUDIT}")
    print(f"Threshold summary:    {OUTPUT_THRESHOLD_SUMMARY}")

    if failures:
        print("\nFailures:")
        for name, reason in failures:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
