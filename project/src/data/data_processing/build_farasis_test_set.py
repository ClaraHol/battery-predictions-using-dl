"""
Processes the Farasis pouch-cell TEST SET (Discovery Learning's own 123
cells) - separate from build_dataset.py, which only handles the six public
training datasets. The test set does NOT get the three-step cleaning (see
README) - this script just standardizes it into the same kind of output.

Expects, for each cell, a pair of files following the pattern already
established for cell 019:
    <cell_id>_single_parquet_with_soc_and_corrections.parquet
    <cell_id>_dchg_cap_rpt.xlsx

*** KNOWN LIMITATION - read before using cycle_life_rpt_units ***
cycle_life is reported in RPT-INDEX units ("reached 90% of initial capacity
by RPT N"), NOT equivalent full cycles. We tried to convert RPT number to
EFC count using the parquet's total_dischg_thrgh_ah_rounded column, but it
gave physically implausible values (large negative numbers) for cell 019,
so that conversion is left unresolved rather than guessed at. If you figure
out the right way to get EFC counts (e.g. from other metadata, or from
understanding what that throughput column actually represents), the fix
goes in build_farasis_cell_record() in parsers.py.

Only RPT0's curve is extracted right now (not RPT-at-50-EFC), for the same
reason - we don't know which RPT number corresponds to ~50 EFC yet.

Usage:
    python build_farasis_test_set.py /path/to/farasis/raw/files/
"""

import sys
import os
from pathlib import Path

import pandas as pd

from parsers import parse_farasis_rpt_metadata, parse_farasis_rpt_curve, build_farasis_cell_record


DATA_ROOT = Path(f"/work3/claho/battery_datasets")
OUTPUT_DIR = Path(f"/work3/claho/battery_datasets/processed")


def find_cell_pairs(raw_dir: Path):
    """Match up parquet + xlsx pairs by the leading cell_id in each filename."""
    pairs = {}
    for f in raw_dir.glob("**/*_dchg_cap_rpt.xlsx"):
        cell_id = f.name.split("_dchg_cap_rpt.xlsx")[0]
        pairs.setdefault(cell_id, {})["metadata"] = f
    for f in raw_dir.glob("**/*_single_parquet_with_soc_and_corrections.parquet"):
        cell_id = f.name.split("_single_parquet_with_soc_and_corrections.parquet")[0]
        pairs.setdefault(cell_id, {})["parquet"] = f
    complete = {cid: files for cid, files in pairs.items() if "metadata" in files and "parquet" in files}
    incomplete = {cid: files for cid, files in pairs.items() if cid not in complete}
    return complete, incomplete


def main(raw_dir: str):
    raw_dir = DATA_ROOT / raw_dir
    OUTPUT_DIR.mkdir(exist_ok=True)

    complete, incomplete = find_cell_pairs(raw_dir)
    if incomplete:
        print(f"Skipping {len(incomplete)} cell(s) missing one of the two required files: "
              f"{list(incomplete.keys())}")
    print(f"Processing {len(complete)} cell(s) with both files present.\n")

    metadata_rows = []
    rpt0_frames = []

    for cell_id, files in complete.items():
        print(f"[{cell_id}]")
        rpt_meta = parse_farasis_rpt_metadata(files["metadata"])
        record = build_farasis_cell_record(cell_id, rpt_meta)
        metadata_rows.append(record)
        print(f"  nominal_capacity_Ah={record['nominal_capacity_Ah']:.2f}, "
              f"cycle_life_rpt_units={record['cycle_life_rpt_units']}")

        curve = parse_farasis_rpt_curve(files["parquet"], rpt_number=12)
        curve["cell_id"] = cell_id
        rpt0_frames.append(curve)
        print(f"  RPT0 curve: {len(curve)} rows")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(OUTPUT_DIR / "farasis_cell_metadata.csv", index=False)
    print(f"\nMetadata -> {OUTPUT_DIR / 'farasis_cell_metadata_12.csv'}")

    if rpt0_frames:
        all_rpt0 = pd.concat(rpt0_frames, ignore_index=True)
        all_rpt0.to_parquet(OUTPUT_DIR / "farasis_rpt0_curves.parquet", index=False)
        print(f"RPT0 curves -> {OUTPUT_DIR / 'farasis_rpt12_curves.parquet'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
