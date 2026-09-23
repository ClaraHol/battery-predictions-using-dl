"""
Processes the Farasis pouch-cell TEST SET (Discovery Learning's own 123
cells) - separate from build_dataset.py, which only handles the six public
training datasets. The test set does NOT get the three-step cleaning (see
README) - this script just standardizes it into the same kind of output.

Expects, for each cell, a pair of files following the pattern already
established for cell 019:
    <cell_id>_single_parquet_with_soc_and_corrections.parquet
    <cell_id>_dchg_cap_rpt.xlsx

***  Cycle-50 handling ***
Per the paper: "the feature at the 50th equivalent full cycle is generally
obtained by interpolating between the features corresponding to the two
nearest reference performance tests before and after it." So this script
no longer extracts a single "cycle 50" RPT curve - it extracts BOTH
bracketing RPTs (e.g. RPT0 and RPT1, if EFC 50 falls between them, which
is common - a single round of cycling for cell 019 covers ~69 EFC, more
than 50), plus the interpolation weight between them. Turning that into
an actual "EFC50 feature" means computing your scalar feature(s) from each
bracketing RPT's curve separately, then combining them with:
    interpolated = (1 - weight) * feature[rpt_before] + weight * feature[rpt_after]
This script does NOT do that combination itself, since it depends on which
features you actually want - it hands you both raw curves and the weight.

Usage:
    python build_pouch_cells_test_set.py /path/to/pouch_cells/raw/files/
"""

import sys
from pathlib import Path

import pandas as pd

from parsers import parse_pouch_cells_rpt_metadata, parse_pouch_cells_rpt_curve, build_pouch_cells_cell_record, extract_primary_charging_curve

DATA_ROOT = Path(f"/work3/claho/battery_datasets")
OUTPUT_DIR = Path(f"/work3/claho/battery_datasets/processed/pouch_cells")

TARGET_EFC = 50


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
    raw_dir = Path(raw_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    complete, incomplete = find_cell_pairs(raw_dir)
    if incomplete:
        print(f"Skipping {len(incomplete)} cell(s) missing one of the two required files: "
              f"{list(incomplete.keys())}")
    print(f"Processing {len(complete)} cell(s) with both files present.\n")

    metadata_rows = []
    rpt0_frames = []
    efc50_before_frames = []
    efc50_after_frames = []


    for cell_id, files in complete.items():
        print(f"[{cell_id}]")
        rpt_meta = parse_pouch_cells_rpt_metadata(files["metadata"])
        record = build_pouch_cells_cell_record(
            cell_id, rpt_meta, parquet_path=str(files["parquet"]), target_efc=TARGET_EFC
        )
        metadata_rows.append(record)
        print(f"  nominal_capacity_Ah={record['nominal_capacity_Ah']:.2f}, "
              f"cycle_life_rpt_units={record['cycle_life_rpt_units']}, "
              f"cycle_life_efc_units={record['cycle_life_efc_units']:.1f}")
        print(f"  EFC{TARGET_EFC} brackets: RPT{record['efc50_rpt_before']} <-> "
              f"RPT{record['efc50_rpt_after']} (weight={record['efc50_interp_weight']:.3f})")

       
        curve0 = parse_pouch_cells_rpt_curve(files["parquet"], rpt_number=0)
        efc0 = extract_primary_charging_curve(curve0) # Extract the charging curve
        efc0["cell_id"] = cell_id
        rpt0_frames.append(efc0)

        rpt_before, rpt_after = record["efc50_rpt_before"], record["efc50_rpt_after"]

        curve_before = parse_pouch_cells_rpt_curve(files["parquet"], rpt_number=rpt_before)
        efc50_before = extract_primary_charging_curve(curve_before) # Extract the charging curve
        efc50_before["cell_id"] = cell_id
        efc50_before_frames.append(efc50_before)
        print(f"  Efc 50 before step number: {efc50_before["step_no"].iloc[0]}")

        if rpt_after != rpt_before:
            curve_after = parse_pouch_cells_rpt_curve(files["parquet"], rpt_number=rpt_after)
            efc50_after = extract_primary_charging_curve(curve_after) # Extract the charging curve
            print(f"  Efc 50 after step number: {efc50_after["step_no"].iloc[0]}")
            efc50_after["cell_id"] = cell_id
            efc50_after_frames.append(efc50_after) 
        print(f"  RPT0 curve: {len(efc0)} rows, "
              f"EFC{TARGET_EFC}-before curve: {len(efc50_before)} rows, "
              f"EFC{TARGET_EFC}-after curve: {len(efc50_after)} rows")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(OUTPUT_DIR / "pouch_cells_cell_metadata.csv", index=False)
    print(f"\nMetadata (includes efc50 bracketing RPTs + weight per cell) -> "
          f"{OUTPUT_DIR / 'pouch_cells_cell_metadata.csv'}")

    if rpt0_frames:
        pd.concat(rpt0_frames, ignore_index=True).to_parquet(OUTPUT_DIR / "pouch_cells_rpt0_curves.parquet", index=False)
        print(f"RPT0 curves -> {OUTPUT_DIR / 'pouch_cells_rpt0_curves.parquet'}")
    if efc50_before_frames:
        pd.concat(efc50_before_frames, ignore_index=True).to_parquet(
            OUTPUT_DIR / "pouch_cells_efc50_before_curves.parquet", index=False)
        print(f"EFC{TARGET_EFC}-before curves -> {OUTPUT_DIR / 'pouch_cells_efc50_before_curves.parquet'}")
    if efc50_after_frames:
        pd.concat(efc50_after_frames, ignore_index=True).to_parquet(
            OUTPUT_DIR / "pouch_cells_efc50_after_curves.parquet", index=False)
        print(f"EFC{TARGET_EFC}-after curves -> {OUTPUT_DIR / 'pouch_cells_efc50_after_curves.parquet'}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
