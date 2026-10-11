"""
Equivalent of build_farasis_test_set.py, but for the six public
cylindrical-cell training datasets (SNL, CMU/VTC6, TUM/VTC5A, LG-MJ1,
Samsung-25R) instead of the Farasis test set.

Per the simplified scope: the 'interpreter' only needs the CHARGING
voltage curve at cycle 1 and cycle 50 (not the full multi-stage curve),
and the 'oracle' only needs temperature + charge/discharge C-rate (already
computed per cell by process_cell() via config_and_cleaning.py). Unlike
the Farasis test set, these ARE run through the three-step cleaning
(C-rate > 10C, cycle life < 70, cluster size <= 6), since that's what the
paper's methodology applies to the training corpus.

*** SNL note - no current column, but confirmed charging-only anyway ***
voltage_timeseries_all_cells.csv (SNL's real format) has no current
column, so we can't verify charge-vs-discharge by current sign the way we
can for the other four sources. However, checking real data confirms
every cell/cycle in this file is a pure, monotonically-rising voltage
curve (0% decreasing steps across every cell/cycle checked, spanning from
near the cell chemistry's low voltage bound to its high one) - i.e. this
export already IS charging-only, just without a current column to prove
it independently. Passed through as-is rather than filtered (there's
nothing to filter).

*** SNL cycle_life is still pending ***
As established earlier: SNL's cycle_summary_all_cells.csv needs the
periodic-capacity-check-cluster logic resolved (the paper's documented
"3 cycles per capacity check" protocol) before cycle_life can be computed
correctly. That work isn't finished, so SNL cells still won't survive the
cycle-life filter in this script until it is.

Usage:
    python build_public_charging_dataset.py
(reads from DATA_ROOT, same layout as build_dataset.py)
"""

import numpy as np
import traceback
from pathlib import Path

import pandas as pd

from parsers import (
    parse_cmu_vtc6, parse_lgmj1, parse_lgmj1_filename,
    parse_samsung25r, parse_samsung25r_filename, parse_snl_voltage_timeseries, parse_snl_cycle_summary
)
from snl_processing import process_snl
from tum_pipeline import build_index, process_cell_vtc5a 
from tum_processing import read_overview
from process_cell import process_cell
from config_and_cleaning import apply_three_step_cleaning, build_cell_record, calculate_efc, find_capacity_checks, NOMINAL_CAPACITY_AH

DATA_ROOT = Path(f"/work3/claho/battery_datasets")
OUTPUT_DIR = Path(f"/work3/claho/battery_datasets/processed")

EFC_EARLY = 1
EFC_LATE = 50


def add_charging_curves(early, late, cell_id, early_frames, late_frames):

    # Only add if both curves exist
    if early is not None and len(early) and late is not None and len(late):
        early["cell_id"] = cell_id
        late["cell_id"] = cell_id
        early_frames.append(early)
        late_frames.append(late)
    else:
        print(f"[{cell_id}] - missing or empty efc curve")
        print(f"          {len(early)},\n      {len(late)} ")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    charging_early_frames = []
    charging_late_frames = []
    errors = []

    # --- Sony-VTC6 (CMU) ---
    
    vtc6_dir = DATA_ROOT / "sony_vtc6"
    vtc6_files = [f for f in vtc6_dir.glob("**/*.csv") if "impedance" not in f.name.lower()]
    print(f"[cmu] found {len(vtc6_files)} files")
    for f in vtc6_files:
        cell_id = f"cmu_{f.stem}"
        try:
            df = parse_cmu_vtc6(f)
            meta, efc0, efc50 = process_cell(df, cell_id, "cmu", efc_early=EFC_EARLY, efc_late=EFC_LATE)
            metadata_rows.append(meta)

            add_charging_curves(efc0, efc50, cell_id, charging_early_frames, charging_late_frames)
        except Exception as e:
            errors.append((cell_id, "cmu", str(e), traceback.format_exc()))

    # --- Sony-VTC5A (TUM), cyclic/dynamic only ---

    tum_dir = DATA_ROOT / "sony_vtc5a"
    idx = build_index(tum_dir)
    overview = read_overview(tum_dir / "bawaii_cell_overview_publication.xlsx")   # from build_cell_csvs
    keep = overview[overview["aging_domain"].str.lower().str.startswith("cycl")
                    | overview["aging_domain"].str.lower().str.startswith("dyn")]
    cells = [f"BW-VTC-{n:03d}" for n in keep.index]
    print(f"[tum] found {len(cells)} cyclic/dynamic files")
    for cell in cells:
        cell_id = f"tum_{cell}"
        try:
            meta, efc0, efc50 = process_cell_vtc5a(cell, idx)
            metadata_rows.append(meta)
            add_charging_curves(efc0, efc50, cell_id, charging_early_frames, charging_late_frames)
        except Exception as e:
            errors.append((cell_id, "tum", str(e), traceback.format_exc()))
  
    # --- LG-MJ1 (4TU) ---
    elt_dir = DATA_ROOT / "lg_mj1"
    elt_files = list(elt_dir.glob("**/*.csv"))
    print(f"[elt] found {len(elt_files)} files")
    
    for f in elt_files:
        cell_id = f"elt_{f.stem}"
        try:
            temp_C, chrg_c, dchrg_c = parse_lgmj1_filename(f)
            df = parse_lgmj1(f)
            if chrg_c == None:
                meta, efc0, efc50 = process_cell(df, cell_id, "elt", efc_early=EFC_EARLY, efc_late=EFC_LATE)
            else:
                meta, efc0, efc50 = process_cell(df, cell_id, "elt", efc_early=EFC_EARLY, efc_late=EFC_LATE, charge_c_rate_override = chrg_c, discharge_c_rate_override= dchrg_c)
            metadata_rows.append(meta)
            add_charging_curves(efc0, efc50, cell_id, charging_early_frames, charging_late_frames)
        except Exception as e:
            errors.append((cell_id, "elt", str(e), traceback.format_exc()))
   
    # --- Samsung-25R (Zhu et al.) ---
    tongji_dir = DATA_ROOT / "samsung_25r"
    tongji_files = list(tongji_dir.glob("**/*.csv"))
    print(f"[tongji] found {len(tongji_files)} files")
    for f in tongji_files:
        cell_id = f"tongji_{f.stem}"
        try:
            temp_C, chrg_c, dchrg_c = parse_samsung25r_filename(f.name)
            df = parse_samsung25r(f)
            df["temp_C"] = temp_C
            meta, efc0, efc50 = process_cell(df, cell_id, "tongji", efc_early=EFC_EARLY, efc_late=EFC_LATE, charge_c_rate_override = chrg_c, discharge_c_rate_override= dchrg_c)
            metadata_rows.append(meta)

            add_charging_curves(efc0, efc50, cell_id, charging_early_frames, charging_late_frames)
        except Exception as e:
            errors.append((cell_id, "tongji", str(e), traceback.format_exc()))

    # --- Sandia SNL (LFP/NMC) - voltage-only, no current column ---
    
    snl_dir = DATA_ROOT / "sandia_snl" 
    snl_files = [f for f in snl_dir.glob("**/*.csv") if "all" not in str(f).lower()]
    snl_meta_data = parse_snl_cycle_summary(snl_dir/"query26_summary_all_cells.csv")
    print(f"[snl] found {len(snl_files)}")

    for f in snl_files:
        try:
            snl_cells = parse_snl_voltage_timeseries(f)
            
            for cell_id, info in snl_cells.items():
                meta, efc_0, efc_50 = process_snl(info, snl_meta_data, cell_id)
                metadata_rows.append(meta)
                    
                add_charging_curves(efc_0, efc_50, cell_id, charging_early_frames, charging_late_frames)    
            
        except Exception as e:
            errors.append((cell_id, "snl", str(e), traceback.format_exc()))

   
    # --- assemble + clean + save ---
    if not metadata_rows:
        print("\nNo cells were successfully parsed.")
    else:
        cells_df = pd.DataFrame(metadata_rows)
        cells_df.to_csv(OUTPUT_DIR / "public_cell_metadata_all.csv", index=False)
        print(f"\nParsed {len(cells_df)} cells -> {OUTPUT_DIR / 'public_cell_metadata_all.csv'}")

        cleaned_df = apply_three_step_cleaning(cells_df)
        cleaned_df.to_csv(OUTPUT_DIR / "public_cell_metadata_cleaned.csv", index=False)
        print(f"Cleaned (oracle-ready) metadata -> {OUTPUT_DIR / 'public_cell_metadata_cleaned.csv'}")

        surviving_ids = set(cleaned_df["cell_id"])
        for name, frames in [("charging_curve_early", charging_early_frames),
                              ("charging_curve_late", charging_late_frames)]:
            to_concat = [d for d in frames if d["cell_id"].iloc[0] in surviving_ids]
            if to_concat:
                pd.concat(to_concat, ignore_index=True).to_csv(
                    OUTPUT_DIR / f"public_{name}.csv", index=False)
                print(f"{name} -> {OUTPUT_DIR / f'public_{name}.csv'}")
            else:
                print(f"No surviving cell had data for {name}.")


    if errors:
        pd.DataFrame(errors, columns=["cell_id", "source_dataset", "error", "traceback"]).to_csv(
            OUTPUT_DIR / "public_parse_errors.csv", index=False)
        print(f"\n{len(errors)} files failed to parse - see {OUTPUT_DIR / 'public_parse_errors.csv'}")


if __name__ == "__main__":
    main()