"""
Main entry point. Run this on the HPC once your raw files are unzipped
under DATA_ROOT (default: /work3/$USER/battery_datasets).

    python build_dataset.py

Outputs (written to OUTPUT_DIR):
    cell_metadata_all.csv       every cell, before cleaning
    cell_metadata_cleaned.csv   after the three-step filter - this is the
                                 dataset to actually train on
    cycle_early_curves.parquet  raw voltage/current/capacity curve at cycle
                                 EFC_EARLY, for every surviving cell (variable
                                 length per cell - not padded/resampled)
    cycle_late_curves.parquet   same, at cycle EFC_LATE

*** BEFORE RUNNING, CHECK THE GLOB PATTERNS BELOW ***
I don't know your exact extracted folder/file layout for each dataset, so
the patterns below are best guesses based on what we discussed. If a
dataset shows "0 files found", fix its pattern (print(list(Path(...).glob))
is your friend) rather than assuming something's broken elsewhere.
"""

import os
import traceback
from pathlib import Path

import pandas as pd

from project.src.data.data_processing.parsers import (
    parse_cmu_vtc6, parse_vtc5a_mat, parse_lgmj1,
    parse_samsung25r, parse_samsung25r_filename, parse_snl_xlsx,
)
from project.src.data.data_processing.process_cell import process_cell
from project.src.data.data_processing.config_and_cleaning import apply_three_step_cleaning

DATA_ROOT = Path(f"/work3/claho/battery_datasets")
OUTPUT_DIR = Path(f"/work3/claho/battery_datasets/processed")

# Exact cycle_number values to pull voltage curves from. NOT necessarily
# the "first" cycle in the data - some sources are 0-indexed. Check
# cell_metadata_all.csv / a quick df["cycle_number"].unique() on one parsed
# cell if you're unsure which convention a given source uses.
EFC_EARLY = 1
EFC_LATE = 50


def safe_process(df, cell_id, source_dataset, errors):
    """Wraps process_cell so one bad file doesn't kill the whole run."""
    try:
        return process_cell(df, cell_id, source_dataset, efc_early=EFC_EARLY, efc_late=EFC_LATE)
    except Exception as e:
        errors.append((cell_id, source_dataset, str(e), traceback.format_exc()))
        return None, None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    efc_early_frames = []
    efc_late_frames = []
    errors = []

    # --- Sony-VTC6 (CMU) ---
    vtc6_dir = DATA_ROOT / "sony_vtc6"
    vtc6_files = list(vtc6_dir.glob("**/*.csv"))
    # skip the *_impedance.csv companion files - not cycling data
    vtc6_files = [f for f in vtc6_files if "impedance" not in f.name.lower()]
    print(f"[cmu] found {len(vtc6_files)} files under {vtc6_dir}")
    for f in vtc6_files:
        cell_id = f"cmu_{f.stem}"
        try:
            df = parse_cmu_vtc6(f)
        except Exception as e:
            errors.append((cell_id, "cmu", str(e), traceback.format_exc()))
            continue
        meta, efc_early, efc_late = safe_process(df, cell_id, "cmu", errors)
        if meta:
            metadata_rows.append(meta)
            if efc_early is not None:
                efc_early_frames.append(efc_early)
            if efc_late is not None:
                efc_late_frames.append(efc_late)

    # --- Sony-VTC5A (TUM) - cyclic/dynamic only, calendar aging excluded ---
    tum_dir = DATA_ROOT / "sony_vtc5a_raw"
    tum_files = list(tum_dir.glob("**/*.mat"))
    tum_files = [f for f in tum_files if "calendar" not in str(f).lower()]
    print(f"[tum] found {len(tum_files)} cyclic/dynamic files under {tum_dir} "
          f"(calendar-aging files excluded)")
    for f in tum_files:
        cell_id = f"tum_{f.stem}"
        try:
            df = parse_vtc5a_mat(f)
        except Exception as e:
            errors.append((cell_id, "tum", str(e), traceback.format_exc()))
            continue
        meta, efc_early, efc_late = safe_process(df, cell_id, "tum", errors)
        if meta:
            metadata_rows.append(meta)
            if efc_early is not None:
                efc_early_frames.append(efc_early)
            if efc_late is not None:
                efc_late_frames.append(efc_late)

    # --- LG-MJ1 (4TU) - NEEDS temperature per file, not yet wired up ---
    elt_dir = DATA_ROOT / "lg_mj1"
    elt_files = list(elt_dir.glob("**/*.csv"))
    print(f"[elt] found {len(elt_files)} files under {elt_dir}")
    if elt_files:
        print("  [SKIPPED] parse_lgmj1() needs temp_C_nameplate per file, decoded "
              "from filename/folder (25C vs 45C) - fill in FILENAME_TO_TEMP below "
              "once you can see real LG-MJ1 filenames, then remove this skip.")
        try:
            df = parse_lgmj1(f)
        except Exception as e:
             errors.append((cell_id, "lg_mj1", str(e), traceback.format_exc()))
        meta, efc0, efc50 = safe_process(df, cell_id, "elt", errors)
        if meta:
            metadata_rows.append(meta)
            if efc0 is not None:
                efc_early_frames.append(efc0)
            if efc50 is not None:
                efc_late_frames.append(efc50)
    # Example of what this loop should look like once filenames are known:
    # FILENAME_TO_TEMP = {"...25C...": 25.0, "...45C...": 45.0}
    # for f in elt_files:
    #     cell_id = f"elt_{f.stem}"
    #     temp = next((t for pat, t in FILENAME_TO_TEMP.items() if pat in f.name), None)
    #     if temp is None:
    #         errors.append((cell_id, "elt", "no temperature match for filename", ""))
    #         continue
    #     df = parse_lgmj1(f, temp_C_nameplate=temp)
    #     meta, efc_early, efc_late = safe_process(df, cell_id, "elt", errors)
    #     ...

    # --- Samsung-25R (Zhu et al.) ---
    tongji_dir = DATA_ROOT / "samsung_25r"
    tongji_files = list(tongji_dir.glob("**/*.csv"))
    print(f"[tongji] found {len(tongji_files)} files under {tongji_dir}")
    for f in tongji_files:
        cell_id = f"tongji_{f.stem}"
        try:
            temp_C, charge_c_nameplate, discharge_c_nameplate = parse_samsung25r_filename(f.name)
            df = parse_samsung25r(f)
            df["temp_C"] = temp_C  # filename-derived, more reliable than any per-row estimate
        except Exception as e:
            errors.append((cell_id, "tongji", str(e), traceback.format_exc()))
            continue
        meta, efc_early, efc_late = safe_process(df, cell_id, "tongji", errors)
        if meta:
            metadata_rows.append(meta)
            if efc_early is not None:
                efc_early_frames.append(efc_early)
            if efc_late is not None:
                efc_late_frames.append(efc_late)

    # --- Sandia SNL (A123-M1A + LG-HG2) - UNVERIFIED parser, see parsers.py ---
    snl_dir = DATA_ROOT / "sandia_snl"
    snl_files = list(snl_dir.glob("**/*.csv"))
    print(f"[snl] found {len(snl_files)} files under {snl_dir}")
    if snl_files:
        print("  Attempting parse with UNVERIFIED schema - failures here are "
              "expected until confirmed against a real file (see parsers.py).")
    for f in snl_files:
        # A123 cells are LFP chemistry (snl_a), LG-HG2 cells are NMC (snl_b) -
        # adjust this split once you know the real filename convention.
        if "lfp" in f.name.lower():
            source = "snl_a"
        elif  "nmc" in f.name.lower():
            source = "snl_b"
        cell_id = f"{source}_{f.stem}"
        try:
            df = parse_snl_xlsx(f)
        except Exception as e:
            errors.append((cell_id, source, str(e), traceback.format_exc()))
            continue
        meta, efc_early, efc_late = safe_process(df, cell_id, source, errors)
        if meta:
            metadata_rows.append(meta)
            if efc_early is not None:
                efc_early_frames.append(efc_early)
            if efc_late is not None:
                efc_late_frames.append(efc_late)

    # --- assemble + clean + save ---
    if not metadata_rows:
        print("\nNo cells were successfully parsed. Check the errors below and "
              "the directory patterns at the top of this script.")
    else:
        cells_df = pd.DataFrame(metadata_rows)
        cells_df.to_csv(OUTPUT_DIR / "cell_metadata_all.csv", index=False)
        print(f"\nParsed {len(cells_df)} cells total -> "
              f"{OUTPUT_DIR / 'cell_metadata_all.csv'}")

        cleaned_df = apply_three_step_cleaning(cells_df)
        cleaned_df.to_csv(OUTPUT_DIR / "cell_metadata_cleaned.csv", index=False)
        print(f"Cleaned dataset -> {OUTPUT_DIR / 'cell_metadata_cleaned.csv'}")

        surviving_ids = set(cleaned_df["cell_id"])
        if efc_early_frames:
            efc_early_all = pd.concat(
                [d for d in efc_early_frames if d["cell_id"].iloc[0] in surviving_ids],
                ignore_index=True,
            )
            efc_early_all.to_parquet(OUTPUT_DIR / "cycle_early_curves.parquet", index=False)
            print(f"cycle EFC_EARLY curves for surviving cells -> {OUTPUT_DIR / 'cycle_early_curves.parquet'}")
        else:
            print(f"No cell had data at cycle_number == {EFC_EARLY} - check EFC_EARLY "
                  f"against real cycle numbering for these sources (see README).")

        if efc_late_frames:
            efc_late_all = pd.concat(
                [d for d in efc_late_frames if d["cell_id"].iloc[0] in surviving_ids],
                ignore_index=True,
            )
            efc_late_all.to_parquet(OUTPUT_DIR / "cycle_late_curves.parquet", index=False)
            print(f"cycle EFC_LATE curves for surviving cells -> {OUTPUT_DIR / 'cycle_late_curves.parquet'}")

    if errors:
        err_df = pd.DataFrame(errors, columns=["cell_id", "source_dataset", "error", "traceback"])
        err_df.to_csv(OUTPUT_DIR / "parse_errors.csv", index=False)
        print(f"\n{len(errors)} files failed to parse - see "
              f"{OUTPUT_DIR / 'parse_errors.csv'} for details "
              f"(first column has the short error message).")


if __name__ == "__main__":
    main()
