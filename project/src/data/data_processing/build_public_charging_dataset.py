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
    parse_cmu_vtc6, parse_vtc5a_mat, parse_lgmj1, parse_lgmj1_filename,
    parse_samsung25r, parse_samsung25r_filename, parse_snl_voltage_timeseries, parse_snl_cycle_summary
)
from process_cell import process_cell
from config_and_cleaning import apply_three_step_cleaning, build_cell_record, calculate_efc, find_capacity_checks, NOMINAL_CAPACITY_AH

DATA_ROOT = Path(f"/work3/claho/battery_datasets")
OUTPUT_DIR = Path(f"/work3/claho/battery_datasets/processed_test_2")

EFC_EARLY = 1
EFC_LATE = 50


def add_charging_curves(early, late, cell_id, early_frames, late_frames):
    if early is not None and len(early):
        early["cell_id"] = cell_id
        early_frames.append(early)

    if late is not None and len(late):
        late["cell_id"] = cell_id
        late_frames.append(late)


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
    tum_files = [f for f in tum_dir.glob("**/*.mat") if "calendar" not in str(f).lower()]
    print(f"[tum] found {len(tum_files)} cyclic/dynamic files")
    for f in tum_files:
        cell_id = f"tum_{f.stem}"
        try:
            
            df = parse_vtc5a_mat(f)
            meta, efc0, efc50 = process_cell(df, cell_id, "tum", efc_early=EFC_EARLY, efc_late=EFC_LATE)
            metadata_rows.append(meta)
            add_charging_curves(efc0, efc50, cell_id, charging_early_frames, charging_late_frames)
        except Exception as e:
            errors.append((cell_id, "tum", str(e), traceback.format_exc()))
  
    # --- LG-MJ1 (4TU) - still needs temp_C_nameplate per file, see build_dataset.py ---
    elt_dir = DATA_ROOT / "lg_mj1"
    elt_files = list(elt_dir.glob("**/*.csv"))
    print(f"[elt] found {len(elt_files)} files")
    
    for f in elt_files:
        cell_id = f"elt_{f.stem}"
        try:
            temp_C, chrg_c, dchrg_c = parse_lgmj1_filename(f)
            df = parse_lgmj1(f)
            df["temp_C"] = temp_C
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
    MAX_CYCLE = {20: 300, 60: 100, 100: 70}
    snl_dir = DATA_ROOT / "sandia_snl" 
    snl_files = [f for f in snl_dir.glob("**/*.csv") if "all" not in str(f).lower()]
    snl_meta_data = parse_snl_cycle_summary(snl_dir/"query26_summary_all_cells.csv")
    print(f"[snl] found {len(snl_files)}")

    for f in snl_files:
        print(f"file name = {f}")
        
        try:
            snl_cells = parse_snl_voltage_timeseries(f)
            
            for cell_id, info in snl_cells.items():
                
                diff = info["dod_high_pct"] - info["dod_low_pct"]
                max_cycle = MAX_CYCLE[diff]
                # Compute SOH from capacity test cycles
                info["discharge_capacity_Ah"] = snl_meta_data[cell_id]["discharge_capacity_Ah"].iloc[:max_cycle]

               
                nom_cap = NOMINAL_CAPACITY_AH[info["source_dataset"]]
    
                if info["dod_low_pct"]==0:
                    checks, cell_processed = find_capacity_checks(snl_meta_data[cell_id], nominal_capacity_Ah=nom_cap, min_fraction_of_nominal=1.5, outlier_fraction_of_nominal=2.5)
                else:
                    checks, cell_processed = find_capacity_checks(snl_meta_data[cell_id], nominal_capacity_Ah=nom_cap)

                Q_ref = checks["capacity_Ah"].iloc[0]
                checks["SOH_pct"] = checks["capacity_Ah"] / Q_ref * 100
                #print(checks)

                # Find first cycle below 90% SOH to compute cycle life
                first_90_cycle = next(
                    (
                        int(checks["first_cycle"].iloc[cycle])
                        for cycle, soh in checks["SOH_pct"].items()
                        if soh <= 90
                    ),
                    None,
                )
                print(f"{first_90_cycle=}")
            
                # Compute EFC
                info = calculate_efc(info, q_nom = nom_cap, cell_id="snl")
            
                
                
                full_cell_id = f"{info['source_dataset']}_{cell_id}"
                metadata_rows.append(build_cell_record(
                    cell_id=full_cell_id,
                    source_dataset=info["source_dataset"],
                    temp_C_raw=info["temp_C"],
                    charge_c_rate_raw=info["charge_c_rate"],
                    discharge_c_rate_raw=info["discharge_c_rate"],
                    cycle_life = first_90_cycle,
                    initial_discharge_capacity_Ah=snl_meta_data[cell_id]["ah_c"].iloc[0],
                ))
                efcs = np.asarray(info["EFC"])
                #print(efcs)
                for target, frames in [(EFC_EARLY, charging_early_frames), (EFC_LATE, charging_late_frames)]:
                    

                    if target == EFC_LATE:
                        closest = efcs[np.argmin(np.abs(efcs - target))]

                        if abs(closest - target) < 0.5:
                            print(f"{closest=}")
                            weight = 1
                            curve = pd.DataFrame({"lower_curve": info["voltage_curves"][closest].copy(), "upper_curve": info["voltage_curves"][closest].copy(), "weight":weight})
                   
                        else:
                            lower = efcs[efcs < target].max() if np.any(efcs < target) else None
                            upper = efcs[efcs > target].min() if np.any(efcs > target) else None
                            print(f"{lower=}")
                            print(f"{upper=}")
                            weight = (target - lower) / (upper - lower)
                            curve = pd.DataFrame({"lower_curve": info["voltage_curves"][lower].copy(), "upper_curve": info["voltage_curves"][upper].copy(), "weight":weight})
                            curve["cell_id"] = full_cell_id
                    else:
                        if target in info["EFC"]:
                            curve = info["voltage_curves"][target].copy()
                            curve["cell_id"] = full_cell_id
                        else:
                            continue

                        frames.append(curve)
                            #print(f"  {len(snl_cells)} cells parsed (cycle_life still pending, curves confirmed charging-only)")
            
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
                pd.concat(to_concat, ignore_index=True).to_parquet(
                    OUTPUT_DIR / f"public_{name}.parquet", index=False)
                print(f"{name} -> {OUTPUT_DIR / f'public_{name}.parquet'}")
            else:
                print(f"No surviving cell had data for {name}.")


    if errors:
        pd.DataFrame(errors, columns=["cell_id", "source_dataset", "error", "traceback"]).to_csv(
            OUTPUT_DIR / "public_parse_errors.csv", index=False)
        print(f"\n{len(errors)} files failed to parse - see {OUTPUT_DIR / 'public_parse_errors.csv'}")


if __name__ == "__main__":
    main()