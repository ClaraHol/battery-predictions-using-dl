"""
Takes one cell's standardized per-row cycling data (from any parser in
parsers.py) and produces:
  1. a metadata dict ready for config_and_cleaning.build_cell_record()
  2. the raw efc0 and efc50 cycle snapshots, for later feature engineering
     (kept separate from cleaning/metadata so re-running feature extraction
     later doesn't require re-parsing every raw file)
"""
import numpy as np
import pandas as pd
from cmu_processing import  cmu_cycle_life_efc
from lg_mj1_processing import lgmj1_cycle_life_from_df, discharge_runs
from config_and_cleaning import build_cell_record, compute_cycle_life_efc, compute_cc_c_rate, calculate_efc, find_capacity_checks


def process_cell(df: pd.DataFrame, cell_id: str, source_dataset: str,
                  efc_early: int = 1, efc_late: int = 50,
                  charge_c_rate_override: float = None,
                  discharge_c_rate_override: float = None):

    """
    df: standard-schema DataFrame for one cell (see parsers.py STANDARD_COLUMNS)
    Returns (metadata_dict, efc_early_df, efc_late_df)

    efc_early / efc_late: the exact cycle_number values to extract voltage curves for. 
    Either curve is None if that cycle number isn't present for this cell 
    (e.g. it died before reaching 50 - the metadata row is still kept, just without that snapshot).

    Curves are NOT forced to equal length - each is just every row in df matching that cycle_number
    """
    from config_and_cleaning import NOMINAL_CAPACITY_AH

    nominal_cap = NOMINAL_CAPACITY_AH[source_dataset]

    # cycle life: first cycle where per-cycle discharge capacity drops to 90% of the initial cycle's discharge capacity
    per_cycle_discharge_cap = df.groupby("cycle_number")["discharge_capacity_Ah"].max()
    df = calculate_efc(df, nominal_cap, cell_id=source_dataset)
    
    if source_dataset == "cmu":
        life = cmu_cycle_life_efc(df)
    elif source_dataset == "elt":
        runs = discharge_runs(df)
        runs[["efc_start", "efc_end"]] /= nominal_cap
        life = lgmj1_cycle_life_from_df(df, nominal_Ah=nominal_cap, pick="max")
    else:
        efc_by_cycle = df.groupby("cycle_number")["EFC"].max()
        life = compute_cycle_life_efc(df, per_cycle_discharge_cap, nominal_cap, efc_by_cycle=efc_by_cycle)
    

    # charge/discharge C-rate: mean over the whole cell's history. 
    # C-rates: constant-current estimate from the data (time-weighted mode per cycle), 
    #          overridden by filename values when supplied 
    charge_c_data = compute_cc_c_rate(df, nominal_cap, "charge")
    discharge_c_data = compute_cc_c_rate(df, nominal_cap, "discharge")
    charge_c = charge_c_rate_override if charge_c_rate_override is not None else charge_c_data
    discharge_c = discharge_c_rate_override if discharge_c_rate_override is not None else discharge_c_data
 
    def _mismatch(a, b):
        return b > 0 and abs(a - b) / b > 0.30

    # --- temperature: mean over the whole cell's history ---
    temp_C = float(df["temp_C"].mean())

    metadata = build_cell_record(
        cell_id=cell_id,
        source_dataset=source_dataset,
        temp_C_raw=temp_C,
        charge_c_rate_raw=charge_c,
        discharge_c_rate_raw=discharge_c,
        cycle_life=life["cycle_life"],
        initial_discharge_capacity_Ah=per_cycle_discharge_cap.iloc[0],
    )
    metadata.update({
        "cycle_life_censored": life["censored"],
        "n_full_cycles": life["n_full_cycles"],
        "c_rate_source": "filename" if (charge_c_rate_override is not None
                                          or discharge_c_rate_override is not None) else "data",
        "charge_c_rate_data": round(charge_c_data, 3),
        "discharge_c_rate_data": round(discharge_c_data, 3),
        "c_rate_mismatch": bool(
            (charge_c_rate_override is not None and _mismatch(charge_c_rate_override, charge_c_data))
            or (discharge_c_rate_override is not None and _mismatch(discharge_c_rate_override, discharge_c_data))
        ),
    })

    efcs = df["EFC"].dropna().drop_duplicates()

    weight = 1
    if efc_early >= efcs.min():
        efc_idx =(efcs - efc_early).abs().idxmin()
        closest = efcs.loc[efc_idx]
        cycle_idx = int(df.loc[efc_idx, "cycle_number"])

        curve = extract_charging_curve(df[df["cycle_number"] == cycle_idx])
        efc_early_df = pd.DataFrame({"early_curve": curve["voltage_V"], "time_s": curve["time_s"]})
        
    else:
        efc_early_df = None
        print(f"{cell_id} - Missing efc {efc_early}")

    if efc_late <= efcs.max():
        efc_idx = (efcs - efc_late).abs().idxmin()
        closest = efcs.loc[efc_idx]

        if abs(closest - efc_late) >= 0.5:
            lower = efcs[efcs < efc_late].max()
            upper = efcs[efcs > efc_late].min()


            lower_idx = int(df.loc[df["EFC"] == lower, "cycle_number"].loc[0])
            upper_idx = int(df.loc[df["EFC"] == upper, "cycle_number"].loc[0])

            weight = (efc_late - lower) / (upper - lower)
        
            lower_curve = extract_charging_curve(df[df["cycle_number"] == lower_idx])
            upper_curve = extract_charging_curve(df[df["cycle_number"] == upper_idx])
            
            efc_late_df = pd.DataFrame({"lower_curve_time": lower_curve["time_s"], "upper_curve_time": upper_curve["time_s"],
                                        "lower_curve": lower_curve["voltage_V"], "upper_curve": upper_curve["voltage_V"], 
                                        "weight":weight})
        else:
            cycle_idx = int(df.loc[efc_idx, "cycle_number"])
            curve = extract_charging_curve(df[df["cycle_number"] == cycle_idx])
            efc_late_df = pd.DataFrame({"lower_curve_time": curve["time_s"], "upper_curve_time": curve["time_s"], 
                                        "lower_curve": curve["voltage_V"],  "upper_curve": curve["voltage_V"], 
                                        "weight":weight})
            

    else:
        efc_late_df = None
        print(f"{cell_id} - Missing efc {efc_late} (max efc = {efcs.max()})")

   

    return metadata, efc_early_df, efc_late_df

def extract_charging_curve(efc_df: pd.DataFrame):
    """
    Isolates just the charging portion of one efc, for the 'interpreter'.
    Returns None if efc_number isn't present in df.
    """
    if len(efc_df) == 0:
        return None
    tbl, d = charge_segments(efc_df, i_thresh_A=0.1)
    first_seg_id = tbl.index[0]
    first_charge_curve = d[d["_seg"] == first_seg_id]
    return first_charge_curve


def charge_segments(df, i_thresh_A=0.05, v_thresh = -0.01, min_rows=20):
    """Table of charge segments in FILE order (no sorting): EFC at segment start, plus the rows."""
    v = df["voltage_V"].to_numpy(float)
    vol = np.diff(np.r_[v, v[-1]]) > v_thresh # Remove big voltage drops
    chg = (df["current_A"].to_numpy(float) > i_thresh_A)
    
    # Both conditions must hold
    mask = chg & vol

    # Segment whenever the combined condition changes
    seg = np.cumsum(np.r_[True, mask[1:] != mask[:-1]])
    d = df.loc[mask].copy()
    d = df.loc[mask].assign(_seg=seg[mask])
    tbl = d.groupby("_seg").agg(efc=("EFC", "first"), n=("EFC", "size"))
    tbl = tbl[tbl["n"] >= min_rows]
    return tbl, d


