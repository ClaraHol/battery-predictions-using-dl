"""
Takes one cell's standardized per-row cycling data (from any parser in
parsers.py) and produces:
  1. a metadata dict ready for config_and_cleaning.build_cell_record()
  2. the raw efc0 and efc50 cycle snapshots, for later feature engineering
     (kept separate from cleaning/metadata so re-running feature extraction
     later doesn't require re-parsing every raw file)
"""

import pandas as pd
from cmu_processing import  cmu_cycle_life
from config_and_cleaning import build_cell_record, compute_cycle_life_robust, compute_cc_c_rate, calculate_efc, find_capacity_checks


def process_cell(df: pd.DataFrame, cell_id: str, source_dataset: str,
                  efc_early: int = 1, efc_late: int = 50,
                  charge_c_rate_override: float = None,
                  discharge_c_rate_override: float = None):

    """
    df: standard-schema DataFrame for one cell (see parsers.py STANDARD_COLUMNS)
    Returns (metadata_dict, efc_early_df, efc_late_df)

    efc_early / efc_late: the exact cycle_number values to extract voltage
    curves for. NOT assumed to be the dataset's first cycle - some sources
    are 0-indexed (first real cycle = 0) and some may start at 1, so pass
    whatever your data actually uses. Either curve is None if that cycle
    number isn't present for this cell (e.g. it died before reaching 50 -
    the metadata row is still kept, just without that snapshot).

    Curves are NOT forced to equal length - each is just every row in df
    matching that cycle_number, whatever length that turns out to be.
    """
    from config_and_cleaning import NOMINAL_CAPACITY_AH

    nominal_cap = NOMINAL_CAPACITY_AH[source_dataset]

    # --- cycle life: first cycle where per-cycle discharge capacity drops
    # to 90% of the initial cycle's discharge capacity ---
    per_cycle_discharge_cap = df.groupby("cycle_number")["discharge_capacity_Ah"].max()
    if source_dataset == "cmu":
        life = cmu_cycle_life(df)
    elif source_dataset == "elt":
        checks, cell_processed = find_capacity_checks(df, nominal_capacity_Ah=nominal_cap)
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
        life = {"cycle_life": first_90_cycle, "censored": False, "n_full_cycles": len(checks["SOH_pct"])}
    else:
      
        life = compute_cycle_life_robust(df, per_cycle_discharge_cap, nominal_cap)
    

    # --- charge/discharge C-rate: mean over the whole cell's history.
    # (The protocol is fixed per cell across its life for these datasets,
    # so this is equivalent to computing it from efc0 alone, but averaging
    # over everything is more robust to a noisy first cycle.) ---
    # --- C-rates: constant-current estimate from the data (time-weighted
    # mode per cycle), overridden by filename values when supplied ---
    charge_c_data = compute_cc_c_rate(df, nominal_cap, "charge")
    discharge_c_data = compute_cc_c_rate(df, nominal_cap, "discharge")
    charge_c = charge_c_rate_override if charge_c_rate_override is not None else charge_c_data
    discharge_c = discharge_c_rate_override if discharge_c_rate_override is not None else discharge_c_data
 
    def _mismatch(a, b):
        return b > 0 and abs(a - b) / b > 0.30

    df = calculate_efc(df, nominal_cap)

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
    #print(efcs)

    if efc_early >= efcs.min():
        closest = efcs.loc[(efcs).abs().idxmin()]
        efc_early_df = df[df["EFC"] == closest].copy()
        efc_early_df["cell_id"] = cell_id
    else:
        efc_early_df = None

    if efc_late <= efcs.max():
        closest = efcs.loc[(efcs - efc_late).abs().idxmin()]

        if abs(closest - efc_late) >= 0.5:
            lower = efcs[efcs < efc_late].max()
            upper = efcs[efcs > efc_late].min()
            print(f"EFC {lower=}, EFC {upper=}")

            weight = (efc_late - lower) / (upper - lower)
            
            lower_curve = extract_charging_curve(df[df["EFC"] == lower])
            upper_curve = extract_charging_curve(df[df["EFC"] == upper])
            efc_late_df = pd.DataFrame({"lower_curve": lower_curve["voltage_V"], "upper_curve": upper_curve["voltage_V"], "weight":weight})
        else:
            print(f"EFC {closest=}")
            curve = extract_charging_curve(df[df["EFC"] == closest])
            weight = 1
            efc_late_df = pd.DataFrame({"lower_curve": curve["voltage_V"], "upper_curve": curve["voltage_V"], "weight":weight})
        

    else:
        efc_late_df = None

    return metadata, efc_early_df, efc_late_df

def extract_charging_curve(efc_df: pd.DataFrame, has_current: bool = True):
    """
    Isolates just the charging portion of one efc, for the 'interpreter'
    (which per clarification only needs the charging voltage curve, not
    the full charge+discharge+rest curve).

    has_current: set False for sources with no current_A column at all
    (currently just SNL's voltage_timeseries_all_cells.csv export) - in
    that case this returns the WHOLE efc unsegmented (charge+discharge
    mixed together), since there's no way to isolate charging without a
    current sign to filter on. Check has_current downstream before
    treating the result as "charging only".

    Returns None if efc_number isn't present in df.
    """
    if len(efc_df) == 0:
        return None
    if not has_current:
        return efc_df.copy()
    return efc_df[efc_df["current_A"] > 0].copy()


# def extract_charging_curve(df: pd.DataFrame, efc_number: int, has_current: bool = True):
#     """
#     Isolates just the charging portion of one efc, for the 'interpreter'
#     (which per clarification only needs the charging voltage curve, not
#     the full charge+discharge+rest curve).

#     has_current: set False for sources with no current_A column at all
#     (currently just SNL's voltage_timeseries_all_cells.csv export) - in
#     that case this returns the WHOLE efc unsegmented (charge+discharge
#     mixed together), since there's no way to isolate charging without a
#     current sign to filter on. Check has_current downstream before
#     treating the result as "charging only".

#     Returns None if efc_number isn't present in df.
#     """
#     efc_df = df[df["EFC"] == efc_number]
#     if len(efc_df) == 0:
#         return None
#     if not has_current:
#         return efc_df.copy()
#     return efc_df[efc_df["current_A"] > 0].copy()