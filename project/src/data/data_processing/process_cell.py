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
from cmu_processing import  cmu_cycle_life_efc, add_cmu_rpt_boundaries
from lg_mj1_processing import lgmj1_cycle_life_from_df, discharge_runs
from config_and_cleaning import build_cell_record, compute_cycle_life_efc, compute_cc_c_rate, calculate_efc


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
    df = df.sort_values("EFC", kind="stable").reset_index(drop=True)
    
    if source_dataset == "cmu":
        life = cmu_cycle_life_efc(df)
        life["caps"] = add_cmu_rpt_boundaries(df, life["caps"])
        caps = life["caps"]
        sub = select_caps_for_targets(caps, efc_early, efc_late)
        efc_early_df, efc_late_df = extract_rpt_discharge_curves(df, sub, efc_early = efc_early, efc_late = efc_late)
    
    elif source_dataset == "elt":
        life = lgmj1_cycle_life_from_df(df, nominal_Ah=nominal_cap, pick="max")
        caps = life["caps"]
        sub = select_caps_for_targets(caps, efc_early, efc_late)
        efc_early_df, efc_late_df = extract_rpt_discharge_curves(df, sub, efc_early = efc_early, efc_late = efc_late)

    else:
        efc_by_cycle = df.groupby("cycle_number")["EFC"].max()
        life = compute_cycle_life_efc(df, per_cycle_discharge_cap, nominal_cap, efc_by_cycle=efc_by_cycle)
        caps = life["caps"]
        sub = select_caps_for_targets(caps, efc_early, efc_late)
        efc_early_df, efc_late_df = extract_rpt_curves(df, sub, efc_early = efc_early, efc_late = efc_late)
    

    # charge/discharge C-rate: mean over the whole cell's history. 
    # C-rates: constant-current estimate from the data (time-weighted mode per cycle), overridden by filename values when supplied 
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

    return metadata, efc_early_df, efc_late_df

def remove_cv_phase(df, voltage_tol=0.005, current_drop_frac=0.05, window=15, min_rows=20):
    """
    Remove the CV phase and retain only the CC phase.

    Detects the CC-CV transition using:
      1. Voltage approaching its maximum.
      2. Current beginning to decrease from its CC level.

    Assumes positive current corresponds to charging.
    """

    if df is None or len(df) < min_rows:
        return df.copy() if df is not None else None

    df = df.sort_values("time_s", kind="stable").reset_index(drop=True).copy()

    v = df["voltage_V"].to_numpy(dtype=float)
    I = df["current_A"].to_numpy(dtype=float)

    # Smooth voltage and current
    v_smooth = pd.Series(v).rolling(5, center=True, min_periods=1).median().to_numpy()
    i_smooth = pd.Series(I).rolling(5, center=True, min_periods=1).median().to_numpy()

    # Upper charging voltage
    v_max = np.quantile(v_smooth, 0.99)

    # Estimate CC current from early charging samples
    n_early = max(5, int(0.2 * len(df)))
    i_cc = np.median(i_smooth[:n_early])

    # Near the upper voltage plateau
    near_vmax = (v_smooth >= v_max - voltage_tol)

    # Detect current taper
    current_taper = (i_smooth < i_cc * (1 - current_drop_frac))

    # Both conditions indicate CV onset
    cv_candidate = near_vmax & current_taper
    sustained = pd.Series(cv_candidate.astype(int)).rolling(window, min_periods=window).sum().eq(window).to_numpy()
    indices = np.flatnonzero(sustained)

    if len(indices) == 0:
        # No CV phase detected: keep the full curve
        return df

    cutoff = indices[0] - window + 1

    if cutoff < min_rows:
        return df

    return df.iloc[:cutoff].copy().reset_index(drop=True)

def select_caps_for_targets(caps, efc_early, efc_late, k=3):
    caps = caps.sort_values("efc").reset_index(drop=True)
    e = caps["efc"].to_numpy()
    keep = set()
    for target in (efc_early, efc_late):
        p = int(np.argmin(np.abs(e - target)))
        keep.update(range(max(0, p - k), min(len(e), p + k + 1)))
    return caps.iloc[sorted(keep)].reset_index(drop=True)




def extract_charging_curve(rpt_df):
    if rpt_df.empty:
        return None

    tbl, d = charge_segments( rpt_df, i_thresh_A=0.1,min_rows=20)

    if tbl.empty:
        return None

    first_seg_id = tbl.index[0]
    curve = (d[d["_seg"] == first_seg_id].copy().reset_index(drop=True))

    # Remove CV, retain CC
    curve = remove_cv_phase(curve)

    if curve is None or curve.empty:
        return None

    curve["time_s"] -= curve["time_s"].iloc[0]

    return curve

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

def get_curve(df, check):
    start = int(check["charge_start_idx"])
    end = int(check["charge_end_idx"])

    if start < 0 or end < start or end >= len(df):
        return None

    # Retrieve the RPT charging measurements
    rpt = df.iloc[start:end + 1].copy()

    # Extract CC charging curve (remove CV)
    curve = extract_charging_curve(rpt)

    if curve is None or curve.empty:
        return None

    return (curve[["time_s", "voltage_V"]] .reset_index(drop=True))

def extract_rpt_curves(df, caps, efc_early=1, efc_late=50, tol_early = 1):
    """
    Extract complete RPT charging curves using caps.
    caps must contain: efc, charge_start_idx, charge_end_idx
    Charging boundaries are inclusive positional indices.
    """

    if caps.empty:
        return None, None

    caps = caps.sort_values("efc").reset_index(drop=True)

    # Keep only checks with valid charging curves
    available = []

    for _, check in caps.iterrows():
        curve = get_curve(df, check)

        if curve is not None:
            available.append({"efc": float(check["efc"]),"curve": curve})

    if not available:
        return None, None

    efcs = np.array([c["efc"] for c in available])

    # EARLY
    early = None
    pos = int(np.argmin(np.abs(efcs - efc_early)))

    if abs(efcs[pos] - efc_early) <= tol_early:      # e.g. 1.0
        curve = available[pos]["curve"]
        early = pd.DataFrame({"time_s": curve["time_s"].to_numpy(),"early_curve": curve["voltage_V"].to_numpy()})

    else:
        print(f"[{df["cell_id"]}] - Missing EFC {efc_early}")
    # LATE
    late = None
    pos = int(np.argmin(np.abs(efcs - efc_late)))

    if abs(efcs[pos] - efc_late) < 0.5:
        curve = available[pos]["curve"]

        t = curve["time_s"].to_numpy()
        v = curve["voltage_V"].to_numpy()

        late = pd.DataFrame({
            "lower_curve_time": t,
            "upper_curve_time": t.copy(),
            "lower_curve": v,
            "upper_curve": v.copy(),
            "weight": 1.0
        })

    elif efcs.min() < efc_late < efcs.max():

        lo = np.where(efcs < efc_late)[0][-1]
        up = np.where(efcs > efc_late)[0][0]

        weight = ((efc_late - efcs[lo]) / (efcs[up] - efcs[lo]))

        lc = available[lo]["curve"]
        uc = available[up]["curve"]

        late = pd.DataFrame({"lower_curve_time": pd.Series(lc["time_s"].to_numpy()),
            "upper_curve_time": pd.Series(uc["time_s"].to_numpy()),
            "lower_curve": pd.Series(lc["voltage_V"].to_numpy()),
            "upper_curve": pd.Series(uc["voltage_V"].to_numpy()),
            "weight": weight
        })

    return early, late


def extract_rpt_discharge_curves(
    df: pd.DataFrame,
    caps: pd.DataFrame,
    efc_early: float = 1,
    efc_late: float = 50,
    discharge_sign: int = -1,
    i_thresh_A: float = 0.05,
):
    """
    Extract early and late RPT discharge curves for ELT.

    caps must contain:
        efc
        capacity_Ah
        discharge_start_idx
        discharge_end_idx

    Boundaries are inclusive positional indices into df.

    Returns:
        efc_early_df
        efc_late_df

    Curves are kept at original resolution.
    """

    if caps.empty:
        return None, None

    required = ["efc", "discharge_start_idx", "discharge_end_idx"]
    missing = [c for c in required if c not in caps.columns]

    if missing:
        raise ValueError(f"Missing ELT caps columns: {missing}")

    caps = caps.sort_values("efc").reset_index(drop=True)

    def get_discharge_curve(check):
        start = int(check["discharge_start_idx"])
        end = int(check["discharge_end_idx"])

        if start < 0 or end < start or end >= len(df):
            return None

        # Retrieve original discharge run
        curve = df.iloc[start:end + 1].copy()

        if curve.empty:
            return None

        # Shift time axis
        curve["time_s"] = (curve["time_s"] - curve["time_s"].iloc[0])

        # Preserve only discharging measurements
        discharge_mask = (curve["current_A"] * discharge_sign > i_thresh_A)

        curve = curve.loc[discharge_mask].copy()

        if len(curve) < 2:
            return None

        # Ensure original chronological order
        #curve = curve.sort_values("time_s", kind="stable").reset_index(drop=True)

        # Remove invalid voltage samples
        curve = curve[curve["voltage_V"] > 1.0].copy()

        if len(curve) < 2:
            return None

        # Reset time to start at zero
        curve["time_s"] -= curve["time_s"].iloc[0]

        return curve[["time_s", "voltage_V"]].reset_index(drop=True)

    # Load all available RPT discharge curves
    available = []

    for _, check in caps.iterrows():

        curve = get_discharge_curve(check)

        if curve is None:
            continue

        available.append({"efc": float(check["efc"]), "curve": curve})

    if not available:
        return None, None

    efcs = np.array([c["efc"] for c in available])

    # ---------------- EARLY ----------------

    early = None

    pos = int(np.argmin(np.abs(efcs - efc_early)))

    if efc_early >= efcs.min():
        curve = available[pos]["curve"]
        early = pd.DataFrame({"time_s": curve["time_s"].to_numpy(), "early_curve": curve["voltage_V"].to_numpy()})

    # ---------------- LATE ----------------

    late = None

    pos = int(np.argmin(np.abs(efcs - efc_late)))
    closest = efcs[pos]

    # Use a single RPT when close enough
    if abs(closest - efc_late) < 0.5:

        curve = available[pos]["curve"]

        t = curve["time_s"].to_numpy()
        v = curve["voltage_V"].to_numpy()

        late = pd.DataFrame({
            "lower_curve_time": t,
            "upper_curve_time": t.copy(),
            "lower_curve": v,
            "upper_curve": v.copy(),
            "weight": 1.0
        })

    # Otherwise use surrounding RPTs
    elif efcs.min() < efc_late < efcs.max():

        lower_pos = np.where(efcs < efc_late)[0][-1]
        upper_pos = np.where(efcs > efc_late)[0][0]

        lower_efc = efcs[lower_pos]
        upper_efc = efcs[upper_pos]

        weight = ((efc_late - lower_efc) / (upper_efc - lower_efc))

        lower_curve = available[lower_pos]["curve"]
        upper_curve = available[upper_pos]["curve"]

        # Independent Series allow different curve lengths
        late = pd.DataFrame({
            "lower_curve_time": pd.Series(lower_curve["time_s"].to_numpy()),
            "upper_curve_time": pd.Series(upper_curve["time_s"].to_numpy()),
            "lower_curve": pd.Series(lower_curve["voltage_V"].to_numpy()),
            "upper_curve": pd.Series(upper_curve["voltage_V"].to_numpy()),
            "weight": weight
        })

    return early, late




