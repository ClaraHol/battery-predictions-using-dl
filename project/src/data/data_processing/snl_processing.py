
import numpy as np
import pandas as pd
from config_and_cleaning import calculate_efc, find_capacity_checks, build_cell_record, NOMINAL_CAPACITY_AH

MAX_CYCLE = {20: 300, 60: 100, 100: 70}

def extract_efc_early(info, efcs, target, cell_id):

    cycle_numbers = np.asarray(info["cycle_number"])
    closest = efcs[np.argmin(np.abs(efcs - target))]
    idx = cycle_numbers[np.argmin(np.abs(efcs - target))]
   
    if target > efcs.min():
        early_curve = info["voltage_curves"][idx]["voltage_V"].to_numpy()
        curve = pd.DataFrame({"time_s":  info["voltage_curves"][idx]["time_s"].to_numpy(),"early_curve": early_curve})
        return curve
    else:
        return None

def extract_efc_late(info, efcs, target, cell_id):

    cycle_numbers = np.asarray(info["cycle_number"])
    closest = efcs[np.argmin(np.abs(efcs - target))]
    idx = cycle_numbers[np.argmin(np.abs(efcs - target))]
   
    if target < efcs.max():
        if abs(closest - target) < 0.5:
            weight = 1
            lower_curve = info["voltage_curves"][idx]["voltage_V"].to_numpy()
            upper_curve = info["voltage_curves"][idx]["voltage_V"].to_numpy()

        else:
            lower_indices = np.where(efcs < target)[0]
            upper_indices = np.where(efcs > target)[0]

            if len(lower_indices) == 0 or len(upper_indices) == 0:
                print(f"Skipping {cell_id}: No lower/upper curve found")
                return None

                    # Positions in the original EFC array
            pos_lower = lower_indices[np.argmax(efcs[lower_indices])]
            pos_upper = upper_indices[np.argmin(efcs[upper_indices])]

            # Actual EFC values
            lower = efcs[pos_lower]
            upper = efcs[pos_upper]

            # Actual cycle numbers
            idx_lower = int(cycle_numbers[pos_lower])
            idx_upper = int(cycle_numbers[pos_upper])

            weight = (target - lower) / (upper - lower)

            lower_curve = info["voltage_curves"][idx_lower]["voltage_V"].to_numpy()
            upper_curve = info["voltage_curves"][idx_upper]["voltage_V"].to_numpy()

        # Validate curves
        if lower_curve.ndim != 1 or upper_curve.ndim != 1:
            print(f"Skipping {cell_id}: Invalid curve dimensions")
            return None

        if len(lower_curve) != len(upper_curve) or len(lower_curve) == 0:
            print(f"Skipping {cell_id}: Invalid curve lengths")
            return None

        curve = pd.DataFrame({
            "lower_curve_time": info["voltage_curves"][idx_lower]["time_s"].to_numpy(),
            "upper_curve_time": info["voltage_curves"][idx_upper]["time_s"].to_numpy(),
            "lower_curve": lower_curve,
            "upper_curve": upper_curve,
            "weight": weight
        })
    else:
        print(f"No EFC 50 - Max EFC is {efcs.max()}")
    return curve

def process_snl(info, snl_meta_data, cell_id, efc_early = 1, efc_late = 50, charge_c_rate_override = None, discharge_c_rate_override = None):

    # Compute depth of charge
    diff = info["dod_high_pct"] - info["dod_low_pct"]
    max_cycle = MAX_CYCLE[diff]
    
    # Compute SOH from capacity test cycles
    info["discharge_capacity_Ah"] = snl_meta_data[cell_id]["discharge_capacity_Ah"].iloc[:max_cycle]
    info["cycle_number"] = snl_meta_data[cell_id]["cycle_number"].iloc[:max_cycle]
    
    # Compute EFC
    nom_cap = NOMINAL_CAPACITY_AH[info["source_dataset"]]
    info = calculate_efc(info, q_nom = nom_cap, cell_id="snl")

    if info["dod_low_pct"]==0:
        checks, cell_processed = find_capacity_checks(snl_meta_data[cell_id], nominal_capacity_Ah=nom_cap, min_fraction_of_nominal=1.5, outlier_fraction_of_nominal=2.5)
    else:
        checks, cell_processed = find_capacity_checks(snl_meta_data[cell_id], nominal_capacity_Ah=nom_cap)

    Q_ref = checks["capacity_Ah"].iloc[0]
    checks["SOH_pct"] = checks["capacity_Ah"] / Q_ref * 100

    # Find first cycle below 90% SOH to compute cycle life
    ref = float(checks["capacity_Ah"].iloc[0])
    thr = 0.9 * ref
    below = np.flatnonzero(checks["capacity_Ah"].to_numpy() <= thr)
    if len(below) == 0:
        life, censored = float(checks["EFC"].iloc[-1]), True       # lower bound
    else:
        i, censored = int(below[0]), False
        if i == 0:
            life = float(checks["EFC"].iloc[0])
        else:
            c0, c1 = checks.iloc[i - 1], checks.iloc[i]
            life = float(c0.EFC + (c0.capacity_Ah - thr) / (c0.capacity_Ah - c1.capacity_Ah) * (c1.EFC - c0.EFC))
    
    full_cell_id = f"{info['source_dataset']}_{cell_id}"
    metadata = build_cell_record(
        cell_id=full_cell_id,
        source_dataset=info["source_dataset"],
        temp_C_raw=info["temp_C"],
        charge_c_rate_raw=info["charge_c_rate"],
        discharge_c_rate_raw=info["discharge_c_rate"],
        cycle_life = life,
        initial_discharge_capacity_Ah=snl_meta_data[cell_id]["ah_c"].iloc[0],
        )

    metadata.update({
        "cycle_life_censored": censored,
        "n_full_cycles": len(checks),
        "c_rate_source": "filename" if (charge_c_rate_override is not None
                                          or discharge_c_rate_override is not None) else "data",
    })
    efcs = np.asarray(info["EFC"])

    efc_early_df = extract_efc_early(info, efcs, efc_early, cell_id)
    efc_late_df = extract_efc_late(info, efcs, efc_late, cell_id)

  
    return metadata, efc_early_df, efc_late_df