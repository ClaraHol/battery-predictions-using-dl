
import numpy as np
import pandas as pd
from config_and_cleaning import calculate_efc, find_capacity_checks, build_cell_record, NOMINAL_CAPACITY_AH

MAX_CYCLE = {20: 300, 60: 100, 100: 70}

def extract_efc(info, efcs, target, cell_id):

    cycle_numbers = np.asarray(info["cycle_number"])
    closest = efcs[np.argmin(np.abs(efcs - target))]
    idx = info["cycle_number"][np.argmin(np.abs(efcs - target))]
    print(idx)
    if target < efcs.max():
        if abs(closest - target) < 0.5:
            print(f"{closest=}")
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
            "lower_curve": lower_curve,
            "upper_curve": upper_curve,
            "weight": weight
        })
    else:
        print(f"No EFC 50 - Max EFC is {efcs.max()}")
    return curve

def process_snl(info, snl_meta_data, cell_id, efc_early = 1, efc_late = 50):
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
    first_90_cycle = next(
        (
            int(checks["EFC"].iloc[cycle])
            for cycle, soh in checks["SOH_pct"].items()
            if soh <= 90
        ),
        None,
    )

    full_cell_id = f"{info['source_dataset']}_{cell_id}"
    metadata = build_cell_record(
        cell_id=full_cell_id,
        source_dataset=info["source_dataset"],
        temp_C_raw=info["temp_C"],
        charge_c_rate_raw=info["charge_c_rate"],
        discharge_c_rate_raw=info["discharge_c_rate"],
        cycle_life = first_90_cycle,
        initial_discharge_capacity_Ah=snl_meta_data[cell_id]["ah_c"].iloc[0],
        )
    efcs = np.asarray(info["EFC"])

    efc_early_df = extract_efc(info, efcs, efc_early, cell_id)
    efc_late_df = extract_efc(info, efcs, efc_late, cell_id)

    return metadata, efc_early_df, efc_late_df