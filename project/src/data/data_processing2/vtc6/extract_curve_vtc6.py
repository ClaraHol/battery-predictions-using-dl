import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def extract_curve_vtc6(file_dir):
    df = pd.read_csv(file_dir)


    q_charge = df["QCharge_mA_h"].to_numpy()
    q_discharge = df["QDischarge_mA_h"].to_numpy()

    charge_increase = np.clip(np.diff(q_charge, prepend=q_charge[0]), 0, None)
    discharge_increase = np.clip(np.diff(q_discharge, prepend=q_discharge[0]), 0, None)

    efc = np.cumsum(charge_increase + discharge_increase) / (2 * 1000 * 3.0)

    current = df["I_mA"].to_numpy()
    ns = df["Ns"].to_numpy()
    cycles = df["cycleNumber"].to_numpy()

    # Boundaries where the raw cycle number changes
    boundaries = np.r_[
        0,
        np.where(np.diff(cycles) != 0)[0] + 1,
        len(df),
    ]

    rpts = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Characteristic VTC6 reference discharge
        mask = (
            np.isin(ns[start:end], [3, 5, 7])
            & (np.abs(current[start:end] + 610) < 40)
        )

        indices = np.flatnonzero(mask) + start

        if len(indices) < 2000:
            continue

        if not {3, 5, 7}.issubset(set(ns[indices])):
            continue

        rpts.append({
            "start_idx": start,
            "efc": efc[start],
            "discharge_indices": indices,
        })


    if not rpts:
        raise ValueError("No RPTs detected")

    rpt_efcs = np.array([rpt["efc"] for rpt in rpts])

    if np.any(np.diff(rpt_efcs) <= 0):
        raise ValueError("RPT EFC values are not strictly increasing")

    upper = np.searchsorted(rpt_efcs, 50)

    if upper == 0 or upper == len(rpts):
        raise ValueError("EFC50 is not bracketed")

    lower = upper - 1

    rpt_efc0 = rpts[0]
    rpt_lower = rpts[lower]
    rpt_upper = rpts[upper]

    efc_lower = rpt_lower["efc"]
    efc_upper = rpt_upper["efc"]

    weight_upper = (50 - efc_lower) / (efc_upper - efc_lower)
    weight_lower = 1 - weight_upper

    def get_curve(rpt, curve_type):
        indices = rpt["discharge_indices"]

        curve = df.iloc[indices][
            ["time_s", "Ecell_V", "I_mA"]
        ].copy()

        curve.columns = ["time_s", "voltage_V", "current_A"]

        curve["time_s"] -= curve["time_s"].iloc[0]
        curve["current_A"] = curve["current_A"].abs() / 1000

        curve["curve_type"] = curve_type
        curve["rpt_efc"] = rpt["efc"]

        return curve

    curves = pd.concat([
        get_curve(rpt_efc0, "efc0"),
        get_curve(rpt_lower, "efc50_lower"),
        get_curve(rpt_upper, "efc50_upper"),
    ], ignore_index=True)

    metadata = {
        "efc0": rpt_efc0["efc"],
        "efc50_lower": efc_lower,
        "efc50_upper": efc_upper,
        "weight_lower": weight_lower,
        "weight_upper": weight_upper,
        "direction": "discharge",
        "n_rpts": len(rpts),
    }

    return curves, metadata