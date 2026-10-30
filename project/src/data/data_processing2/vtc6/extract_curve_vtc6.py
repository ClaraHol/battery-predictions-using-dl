from pathlib import Path

import numpy as np
import pandas as pd


RATED_CAPACITY_AH = 3.0
RPT_CURRENT_MA = -610.0
RPT_CURRENT_TOLERANCE_MA = 40.0
RPT_MIN_POINTS = 2000
LIFETIME_SOH_THRESHOLD = 0.90
CONDITION_EFC_MAX = 50.0
C_RATE_ROUND_STEP = 0.5
TEMPERATURE_ROUND_STEP_C = 5.0
ACTIVE_CURRENT_MA = 50.0

# VTC6 ageing-protocol conditions from the source dataset.
# Unlisted cells use the baseline protocol: 25 C ambient, 1C CC charge,
# and 4.2 V end-of-charge voltage.
VTC6_PROTOCOL_OVERRIDES = {
    "VAH06": {"charge_cc_c_rate_protocol": 0.5},
    "VAH07": {"charge_voltage_V": 4.0},
    "VAH09": {"ambient_temperature_C": 20.0},
    "VAH10": {"ambient_temperature_C": 30.0},
    "VAH16": {"charge_cc_c_rate_protocol": 1.5},
    "VAH20": {"charge_cc_c_rate_protocol": 1.5},
    "VAH23": {"charge_voltage_V": 4.1},
    "VAH24": {"charge_cc_c_rate_protocol": 0.5},
    "VAH25": {"ambient_temperature_C": 20.0},
    "VAH30": {"ambient_temperature_C": 35.0},
}


def _round_to_step(value, step):
    if not np.isfinite(value):
        return np.nan
    # All values here are non-negative. Avoid Python's bankers-rounding at ties.
    return float(step * np.floor(value / step + 0.5))


def _protocol_for_cell(file_dir):
    cell_name = Path(file_dir).stem.upper()
    protocol = {
        "ambient_temperature_C": 25.0,
        "charge_cc_c_rate_protocol": 1.0,
        "charge_voltage_V": 4.2,
    }
    protocol.update(VTC6_PROTOCOL_OVERRIDES.get(cell_name, {}))
    return cell_name, protocol


def _positive_counter_increments(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    delta = np.diff(values)
    delta = delta[np.isfinite(delta)]
    return float(np.clip(delta, 0.0, None).sum())


def _integrate_capacity_ah(df, indices):
    """Integrate discharge current over contiguous pieces of an RPT curve."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return np.nan

    # Ns=3,5,7 are separated by interruptions (Ns=4,6), so do not integrate
    # across the time gaps between those pieces.
    split_at = np.where(np.diff(indices) > 1)[0] + 1
    runs = np.split(indices, split_at)

    capacity_ah = 0.0
    for run in runs:
        if len(run) < 2:
            continue
        t = df.loc[run, "time_s"].to_numpy(dtype=float)
        current_a = -df.loc[run, "I_mA"].to_numpy(dtype=float) / 1000.0

        valid = np.isfinite(t) & np.isfinite(current_a) & (current_a > 0)
        t = t[valid]
        current_a = current_a[valid]
        if len(t) < 2:
            continue

        order = np.argsort(t)
        t = t[order]
        current_a = current_a[order]
        dt = np.diff(t)
        good = dt > 0
        if not np.any(good):
            continue

        # Trapezoidal integration, applied only between consecutive valid samples.
        capacity_ah += np.sum(
            0.5 * (current_a[:-1][good] + current_a[1:][good]) * dt[good]
        ) / 3600.0

    return float(capacity_ah)


def _rpt_counter_capacity_ah(df, indices):
    """Independent capacity estimate from QDischarge_mA_h for diagnostics."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return np.nan

    # Include the short interruptions between Ns=3,5,7. The discharge counter
    # should be flat during interruptions. Counter resets are ignored by summing
    # only positive increments.
    start = int(indices[0])
    end = int(indices[-1]) + 1
    q = df.iloc[start:end]["QDischarge_mA_h"].to_numpy(dtype=float)
    return _positive_counter_increments(q) / 1000.0


def _cycle_life_from_rpts(rpt_efc, soh, threshold=LIFETIME_SOH_THRESHOLD):
    """Return first interpolated SOH-threshold crossing or a censored lower bound."""
    rpt_efc = np.asarray(rpt_efc, dtype=float)
    soh = np.asarray(soh, dtype=float)

    valid = np.isfinite(rpt_efc) & np.isfinite(soh)
    rpt_efc = rpt_efc[valid]
    soh = soh[valid]
    if len(rpt_efc) < 2:
        raise ValueError("Need at least two valid RPT capacities for lifetime")
    if soh[0] <= threshold:
        raise ValueError("Initial RPT is already at or below lifetime SOH threshold")

    for j in range(1, len(soh)):
        if soh[j] <= threshold and soh[j - 1] > threshold:
            x0, x1 = rpt_efc[j - 1], rpt_efc[j]
            y0, y1 = soh[j - 1], soh[j]
            if y1 == y0:
                crossing = x1
            else:
                crossing = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            return float(crossing), False

    # The threshold was not observed. The last RPT EFC is a lower bound only.
    return float(rpt_efc[-1]), True


def _condition_features(df, blocks, rpt_start_indices, efc, initial_capacity_ah):
    """
    Estimate paper-style charge/discharge C-rate features from ageing cycles.

    RPT blocks are excluded. Only cycles beginning within the first 50 EFC are
    used so that the operating-condition feature is not dominated by late-life
    voltage/resistance changes. Within each ageing cycle, SOC is reconstructed
    from the charge/discharge counters relative to the initial RPT capacity.
    C-rate is averaged in the four SOC ranges used by the paper, and those bin
    means are then averaged.
    """
    soc_edges = np.array([0.0, 0.2, 0.5, 0.8, 1.0000001])
    charge_by_bin = [[] for _ in range(4)]
    discharge_by_bin = [[] for _ in range(4)]
    condition_indices = []
    n_blocks = 0

    for start, end in blocks:
        if start in rpt_start_indices:
            continue
        if not np.isfinite(efc[start]) or efc[start] > CONDITION_EFC_MAX:
            continue
        if end - start < 2:
            continue

        block = df.iloc[start:end]
        current_ma = block["I_mA"].to_numpy(dtype=float)
        q_charge = block["QCharge_mA_h"].to_numpy(dtype=float)
        q_discharge = block["QDischarge_mA_h"].to_numpy(dtype=float)

        dq_charge_ah = np.r_[0.0, np.clip(np.diff(q_charge), 0.0, None)] / 1000.0
        dq_discharge_ah = np.r_[0.0, np.clip(np.diff(q_discharge), 0.0, None)] / 1000.0

        cum_discharge_ah = np.cumsum(dq_discharge_ah)
        total_discharge_ah = float(cum_discharge_ah[-1])
        cum_charge_ah = np.cumsum(dq_charge_ah)

        # Ageing mission starts essentially full after the previous CC-CV charge.
        discharge_soc = np.clip(
            1.0 - cum_discharge_ah / initial_capacity_ah, 0.0, 1.0
        )
        charge_start_soc = np.clip(
            1.0 - total_discharge_ah / initial_capacity_ah, 0.0, 1.0
        )
        charge_soc = np.clip(
            charge_start_soc + cum_charge_ah / initial_capacity_ah, 0.0, 1.0
        )

        charge_mask = np.isfinite(current_ma) & (current_ma > ACTIVE_CURRENT_MA)
        discharge_mask = np.isfinite(current_ma) & (current_ma < -ACTIVE_CURRENT_MA)

        if not (np.any(charge_mask) or np.any(discharge_mask)):
            continue

        c_rate = np.abs(current_ma) / (1000.0 * RATED_CAPACITY_AH)

        for b in range(4):
            lo, hi = soc_edges[b], soc_edges[b + 1]
            cmask = charge_mask & (charge_soc >= lo) & (charge_soc < hi)
            dmask = discharge_mask & (discharge_soc >= lo) & (discharge_soc < hi)
            if np.any(cmask):
                charge_by_bin[b].append(float(np.nanmean(c_rate[cmask])))
            if np.any(dmask):
                discharge_by_bin[b].append(float(np.nanmean(c_rate[dmask])))

        condition_indices.extend(range(start, end))
        n_blocks += 1

    def average_bin_means(values_by_bin):
        bin_means = [np.mean(v) for v in values_by_bin if len(v) > 0]
        return float(np.mean(bin_means)) if bin_means else np.nan, len(bin_means)

    charge_c_rate_raw, charge_bins_used = average_bin_means(charge_by_bin)
    discharge_c_rate_raw, discharge_bins_used = average_bin_means(discharge_by_bin)

    temperature_mean = np.nan
    temperature_p05 = np.nan
    temperature_max = np.nan
    if condition_indices:
        temp = df.iloc[np.asarray(condition_indices, dtype=int)]["Temperature__C"].to_numpy(dtype=float)
        temp = temp[np.isfinite(temp)]
        if len(temp):
            temperature_mean = float(np.mean(temp))
            temperature_p05 = float(np.percentile(temp, 5))
            temperature_max = float(np.max(temp))

    return {
        "charge_c_rate_raw": charge_c_rate_raw,
        "discharge_c_rate_raw": discharge_c_rate_raw,
        "charge_c_rate_bins_used": charge_bins_used,
        "discharge_c_rate_bins_used": discharge_bins_used,
        "n_condition_cycles": n_blocks,
        "surface_temperature_mean_C": temperature_mean,
        "surface_temperature_p05_C": temperature_p05,
        "surface_temperature_max_C": temperature_max,
    }


def extract_curve_vtc6(file_dir, return_rpt_summary=False):
    df = pd.read_csv(file_dir)

    required_columns = {
        "time_s",
        "Ecell_V",
        "I_mA",
        "QCharge_mA_h",
        "QDischarge_mA_h",
        "Temperature__C",
        "cycleNumber",
        "Ns",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required VTC6 columns: {sorted(missing)}")

    cell_name, protocol = _protocol_for_cell(file_dir)

    q_charge = df["QCharge_mA_h"].to_numpy(dtype=float)
    q_discharge = df["QDischarge_mA_h"].to_numpy(dtype=float)

    charge_increase = np.clip(np.diff(q_charge, prepend=q_charge[0]), 0, None)
    discharge_increase = np.clip(np.diff(q_discharge, prepend=q_discharge[0]), 0, None)
    efc = np.cumsum(charge_increase + discharge_increase) / (
        2 * 1000 * RATED_CAPACITY_AH
    )

    current = df["I_mA"].to_numpy(dtype=float)
    ns = df["Ns"].to_numpy()
    cycles = df["cycleNumber"].to_numpy()

    # cycleNumber is not globally reliable, but changes in the raw counter are
    # still useful as local contiguous block boundaries.
    boundaries = np.r_[
        0,
        np.where(np.diff(cycles) != 0)[0] + 1,
        len(df),
    ]
    blocks = list(zip(boundaries[:-1], boundaries[1:]))

    rpts = []
    for start, end in blocks:
        mask = (
            np.isin(ns[start:end], [3, 5, 7])
            & (np.abs(current[start:end] - RPT_CURRENT_MA) < RPT_CURRENT_TOLERANCE_MA)
        )
        indices = np.flatnonzero(mask) + start

        if len(indices) < RPT_MIN_POINTS:
            continue
        if not {3, 5, 7}.issubset(set(ns[indices])):
            continue

        capacity_ah = _integrate_capacity_ah(df, indices)
        counter_capacity_ah = _rpt_counter_capacity_ah(df, indices)
        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            raise ValueError(f"Invalid RPT capacity at row {start}")

        rpts.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "efc": float(efc[start]),
            "discharge_indices": indices,
            "capacity_Ah": capacity_ah,
            "counter_capacity_Ah": counter_capacity_ah,
        })

    if not rpts:
        raise ValueError("No RPTs detected")

    rpt_efcs = np.array([rpt["efc"] for rpt in rpts], dtype=float)
    if np.any(np.diff(rpt_efcs) <= 0):
        raise ValueError("RPT EFC values are not strictly increasing")

    capacities = np.array([rpt["capacity_Ah"] for rpt in rpts], dtype=float)
    initial_capacity_ah = capacities[0]
    if not np.isfinite(initial_capacity_ah) or initial_capacity_ah <= 0:
        raise ValueError("Invalid initial RPT capacity")
    soh = capacities / initial_capacity_ah

    cycle_life_efc, cycle_life_censored = _cycle_life_from_rpts(
        rpt_efcs, soh, LIFETIME_SOH_THRESHOLD
    )

    upper = np.searchsorted(rpt_efcs, 50.0)
    if upper == 0 or upper == len(rpts):
        raise ValueError("EFC50 is not bracketed")
    lower = upper - 1

    rpt_efc0 = rpts[0]
    rpt_lower = rpts[lower]
    rpt_upper = rpts[upper]

    efc_lower = rpt_lower["efc"]
    efc_upper = rpt_upper["efc"]
    weight_upper = (50.0 - efc_lower) / (efc_upper - efc_lower)
    weight_lower = 1.0 - weight_upper

    def get_curve(rpt, curve_type):
        indices = rpt["discharge_indices"]
        curve = df.iloc[indices][["time_s", "Ecell_V", "I_mA"]].copy()
        curve.columns = ["time_s", "voltage_V", "current_A"]
        curve["time_s"] -= curve["time_s"].iloc[0]
        curve["current_A"] = curve["current_A"].abs() / 1000.0
        curve["curve_type"] = curve_type
        curve["rpt_efc"] = rpt["efc"]
        return curve

    curves = pd.concat([
        get_curve(rpt_efc0, "efc0"),
        get_curve(rpt_lower, "efc50_lower"),
        get_curve(rpt_upper, "efc50_upper"),
    ], ignore_index=True)

    condition = _condition_features(
        df=df,
        blocks=blocks,
        rpt_start_indices={rpt["start_idx"] for rpt in rpts},
        efc=efc,
        initial_capacity_ah=initial_capacity_ah,
    )

    # Paper-compatible clustering features. C-rates are rounded to 0.5 C and
    # ambient temperature to 5 C. Keep unrounded values alongside them.
    charge_c_rate = _round_to_step(condition["charge_c_rate_raw"], C_RATE_ROUND_STEP)
    if not np.isfinite(charge_c_rate):
        charge_c_rate = protocol["charge_cc_c_rate_protocol"]

    discharge_c_rate = _round_to_step(
        condition["discharge_c_rate_raw"], C_RATE_ROUND_STEP
    )

    ambient_temperature_raw = float(protocol["ambient_temperature_C"])
    temperature_c = _round_to_step(
        ambient_temperature_raw, TEMPERATURE_ROUND_STEP_C
    )

    metadata = {
        "efc0": rpt_efc0["efc"],
        "efc50_lower": efc_lower,
        "efc50_upper": efc_upper,
        "weight_lower": weight_lower,
        "weight_upper": weight_upper,
        "direction": "discharge",
        "n_rpts": len(rpts),
        # Lifetime from RPT capacity retention.
        "cycle_life_efc": cycle_life_efc,
        "cycle_life_censored": bool(cycle_life_censored),
        "cycle_life_soh_threshold": LIFETIME_SOH_THRESHOLD,
        "initial_rpt_capacity_Ah": initial_capacity_ah,
        "last_rpt_capacity_Ah": float(capacities[-1]),
        "last_rpt_soh": float(soh[-1]),
        "last_rpt_efc": float(rpt_efcs[-1]),
        # Paper-style cycling-condition features.
        "temperature_C": temperature_c,
        "ambient_temperature_raw_C": ambient_temperature_raw,
        "temperature_source": "vtc6_protocol",
        "charge_c_rate": charge_c_rate,
        "charge_c_rate_raw": condition["charge_c_rate_raw"],
        "discharge_c_rate": discharge_c_rate,
        "discharge_c_rate_raw": condition["discharge_c_rate_raw"],
        # Protocol/audit information.
        "charge_cc_c_rate_protocol": protocol["charge_cc_c_rate_protocol"],
        "charge_voltage_V": protocol["charge_voltage_V"],
        "charge_c_rate_bins_used": condition["charge_c_rate_bins_used"],
        "discharge_c_rate_bins_used": condition["discharge_c_rate_bins_used"],
        "n_condition_cycles": condition["n_condition_cycles"],
        "surface_temperature_mean_C": condition["surface_temperature_mean_C"],
        "surface_temperature_p05_C": condition["surface_temperature_p05_C"],
        "surface_temperature_max_C": condition["surface_temperature_max_C"],
    }

    rpt_summary = pd.DataFrame({
        "rpt_number": np.arange(len(rpts), dtype=int),
        "rpt_efc": rpt_efcs,
        "capacity_Ah": capacities,
        "capacity_counter_Ah": [r["counter_capacity_Ah"] for r in rpts],
        "soh": soh,
    })

    if return_rpt_summary:
        return curves, metadata, rpt_summary
    return curves, metadata
