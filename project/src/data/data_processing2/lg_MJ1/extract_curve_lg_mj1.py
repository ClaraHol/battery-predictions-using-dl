"""Extract LG-MJ1 VITO/EVERLASTING RPT curves, lifetime and conditions.

The VITO CSV files are very large (roughly 0.4--0.8 GB each), so this extractor
streams them in chunks and never loads the complete time series into memory.

Important dataset-specific points
---------------------------------
* ``Step`` is deliberately NOT used.  It is not a reliable chronological key in
  these files and may jump by large amounts.
* ``Total_Time_Seconds`` is also not assumed to be globally monotone.  Some files
  reset/rebase this counter during the experiment.  We reconstruct a monotone
  elapsed-time axis by accumulating sample-to-sample increments.  Positive raw
  counter increments are used directly; at a reset/non-positive increment the
  timestamp column is used to bridge the boundary, with a local cadence fallback
  only if neither source provides a positive interval.
* LG INR18650-MJ1 nominal capacity is 3.5 Ah.
* VITO defines SOH from the SECOND 1C constant-current discharge in each
  periodic check-up, relative to the second 1C discharge in the initial check-up.
* Ageing is periodically interrupted by a reference/check-up test.  We detect
  the two standard 1C discharges by behaviour: approximately -3.5 A, starting
  near the upper voltage limit and ending near the lower voltage limit.
* EFC is reset to zero at the beginning of the initial check-up and uses total
  bidirectional Ah throughput divided by 2*Q_nom.
* Curves are stored with discharge current as a positive magnitude.

Returns
-------
curves : DataFrame
    The three model observations: efc0, efc50_lower and efc50_upper.
metadata : dict
    EFC interpolation, lifetime, temperature/C-rate and provenance metadata.
rpt_summary : DataFrame
    One row per detected check-up, using the selected second 1C discharge.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


Q_NOM_AH = 3.5
TARGET_EFC = 50.0
LIFETIME_SOH_THRESHOLD = 0.90
REFERENCE_DISCHARGE_A = 3.5  # 1C standard capacity discharge

CHUNK_SIZE = 250_000
ACTIVE_CURRENT_A = 0.05
REFERENCE_CURRENT_TOL_A = 0.40
MAX_SEGMENT_GAP_S = 180.0
MIN_STABILIZATION_REST_H = 7.0
MAX_REST_TO_CHECKUP_H = 12.0
MAX_PAIR_EFC_GAP = 3.0
MAX_PAIR_TIME_H = 12.0

NUMERIC_USECOLS = [
    "Total_Time_Seconds",
    "Voltage_V",
    "Current_A",
    "Charge_Capacity_Ah",
    "Discharge_Capacity_Ah",
]

CYCLIC_RE = re.compile(
    r"^Cycl_T(?P<temperature>\d+)_SOC(?P<soc_lo>\d+)-(?P<soc_hi>\d+)_"
    r"Dch(?P<discharge>[0-9.]+)C_Ch(?P<charge>[0-9.]+)C_"
    r"Vito_Cell(?P<cell>\d+)_AllData\.csv$",
    re.IGNORECASE,
)

DRIVING_RE = re.compile(
    r"^(?P<kind>DrivingAgeing(?:Power)?)_T(?P<temperature>\d+)_"
    r"SOC(?P<soc_lo>\d+)-(?P<soc_hi>\d+)_Vito_Cell(?P<cell>\d+)_AllData\.csv$",
    re.IGNORECASE,
)


def _round_to_5(value: float) -> float:
    return float(np.floor(float(value) / 5.0 + 0.5) * 5.0)


def _round_to_half(value: float) -> float:
    return float(np.floor(float(value) / 0.5 + 0.5) * 0.5)


def parse_protocol(file_path: str | Path) -> dict[str, Any]:
    """Parse the stable protocol information encoded in a VITO filename."""
    path = Path(file_path)

    m = CYCLIC_RE.match(path.name)
    if m is not None:
        temperature = float(m.group("temperature"))
        return {
            "cell_number": int(m.group("cell")),
            "test_type": "cyclic",
            "temperature_C_raw": temperature,
            "temperature_C": _round_to_5(temperature),
            "soc_lower_pct": float(m.group("soc_lo")),
            "soc_upper_pct": float(m.group("soc_hi")),
            "charge_c_rate_filename": float(m.group("charge")),
            "discharge_c_rate_filename": float(m.group("discharge")),
            "condition": path.stem.rsplit("_Vito_Cell", 1)[0],
        }

    m = DRIVING_RE.match(path.name)
    if m is not None:
        temperature = float(m.group("temperature"))
        kind = m.group("kind").lower()
        test_type = "driving_power" if "power" in kind else "driving_current"
        return {
            "cell_number": int(m.group("cell")),
            "test_type": test_type,
            "temperature_C_raw": temperature,
            "temperature_C": _round_to_5(temperature),
            "soc_lower_pct": float(m.group("soc_lo")),
            "soc_upper_pct": float(m.group("soc_hi")),
            "charge_c_rate_filename": np.nan,
            "discharge_c_rate_filename": np.nan,
            "condition": path.stem.rsplit("_Vito_Cell", 1)[0],
        }

    raise ValueError(
        f"Unexpected LG-MJ1 VITO filename: {path.name}. "
        "Expected Cycl_..., DrivingAgeing_... or DrivingAgeingPower_..."
    )


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _positive_counter_increment(values: np.ndarray, previous: float | None) -> tuple[np.ndarray, float]:
    """Reset-safe positive increments for one cumulative capacity counter."""
    if len(values) == 0:
        return np.empty(0, dtype=float), previous if previous is not None else np.nan

    if previous is None:
        prior = np.r_[values[0], values[:-1]]
    else:
        prior = np.r_[previous, values[:-1]]
    delta = values - prior
    delta = np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)
    return delta, float(values[-1])


def _reconstruct_elapsed_time(
    raw_time_s: np.ndarray,
    timestamps: pd.Series,
    *,
    previous_raw_time_s: float | None,
    previous_timestamp: pd.Timestamp | None,
    previous_elapsed_time_s: float | None,
    previous_positive_dt_s: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct a strictly increasing elapsed-time axis across timer resets.

    ``Total_Time_Seconds`` is treated as a *local* timer rather than a global
    clock.  Within monotone stretches its positive increment is trustworthy and
    is accumulated.  If the timer resets, repeats, or moves backwards, the
    wall-clock timestamp increment is used for that boundary.  Only when both
    are unusable do we fall back to the most recent positive sampling interval
    (1 s for the very first unresolved interval).

    This preserves real long rests and acquisition gaps: positive raw timer
    increments are never clipped merely because they are large.
    """
    raw = np.asarray(raw_time_s, dtype=float)
    ts = pd.to_datetime(timestamps, errors="coerce")
    n = len(raw)
    elapsed = np.empty(n, dtype=float)

    prev_raw = previous_raw_time_s
    prev_ts = previous_timestamp
    prev_elapsed = 0.0 if previous_elapsed_time_s is None else float(previous_elapsed_time_s)
    last_positive_dt = previous_positive_dt_s

    n_resets = 0
    n_timestamp_repairs = 0
    n_fallback_repairs = 0
    max_backward_jump_s = 0.0

    for j in range(n):
        current_raw = float(raw[j])
        current_ts = ts.iloc[j]

        # The first sample defines elapsed t=0 unless continuing from a prior
        # chunk, in which case it is handled exactly like any other boundary.
        if prev_raw is None and previous_elapsed_time_s is None and j == 0:
            dt = 0.0
        else:
            raw_dt = current_raw - float(prev_raw) if prev_raw is not None else np.nan
            if np.isfinite(raw_dt) and raw_dt > 0.0:
                dt = float(raw_dt)
                last_positive_dt = dt
            else:
                n_resets += 1
                if np.isfinite(raw_dt) and raw_dt < 0.0:
                    max_backward_jump_s = max(max_backward_jump_s, float(-raw_dt))

                wall_dt = np.nan
                if prev_ts is not None and not pd.isna(prev_ts) and not pd.isna(current_ts):
                    wall_dt = float((current_ts - prev_ts).total_seconds())

                if np.isfinite(wall_dt) and wall_dt > 0.0:
                    dt = wall_dt
                    last_positive_dt = dt
                    n_timestamp_repairs += 1
                else:
                    dt = float(last_positive_dt) if last_positive_dt and last_positive_dt > 0 else 1.0
                    n_fallback_repairs += 1

        prev_elapsed += dt
        elapsed[j] = prev_elapsed
        prev_raw = current_raw
        if not pd.isna(current_ts):
            prev_ts = current_ts

    # Floating-point-safe strict monotonicity. The only permitted zero is the
    # very first elapsed time in the whole file.
    if previous_elapsed_time_s is not None:
        if np.any(np.diff(np.r_[float(previous_elapsed_time_s), elapsed]) <= 0):
            raise ValueError("Failed to reconstruct a strictly increasing elapsed-time axis")
    elif len(elapsed) > 1 and np.any(np.diff(elapsed) <= 0):
        raise ValueError("Failed to reconstruct a strictly increasing elapsed-time axis")

    state = {
        "previous_raw_time_s": float(prev_raw) if prev_raw is not None else None,
        "previous_timestamp": prev_ts,
        "previous_elapsed_time_s": float(elapsed[-1]) if len(elapsed) else previous_elapsed_time_s,
        "previous_positive_dt_s": float(last_positive_dt) if last_positive_dt is not None else None,
        "n_time_resets": int(n_resets),
        "n_timestamp_time_repairs": int(n_timestamp_repairs),
        "n_fallback_time_repairs": int(n_fallback_repairs),
        "max_raw_time_backward_jump_s": float(max_backward_jump_s),
    }
    return elapsed, state


def _run_slices(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start,end) slices for True runs in a boolean mask."""
    if not np.any(mask):
        return []
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _finalize_reference_segment(segment: pd.DataFrame | None) -> dict[str, Any] | None:
    if segment is None or len(segment) < 3:
        return None

    t = segment["time"].to_numpy(dtype=float)
    v = segment["voltage"].to_numpy(dtype=float)
    i = -segment["current"].to_numpy(dtype=float)  # positive magnitude

    if not np.isfinite(t).all() or not np.isfinite(v).all() or not np.isfinite(i).all():
        return None
    if np.any(np.diff(t) <= 0):
        return None

    duration_h = (t[-1] - t[0]) / 3600.0
    capacity_ah = _trapz(i, t) / 3600.0
    median_current = float(np.median(i))

    # A standard VITO capacity cycle is a complete 1C discharge.  Partial
    # 10--90% ageing discharges and dynamic driving segments fail the voltage
    # and/or capacity criteria even if they happen to pass through 3.5 A.
    if not 0.65 <= duration_h <= 1.30:
        return None
    if v[0] < 4.00 or v[-1] > 2.65:
        return None
    if not 2.3 <= capacity_ah <= 4.0:
        return None
    if abs(median_current - REFERENCE_DISCHARGE_A) > REFERENCE_CURRENT_TOL_A:
        return None

    q = segment["q_dis"].to_numpy(dtype=float)
    counter_capacity = float(np.nanmax(q) - np.nanmin(q)) if len(q) else np.nan

    curve = pd.DataFrame(
        {
            "time_s": t - t[0],
            "voltage_V": v,
            "current_A": i,
        }
    )

    return {
        "start_time_s": float(t[0]),
        "end_time_s": float(t[-1]),
        "start_raw_efc": float(segment["raw_efc"].iloc[0]),
        "end_raw_efc": float(segment["raw_efc"].iloc[-1]),
        "start_row": int(segment["row_number"].iloc[0]),
        "end_row": int(segment["row_number"].iloc[-1]),
        "duration_h": float(duration_h),
        "capacity_Ah": float(capacity_ah),
        "counter_capacity_Ah": counter_capacity,
        "median_current_A": median_current,
        "voltage_start_V": float(v[0]),
        "voltage_end_V": float(v[-1]),
        "curve": curve,
    }


def _finalize_rest_segment(rest: dict[str, float] | None) -> dict[str, float] | None:
    if rest is None:
        return None
    duration_h = (rest["end_time_s"] - rest["start_time_s"]) / 3600.0
    if duration_h < MIN_STABILIZATION_REST_H:
        return None
    rest = dict(rest)
    rest["duration_h"] = float(duration_h)
    return rest


def _scan_file(file_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    """Stream one huge CSV, repair its time base and detect full 1C discharges."""
    candidates: list[dict[str, Any]] = []
    long_rests: list[dict[str, float]] = []

    reference_carry: pd.DataFrame | None = None
    rest_carry: dict[str, float] | None = None

    prev_q_ch: float | None = None
    prev_q_dis: float | None = None
    throughput_ah = 0.0
    row_offset = 0

    # Time-reconstruction state.  Do not use Step for chronology.
    previous_raw_time_s: float | None = None
    previous_timestamp: pd.Timestamp | None = None
    previous_elapsed_time_s: float | None = None
    previous_positive_dt_s: float | None = None
    n_time_resets = 0
    n_timestamp_time_repairs = 0
    n_fallback_time_repairs = 0
    max_raw_time_backward_jump_s = 0.0

    # The first column is the wall-clock timestamp in the VITO files. Its
    # header is blank in the supplied CSV and pandas commonly calls it
    # ``Unnamed: 0``; determine it from the file rather than hard-coding it.
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    missing = [c for c in NUMERIC_USECOLS if c not in header]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    timestamp_column = next((c for c in header if c not in NUMERIC_USECOLS and c != "Step"), None)
    # Some pre-trimmed exports omit the wall-clock timestamp entirely.  That is
    # still processable: timer resets are bridged with the most recent positive
    # sampling interval instead of rejecting the cell.
    read_columns = ([timestamp_column] if timestamp_column is not None else []) + NUMERIC_USECOLS

    # For dynamic profiles we retain a transparent effective-rate diagnostic.
    charge_amp_seconds = 0.0
    charge_active_seconds = 0.0
    discharge_amp_seconds = 0.0
    discharge_active_seconds = 0.0

    for chunk in pd.read_csv(file_path, usecols=read_columns, chunksize=CHUNK_SIZE):
        for column in NUMERIC_USECOLS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        if chunk[NUMERIC_USECOLS].isna().any().any():
            bad = chunk[NUMERIC_USECOLS].columns[chunk[NUMERIC_USECOLS].isna().any()].tolist()
            raise ValueError(f"Non-numeric/NaN values in {bad}")

        raw_t = chunk["Total_Time_Seconds"].to_numpy(dtype=float)
        v = chunk["Voltage_V"].to_numpy(dtype=float)
        i = chunk["Current_A"].to_numpy(dtype=float)
        q_ch = chunk["Charge_Capacity_Ah"].to_numpy(dtype=float)
        q_dis = chunk["Discharge_Capacity_Ah"].to_numpy(dtype=float)

        if len(raw_t) == 0:
            continue

        timestamp_values = (
            chunk[timestamp_column]
            if timestamp_column is not None
            else pd.Series(pd.NaT, index=chunk.index)
        )
        t, time_state = _reconstruct_elapsed_time(
            raw_t,
            timestamp_values,
            previous_raw_time_s=previous_raw_time_s,
            previous_timestamp=previous_timestamp,
            previous_elapsed_time_s=previous_elapsed_time_s,
            previous_positive_dt_s=previous_positive_dt_s,
        )
        previous_raw_time_s = time_state["previous_raw_time_s"]
        previous_timestamp = time_state["previous_timestamp"]
        previous_elapsed_time_s = time_state["previous_elapsed_time_s"]
        previous_positive_dt_s = time_state["previous_positive_dt_s"]
        n_time_resets += time_state["n_time_resets"]
        n_timestamp_time_repairs += time_state["n_timestamp_time_repairs"]
        n_fallback_time_repairs += time_state["n_fallback_time_repairs"]
        max_raw_time_backward_jump_s = max(
            max_raw_time_backward_jump_s,
            time_state["max_raw_time_backward_jump_s"],
        )

        dq_ch, prev_q_ch = _positive_counter_increment(q_ch, prev_q_ch)
        dq_dis, prev_q_dis = _positive_counter_increment(q_dis, prev_q_dis)
        chunk_throughput = dq_ch + dq_dis
        raw_efc = (throughput_ah + np.cumsum(chunk_throughput)) / (2.0 * Q_NOM_AH)
        throughput_ah += float(chunk_throughput.sum())

        # Time-weighted active-current mean used only for driving-profile
        # operational-condition metadata. Ignore large acquisition gaps here;
        # long gaps are still retained in the repaired global time axis.
        if row_offset == 0:
            t_prior = np.r_[t[0], t[:-1]]
        else:
            # The prior repaired time is the current chunk's first time minus
            # the already reconstructed first boundary increment. We only need
            # active-current weighting, so a zero first dt is conservative.
            t_prior = np.r_[t[0], t[:-1]]
        dt = t - t_prior
        valid_dt = np.where((dt > 0.0) & (dt <= 300.0), dt, 0.0)
        charge_mask = i > ACTIVE_CURRENT_A
        discharge_mask = i < -ACTIVE_CURRENT_A
        charge_amp_seconds += float(np.sum(i[charge_mask] * valid_dt[charge_mask]))
        charge_active_seconds += float(np.sum(valid_dt[charge_mask]))
        discharge_amp_seconds += float(np.sum((-i[discharge_mask]) * valid_dt[discharge_mask]))
        discharge_active_seconds += float(np.sum(valid_dt[discharge_mask]))

        work = pd.DataFrame(
            {
                "time": t,
                "voltage": v,
                "current": i,
                "q_dis": q_dis,
                "raw_efc": raw_efc,
                "row_number": np.arange(row_offset, row_offset + len(chunk), dtype=np.int64),
            }
        )

        # --- Full 1C discharge candidate runs ---
        ref_mask = np.abs(i + REFERENCE_DISCHARGE_A) <= REFERENCE_CURRENT_TOL_A
        slices = _run_slices(ref_mask)

        # If the previous chunk ended inside a run, merge it with this chunk's
        # first run only when the repaired-time acquisition gap is contiguous.
        if reference_carry is not None:
            if slices and slices[0][0] == 0:
                s, e = slices.pop(0)
                head = work.iloc[s:e]
                gap = float(head["time"].iloc[0] - reference_carry["time"].iloc[-1])
                if 0 < gap <= MAX_SEGMENT_GAP_S:
                    reference_carry = pd.concat([reference_carry, head], ignore_index=True)
                else:
                    candidate = _finalize_reference_segment(reference_carry)
                    if candidate is not None:
                        candidates.append(candidate)
                    reference_carry = head.copy()
                if e < len(work):
                    candidate = _finalize_reference_segment(reference_carry)
                    if candidate is not None:
                        candidates.append(candidate)
                    reference_carry = None
            else:
                candidate = _finalize_reference_segment(reference_carry)
                if candidate is not None:
                    candidates.append(candidate)
                reference_carry = None

        for s, e in slices:
            run = work.iloc[s:e].copy()
            if e == len(work):
                reference_carry = run
            else:
                candidate = _finalize_reference_segment(run)
                if candidate is not None:
                    candidates.append(candidate)

        # --- Long zero-current stabilization rests ---
        rest_mask = np.abs(i) <= 0.02
        rest_slices = _run_slices(rest_mask)
        if rest_carry is not None:
            if rest_slices and rest_slices[0][0] == 0:
                s, e = rest_slices.pop(0)
                end_time = float(work["time"].iloc[e - 1])
                rest_carry["end_time_s"] = end_time
                rest_carry["end_raw_efc"] = float(work["raw_efc"].iloc[e - 1])
                if e < len(work):
                    finalized = _finalize_rest_segment(rest_carry)
                    if finalized is not None:
                        long_rests.append(finalized)
                    rest_carry = None
            else:
                finalized = _finalize_rest_segment(rest_carry)
                if finalized is not None:
                    long_rests.append(finalized)
                rest_carry = None

        for s, e in rest_slices:
            rest = {
                "start_time_s": float(work["time"].iloc[s]),
                "end_time_s": float(work["time"].iloc[e - 1]),
                "start_raw_efc": float(work["raw_efc"].iloc[s]),
                "end_raw_efc": float(work["raw_efc"].iloc[e - 1]),
            }
            if e == len(work):
                rest_carry = rest
            else:
                finalized = _finalize_rest_segment(rest)
                if finalized is not None:
                    long_rests.append(finalized)

        row_offset += len(chunk)

    if reference_carry is not None:
        candidate = _finalize_reference_segment(reference_carry)
        if candidate is not None:
            candidates.append(candidate)
    if rest_carry is not None:
        finalized = _finalize_rest_segment(rest_carry)
        if finalized is not None:
            long_rests.append(finalized)

    condition_stats = {
        "charge_active_mean_A": (
            charge_amp_seconds / charge_active_seconds if charge_active_seconds > 0 else np.nan
        ),
        "discharge_active_mean_A": (
            discharge_amp_seconds / discharge_active_seconds if discharge_active_seconds > 0 else np.nan
        ),
        "total_throughput_Ah": float(throughput_ah),
        "n_rows": int(row_offset),
        "reconstructed_duration_s": float(previous_elapsed_time_s or 0.0),
        "n_time_resets": int(n_time_resets),
        "n_timestamp_time_repairs": int(n_timestamp_time_repairs),
        "n_fallback_time_repairs": int(n_fallback_time_repairs),
        "max_raw_time_backward_jump_s": float(max_raw_time_backward_jump_s),
        "time_axis_method": "cumulative_positive_raw_dt_with_timestamp_reset_repair",
    }
    return candidates, long_rests, condition_stats


def _group_checkups(
    candidates: list[dict[str, Any]], long_rests: list[dict[str, float]]
) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("No complete 1C reference discharges detected")

    candidates = sorted(candidates, key=lambda x: x["start_time_s"])
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for candidate in candidates:
        if not current:
            current = [candidate]
            continue
        previous = current[-1]
        efc_gap = candidate["start_raw_efc"] - previous["start_raw_efc"]
        time_gap_h = (candidate["start_time_s"] - previous["end_time_s"]) / 3600.0
        if 0 <= efc_gap <= MAX_PAIR_EFC_GAP and 0 <= time_gap_h <= MAX_PAIR_TIME_H:
            current.append(candidate)
        else:
            groups.append(current)
            current = [candidate]
    if current:
        groups.append(current)

    checkups: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        first = group[0]

        # The VITO definition uses the second standard 1C cycle. If only one
        # full 1C discharge is detectable we retain it but mark it explicitly;
        # verification will warn rather than silently pretending it was the
        # documented second cycle.
        selected_pos = 1 if len(group) >= 2 else 0
        selected = group[selected_pos]
        selection_status = "second_standard_cycle" if len(group) >= 2 else "single_detected_cycle"

        checkup_raw_efc = first["start_raw_efc"]
        checkup_start_source = "first_reference_discharge"
        preceding = [
            rest
            for rest in long_rests
            if rest["end_time_s"] <= first["start_time_s"]
            and (first["start_time_s"] - rest["end_time_s"]) / 3600.0 <= MAX_REST_TO_CHECKUP_H
        ]
        if preceding:
            rest = max(preceding, key=lambda x: x["end_time_s"])
            checkup_raw_efc = rest["start_raw_efc"]
            checkup_start_source = "preceding_stabilization_rest"

        checkups.append(
            {
                "rpt_index": int(index),
                "checkup_raw_efc": float(checkup_raw_efc),
                "checkup_start_source": checkup_start_source,
                "n_full_1c_discharges": int(len(group)),
                "selected_cycle_in_checkup": int(selected_pos + 1),
                "selection_status": selection_status,
                "capacity_Ah": float(selected["capacity_Ah"]),
                "counter_capacity_Ah": float(selected["counter_capacity_Ah"]),
                "median_current_A": float(selected["median_current_A"]),
                "duration_h": float(selected["duration_h"]),
                "voltage_start_V": float(selected["voltage_start_V"]),
                "voltage_end_V": float(selected["voltage_end_V"]),
                "selected_start_time_s": float(selected["start_time_s"]),
                "selected_end_time_s": float(selected["end_time_s"]),
                "selected_start_row": int(selected["start_row"]),
                "selected_end_row": int(selected["end_row"]),
                "curve": selected["curve"],
            }
        )

    # EFC origin is the beginning of the initial check-up.
    origin = checkups[0]["checkup_raw_efc"]
    for checkup in checkups:
        value = checkup["checkup_raw_efc"] - origin
        checkup["rpt_efc"] = float(0.0 if abs(value) < 1e-10 else value)

    efcs = np.array([x["rpt_efc"] for x in checkups], dtype=float)
    if np.any(np.diff(efcs) <= 0):
        raise ValueError(f"Detected check-up EFC positions are not strictly increasing: {efcs}")

    return checkups


def _summary_from_checkups(checkups: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for checkup in checkups:
        rows.append({k: v for k, v in checkup.items() if k != "curve"})
    summary = pd.DataFrame(rows)
    initial_capacity = float(summary.iloc[0]["capacity_Ah"])
    if not 2.5 <= initial_capacity <= 4.0:
        raise ValueError(f"Implausible initial 1C capacity {initial_capacity:.3f} Ah")
    summary["soh"] = summary["capacity_Ah"] / initial_capacity
    return summary


def _lifetime_from_summary(summary: pd.DataFrame) -> dict[str, Any]:
    threshold = LIFETIME_SOH_THRESHOLD
    soh = summary["soh"].to_numpy(dtype=float)
    crossing = np.flatnonzero(soh <= threshold)

    if len(crossing) == 0:
        last = summary.iloc[-1]
        return {
            "cycle_life_efc": float(last["rpt_efc"]),
            "cycle_life_censored": True,
            "cycle_life_soh_threshold": threshold,
            "lifetime_status": "censored_end_of_file",
            "lifetime_lower_rpt_index": int(last["rpt_index"]),
            "lifetime_upper_rpt_index": np.nan,
            "last_observed_soh": float(last["soh"]),
        }

    upper_idx = int(crossing[0])
    if upper_idx == 0:
        raise ValueError("Initial check-up is already at/below the SOH threshold")

    lower = summary.iloc[upper_idx - 1]
    upper = summary.iloc[upper_idx]
    s0, s1 = float(lower["soh"]), float(upper["soh"])
    e0, e1 = float(lower["rpt_efc"]), float(upper["rpt_efc"])
    if not (s0 > threshold >= s1):
        raise ValueError("Invalid SOH threshold bracket")

    if np.isclose(s1, s0):
        life = e1
    else:
        weight = (threshold - s0) / (s1 - s0)
        life = e0 + weight * (e1 - e0)

    return {
        "cycle_life_efc": float(life),
        "cycle_life_censored": False,
        "cycle_life_soh_threshold": threshold,
        "lifetime_status": "observed",
        "lifetime_lower_rpt_index": int(lower["rpt_index"]),
        "lifetime_upper_rpt_index": int(upper["rpt_index"]),
        "last_observed_soh": float(summary.iloc[-1]["soh"]),
    }


def _condition_features(protocol: dict[str, Any], stats: dict[str, float]) -> dict[str, Any]:
    if protocol["test_type"] == "cyclic":
        charge_raw = float(protocol["charge_c_rate_filename"])
        discharge_raw = float(protocol["discharge_c_rate_filename"])
        method = "filename_protocol"
    else:
        charge_raw = float(stats["charge_active_mean_A"] / Q_NOM_AH)
        discharge_raw = float(stats["discharge_active_mean_A"] / Q_NOM_AH)
        method = "time_weighted_active_current_full_file"

    return {
        "charge_c_rate_raw": charge_raw,
        "discharge_c_rate_raw": discharge_raw,
        "charge_c_rate": _round_to_half(charge_raw) if np.isfinite(charge_raw) else np.nan,
        "discharge_c_rate": _round_to_half(discharge_raw) if np.isfinite(discharge_raw) else np.nan,
        "c_rate_method": method,
    }


def extract_curve_lg_mj1(
    file_path: str | Path,
    *,
    target_efc: float = TARGET_EFC,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Extract three RPT discharge curves, metadata and RPT SOH summary."""
    file_path = Path(file_path)
    protocol = parse_protocol(file_path)
    candidates, long_rests, stats = _scan_file(file_path)
    checkups = _group_checkups(candidates, long_rests)
    summary = _summary_from_checkups(checkups)

    efcs = summary["rpt_efc"].to_numpy(dtype=float)
    lower_pos = np.flatnonzero(efcs < target_efc)
    upper_pos = np.flatnonzero(efcs > target_efc)
    if not len(lower_pos) or not len(upper_pos):
        raise ValueError(
            f"EFC{target_efc:g} is not strictly bracketed by detected check-ups; "
            f"RPT EFCs={np.round(efcs, 3).tolist()}"
        )

    lower_idx = int(lower_pos[-1])
    upper_idx = int(upper_pos[0])
    lower = summary.iloc[lower_idx]
    upper = summary.iloc[upper_idx]
    lower_efc = float(lower["rpt_efc"])
    upper_efc = float(upper["rpt_efc"])
    weight_upper = (target_efc - lower_efc) / (upper_efc - lower_efc)

    selection = {
        "efc0": 0,
        "efc50_lower": lower_idx,
        "efc50_upper": upper_idx,
    }
    frames = []
    for curve_type, checkup_pos in selection.items():
        source = checkups[checkup_pos]
        curve = source["curve"].copy()
        curve["curve_type"] = curve_type
        curve["rpt_efc"] = float(source["rpt_efc"])
        curve["rpt_index"] = int(source["rpt_index"])
        frames.append(curve)

    lifetime = _lifetime_from_summary(summary)
    conditions = _condition_features(protocol, stats)

    metadata = {
        "efc0": 0.0,
        "efc50_lower": lower_efc,
        "efc50_upper": upper_efc,
        "weight_lower": float(1.0 - weight_upper),
        "weight_upper": float(weight_upper),
        "direction": "discharge",
        "efc0_rpt_index": int(summary.iloc[0]["rpt_index"]),
        "efc50_lower_rpt_index": int(lower["rpt_index"]),
        "efc50_upper_rpt_index": int(upper["rpt_index"]),
        "n_rpts": int(len(summary)),
        "initial_capacity_Ah": float(summary.iloc[0]["capacity_Ah"]),
        "temperature_C_raw": protocol["temperature_C_raw"],
        "temperature_C": protocol["temperature_C"],
        "soc_lower_pct": protocol["soc_lower_pct"],
        "soc_upper_pct": protocol["soc_upper_pct"],
        "test_type": protocol["test_type"],
        "condition": protocol["condition"],
        "cell_number": protocol["cell_number"],
        "source_file": file_path.name,
        "source_dataset": "EVERLASTING_VITO_25_45C",
        "dataset_provenance": "EVERLASTING_VITO",
        "source_family": "lg_mj1",
        "efc_definition": "bidirectional Ah throughput/(2*3.5Ah), zeroed at initial check-up start",
        "rpt_definition": "second complete 1C CC discharge in each VITO check-up",
        "n_rows_scanned": int(stats["n_rows"]),
        "total_throughput_Ah": float(stats["total_throughput_Ah"]),
        "reconstructed_duration_s": float(stats["reconstructed_duration_s"]),
        "time_axis_method": stats["time_axis_method"],
        "n_time_resets": int(stats["n_time_resets"]),
        "n_timestamp_time_repairs": int(stats["n_timestamp_time_repairs"]),
        "n_fallback_time_repairs": int(stats["n_fallback_time_repairs"]),
        "max_raw_time_backward_jump_s": float(stats["max_raw_time_backward_jump_s"]),
        **conditions,
        **lifetime,
    }

    # Useful audit flags without rejecting otherwise usable cells.
    metadata["n_rpts_single_detected_cycle"] = int(
        (summary["selection_status"] == "single_detected_cycle").sum()
    )
    metadata["n_rpts_nonpair"] = int((summary["n_full_1c_discharges"] != 2).sum())

    return pd.concat(frames, ignore_index=True), metadata, summary
