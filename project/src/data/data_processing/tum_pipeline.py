
"""
VTC5a -> existing pipeline (replaces parse_vtc5a + the vtc5a branch of process_cell).

Needs (same folder / on PYTHONPATH):
  build_cell_csvs.py   (loader helpers)
  config_and_cleaning  (your build_cell_record)
and the per-cell summaries written by build_cell_csvs.py (SUMMARY_DIR).

Usage:
    idx = build_index("/path/to/dataset_root")          # once, scans all .mat files
    meta, efc0_df, efc50_df = process_cell_vtc5a("BW-VTC-164", idx)
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tum_processing import load_fields, to_hours, I_THR, C_NOM

DISCHARGE_END_I_THR = 0.05  
SUMMARY_DIR = Path("/work3/claho/battery_datasets/sony_vtc5a/summaries")         # <- output folder of build_cell_csvs.py
DATA_ROOT = Path("/work3/claho/battery_datasets/sony_vtc5a/")

# ------------------------------------------------------------------ file index
def build_index(root):
    idx = defaultdict(
        lambda: {
            "cu": defaultdict(list),
            "cyc": defaultdict(list)
        }
    )

    for f in sorted(Path(root).rglob("*.mat")):
        m_id = re.search(r"BW-VTC-(\d+)", f.name)
        m_cu = re.search(r"(?:^|[\\/])CU(\d+)(?:_[^\\/]+)?(?:[\\/])", str(f), re.I)
        m_cy = re.search(r"(?:^|[\\/])Cyc(\d+)(?:[\\/])",str(f), re.I)

        if not m_id or not (m_cu or m_cy):
            continue

        cell = int(m_id.group(1))

        if m_cu: idx[cell]["cu"][int(m_cu.group(1))].append(f)
        else:  idx[cell]["cyc"][int(m_cy.group(1))].append(f)

    return idx

def debug_tum_files(root, cell_num=389):
    root = Path(root)

    print(f"Scanning: {root.resolve()}")
    print(f"Root exists: {root.exists()}")

    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".mat"
        and re.search(
            rf"(?<!\d){cell_num}(?!\d)",
            str(p)
        )
    ]

    print(f"\nFound {len(files)} possible files:")

    for f in files:
        m_id = re.search(r"BW-VTC-(\d+)", f.name, re.I)
        m_cu = re.search(r"(?:^|[\\/])CU(\d+)(?:_[^\\/]+)?(?:[\\/])", str(f), re.I)
        m_cy = re.search(r"(?:^|[\\/])Cyc(\d+)(?:[\\/])", str(f), re.I)

        print(
            f.name,
            "cell:", m_id.group(1) if m_id else None,
            "CU:", m_cu.group(1) if m_cu else None,
            "Cyc:", m_cy.group(1) if m_cy else None
        )

    return files

def _num(cell_id):
    return int(re.search(r"(\d+)", str(cell_id)).group(1))


def _summary(cell_id, summary_dir=SUMMARY_DIR):
    return pd.read_csv(Path(summary_dir) / f"BW-VTC-{_num(cell_id):03d}.csv")


# ------------------------------------------------------------------ EFC scan
def _make_cref(summary):
    """C_new for interval k = capacity at the last check-up before k (paper definition)."""
    caps = {int(r.CU): r.capacity_Ah for r in summary.itertuples() if pd.notna(r.capacity_Ah)}
    keys = sorted(caps)

    def cref(k):
        prev = [m for m in keys if m < k]
        if prev:
            return caps[prev[-1]]
        return caps[keys[0]] if keys else C_NOM
    return cref


MIN_DISCHARGE_H, MIN_DISCHARGE_AH = 0.1, -0.25      # a "charge" = sustained positive current (not regen pulses)


def _charge_mask(d):
    """Boolean mask of rows belonging to a charging phase.
    With Command: Command == 'Charge'. Without it (dynamic cells 447-458): runs of I > I_THR
    lasting >= MIN_DISCHARGE_H hours AND >= MIN_DISCHARGE_AH Ah, which excludes regenerative pulses."""
    if "Command" in d:
        return d["Command"] == "Discharge"
    t, I = d["Time"], d["I"]
    n = len(I)
    pos = I < -I_THR
    edge = np.flatnonzero(np.diff(np.r_[False, pos, False].astype(int)))
    s, e = edge[::2], edge[1::2]
    dq0 = np.r_[0.0, np.cumsum(np.where(pos, I * np.r_[np.clip(np.diff(t), 0, None), 0.0], 0.0))]
    ah = dq0[e] - dq0[s]
    dur = t[np.minimum(e, n - 1)] - t[s]
    mask = np.zeros(n, bool)
    for a, b in zip(s[(dur >= MIN_DISCHARGE_H) & (ah <= MIN_DISCHARGE_AH)], e[(dur >= MIN_DISCHARGE_H) & (ah <= MIN_DISCHARGE_AH)]):
        mask[a:b] = True
    return mask


def _load(path):
    d = load_fields(path, ["Time", "U", "I", "T1", "Command"])
    d["Time"], _ = to_hours(d["Time"])                     # -> hours
    return d


def _discharge_curve(d):
    """
    Extract the largest contiguous discharge phase from
    a capacity-check measurement.

    Assumes negative current corresponds to discharge.
    """

    t = np.asarray(d["Time"], dtype=float)
    I = np.asarray(d["I"], dtype=float)
    V = np.asarray(d["U"], dtype=float)

    if len(t) == 0:
        return None

    # Identify discharge samples
    if "Command" in d:
        command = np.asarray(d["Command"])
        mask = np.array([str(x).strip().lower() == "discharge" for x in command])
    else:
        mask = I < -abs(I_THR)

    # Split into contiguous discharge segments
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))

    starts = edges[::2]
    ends = edges[1::2]

    if len(starts) == 0:
        return None

    # Choose the discharge segment with largest Ah throughput
    best = None
    best_capacity = -np.inf

    for s, e in zip(starts, ends):

        if e - s < 2:
            continue

        # Remove very long segments
        duration_s = (t[e - 1] - t[s]) * 3600

        if duration_s > 20000:
            continue

        dt = np.clip(np.diff(t[s:e]), 0, None)
        current = I[s:e - 1]
        capacity = np.sum(np.maximum(-current, 0) * dt)

        if capacity > best_capacity:
            best_capacity = capacity
            best = (s, e)

    if best is None:
        return None

    s, e = best

    return pd.DataFrame({
        "time_s": (t[s:e] - t[s]) * 3600,
        "voltage_V": V[s:e],
        "current_A": I[s:e]
    }).reset_index(drop=True)

def _trim_flat_tail(curve, max_range_v=0.005, min_duration_s=60, min_points=10):
    """
    Remove a terminal constant-voltage / flat-voltage tail.

    Starting from the final point, walk backwards while the voltage
    remains within max_range_v of the terminal plateau.

    Parameters
    ----------
    max_range_v : float
        Maximum voltage variation allowed in the flat tail.
        0.005 = 5 mV.
    min_duration_s : float
        Minimum duration required before considering it a real flat tail.
    min_points : int
        Minimum number of samples required in the flat tail.
    """

    if curve is None or len(curve) < min_points + 2:
        return curve

    t = curve["time_s"].to_numpy()
    V = curve["voltage_V"].to_numpy()

    n = len(V)

    # Start from final voltage
    vmin = V[-1]
    vmax = V[-1]
    start = n - 1

    # Walk backwards while voltage remains in a narrow band
    for i in range(n - 2, -1, -1):

        new_min = min(vmin, V[i])
        new_max = max(vmax, V[i])

        if new_max - new_min > max_range_v:
            break

        vmin = new_min
        vmax = new_max
        start = i

    tail_points = n - start
    tail_duration = t[-1] - t[start]

    # Only trim if it is clearly a real terminal plateau
    if (tail_points >= min_points and tail_duration >= min_duration_s and start > 1):
        # Keep the first point of the plateau as the endpoint
        curve = curve.iloc[:start + 1].copy()

    return curve.reset_index(drop=True)

def _capacity_check_curves_old(cell_id, idx, summary_dir):
    """
    Load discharge curves from capacity-check files only.

    Return a list of:
        EFC, CU, curve
    """

    cu_files = idx[_num(cell_id)]["cu"]

    if not cu_files:
        return []

    summary = _summary(cell_id, summary_dir)
    checks = []

    for cu, paths in sorted(cu_files.items()):
        rows = summary.loc[(summary["CU"] == cu) & summary["EFC"].notna()]
        if rows.empty:
            continue

        efc = float(rows["EFC"].iloc[0])
        best_curve = None
        best_capacity = -np.inf

        for path in paths:
            d = _load(path)
            curve = _discharge_curve(d)

            if curve is None:
                continue

            t_h = curve["time_s"].to_numpy() / 3600
            I = curve["current_A"].to_numpy()
            capacity = np.sum(np.maximum(-I[:-1], 0) * np.diff(t_h))

            if capacity > best_capacity:
                best_capacity = capacity
                best_curve = curve

        if best_curve is not None:
            checks.append({"EFC": efc, "CU": cu, "curve": best_curve})

    return sorted(checks, key=lambda x: x["EFC"])

def _capacity_check_curves(cell_id, idx, summary_dir, verbose=True):
    num = _num(cell_id)
    cu_files = idx[num]["cu"]
    why = []

    if not cu_files:
        n_with_cu = sum(1 for v in idx.values() if v["cu"])
        why.append(
            f"idx has no CU files for cell {num} "
            f"(idx has CU files for {n_with_cu} cells)"
        )

    summary = _summary(cell_id, summary_dir)
    checks = []

    for cu, paths in sorted(cu_files.items()):

        rows = summary.loc[
            (summary["CU"] == cu)
            & summary["EFC_total"].notna()
        ]

        if rows.empty:
            why.append(
                f"CU{cu}: no summary row with a valid EFC "
                f"(summary CU numbers: "
                f"{sorted(summary['CU'].astype(int).unique())[:6]}...)"
            )
            continue

        efc = float(rows["EFC_total"].iloc[0])

        best_curve = None
        best_capacity = -np.inf

        for path in paths:

            d = _load(path)
            curve = _discharge_curve(d)

            if curve is None:

                cmd = (
                    list(
                        pd.unique(
                            pd.Series(
                                np.asarray(d["Command"])
                            ).astype(str).str.strip()
                        )
                    )[:5]
                    if "Command" in d
                    else "no Command field"
                )

                I = np.asarray(d["I"], float)

                why.append(
                    f"CU{cu} {Path(path).name}: "
                    f"no discharge segment found | "
                    f"fields={list(d)} | "
                    f"Command values={cmd} | "
                    f"I range=({np.nanmin(I):.2f}, "
                    f"{np.nanmax(I):.2f}) | "
                    f"rows={len(I)}"
                )

                continue

            # -----------------------------------------
            # Remove constant-voltage tail
            # -----------------------------------------
            curve = _trim_flat_tail(
                curve,
                max_range_v=0.005,
                min_duration_s=60,
                min_points=10
            )

            if len(curve) < 2:
                continue

            # -----------------------------------------
            # Calculate capacity
            # -----------------------------------------
            t_h = curve["time_s"].to_numpy() / 3600
            I = curve["current_A"].to_numpy()

            capacity = np.sum(
                np.maximum(-I[:-1], 0) * np.diff(t_h)
            )

            if capacity > best_capacity:
                best_capacity = capacity
                best_curve = curve

        if best_curve is not None:
            checks.append({
                "EFC": efc,
                "CU": cu,
                "curve": best_curve
            })

    if verbose and not checks:
        print(
            f"[{cell_id}] checks=[] because:\n  "
            + "\n  ".join(why[:4])
            + (
                f"\n  ... (+{len(why) - 4} more)"
                if len(why) > 4
                else ""
            )
        )

    return sorted(checks, key=lambda x: x["EFC"])


def vtc5a_curves(cell_id, idx, summary_dir=SUMMARY_DIR, efc_early=0.0, efc_late=50.0):
    """
    Extract discharge curves ONLY from capacity checks.

    Early:  Closest available capacity-check discharge curve.
    Late:   Exact/nearby capacity-check discharge curve,
            or two capacity-check curves bracketing target EFC.
    """

    checks = _capacity_check_curves(cell_id, idx, summary_dir)
    if not checks:
        return None, None

    efcs = np.array([c["EFC"] for c in checks])

    # ---------------- EARLY ----------------
    early = None
    i = int(np.argmin(np.abs(efcs - efc_early)))
    if abs(efcs[i] - efc_early) <= 1.0:
        curve = checks[i]["curve"]
        early = pd.DataFrame({"time_s": curve["time_s"].to_numpy(),"early_curve": curve["voltage_V"].to_numpy()})

    # ---------------- LATE ----------------
    late = None
    i = int(np.argmin(np.abs(efcs - efc_late)))
    if abs(efcs[i] - efc_late) < 0.5:
        curve = checks[i]["curve"]
        v = curve["voltage_V"].to_numpy()
        t = curve["time_s"].to_numpy()

        late = pd.DataFrame({
            "lower_curve_time": t,
            "upper_curve_time": t.copy(),
            "lower_curve": v,
            "upper_curve": v.copy(),
            "weight": 1.0
        })
    else:

        lower = np.where(efcs < efc_late)[0]
        upper = np.where(efcs > efc_late)[0]

        if len(lower) == 0 or len(upper) == 0:
            return early, None

        lo = lower[-1]
        up = upper[0]

        w = (efc_late - efcs[lo])/ (efcs[up] - efcs[lo])

        lc = checks[lo]["curve"]
        uc = checks[up]["curve"]
        lct = lc["time_s"]
        uct = uc["time_s"]
        lcv = lc["voltage_V"]
        ucv = uc["voltage_V"]

        late = pd.DataFrame({
            "lower_curve_time": pd.Series(np.asarray(lct)),
            "upper_curve_time": pd.Series(np.asarray(uct)),
            "lower_curve": pd.Series(np.asarray(lcv)),
            "upper_curve": pd.Series(np.asarray(ucv)),
            "weight": w
        })
    
    return early, late


# ------------------------------------------------------------------ metadata
def vtc5a_cycle_life(cell_id, soh=0.9, summary_dir=SUMMARY_DIR):
    s = _summary(cell_id, summary_dir)
    life = float(s[f"cycle_life_EFC_soh{int(round(soh * 100))}"].iloc[0])
    censored = bool(np.isnan(life))
    if censored:                                            # right-censored: last observed EFC
        life = float(s["EFC"].max())
    return {"cycle_life": life, "censored": censored, "n_full_cycles": int(len(s))}


def process_cell_vtc5a(cell_id, idx, summary_dir=SUMMARY_DIR, soh=0.9,
                       efc_early=0.0, efc_late=50.0):
    """Same return as process_cell: (metadata_dict, efc_early_df, efc_late_df)."""
    from config_and_cleaning import build_cell_record

    s = _summary(cell_id, summary_dir)
    r = s.iloc[0]
    life = vtc5a_cycle_life(cell_id, soh, summary_dir)

    num = lambda v: None if pd.isna(pd.to_numeric(v, errors="coerce")) else float(v)
    ch_nom, dis_nom = num(r.get("C_ch_nominal")), num(r.get("C_dis_nominal"))
    ch_data, dis_data = num(r["mean_C_ch_measured"]), num(r["mean_C_dis_measured"])
    charge_c = ch_nom if ch_nom is not None else ch_data
    discharge_c = dis_nom if dis_nom is not None else dis_data
    temp_C = num(r.get("T_ambient_C"))
    temp_C = temp_C if temp_C is not None else float(r["mean_T_surface_C"])

    meta = build_cell_record(
        cell_id="tum_"+str(cell_id), source_dataset="tum", temp_C_raw=temp_C,
        charge_c_rate_raw=charge_c, discharge_c_rate_raw=discharge_c,
        cycle_life=life["cycle_life"],
        initial_discharge_capacity_Ah=float(s["capacity_Ah"].dropna().iloc[0]),
    )
    mismatch = lambda a, b: a is not None and b not in (None, 0) and abs(a - b) / b > 0.30
    meta.update({
        "cycle_life_censored": life["censored"],
        "n_full_cycles": life["n_full_cycles"],
        "c_rate_source": "overview" if (ch_nom is not None or dis_nom is not None) else "data",
        "charge_c_rate_data": None if ch_data is None else round(ch_data, 3),
        "discharge_c_rate_data": None if dis_data is None else round(dis_data, 3),
        "c_rate_mismatch": bool(mismatch(ch_nom, ch_data) or mismatch(dis_nom, dis_data)),
    })
    early, late = vtc5a_curves(cell_id, idx, summary_dir, efc_early, efc_late)
    return meta, early, late

if __name__ == "__main__":
    import numpy as np
    from tum_processing import load_fields

    def diagnose_time_units(path):
        d = load_fields(path, ["Time", "I"])

        t = np.asarray(d["Time"], dtype=float)
        I = np.asarray(d["I"], dtype=float)

        dt = np.diff(t)
        positive_dt = dt[dt > 0]

        if len(positive_dt) == 0:
            return

        median_dt = np.median(positive_dt)

        # Current held until next measurement
        charge_raw = np.sum(
            np.maximum(I[:-1], 0) *
            np.clip(dt, 0, None)
        )

        # Interpretation A: raw Time is in hours
        ah_if_hours = charge_raw

        # Interpretation B: raw Time is in seconds
        ah_if_seconds = charge_raw / 3600

        print(f"\nFile: {path}")
        print(f"Median raw dt: {median_dt:.8f}")
        print(f"Charge if hours:   {ah_if_hours:.4f} Ah")
        print(f"Charge if seconds: {ah_if_seconds:.4f} Ah")
        print(f"Current range: {I.min():.4f} to {I.max():.4f} A")
    
    
    


    idx = build_index(DATA_ROOT)

    cell = 280

    for interval, paths in sorted(idx[cell]["cyc"].items()):

        if interval > 5:
            break

        print(f"\n===== Cycling interval {interval} =====")

        for path in paths:
            diagnose_time_units(path)

    
    #files = debug_tum_files(DATA_ROOT, 389)




