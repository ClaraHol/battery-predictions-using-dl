
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

 
SUMMARY_DIR = Path("/work3/claho/battery_datasets/sony_vtc5a/processed_new")         # <- output folder of build_cell_csvs.py


# ------------------------------------------------------------------ file index
def build_index(root):
    """idx[cell_num]['cu' | 'cyc'][interval_no] -> sorted list of .mat paths."""
    idx = defaultdict(lambda: {"cu": defaultdict(list), "cyc": defaultdict(list)})
    for f in sorted(Path(root).rglob("*.mat")):
        m_id = re.search(r"BW-VTC-(\d+)", f.name)
        m_cu = re.search(r"(?:^|[\\/])CU(\d+)(?:[\\/])", str(f), re.I)
        m_cy = re.search(r"(?:^|[\\/])Cyc(\d+)(?:[\\/])", str(f), re.I)
        if not m_id or not (m_cu or m_cy):
            continue
        cell = int(m_id.group(1))
        if m_cu:
            idx[cell]["cu"][int(m_cu.group(1))].append(f)
        else:
            idx[cell]["cyc"][int(m_cy.group(1))].append(f)
    return idx


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


MIN_CHARGE_H, MIN_CHARGE_AH = 0.1, 0.25      # a "charge" = sustained positive current (not regen pulses)


def _charge_mask(d):
    """Boolean mask of rows belonging to a charging phase.
    With Command: Command == 'Charge'. Without it (dynamic cells 447-458): runs of I > I_THR
    lasting >= MIN_CHARGE_H hours AND >= MIN_CHARGE_AH Ah, which excludes regenerative pulses."""
    if "Command" in d:
        return d["Command"] == "Charge"
    t, I = d["Time"], d["I"]
    n = len(I)
    pos = I > I_THR
    edge = np.flatnonzero(np.diff(np.r_[False, pos, False].astype(int)))
    s, e = edge[::2], edge[1::2]
    dq0 = np.r_[0.0, np.cumsum(np.where(pos, I * np.r_[np.clip(np.diff(t), 0, None), 0.0], 0.0))]
    ah = dq0[e] - dq0[s]
    dur = t[np.minimum(e, n - 1)] - t[s]
    mask = np.zeros(n, bool)
    for a, b in zip(s[(dur >= MIN_CHARGE_H) & (ah >= MIN_CHARGE_AH)], e[(dur >= MIN_CHARGE_H) & (ah >= MIN_CHARGE_AH)]):
        mask[a:b] = True
    return mask


def _load(path):
    d = load_fields(path, ["Time", "U", "I", "T1", "Command"])
    d["Time"], _ = to_hours(d["Time"])                     # -> hours
    return d


def _segment(d, carry, cref):
    """Split one file into charge-started cycles. Returns (cycles, carry_out).
    A cycle starts where Command switches into 'Charge' and ends at the next such start.
    cycle['efc'] = cumulative EFC (charge throughput / C_new) at the cycle's start."""
    t, I = d["Time"], d["I"]
    dt_next = np.r_[np.clip(np.diff(t), 0, None), 0.0]     # I[k] is held until t[k+1]
    dq = np.where(I > I_THR, I * dt_next, 0.0)
    cs = np.cumsum(dq)
    ch = _charge_mask(d)
    starts = np.flatnonzero(ch & ~np.r_[False, ch[:-1]])
    ends = np.r_[starts[1:], len(t)]
    cycles = [dict(efc=carry + (cs[s - 1] if s > 0 else 0.0) / cref, s=int(s), e=int(e))
              for s, e in zip(starts, ends)]
    return cycles, carry + cs[-1] / cref


def _scan(cyc_files, cref, stop_efc):
    """Walk cycling files in chronological order, collect (efc, file, rows) of every cycle
    until EFC exceeds stop_efc. Only the small per-cycle bookkeeping is kept."""
    cands, carry, n_cyc = [], 0.0, 0
    for n in sorted(cyc_files):
        for path in cyc_files[n]:
            d = _load(path)
            cycles, carry = _segment(d, carry, cref(n))
            for j, c in enumerate(cycles):
                cands.append(dict(c, path=path, interval=n, cycle_no=n_cyc + j))
            n_cyc += len(cycles)
        if carry > stop_efc:
            break
    return cands


def _cycle_frame(c, cell_id):
    """Standard-schema DataFrame for one cycle (+EFC, cell_id)."""
    d = _load(c["path"])
    s, e = c["s"], c["e"]
    t, I = d["Time"][s:e], d["I"][s:e]
    dt = np.r_[np.clip(np.diff(t), 0, None), 0.0]
    dq = I * dt
    out = pd.DataFrame({
        "time_s": (t - t[0]) * 3600.0,
        "voltage_V": d["U"][s:e],
        "current_A": I,
        "temp_C": d["T1"][s:e],
        "cycle_number": c["cycle_no"],
        "charge_capacity_Ah": np.cumsum(np.where(I > 0, dq, 0.0)),
        "discharge_capacity_Ah": np.cumsum(np.where(I < 0, -dq, 0.0)),
    })
    out["charge_block"] = _charge_mask(d)[s:e]      # True only in the real charging phase
    out["EFC"] = c["efc"]
    out["cell_id"] = cell_id
    return out

def remove_cv_phase(df, voltage_tol=0.005, window=15):
    """
    Truncate charging curve at the beginning of the
    sustained upper-voltage plateau.
    """

    if len(df) < window:
        return df.copy()

    v = df["voltage_V"].to_numpy(dtype=float)

    # Smooth voltage
    v_smooth = (
        pd.Series(v)
        .rolling(5, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # Approximate maximum voltage
    v_max = np.max(v_smooth)

    # Detect a nearly constant voltage window
    voltage_range = (
        pd.Series(v_smooth).rolling(window).max()
        - pd.Series(v_smooth).rolling(window).min()
    ).to_numpy()

    plateau = (
        (voltage_range < voltage_tol)
        & (v_smooth >= v_max - voltage_tol)
    )

    indices = np.flatnonzero(plateau)

    if len(indices) == 0:
        return df.copy()

    cutoff = indices[0] - window + 1

    # Avoid returning an empty curve
    if cutoff < 20:
        return df.copy()

    return df.iloc[:cutoff].copy().reset_index(drop=True)

def _charging_curve(df):
    """Charging phase only. Uses charge_block (not current > 0, which would also pick up
    regenerative braking pulses in the dynamic cells)."""
    out = remove_cv_phase(df[df["charge_block"]].copy().reset_index(drop=True))
    return out


def vtc5a_curves(cell_id, idx, summary_dir=SUMMARY_DIR, efc_early=0.0, efc_late=50.0):
    """(efc_early_df, efc_late_df) in the same formats process_cell returns.
    early: full cycle (all rows) closest to efc_early.
    late : DataFrame(lower_curve, upper_curve, weight) of charging voltages, interpolated
           between the cycles bracketing efc_late (single cycle if within 0.5 EFC)."""
    cyc_files = idx[_num(cell_id)]["cyc"]
    if not cyc_files:
        return None, None                                   # calendar cell
    cref = _make_cref(_summary(cell_id, summary_dir))
    cands = _scan(cyc_files, cref, stop_efc=max(efc_early, efc_late) + 5)
    if not cands:
        return None, None
    efcs = np.array([c["efc"] for c in cands])
    cid = str(cell_id)

    early = None
    if efcs.min() <= efc_early + 1.0:   
        curve = _charging_curve(_cycle_frame(cands[int(np.abs(efcs - efc_early).argmin())], cid))
        v = curve["voltage_V"]                 # a cycle within 1 EFC of the target exists
        vt = curve["time_s"]
        early = early = pd.DataFrame({"time_s": vt, "early_curve": v})

    late = None
    if efc_late <= efcs.max():
        i = int(np.abs(efcs - efc_late).argmin())
        if abs(efcs[i] - efc_late) >= 0.5 and (efcs < efc_late).any() and (efcs > efc_late).any():
            lo = int(np.where(efcs == efcs[efcs < efc_late].max())[0][-1])
            up = int(np.where(efcs == efcs[efcs > efc_late].min())[0][0])
            w = (efc_late - efcs[lo]) / (efcs[up] - efcs[lo])
            lc = _charging_curve(_cycle_frame(cands[lo], cid))["voltage_V"]
            uc = _charging_curve(_cycle_frame(cands[up], cid))["voltage_V"]
            lct = _charging_curve(_cycle_frame(cands[lo], cid))["time_s"]
            uct = _charging_curve(_cycle_frame(cands[up], cid))["time_s"]
            late = pd.DataFrame({"lower_curve_time": lct, "upper_curve_time": uct, "lower_curve": lc, "upper_curve": uc, "weight": w})
        else:
            v = _charging_curve(_cycle_frame(cands[i], cid))["voltage_V"]
            vt = _charging_curve(_cycle_frame(cands[i], cid))["time_s"]
            late = pd.DataFrame({"lower_curve_time": vt, "upper_curve_time": vt, "lower_curve": v, "upper_curve": v, "weight": 1})
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



