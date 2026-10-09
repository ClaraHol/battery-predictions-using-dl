"""
VTC5a integration for the existing pipeline.

1. parse_vtc5a(csv_path)      -> drop-in replacement for the parser in parsers.py
2. vtc5a_cycle_life(...)      -> returns the same dict as the cmu/elt branches
3. vtc5a_c_rates(...)         -> optional C-rate overrides for process_cell

The cell summaries come from build_cell_csvs.py (run it once, point SUMMARY_DIR at its output).
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from parsers import STANDARD_COLUMNS
except ImportError:  # fallback so this file can be tested standalone
    STANDARD_COLUMNS = ["time_s", "voltage_V", "current_A", "temp_C", "cycle_number",
                        "charge_capacity_Ah", "discharge_capacity_Ah"]

SUMMARY_DIR = Path("vtc5a_summaries")      # <- output folder of build_cell_csvs.py


# ---------------------------------------------------------------- 1. parser
def parse_vtc5a(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)             # keep acquisition order (no sort by Time, see below)

    t = np.asarray(raw["Time"], dtype=float)
    d = np.diff(t)
    # README says hours; detect seconds from the median sample spacing
    to_s = 1.0 if np.median(d[d > 0]) >= 0.05 else 3600.0
    # Time restarts at file boundaries -> rebuild one continuous clock instead of sorting
    dt_s = np.r_[0.0, np.clip(d, 0, None)] * to_s
    n_resets = int((d < 0).sum())
    if n_resets:
        print(f"{csv_path}: Time resets {n_resets}x (file boundaries?) - continuous clock rebuilt")

    cyc = np.asarray(raw["CycCount"], dtype=int)
    cur = np.asarray(raw["I"], dtype=float)

    out = pd.DataFrame({
        "time_s": np.cumsum(dt_s),
        "voltage_V": np.asarray(raw["U"], dtype=float),
        "current_A": cur,
        "temp_C": np.asarray(raw["T1"], dtype=float),
        "cycle_number": cyc,
    })

    # 'Ah' is a signed cumulative counter -> NOT a per-cycle capacity. Integrate |I| within
    # each cycle_number instead (first row of a cycle gets dt=0 so gaps are not counted).
    new_cycle = np.r_[True, cyc[1:] != cyc[:-1]]
    dq = cur * np.where(new_cycle, 0.0, dt_s) / 3600.0
    out["charge_capacity_Ah"] = pd.Series(np.where(cur > 0, dq, 0.0)).groupby(cyc).cumsum().to_numpy()
    out["discharge_capacity_Ah"] = pd.Series(np.where(cur < 0, -dq, 0.0)).groupby(cyc).cumsum().to_numpy()
    return out[STANDARD_COLUMNS]


# ---------------------------------------------------------------- 2. cycle life
def _summary(cell_id: str, summary_dir=SUMMARY_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(summary_dir) / f"{cell_id}.csv")


def vtc5a_cycle_life(cell_id: str, soh: float = 0.9, summary_dir=SUMMARY_DIR) -> dict:
    """Cycle life in EFC at `soh` (from check-up capacities). Same dict as the other branches.
    If SOH never reaches the threshold: cycle_life = last observed EFC, censored=True."""
    s = _summary(cell_id, summary_dir)
    col = f"cycle_life_EFC_soh{int(round(soh * 100))}"
    life = float(s[col].iloc[0])
    censored = bool(np.isnan(life))
    if censored:
        life = float(s["EFC"].max())
    return {"cycle_life": life, "censored": censored, "n_full_cycles": int(len(s))}


# ---------------------------------------------------------------- 3. C-rates
def vtc5a_c_rates(cell_id: str, summary_dir=SUMMARY_DIR):
    """(charge, discharge) nominal C-rates from the overview, None where not defined
    (e.g. dynamic/automotive cells) so process_cell falls back to the data-derived value."""
    s = _summary(cell_id, summary_dir).iloc[0]
    f = lambda v: None if pd.isna(pd.to_numeric(v, errors="coerce")) else float(v)
    return f(s["C_ch_nominal"]), f(s["C_dis_nominal"])


#===============================================================================================
# Make a summary file over the TUM cells
#===============================================================================================
#!/usr/bin/env python3
"""
Build one summary CSV per cell (one row per check-up) from the Wildfeuer et al.
VTC5a aging dataset: EFC, capacity/SOH, cycle life (SOH 90% and 80%), mean
temperatures and mean C-rates, plus the overview metadata.

Usage:
    python build_cell_csvs.py DATA_ROOT bawaii_cell_overview_publication.xlsx OUT_DIR
"""
import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

C_NOM = 2.5                   # Ah, nominal capacity (README)
I_THR = 0.02                  # A, |I| below this counts as rest
U_FULL, U_EMPTY = 4.1, 2.7    # V, a "full-range" discharge starts above / ends below these
EOL_LEVELS = (0.9, 0.8)       # SOH levels for cycle life
NEEDED = ["Time", "U", "I", "T1", "Line", "Command", "State"]
TEXT = {"Command"}


# ------------------------------------------------------------------ loading
def _h5_ref():
    try:
        import h5py
        return h5py.Reference
    except ImportError:
        return ()


def _to_array(v, k, f=None):
    v = np.asarray(v)
    if k in TEXT:
        out = []
        for x in np.atleast_1d(v).ravel():
            if isinstance(x, _h5_ref()) and f is not None:      # HDF5 object reference
                x = "".join(chr(c) for c in np.asarray(f[x]).ravel())
            out.append(str(x).strip())
        return np.array(out)
    return np.atleast_1d(v.astype(float)).ravel()


def load_fields(path, fields=NEEDED):
    """Load only the needed fields from Dataset (handles v7 and v7.3/HDF5 files)."""
    try:
        ds = loadmat(path, squeeze_me=True, struct_as_record=False,
                     variable_names=["Dataset"])["Dataset"]
        return {k: _to_array(getattr(ds, k), k) for k in fields if hasattr(ds, k)}
    except NotImplementedError:  # MATLAB v7.3
        import h5py
        with h5py.File(path, "r") as f:
            ds = f["Dataset"]
            return {k: _to_array(ds[k][()], k, f) for k in fields if k in ds}


def to_hours(t):
    """README says Time is in h, but guard against seconds: median sample spacing
    >= 3 min in 'h' would be implausible for a logger, so treat it as seconds."""
    dts = np.diff(t)
    dts = dts[dts > 0]
    if len(dts) and np.median(dts) >= 0.05:
        return t / 3600.0, "s"
    return t, "h"


# ------------------------------------------------------------------ analysis
def throughput_stats(d):
    t, i = d["Time"], d["I"]
    dt = np.clip(np.diff(t), 0, None)          # h; guards against time resets
    i0 = i[:-1]
    ch, dis = i0 > I_THR, i0 < -I_THR
    T1 = d.get("T1", np.full_like(t, np.nan))[:-1]
    return dict(ah_ch=float((i0[ch] * dt[ch]).sum()),
                ah_dis=float((-i0[dis] * dt[dis]).sum()),
                t_ch=float(dt[ch].sum()), t_dis=float(dt[dis].sum()),
                t_tot=float(dt.sum()), t1_int=float(np.nansum(T1 * dt)))


def add_stats(a, b):
    return {k: a[k] + b[k] for k in a}


def checkup_capacity(d):
    """Capacity of the SECOND CCCV discharge of a check-up (definition of SOH_C in
    Wildfeuer et al., Sec. 3.1). A CCCV discharge = contiguous 'Discharge' block (same
    Line) starting above U_FULL and ending below U_EMPTY, plus the directly following
    State==0 discharge block (CV tail).
    Returns (capacity, n_found). If fewer than two such discharges exist the last one
    found is used as a fallback (n_found < 2 flags this); NaN if none."""
    t, i, u = d["Time"], d["I"], d["U"]
    dt = np.r_[0.0, np.clip(np.diff(t), 0, None)]
    dis = d["Command"] == "Discharge"
    change = np.r_[True, (np.diff(d["Line"]) != 0) | (np.diff(dis.astype(int)) != 0)]
    blk = np.cumsum(change) - 1
    w = np.where(dis, -i * dt, 0.0)
    w[change] = 0.0                                   # drop dt spanning the preceding pause
    ah = np.bincount(blk, weights=w)
    first = np.flatnonzero(change)
    last = np.r_[first[1:] - 1, len(t) - 1]
    caps = []
    for b in range(len(first)):
        if dis[first[b]] and u[first[b]] > U_FULL and u[last[b]] < U_EMPTY:
            q = ah[b]
            if b + 1 < len(first) and dis[first[b + 1]] and d["State"][first[b + 1]] == 0:
                q += ah[b + 1]
            caps.append(q)
    if not caps:
        return np.nan, 0
    return float(caps[1] if len(caps) >= 2 else caps[-1]), len(caps)


def cycle_life(efc, soh, eol):
    """EFC at which SOH first drops to <= eol (linear interpolation); NaN if never."""
    for k in range(1, len(soh)):
        if soh[k] <= eol < soh[k - 1]:
            f = (soh[k - 1] - eol) / (soh[k - 1] - soh[k])
            return efc[k - 1] + f * (efc[k] - efc[k - 1])
    return np.nan


# ------------------------------------------------------------------ overview
def read_overview(xlsx):
    df = pd.read_excel(xlsx)
    cols = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}

    def find(prefix):
        for c, n in cols.items():
            if n.startswith(prefix):
                return c
        raise KeyError(f"No overview column starting with '{prefix}'. Have: {list(df.columns)}")

    out = pd.DataFrame({
        "cell_num": df[find("inte")].astype(str).str.extract(r"(\d+)")[0].astype(int),
        "TP": df[find("tp")], "group": df[find("group")],
        "aging_domain": df[find("aging domain")], "operation": df[find("operation")],
        "T_ambient_C": df[find("t in")],
        "DOD_pct": df[find("dod")], "SOC_mean_pct": df[find("soc_mean")],
        "C_ch_nominal": df[find("c_ch")], "C_dis_nominal": df[find("c_dis")],
        "total_EFC_overview": df[find("total efc")],
        "total_Ah_throughput_overview": df[find("total ah")],
        "total_days_overview": df[find("total time")],
    })
    return out.set_index("cell_num")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("overview"); ap.add_argument("out")
    ap.add_argument("--efc-basis", choices=["cycling", "total"], default="cycling",
                    help="EFC used for cycle life: cycling only (paper's CU interval of ~50 EFC) "
                         "or cycling + check-up throughput")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    overview = read_overview(a.overview)

    cu = defaultdict(dict)    # cell -> {cu_no: ([capacities], stats)}
    cyc = defaultdict(dict)   # cell -> {cyc_no: stats}
    units = defaultdict(set)

    for f in sorted(root.rglob("*.mat")):
        m_id = re.search(r"BW-VTC-(\d+)", f.name)
        m_cu = re.search(r"(?:^|[\\/])CU(\d+)(?:_[^\\/]+)?(?:[\\/])", str(f), re.I)
        m_cy = re.search(r"(?:^|[\\/])Cyc(\d+)(?:[\\/])", str(f), re.I)
        if not m_id or not (m_cu or m_cy):
            print(f"skip (cannot parse): {f}")
            continue
        cell = int(m_id.group(1))
        d = load_fields(f)
        if "Time" not in d or "I" not in d:
            print(f"skip (missing Time/I): {f}")
            continue
        d["Time"], unit = to_hours(d["Time"])
        units[cell].add(unit)
        st = throughput_stats(d)
        if m_cu:
            n = int(m_cu.group(1))
            caps, prev = cu[cell].get(n, ([], None))
            cap, n_found = checkup_capacity(d)
            cu[cell][n] = (caps + [(cap, n_found)], st if prev is None else add_stats(prev, st))
        else:
            n = int(m_cy.group(1))
            prev = cyc[cell].get(n)      # several files per interval are summed, not overwritten
            cyc[cell][n] = st if prev is None else add_stats(prev, st)

    for cell in sorted(set(cu) | set(cyc)):
        cyc_n, cu_n = sorted(cyc[cell]), sorted(cu[cell])
        if not cu_n:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cap_by_cu = {n: np.nanmean([c for c, _ in cu[cell][n][0]]) for n in cu_n}
        n_found = {n: max(k for _, k in cu[cell][n][0]) for n in cu_n}
        valid_cu = [n for n in cu_n if not np.isnan(cap_by_cu[n])]

        def c_ref(k):
            """Capacity measured at the last check-up before interval/check-up k (C_new).
            Fallback (flagged): first valid check-up, then nominal capacity."""
            prev = [m for m in valid_cu if m < k]
            if prev:
                return cap_by_cu[prev[-1]], False
            return (cap_by_cu[valid_cu[0]] if valid_cu else C_NOM), True

        # Paper definition: 1 EFC = charge throughput of one full cycle at capacity C_new
        efc_int, efc_cu, fallback = {}, {}, False
        for k in cyc_n:
            c, fb = c_ref(k)
            efc_int[k] = cyc[cell][k]["ah_ch"] / c
            fallback |= fb
        for m in cu_n:
            efc_cu[m] = cu[cell][m][1]["ah_ch"] / c_ref(m)[0]

        rows = []
        for n in cu_n:
            e_cyc = sum(efc_int[k] for k in cyc_n if k <= n)
            e_cu = sum(efc_cu[m] for m in cu_n if m < n)
            rows.append(dict(
                CU=n, capacity_Ah=cap_by_cu[n], n_cccv_discharges=n_found[n],
                EFC_cycling=e_cyc, EFC_checkups=e_cu, EFC_total=e_cyc + e_cu,
                Ah_ch_cycling_cum=sum(cyc[cell][k]["ah_ch"] for k in cyc_n if k <= n),
                Ah_dis_cycling_cum=sum(cyc[cell][k]["ah_dis"] for k in cyc_n if k <= n)))
        df = pd.DataFrame(rows)
        df["EFC"] = df["EFC_cycling"] if a.efc_basis == "cycling" else df["EFC_total"]
        first_cap = df["capacity_Ah"].dropna()
        df["SOH"] = df["capacity_Ah"] / first_cap.iloc[0] if len(first_cap) else np.nan

        allc = list(cyc[cell].values()) or [v[1] for v in cu[cell].values()]
        sm = lambda k: sum(s_[k] for s_ in allc)
        everything = list(cyc[cell].values()) + [v[1] for v in cu[cell].values()]
        ah_ch_all = sum(s_["ah_ch"] for s_ in everything)
        ah_dis_all = sum(s_["ah_dis"] for s_ in everything)

        df["cell_id"] = f"BW-VTC-{cell:03d}"
        df["mean_C_ch_measured"] = sm("ah_ch") / sm("t_ch") / C_NOM if sm("t_ch") > 0 else np.nan
        df["mean_C_dis_measured"] = sm("ah_dis") / sm("t_dis") / C_NOM if sm("t_dis") > 0 else np.nan
        df["mean_T_surface_C"] = sm("t1_int") / sm("t_tot") if sm("t_tot") > 0 else np.nan
        sv = df.dropna(subset=["SOH"])
        for lvl in EOL_LEVELS:
            life = cycle_life(sv["EFC"].to_numpy(), sv["SOH"].to_numpy(), lvl)
            df[f"cycle_life_EFC_soh{int(lvl*100)}"] = life
            df[f"EOL_reached_soh{int(lvl*100)}"] = not np.isnan(life)
        df["c_ref_fallback_used"] = fallback
        df["EFC_all_cycling"] = sum(efc_int.values())
        df["EFC_all_total"] = sum(efc_int.values()) + sum(efc_cu.values())
        df["Ah_ch_all"], df["Ah_dis_all"] = ah_ch_all, ah_dis_all
        df["time_unit_seconds_detected"] = "s" in units[cell]
        if cell in overview.index:
            for k, v in overview.loc[cell].items():
                df[k] = v
            ref_e = overview.loc[cell, "total_EFC_overview"]
            ref_a = overview.loc[cell, "total_Ah_throughput_overview"]
            rc = df["EFC_all_cycling"].iloc[0] / ref_e if ref_e and ref_e > 0 else np.nan
            rt = df["EFC_all_total"].iloc[0] / ref_e if ref_e and ref_e > 0 else np.nan
            df["EFC_ratio_cycling_vs_overview"], df["EFC_ratio_total_vs_overview"] = rc, rt
            # which throughput definition does the overview's Ah-throughput follow?
            df["AhTP_ratio_charge_only"] = ah_ch_all / ref_a if ref_a and ref_a > 0 else np.nan
            df["AhTP_ratio_charge_plus_dis"] = (ah_ch_all + ah_dis_all) / ref_a if ref_a and ref_a > 0 else np.nan
            if not any(0.8 < r < 1.25 for r in (rc, rt)):
                print(f"WARNING BW-VTC-{cell:03d}: EFC cycling {df['EFC_all_cycling'].iloc[0]:.0f} / "
                      f"total {df['EFC_all_total'].iloc[0]:.0f} vs overview {ref_e}")
        df.to_csv(out / f"BW-VTC-{cell:03d}.csv", index=False)
        print(f"BW-VTC-{cell:03d}: {len(df)} CUs, final SOH {df['SOH'].iloc[-1]:.3f}, "
              f"life90 {df['cycle_life_EFC_soh90'].iloc[0]:.0f} EFC")


if __name__ == "__main__":
    main()