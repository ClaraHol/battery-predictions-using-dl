#!/usr/bin/env python3
"""
Build one CSV per cell (EFC, capacity/SOH, cycle life, mean temps, mean C-rates)
from the Wildfeuer et al. VTC5a aging dataset.

Usage:
    python build_cell_csvs.py DATA_ROOT bawaii_cell_overview_publication.xlsx OUT_DIR [--eol 0.8]

DATA_ROOT contains CU_Calendar, CU_Cyclic, CU_Dynamic, CYC_Cyclic, CYC_Dynamic.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

C_NOM = 2.5      # Ah, nominal capacity (README)
I_THR = 0.02     # A, |I| below this counts as rest
U_FULL, U_EMPTY = 4.1, 2.7   # V, a "full-range" discharge starts above / ends below these
NEEDED = ["Time", "U", "I", "T1", "Line", "Command", "State"]
TEXT = {"Command"}



def _to_array(v, k, f=None):
    v = np.asarray(v)
    if k in TEXT:
        out = []
        for x in np.atleast_1d(v).ravel():
            if isinstance(x, h5py_ref()) and f is not None:        # HDF5 object reference
                x = "".join(chr(c) for c in np.asarray(f[x]).ravel())
            out.append(str(x).strip())
        return np.array(out)
    return np.atleast_1d(v.astype(float)).ravel()


def h5py_ref():
    try:
        import h5py
        return h5py.Reference
    except ImportError:
        return ()


# ------------------------------------------------------------------ loading
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


# ------------------------------------------------------------------ analysis
def throughput_stats(d):
    """Integrate current over time -> Ah, time spent charging/discharging, T1 integral."""
    t, i = d["Time"], d["I"]
    dt = np.clip(np.diff(t), 0, None)          # h; guards against time resets
    i0 = i[:-1]
    ch, dis = i0 > I_THR, i0 < -I_THR
    T1 = d.get("T1", np.full_like(t, np.nan))[:-1]
    return dict(
        ah_ch=float((i0[ch] * dt[ch]).sum()),
        ah_dis=float((-i0[dis] * dt[dis]).sum()),
        t_ch=float(dt[ch].sum()),
        t_dis=float(dt[dis].sum()),
        t_tot=float(dt.sum()),
        t1_int=float(np.nansum(T1 * dt)),
    )


def checkup_capacity(d):
    """Capacity at the 1C reference discharge of the check-up.

    A reference discharge = contiguous Command=='Discharge' block (same program Line)
    that starts above U_FULL and ends below U_EMPTY (CC phase), plus the directly
    following discharge block with State==0 (CV tail), if present.
    Returns the mean over all such discharges in the file (the program repeats it),
    or NaN if none is found."""
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
            if b + 1 < len(first) and dis[first[b + 1]] and d.get("State", [1])[first[b + 1]] == 0:
                q += ah[b + 1]                        # CV tail
            caps.append(q)
    return float(np.mean(caps)) if caps else np.nan


def cycle_life(efc, soh, eol):
    """EFC at which SOH first drops to `eol` (linear interpolation). NaN if never."""
    for k in range(1, len(soh)):
        if soh[k] <= eol < soh[k - 1]:
            f = (soh[k - 1] - eol) / (soh[k - 1] - soh[k])
            return efc[k - 1] + f * (efc[k] - efc[k - 1])
    return np.nan if (len(soh) == 0 or soh[0] > eol) else efc[0]


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
        "total_days_overview": df[find("total time")],
    })
    return out.set_index("cell_num")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("overview"); ap.add_argument("out")
    ap.add_argument("--eol", type=float, default=0.8, help="SOH defining end of life")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    overview = read_overview(a.overview)

    cu = defaultdict(dict)    # cell -> {cu_no: (capacity, stats)}
    cyc = defaultdict(dict)   # cell -> {cyc_no: stats}
   
    print("Processing cells")
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
        if m_cu:
            cu[cell][int(m_cu.group(1))] = (checkup_capacity(d), throughput_stats(d))
        else:
            cyc[cell][int(m_cy.group(1))] = throughput_stats(d)
    print("Combining cells")

    for cell in sorted(set(cu) | set(cyc)):
        # --- cumulative cycling throughput at each check-up (Cyc n happens before CU n)
        cyc_n = sorted(cyc[cell])
        cu_n = sorted(cu[cell])
        rows, cum_cyc, cum_cu = [], 0.0, 0.0
        for n in cu_n:
            cum_cyc = sum(cyc[cell][k]["ah_ch"] + cyc[cell][k]["ah_dis"] for k in cyc_n if k <= n)
            cum_cu = sum(cu[cell][k][1]["ah_ch"] + cu[cell][k][1]["ah_dis"] for k in cu_n if k < n)
            cap = cu[cell][n][0]
            rows.append(dict(CU=n,
                             EFC_cycling=cum_cyc / (2 * C_NOM),
                             EFC_total=(cum_cyc + cum_cu) / (2 * C_NOM),
                             capacity_Ah=cap))
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["SOH"] = df["capacity_Ah"] / df["capacity_Ah"].iloc[0]

        # --- cell-level constants
        allc = list(cyc[cell].values()) or [v[1] for v in cu[cell].values()]
        sm = lambda k: sum(s[k] for s in allc)
        life = cycle_life(df["EFC_total"].to_numpy(), df["SOH"].to_numpy(), a.eol)
        df["cell_id"] = f"BW-VTC-{cell:03d}"
        df["mean_C_ch_measured"] = sm("ah_ch") / sm("t_ch") / C_NOM if sm("t_ch") > 0 else np.nan
        df["mean_C_dis_measured"] = sm("ah_dis") / sm("t_dis") / C_NOM if sm("t_dis") > 0 else np.nan
        df["mean_T_surface_C"] = sm("t1_int") / sm("t_tot") if sm("t_tot") > 0 else np.nan
        df["cycle_life_EFC"] = life
        df["EOL_reached"] = not np.isnan(life)
        df["EOL_SOH"] = a.eol
        if cell in overview.index:
            for k, v in overview.loc[cell].items():
                df[k] = v
        df.to_csv(out / f"BW-VTC-{cell:03d}.csv", index=False)
        print(f"BW-VTC-{cell:03d}: {len(df)} check-ups, cycle life = {life}")


if __name__ == "__main__":
    main()