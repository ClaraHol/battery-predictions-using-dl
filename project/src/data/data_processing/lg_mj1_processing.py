
import numpy as np
import pandas as pd

RESET_THRESH_AH = 0.1


def add_efc(df, q_nom, reset_threshold=RESET_THRESH_AH):
    q = df["discharge_capacity_Ah"].ffill().fillna(0).to_numpy(float)
    prev = np.r_[q[0], q[:-1]]
    offset = np.cumsum(np.where(q - prev < -reset_threshold, prev, 0.0))
    out = df.copy()
    out["EFC"] = (q + offset) / q_nom
    return out

def discharge_runs(df, discharge_sign=-1, i_thresh_A=0.05, max_dt_s=60.0, v_floor=1.0):
    d = df[df["voltage_V"] > v_floor].reset_index(drop=True)      # drop the 0.02 V glitch rows; NO sort
    t = d["time_s"].to_numpy(float)
    I = d["current_A"].to_numpy(float) * discharge_sign
    V = d["voltage_V"].to_numpy(float)

    dis = I > i_thresh_A
    run = np.cumsum(np.r_[True, dis[1:] != dis[:-1]])
    dt = np.diff(t, prepend=t[0])
    dt = np.where((np.r_[False, run[1:] == run[:-1]]) & (dt > 0) & (dt <= max_dt_s), dt, 0.0)
    dQ = np.where(dis, I, 0.0) * dt / 3600.0

    tmp = pd.DataFrame({"run": run, "dis": dis, "dQ": dQ, "V": V, "I": I, "t": t})
    runs = (tmp[tmp.dis].groupby("run")
            .agg(Q_Ah=("dQ", "sum"), v_start=("V", "first"), v_end=("V", "last"),
                 mean_I_A=("I", "mean"), t_start=("t", "first"), t_end=("t", "last"))
            .reset_index(drop=True))
    cum = runs["Q_Ah"].cumsum()
    runs["efc_start"] = (cum - runs["Q_Ah"])    # divide by q_nom below
    runs["efc_end"] = cum

    return runs

def lgmj1_cycle_life(runs, nominal_Ah, v_end_max=2.6, min_frac=0.8,
                     cluster_efc_gap=20.0, eol=0.9, confirm=1,
                     pick="first"):
    rpt = runs[(runs.v_end <= v_end_max) &
               (runs.Q_Ah >= min_frac * nominal_Ah)].sort_values("efc_start")

    # new RPT whenever the EFC jumps by more than cluster_efc_gap since the previous RPT discharge
    new = rpt["efc_start"].diff().fillna(np.inf) > cluster_efc_gap
    g = new.cumsum()
    agg = {"first": "first", "max": "max", "median": "median"}[pick]
    caps = (rpt.groupby(g)
               .agg(capacity_Ah=("Q_Ah", agg), efc=("efc_start", "min"),
                    n_runs=("Q_Ah", "size"))
               .reset_index(drop=True))
    if caps.empty:
        raise ValueError("No RPT runs found - check v_end_max / min_frac")

    ref = float(caps.capacity_Ah.iloc[0]); thr = eol * ref
    ok = (caps.capacity_Ah <= thr).rolling(confirm).sum().eq(confirm)
    if not ok.any():
        return dict(cycle_life=float(runs.efc_end.iloc[-1]), censored=True,
                    n_full_cycles=len(caps), reference_Ah=ref, caps=caps)
    i = int(ok.idxmax()) - confirm + 1
    if i == 0:
        life = float(caps.efc.iloc[0])
    else:
        c0, c1 = caps.loc[i - 1], caps.loc[i]
        life = float(c0.efc + (c0.capacity_Ah - thr) / (c0.capacity_Ah - c1.capacity_Ah) * (c1.efc - c0.efc))
    return dict(cycle_life=life, censored=False, n_full_cycles=len(caps),
                reference_Ah=ref, caps=caps)

def lgmj1_cycle_life_from_df(df, nominal_Ah, **kw):
    runs = discharge_runs(df)
    runs[["efc_start", "efc_end"]] /= nominal_Ah
    return lgmj1_cycle_life(runs, nominal_Ah, **kw)
"""
def lgmj1_cycle_life(df, nominal_Ah, eol=0.9, min_frac=0.8,
                     v_rpt_max=None, confirm=1):
    d = add_efc(df.sort_values("time_s", kind="stable"), nominal_Ah)

    cyc = (d.groupby("cycle_number")
             .agg(dis_Ah=("discharge_capacity_Ah", "max"),
                  v_min=("voltage_V", "min"),
                  efc_start=("EFC", "min"),
                  efc_end=("EFC", "max"))
             .reset_index())
    
    print(cyc.sort_values("v_min").head(20))
    print(cyc.v_min.describe())
    if v_rpt_max is None:                       # heuristic: look at cyc.v_min first
        v_rpt_max = cyc["v_min"].min() + 0.2

    is_rpt = (cyc["dis_Ah"] >= min_frac * nominal_Ah * (-1)) & (cyc["v_min"] <= v_rpt_max)
    key = (~is_rpt).cumsum()                    # consecutive RPT cycles share a key
    caps = (cyc[is_rpt].groupby(key[is_rpt])
              .agg(capacity_Ah=("dis_Ah", "max"),
                   efc=("efc_start", "min"),
                   cycle=("cycle_number", "min"))
              .reset_index(drop=True))
    if caps.empty:
        raise ValueError("No RPT discharges found - check v_rpt_max / min_frac")

    ref = float(caps["capacity_Ah"].iloc[0])
    thr = eol * ref
    below = caps["capacity_Ah"] <= thr
    ok = below.rolling(confirm).sum().eq(confirm)

    if not ok.any():
        return dict(cycle_life=float(cyc["efc_end"].iloc[-1]), censored=True,
                    n_full_cycles=len(caps), reference_Ah=ref, caps=caps)

    i = int(ok.idxmax()) - confirm + 1          # first of the confirmed-below tests
    if i == 0:
        life = float(caps.loc[0, "efc"])
    else:
        c0, c1 = caps.loc[i - 1], caps.loc[i]
        frac = (c0.capacity_Ah - thr) / (c0.capacity_Ah - c1.capacity_Ah)
        life = float(c0.efc + frac * (c1.efc - c0.efc))
    return dict(cycle_life=life, censored=False, n_full_cycles=len(caps),
                reference_Ah=ref, caps=caps)
"""