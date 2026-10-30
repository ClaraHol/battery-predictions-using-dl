
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

def discharge_runs(
    df,
    discharge_sign=-1,
    i_thresh_A=0.05,
    max_dt_s=60.0,
    v_floor=1.0
):
    t = df["time_s"].to_numpy(dtype=float)
    I = df["current_A"].to_numpy(dtype=float) * discharge_sign
    V = df["voltage_V"].to_numpy(dtype=float)

    # Keep the original row indices.
    rows = np.arange(len(df))

    # Identify valid discharge samples without deleting rows.
    dis = (
        (I > i_thresh_A)
        & (V > v_floor)
        & np.isfinite(t)
        & np.isfinite(I)
        & np.isfinite(V)
    )

    # Actual time difference between consecutive original samples.
    dt = np.diff(t, prepend=t[0])

    # Force a new run at:
    # 1. Current-state transitions
    # 2. Large time gaps
    # 3. Non-increasing timestamps
    new_run = np.r_[
        True,
        (dis[1:] != dis[:-1])
        | (np.diff(t) > max_dt_s)
        | (np.diff(t) <= 0)
        | ~np.isfinite(np.diff(t))
    ]

    run = np.cumsum(new_run)

    # Don't integrate across run boundaries.
    same_run = ~new_run
    valid_dt = same_run & dis

    dQ = np.where(valid_dt, I * dt / 3600.0, 0.0)

    tmp = pd.DataFrame({
        "run": run,
        "dis": dis,
        "dQ": dQ,
        "V": V,
        "I": I,
        "t": t,
        "row": rows
    })

    runs = (
        tmp[tmp.dis]
        .groupby("run", sort=True)
        .agg(
            Q_Ah=("dQ", "sum"),
            v_start=("V", "first"),
            v_end=("V", "last"),
            mean_I_A=("I", "mean"),
            t_start=("t", "first"),
            t_end=("t", "last"),
            discharge_start_idx=("row", "first"),
            discharge_end_idx=("row", "last")
        )
        .reset_index(drop=True)
    )

    runs["duration_s"] = runs["t_end"] - runs["t_start"]

    cum = runs["Q_Ah"].cumsum()
    runs["efc_start"] = cum - runs["Q_Ah"]
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
    
    if pick not in ("first", "max", "median"):
        raise ValueError(f"Invalid pick: {pick}")

    rpt = rpt.copy()
    rpt["rpt_id"] = g.to_numpy() - 1

    records = []

    for rpt_id, group in rpt.groupby("rpt_id"):

        if pick == "first":
            selected = group.iloc[0]
            capacity = float(selected["Q_Ah"])

        elif pick == "max":
            selected = group.loc[group["Q_Ah"].idxmax()]
            capacity = float(selected["Q_Ah"])

        else:
            capacity = float(group["Q_Ah"].median())
            selected = group.iloc[np.argmin(abs(group["Q_Ah"] - capacity))]

        records.append({
            "rpt_id": int(rpt_id),
            "capacity_Ah": capacity,
            "efc": float(group["efc_start"].min()),
            "discharge_start_idx": int(selected["discharge_start_idx"]),
            "discharge_end_idx": int(selected["discharge_end_idx"]),
            "n_runs": len(group)
        })

    caps = (pd.DataFrame(records).sort_values("efc").reset_index(drop=True))

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