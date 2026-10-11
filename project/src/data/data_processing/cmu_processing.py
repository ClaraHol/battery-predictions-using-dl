import numpy as np
import pandas as pd

def reconstruct_cmu_mission_cycles(df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct monotonically increasing VTC6 mission-cycle numbers.

    CMU's raw cycleNumber cannot be used globally because the original
    tester files were concatenated and the tester cycle counter resets.

    A recurring eVTOL mission cycle contains:
        Ns 0 -> 1 -> 3 -> 4 -> 5 -> 6 -> 7

    We identify a new physical mission cycle from the beginning of the
    charge sequence (Ns == 0), but only retain blocks that subsequently
    contain the characteristic mission-discharge state Ns == 4.
    """
    df = df.sort_values("time_s")
    ns = df["Ns"]

    # Beginning of every new Ns=0 block.
    start = ns.eq(0) & ~ns.shift(fill_value=-1).eq(0)
    block = start.cumsum() - 1

    # Determine which blocks actually contain the mission-discharge phase.
    has_mission = df.groupby(block)["Ns"].transform(lambda x: (x == 4).any())

    # Give mission blocks consecutive global cycle numbers.
    mission_blocks = (
        pd.DataFrame({
            "block": block,
            "has_mission": has_mission
        })
        .drop_duplicates("block")
    )
    valid_blocks = mission_blocks.loc[mission_blocks["has_mission"], "block"]

    block_to_cycle = {
        b: i + 1
        for i, b in enumerate(valid_blocks)
    }

    cycle = block.map(block_to_cycle)
    return cycle.astype("Int64")


def label_cmu_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time_s", kind="stable").reset_index(drop=True).copy()
    ns = df["Ns"]
    start = ns.eq(0) & ~ns.shift(fill_value=-1).eq(0)
    df["block"] = start.cumsum()

    g = df["block"]
    has4 = ns.eq(4).groupby(g).transform("max")
    has_cap = (ns.eq(2) | ns.eq(8)).groupby(g).transform("max")  # capacity-test marker
    df["is_capacity"] = has_cap
    df["is_mission"] = has4 & ~has_cap

    blk = df.drop_duplicates("block").set_index("block")["is_mission"]
    mission_count = blk.astype(int).cumsum()          # capacity tests inherit prior count
    df["mission_cycle"] = df["block"].map(mission_count).astype(int)
    return df

def per_block_table(df, max_dt_s=30.0, discharge_sign=1):
    "EFC should already be calculated"
    t = df["time_s"].to_numpy(float)
    I_dis = np.clip(df["current_A"].to_numpy(float) * discharge_sign, 0, None)
    blk = df["block"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    same = np.r_[False, blk[1:] == blk[:-1]]
    dt = np.where(same & (dt <= max_dt_s), dt, 0.0)
    dQ = np.r_[0, 0.5 * (I_dis[1:] + I_dis[:-1])] * dt / 3600.0

    out = (df.assign(dQ=dQ).groupby("block").agg(
        t_start=("time_s", "min"), t_end=("time_s", "max"),
        is_mission=("is_mission", "first"),
        is_capacity=("is_capacity", "first"),
        mission_cycle=("mission_cycle", "first"),
        discharge_Ah=("dQ", "sum"),
        efc_start=("EFC", "min"),
        efc_end=("EFC", "max"),
    ).reset_index())
    return out


def cmu_cycle_life_efc(df, nominal_Ah=3.0, eol=0.9, min_frac=0.8,
                       outlier_frac=1.2, n_ref=1):
    blocks = per_block_table(label_cmu_blocks(df))

    caps = blocks[blocks.is_capacity &
                  (blocks.discharge_Ah >= min_frac * nominal_Ah) &
                  (blocks.discharge_Ah <= outlier_frac * nominal_Ah)].copy()
    caps = (caps.groupby("mission_cycle", as_index=False)
                .agg(capacity_Ah=("discharge_Ah", "max"),
                     efc=("efc_start", "min")))   # EFC accumulated before the test
    caps = caps.sort_values("efc").reset_index(drop=True)

    ref = caps["capacity_Ah"].iloc[:n_ref].median()
    thresh = eol * ref
    below = caps[caps["capacity_Ah"] <= thresh]

    if below.empty:
        return dict(cycle_life=float(blocks.efc_end.iloc[-1]), censored=True, 
                    n_full_cycles = int(blocks.mission_cycle.max()),
                    reference_Ah=ref, caps=caps)

    i = below.index[0]
    if i == 0:
        life = caps.loc[i, "efc"]
    else:
        c0, c1 = caps.loc[i - 1], caps.loc[i]
        frac = (c0.capacity_Ah - thresh) / (c0.capacity_Ah - c1.capacity_Ah)
        life = c0.efc + frac * (c1.efc - c0.efc)
    return dict(cycle_life=float(life), censored=False, n_full_cycles = int(blocks.mission_cycle.max()),
                reference_Ah=ref, caps=caps)

def cmu_cycle_life(df, nominal_Ah=3.0, eol=0.9, min_frac=0.8,
                   outlier_frac=1.2, n_ref=3):
    blocks = per_block_table(label_cmu_blocks(df))
    
    caps = blocks[blocks.is_capacity &
                (blocks.discharge_Ah >= min_frac * nominal_Ah) &
                (blocks.discharge_Ah <= outlier_frac * nominal_Ah)].copy()

    # collapse back-to-back tests (VAH25, VAH28) that share a mission_cycle
    caps = (caps.groupby("mission_cycle", as_index=False)
                .agg(capacity_Ah=("discharge_Ah", "max"), t_start=("t_start", "min")))

    ref = caps["capacity_Ah"].iloc[:n_ref].median()
    thresh = eol * ref
    below = caps[caps["capacity_Ah"] <= thresh]

    if below.empty:
        return dict(cycle_life=None, censored=True,
                    reference_Ah=ref, caps=caps, n_full_cycles=int(blocks.mission_cycle.max()))

    i = below.index[0]
    if i == caps.index[0]:
        life = int(caps.loc[i, "mission_cycle"])
    else:                                              # interpolate between bracketing tests
        c0, c1 = caps.loc[i - 1], caps.loc[i]
        frac = (c0.capacity_Ah - thresh) / (c0.capacity_Ah - c1.capacity_Ah)
        life = int(round(c0.mission_cycle + frac * (c1.mission_cycle - c0.mission_cycle)))
    return dict(cycle_life=life, censored=False, reference_Ah=ref, caps=caps, n_full_cycles=int(blocks.mission_cycle.max()))