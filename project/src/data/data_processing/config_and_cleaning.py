"""
Shared definitions and cleaning logic for the public battery-aging datasets
(A123-M1A, LG-HG2, Sony-VTC6, Sony-VTC5A, LG-MJ1, Samsung-25R).

This module is format-agnostic: every per-format parser in parsers.py
converts its raw files into ONE standard per-row schema, and everything
here operates only on that standard schema. This keeps the three-step
cleaning identical across all six sources instead of re-implementing it
per format.

STANDARD PER-ROW SCHEMA (one row per timestamped measurement, per cell):
    time_s                  float   seconds since cell test start
    voltage_V               float
    current_A               float   sign convention: positive = charge,
                                     negative = discharge (flip at parse
                                     time if a source uses the opposite)
    temp_C                  float   cell/chamber temperature
    cycle_number             int    0-indexed cycle number
    charge_capacity_Ah      float   cumulative Ah charged so far THIS cycle
    discharge_capacity_Ah   float   cumulative Ah discharged so far THIS cycle

Each parser also returns a `nominal_capacity_Ah` for the cell (from the
lookup table below, since none of the raw files log rated capacity
directly), and, where the source encodes it, a fixed `temp_C_nameplate`
independent of measurement noise.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nominal capacities (Ah),from the supplementary notes in the DL paper
# ---------------------------------------------------------------------------
#
NOMINAL_CAPACITY_AH = {
    "snl_lfp": 1.1,  # A123 ANR26650M1B (LFP)          
    "snl_nmc": 3.0,  # LG INR18650-HG2                  
    "cmu": 3.0,  # Sony/Murata US18650VTC6          
    "tum": 2.5,  # Sony/Murata US18650VTC5A        
    "elt": 3.5,  # LG INR18650-MJ1                  
    "tongji": 2.5,  # Samsung INR18650-25R             
}

# End-of-life threshold, per the paper's supplementary info:
# "cycle life is defined as the number of cycles until capacity ... drops
#  to 90% of its initial value."
EOL_FRACTION_OF_INITIAL = 0.90

# Cleaning thresholds from the paper's methods section
MIN_CYCLE_LIFE = 70
MAX_C_RATE = 10.0

# Rounding grid for the "operational condition" cluster coordinate, taken
# directly from the capsule's data_preparation_k_fold.py:
#   temperature -> nearest 5 degrees C
#   charge/discharge C-rate -> nearest 0.5 C
TEMP_ROUND_TO = 5.0
CRATE_ROUND_TO = 0.5

MIN_CLUSTER_SIZE = (
    6  # clusters with <= this many cells are dropped (paper: "six or fewer")
)


def round_to_grid(value, grid):
    """Round `value` to the nearest multiple of `grid`."""
    return grid * np.round((value + 1e-9) / grid)


def compute_cycle_life(discharge_capacity_by_cycle: pd.Series) -> int:
    """
    discharge_capacity_by_cycle: index = cycle_number, value = that cycle's
    end-of-discharge capacity (Ah), for RPT/checkup cycles ideally, but a
    per-cycle max discharge capacity works too for continuously-cycled data.

    Returns the first cycle number at which capacity drops to
    EOL_FRACTION_OF_INITIAL of the initial capacity, or the last recorded
    cycle number if the cell never crosses that threshold (censored data -
    caller should treat this as a lower bound, not a confirmed cycle life).
    """
    s = discharge_capacity_by_cycle.dropna().sort_index()
    if len(s) == 0:
        raise ValueError("No discharge capacity data to compute cycle life from")
    initial_cap = s.iloc[0]
    threshold = EOL_FRACTION_OF_INITIAL * initial_cap
    below = s[s <= threshold]
    if len(below) == 0:
        return int(s.index[-1])  # censored: cell outlived the data
    return int(below.index[0])


def compute_c_rate(
    current_A: pd.Series, nominal_capacity_Ah: float, positive_only: bool = None
) -> float:
    """
    Mean C-rate over the given current samples.
    positive_only=True  -> only average positive (charge) current
    positive_only=False -> only average negative (discharge) current, returned as a positive C-rate
    positive_only=None  -> average the absolute value of everything passed in
    """
    cur = current_A.copy()
    if positive_only is True:
        cur = cur[cur > 0]
    elif positive_only is False:
        cur = cur[cur < 0]
    if len(cur) == 0:
        return 0.0
    return float(np.abs(cur).mean()) / nominal_capacity_Ah

def compute_cc_c_rate(df: pd.DataFrame, nominal_capacity_Ah: float, direction: str) -> float:
    """
    Estimates the constant-current (CC) C-rate a cell was cycled at, for
    direction 'charge' or 'discharge'.
 
    Replaces the plain mean-of-|current| in compute_c_rate() for per-cycle
    cycling data. That mean was biased LOW: every charge/discharge ends in
    a constant-voltage taper where current decays toward zero and rows are
    logged frequently, so taper samples dragged the average far below the
    real CC level (reported 0.5C for cells whose filenames say 1.5C).
 
    Method: within each cycle, build a TIME-WEIGHTED histogram of current
    magnitude (bins of 2% of nominal capacity, weighted by the time each
    sample covers, gaps clipped to 10 min) and take the heaviest bin - the
    CC plateau, where the cell spends the most time at one current. Then
    take the median of that value across cycles, so a few check-up cycles
    at a different rate don't move the answer.
 
    Returns 0.0 if no usable samples in that direction.
    """
    sign = 1.0 if direction == "charge" else -1.0
    bin_width = 0.02 * nominal_capacity_Ah
    per_cycle = []
    for _, g in df.groupby("cycle_number"):
        t = g["time_s"].to_numpy(dtype=float)
        cur = sign * g["current_A"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        dt = np.clip(np.diff(t, append=t[-1]), 0.0, 600.0)
        mask = cur > 2 * bin_width  # ignore ~zero currents (rest, noise)
        if mask.sum() < 3 or dt[mask].sum() <= 0:
            continue
        c, w = cur[mask], dt[mask]
        bins = np.floor(c / bin_width).astype(int)
        weight_by_bin = pd.Series(w).groupby(bins).sum()
        top_bin = weight_by_bin.idxmax()
        per_cycle.append(float(np.median(c[bins == top_bin])))
    if not per_cycle:
        return 0.0
    return float(np.median(per_cycle)) / nominal_capacity_Ah


def compute_cycle_life_robust(discharge_capacity_by_cycle: pd.Series,
                               nominal_capacity_Ah: float,
                               window: int = 5, n_reference: int = 5,
                               min_fraction_of_nominal: float = 0.5) -> dict:
    """
    Cycle life for NOISY per-cycle capacity data (per-cycle max discharge
    Ah from continuous cycling logs). The simple compute_cycle_life()
    compares every single cycle to iloc[0], which fails two ways on real
    data: a partial first cycle (fresh cell, 0.83 Ah vs ~3.3 Ah full) gives
    a wrong reference, and any single interrupted/partial cycle ends the
    cell's life immediately.
 
    Steps:
      1. Keep only 'full' cycles: capacity >= min_fraction_of_nominal *
         nominal. Drops partial, interrupted and formation cycles.
      2. Reference = median of the first n_reference full cycles.
      3. Smooth with a centered rolling median (window cycles - centered so it adds no lag).
      4. Life = first cycle_number where the smoothed capacity is at or
         below EOL_FRACTION_OF_INITIAL * reference.
 
    Returns {'cycle_life': int, 'censored': bool, 'reference_capacity_Ah': float,
             'n_full_cycles': int}. censored=True means the cell never
    crossed the threshold in the recorded data, so cycle_life is only a
    lower bound (its last recorded full cycle).
    """
    s = discharge_capacity_by_cycle.dropna().sort_index()
    s = s[s >= min_fraction_of_nominal * nominal_capacity_Ah]
    if len(s) == 0:
        raise ValueError("No full-capacity cycles found - cannot compute cycle life")
    reference = float(s.iloc[:n_reference].median())
    smoothed = s.rolling(window, center=True, min_periods=1).median()
    below = smoothed[smoothed <= EOL_FRACTION_OF_INITIAL * reference]
    censored = len(below) == 0
    life = int(s.index[-1]) if censored else int(below.index[0])
    return {"cycle_life": life, "censored": censored,
            "reference_capacity_Ah": reference, "n_full_cycles": int(len(s))}


def calculate_efc(df, q_nom, reset_threshold=0.1, cell_id = None):

    df = df.copy()
    discharge_capacity = df["discharge_capacity_Ah"].astype(float).to_numpy()

    # Difference between consecutive capacity values
    dQ = np.diff(discharge_capacity, prepend=discharge_capacity[0])

    # A significant negative jump indicates that the
    # discharge-capacity counter has reset.
    reset = dQ < -reset_threshold

    # Accumulated capacity before each reset
    offset = np.zeros(len(df))

    accumulated_capacity = 0.0
    if cell_id=="snl":
        df["EFC"] = discharge_capacity/q_nom
    else:
        offset = np.zeros(len(df))
        for i in range(len(df)):
            if reset[i]:
                accumulated_capacity += discharge_capacity[i - 1]

            offset[i] = accumulated_capacity

        # Global discharged capacity
        #df["discharge_capacity_global_Ah"] = discharge_capacity + offset

        # Global discharge-based EFC
        df["EFC"] = (discharge_capacity + offset) / q_nom

    return df


def build_cell_record(
    cell_id,
    source_dataset,
    temp_C_raw,
    charge_c_rate_raw,
    discharge_c_rate_raw,
    cycle_life,
    initial_discharge_capacity_Ah,
):
    """Assemble one row of the per-cell metadata table with rounding applied."""
    return {
        "cell_id": cell_id,
        "source_dataset": source_dataset,
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH[source_dataset],
        "temp_C": float(round_to_grid(temp_C_raw, TEMP_ROUND_TO)),
        "charge_c_rate": float(round_to_grid(charge_c_rate_raw, CRATE_ROUND_TO)),
        "discharge_c_rate": float(round_to_grid(discharge_c_rate_raw, CRATE_ROUND_TO)),
        "cycle_life": int(cycle_life),
        "initial_discharge_capacity_Ah": float(initial_discharge_capacity_Ah),
    }


def find_capacity_checks(cell_df, min_jump_ratio=1.15):
    """
    Identify the 3-cycle capacity-check blocks in one cell.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Must contain: cycle_index, ah_d
    min_jump_ratio : float
        Minimum ratio between the first cycle of a capacity-check block
        and the preceding normal-cycle capacity.

    Returns
    -------
    checks : pd.DataFrame
        One row per capacity-check block.
    df : pd.DataFrame
        Original data with capacity-check information added.
    """

    df = cell_df.sort_values("cycle_index").copy()

    # Difference/ratio between consecutive cycles
    df["ah_d_prev"] = df["ah_d"].shift(1)
    df["ah_d_ratio"] = df["ah_d"] / df["ah_d_prev"]

    # A capacity check starts when ah_d jumps substantially
    # relative to the preceding normal cycling value.
    starts = df.index[
        df["ah_d_ratio"] >= min_jump_ratio
    ]

    check_blocks = []

    for start_idx in starts:
        pos = df.index.get_loc(start_idx)

        # Need 3 consecutive rows/cycles
        if pos + 2 >= len(df):
            continue

        block = df.iloc[pos:pos + 3]

        # Make sure the cycle indices really are consecutive
        if (
            block["cycle_index"].diff().iloc[1:] == 1
        ).all():

            check_blocks.append(block.index.tolist())

    # Remove duplicate/overlapping detections
    unique_blocks = []
    used = set()

    for block in check_blocks:
        if not any(i in used for i in block):
            unique_blocks.append(block)
            used.update(block)

    # Mark capacity-check cycles
    df["capacity_check"] = False
    df["capacity_check_id"] = np.nan

    for check_id, block in enumerate(unique_blocks):
        df.loc[block, "capacity_check"] = True
        df.loc[block, "capacity_check_id"] = check_id

    # One capacity value per capacity-check block
    checks = (
        df[df["capacity_check"]]
        .groupby("capacity_check_id")
        .agg(
            first_cycle=("cycle_index", "min"),
            last_cycle=("cycle_index", "max"),
            capacity_Ah=("ah_d", "mean"),
        )
        .reset_index()
    )

    return checks, df


def apply_three_step_cleaning(
    cells_df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Apply, in order:
      1. C-rate filter: drop cells where |charge_c_rate| > 10 or |discharge_c_rate| > 10
      2. Cycle-life filter: drop cells where cycle_life < 70
      3. Operational-condition cluster filter: group by (temp_C, charge_c_rate,
         discharge_c_rate) and drop any cluster with <= 6 cells.

    Adds an `operational_cluster` column (tuple) to the surviving rows.
    """
    df = cells_df.copy()
    n0 = len(df)

    # Step 1: C-rate
    mask_crate = (df["charge_c_rate"].abs() <= MAX_C_RATE) & (
        df["discharge_c_rate"].abs() <= MAX_C_RATE
    )
    df = df[mask_crate]
    n1 = len(df)

    # Step 2: cycle life
    df = df[df["cycle_life"] >= MIN_CYCLE_LIFE]
    n2 = len(df)

    # Step 3: cluster size
    df["operational_cluster"] = list(
        zip(df["temp_C"], df["charge_c_rate"], df["discharge_c_rate"])
    )
    cluster_sizes = df["operational_cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes <= MIN_CLUSTER_SIZE].index
    df = df[~df["operational_cluster"].isin(small_clusters)]
    n3 = len(df)

    if verbose:
        print(
            f"Cleaning: {n0} cells -> {n1} after C-rate filter (-{n0 - n1}) "
            f"-> {n2} after cycle-life filter (-{n1 - n2}) "
            f"-> {n3} after cluster-size filter (-{n2 - n3})"
        )
        print(f"Surviving operational clusters: {df['operational_cluster'].nunique()}")

    return df.reset_index(drop=True)
