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
# Nominal capacities (Ah), from manufacturer datasheets - verified via web
# search against Murata/Samsung/A123/LG product data, Sep 2026. If a cell's
# measured max discharge capacity differs from this by more than ~10%,
# re-check the datasheet before trusting downstream C-rate values.
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY_AH = {
    "snl_a": 2.5,     # A123 ANR26650M1B (LFP)          - confirmed
    "snl_b": 3.0,     # LG INR18650-HG2                  - confirmed
    "cmu": 3.12,      # Sony/Murata US18650VTC6          - confirmed (NOT 3.0 - that's rated minimum, not nominal)
    "tum": 2.6,       # Sony/Murata US18650VTC5A         - confirmed
    "elt": 3.5,       # LG INR18650-MJ1                  - commonly cited figure, NOT independently confirmed.
                       #    Sanity-check against real data: max observed discharge_capacity_Ah
                       #    at a low-rate cycle should land close to 3.5 Ah.
    "tongji": 2.5,    # Samsung INR18650-25R             - confirmed
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

MIN_CLUSTER_SIZE = 6  # clusters with <= this many cells are dropped (paper: "six or fewer")


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


def compute_c_rate(current_A: pd.Series, nominal_capacity_Ah: float, positive_only: bool = None) -> float:
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


def build_cell_record(cell_id, source_dataset, temp_C_raw, charge_c_rate_raw,
                       discharge_c_rate_raw, cycle_life, initial_discharge_capacity_Ah):
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


def apply_three_step_cleaning(cells_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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
    mask_crate = (df["charge_c_rate"].abs() <= MAX_C_RATE) & (df["discharge_c_rate"].abs() <= MAX_C_RATE)
    df = df[mask_crate]
    n1 = len(df)

    # Step 2: cycle life
    df = df[df["cycle_life"] >= MIN_CYCLE_LIFE]
    n2 = len(df)

    # Step 3: cluster size
    df["operational_cluster"] = list(zip(df["temp_C"], df["charge_c_rate"], df["discharge_c_rate"]))
    cluster_sizes = df["operational_cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes <= MIN_CLUSTER_SIZE].index
    df = df[~df["operational_cluster"].isin(small_clusters)]
    n3 = len(df)

    if verbose:
        print(f"Cleaning: {n0} cells -> {n1} after C-rate filter (-{n0-n1}) "
              f"-> {n2} after cycle-life filter (-{n1-n2}) "
              f"-> {n3} after cluster-size filter (-{n2-n3})")
        print(f"Surviving operational clusters: {df['operational_cluster'].nunique()}")

    return df.reset_index(drop=True)
