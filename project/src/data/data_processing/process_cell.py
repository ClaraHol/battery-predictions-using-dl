"""
Takes one cell's standardized per-row cycling data (from any parser in
parsers.py) and produces:
  1. a metadata dict ready for config_and_cleaning.build_cell_record()
  2. the raw efc0 and efc50 cycle snapshots, for later feature engineering
     (kept separate from cleaning/metadata so re-running feature extraction
     later doesn't require re-parsing every raw file)
"""

import pandas as pd
from project.src.data.data_processing.config_and_cleaning import build_cell_record, compute_cycle_life, compute_c_rate


def process_cell(df: pd.DataFrame, cell_id: str, source_dataset: str, efc_target: int = 50):
    """
    df: standard-schema DataFrame for one cell (see parsers.py STANDARD_COLUMNS)
    Returns (metadata_dict, efc0_df, efc_target_df)
    efc0_df / efc_target_df are None if that cycle isn't present in the data
    (e.g. the cell died before reaching cycle 50 - still worth keeping the
    metadata row, just without an efc50 snapshot).
    """
    from project.src.data.data_processing.config_and_cleaning import NOMINAL_CAPACITY_AH

    nominal_cap = NOMINAL_CAPACITY_AH[source_dataset]

    # --- cycle life: first cycle where per-cycle discharge capacity drops
    # to 90% of the initial cycle's discharge capacity ---
    per_cycle_discharge_cap = df.groupby("cycle_number")["discharge_capacity_Ah"].max()
    cycle_life = compute_cycle_life(per_cycle_discharge_cap)

    # --- charge/discharge C-rate: mean over the whole cell's history.
    # (The protocol is fixed per cell across its life for these datasets,
    # so this is equivalent to computing it from efc0 alone, but averaging
    # over everything is more robust to a noisy first cycle.) ---
    charge_c = compute_c_rate(df["current_A"], nominal_cap, positive_only=True)
    discharge_c = compute_c_rate(df["current_A"], nominal_cap, positive_only=False)

    # --- temperature: mean over the whole cell's history ---
    temp_C = float(df["temp_C"].mean())

    metadata = build_cell_record(
        cell_id=cell_id,
        source_dataset=source_dataset,
        temp_C_raw=temp_C,
        charge_c_rate_raw=charge_c,
        discharge_c_rate_raw=discharge_c,
        cycle_life=cycle_life,
        initial_discharge_capacity_Ah=per_cycle_discharge_cap.iloc[0],
    )

    efc0_df = df[df["cycle_number"] == df["cycle_number"].min()].copy()
    efc0_df["cell_id"] = cell_id

    target_cycles_present = df["cycle_number"].unique()
    if efc_target in target_cycles_present:
        efc_target_df = df[df["cycle_number"] == efc_target].copy()
        efc_target_df["cell_id"] = cell_id
    else:
        efc_target_df = None

    return metadata, efc0_df, efc_target_df
