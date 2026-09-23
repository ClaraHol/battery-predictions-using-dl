"""
Builds oracle (meta-predictor) features for the Farasis test set: per-cell
temperature, charge C-rate, and discharge C-rate.

The paper doesn't give per-cell C-rate/temperature for the test set, only
per-GROUP averages (Table below, transcribed from the paper's supplementary
material). Combined with number_relationship.xlsx's cell_index -> group_index
mapping (already confirmed present in that file), every cell gets its
group's average as an approximation of its own operational condition.

*** IMPORTANT CAVEAT ***
Seven cells use industrial driving-cycle (mission-profile) protocols with
proprietary raw data that couldn't be shared - these are NOT constant-
current cycled, so a single "charge/discharge C-rate" is a much rougher
summary for them than for the rest of the dataset (their group's average
was almost certainly computed mostly/entirely from the OTHER, constant-
current cells sharing that group, not from these driving cells
themselves). They're flagged via is_driving_cycle_cell rather than
silently treated the same as every other cell:
    PB-b1: 029
    PB-b2: 030, 031
    PC-b1: 111, 112
    PC-b2: 041, 042
"""

import pandas as pd

# Group index -> (T_avg [C], Charge_avg [C-rate], Discharge_avg [C-rate])
# Transcribed from the paper's supplementary group-average table.
GROUP_AVERAGES = {
    1: (25.0, 1.07, 0.99),
    2: (25.0, 0.31, 0.33),
    3: (45.0, 1.10, 0.99),
    4: (25.0, 0.31, 0.33),
    5: (25.0, 1.07, 0.99),
    6: (45.0, 0.33, 0.33),
    7: (25.0, 1.07, 0.99),
    8: (45.0, 1.10, 0.99),
    9: (25.0, 0.31, 0.33),
    10: (45.0, 0.33, 0.33),
    11: (35.0, 0.88, 1.00),
    12: (45.0, 0.92, 1.00),
    13: (25.0, 0.50, 1.00),
    14: (30.0, 1.02, 1.00),
    15: (45.0, 1.17, 1.00),
    16: (32.5, 0.39, 0.56),
    17: (32.5, 0.39, 0.56),
    18: (45.0, 0.94, 1.00),
    19: (45.0, 0.95, 1.00),
    20: (30.0, 0.32, 0.33),
    21: (45.0, 0.91, 1.00),
    22: (45.0, 0.92, 1.00),
    23: (50.0, 0.95, 1.00),
    24: (45.0, 0.94, 1.00),
    25: (50.0, 0.93, 1.00),
    26: (45.0, 0.96, 1.00),
    27: (32.5, 0.50, 0.56),
    28: (45.0, 0.94, 1.00),
    29: (45.0, 0.95, 1.00),
    30: (25.0, 0.36, 0.37),
    31: (50.0, 0.93, 1.00),
    32: (45.0, 0.96, 1.00),
    33: (25.0, 1.42, 0.34),
    34: (25.0, 0.31, 0.34),
    35: (45.0, 0.34, 0.34),
    36: (20.0, 1.19, 1.01),
    37: (25.0, 1.03, 1.01),
}
assert len(GROUP_AVERAGES) == 37, f"expected 37 groups, got {len(GROUP_AVERAGES)}"

# Cell IDs known to use proprietary industrial driving-cycle protocols
# (not constant-current) - their group's average C-rate is a much rougher
# approximation for these specific cells than for the rest of that group.
DRIVING_CYCLE_CELL_IDS = {"029", "030", "031", "111", "112", "041", "042"}


def build_test_set_oracle_features(number_relationship_path: str) -> pd.DataFrame:
    """
    Returns one row per test-set cell: cell_index, group_index, temp_C,
    charge_c_rate, discharge_c_rate, is_driving_cycle_cell.

    cell_index formatting: number_relationship.xlsx's cell_index may be an
    int (e.g. 19) or already zero-padded depending on how it was read -
    this function checks membership in DRIVING_CYCLE_CELL_IDS against both
    the zero-padded 3-digit string form and the plain string form, so it
    matches regardless.
    """
    df = pd.read_excel(number_relationship_path, sheet_name="Sheet1")
    df = df.rename(columns={"cell index": "cell_index", "group index": "group_index"})

    missing_groups = set(df["group_index"].unique()) - set(GROUP_AVERAGES.keys())
    if missing_groups:
        raise ValueError(
            f"number_relationship.xlsx references group(s) {missing_groups} "
            f"not present in GROUP_AVERAGES - the transcribed table may be "
            f"incomplete or the group numbering doesn't match."
        )

    records = []
    for _, row in df.iterrows():
        group = row["group_index"]
        temp_C, charge_c, discharge_c = GROUP_AVERAGES[group]
        cell_id_str = str(row["cell_index"])
        is_driving = (
            cell_id_str in DRIVING_CYCLE_CELL_IDS
            or cell_id_str.zfill(3) in DRIVING_CYCLE_CELL_IDS
        )
        records.append({
            "cell_index": row["cell_index"],
            "group_index": group,
            "temp_C": temp_C,
            "charge_c_rate": charge_c,
            "discharge_c_rate": discharge_c,
            "is_driving_cycle_cell": is_driving,
        })

    result = pd.DataFrame(records)
    n_driving = result["is_driving_cycle_cell"].sum()
    print(f"Built oracle features for {len(result)} cells "
          f"({n_driving} flagged as driving-cycle protocol cells)")
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "number_relationship.xlsx"
    out = build_test_set_oracle_features(path)
    out.to_csv("/work3/claho/battery_datasets/processed/pouch_cells/farasis_oracle_features.csv", index=False)
    print(out.to_string())
