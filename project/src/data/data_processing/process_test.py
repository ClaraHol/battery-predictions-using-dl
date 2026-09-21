from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================



DATA_ROOT = Path(f"/work3/claho/battery_datasets")
INPUT_DIR = DATA_ROOT / "pouch_cells" /"test_data"
BATTERY_TYPE = "PA-b1"



PARQUET_FILE = INPUT_DIR / BATTERY_TYPE / "001_single_parquet_with_soc_and_corrections.parquet"

OUTPUT_DIR = DATA_ROOT/ "processed"/ "pouch_cells" / "rpt_extracted"
OUTPUT_DIR.mkdir(exist_ok=True)

# Nominal capacity of the cell.
# For an 80 Ah cell this is 80.
NOMINAL_CAPACITY_AH = 73.0

TARGET_EFC = 50.0

# Number of points in the voltage curve supplied to the model.
N_CURVE_POINTS = 100

# Expected C/3 current.
EXPECTED_C3_CURRENT = NOMINAL_CAPACITY_AH / 3.0

# Tolerance for identifying the C/3 step.
# 20% is deliberately fairly broad initially.
C3_CURRENT_TOLERANCE = 0.20

# Minimum duration for a candidate discharge step.
MIN_DISCHARGE_DURATION_S = 300


# ============================================================
# LOAD PARQUET
# ============================================================

df = pd.read_parquet(PARQUET_FILE)

print("Loaded:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# STANDARDIZE TYPES
# ============================================================

df["time_stamp"] = pd.to_datetime(df["time_stamp"])

numeric_columns = [
    "current (A)",
    "voltage (V)",
    "step_no",
    "temp",
    "rpt",
    "soc",
    "net_thrgh_ah",
    "total_chg_thrgh_ah_rounded",
    "total_dischg_thrgh_ah_rounded",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

df = df.sort_values("time_stamp").reset_index(drop=True)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nRPT values:")
print(
    df["rpt"]
    .dropna()
    .unique()
)


# ============================================================
# CREATE STEP SUMMARY
# ============================================================

def make_step_summary(df):

    rows = []

    for (rpt, step), x in df.groupby(
        ["rpt", "step_no"],
        sort=True
    ):

        x = x.sort_values("time_stamp")

        if len(x) < 2:
            continue

        t0 = x["time_stamp"].iloc[0]
        t1 = x["time_stamp"].iloc[-1]

        duration_s = (
            t1 - t0
        ).total_seconds()

        current = x["current (A)"].values

        voltage = x["voltage (V)"].values

        # Mean current excluding NaN
        mean_current = np.nanmean(current)

        mean_abs_current = np.nanmean(
            np.abs(current)
        )

        # Estimate charge transferred using
        # trapezoidal integration.
        time_s = (
            x["time_stamp"] - t0
        ).dt.total_seconds().values

        valid = (
            np.isfinite(time_s) &
            np.isfinite(current)
        )

        time_s = time_s[valid]
        current = current[valid]

        if len(time_s) > 1:

            charge_Ah = np.trapezoid(
                np.abs(current),
                time_s
            ) / 3600.0

        else:
            charge_Ah = 0.0

        rows.append({

            "rpt": rpt,
            "step_no": step,

            "n_points": len(x),

            "duration_s": duration_s,
            "duration_min": duration_s / 60,

            "mean_current_A": mean_current,
            "mean_abs_current_A":
                mean_abs_current,

            "charge_Ah":
                charge_Ah,

            "start_voltage_V":
                voltage[0],

            "end_voltage_V":
                voltage[-1],

            "voltage_change_V":
                voltage[-1] - voltage[0],

            "start_soc":
                x["soc"].iloc[0]
                if "soc" in x
                else np.nan,

            "end_soc":
                x["soc"].iloc[-1]
                if "soc" in x
                else np.nan,

        })

    return pd.DataFrame(rows)


step_summary = make_step_summary(df)

step_summary.to_csv(
    OUTPUT_DIR / "rpt_step_summary.csv",
    index=False
)

print(
    "\nSaved step summary to:",
    OUTPUT_DIR / "rpt_step_summary.csv"
)


# ============================================================
# IDENTIFY C/3 DISCHARGE STEP
# ============================================================

def find_c3_discharge(rpt_df):

    candidates = []

    for step, x in rpt_df.groupby(
        "step_no",
        sort=True
    ):

        x = x.sort_values("time_stamp")

        if len(x) < 2:
            continue

        t = (
            x["time_stamp"] -
            x["time_stamp"].iloc[0]
        ).dt.total_seconds().values

        I = x["current (A)"].values

        V = x["voltage (V)"].values

        duration_s = t[-1]

        if duration_s < MIN_DISCHARGE_DURATION_S:
            continue

        # ----------------------------------------------------
        # Determine discharge direction automatically.
        #
        # Here we don't assume whether discharge current is
        # positive or negative.
        #
        # We use the magnitude of current.
        # ----------------------------------------------------

        mean_current = np.nanmean(I)

        mean_abs_current = np.nanmean(
            np.abs(I)
        )

        # How close is this step to C/3?
        relative_error = (
            abs(mean_abs_current -
                EXPECTED_C3_CURRENT)
            / EXPECTED_C3_CURRENT
        )

        # We also want a genuine discharge:
        # voltage should generally decrease.
        voltage_change = (
            V[-1] - V[0]
        )

        # SOC should generally decrease.
        soc_change = np.nan

        if "soc" in x:

            soc_change = (
                x["soc"].iloc[-1]
                - x["soc"].iloc[0]
            )

        candidates.append({

            "step_no": step,

            "duration_s": duration_s,

            "mean_current_A":
                mean_current,

            "mean_abs_current_A":
                mean_abs_current,

            "relative_current_error":
                relative_error,

            "voltage_change_V":
                voltage_change,

            "soc_change":
                soc_change,

            "start_voltage_V":
                V[0],

            "end_voltage_V":
                V[-1],

        })

    candidates = pd.DataFrame(candidates)

    if len(candidates) == 0:
        return None, candidates

    # Prefer steps close to C/3.
    #
    # A decreasing voltage/SOC receives preference.

    candidates["score"] = (
        candidates["relative_current_error"]
    )

    candidates.loc[
        candidates["voltage_change_V"] >= 0,
        "score"
    ] += 10

    if "soc_change" in candidates:
        candidates.loc[
            candidates["soc_change"] >= 0,
            "score"
        ] += 10

    candidates = candidates.sort_values(
        "score"
    )

    selected_step = candidates.iloc[0]["step_no"]

    return selected_step, candidates


# ============================================================
# IDENTIFY C/3 STEP FOR EVERY RPT
# ============================================================

selected_rpt_steps = []

candidate_tables = []

for rpt, rpt_df in df.groupby(
    "rpt",
    sort=True
):

    step, candidates = find_c3_discharge(
        rpt_df
    )

    if step is None:
        print(
            f"WARNING: no C/3 candidate found "
            f"for RPT {rpt}"
        )
        continue

    selected_rpt_steps.append({

        "rpt": rpt,
        "selected_step": step,

        "mean_abs_current_A":
            candidates.iloc[0][
                "mean_abs_current_A"
            ],

        "duration_s":
            candidates.iloc[0][
                "duration_s"
            ],

        "start_voltage_V":
            candidates.iloc[0][
                "start_voltage_V"
            ],

        "end_voltage_V":
            candidates.iloc[0][
                "end_voltage_V"
            ],

    })

    candidates["rpt"] = rpt

    candidate_tables.append(candidates)


selected_steps = pd.DataFrame(
    selected_rpt_steps
)

selected_steps.to_csv(
    OUTPUT_DIR / "selected_c3_steps.csv",
    index=False
)

print("\nSelected C/3 steps:")
print(selected_steps)


# ============================================================
# FUNCTION:
# EXTRACT VOLTAGE CURVE
# ============================================================

def extract_voltage_curve(
    x,
    n_points=200
):

    x = x.sort_values("time_stamp").copy()

    # --------------------------------------------------------
    # Calculate cumulative discharged capacity
    # --------------------------------------------------------

    time_s = (
        x["time_stamp"] -
        x["time_stamp"].iloc[0]
    ).dt.total_seconds().values

    current = x["current (A)"].values

    voltage = x["voltage (V)"].values

    # --------------------------------------------------------
    # Determine current direction.
    #
    # We only need the magnitude for capacity.
    # --------------------------------------------------------

    abs_current = np.abs(current)

    dt = np.diff(time_s)

    dq = (
        0.5 *
        (
            abs_current[:-1] +
            abs_current[1:]
        ) *
        dt /
        3600.0
    )

    q = np.concatenate([
        [0],
        np.cumsum(dq)
    ])

    # --------------------------------------------------------
    # Remove duplicate q values
    # --------------------------------------------------------

    unique_q, idx = np.unique(
        q,
        return_index=True
    )

    voltage = voltage[idx]

    q = unique_q

    # --------------------------------------------------------
    # Normalize discharge capacity
    #
    # 0 = beginning of discharge
    # 1 = end of discharge
    # --------------------------------------------------------

    if q[-1] <= 0:
        return None

    q_norm = q / q[-1]

    # --------------------------------------------------------
    # Remove NaNs
    # --------------------------------------------------------

    valid = (
        np.isfinite(q_norm) &
        np.isfinite(voltage)
    )

    q_norm = q_norm[valid]
    voltage = voltage[valid]

    # --------------------------------------------------------
    # Fixed grid
    # --------------------------------------------------------

    grid = np.linspace(
        0,
        1,
        n_points
    )

    voltage_grid = np.interp(
        grid,
        q_norm,
        voltage
    )

    return {

        "q_total_Ah": q[-1],

        "q_norm": grid,

        "voltage": voltage_grid,

    }


# ============================================================
# EXTRACT ALL RPT VOLTAGE CURVES
# ============================================================

rpt_curves = {}

for _, row in selected_steps.iterrows():

    rpt = row["rpt"]
    step = row["selected_step"]

    x = df[
        (df["rpt"] == rpt) &
        (df["step_no"] == step)
    ].copy()

    curve = extract_voltage_curve(
        x,
        N_CURVE_POINTS
    )

    if curve is None:
        continue

    rpt_curves[rpt] = curve


# ============================================================
# CREATE RPT FEATURE TABLE
# ============================================================

rpt_features = []

for rpt, curve in rpt_curves.items():

    rpt_features.append({

        "rpt": rpt,

        "capacity_Ah":
            curve["q_total_Ah"],

    })

rpt_features = pd.DataFrame(
    rpt_features
)

print("\nExtracted RPT capacities:")
print(rpt_features)


# ============================================================
# EFC
# ============================================================

# IMPORTANT:
#
# We should ideally determine EFC from the aging-cycle
# throughput, NOT simply from the total discharge throughput
# at the END of the RPT, because the RPT itself contains
# discharge capacity.
#
# For now we calculate two quantities:
#
#   1. EFC based on total discharge throughput
#   2. EFC based on the first row of each RPT
#
# This lets us inspect the difference.
# ============================================================

rpt_efc = []

for rpt in sorted(
    df["rpt"].dropna().unique()
):

    x = df[
        df["rpt"] == rpt
    ].sort_values("time_stamp")

    first = x.iloc[0]

    last = x.iloc[-1]

    total_dischg_start = (
        first[
            "total_dischg_thrgh_ah_rounded"
        ]
    )

    total_dischg_end = (
        last[
            "total_dischg_thrgh_ah_rounded"
        ]
    )

    rpt_efc.append({

        "rpt": rpt,

        "total_dischg_start_Ah":
            total_dischg_start,

        "total_dischg_end_Ah":
            total_dischg_end,

        "efc_start":
            total_dischg_start /
            NOMINAL_CAPACITY_AH,

        "efc_end":
            total_dischg_end /
            NOMINAL_CAPACITY_AH,

    })


rpt_efc = pd.DataFrame(rpt_efc)

print("\nRPT EFC diagnostics:")
print(rpt_efc)


# ============================================================
# SAVE EFC DIAGNOSTICS
# ============================================================

rpt_efc.to_csv(
    OUTPUT_DIR / "rpt_efc_diagnostics.csv",
    index=False
)


# ============================================================
# MERGE FEATURES + EFC
# ============================================================

rpt_info = rpt_features.merge(
    rpt_efc,
    on="rpt",
    how="left"
)

print("\nRPT information:")
print(rpt_info)


# ============================================================
# SELECT EFC COORDINATE
# ============================================================

# For the first implementation we use efc_start.
#
# This is preferable to efc_end because the RPT discharge
# itself should not count as aging-cycle throughput.
#
# We can change this after inspecting the paper/dataset.
# ============================================================

rpt_info["efc"] = (
    rpt_info["efc_start"]
)


# ============================================================
# CYCLE-1 / FIRST-RPT FEATURE
# ============================================================

rpt_info = rpt_info.sort_values(
    "efc"
).reset_index(drop=True)

first_rpt = rpt_info.iloc[0]

cycle1_rpt = first_rpt["rpt"]

cycle1_capacity = (
    first_rpt["capacity_Ah"]
)

cycle1_voltage = (
    rpt_curves[cycle1_rpt]["voltage"]
)

print(step_summary[
    step_summary["rpt"] == 0
].to_string(index=False))

# ============================================================
# FIND RPTS AROUND 50 EFC
# ============================================================

before = rpt_info[
    rpt_info["efc"] <= TARGET_EFC
]

after = rpt_info[
    rpt_info["efc"] >= TARGET_EFC
]

if len(before) == 0:

    raise ValueError(
        "No RPT before 50 EFC."
    )

if len(after) == 0:

    raise ValueError(
        "No RPT after 50 EFC."
    )


rpt_a = before.iloc[-1]
rpt_b = after.iloc[0]

rpt_a_id = rpt_a["rpt"]
rpt_b_id = rpt_b["rpt"]

efc_a = rpt_a["efc"]
efc_b = rpt_b["efc"]


# ============================================================
# INTERPOLATION FACTOR
# ============================================================

if abs(efc_b - efc_a) < 1e-12:

    alpha = 0.0

else:

    alpha = (
        TARGET_EFC - efc_a
    ) / (
        efc_b - efc_a
    )


print("\n50 EFC interpolation:")
print(
    f"RPT A = {rpt_a_id}, "
    f"EFC = {efc_a:.4f}"
)

print(
    f"RPT B = {rpt_b_id}, "
    f"EFC = {efc_b:.4f}"
)

print(
    f"alpha = {alpha:.6f}"
)


# ============================================================
# INTERPOLATE CAPACITY
# ============================================================

capacity_a = rpt_a["capacity_Ah"]
capacity_b = rpt_b["capacity_Ah"]

capacity_50 = (
    capacity_a +
    alpha *
    (capacity_b - capacity_a)
)


# ============================================================
# INTERPOLATE VOLTAGE CURVE
# ============================================================

voltage_a = (
    rpt_curves[rpt_a_id]["voltage"]
)

voltage_b = (
    rpt_curves[rpt_b_id]["voltage"]
)

q_grid = (
    rpt_curves[rpt_a_id]["q_norm"]
)

voltage_50 = (
    voltage_a +
    alpha *
    (voltage_b - voltage_a)
)


# ============================================================
# SAVE MODEL FEATURES
# ============================================================

# Columns:
#
# normalized discharge capacity
# cycle-1 voltage
# 50-EFC voltage
#

voltage_features = pd.DataFrame({

    "normalized_discharge_capacity":
        q_grid,

    "voltage_cycle_1_V":
        cycle1_voltage,

    "voltage_50_EFC_V":
        voltage_50,

})


voltage_features.to_csv(
    OUTPUT_DIR /
    "model_voltage_features.csv",
    index=False
)


# ============================================================
# SAVE NUMPY ARRAY
# ============================================================

X_voltage = np.vstack([

    cycle1_voltage,

    voltage_50,

])

np.save(
    OUTPUT_DIR /
    "X_voltage.npy",

    X_voltage
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = pd.DataFrame({

    "quantity": [

        "first_rpt",
        "cycle_1_capacity_Ah",

        "rpt_before_50",
        "efc_before_50",

        "rpt_after_50",
        "efc_after_50",

        "interpolation_alpha",
        "capacity_50_EFC_Ah",

    ],

    "value": [

        cycle1_rpt,
        cycle1_capacity,

        rpt_a_id,
        efc_a,

        rpt_b_id,
        efc_b,

        alpha,
        capacity_50,

    ]

})

metadata.to_csv(
    OUTPUT_DIR /
    "model_metadata.csv",
    index=False
)


print("\n======================================")
print("EXTRACTION COMPLETE")
print("======================================")

print(
    f"Cycle 1 / first RPT: {cycle1_rpt}"
)

print(
    f"Cycle 1 capacity: "
    f"{cycle1_capacity:.4f} Ah"
)

print(
    f"50 EFC capacity: "
    f"{capacity_50:.4f} Ah"
)

print(
    f"50 EFC interpolation: "
    f"{rpt_a_id} → {rpt_b_id}"
)

print(
    f"Voltage feature shape: "
    f"{X_voltage.shape}"
)

print(
    f"\nOutput directory: "
    f"{OUTPUT_DIR.resolve()}"
)