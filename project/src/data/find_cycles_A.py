import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Q_NOM = 3.7


# ============================================================
# 1. Calculate global discharge-based EFC
# ============================================================

def calculate_efc(df, q_nom=Q_NOM, reset_threshold=0.1):

    df = df.copy()

    discharge_capacity = (
        df["Discharge_Capacity_Ah"]
        .astype(float)
        .to_numpy()
    )

    # Difference between consecutive capacity values
    dQ = np.diff(
        discharge_capacity,
        prepend=discharge_capacity[0]
    )

    # A significant negative jump indicates that the
    # discharge-capacity counter has reset.
    reset = dQ < -reset_threshold

    # Accumulated capacity before each reset
    offset = np.zeros(len(df))

    accumulated_capacity = 0.0

    for i in range(len(df)):

        if reset[i]:
            accumulated_capacity += (
                discharge_capacity[i - 1]
            )

        offset[i] = accumulated_capacity

    # Global discharged capacity
    df["Discharge_Capacity_Global_Ah"] = (
        discharge_capacity + offset
    )

    # Global discharge-based EFC
    df["EFC"] = (
        df["Discharge_Capacity_Global_Ah"]
        / q_nom
    )

    return df


# ============================================================
# 2. Find individual full charge events
# ============================================================

def find_charge_events(
    df,
    current_threshold=3,
    current_deviation_fraction=0.05,
    min_deviation_duration_s=20,
    min_duration_h=0.1,
    max_duration_h=3,
):

    current = df["Current_A"].to_numpy()
    time = df["Total_Time_Seconds"].to_numpy()

    # --------------------------------------------------------
    # Identify regions where the battery is charging.
    #
    # Positive current = charging
    # --------------------------------------------------------

    is_charge = current > current_threshold

    starts = np.where(
        is_charge &
        ~np.r_[False, is_charge[:-1]]
    )[0]

    events = []

    for start in starts:

        # ----------------------------------------------------
        # Find the initial continuous positive-current region
        # ----------------------------------------------------

        i = start

        while (
            i < len(df)
            and current[i] > current_threshold
        ):
            i += 1

        preliminary_end = i - 1

        if preliminary_end <= start:
            continue

        # ----------------------------------------------------
        # Estimate characteristic charge current
        #
        # Use the middle 80% to avoid the current transitions
        # at the beginning and end affecting the reference.
        # ----------------------------------------------------

        segment = current[
            start:preliminary_end + 1
        ]

        n = len(segment)

        lo = int(0.10 * n)
        hi = int(0.90 * n)

        if hi <= lo:
            continue

        characteristic_current = np.median(
            segment[lo:hi]
        )

        # ----------------------------------------------------
        # Calculate relative deviation from the characteristic
        # charge current.
        # ----------------------------------------------------

        deviation = (
            np.abs(
                current[start:preliminary_end + 1]
                - characteristic_current
            )
            / abs(characteristic_current)
        )

        # ----------------------------------------------------
        # Find a sustained deviation.
        #
        # A single noisy sample is not enough to terminate
        # the charge event.
        # ----------------------------------------------------

        end = preliminary_end

        deviation_start = None

        for j, dev in enumerate(deviation):

            if dev > current_deviation_fraction:

                if deviation_start is None:
                    deviation_start = j

                t0 = time[
                    start + deviation_start
                ]

                t1 = time[
                    start + j
                ]

                if (
                    t1 - t0
                    >= min_deviation_duration_s
                ):

                    # Charge ends immediately before the
                    # sustained current deviation.
                    end = (
                        start
                        + deviation_start
                        - 1
                    )

                    break

            else:

                # Current returned to the normal
                # charge-current band.
                deviation_start = None

        # ----------------------------------------------------
        # Calculate charge duration
        # ----------------------------------------------------

        if end <= start:
            continue

        duration_h = (
            time[end] - time[start]
        ) / 3600

        # ----------------------------------------------------
        # Keep approximately full charge events
        # ----------------------------------------------------

        if not (
            min_duration_h
            <= duration_h
            <= max_duration_h
        ):
            continue

        events.append({
            "start_idx": start,
            "end_idx": end,

            "duration_h": duration_h,

            "current_A": characteristic_current,

            # EFC is discharge-based, so it is a single
            # reference value for the charge event.
            "efc": df.loc[
                start,
                "EFC"
            ],
        })

    return events


# ============================================================
# 3. Find EFC index
# ============================================================

def find_efc_index(df, target_efc):

    idx = (
        df["EFC"]
        .sub(target_efc)
        .abs()
        .idxmin()
    )

    return idx


# ============================================================
# 4. Distance between a charge event and an EFC index
# ============================================================

def distance_to_index(event, target_idx):

    if target_idx < event["start_idx"]:
        return (
            event["start_idx"]
            - target_idx
        )

    if target_idx > event["end_idx"]:
        return (
            target_idx
            - event["end_idx"]
        )

    return 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    csv_file = (
        "/work3/claho/battery_datasets/lg_mj1/"
        "Cycl_T25_SOC10-90_Dch1.5C_Ch1.0C_"
        "Vito_Cell85_AllData.csv"
    )

    df = pd.read_csv(csv_file)


    # ------------------------------------------------------------
    # Calculate global discharge-based EFC
    # ------------------------------------------------------------

    df = calculate_efc(
        df,
        q_nom=Q_NOM,
    )


    # ------------------------------------------------------------
    # Find the actual dataframe index closest to EFC = 50
    # ------------------------------------------------------------

    efc50_idx = find_efc_index(
        df,
        target_efc=50.0,
    )

    print(
        f"EFC 50 index: {efc50_idx}"
    )

    print(
        f"EFC at index: "
        f"{df.loc[efc50_idx, 'EFC']:.6f}"
    )


    # ------------------------------------------------------------
    # Find full charge events
    # ------------------------------------------------------------

    events = find_charge_events(
        df,

        # Positive current above this value
        current_threshold=3,

        # End when current deviates by >5%
        current_deviation_fraction=0.05,

        # Deviation must persist for 20 seconds
        min_deviation_duration_s=20,

        # Charge duration limits
        min_duration_h=0.1,
        max_duration_h=3,
    )

    print(
        f"Number of full charge events: "
        f"{len(events)}"
    )


    # ------------------------------------------------------------
    # First full charge
    # ------------------------------------------------------------

    first_charge = events[0]


    # ------------------------------------------------------------
    # Charge closest to EFC 50
    # ------------------------------------------------------------

    efc50_charge = min(
        events,
        key=lambda event:
            distance_to_index(
                event,
                efc50_idx,
            )
    )


    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("\nFirst full charge")

    print(
        f"  Index: "
        f"{first_charge['start_idx']} "
        f"-> "
        f"{first_charge['end_idx']}"
    )

    print(
        f"  EFC: "
        f"{first_charge['efc']:.4f}"
    )

    print(
        f"  Duration: "
        f"{first_charge['duration_h']:.3f} h"
    )

    print(
        f"  Characteristic current: "
        f"{first_charge['current_A']:.3f} A"
    )


    print("\nCharge closest to EFC 50")

    print(
        f"  Index: "
        f"{efc50_charge['start_idx']} "
        f"-> "
        f"{efc50_charge['end_idx']}"
    )

    print(
        f"  EFC: "
        f"{efc50_charge['efc']:.4f}"
    )

    print(
        f"  Duration: "
        f"{efc50_charge['duration_h']:.3f} h"
    )

    print(
        f"  Characteristic current: "
        f"{efc50_charge['current_A']:.3f} A"
    )


    # ============================================================
    # Extract voltage curves
    # ============================================================

    first_data = df.loc[
        first_charge["start_idx"]:
        first_charge["end_idx"]
    ]

    efc50_data = df.loc[
        efc50_charge["start_idx"]:
        efc50_charge["end_idx"]
    ]


    first_time = (
        first_data["Total_Time_Seconds"]
        - first_data["Total_Time_Seconds"].iloc[0]
    )

    efc50_time = (
        efc50_data["Total_Time_Seconds"]
        - efc50_data["Total_Time_Seconds"].iloc[0]
    )


    first_voltage = first_data["Voltage_V"]

    efc50_voltage = efc50_data["Voltage_V"]


    # ============================================================
    # Plot voltage curves
    # ============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    axes[0].plot(
        first_time / 3600,
        first_voltage,
    )

    axes[0].set_title(
        "First full charge"
    )

    axes[0].set_xlabel(
        "Time (h)"
    )

    axes[0].set_ylabel(
        "Voltage (V)"
    )

    axes[0].grid(True)


    axes[1].plot(
        efc50_time / 3600,
        efc50_voltage,
    )

    axes[1].set_title(
        "Charge closest to EFC 50"
    )

    axes[1].set_xlabel(
        "Time (h)"
    )

    axes[1].grid(True)


    plt.tight_layout()

    output_dir = Path(
        "project/reports/figures/LG_MJ1"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_dir / "correct_voltage_charge_curves.png",
        dpi=300,
    )

    plt.close()