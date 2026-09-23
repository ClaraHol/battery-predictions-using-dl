import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

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
    start_threshold=0.1,
    reference_duration_s=60,
    current_deviation_fraction=0.10,
    min_deviation_duration_s=20,
    min_duration_h=0.1,
    max_duration_h=3.0,
):
    current = df["Current_A"].to_numpy()
    time = df["Total_Time_Seconds"].to_numpy()

    events = []

    i = 0
    n = len(df)

    while i < n:

        # ------------------------------------------------------
        # 1. Find potential start of positive charging current
        # ------------------------------------------------------
        if current[i] <= start_threshold:
            i += 1
            continue

        start = i

        # ------------------------------------------------------
        # 2. Collect a short initial charging-current window
        # ------------------------------------------------------
        reference_indices = []

        j = start

        while j < n:
            if current[j] > start_threshold:
                reference_indices.append(j)

            if time[j] - time[start] >= reference_duration_s:
                break

            j += 1

        if len(reference_indices) < 3:
            i += 1
            continue

        characteristic_current = np.median(
            current[reference_indices]
        )

        # Reject tiny/noisy positive-current regions
        if characteristic_current <= start_threshold:
            i += 1
            continue

        # ------------------------------------------------------
        # 3. Follow the ORIGINAL current signal forward.
        #
        # The start_threshold is no longer involved.
        # ------------------------------------------------------
        deviation_start = None
        end = None

        j = start

        while j < n:

            relative_deviation = (
                abs(
                    current[j]
                    - characteristic_current
                )
                / abs(characteristic_current)
            )

            if relative_deviation > current_deviation_fraction:

                if deviation_start is None:
                    deviation_start = j

                if (
                    time[j]
                    - time[deviation_start]
                    >= min_deviation_duration_s
                ):
                    end = deviation_start - 1
                    break

            else:
                # It returned to the normal charging-current band
                deviation_start = None

            j += 1

        if end is None:
            break

        # ------------------------------------------------------
        # 4. Duration filtering
        # ------------------------------------------------------
        if end <= start:
            i += 1
            continue

        duration_h = (
            time[end] - time[start]
        ) / 3600.0

        if (
            min_duration_h
            <= duration_h
            <= max_duration_h
        ):
            events.append({
                "start_idx": start,
                "end_idx": end,
                "duration_h": duration_h,
                "current_A": characteristic_current,

                # EFC is essentially constant during charge,
                # so one value is enough.
                "efc": df.loc[start, "EFC"],
            })

        # ------------------------------------------------------
        # 5. Continue searching AFTER this event
        # ------------------------------------------------------
        i = max(end + 1, i + 1)

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

def process_battery_file(file_path, save_dir):
    battery_name = os.path.splitext(os.path.basename(file_path))[0]

    print("\n" + "=" * 80)
    print(f"Processing battery: {battery_name}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(file_path)

    # ------------------------------------------------------------------
    # Calculate global EFC
    # ------------------------------------------------------------------
    df = calculate_efc(
        df,
        q_nom=Q_NOM,
        reset_threshold=0.1
    )

    # ------------------------------------------------------------------
    # Identify charge events
    # ------------------------------------------------------------------
    events = find_charge_events(
        df,
        start_threshold=1,
        current_deviation_fraction=0.05,
        min_deviation_duration_s=20,
        min_duration_h=0.1,
        max_duration_h=4,
    )

    if len(events) < 2:
        print(
            f"WARNING: Only {len(events)} charge event(s) "
            f"found for {battery_name}"
        )
        return

    print(f"Found {len(events)} charge events.")

    # ------------------------------------------------------------------
    # Find the charge event closest to EFC = 50
    # ------------------------------------------------------------------
    efc_target = 50.0

    first_event = events[0]

    efc50_event = min(
        events,
        key=lambda event: abs(event["efc"] - efc_target)
    )

    # Avoid selecting the same event twice
    if efc50_event["start_idx"] == first_event["start_idx"]:
        print(
            "WARNING: The first charge event is also the event "
            "closest to EFC 50."
        )
        selected_events = [first_event]
    else:
        selected_events = [first_event, efc50_event]

    # ------------------------------------------------------------------
    # Extract voltage curves and reset time to zero
    # ------------------------------------------------------------------
    curves = []

    for i, event in enumerate(selected_events, start=1):

        if i == 1:
            event_label = "First charge"
        else:
            event_label = "EFC 50 charge"

        start_idx = event["start_idx"]
        end_idx = event["end_idx"]

        event_df = df.loc[start_idx:end_idx].copy()

        event_time = (
            event_df["Total_Time_Seconds"]
            - event_df["Total_Time_Seconds"].iloc[0]
        )

        voltage = event_df["Voltage_V"].to_numpy()

        curves.append(
            {
                "time": event_time.to_numpy(),
                "voltage": voltage,
                "efc": event["efc"],
            }
        )

        print(
            f"  {event_label}: "
            f"index {start_idx} -> {end_idx}, "
            f"EFC = {event['efc']:.3f}, "
            f"duration = {event_time.iloc[-1] / 3600:.3f} h"
        )

    # ------------------------------------------------------------------
    # Create DataFrame containing the two voltage curves
    #
    # Each curve has its own time column because the number of samples
    # may differ between the two events.
    # ------------------------------------------------------------------
    max_length = max(
        len(curve["time"])
        for curve in curves
    )

    processed_df = pd.DataFrame(
        {
            "Time_1_s": pd.Series(
                curves[0]["time"]
            ),
            "Voltage_1_V": pd.Series(
                curves[0]["voltage"]
            ),
            "Time_2_s": pd.Series(
                curves[1]["time"]
            ),
            "Voltage_2_V": pd.Series(
                curves[1]["voltage"]
            ),
        }
    )

    # ------------------------------------------------------------------
    # Save processed voltage curves
    # ------------------------------------------------------------------
    output_csv = os.path.join(
        save_dir,
        f"Processed_{battery_name}.csv"
    )

    processed_df.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved voltage curves to:")
    print(f"  {output_csv}")

    # ------------------------------------------------------------------
    # Plot the two charge curves
    # ------------------------------------------------------------------
  # ------------------------------------------------------------------
# Plot first charge and EFC50 charge as two subplots
# ------------------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        sharey=True
    )

    # First charge event (EFC ~ 0)
    axes[0].plot(
        curves[0]["time"] / 60.0,
        curves[0]["voltage"]
    )

    axes[0].set_title(
        f"First charge (EFC = {curves[0]['efc']:.2f})"
    )
    axes[0].set_xlabel("Time [min]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True)


    # Charge event closest to EFC 50
    axes[1].plot(
        curves[1]["time"] / 60.0,
        curves[1]["voltage"]
    )

    axes[1].set_title(
        f"EFC 50 charge (EFC = {curves[1]['efc']:.2f})"
    )
    axes[1].set_xlabel("Time [min]")
    axes[1].grid(True)


    # Overall figure title
    fig.suptitle(
        f"{battery_name} - Voltage Curves",
        fontsize=14
    )

    plt.tight_layout()


    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    plot_dir = Path("project/reports/figures/LG_MJ1/")
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_plot = plot_dir / f"{battery_name}_voltage_curves.png"

    plt.savefig(
        output_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved plot to:")
    print(f"  {output_plot}")


# ==========================================================================
# MAIN


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ### THIS SHOULD BE CHANGED TO MATCH YOUR DESIRED DIRECTORY.
    save_dir = "/work3/claho/battery_datasets/"
    DATA_DIR = (
        "/work3/claho/battery_datasets/lg_mj1/"
    )
    save_dir
    csv_files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )
    )

    # Do not accidentally process files generated by this script
    csv_files = [
        f for f in csv_files
        if not os.path.basename(f).startswith("Processed_")
    ]

    print(f"Found {len(csv_files)} battery files.")

    for file_path in csv_files:

        try:
            process_battery_file(file_path)

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{os.path.basename(file_path)}:"
            )
            print(e)