import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os

DATA_ROOT = Path(f"/work3/claho/battery_datasets/processed/pouch_cells")
file = "farasis_efc0_curves.parquet"
file_name = file.split(".")[0]

OUTPUT_DIR = Path(f"project/reports/figures/{file_name}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


loadpath = DATA_ROOT / file

data=pq.read_table(loadpath).to_pandas()
print(data.head())



for cell_id, group in data.groupby("cell_id"):
    fig, ax = plt.subplots(figsize=(14, 6))
    print(f"Plotting voltage for for {cell_id}")
    
    group["time_step"] = np.array(group["time_s"]).cumsum()/36000
    group["time_step"] = group["time_step"] - group["time_step"].min()

    # Plot voltage
    ax.plot(
        group["time_step"],
        group["voltage_V"],
        linewidth=1,
        label = cell_id
    )

    # Add vertical lines and labels for each step
    for step, time_group in group.groupby("step_no"):
        start = time_group["time_step"].iloc[0]

        ax.axvline(
            start,
            linestyle="--",
            alpha=0.4
        )

        # Put step number near the top of the plot
        ax.text(
            start,
            group["voltage_V"].max() + 0.02,
            str(int(step)),
            rotation=90,
            ha="right",
            va="bottom"
        )

    ax.set_xlabel("Time from start of RPT 0 (hours)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("RPT 0 — Voltage profile and test steps")

    ax.set_ylim(
        group["voltage_V"].min() - 0.05,
        group["voltage_V"].max() + 0.15
    )

    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"{cell_id}")
