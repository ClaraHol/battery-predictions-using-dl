from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import pandas as pd

DATA_ROOT = Path("/work3/claho/battery_datasets/processed/")
file = "farasis_efc0_curves.parquet"
file_name = file.split(".")[0]
pouch_name = "pouch_cells/pouch_cells_metadata.csv"
public_name = "public_cell_metadata_cleaned.csv"


OUTPUT_DIR = Path(f"project/reports/figures/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


loadpath = DATA_ROOT / file

#data = pq.read_table(loadpath).to_pandas()
pouch_metadata = pd.read_csv(DATA_ROOT / pouch_name)
public_metadata = pd.read_csv(DATA_ROOT / public_name)
#print(data.head())

pouch_lives = pouch_metadata["cycle_life_efc_units"].to_numpy()
public_lives = public_metadata["cycle_life"].to_numpy()

lives = [pouch_lives, public_lives]
plt.figure()
plt.boxplot(lives, tick_labels=["Validation cells", "Training cells"])
plt.ylabel("Cycle life (EFC)")
plt.title("Distribution of cycle life of training and validation cells")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cycle_life_distribution.png", dpi=300)


public_mean = public_lives.mean()
public_std = public_lives.std()
pouch_mean = pouch_lives.mean()
pouch_std = pouch_lives.std()
mape_pouch = 100 * (np.abs(pouch_lives-pouch_mean)/pouch_lives).mean()
mape_public = 100 * (np.abs(public_lives-public_mean)/public_lives).mean()

print(f" {mape_pouch=}, {pouch_mean=}, {pouch_std=}")
print(f"{mape_public=}, {public_mean=}, {public_std=}")

"""
for cell_id, group in data.groupby("cell_id"):
    fig, ax = plt.subplots(figsize=(14, 6))
    print(f"Plotting voltage for for {cell_id}")

    group["time_step"] = np.array(group["time_s"]).cumsum() / 36000
    group["time_step"] = group["time_step"] - group["time_step"].min()

    # Plot voltage
    ax.plot(group["time_step"], group["voltage_V"], linewidth=1, label=cell_id)

    # Add vertical lines and labels for each step
    for step, time_group in group.groupby("step_no"):
        start = time_group["time_step"].iloc[0]

        ax.axvline(start, linestyle="--", alpha=0.4)

        # Put step number near the top of the plot
        ax.text(
            start,
            group["voltage_V"].max() + 0.02,
            str(int(step)),
            rotation=90,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("Time from start of RPT 0 (hours)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("RPT 0 — Voltage profile and test steps")

    ax.set_ylim(group["voltage_V"].min() - 0.05, group["voltage_V"].max() + 0.15)

    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{cell_id}")
"""
