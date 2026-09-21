import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path(f"/work3/{os.environ.get('USER', '')}/battery_datasets/sandia_snl")
OUTPUT_DIR = Path(f"project/reports/figures/snl")
df = pd.read_csv(DATA_ROOT/"voltage_timeseries_all_cells.csv")

# Keep only cycle 100
CYCLE = 100
df100 = df[df["cycle"] == CYCLE].copy()

df100["chemistry"] = df100["cell_id"].str.extract(
    r"SNL_18650_(LFP|NCA|NMC)_"
)
print(df100)
for chemistry, group_chem in df100.groupby("chemistry"):

    plt.figure(figsize=(9, 6))

    for cell_id, group in group_chem.groupby("cell_id"):
        group = group.sort_values("cycle_time")
        plt.plot(group["cycle_time"], group["v"], label=cell_id)

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"{chemistry} — Cycle {CYCLE}")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"snl_voltage_{chemistry}_{CYCLE}")