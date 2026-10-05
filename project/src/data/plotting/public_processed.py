from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_ROOT = Path("/work3/claho/battery_datasets/processed/")
OUTPUT_DIR = Path("project/reports/figures/public_cells")
metadata_path = "public_cell_metadata_cleaned.csv"
metadata = pd.read_csv(DATA_ROOT/ metadata_path)


cycle_lives = {}
chrg_rates = {}
dchrg_rates = {}
temps = {}
for source, group in metadata.groupby("source_dataset"):
    cycle_lives[source]=  group["cycle_life"]
    chrg_rates[source] = group["charge_c_rate"]
    dchrg_rates[source] = group["discharge_c_rate"]
    temps[source] = group["temp_C"]


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].boxplot(cycle_lives.values(), tick_labels = cycle_lives.keys())
ax[0,0].set_title("Cycle life")
ax[0,1].boxplot(temps.values(), tick_labels = temps.keys())
ax[0,1].set_title("Temperature")
ax[1,0].boxplot(chrg_rates.values(), tick_labels = chrg_rates.keys())
ax[1,0].set_title("Charge C-Rate")
ax[1,1].boxplot(dchrg_rates.values(), tick_labels = dchrg_rates.keys())
ax[1,1].set_title("Discharge C-Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"metadata_boxplot")
import pyarrow.parquet as pq