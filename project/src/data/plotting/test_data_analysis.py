import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os

# Load the parquet file (here is an example of PD/cell 113)
# loadpath=r"C:/Users\Anton Espholm\Downloads\raw_data\raw_data\PD\113_single_parquet_with_soc_and_corrections.parquet"

# Print full dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

# Set up
DATA_ROOT = Path(f"/work3/claho/battery_datasets/pouch_cells/test_data")
BATTERY_TYPE = "PA-b1"

OUTPUT_DIR = Path(f"project/reports/figures/{BATTERY_TYPE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


loadpath = DATA_ROOT / BATTERY_TYPE 
files = glob.glob("*.parquet", root_dir=loadpath)

RPT = 0
STEP = 4

# Load the quantities you'd like to use

for file in files:
    fig, ax = plt.subplots(figsize=(14, 6))
    file_name = file.split(".")[0]
    cell_id =file.split("_")[0]
    print(f"Plotting {file_name}")
    
    data=pq.read_table(loadpath / file).to_pandas()
    print(data.head())

    rpt1 = data[data["rpt"] == RPT].copy()
    rpt1["step_instance"] = (rpt1["step_no"].ne(rpt1["step_no"].shift())).cumsum() -1   

    #rpt1 = rpt1[rpt1["step_instance"] == STEP].copy()
    rpt1["time_step"] = np.array(rpt1["del_time"]).cumsum()/36000
    #print(rpt1["step_no"])
    

    

    # Plot voltage
    ax.plot(
        rpt1["time_step"],
        rpt1["voltage (V)"],
        linewidth=1,
        label = cell_id
    )
    

    # Add vertical lines and labels for each step
    for step, group in rpt1.groupby("step_instance"):
        start = group["time_step"].iloc[0]

        ax.axvline(
            start,
            linestyle="--",
            alpha=0.4
        )

        # Put step number near the top of the plot
        ax.text(
            start,
            rpt1["voltage (V)"].max() + 0.02,
            str(int(step)),
            rotation=90,
            ha="right",
            va="bottom"
        )

    ax.set_xlabel("Time from start of RPT 1 (hours)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"RPT {RPT} — Voltage profile")

    ax.set_ylim(
        rpt1["voltage (V)"].min() - 0.05,
        rpt1["voltage (V)"].max() + 0.15
    )

    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"{BATTERY_TYPE}_{cell_id}_rpt_{RPT}")

  

# You can also save the data as other format, like .csv
# savepath="/save_path/"
# data.to_csv(savepath+'XXX.csv',sep=',', index=False, mode='w', encoding='utf-8')

def normalize(A):
    return (A-A.min())/(A.max()-A.min())

# You can also visualize them
# my_dict_norm = {"current (A)": normalize(current), "voltage (V)": normalize(voltage), "total_dischg_thrgh_ah_rounded": normalize(total_dischg_thrgh_ah_rounded)}
# my_dict = {"current (A)": current, "voltage (V)": voltage}
# plt.boxplot(my_dict.values())
# plt.show()

# plt.boxplot(my_dict_norm.values())
# plt.show()
