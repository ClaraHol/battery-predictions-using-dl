import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load the parquet file (here is an example of PD/cell 113)
# loadpath=r"C:/Users\Anton Espholm\Downloads\raw_data\raw_data\PD\113_single_parquet_with_soc_and_corrections.parquet"
ORIG_DIR="./project/data/raw_data/"
BATTERY_TYPE = "PD"
loadpath = ORIG_DIR + BATTERY_TYPE 
files = glob.glob("*.parquet", root_dir=loadpath)


RPT = 1
STEP = 8

# Load the quantities you'd like to use
for file in files:
    
    file_name = file.split(".")[0]
    print(f"Plotting {file_name}")
    data=pq.read_table(loadpath+"/"+file).to_pandas()
    data["time_step"] = np.array(data["del_time"]).cumsum()/36000
    rpt1 = data[data["rpt"] == 1].copy()
    #print(data.columns.tolist())

    #cyc1 = charge = data[(data["rpt"] == RPT) & (data["step_no"] == STEP)].copy()


    #cyc1 = cyc1[cyc1["total_dischg_thrgh_ah_rounded"]>0]
    # voltage = np.array(cyc1["voltage (V)"])
    # current = np.array(cyc1["current (A)"])
    # time = np.array(cyc1["time_step"])


    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot voltage
    ax.plot(
        rpt1["time_step"],
        rpt1["voltage (V)"],
        linewidth=1,
        label = "voltage"
    )
    ax.plot(
        rpt1["time_step"],
        rpt1["current (A)"],
        linewidth=1,
        label = "current"
    )

    # Add vertical lines and labels for each step
    for step, group in rpt1.groupby("step_no"):
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
    ax.set_ylabel("Voltage (V)/Current (A)")
    ax.set_title("RPT 1 — Voltage/Current profile and test steps")

    # ax.set_ylim(
    #     rpt1["voltage (V)"].min() - 0.05,
    #     rpt1["voltage (V)"].max() + 0.15
    # )

    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"project/reports/figures/{file_name}_rpt_{RPT}")
    plt.show()

    # Make time-voltage plot
    # plt.figure()
    # plt.plot(time, voltage, label = "voltage")
    # plt.plot(time, current, label = "current")
    # plt.legend()
    # plt.title(f" {BATTERY_TYPE} cycle {STEP}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Voltage (V)")
    # plt.savefig(f"project/reports/figures/{file_name}_c_v")
    break
# 
# step = np.array(data["step_no"])
# rpt = np.array(data["rpt"])
# soc = np.array(data["soc"])
#total_dischg_thrgh_ah_rounded = np.array(data["total_dischg_thrgh_ah_rounded"])

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
