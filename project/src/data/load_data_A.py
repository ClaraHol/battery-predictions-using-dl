import numpy as np
from matplotlib import pyplot as pyplot


import pandas as pd
from pathlib import Path

scratch_dir = Path("/work3/claho/battery_datasets/lg_mj1")
# file_dir = "DrivingAgeing_T25_SOC10-90_Vito_Cell88_AllData"
# file_dir = "Cycl_T25_SOC10-90_Dch0.5C_Ch0.5C_Vito_Cell78_AllData"
# file_dir = "Cycl_T25_SOC10-90_Dch1.5C_Ch1.0C_Vito_Cell85_AllData"
# file_dir = "Cycl_T45_SOC10-90_Dch0.5C_Ch0.5C_Vito_Cell63_AllData"
file_dir = "DrivingAgeing_T25_SOC70-90_Vito_Cell86_AllData"
# Single file

df = pd.read_csv(scratch_dir / (file_dir +".csv"))
print(list(df.columns))

# Write the first half of the time steps to a separate CSV file.


# Plot voltage as a function of time.
time = df["Total_Time_Seconds"]
voltage = df["Voltage_V"]
current = df["Current_A"]

charge_cap_Ah = df["Charge_Capacity_Ah"]
discharge_cap_Ah = df["Discharge_Capacity_Ah"]

charge_cap_Wh = df["Charge_Capacity_Wh"]
discharge_cap_Wh = df["Discharge_Capacity_Wh"]

unnamed = df["Unnamed: 0"]
step = df["Step"]

n = time.shape[0]
frac = 0.01
num_points = int(np.ceil(n*frac))


lower =  648333 - 5000
upper =  648347


time = time[lower:upper]
voltage = voltage[lower:upper]
current = current[lower:upper]
charge_cap_Ah = charge_cap_Ah[lower:upper]
discharge_cap_Ah = discharge_cap_Ah[lower:upper]
charge_cap_Wh = charge_cap_Wh[lower:upper]
discharge_cap_Wh = discharge_cap_Wh[lower:upper]
unnamed = unnamed[lower:upper]
step = step[lower:upper]


figure, axes = pyplot.subplots(1, 2, figsize=(12, 5))

axes[0].plot(time, voltage)
axes[0].set_xlabel("Time Seconds")
axes[0].set_ylabel("Voltage V")
axes[0].set_title("Voltage vs. Time")
axes[0].grid(True)

axes[1].plot(time, current)
axes[1].set_xlabel("Time Seconds")
axes[1].set_ylabel("Current A")
axes[1].set_title("Current vs. Time")
axes[1].grid(True)

pyplot.tight_layout()
output_dir = Path("project/reports/figures/LG_MJ1")
output_dir.mkdir(parents=True, exist_ok=True)
pyplot.savefig(output_dir / "voltage_vs_time.png", dpi=300)
pyplot.close()
