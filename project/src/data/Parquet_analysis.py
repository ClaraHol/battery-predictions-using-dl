import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the parquet file (here is an example of PD/cell 113)
SCRATCH = Path("/work3/claho")
DEST_ROOT = SCRATCH / "battery_datasets/processed"

loadpath  =DEST_ROOT/"efc50_curves.parquet"

# Load the quantities you'd like to use
data=pq.read_table(loadpath).to_pandas()
print(data.columns.tolist())

current = np.array(data["current_A"])
voltage = np.array(data["voltage_V"])
step = np.array(data["cycle_number"])
rpt = np.array(data["temp_C"])
soc = np.array(data["cell_id"])

# You can also save the data as other format, like .csv
# savepath="/save_path/"
# data.to_csv(savepath+'XXX.csv',sep=',', index=False, mode='w', encoding='utf-8')
print(step)
# You can also visualize them
plt.plot(voltage, linewidth=1.2)
plt.plot(current, linewidth=1.2)
plt.savefig("test_parquet")
