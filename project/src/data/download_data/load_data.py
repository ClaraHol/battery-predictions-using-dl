from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# Set path to scratch folder, where data is stored
# --------------------------------------------------------------------------
SCRATCH = Path("/work3/claho")
DEST_ROOT = SCRATCH / "battery_datasets"

dest_dir = DEST_ROOT / "sony_vtc5a_raw" / "CU_Calendar" / "CU000"
file_name = "BW-VTC-151_2119_CU_cal_000_BW-VTC-CAL.mat"
import scipy.io as sio

mat = sio.loadmat(
    "/work3/claho/battery_datasets/sony_vtc5a_raw/CU_Calendar/CU000/BW-VTC-151_2119_CU_cal_000_BW-VTC-CAL.mat",
    struct_as_record=False,
    squeeze_me=True,
)


ds = mat["Dataset"]


def describe(obj, prefix=""):
    if hasattr(obj, "_fieldnames"):
        for f in obj._fieldnames:
            val = getattr(obj, f)
            print(f"{prefix}{f}: {type(val).__name__}", getattr(val, "shape", ""))
    else:
        print(f"{prefix}: {type(obj).__name__}", getattr(obj, "shape", ""))


# describe(ds)
describe(ds.Info)
# --------------------------------------------------------------------------
# Load samsung_25r
# --------------------------------------------------------------------------

# Header:
# time/s  control/V/mA   Ecell/V       <I>/mA  Q discharge/mA.h  Q charge/mA.h  control/V  control/mA  cycle number
dest_dir = Path(DEST_ROOT / "samsung_25r_raw" / "Dataset_1_NCA_battery")
df = pd.read_csv(dest_dir / "CY25-1_1-#1.csv")
glued_data = pd.DataFrame()
# for file_name in dest_dir.glob('*.csv'):
#    x = pd.read_csv(file_name, low_memory=False)
#    C_inits = x.loc[x["Q discharge/mA.h"]>0]
# C_init = C_inits.loc[0]
#    print(C_inits)
# cap = x["Q discharge/mA.h"]/C_init
# glued_data = pd.concat([glued_data,cap],axis=0)
C_init = df.loc[df["cycle number"] == 2.0, "Q discharge/mA.h"].max()
max_discharge = df.groupby("cycle number")["Q discharge/mA.h"].max().reset_index()
# print(max_discharge)
# print(C_init)
cap = max_discharge["Q discharge/mA.h"] / C_init
cap.plot()

# print(glued_data)
plt.savefig("capacity_plot")
