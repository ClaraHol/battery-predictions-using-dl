
#!/usr/bin/env python3
"""Summarise one check-up file block by block, to find which Line is the capacity test.
Usage: python inspect_checkup.py path/to/BW-VTC-xxx_CU_...mat
"""
import sys
import numpy as np, pandas as pd
from build_cell_csvs import load_fields
print("Hey")
 
d = load_fields(sys.argv[1], ["Time", "U", "I", "Line", "Command", "State"])
print("Loaded file")
n = len(d["Time"])
dt = np.r_[0.0, np.clip(np.diff(d["Time"]), 0, None)]
df = pd.DataFrame({k: v for k, v in d.items() if len(v) == n})
df["dt"] = dt
df["block"] = ((df.Line.diff() != 0) | (df.Command != df.Command.shift())).cumsum()
g = df.groupby("block")
summary = pd.DataFrame({
    "Line": g.Line.first(), "Command": g.Command.first(), "State": g.State.first(),
    "dur_min": g.dt.sum() * 60,
    "Ah": (df.I * df.dt).groupby(df.block).sum(),
    "I_mean": g.I.mean(), "U_start": g.U.first(), "U_end": g.U.last(),
})
pd.set_option("display.width", 200, "display.max_rows", 500)
print(summary.round(3))
print("\nLargest discharge blocks:")
print(summary[summary.Command == "Discharge"].sort_values("Ah").head(5).round(3))
