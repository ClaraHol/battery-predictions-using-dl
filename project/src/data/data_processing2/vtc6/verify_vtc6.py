import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/vtc6")

curves = pd.read_parquet(DATA_DIR / "vtc6_all_curves.parquet")
metadata = pd.read_parquet(DATA_DIR / "vtc6_all_metadata.parquet")

print("Metadata header:")
print(metadata.head())

expected = {"efc0", "efc50_lower", "efc50_upper"}
problems = []

for cell_id, cell in curves.groupby("cell_id"):
    missing = expected - set(cell["curve_type"])

    if missing:
        problems.append((cell_id, f"Missing curves: {missing}"))

    for curve_type, df in cell.groupby("curve_type"):
        t = df["time_s"].to_numpy()
        v = df["voltage_V"].to_numpy()
        i = df["current_A"].to_numpy()

        duration_h = (t[-1] - t[0]) / 3600
        current_med = np.median(i)

        issues = []

        if len(t) < 2000:
            issues.append("Too few samples")
        if not (3 <= duration_h <= 6):
            issues.append("Unusual duration")
        if v[0] < 3.7:
            issues.append("Low starting voltage")
        if v[-1] > 2.65:
            issues.append("High ending voltage")
        if abs(current_med - 0.61) > 0.08:
            issues.append("Unexpected current")
        if not (np.isfinite(t).all() and np.isfinite(v).all() and np.isfinite(i).all()):
            issues.append("Non-finite values")
        if np.any(np.diff(t) <= 0):
            issues.append("Non-increasing time")
        if np.any(np.diff(t) > 120):
            issues.append("Large time gap")

        status = "OK" if not issues else "CHECK"
        if issues:
            problems.append((cell_id, f"{curve_type}: {', '.join(issues)}"))

        print(
            f"{cell_id:12} {curve_type:13} "
            f"N={len(t):6} "
            f"T={duration_h:5.2f}h "
            f"V={v[0]:.2f}->{v[-1]:.2f} "
            f"I={current_med:.3f}A "
            f"{status}"
        )

# Check interpolation metadata
for row in metadata.itertuples(index=False):
    if not (
        row.efc50_lower < 50 < row.efc50_upper
        and np.isclose(row.weight_lower + row.weight_upper, 1)
    ):
        problems.append((row.cell_id, "Invalid EFC50 interpolation"))

print("\n========== SUMMARY ==========")
print(f"Cells:            {curves['cell_id'].nunique()}")
print(f"Extracted curves: {curves.groupby(['cell_id', 'curve_type']).ngroups}")
print(f"Expected curves:  {len(metadata) * 3}")
print(f"Flagged issues:   {len(problems)}")

for cell_id, issue in problems:
    print(f"  {cell_id}: {issue}")