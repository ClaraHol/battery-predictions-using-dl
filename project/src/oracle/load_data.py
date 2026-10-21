import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

data_dir = "/work3/claho/battery_datasets/processed/"


efc0 = pd.read_csv(data_dir + "public_charging_curve_early_all.csv")
efc50 = pd.read_csv(data_dir + "public_charging_curve_late_all.csv")
source_to_battery = {
    "snl_lfp": "A123-M1A",
    "snl_nmc": "LG-HG2",
    "tongji": "Samsung-25R",
    "tum": "Sony-VTC5A",
    "cmu": "Sony-VTC6",
    "elt": "LG-MJ1"
}


def get_source(cell_id):
    if cell_id.startswith("snl_lfp_"):
        return "snl_lfp"
    if cell_id.startswith("snl_nmc_"):
        return "snl_nmc"
    if cell_id.startswith("tongji_"):
        return "tongji"
    if cell_id.startswith("tum_"):
        return "tum"
    if cell_id.startswith("cmu_"):
        return "cmu"
    if cell_id.startswith("elt_"):
        return "elt"

    return None


for df in [efc0, efc50]:
    df["source_dataset"] = df["cell_id"].map(get_source)
    df["battery_type"] = df["source_dataset"].map(source_to_battery)

print(
    "Unique battery_types:",
    pd.concat([efc0["battery_type"], efc50["battery_type"]]).dropna().unique(),
)


print("Total unique cells:")
print("efc0 :", efc0["cell_id"].nunique())
print("efc50:", efc50["cell_id"].nunique())

print("\nNumber of cells:", len(efc50["cell_id"]))
print("efc50 header:", efc50.columns.tolist())
print(efc50.head())

print("efc0 header", efc0.columns.tolist())

print('efc0.head() for source_dataset = "tum":')
print(efc0.loc[efc0["source_dataset"] == "tum"].head())




out_dir = Path("project/reports/figures/oracle_testing")
out_dir.mkdir(parents=True, exist_ok=True)
cell_indices = random.sample(
    range(efc0["cell_id"].drop_duplicates().shape[0]), 9
)
cell_ids = efc0["cell_id"].drop_duplicates().iloc[cell_indices]

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, cell_id in zip(axes.flat, cell_ids):
    cell_data = efc0[efc0["cell_id"] == cell_id]
    ax.plot(cell_data["early_curve"].to_numpy())
    ax.set_xlabel("Sample")
    ax.set_ylabel("Lower curve")
    ax.set_title(f"Lower curve for {cell_id}")
fig.tight_layout()
fig.savefig(out_dir / "lower_curves_efc0_cells.png")
plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, cell_id in zip(axes.flat, cell_ids):
    cell_data = efc50[efc50["cell_id"] == cell_id]
    ax.plot(cell_data["upper_curve"].to_numpy())
    ax.set_xlabel("Sample")
    ax.set_ylabel("Lower curve")
    ax.set_title(f"Lower curve for {cell_id} at efc50")
fig.tight_layout()
fig.savefig(out_dir / "lower_curves_efc50_cells.png")
plt.close(fig)


selected_cell_ids = [
    "tum_BW-VTC-234",
    "tum_BW-VTC-221",
    "tum_BW-VTC-431",
    "snl_nmc_SNL_18650_NMC_25C_0-100_0.5/2C_b",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, cell_id in zip(axes.flat, selected_cell_ids):
    cell_data = efc50[efc50["cell_id"] == cell_id]
    voltage = cell_data["lower_curve"].to_numpy()
    time = cell_data["lower_curve_time"].to_numpy()
    ax.plot(time, voltage)
    ax.set_xlabel("Time (sample)")
    ax.set_ylabel("Voltage")
    ax.set_title(f"Voltage over time for {cell_id}")
    ax.grid(True)
fig.tight_layout()
fig.savefig(out_dir / "time_voltage_selected_cells.png")
plt.close(fig)


