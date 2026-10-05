import pandas as pd

data_dir = "/work3/claho/battery_datasets/processed/"


efc0 = pd.read_parquet(data_dir + "public_charging_curve_early.parquet")
efc50 = pd.read_parquet(data_dir + "public_charging_curve_late.parquet")
metadata = pd.read_csv(data_dir + "public_cell_metadata_all.csv")


source_to_battery = {
    "snl_lfp": "A123-M1A",
    "snl_nmc": "LG-HG2",
    "tongji": "Samsung-25R",
    "tum": "Sony-VTC5A",
    "cmu": "Sony-VTC6",
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

    return None


for df in [efc0, efc50]:
    df["source_dataset"] = df["cell_id"].map(get_source)
    df["battery_type"] = df["source_dataset"].map(source_to_battery)

battery_data = {
    battery: {
        "efc0": efc0[efc0["battery_type"] == battery].copy(),
        "efc50": efc50[efc50["battery_type"] == battery].copy(),
    }
    for battery in source_to_battery.values()
}

for battery, data in battery_data.items():
    print(
        battery,
        data["efc0"]["cell_id"].nunique(),
        data["efc50"]["cell_id"].nunique(),
    )

print("Total unique cells:")
print("efc0 :", efc0["cell_id"].nunique())
print("efc50:", efc50["cell_id"].nunique())

print("\nAll EFC0 cell IDs:")
print(efc0["cell_id"].drop_duplicates().tolist())