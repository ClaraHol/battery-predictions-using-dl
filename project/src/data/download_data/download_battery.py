import csv
import os
import requests

URL = "https://database.batteryarchive.org/api/queries/40/results"

HEADERS = {
    "Authorization": "Key 61PzyYn9u56njtPDOh37VVBFasGusUlxc7WZ1FSj",
    "Content-Type": "application/json",
}

CELLS = [
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_a",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_b",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_c",
    "SNL_18650_LFP_25C_20-80_0.5/0.5C_d",

    "SNL_18650_LFP_25C_20-80_0.5/3C_a",

    "SNL_18650_LFP_25C_40-60_0.5/0.5C_a",
    "SNL_18650_LFP_25C_40-60_0.5/0.5C_b",
    "SNL_18650_LFP_25C_40-60_0.5/3C_a",
    "SNL_18650_LFP_25C_40-60_0.5/3C_b",

    "SNL_18650_NCA_25C_20-80_0.5/0.5C_a",
    "SNL_18650_NCA_25C_20-80_0.5/0.5C_b",
    "SNL_18650_NCA_25C_20-80_0.5/0.5C_c",
    "SNL_18650_NCA_25C_20-80_0.5/0.5C_d",

    "SNL_18650_NCA_25C_40-60_0.5/0.5C_a",
    "SNL_18650_NCA_25C_40-60_0.5/0.5C_b",

    "SNL_18650_NMC_25C_20-80_0.5/0.5C_a",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_b",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_c",
    "SNL_18650_NMC_25C_20-80_0.5/0.5C_d",

    "SNL_18650_NMC_25C_20-80_0.5/3C_a",
    "SNL_18650_NMC_25C_20-80_0.5/3C_b",

    "SNL_18650_NMC_25C_40-60_0.5/0.5C_a",
    "SNL_18650_NMC_25C_40-60_0.5/0.5C_b",
    "SNL_18650_NMC_25C_40-60_0.5/3C_a",
    "SNL_18650_NMC_25C_40-60_0.5/3C_b",
]

OUTPUT_DIR = "battery_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_cell(cell_id):
    print(f"\nDownloading: {cell_id}")

    payload = {
        "id": 40,
        "parameters": {
            "cell_id": [cell_id]
        },
        "apply_auto_limit": False,
    }

    response = requests.post(
        URL,
        headers=HEADERS,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()

    data = response.json()["query_result"]["data"]
    rows = data["rows"]

    # Make a safe filename
    filename = cell_id.replace("/", "_") + ".csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["series", "cycle_index", "test_time", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Rows: {len(rows):,}")
    print(f"  Saved: {filepath}")


for i, cell in enumerate(CELLS, start=1):
    print(f"\n[{i}/{len(CELLS)}]")
    download_cell(cell)

print("\nAll downloads complete!")