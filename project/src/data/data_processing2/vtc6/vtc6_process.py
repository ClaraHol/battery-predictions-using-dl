import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys



sys.path.append(str(Path(__file__).resolve().parents[5]))

from project.src.data.data_processing2.vtc6.extract_curve_vtc6 import extract_curve_vtc6

DATA_DIR = "/work3/claho/battery_datasets/sony_vtc6/"
OUT_DIR = "/work3/s214728/battery-predictions/processed_data/vtc6"

files = sorted(
    file_path
    for file_path in Path(DATA_DIR).glob("*.csv")
    if "_" not in file_path.name
)

all_curves = []
all_metadata = []

for file_dir in files:
    print(file_dir)
    try:
        curves, metadata = extract_curve_vtc6(file_dir)

        cell_id = f"cmu_{file_dir.stem}"

        curves["cell_id"] = cell_id
        curves["chemistry"] = "VTC6"

        metadata["cell_id"] = cell_id
        metadata["chemistry"] = "VTC6"

        all_curves.append(curves)
        all_metadata.append(metadata)

    except ValueError as e:
        print(f"Skipping {file_dir.name}: {e}")


output_dir = Path(OUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

pd.concat(all_curves, ignore_index=True).to_parquet(
    output_dir / "vtc6_all_curves.parquet", index=False
)
pd.DataFrame(all_metadata).to_parquet(
    output_dir / "vtc6_all_metadata.parquet", index=False
)


