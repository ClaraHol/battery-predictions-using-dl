import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
all_rpts = []

for file_dir in files:
    print(file_dir)
    try:
        curves, metadata, rpt_summary = extract_curve_vtc6(
            file_dir, return_rpt_summary=True
        )

        cell_id = f"cmu_{file_dir.stem}"

        curves["cell_id"] = cell_id
        curves["chemistry"] = "VTC6"

        metadata["cell_id"] = cell_id
        metadata["chemistry"] = "VTC6"

        rpt_summary["cell_id"] = cell_id
        rpt_summary["chemistry"] = "VTC6"

        # Optional cross-check against the published impedance/RPT file.
        impedance_file = file_dir.with_name(f"{file_dir.stem}_impedance.csv")
        if impedance_file.is_file():
            impedance = pd.read_csv(impedance_file)
            metadata["n_impedance_rpts"] = len(impedance)
            metadata["rpt_count_matches_impedance"] = (
                len(impedance) == metadata["n_rpts"]
            )
        else:
            metadata["n_impedance_rpts"] = np.nan
            metadata["rpt_count_matches_impedance"] = np.nan

        all_curves.append(curves)
        all_metadata.append(metadata)
        all_rpts.append(rpt_summary)

    except ValueError as exc:
        print(f"Skipping {file_dir.name}: {exc}")


if not all_curves:
    raise RuntimeError("No VTC6 cells were processed successfully")

output_dir = Path(OUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

curves_df = pd.concat(all_curves, ignore_index=True)
metadata_df = pd.DataFrame(all_metadata)
rpt_df = pd.concat(all_rpts, ignore_index=True)

curves_df.to_parquet(
    output_dir / "vtc6_all_curves.parquet", index=False
)
metadata_df.to_parquet(
    output_dir / "vtc6_all_metadata.parquet", index=False
)
rpt_df.to_parquet(
    output_dir / "vtc6_rpt_summary.parquet", index=False
)

print(f"\nProcessed {len(metadata_df)} VTC6 cells")
print(
    metadata_df[
        [
            "cell_id",
            "cycle_life_efc",
            "cycle_life_censored",
            "temperature_C",
            "charge_c_rate",
            "discharge_c_rate",
            "charge_voltage_V",
            "n_rpts",
        ]
    ].to_string(index=False)
)

print(
    f"\nLifetime threshold not reached (censored): "
    f"{metadata_df['cycle_life_censored'].sum()} / {len(metadata_df)}"
)

mismatch = metadata_df[
    metadata_df["rpt_count_matches_impedance"].eq(False)
]
if not mismatch.empty:
    print("\nWarning: RPT count differs from impedance-file count for:")
    print(
        mismatch[["cell_id", "n_rpts", "n_impedance_rpts"]].to_string(index=False)
    )
