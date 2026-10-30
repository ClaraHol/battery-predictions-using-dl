"""Batch-process the Zhu2022 Dataset_2_NCM_battery MJ1 cells into common Parquet files.

Important: Dataset_2_NCM_battery is the 3.5 Ah NCM/MJ1 cell set.  Samsung 25R
is Dataset_3_NCM_NCA_battery (2.5 Ah), so this script deliberately scans only
the NCM folder requested by the user and labels the chemistry as MJ1.
"""

from pathlib import Path

import pandas as pd

from extract_curve_samsung_mj1 import extract_curve_samsung_mj1


DATA_DIR = Path("/work3/claho/battery_datasets/samsung_25r/Dataset_2_NCM_battery")
OUT_DIR = Path("/work3/s214728/battery-predictions/processed_data/samsung_mj1")
SELECTED_FILES = None  # e.g. ["CY25-05_1-#1.csv"] or None for all


def main():
    root = DATA_DIR.expanduser()
    out = OUT_DIR.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files = sorted(root.glob("*.csv"))
    if SELECTED_FILES is not None:
        selected = set(SELECTED_FILES)
        files = [path for path in files if path.name in selected]
        missing = selected - {path.name for path in files}
        if missing:
            raise FileNotFoundError(f"Selected files not found: {sorted(missing)}")
    if not files:
        raise RuntimeError(f"No CSV files found in {root}")

    all_curves = []
    all_metadata = []
    all_cycles = []
    records = []

    for path in files:
        cell_id = f"samsung_mj1_{path.stem}"
        try:
            curves, metadata, cycles = extract_curve_samsung_mj1(path)

            curves["cell_id"] = cell_id
            curves["chemistry"] = "MJ1"

            metadata.update({"cell_id": cell_id, "chemistry": "MJ1"})

            cycles["cell_id"] = cell_id
            cycles["chemistry"] = "MJ1"

            all_curves.append(curves)
            all_metadata.append(metadata)
            all_cycles.append(cycles)
            records.append({"cell_id": cell_id, "source_file": path.name, "status": "OK", "reason": ""})

            life = metadata["cycle_life_efc"]
            life_text = f">={life:.1f}" if metadata["cycle_life_censored"] else f"{life:.1f}"
            print(
                f"OK   {path.name}: cycles={metadata['n_cycles']:4d}, "
                f"EFC50={metadata['efc50_lower']:.2f}..{metadata['efc50_upper']:.2f}, "
                f"life={life_text} EFC, T={metadata['temperature_C']:.0f}C, "
                f"Cchg={metadata['charge_c_rate']:.2f}, Cdis={metadata['discharge_c_rate']:.2f}"
            )
        except (ValueError, KeyError, IndexError, OSError) as exc:
            records.append({"cell_id": cell_id, "source_file": path.name, "status": "SKIPPED", "reason": str(exc)})
            print(f"SKIP {path.name}: {exc}")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "samsung_mj1_processing_log.csv", index=False)

    if not all_curves:
        raise RuntimeError("No Samsung-MJ1 cells were processed; see samsung_mj1_processing_log.csv")

    curves = pd.concat(all_curves, ignore_index=True)
    metadata = pd.DataFrame(all_metadata)
    cycles = pd.concat(all_cycles, ignore_index=True)

    curves.to_parquet(out / "samsung_mj1_all_curves.parquet", index=False)
    metadata.to_parquet(out / "samsung_mj1_all_metadata.parquet", index=False)
    cycles.to_parquet(out / "samsung_mj1_cycle_summary.parquet", index=False)

    print("\n========== SUMMARY ==========")
    print(f"Input files:           {len(files)}")
    print(f"Processed cells:       {len(metadata)}")
    print(f"Skipped cells:         {sum(r['status'] == 'SKIPPED' for r in records)}")
    print(f"Extracted curves:      {curves.groupby(['cell_id', 'curve_type']).ngroups}")
    print(f"Cycle-summary rows:    {len(cycles)}")
    print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
    print("Temperature counts:")
    print(metadata["temperature_C"].value_counts().sort_index().to_string())
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
