"""Batch-process the VITO/EVERLASTING LG-MJ1 25/45 C CSV files.

The raw files are very large, but ``extract_curve_lg_mj1`` streams each file in
chunks.  Only the three selected curves and a compact RPT summary are retained.
"""

from pathlib import Path

import pandas as pd

from extract_curve_lg_mj1 import extract_curve_lg_mj1, parse_protocol


DATA_DIR = Path("/work3/claho/battery_datasets/lg_mj1")
OUT_DIR = Path("/work3/s214728/battery-predictions/processed_data/lg_mj1")
SELECTED_FILES = None  # e.g. ["Cycl_T25_SOC10-90_Dch0.5C_Ch0.5C_Vito_Cell78_AllData.csv"]


def main():
    root = DATA_DIR.expanduser()
    out = OUT_DIR.expanduser()

    if not root.is_dir():
        raise FileNotFoundError(f"LG-MJ1 dataset directory not found: {root}")

    files = sorted(root.glob("*.csv"))
    if SELECTED_FILES is not None:
        wanted = set(SELECTED_FILES)
        files = [path for path in files if path.name in wanted]
        missing = wanted - {path.name for path in files}
        if missing:
            raise FileNotFoundError(f"Requested files not found: {sorted(missing)}")

    if not files:
        raise RuntimeError(f"No CSV files found in {root}")

    all_curves = []
    all_metadata = []
    all_rpts = []
    records = []

    for path in files:
        try:
            protocol = parse_protocol(path)
            cell_id = f"vito_mj1_cell{protocol['cell_number']:02d}"
            print(f"Processing {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

            curves, metadata, rpt_summary = extract_curve_lg_mj1(path)

            curves["cell_id"] = cell_id
            curves["chemistry"] = "MJ1"

            metadata.update({"cell_id": cell_id, "chemistry": "MJ1"})

            rpt_summary = rpt_summary.copy()
            rpt_summary["cell_id"] = cell_id
            rpt_summary["chemistry"] = "MJ1"
            rpt_summary["source_file"] = path.name

            all_curves.append(curves)
            all_metadata.append(metadata)
            all_rpts.append(rpt_summary)
            records.append(
                {
                    "cell_id": cell_id,
                    "source_file": path.name,
                    "status": "OK",
                    "reason": "",
                    "n_rpts": metadata["n_rpts"],
                    "cycle_life_censored": metadata["cycle_life_censored"],
                    "n_time_resets": metadata["n_time_resets"],
                    "n_timestamp_time_repairs": metadata["n_timestamp_time_repairs"],
                    "n_fallback_time_repairs": metadata["n_fallback_time_repairs"],
                }
            )

            life = (
                f">={metadata['cycle_life_efc']:.1f} EFC (censored)"
                if metadata["cycle_life_censored"]
                else f"{metadata['cycle_life_efc']:.1f} EFC"
            )
            print(
                f"OK   {cell_id}: RPT EFC={metadata['efc50_lower']:.2f}.."
                f"{metadata['efc50_upper']:.2f}, life={life}, "
                f"T={metadata['temperature_C']:.0f}C, "
                f"Cchg={metadata['charge_c_rate']:.1f}, "
                f"Cdis={metadata['discharge_c_rate']:.1f}, "
                f"time_resets={metadata['n_time_resets']}"
            )

        except (ValueError, KeyError, OSError, pd.errors.ParserError) as error:
            cell_id = path.stem
            try:
                protocol = parse_protocol(path)
                cell_id = f"vito_mj1_cell{protocol['cell_number']:02d}"
            except ValueError:
                pass
            print(f"SKIP {path.name}: {error}")
            records.append(
                {
                    "cell_id": cell_id,
                    "source_file": path.name,
                    "status": "SKIPPED",
                    "reason": str(error),
                    "n_rpts": 0,
                    "cycle_life_censored": None,
                    "n_time_resets": None,
                    "n_timestamp_time_repairs": None,
                    "n_fallback_time_repairs": None,
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "lg_mj1_processing_log.csv", index=False)

    if not all_curves:
        raise RuntimeError("No LG-MJ1 cells were processed; see lg_mj1_processing_log.csv")

    curves = pd.concat(all_curves, ignore_index=True)
    metadata = pd.DataFrame(all_metadata)
    rpts = pd.concat(all_rpts, ignore_index=True)

    curves.to_parquet(out / "lg_mj1_all_curves.parquet", index=False)
    metadata.to_parquet(out / "lg_mj1_all_metadata.parquet", index=False)
    rpts.to_parquet(out / "lg_mj1_rpt_summary.parquet", index=False)

    print("\n========== SUMMARY ==========")
    print(f"Input CSV files:       {len(files)}")
    print(f"Processed cells:       {len(metadata)}")
    print(f"Skipped cells:         {sum(r['status'] == 'SKIPPED' for r in records)}")
    print(f"Extracted curves:      {curves.groupby(['cell_id', 'curve_type']).ngroups}")
    print(f"RPT rows:              {len(rpts)}")
    print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
