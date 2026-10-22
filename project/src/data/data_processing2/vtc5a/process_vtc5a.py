"""Build VTC5A reference-test Parquet files using extract_curve_vtc5a.py.

Run once for all published cells, from a dataset root containing CU_Cyclic,
CU_Calendar, CU_Dynamic, CYC_Cyclic, CYC_Dynamic and the Excel overview.

Example:
    python process_vtc5a.py --data-dir /work3/claho/battery_datasets/sony_vtc5a \
        --out-dir /work3/s214728/battery-predictions/processed_data/vtc5a

To process a single cell:
    python process_vtc5a.py --data-dir /path/to/vtc5a --cell BW-VTC-210

Requirements: numpy, scipy, pandas, openpyxl, pyarrow.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from extract_curve_vtc5a import extract_curve_vtc5a

CELL_RE = re.compile(r"BW[-_]VTC[-_](\d+)", re.IGNORECASE)
CURVE_TYPES = {"efc0", "efc50_lower", "efc50_upper"}


def cell_id(value):
    match = CELL_RE.search(str(value))
    return f"BW-VTC-{int(match.group(1)):03d}" if match else None


def overview_cells(path):
    """Read publication overview; omit non-cell footnotes and extra rows."""
    if path.suffix.lower() == ".csv":
        overview = pd.read_csv(path)
    else:
        overview = pd.read_excel(path)
    required = {"Internal ID", "Aging domain", "Operation"}
    if not required.issubset(overview.columns):
        raise ValueError(f"Overview missing columns: {sorted(required - set(overview.columns))}")
    overview = overview.copy()
    overview["cell_id"] = overview["Internal ID"].map(cell_id)
    overview = overview.loc[overview["cell_id"].notna()].copy()
    if "TP" in overview.columns:
        overview = overview.loc[pd.to_numeric(overview["TP"], errors="coerce").notna()]
    if overview["cell_id"].duplicated().any():
        repeated = overview.loc[overview["cell_id"].duplicated(), "cell_id"].tolist()
        raise ValueError(f"Duplicate cell IDs in overview: {repeated}")
    return overview


def index_mat_files(root):
    """Find CU/CYC files in the documented folder families, keyed by cell."""
    cu = defaultdict(list)
    cyc = defaultdict(list)
    other = []
    for path in root.rglob("*.mat"):
        relative = path.relative_to(root)
        folder = relative.parts[0].upper()
        if folder.startswith("CU_"):
            collection = cu
        elif folder.startswith("CYC_"):
            collection = cyc
        else:
            continue  # e.g., undocumented processed or combined folders
        key = cell_id(path.name)
        if key is None:
            other.append(str(relative))
        else:
            collection[key].append(path)
    return cu, cyc, other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root containing CU_*/CYC_* folders")
    parser.add_argument("--overview", type=Path, help="Cell overview XLSX/CSV (default: within data-dir)")
    parser.add_argument("--out-dir", type=Path, default=Path("/work3/s214728/battery-predictions/processed_data/vtc5a"))
    parser.add_argument("--cell", action="append", help="Optional BW-VTC cell ID; repeat to select several")
    args = parser.parse_args()

    root = args.data_dir.expanduser().resolve()
    overview_path = args.overview or (root / "bawaii_cell_overview_publication.xlsx")
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    if not overview_path.is_file():
        parser.error(f"Cell overview not found: {overview_path}")

    overview = overview_cells(overview_path)
    selected = {cell_id(s) for s in args.cell} if args.cell else None
    if selected is not None and (None in selected or not selected):
        parser.error("--cell must contain a valid ID such as BW-VTC-210")
    if selected is not None:
        missing = selected - set(overview["cell_id"])
        if missing:
            parser.error(f"Cells not listed in overview: {sorted(missing)}")
        overview = overview.loc[overview["cell_id"].isin(selected)]

    cu_by_cell, cyc_by_cell, unrecognized = index_mat_files(root)
    published = set(overview_cells(overview_path)["cell_id"])
    unlisted = (set(cu_by_cell) | set(cyc_by_cell)) - published
    print(f"Overview: {len(overview)} selected cells")
    print(f"Indexed: {sum(map(len, cu_by_cell.values()))} CU and {sum(map(len, cyc_by_cell.values()))} CYC files")
    print(f"Not in overview (excluded): {', '.join(sorted(unlisted)) or 'none'}")
    if unrecognized:
        print(f"WARNING: {len(unrecognized)} CU/CYC files lack recognizable cell IDs")

    all_curves, all_metadata, records = [], [], []
    for row in overview.itertuples(index=False):
        key = row.cell_id
        cu_files = cu_by_cell.get(key, [])
        cyc_files = cyc_by_cell.get(key, [])
        # CU-only cells have no CYC intervals, and calendar-aged cells need
        # not have CYC files. Other cells must have complete cycling intervals.
        original = overview.loc[overview["cell_id"].eq(key)].iloc[0]
        domain = str(original["Aging domain"]).lower()
        operation = str(original["Operation"]).lower()
        require_cyc = "calendar" not in domain and "cu-only" not in operation
        record = {"cell_id": key, "aging_domain": domain,
                  "n_cu_files": len(cu_files), "n_cyc_files": len(cyc_files)}
        try:
            if not cu_files:
                raise ValueError("No CU files")
            curves, meta = extract_curve_vtc5a(
                cu_files, cyc_files, require_cyc_intervals=require_cyc
            )
            if set(curves["curve_type"].unique()) != CURVE_TYPES:
                raise ValueError("Extractor did not return exactly the three required curve types")
            if not (meta["efc50_lower"] < 50 < meta["efc50_upper"]):
                raise ValueError("EFC50 not bracketed")
            if not np.isclose(meta["weight_lower"] + meta["weight_upper"], 1):
                raise ValueError("Interpolation weights do not sum to one")
            curves["cell_id"] = f"tum_{key}"
            curves["chemistry"] = "VTC5A"
            meta.update({"cell_id": f"tum_{key}", "chemistry": "VTC5A"})
            all_curves.append(curves)
            all_metadata.append(meta)
            record.update(status="OK", reason="")
            print(f"OK   {key}: CU{meta['efc50_lower_cu_number']:03d} / "
                  f"CU{meta['efc50_upper_cu_number']:03d}, "
                  f"EFC={meta['efc50_lower']:.2f}..{meta['efc50_upper']:.2f}, "
                  f"w_upper={meta['weight_upper']:.3f}")
        except (ValueError, KeyError, IndexError, OSError) as error:
            record.update(status="SKIPPED", reason=str(error))
            print(f"SKIP {key}: {error}")
        records.append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_dir / "vtc5a_processing_log.csv", index=False)
    if not all_curves:
        raise RuntimeError("No cells could be processed; see vtc5a_processing_log.csv")

    curves = pd.concat(all_curves, ignore_index=True)
    metadata = pd.DataFrame(all_metadata)
    curves.to_parquet(args.out_dir / "vtc5a_all_curves.parquet", index=False)
    metadata.to_parquet(args.out_dir / "vtc5a_all_metadata.parquet", index=False)
    print(f"\nSaved {metadata.shape[0]} cells, {len(curves)} curve rows "
          f"({3 * len(metadata)} curves); skipped {sum(r['status']=='SKIPPED' for r in records)} cells.")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
