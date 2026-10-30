"""Build VTC5A curve, metadata, and RPT/SOH Parquet files.

Run once for all published cells from a dataset root containing CU_Cyclic,
CU_Calendar, CU_Dynamic, CYC_Cyclic, CYC_Dynamic and the publication overview.

Edit DATA_DIR / OUT_DIR at the top, then run:
    python process_vtc5a.py
"""

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from extract_curve_vtc5a import extract_curve_vtc5a

DATA_DIR = Path("/work3/claho/battery_datasets/sony_vtc5a")
OUT_DIR = Path("/work3/s214728/battery-predictions/processed_data/vtc5a")
CORRECTED_OVERVIEW = DATA_DIR / "bawaii_cell_overview_publication_corrected.xlsx"
OVERVIEW_PATH = (
    CORRECTED_OVERVIEW
    if CORRECTED_OVERVIEW.is_file()
    else DATA_DIR / "bawaii_cell_overview_publication.xlsx"
)
SELECTED_CELLS = None  # e.g. ["BW-VTC-210"] or None for all

CELL_RE = re.compile(r"BW[-_]VTC[-_](\d+)", re.IGNORECASE)
CURVE_TYPES = {"efc0", "efc50_lower", "efc50_upper"}


def cell_id(value):
    match = CELL_RE.search(str(value))
    return f"BW-VTC-{int(match.group(1)):03d}" if match else None


def overview_cells(path):
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
            continue
        key = cell_id(path.name)
        if key is None:
            other.append(str(relative))
        else:
            collection[key].append(path)
    return cu, cyc, other


def main():
    root = DATA_DIR.expanduser().resolve()
    overview_path = OVERVIEW_PATH.expanduser()
    output_dir = OUT_DIR.expanduser()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not overview_path.is_file():
        raise FileNotFoundError(f"Cell overview not found: {overview_path}")

    overview_all = overview_cells(overview_path)
    overview = overview_all
    if SELECTED_CELLS is not None:
        selected = {cell_id(value) for value in SELECTED_CELLS}
        if None in selected or not selected:
            raise ValueError("SELECTED_CELLS must contain valid IDs such as BW-VTC-210")
        missing = selected - set(overview_all["cell_id"])
        if missing:
            raise ValueError(f"Cells not listed in overview: {sorted(missing)}")
        overview = overview_all.loc[overview_all["cell_id"].isin(selected)]

    cu_by_cell, cyc_by_cell, unrecognized = index_mat_files(root)
    published = set(overview_all["cell_id"])
    unlisted = (set(cu_by_cell) | set(cyc_by_cell)) - published
    print(f"Overview: {len(overview)} selected cells")
    print(
        f"Indexed: {sum(map(len, cu_by_cell.values()))} CU and "
        f"{sum(map(len, cyc_by_cell.values()))} CYC files"
    )
    print(f"Not in overview (excluded): {', '.join(sorted(unlisted)) or 'none'}")
    if unrecognized:
        print(f"WARNING: {len(unrecognized)} CU/CYC files lack recognizable cell IDs")

    all_curves = []
    all_metadata = []
    all_rpts = []
    records = []

    for row in overview.itertuples(index=False):
        key = row.cell_id
        cu_files = cu_by_cell.get(key, [])
        cyc_files = cyc_by_cell.get(key, [])
        original = overview.loc[overview["cell_id"].eq(key)].iloc[0]
        domain = str(original["Aging domain"]).lower()
        operation = str(original["Operation"]).lower()
        is_calendar = "calendar" in domain or "cu-only" in operation
        record = {
            "cell_id": key,
            "aging_domain": domain,
            "n_cu_files": len(cu_files),
            "n_cyc_files": len(cyc_files),
        }

        if is_calendar:
            record.update(
                status="EXCLUDED",
                reason="Calendar/CU-only cell cannot provide a genuine EFC50 ageing bracket",
            )
            records.append(record)
            print(f"EXCL {key}: calendar/CU-only ageing; no EFC50 cycling bracket")
            continue

        try:
            if not cu_files:
                raise ValueError("No CU files")

            curves, meta, rpt_summary = extract_curve_vtc5a(
                cu_files,
                cyc_files,
                require_cyc_intervals=True,
            )
            if set(curves["curve_type"].unique()) != CURVE_TYPES:
                raise ValueError("Extractor did not return exactly the three required curve types")
            if not (meta["efc50_lower"] < 50 < meta["efc50_upper"]):
                raise ValueError("EFC50 not bracketed")
            if not np.isclose(meta["weight_lower"] + meta["weight_upper"], 1):
                raise ValueError("Interpolation weights do not sum to one")

            full_id = f"tum_{key}"
            curves["cell_id"] = full_id
            curves["chemistry"] = "VTC5A"
            rpt_summary["cell_id"] = full_id
            rpt_summary["chemistry"] = "VTC5A"

            meta.update({
                "cell_id": full_id,
                "chemistry": "VTC5A",
                "aging_domain": domain,
                "operation": operation,
            })
            if "TP" in original.index:
                meta["test_point"] = original["TP"]

            all_curves.append(curves)
            all_metadata.append(meta)
            all_rpts.append(rpt_summary)
            record.update(status="OK", reason="")

            life_text = (
                f">={meta['cycle_life_efc']:.1f} EFC (censored)"
                if meta["cycle_life_censored"]
                else f"{meta['cycle_life_efc']:.1f} EFC"
            )
            print(
                f"OK   {key}: CU{meta['efc50_lower_cu_number']:03d} / "
                f"CU{meta['efc50_upper_cu_number']:03d}, "
                f"EFC={meta['efc50_lower']:.2f}..{meta['efc50_upper']:.2f}, "
                f"life={life_text}, T={meta['temperature_C']:.1f}C, "
                f"Cchg={meta['charge_c_rate']:.1f}, Cdis={meta['discharge_c_rate']:.1f}"
            )
        except (ValueError, KeyError, IndexError, OSError) as error:
            record.update(status="SKIPPED", reason=str(error))
            print(f"SKIP {key}: {error}")
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "vtc5a_processing_log.csv", index=False)
    if not all_curves:
        raise RuntimeError("No cells could be processed; see vtc5a_processing_log.csv")

    curves = pd.concat(all_curves, ignore_index=True)
    metadata = pd.DataFrame(all_metadata)
    rpt_summary = pd.concat(all_rpts, ignore_index=True)

    curves.to_parquet(output_dir / "vtc5a_all_curves.parquet", index=False)
    metadata.to_parquet(output_dir / "vtc5a_all_metadata.parquet", index=False)
    rpt_summary.to_parquet(output_dir / "vtc5a_rpt_summary.parquet", index=False)

    n_skipped = sum(r["status"] == "SKIPPED" for r in records)
    n_excluded = sum(r["status"] == "EXCLUDED" for r in records)
    n_censored = int(metadata["cycle_life_censored"].sum())
    n_gap = int((metadata.get("lifetime_status", pd.Series(dtype=str)) == "censored_data_gap").sum())
    print(
        f"\nSaved {len(metadata)} cells, {len(curves)} curve rows "
        f"({3 * len(metadata)} curves), {len(rpt_summary)} scanned RPTs; "
        f"skipped {n_skipped}, excluded-as-not-applicable {n_excluded}."
    )
    print(f"Censored lifetimes: {n_censored} / {len(metadata)} "
          f"({n_gap} due to late data/format gaps)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
