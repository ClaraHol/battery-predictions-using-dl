# Battery dataset cleaning pipeline

Standardizes the six public cylindrical-cell datasets into one schema, then
applies the paper's three-step cleaning (C-rate > 10C, cycle life < 70,
operational-condition clusters with <= 6 cells).

## Files
- `config_and_cleaning.py` — nominal capacities, rounding rules, cycle-life
  definition (90% of initial capacity), the three-step filter
- `parsers.py` — one parser per raw format (csv/xlsx/mat) -> standard schema
- `process_cell.py` — turns one cell's standard-schema data into a metadata
  row + efc0/efc50 curve snapshots
- `build_dataset.py` — walks `DATA_ROOT`, runs everything, saves outputs

## Running it
```bash
cd battery_pipeline
python3 build_dataset.py
```
Outputs land in `/work3/$USER/battery_datasets/processed/`:
- `cell_metadata_all.csv` — every cell parsed, before cleaning
- `cell_metadata_cleaned.csv` — after all three filters, this is the
  training-ready per-cell table
- `cycle_early_curves.parquet` / `cycle_late_curves.parquet` — raw
  voltage/current/capacity curves at `cycle_number == EFC_EARLY` /
  `EFC_LATE` (set at the top of `build_dataset.py`, default 1 and 50),
  for surviving cells only. Each cell's curve is whatever length that
  cycle naturally has — not padded or resampled to match — so don't
  assume every cell has the same row count when loading this.
- `parse_errors.csv` — written if any files failed to parse, with the
  short error message per file

## Status per dataset — check before trusting the output

| Dataset | Status | Notes |
|---|---|---|
| Sony-VTC6 (cmu) | Ready | Native `cycleNumber` column, tested |
| Sony-VTC5A (tum) | Ready | Native `CycCount`; `Ah` split into charge/discharge by current sign |
| LG-MJ1 (elt) | **Blocked** | Cycle-boundary logic tested and works, but the raw file has no temperature column, and we still don't have real filenames to decode 25C-vs-45C from. `build_dataset.py` currently **skips** this dataset — see the `FILENAME_TO_TEMP` comment block in that file |
| Samsung-25R (tongji) | Ready, one assumption | Temperature/C-rate decoded from filename per `CYX-Y_Z-#N` — the underscore-as-decimal-point convention hasn't been independently verified against a real filename, only against the pattern as described |
| Sandia SNL (snl_lfp / snl_nca / snl_nmc) | Parses correctly, cycle_life pending | Confirmed against a real export (`voltage_timeseries_all_cells.csv`): 3 chemistries (LFP/NCA/NMC, not the 2 originally assumed), voltage-only, already filtered to cycles 1/50/100 across 24 cells. Temperature and C-rate come straight from `cell_id`, not computed. **This file has no current or capacity data**, so `cycle_life` is `NaN` for every SNL cell until a companion capacity/cycle-life file is found - until then these cells are correctly excluded by the cleaning step, not silently kept with a wrong value |

## Cycle numbering caveat
`EFC_EARLY`/`EFC_LATE` in `build_dataset.py` (default 1 and 50) are matched
against `cycle_number` **exactly** — no "closest cycle" fallback. Some
sources are 0-indexed (first real cycle labeled `0`, not `1`), so verify
which convention each source uses before trusting `cycle_early_curves.parquet`
is showing what you think it's showing; a cell simply won't appear in that
file if it has no row at exactly that cycle number, with no separate error
raised (this is expected and normal near a cell's end-of-life for the late
cycle, but worth checking for the early cycle since that would signal an
indexing mismatch rather than a dead cell).

## Farasis test set (separate from the training pipeline above)

`build_farasis_test_set.py` processes the Discovery Learning pouch-cell
test set (123 cells) — separate from `build_dataset.py` because the test
set doesn't get the three-step cleaning (see earlier discussion: all 123
cells pass every filter trivially). Point it at a directory containing
`<cell_id>_dchg_cap_rpt.xlsx` + `<cell_id>_single_parquet_with_soc_and_corrections.parquet`
pairs:
```bash
python3 build_farasis_test_set.py /path/to/farasis/raw/files/
```
Outputs `farasis_cell_metadata.csv` and `farasis_rpt0_curves.parquet`.

**Open limitation:** `cycle_life_rpt_units` is cycle life in *RPT-index*
units ("reached 90% of initial capacity by RPT N"), not equivalent full
cycles. Converting RPT number to actual EFC count would need the parquet's
`total_dischg_thrgh_ah_rounded` column, but that gave physically
implausible values (large negative numbers, inconsistent with a lifetime-
cumulative counter on a ~75 Ah cell) when tested against real cell-019
data — so `cycle_life_efc_units` is left as `None` rather than computed
from a number we can't sanity-check. Only RPT0's curve is extracted for
the same reason: we don't yet know which RPT number corresponds to ~50 EFC.

Also worth knowing: each RPT's raw curve (`parse_farasis_rpt_curve`)
includes both the capacity-check cycles *and* a DC-resistance pulse-test
sub-phase, mixed together — confirmed against real RPT0 data (long
~11,000-row steps alternating charge/discharge at a steady current,
interleaved with much shorter ~10–550-row steps too brief to be genuine
charge/discharge). No reliable way to cleanly separate these via `step_no`
has been confirmed yet.

- LG-MJ1 nominal capacity (3.5 Ah) is a commonly-cited figure, not
  independently confirmed — check it against the max observed
  `discharge_capacity_Ah` in a real low-rate cycle
- C-rate and temperature are averaged over each cell's *entire* history
  (not just efc0), on the assumption the cycling protocol is fixed per
  cell — true for these datasets as far as we've confirmed, but worth a
  spot-check
- `cycle_life` for a cell that never crosses 90%-of-initial capacity
  within the recorded data returns its last recorded cycle as a censored
  lower bound, not a confirmed cycle life — these are flagged nowhere
  automatically yet, so if you need to distinguish censored from
  confirmed lifetimes, that's a small addition to `compute_cycle_life()`

## Directory layout this expects
`build_dataset.py` assumes raw files live under:
```
/work3/$USER/battery_datasets/
    sony_vtc6/**/*.csv
    sony_vtc5a_raw/**/*.mat        (excluding anything with "calendar" in the path)
    lg_mj1/**/*.csv
    samsung_25r/**/*.csv
    sandia_snl/**/*.xlsx
```
If a dataset prints "0 files found" when you run it, your actual extracted
folder structure doesn't match these glob patterns — adjust the `*_dir`
variable at the top of the matching block in `build_dataset.py`.
