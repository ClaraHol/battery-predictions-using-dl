"""
Per-format parsers. Each function takes a path (and any format-specific
extra info, e.g. a filename to decode) and returns a pandas DataFrame in
the STANDARD_COLUMNS schema defined in config_and_cleaning.py, ready to
hand to build_cell_record().

STATUS PER PARSER (check before trusting output):
    parse_cmu_vtc6      READY   - cycleNumber column is native, no inference needed
    parse_vtc5a_mat      READY   - CycCount is native; T1 is measured temperature
    parse_lgmj1          READY   - cycle boundaries derived from Discharge_Capacity_Ah
                                    resets (verified against real sample data).
                                    ** Temperature is NOT in the raw file - must be
                                    passed in explicitly (see NEEDS INPUT below). **
    parse_samsung_25r    READY   - cycle number column is native; C-rate/temp decoded
                                    from filename per the CYX-Y_Z-#N convention
    parse_snl_xlsx       *** NOT VERIFIED ***
                                    We have not yet seen a correct Sandia SNL cycling
                                    file (the first download turned out to be a nail-
                                    penetration abuse-test file, not cycling data).
                                    This function is written against the commonly
                                    documented Battery Archive schema but WILL likely
                                    need column-name fixes once you get the real file.
                                    It raises a clear error rather than silently
                                    parsing garbage if expected columns are missing.

NEEDS INPUT (can't be inferred from the files alone):
    - LG-MJ1 temperature: the 4TU dataset was cycled at 25C and 45C, but the raw
      csv has no temperature column. You'll need to pass temp_C_nameplate based
      on the filename or folder the file came from (once you can see real
      filenames - we haven't yet).
"""

import re
import numpy as np
import pandas as pd

STANDARD_COLUMNS = [
    "time_s", "voltage_V", "current_A", "temp_C",
    "cycle_number", "charge_capacity_Ah", "discharge_capacity_Ah",
]


# ---------------------------------------------------------------------------
# Sony-VTC6 (CMU / KiltHub)
# ---------------------------------------------------------------------------
def parse_cmu_vtc6(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "time_s": df["time_s"],
        "voltage_V": df["Ecell_V"],
        "current_A": df["I_mA"] / 1000.0,
        "temp_C": df["Temperature__C"],
        "cycle_number": df["cycleNumber"].astype(int),
        "charge_capacity_Ah": df["QCharge_mA_h"] / 1000.0,
        "discharge_capacity_Ah": df["QDischarge_mA_h"] / 1000.0,
    })
    return out[STANDARD_COLUMNS]


# ---------------------------------------------------------------------------
# Sony-VTC5A (TUM / mediaTUM) - MATLAB v5 struct
# ---------------------------------------------------------------------------
def parse_vtc5a_mat(mat_path: str) -> pd.DataFrame:
    import scipy.io as sio

    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    ds = mat["Dataset"]

    # The high-resolution arrays (Time, U, I, Ah, T1, CycCount, ...) are all
    # the same length; the low-res setpoint arrays (DateTime, AhChSet, ...)
    # are a different, shorter length and are NOT used here.
    n = len(ds.Time)
    out = pd.DataFrame({
        "time_s": np.asarray(ds.Time, dtype=float),
        "voltage_V": np.asarray(ds.U, dtype=float),
        "current_A": np.asarray(ds.I, dtype=float),
        "temp_C": np.asarray(ds.T1, dtype=float),
        "cycle_number": np.asarray(ds.CycCount, dtype=int),
    })

    # Ah is a signed cumulative counter in this dataset (not separate
    # charge/discharge columns) - split by sign of current to reconstruct
    # per-cycle charge/discharge capacity the same way as the other sources.
    ah = np.asarray(ds.Ah, dtype=float)
    is_charge = out["current_A"] > 0
    out["charge_capacity_Ah"] = np.where(is_charge, ah, np.nan)
    out["discharge_capacity_Ah"] = np.where(~is_charge, -ah, np.nan)
    # forward-fill within each cycle so both columns are always defined
    out["charge_capacity_Ah"] = out.groupby("cycle_number")["charge_capacity_Ah"].ffill().fillna(0.0)
    out["discharge_capacity_Ah"] = out.groupby("cycle_number")["discharge_capacity_Ah"].ffill().fillna(0.0)

    return out[STANDARD_COLUMNS]


# ---------------------------------------------------------------------------
# LG-MJ1 (4TU / VITO) - no native cycle number, no temperature column
# ---------------------------------------------------------------------------
def parse_lgmj1(csv_path: str, temp_C_nameplate: float) -> pd.DataFrame:
    """
    temp_C_nameplate: REQUIRED. The 4TU dataset was cycled at a fixed 25C or
    45C per cell/folder, but that value isn't in the raw csv - pass it in
    based on the filename/folder convention once you can see real filenames.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # first column is an unlabeled timestamp column in the sample we saw
    ts_col = df.columns[0]

    # A new cycle starts wherever Discharge_Capacity_Ah resets back near
    # zero after having been nonzero - i.e. a sharp negative jump, not the
    # gentle staying-at-zero during a rest step. Verified against a real
    # sample: this fires exactly once per discharge->charge transition.
    dcap = df["Discharge_Capacity_Ah"]
    reset = dcap.diff() < -1e-4
    cycle_number = reset.cumsum()

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df["Total_Time_Seconds"], errors="coerce"),
        "voltage_V": df["Voltage_V"],
        "current_A": df["Current_A"],
        "temp_C": temp_C_nameplate,
        "cycle_number": cycle_number.astype(int),
        "charge_capacity_Ah": df["Charge_Capacity_Ah"],
        "discharge_capacity_Ah": df["Discharge_Capacity_Ah"],
    })
    return out[STANDARD_COLUMNS]


# ---------------------------------------------------------------------------
# Samsung-25R (Zhu et al. / Zenodo) - condition encoded in filename
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(r"CY(?P<temp>-?\d+)-(?P<charge>[\d_]+)_(?P<discharge>[\d_]+)-#(?P<tag>\d+)")


def parse_samsung25r_filename(filename: str):
    """
    Decodes 'CYX-Y_Z-#N' -> temperature X, charge rate Y, discharge rate Z, cell tag N.
    Y/Z use underscore as decimal point (e.g. '0_5' -> 0.5C) based on the
    convention as described; VERIFY this against a couple of real filenames
    before trusting it at scale, since we haven't independently confirmed
    the decimal convention.
    """
    m = FILENAME_RE.search(filename)
    if not m:
        raise ValueError(f"Filename '{filename}' doesn't match expected CYX-Y_Z-#N pattern")
    temp_C = float(m.group("temp"))
    charge_c = float(m.group("charge").replace("_", "."))
    discharge_c = float(m.group("discharge").replace("_", "."))
    return temp_C, charge_c, discharge_c


def parse_samsung25r(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "time_s": df["time/s"],
        "voltage_V": df["Ecell/V"],
        "current_A": df["<I>/mA"] / 1000.0,
        "temp_C": np.nan,  # not logged per-row; comes from filename instead
        "cycle_number": df["cycle number"].astype(int),
        "charge_capacity_Ah": df["Q charge/mA.h"] / 1000.0,
        "discharge_capacity_Ah": df["Q discharge/mA.h"] / 1000.0,
    })
    return out[STANDARD_COLUMNS]


SNL_CELL_ID_RE = re.compile(
    r"SNL_18650_(?P<chem>LFP|NCA|NMC)_(?P<temp>-?\d+)C_"
    r"(?P<dod_low>\d+)-(?P<dod_high>\d+)_"
    r"(?P<charge>[\d.]+)/(?P<discharge>[\d.]+)C_(?P<tag>[a-z]+)"
)

SNL_CHEM_TO_SOURCE = {"LFP": "snl_lfp", "NCA": "snl_nca", "NMC": "snl_nmc"}


def parse_snl_cell_id(cell_id: str):
    """
    Decodes 'SNL_18650_LFP_25C_20-80_0.5/0.5C_a' into its components.
    Returns (source_dataset, temp_C, dod_low_pct, dod_high_pct, charge_c, discharge_c, tag).
    """
    m = SNL_CELL_ID_RE.search(cell_id)
    if not m:
        raise ValueError(f"cell_id '{cell_id}' doesn't match expected SNL naming pattern")
    return (
        SNL_CHEM_TO_SOURCE[m.group("chem")],
        float(m.group("temp")),
        int(m.group("dod_low")),
        int(m.group("dod_high")),
        float(m.group("charge")),
        float(m.group("discharge")),
        m.group("tag"),
    )


def parse_snl_voltage_timeseries(csv_path: str) -> dict:
    """
    Parses the real SNL voltage-only export (columns: cell_id, cycle,
    cycle_time, v, row_m, label, row) covering many cells in one file.

    Unlike the other parsers, this does NOT return a single standard-schema
    DataFrame, because this file has no current or capacity data at all -
    current_A/charge_capacity_Ah/discharge_capacity_Ah are meaningless here
    and filling them with NaN would silently break compute_c_rate() and
    compute_cycle_life() if run through process_cell() as-is.

    Returns a dict: {cell_id: {"voltage_curves": {cycle_number: DataFrame},
                                "temp_C": ..., "charge_c_rate": ...,
                                "discharge_c_rate": ..., "dod_low_pct": ...,
                                "dod_high_pct": ..., "source_dataset": ...}}

    charge_c_rate / discharge_c_rate / temp_C come directly from the
    cell_id naming convention (confirmed against Sandia's SNL study
    documentation), NOT computed from data - there's no current column to
    compute them from in this file.

    cycle_life is NOT included - this file has no capacity data, so cycle
    life can't be determined from it alone. A separate capacity/cycle-life
    summary file for these same cells is needed to complete the metadata.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for cell_id, group in df.groupby("cell_id"):
        source_dataset, temp_C, dod_low, dod_high, charge_c, discharge_c, tag = parse_snl_cell_id(cell_id)
        curves = {}
        for cycle_number, cycle_group in group.groupby("cycle"):
            curves[int(cycle_number)] = pd.DataFrame({
                "time_s": cycle_group["cycle_time"].values,
                "voltage_V": cycle_group["v"].values,
                "cycle_number": int(cycle_number),
                "cell_id": cell_id,
            })
        out[cell_id] = {
            "voltage_curves": curves,
            "temp_C": temp_C,
            "charge_c_rate": charge_c,
            "discharge_c_rate": discharge_c,
            "dod_low_pct": dod_low,
            "dod_high_pct": dod_high,
            "source_dataset": source_dataset,
        }
    return out


def parse_snl_cycle_summary(csv_path: str) -> dict:
    """
    Parses the companion capacity-summary export (columns: cell_id,
    cycle_index, test_time, ah_c, ah_d, e_c, e_d, v_c_mean, v_d_mean,
    v_max, v_min) - one row per cycle per cell, spanning each cell's full
    life. This is what feeds config_and_cleaning.compute_snl_cycle_life();
    it does NOT include voltage curves (use parse_snl_voltage_timeseries
    for those).

    Returns {cell_id: DataFrame[cycle_index, ah_d]} - only the columns
    actually needed for cycle-life computation are kept.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for cell_id, group in df.groupby("cell_id"):
        out[cell_id] = group[["cycle_index", "ah_d"]].sort_values("cycle_index").reset_index(drop=True)
    return out


 
def parse_farasis_rpt_metadata(xlsx_path: str) -> pd.DataFrame:
    """
    Parses a '<cell>_dchg_cap_rpt.xlsx' file. Confirmed to come in at least
    two different schemas across cells:
 
    Schema A (e.g. cell 019): one clean row per RPT, column 'discharge_cap'.
 
    Schema B (e.g. cell 034): MULTIPLE rows per RPT (typically 2-3, from
    repeated/partial discharge measurements within that RPT's capacity
    check), column 'discharge_ah' instead of 'discharge_cap', and an
    explicit boolean 'is_outlier' column flagging which rows to exclude
    (e.g. a row at roughly half the others' value - probably a different
    sub-test, not the main capacity-check reading). No cluster-detection
    heuristic needed here, unlike the SNL case - just trust is_outlier.
 
    Note: at least one cell (034) has an RPT number that actually spans
    TWO separate checkup events ~16 days apart, sharing the same RPT
    label - likely a data-labeling quirk upstream. This function averages
    all non-outlier rows within an RPT number, which slightly blurs such
    cases together rather than treating them as separate checkups. Worth
    knowing if you need RPT-level dates to line up precisely with a
    calendar timeline.
 
    Returns a DataFrame indexed by integer RPT number with a
    discharge_cap column (normalized name regardless of source schema).
    """
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={"RPT Number": "rpt_number"})
    df = df.rename(columns={"rpt": "rpt_number"})
    df["rpt_number"] = df["rpt_number"].str.lower().str.replace("rpt", "", regex=False).astype(int)
 
    if "is_outlier" in df.columns:
        df = df[~df["is_outlier"]]
 
    if "discharge_ah" in df.columns and "discharge_cap" not in df.columns:
        df = df.rename(columns={"discharge_ah": "discharge_cap"})
    elif "discharge_cap" not in df.columns:
        raise KeyError(
            f"Neither 'discharge_cap' nor 'discharge_ah' found in {xlsx_path}. "
            f"Actual columns: {list(df.columns)}"
        )
 
    # average multiple readings per RPT (schema B); a no-op for schema A,
    # which already has exactly one row per RPT
    result = df.groupby("rpt_number")["discharge_cap"].mean().to_frame()
    return result.sort_index()
 
 
def parse_farasis_rpt_curve(parquet_path: str, rpt_number: int, batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    Streams through the (typically huge, ~100-200MB / tens of millions of
    rows) raw parquet and pulls out every row belonging to one RPT number.
    Does NOT load the whole file into memory - only rows matching rpt_number
    are kept as batches are scanned, avoiding the earlier OOM issue.
 
    Returns the raw multi-stage curve for that RPT: every charge/discharge/
    rest step it contains, INCLUDING the DC-resistance pulse-test sub-phase
    (short-duration steps mixed in with the longer capacity-check cycles -
    confirmed against real RPT0 data: long ~11000-row steps alternating
    charge/discharge at a consistent current, interleaved with much
    shorter ~10-550 row steps that are too brief to be genuine
    charge/discharge and are almost certainly the DCR pulse test). We
    haven't confirmed a reliable way to separate these sub-phases purely
    from step_no, so if you need just the capacity-check portion, filter
    further downstream using step_no once/if that's pinned down - this
    function returns everything tagged with that RPT number as-is.
 
    Output columns match this pipeline's standard schema as closely as
    possible, plus soc/step_no which are Farasis-specific extras:
    time_s, voltage_V, current_A, temp_C, soc, step_no, cycle_number
    (= rpt_number, for compatibility with code elsewhere expecting a
    'cycle_number' column).
    """
    import pyarrow.parquet as pq
 
    pf = pq.ParquetFile(parquet_path)
    chunks = []
    for batch in pf.iter_batches(
        columns=["del_time", "current (A)", "voltage (V)", "step_no", "temp", "rpt", "soc"],
        batch_size=batch_size,
    ):
        df = batch.to_pandas()
        sub = df[df["rpt"] == rpt_number]
        if len(sub):
            chunks.append(sub)
 
    if not chunks:
        raise ValueError(f"No rows found for rpt == {rpt_number} in {parquet_path}")
 
    raw = pd.concat(chunks, ignore_index=True)
    out = pd.DataFrame({
        "time_s": raw["del_time"],
        "voltage_V": raw["voltage (V)"],
        "current_A": raw["current (A)"],
        "temp_C": raw["temp"],
        "soc": raw["soc"],
        "step_no": raw["step_no"],
        "cycle_number": rpt_number,
    })
    return out.sort_values("time_s").reset_index(drop=True)
 
 
def build_farasis_cell_record(cell_id: str, rpt_metadata: pd.DataFrame):
    """
    cycle_life is computed in RPT-index units (NOT equivalent full cycles -
    we don't have a trustworthy way to convert RPT number to EFC count yet;
    the parquet's total_dischg_thrgh_ah_rounded column gave physically
    implausible values (huge negatives) when we tried this, so it's left
    as an open problem rather than guessed at). Treat cycle_life_rpt_units
    as "reached 90% of initial capacity by RPT N", not "by cycle N".
    """
    from config_and_cleaning import compute_cycle_life
 
    cap_series = rpt_metadata["discharge_cap"]
    cycle_life_in_rpt_units = compute_cycle_life(cap_series)
    return {
        "cell_id": cell_id,
        "nominal_capacity_Ah": float(cap_series.iloc[0]),  # rpt0's discharge_cap
        "cycle_life_rpt_units": cycle_life_in_rpt_units,
        "cycle_life_efc_units": None,  # TBD - RPT-to-EFC mapping still unresolved
    }


# ---------------------------------------------------------------------------
# Sandia SNL (A123-M1A, LG-HG2) - *** SUPERSEDED - see parse_snl_voltage_timeseries
# above, which is written against a real confirmed file. This function below
# was written from documentation before any real SNL file was available and
# assumed a per-cell csv/xlsx with current+capacity columns, which the real
# export does NOT have. Kept here in case a different SNL export (e.g. a
# full cycling log with current/capacity) turns up later and matches this
# shape - otherwise, prefer parse_snl_voltage_timeseries. ***
# ---------------------------------------------------------------------------
def parse_snl_xlsx(xlsx_path: str) -> pd.DataFrame:
    """
    Battery Archive's published per-cell timeseries export commonly uses
    columns like: 'Cycle_Index', 'Test_Time (s)', 'Voltage (V)',
    'Current (A)', 'Cell_Temperature (C)', 'Charge_Capacity (Ah)',
    'Discharge_Capacity (Ah)' - but we have NOT confirmed this against your
    actual downloaded file (the one we inspected earlier was a mechanical
    abuse test, not cycling data). This function will raise a clear
    KeyError listing available columns if the guessed names are wrong -
    when that happens, send me the real column list and I'll fix this
    function rather than you having to guess at it.
    """
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    expected = {
        "Cycle_Index": "cycle_number",
        "Test_Time (s)": "time_s",
        "Voltage (V)": "voltage_V",
        "Current (A)": "current_A",
        "Cell_Temperature (C)": "temp_C",
        "Charge_Capacity (Ah)": "charge_capacity_Ah",
        "Discharge_Capacity (Ah)": "discharge_capacity_Ah",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"parse_snl_xlsx: expected columns {missing} not found in {xlsx_path}.\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"This parser was written from documentation, not a verified real file - "
            f"please share these actual column names so this can be fixed."
        )

    out = df.rename(columns=expected)[list(expected.values())]
    out["cycle_number"] = out["cycle_number"].astype(int)
    return out[STANDARD_COLUMNS]
