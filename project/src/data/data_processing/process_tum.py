from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# CONFIGURATION
# ============================================================

# Root folder containing:
#   CYC_Cyclic/
#   CYC_Dynamic/
DATA_ROOT = Path(r"/work3/claho/battery_datasets/sony_vtc5a")

# Where combined files should be stored
OUTPUT_DIR = DATA_ROOT / "combined_cells"

# Fields to extract from Dataset
FIELDS = [
    "Time",
    "U",
    "I",
    "Ah",
    "Wh",
    "T1",
    "DateTime",
]

# Output type:
# "parquet" is strongly recommended for large datasets.
# You can change this to "csv" if desired.
OUTPUT_FORMAT = "csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_cell_id(filename):
    """
    Extract an Internal ID such as BW-VTC-247 from a filename.
    """
    match = re.search(r"(BW-VTC-\d+)", filename)
    if match:
        return match.group(1)

    return None


def get_cycle_number(path):
    """
    Determine cycling interval from folder names such as:
        Cyc01
        Cyc2
        Cyc18

    Returns a large number if no cycle number can be found,
    so unidentified files are placed at the end.
    """

    # Look through all parts of the path
    for part in path.parts:
        match = re.fullmatch(r"Cyc(\d+)", part, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))

    # Fallback: try filename
    match = re.search(r"Cyc(\d+)", path.name, flags=re.IGNORECASE)

    if match:
        return int(match.group(1))

    return 999999


def flatten_array(x):
    """
    Convert MATLAB arrays to flat NumPy arrays.
    """
    if x is None:
        return None

    x = np.asarray(x)

    return x.squeeze().reshape(-1)


def convert_datetime(values):
    """
    Convert DateTime values to pandas datetime.

    Handles:
    1. MATLAB serial datenumbers
    2. Strings such as '23.04.2020 09:10:29'
    """

    values = np.asarray(values).reshape(-1)

    # --------------------------------------------------------
    # Case 1: strings / object arrays
    # --------------------------------------------------------
    if values.dtype.kind in {"U", "S", "O"}:
        try:
            return pd.to_datetime(
                values.astype(str),
                format="%d.%m.%Y %H:%M:%S",
                errors="coerce"
            )
        except Exception:
            return pd.to_datetime(
                values.astype(str),
                dayfirst=True,
                errors="coerce"
            )

    # --------------------------------------------------------
    # Case 2: numeric MATLAB datenum
    # --------------------------------------------------------
    try:
        numeric_values = values.astype(float)

        return pd.to_datetime(
            numeric_values - 719529,
            unit="D",
            origin="unix",
            errors="coerce"
        )

    except Exception:
        return pd.to_datetime(
            values,
            errors="coerce"
        )


def load_dataset_from_mat(mat_file):
    """
    Load the 'Dataset' struct from a MATLAB .mat file.

    This function works with ordinary MATLAB MAT files
    supported by scipy.io.loadmat.
    """

    data = loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False
    )

    if "Dataset" not in data:
        raise KeyError(
            f"'Dataset' struct not found in {mat_file}"
        )

    dataset = data["Dataset"]

    result = {}

    for field in FIELDS:
        if hasattr(dataset, field):
            result[field] = flatten_array(
                getattr(dataset, field)
            )

    return result


def dataset_to_dataframe(dataset, mat_file, cycle_number, cell_id):
    """
    Convert fields from one MATLAB Dataset struct into a DataFrame.
    """

    # Determine number of samples from available fields
    lengths = []

    for field, values in dataset.items():
        if values is not None:
            lengths.append(len(values))

    if not lengths:
        raise ValueError(f"No usable data found in {mat_file}")

    # Usually all fields should have the same length.
    # Use the most common length in case some metadata field differs.
    sample_count = max(set(lengths), key=lengths.count)

    df = pd.DataFrame()

    # ----------------------------------------------------------------
    # Add identifying metadata
    # ----------------------------------------------------------------

    df["cell_id"] = [cell_id] * sample_count
    df["cycle_interval"] = [cycle_number] * sample_count
    df["source_file"] = [mat_file.name] * sample_count

    # ----------------------------------------------------------------
    # Add measurement fields
    # ----------------------------------------------------------------

    for field, values in dataset.items():

        if values is None:
            continue

        # Only use fields that correspond to the time-series length
        if len(values) != sample_count:
            print(
                f"  Warning: skipping {field} in {mat_file.name}: "
                f"length {len(values)} != {sample_count}"
            )
            continue

        if field == "DateTime":
            try:
                df["DateTime"] = convert_datetime(values)
            except Exception as exc:
                print(
                    f"  Warning: could not convert DateTime "
                    f"in {mat_file.name}: {exc}"
                )
        else:
            df[field] = values

    # Rename local quantities so their meaning is explicit
    rename_columns = {
        "Time": "Time_local_h",
        "Ah": "Ah_local",
        "Wh": "Wh_local",
    }

    df = df.rename(columns=rename_columns)

    return df


# ============================================================
# DISCOVER ALL CYCLING FILES
# ============================================================

def find_cycling_files(data_root):
    """
    Recursively find all .mat files under:
        CYC_Cyclic
        CYC_Dynamic
    """

    cycle_roots = [
        data_root / "CYC_Cyclic",
        data_root / "CYC_Dynamic",
    ]

    files = []

    for root in cycle_roots:

        if not root.exists():
            print(f"Folder not found, skipping: {root}")
            continue

        found = list(root.rglob("*.mat"))

        print(f"Found {len(found)} files in {root.name}")

        files.extend(found)

    return files


# ============================================================
# GROUP FILES BY CELL
# ============================================================

def group_files_by_cell(files):
    grouped = {}

    for file in files:

        cell_id = get_cell_id(file.name)

        if cell_id is None:
            print(
                f"Warning: could not determine cell ID from "
                f"{file.name}. Skipping."
            )
            continue

        grouped.setdefault(cell_id, []).append(file)

    return grouped


# ============================================================
# COMBINE ONE CELL
# ============================================================

def combine_cell(cell_id, files):
    """
    Load and concatenate all cycling files belonging to one cell.
    """

    # Sort first by cycling interval, then by filename
    files = sorted(
        files,
        key=lambda x: (
            get_cycle_number(x),
            x.name
        )
    )

    frames = []

    print(f"\nProcessing {cell_id}")
    print(f"  {len(files)} cycling files")

    for mat_file in files:

        cycle_number = get_cycle_number(mat_file)

        print(
            f"  Cyc {cycle_number:>3}: "
            f"{mat_file.name}"
        )

        try:
            dataset = load_dataset_from_mat(mat_file)

            df = dataset_to_dataframe(
                dataset=dataset,
                mat_file=mat_file,
                cycle_number=cycle_number,
                cell_id=cell_id,
            )

            frames.append(df)

        except Exception as exc:
            print(
                f"  ERROR loading {mat_file}: {exc}"
            )

    if not frames:
        return None

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    if "DateTime" in combined.columns:

        # Sorting by actual recorded date/time is safer than relying
        # purely on Cyc01, Cyc02, ...
        combined = combined.sort_values(
            ["DateTime", "cycle_interval"],
            na_position="last"
        ).reset_index(drop=True)

        # ====================================================
        # GLOBAL CONTINUOUS TIME
        # ====================================================

        valid_times = combined["DateTime"].dropna()

        if len(valid_times) > 0:

            t0 = valid_times.iloc[0]

            combined["Time_global_h"] = (
                combined["DateTime"] - t0
            ).dt.total_seconds() / 3600

    else:

        print(
            f"  Warning: no DateTime found for {cell_id}. "
            "Time_global_h cannot be constructed from real time."
        )

    # ========================================================
    # GLOBAL ABSOLUTE Ah THROUGHPUT
    # ========================================================

    #
    # Rather than concatenating Ah_local directly, calculate
    # throughput from the current and elapsed time.
    #
    # This is robust to Ah resetting at each MATLAB file.
    #

    if (
        "I" in combined.columns
        and "DateTime" in combined.columns
    ):

        dt_h = (
            combined["DateTime"]
            .diff()
            .dt.total_seconds()
            / 3600
        )

        # Prevent huge integration jumps across gaps between files.
        #
        # We calculate current integration separately within
        # each original cycling MAT file.
        same_file = (
            combined["source_file"]
            == combined["source_file"].shift(1)
        )

        dt_h = dt_h.where(same_file, 0)

        # Trapezoidal integration
        current_previous = combined["I"].shift(1)

        delta_ah = (
            0.5
            * (
                combined["I"].abs()
                + current_previous.abs()
            )
            * dt_h
        )

        delta_ah = delta_ah.fillna(0)

        combined["Ah_throughput_global"] = (
            delta_ah.cumsum()
        )

        # Equivalent Full Cycles based on nominal capacity = 2.5 Ah.
        #
        # Full charge + full discharge = 2 * 2.5 Ah = 5 Ah
        # absolute bidirectional throughput.
        combined["EFC_estimated"] = (
            combined["Ah_throughput_global"] / 5.0
        )

    return combined


# ============================================================
# SAVE
# ============================================================

def save_cell(df, cell_id, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FORMAT.lower() == "parquet":

        output_file = output_dir / f"{cell_id}_cycling.parquet"

        df.to_parquet(
            output_file,
            index=False
        )

    elif OUTPUT_FORMAT.lower() == "csv":

        output_file = output_dir / f"{cell_id}_cycling.csv"

        df.to_csv(
            output_file,
            index=False
        )

    else:
        raise ValueError(
            "OUTPUT_FORMAT must be 'parquet' or 'csv'"
        )

    print(
        f"  Saved {len(df):,} rows -> {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BAWAII cycling-data combiner")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Find files
    # --------------------------------------------------------

    all_files = find_cycling_files(DATA_ROOT)

    print(
        f"\nTotal cycling .mat files found: "
        f"{len(all_files)}"
    )

    # --------------------------------------------------------
    # 2. Group by cell
    # --------------------------------------------------------

    grouped = group_files_by_cell(all_files)

    print(
        f"Unique cells found: "
        f"{len(grouped)}"
    )

    # --------------------------------------------------------
    # 3. Combine each cell
    # --------------------------------------------------------

    for cell_id in sorted(grouped):

        combined = combine_cell(
            cell_id,
            grouped[cell_id]
        )

        if combined is None:
            print(
                f"No usable data for {cell_id}"
            )
            continue

        save_cell(
            combined,
            cell_id,
            OUTPUT_DIR
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()