import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data/vtc6")

CURVES_FILE = DATA_DIR / "vtc6_all_curves.parquet"
METADATA_FILE = DATA_DIR / "vtc6_all_metadata.parquet"
RPT_FILE = DATA_DIR / "vtc6_rpt_summary.parquet"

EXPECTED_CURVES = {"efc0", "efc50_lower", "efc50_upper"}
EXPECTED_TEMPERATURES_C = {20.0, 25.0, 30.0, 35.0}
EXPECTED_CHARGE_PROTOCOL_C_RATES = {0.5, 1.0, 1.5}
EXPECTED_CHARGE_VOLTAGES_V = {4.0, 4.1, 4.2}
LIFETIME_SOH_THRESHOLD = 0.90


REQUIRED_METADATA_COLUMNS = {
    "cell_id",
    "chemistry",
    "efc0",
    "efc50_lower",
    "efc50_upper",
    "weight_lower",
    "weight_upper",
    "direction",
    "n_rpts",
    "cycle_life_efc",
    "cycle_life_censored",
    "cycle_life_soh_threshold",
    "initial_rpt_capacity_Ah",
    "last_rpt_capacity_Ah",
    "last_rpt_soh",
    "last_rpt_efc",
    "temperature_C",
    "ambient_temperature_raw_C",
    "temperature_source",
    "charge_c_rate",
    "charge_c_rate_raw",
    "discharge_c_rate",
    "discharge_c_rate_raw",
    "charge_cc_c_rate_protocol",
    "charge_voltage_V",
    "charge_c_rate_bins_used",
    "discharge_c_rate_bins_used",
    "n_condition_cycles",
    "surface_temperature_mean_C",
    "surface_temperature_p05_C",
    "surface_temperature_max_C",
    "n_impedance_rpts",
    "rpt_count_matches_impedance",
}

REQUIRED_RPT_COLUMNS = {
    "cell_id",
    "chemistry",
    "rpt_number",
    "rpt_efc",
    "capacity_Ah",
    "capacity_counter_Ah",
    "soh",
}


def round_to_step(value, step):
    if not np.isfinite(value):
        return np.nan
    return float(step * np.floor(value / step + 0.5))


def recompute_lifetime(rpt_cell, threshold):
    rpt_cell = rpt_cell.sort_values("rpt_efc")
    efc = rpt_cell["rpt_efc"].to_numpy(dtype=float)
    capacity = rpt_cell["capacity_Ah"].to_numpy(dtype=float)

    if len(efc) < 2:
        raise ValueError("Need at least two RPTs")
    if not np.isfinite(efc).all() or not np.isfinite(capacity).all():
        raise ValueError("Non-finite RPT EFC/capacity")
    if np.any(np.diff(efc) <= 0):
        raise ValueError("RPT EFC is not strictly increasing")
    if capacity[0] <= 0:
        raise ValueError("Initial RPT capacity is non-positive")

    soh = capacity / capacity[0]
    if soh[0] <= threshold:
        raise ValueError("Initial RPT is already below lifetime threshold")

    for j in range(1, len(soh)):
        if soh[j] <= threshold and soh[j - 1] > threshold:
            x0, x1 = efc[j - 1], efc[j]
            y0, y1 = soh[j - 1], soh[j]
            if y1 == y0:
                crossing = x1
            else:
                crossing = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            return float(crossing), False, soh

    return float(efc[-1]), True, soh


def add_problem(problems, cell_id, issue):
    problems.append((str(cell_id), issue))


def add_warning(warnings, cell_id, issue):
    warnings.append((str(cell_id), issue))


curves = pd.read_parquet(CURVES_FILE)
metadata = pd.read_parquet(METADATA_FILE)

if not RPT_FILE.is_file():
    raise FileNotFoundError(
        f"Missing {RPT_FILE}. The updated verifier requires vtc6_rpt_summary.parquet."
    )
rpts = pd.read_parquet(RPT_FILE)

print("Metadata header:")
print(metadata.head().to_string(index=False))
print("\nRPT summary header:")
print(rpts.head().to_string(index=False))

problems = []
warnings = []


# -----------------------------------------------------------------------------
# File/schema checks
# -----------------------------------------------------------------------------
missing_metadata_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
if missing_metadata_columns:
    add_problem(
        problems,
        "GLOBAL",
        f"Missing metadata columns: {sorted(missing_metadata_columns)}",
    )

missing_rpt_columns = REQUIRED_RPT_COLUMNS - set(rpts.columns)
if missing_rpt_columns:
    add_problem(
        problems,
        "GLOBAL",
        f"Missing RPT-summary columns: {sorted(missing_rpt_columns)}",
    )

if metadata["cell_id"].duplicated().any():
    duplicate_ids = metadata.loc[metadata["cell_id"].duplicated(), "cell_id"].tolist()
    add_problem(problems, "GLOBAL", f"Duplicate metadata cell IDs: {duplicate_ids}")

curve_cells = set(curves["cell_id"].astype(str))
metadata_cells = set(metadata["cell_id"].astype(str))
rpt_cells = set(rpts["cell_id"].astype(str))

for cell_id in sorted(curve_cells - metadata_cells):
    add_problem(problems, cell_id, "Present in curves but missing from metadata")
for cell_id in sorted(metadata_cells - curve_cells):
    add_problem(problems, cell_id, "Present in metadata but missing from curves")
for cell_id in sorted(metadata_cells - rpt_cells):
    add_problem(problems, cell_id, "Present in metadata but missing from RPT summary")
for cell_id in sorted(rpt_cells - metadata_cells):
    add_problem(problems, cell_id, "Present in RPT summary but missing from metadata")


# -----------------------------------------------------------------------------
# Curve checks
# -----------------------------------------------------------------------------
for cell_id, cell in curves.groupby("cell_id", sort=True):
    missing = EXPECTED_CURVES - set(cell["curve_type"])
    extra = set(cell["curve_type"]) - EXPECTED_CURVES

    if missing:
        add_problem(problems, cell_id, f"Missing curves: {sorted(missing)}")
    if extra:
        add_problem(problems, cell_id, f"Unexpected curves: {sorted(extra)}")

    for curve_type, df in cell.groupby("curve_type", sort=True):
        df = df.sort_values("time_s")
        t = df["time_s"].to_numpy(dtype=float)
        v = df["voltage_V"].to_numpy(dtype=float)
        i = df["current_A"].to_numpy(dtype=float)

        issues = []
        if len(t) == 0:
            issues.append("Empty curve")
            add_problem(problems, cell_id, f"{curve_type}: Empty curve")
            continue

        duration_h = (t[-1] - t[0]) / 3600.0
        current_med = float(np.median(i))

        if len(t) < 2000:
            issues.append("Too few samples")
        if not (3 <= duration_h <= 6):
            issues.append("Unusual duration")
        if v[0] < 3.7:
            issues.append("Low starting voltage")
        if v[-1] > 2.65:
            issues.append("High ending voltage")
        if abs(current_med - 0.61) > 0.08:
            issues.append("Unexpected current")
        if not (
            np.isfinite(t).all()
            and np.isfinite(v).all()
            and np.isfinite(i).all()
        ):
            issues.append("Non-finite values")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            issues.append("Non-increasing time")
        if len(t) > 1 and np.any(np.diff(t) > 120):
            issues.append("Large time gap")

        for issue in issues:
            add_problem(problems, cell_id, f"{curve_type}: {issue}")

        status = "OK" if not issues else "CHECK"
        print(
            f"{cell_id:12} {curve_type:13} "
            f"N={len(t):6} "
            f"T={duration_h:5.2f}h "
            f"V={v[0]:.2f}->{v[-1]:.2f} "
            f"I={current_med:.3f}A "
            f"{status}"
        )


# -----------------------------------------------------------------------------
# Metadata + RPT checks
# -----------------------------------------------------------------------------
metadata_by_cell = metadata.set_index("cell_id", drop=False)

for row in metadata.itertuples(index=False):
    cell_id = row.cell_id

    # Basic identity/provenance.
    if row.chemistry != "VTC6":
        add_problem(problems, cell_id, f"Unexpected chemistry: {row.chemistry}")
    if row.direction != "discharge":
        add_problem(problems, cell_id, f"Unexpected direction: {row.direction}")
    if row.temperature_source != "vtc6_protocol":
        add_problem(
            problems,
            cell_id,
            f"Unexpected temperature source: {row.temperature_source}",
        )

    # EFC50 interpolation.
    interpolation_values = np.array(
        [
            row.efc0,
            row.efc50_lower,
            row.efc50_upper,
            row.weight_lower,
            row.weight_upper,
        ],
        dtype=float,
    )
    if not np.isfinite(interpolation_values).all():
        add_problem(problems, cell_id, "Non-finite EFC interpolation metadata")
    else:
        if not row.efc50_lower < 50 < row.efc50_upper:
            add_problem(problems, cell_id, "EFC50 is not genuinely bracketed")
        if not (0 <= row.weight_lower <= 1 and 0 <= row.weight_upper <= 1):
            add_problem(problems, cell_id, "Interpolation weight outside [0, 1]")
        if not np.isclose(row.weight_lower + row.weight_upper, 1.0, atol=1e-10):
            add_problem(problems, cell_id, "Interpolation weights do not sum to 1")

        reconstructed_efc = (
            row.weight_lower * row.efc50_lower
            + row.weight_upper * row.efc50_upper
        )
        if not np.isclose(reconstructed_efc, 50.0, atol=1e-8):
            add_problem(
                problems,
                cell_id,
                f"Interpolation reconstructs EFC={reconstructed_efc:.8f}, not 50",
            )

    # Paper-style operating-condition features.
    condition_values = np.array(
        [row.temperature_C, row.charge_c_rate, row.discharge_c_rate], dtype=float
    )
    if not np.isfinite(condition_values).all():
        add_problem(problems, cell_id, "Non-finite operating-condition feature")

    if np.isfinite(row.temperature_C):
        if not np.isclose(row.temperature_C % 5.0, 0.0, atol=1e-10):
            add_problem(problems, cell_id, "temperature_C is not rounded to 5 C")
        if row.temperature_C not in EXPECTED_TEMPERATURES_C:
            add_warning(
                warnings,
                cell_id,
                f"Unexpected VTC6 ambient temperature: {row.temperature_C:g} C",
            )

    expected_temp = round_to_step(row.ambient_temperature_raw_C, 5.0)
    if np.isfinite(expected_temp) and not np.isclose(row.temperature_C, expected_temp):
        add_problem(
            problems,
            cell_id,
            "temperature_C does not match rounded ambient_temperature_raw_C",
        )

    for name, value in (
        ("charge_c_rate", row.charge_c_rate),
        ("discharge_c_rate", row.discharge_c_rate),
    ):
        if np.isfinite(value):
            if value <= 0 or value > 10:
                add_problem(problems, cell_id, f"{name} outside (0, 10]")
            if not np.isclose((value / 0.5) % 1.0, 0.0, atol=1e-10):
                add_problem(problems, cell_id, f"{name} is not rounded to 0.5 C")

    if np.isfinite(row.charge_c_rate_raw):
        expected_charge = round_to_step(row.charge_c_rate_raw, 0.5)
        if not np.isclose(row.charge_c_rate, expected_charge):
            add_problem(
                problems,
                cell_id,
                "charge_c_rate does not match rounded charge_c_rate_raw",
            )
    elif not np.isclose(row.charge_c_rate, row.charge_cc_c_rate_protocol):
        add_problem(
            problems,
            cell_id,
            "Missing raw charge C-rate but final value does not fall back to protocol",
        )

    if np.isfinite(row.discharge_c_rate_raw):
        expected_discharge = round_to_step(row.discharge_c_rate_raw, 0.5)
        if not np.isclose(row.discharge_c_rate, expected_discharge):
            add_problem(
                problems,
                cell_id,
                "discharge_c_rate does not match rounded discharge_c_rate_raw",
            )
    else:
        add_problem(problems, cell_id, "Missing raw discharge C-rate")

    if row.charge_cc_c_rate_protocol not in EXPECTED_CHARGE_PROTOCOL_C_RATES:
        add_warning(
            warnings,
            cell_id,
            f"Unexpected charge protocol C-rate: {row.charge_cc_c_rate_protocol:g} C",
        )
    if row.charge_voltage_V not in EXPECTED_CHARGE_VOLTAGES_V:
        add_warning(
            warnings,
            cell_id,
            f"Unexpected charge voltage: {row.charge_voltage_V:g} V",
        )

    if not (0 <= row.charge_c_rate_bins_used <= 4):
        add_problem(problems, cell_id, "charge_c_rate_bins_used outside [0, 4]")
    if not (0 <= row.discharge_c_rate_bins_used <= 4):
        add_problem(problems, cell_id, "discharge_c_rate_bins_used outside [0, 4]")
    if row.n_condition_cycles <= 0:
        add_problem(problems, cell_id, "No ageing cycles used for condition features")

    # Surface-temperature telemetry is retained only as an audit diagnostic.
    surface = np.array(
        [
            row.surface_temperature_p05_C,
            row.surface_temperature_mean_C,
            row.surface_temperature_max_C,
        ],
        dtype=float,
    )
    if not np.isfinite(surface).all():
        add_warning(warnings, cell_id, "Missing surface-temperature diagnostics")
    else:
        if not surface[0] <= surface[1] <= surface[2]:
            add_problem(
                problems,
                cell_id,
                "Surface temperature does not satisfy p05 <= mean <= max",
            )
        if surface[2] >= 70:
            add_warning(
                warnings,
                cell_id,
                f"Surface temperature reaches {surface[2]:.1f} C",
            )

    # RPT/lifetime consistency.
    if cell_id not in rpt_cells:
        continue

    rpt_cell = rpts[rpts["cell_id"] == cell_id].sort_values("rpt_efc")

    if len(rpt_cell) != int(row.n_rpts):
        add_problem(
            problems,
            cell_id,
            f"n_rpts={row.n_rpts}, but RPT summary contains {len(rpt_cell)} rows",
        )

    if rpt_cell["rpt_number"].duplicated().any():
        add_problem(problems, cell_id, "Duplicate rpt_number values")

    if set(rpt_cell["chemistry"].astype(str)) != {"VTC6"}:
        add_problem(problems, cell_id, "Unexpected chemistry in RPT summary")

    rpt_efc = rpt_cell["rpt_efc"].to_numpy(dtype=float)
    capacity = rpt_cell["capacity_Ah"].to_numpy(dtype=float)
    counter_capacity = rpt_cell["capacity_counter_Ah"].to_numpy(dtype=float)
    stored_soh = rpt_cell["soh"].to_numpy(dtype=float)

    if np.any(np.diff(rpt_efc) <= 0):
        add_problem(problems, cell_id, "RPT EFC is not strictly increasing")
    if not np.isfinite(capacity).all() or np.any(capacity <= 0):
        add_problem(problems, cell_id, "Invalid integrated RPT capacity")

    if len(capacity) and capacity[0] > 0:
        recomputed_soh = capacity / capacity[0]
        if not np.allclose(stored_soh, recomputed_soh, rtol=1e-6, atol=1e-8):
            add_problem(problems, cell_id, "Stored RPT SOH does not match capacity / initial capacity")

    if len(rpt_cell):
        first = rpt_cell.iloc[0]
        last = rpt_cell.iloc[-1]
        if not np.isclose(row.efc0, first.rpt_efc, atol=1e-8):
            add_problem(problems, cell_id, "efc0 does not match first RPT EFC")
        if not np.isclose(row.initial_rpt_capacity_Ah, first.capacity_Ah, rtol=1e-6):
            add_problem(problems, cell_id, "initial_rpt_capacity_Ah does not match RPT summary")
        if not np.isclose(row.last_rpt_efc, last.rpt_efc, atol=1e-8):
            add_problem(problems, cell_id, "last_rpt_efc does not match RPT summary")
        if not np.isclose(row.last_rpt_capacity_Ah, last.capacity_Ah, rtol=1e-6):
            add_problem(problems, cell_id, "last_rpt_capacity_Ah does not match RPT summary")
        if not np.isclose(row.last_rpt_soh, last.soh, rtol=1e-6, atol=1e-8):
            add_problem(problems, cell_id, "last_rpt_soh does not match RPT summary")

    if not np.isclose(row.cycle_life_soh_threshold, LIFETIME_SOH_THRESHOLD):
        add_problem(
            problems,
            cell_id,
            f"Unexpected lifetime SOH threshold: {row.cycle_life_soh_threshold}",
        )

    try:
        expected_life, expected_censored, _ = recompute_lifetime(
            rpt_cell, row.cycle_life_soh_threshold
        )
        if not np.isclose(row.cycle_life_efc, expected_life, rtol=1e-8, atol=1e-6):
            add_problem(
                problems,
                cell_id,
                f"cycle_life_efc={row.cycle_life_efc:.4f}, recomputed={expected_life:.4f}",
            )
        if bool(row.cycle_life_censored) != bool(expected_censored):
            add_problem(
                problems,
                cell_id,
                f"cycle_life_censored={row.cycle_life_censored}, recomputed={expected_censored}",
            )
    except ValueError as exc:
        add_problem(problems, cell_id, f"Could not recompute lifetime: {exc}")

    # Independent RPT-capacity counter cross-check. This is diagnostic rather
    # than a hard failure because raw counters may contain resets/artifacts.
    valid_counter = (
        np.isfinite(capacity)
        & np.isfinite(counter_capacity)
        & (capacity > 0)
    )
    if np.any(valid_counter):
        rel_error = np.abs(counter_capacity[valid_counter] - capacity[valid_counter]) / capacity[valid_counter]
        max_rel_error = float(np.max(rel_error))
        if max_rel_error > 0.05:
            add_warning(
                warnings,
                cell_id,
                f"Integrated/counter RPT capacity differs by up to {100 * max_rel_error:.1f}%",
            )

    # Optional impedance-file RPT count cross-check.
    if np.isfinite(row.n_impedance_rpts):
        if int(row.n_impedance_rpts) != int(row.n_rpts):
            add_problem(
                problems,
                cell_id,
                f"n_impedance_rpts={int(row.n_impedance_rpts)} != n_rpts={int(row.n_rpts)}",
            )
        if not bool(row.rpt_count_matches_impedance):
            add_problem(problems, cell_id, "rpt_count_matches_impedance is False")

    # Extracted curve RPT EFC values should agree with metadata.
    curve_cell = curves[curves["cell_id"] == cell_id]
    expected_curve_efc = {
        "efc0": row.efc0,
        "efc50_lower": row.efc50_lower,
        "efc50_upper": row.efc50_upper,
    }
    for curve_type, expected_efc in expected_curve_efc.items():
        curve = curve_cell[curve_cell["curve_type"] == curve_type]
        if curve.empty:
            continue
        unique_efc = curve["rpt_efc"].dropna().unique()
        if len(unique_efc) != 1 or not np.isclose(unique_efc[0], expected_efc, atol=1e-8):
            add_problem(
                problems,
                cell_id,
                f"{curve_type} rpt_efc does not match metadata",
            )


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n========== METADATA SUMMARY ==========")
summary_columns = [
    "cell_id",
    "cycle_life_efc",
    "cycle_life_censored",
    "last_rpt_soh",
    "temperature_C",
    "charge_c_rate",
    "discharge_c_rate",
    "charge_voltage_V",
    "n_rpts",
]
print(metadata[summary_columns].sort_values("cell_id").to_string(index=False))

print("\n========== SUMMARY ==========")
print(f"Cells in curves:       {curves['cell_id'].nunique()}")
print(f"Cells in metadata:     {metadata['cell_id'].nunique()}")
print(f"Cells in RPT summary:  {rpts['cell_id'].nunique()}")
print(f"Extracted curves:      {curves.groupby(['cell_id', 'curve_type']).ngroups}")
print(f"Expected curves:       {len(metadata) * 3}")
print(f"RPT rows:              {len(rpts)}")
print(f"Censored lifetimes:    {int(metadata['cycle_life_censored'].sum())} / {len(metadata)}")
print(f"Flagged problems:      {len(problems)}")
print(f"Warnings:              {len(warnings)}")

if problems:
    print("\nProblems:")
    for cell_id, issue in problems:
        print(f"  {cell_id}: {issue}")
else:
    print("\nNo verification problems found.")

if warnings:
    print("\nWarnings:")
    for cell_id, issue in warnings:
        print(f"  {cell_id}: {issue}")
