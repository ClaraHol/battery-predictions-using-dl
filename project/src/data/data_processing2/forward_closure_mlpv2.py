#!/usr/bin/env python3
"""Forward-closure test for the MLP-v2 physical features.

Run this script from the data_processing2 directory, alongside
extract_features_mlpv2.py.

For each cell/stage in experimental_inference_mlp2.csv:
  1. Load the 11 inferred physical parameters from the MLP-v2 output.
  2. Build the chemistry-specific PyBaMM parameter dictionary.
  3. Run the same battery simulator/configuration used to generate the MLP
     training data.
  4. Compare the simulated voltage response with the experimental RPT.

EFC0 is compared directly with its experimental RPT.
For EFC50, the MLP-v2 file contains an interpolated parameter estimate rather
than a measured EFC50 curve. Therefore the script compares the simulated
EFC50 response against both bracketing RPTs and also against a weighted
interpolation of those two RPT voltage curves. The latter is diagnostic only;
it is not part of the original feature-extraction method.

The script also computes the 100-D observation-space RMSE using the same
preprocessing as extract_features_mlpv2.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# User-editable settings
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# The MLP-v2 extractor writes this file here in the normal project layout.
FEATURE_CSV = None  # None -> project/models/inference_results/experimental_inference_mlp2.csv

# Output directory.
OUTPUT_DIR = None  # None -> project/models/forward_closure_mlpv2

# Save voltage plots for the worst cases. Set to False for a faster run.
SAVE_PLOTS = True
TOP_N_PLOTS_PER_CHEMISTRY = 5

# By default use the exact current used when generating the MLP training data.
# Set True only if you deliberately want to test sensitivity to the measured
# experimental current instead.
USE_EXPERIMENTAL_CURRENT = False

# Use the simulator's training-time simulation horizon (3700/3600 of a nominal
# full C-rate period). If True, use a slightly larger horizon based on the
# experimental duration when needed.
EXTEND_SIMULATION_TO_EXPERIMENT = True
SIMULATION_HORIZON_FACTOR = 1.10

# Number of points for normalized-time voltage comparison.
N_COMPARE_POINTS = 200


# -----------------------------------------------------------------------------
# Project/module discovery
# -----------------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected an ancestor containing both "
        "'src' and 'models'."
    )


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the v2 extractor's chemistry/source definitions and observation
# preprocessing rather than duplicating those details.
extractor = importlib.import_module("extract_features_mlpv2")

from src.simulation.battery_simulator import simulator  # noqa: E402


# -----------------------------------------------------------------------------
# Training-time simulator configuration from generate_simulations.py
# -----------------------------------------------------------------------------
# These signs/cutoffs intentionally follow generate_simulations.py, not the
# magnitude-only experimental current stored by the v2 extractor.
SIM_CONFIG = {
    "25R": {"Q_rated": 2.5, "current": -1.25, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.2},
    "HG2": {"Q_rated": 3.0, "current": -1.5, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.2},
    "M1A": {"Q_rated": 1.1, "current": -0.55, "dod": 1.0, "V_cut_lb": 2.0, "V_cut_ub": 2.6},
    "MJ1": {"Q_rated": 3.5, "current": 3.5, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.2},
    "PA": {"Q_rated": 74.273, "current": 24.33, "dod": 1.0, "V_cut_lb": 2.75, "V_cut_ub": 4.2},
    "PBC": {"Q_rated": 76.0, "current": 25.43, "dod": 1.0, "V_cut_lb": 2.75, "V_cut_ub": 4.2},
    "PD": {"Q_rated": 83.456, "current": 28.0, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.15},
    "VTC5A": {"Q_rated": 2.5, "current": 2.5, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.25},
    "VTC6": {"Q_rated": 3.0, "current": 0.6, "dod": 1.0, "V_cut_lb": 2.5, "V_cut_ub": 4.25},
}

PARAMETER_NAMES = [
    "D_s_n [m2/s]",
    "D_s_p [m2/s]",
    "k_n [(A/m2)(mol/m3)^(-1.5)]",
    "k_p [(A/m2)(mol/m3)^(-1.5)]",
    "D_e",
    "sigma_e [S/m]",
    "R_f [ohm*m^2]",
    "eps_s_n",
    "eps_s_p",
    "theta_n_init",
    "theta_p_init",
]
PHYSICAL_COLUMNS = [f"{name}_physical" for name in PARAMETER_NAMES]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def resolve_feature_csv() -> Path:
    if FEATURE_CSV is not None:
        return Path(FEATURE_CSV).expanduser().resolve()
    return PROJECT_ROOT / "models" / "inference_results" / "experimental_inference_mlp2.csv"


def resolve_output_dir() -> Path:
    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR).expanduser().resolve()
    return PROJECT_ROOT / "models" / "forward_closure_mlpv2"


def load_features() -> pd.DataFrame:
    path = resolve_feature_csv()
    if not path.is_file():
        raise FileNotFoundError(f"MLP-v2 feature CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"cell_id", "chemistry", "curve_type", *PHYSICAL_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")

    df = df.copy()
    for col in PHYSICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = ~np.isfinite(df[PHYSICAL_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} rows with non-finite inferred parameters")
        df = df.loc[~bad].copy()

    return df


def load_curve_sources():
    """Load the same long-form curve sources used by extract_features_mlpv2."""
    datasets = {}
    for source_name, config in extractor.DATA_SOURCES.items():
        curve_path = Path(config["curves"])
        metadata_path = Path(config["metadata"])

        if not curve_path.is_file():
            raise FileNotFoundError(
                f"Curve file for {source_name} not found: {curve_path}\n"
                "Set PROCESSED_DATA_DIR in extract_features_mlpv2.py or adjust the "
                "paths there."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        curves = pd.read_parquet(curve_path)
        metadata = pd.read_parquet(metadata_path)
        datasets[source_name] = {
            "chemistry": config["chemistry"],
            "curves": curves,
            "metadata": metadata,
        }
        print(
            f"Loaded {source_name}: chemistry={config['chemistry']}, "
            f"cells={curves['cell_id'].nunique()}, points={len(curves):,}"
        )

    return datasets


def build_curve_lookup(datasets):
    lookup = {}
    for source_name, data in datasets.items():
        curves = data["curves"]
        for (cell_id, curve_type), group in curves.groupby(
            ["cell_id", "curve_type"], sort=False
        ):
            lookup[(str(cell_id), str(curve_type))] = {
                "source": source_name,
                "chemistry": data["chemistry"],
                "data": group.sort_values("time_s"),
            }
    return lookup


def build_metadata_lookup(datasets):
    lookup = {}
    for source_name, data in datasets.items():
        metadata = data["metadata"]
        for row in metadata.itertuples(index=False):
            lookup[(str(row.cell_id), data["chemistry"])] = {
                "source": source_name,
                "row": row,
            }
    return lookup


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------
def get_parameter_setting(chemistry):
    module = importlib.import_module(f"src.simulation.parameters.parameter_{chemistry}")
    return module.params_setting


def simulation_current(chemistry, exp_curve=None):
    cfg = SIM_CONFIG[chemistry]
    current = float(cfg["current"])
    if USE_EXPERIMENTAL_CURRENT and exp_curve is not None:
        magnitude = float(np.nanmedian(np.abs(exp_curve["current_A"].to_numpy(dtype=float))))
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(f"Invalid experimental current for {chemistry}")
        current = np.sign(current) * magnitude
    return current


def simulate_from_row(row: pd.Series, chemistry: str, exp_curve=None):
    cfg = SIM_CONFIG[chemistry]
    params_setting = get_parameter_setting(chemistry)

    values = [float(row[col]) for col in PHYSICAL_COLUMNS]
    params = params_setting(values)

    current = simulation_current(chemistry, exp_curve)
    training_horizon = abs(3700.0 * cfg["Q_rated"] * cfg["dod"] / current)

    horizon = training_horizon
    if EXTEND_SIMULATION_TO_EXPERIMENT and exp_curve is not None:
        exp_time = np.asarray(exp_curve["time_s"], dtype=float)
        exp_duration = float(np.nanmax(exp_time) - np.nanmin(exp_time))
        horizon = max(horizon, SIMULATION_HORIZON_FACTOR * exp_duration)

    result = simulator(params, current, abs(float(horizon)))
    if isinstance(result, str):
        raise RuntimeError(result)

    sim_voltage = np.asarray(result["Terminal voltage [V]"].entries, dtype=float)
    sim_time = np.asarray(result["Time [s]"].entries, dtype=float)

    mask = np.isfinite(sim_time) & np.isfinite(sim_voltage)
    sim_time = sim_time[mask]
    sim_voltage = sim_voltage[mask]

    if len(sim_time) < 2:
        raise RuntimeError("Simulator returned fewer than two valid points")

    order = np.argsort(sim_time)
    sim_time = sim_time[order]
    sim_voltage = sim_voltage[order]

    sim_time = sim_time - sim_time[0]
    return sim_time, sim_voltage, current


# -----------------------------------------------------------------------------
# Comparison utilities
# -----------------------------------------------------------------------------
def clean_curve(curve):
    t = curve["time_s"].to_numpy(dtype=float)
    v = curve["voltage_V"].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    t, unique_idx = np.unique(t, return_index=True)
    v = v[unique_idx]
    t = t - t[0]
    if len(t) < 2 or t[-1] <= 0:
        raise ValueError("Experimental curve has insufficient time span")
    return t, v


def normalized_voltage_curve(time_s, voltage_v, n=N_COMPARE_POINTS):
    u = np.linspace(0.0, 1.0, n)
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(voltage_v, dtype=float)
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("Invalid curve duration")
    un = (t - t[0]) / duration
    return u, np.interp(u, un, v)


def compare_voltage(sim_t, sim_v, exp_t, exp_v):
    u, sim_norm = normalized_voltage_curve(sim_t, sim_v)
    _, exp_norm = normalized_voltage_curve(exp_t, exp_v)
    err = sim_norm - exp_norm

    return {
        "duration_sim_s": float(sim_t[-1]),
        "duration_exp_s": float(exp_t[-1]),
        "duration_ratio_sim_exp": float(sim_t[-1] / exp_t[-1]),
        "start_voltage_sim_V": float(sim_v[0]),
        "start_voltage_exp_V": float(exp_v[0]),
        "end_voltage_sim_V": float(sim_v[-1]),
        "end_voltage_exp_V": float(exp_v[-1]),
        "start_voltage_error_V": float(sim_v[0] - exp_v[0]),
        "end_voltage_error_V": float(sim_v[-1] - exp_v[-1]),
        "voltage_rmse_V": float(np.sqrt(np.mean(err**2))),
        "voltage_mae_V": float(np.mean(np.abs(err))),
        "voltage_max_abs_error_V": float(np.max(np.abs(err))),
        "sim_u": u,
        "sim_v_norm": sim_norm,
        "exp_v_norm": exp_norm,
    }


def observation_from_curve(time_s, voltage_v, chemistry):
    return extractor.make_observation(voltage_v, time_s, chemistry)


def observation_rmse(sim_t, sim_v, exp_t, exp_v, chemistry):
    x_sim = observation_from_curve(sim_t, sim_v, chemistry)
    x_exp = observation_from_curve(exp_t, exp_v, chemistry)
    diff = x_sim - x_exp
    return float(np.sqrt(np.mean(diff**2)))


# -----------------------------------------------------------------------------
# EFC50 bracket handling
# -----------------------------------------------------------------------------
def get_efc50_brackets(cell_id, chemistry, curve_lookup, metadata_lookup):
    lower_key = (cell_id, "efc50_lower")
    upper_key = (cell_id, "efc50_upper")
    if lower_key not in curve_lookup or upper_key not in curve_lookup:
        raise KeyError(f"Missing EFC50 bracket curves for {cell_id}")

    meta_entry = metadata_lookup.get((cell_id, chemistry))
    if meta_entry is None:
        raise KeyError(f"Missing metadata for {cell_id} / {chemistry}")

    meta = meta_entry["row"]
    w_lower = float(meta.weight_lower)
    w_upper = float(meta.weight_upper)
    if not np.isfinite(w_lower + w_upper) or abs(w_lower + w_upper - 1.0) > 1e-6:
        raise ValueError(f"Invalid EFC50 weights for {cell_id}")

    return (
        curve_lookup[lower_key]["data"],
        curve_lookup[upper_key]["data"],
        w_lower,
        w_upper,
    )


def weighted_bracket_voltage(lower_t, lower_v, upper_t, upper_v, w_lower, w_upper):
    """Diagnostic voltage curve at EFC50.

    This interpolates the two bracket voltage curves on normalized time. It is
    only a visualization/closure diagnostic; v2 itself interpolates inferred
    parameters, not raw voltage curves.
    """
    u, lv = normalized_voltage_curve(lower_t, lower_v, N_COMPARE_POINTS)
    _, uv = normalized_voltage_curve(upper_t, upper_v, N_COMPARE_POINTS)
    return u, w_lower * lv + w_upper * uv


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_plot(path, title, sim_t, sim_v, exp_t, exp_v, extra=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(exp_t, exp_v, label="experimental")
    ax.plot(sim_t, sim_v, label="simulated")
    if extra is not None:
        for extra_t, extra_v, label in extra:
            ax.plot(extra_t, extra_v, linestyle="--", alpha=0.7, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("voltage [V]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    feature_df = load_features()
    datasets = load_curve_sources()
    curve_lookup = build_curve_lookup(datasets)
    metadata_lookup = build_metadata_lookup(datasets)

    output_dir = resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if SAVE_PLOTS:
        plot_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    # The feature CSV contains EFC0 and EFC50 parameter rows.
    for chemistry in sorted(feature_df["chemistry"].dropna().unique()):
        chem_rows = feature_df.loc[feature_df["chemistry"] == chemistry].copy()
        chem_rows = chem_rows.sort_values(["cell_id", "curve_type"])
        print(f"\n=== {chemistry}: {chem_rows['cell_id'].nunique()} cells ===")

        for _, row in chem_rows.iterrows():
            cell_id = str(row["cell_id"])
            stage = str(row["curve_type"])

            try:
                if stage == "efc0":
                    exp_curve = curve_lookup[(cell_id, "efc0")]["data"]
                    exp_t, exp_v = clean_curve(exp_curve)
                    sim_t, sim_v, sim_current = simulate_from_row(
                        row, chemistry, exp_curve
                    )
                    metrics = compare_voltage(sim_t, sim_v, exp_t, exp_v)
                    x_rmse = observation_rmse(
                        sim_t, sim_v, exp_t, exp_v, chemistry
                    )

                    result = {
                        "cell_id": cell_id,
                        "chemistry": chemistry,
                        "stage": "efc0",
                        "source": curve_lookup[(cell_id, "efc0")]["source"],
                        "sim_current_A": sim_current,
                        "exp_current_A_median": float(
                            np.nanmedian(np.abs(exp_curve["current_A"].to_numpy(dtype=float)))
                        ),
                        "observation_rmse": x_rmse,
                        **{k: v for k, v in metrics.items() if not k.startswith("sim_") and not k.startswith("exp_v_")},
                    }
                    results.append(result)

                    if SAVE_PLOTS:
                        result["plot_payload"] = (sim_t, sim_v, exp_t, exp_v, None)

                elif stage == "efc50":
                    lower_curve, upper_curve, w_lower, w_upper = get_efc50_brackets(
                        cell_id, chemistry, curve_lookup, metadata_lookup
                    )
                    lower_t, lower_v = clean_curve(lower_curve)
                    upper_t, upper_v = clean_curve(upper_curve)

                    # Simulate from the interpolated EFC50 parameter estimate.
                    sim_t, sim_v, sim_current = simulate_from_row(
                        row, chemistry, lower_curve
                    )

                    lower_metrics = compare_voltage(sim_t, sim_v, lower_t, lower_v)
                    upper_metrics = compare_voltage(sim_t, sim_v, upper_t, upper_v)

                    u, bracket_v = weighted_bracket_voltage(
                        lower_t, lower_v, upper_t, upper_v, w_lower, w_upper
                    )
                    _, sim_v_norm = normalized_voltage_curve(sim_t, sim_v, N_COMPARE_POINTS)
                    bracket_rmse = float(np.sqrt(np.mean((sim_v_norm - bracket_v) ** 2)))

                    # Observation-space comparison against the weighted bracket
                    # observation is also diagnostic only.
                    x_lower = observation_from_curve(lower_t, lower_v, chemistry)
                    x_upper = observation_from_curve(upper_t, upper_v, chemistry)
                    x_bracket = w_lower * x_lower + w_upper * x_upper
                    x_sim = observation_from_curve(sim_t, sim_v, chemistry)
                    x_bracket_rmse = float(np.sqrt(np.mean((x_sim - x_bracket) ** 2)))

                    result = {
                        "cell_id": cell_id,
                        "chemistry": chemistry,
                        "stage": "efc50",
                        "source": curve_lookup[(cell_id, "efc50_lower")]["source"],
                        "sim_current_A": sim_current,
                        "exp_current_A_median": float(
                            np.nanmedian(np.abs(lower_curve["current_A"].to_numpy(dtype=float)))
                        ),
                        "efc50_weight_lower": w_lower,
                        "efc50_weight_upper": w_upper,
                        "lower_voltage_rmse_V": lower_metrics["voltage_rmse_V"],
                        "upper_voltage_rmse_V": upper_metrics["voltage_rmse_V"],
                        "weighted_bracket_voltage_rmse_V": bracket_rmse,
                        "weighted_bracket_observation_rmse": x_bracket_rmse,
                        "sim_duration_s": float(sim_t[-1]),
                        "lower_duration_s": float(lower_t[-1]),
                        "upper_duration_s": float(upper_t[-1]),
                        "sim_duration_over_lower": float(sim_t[-1] / lower_t[-1]),
                        "sim_duration_over_upper": float(sim_t[-1] / upper_t[-1]),
                        "lower_end_voltage_error_V": lower_metrics["end_voltage_error_V"],
                        "upper_end_voltage_error_V": upper_metrics["end_voltage_error_V"],
                    }
                    results.append(result)

                    if SAVE_PLOTS:
                        # Plot against the weighted diagnostic curve on normalized time.
                        plot_payload = (sim_t, sim_v, lower_t, lower_v, upper_t, upper_v, w_lower, w_upper)
                        result["plot_payload"] = plot_payload

                else:
                    print(f"Skipping unsupported curve_type={stage}: {cell_id}")

            except Exception as exc:
                failures.append(
                    {
                        "cell_id": cell_id,
                        "chemistry": chemistry,
                        "stage": stage,
                        "error": repr(exc),
                    }
                )
                print(f"FAILED {chemistry} {cell_id} {stage}: {exc}")

    result_df = pd.DataFrame(results)

    # Keep plot payloads out of CSVs.
    plot_payloads = {}
    if "plot_payload" in result_df.columns:
        for idx, payload in result_df["plot_payload"].items():
            plot_payloads[idx] = payload
        result_df = result_df.drop(columns=["plot_payload"])

    result_df.to_csv(output_dir / "forward_closure_results.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "forward_closure_failures.csv", index=False)

    # Summary by chemistry and stage.
    if not result_df.empty:
        summary_rows = []
        for (chem, stage), group in result_df.groupby(["chemistry", "stage"]):
            row = {
                "chemistry": chem,
                "stage": stage,
                "n": len(group),
            }
            for metric in [
                "voltage_rmse_V",
                "voltage_mae_V",
                "voltage_max_abs_error_V",
                "observation_rmse",
                "lower_voltage_rmse_V",
                "upper_voltage_rmse_V",
                "weighted_bracket_voltage_rmse_V",
                "weighted_bracket_observation_rmse",
            ]:
                if metric in group:
                    row[f"median_{metric}"] = float(group[metric].median())
                    row[f"mean_{metric}"] = float(group[metric].mean())
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "forward_closure_summary.csv", index=False
        )

    # Save plots for worst cases by voltage/observation error.
    if SAVE_PLOTS and not result_df.empty:
        for chemistry, group in result_df.groupby("chemistry"):
            if "observation_rmse" in group:
                score = group["observation_rmse"].fillna(-np.inf)
            elif "weighted_bracket_observation_rmse" in group:
                score = group["weighted_bracket_observation_rmse"].fillna(-np.inf)
            else:
                score = pd.Series(-np.inf, index=group.index)

            selected = score.sort_values(ascending=False).head(TOP_N_PLOTS_PER_CHEMISTRY)
            for idx in selected.index:
                row = result_df.loc[idx]
                payload = plot_payloads.get(idx)
                if payload is None:
                    continue

                cell_id = str(row["cell_id"]).replace("/", "_")
                stage = str(row["stage"])
                filename = plot_dir / f"{chemistry}_{cell_id}_{stage}.png"

                if stage == "efc0":
                    sim_t, sim_v, exp_t, exp_v, _ = payload
                    save_plot(
                        filename,
                        f"{chemistry} {cell_id} EFC0 forward closure",
                        sim_t,
                        sim_v,
                        exp_t,
                        exp_v,
                    )
                else:
                    sim_t, sim_v, lower_t, lower_v, upper_t, upper_v, w_lower, w_upper = payload
                    save_plot(
                        filename,
                        f"{chemistry} {cell_id} EFC50 forward closure",
                        sim_t,
                        sim_v,
                        lower_t,
                        lower_v,
                        extra=[
                            (upper_t, upper_v, "experimental upper bracket"),
                        ],
                    )

    print("\n=== Forward closure complete ===")
    print(f"Feature rows processed: {len(result_df)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
