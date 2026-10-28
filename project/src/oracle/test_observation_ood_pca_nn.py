"""PCA + nearest-neighbour OOD test for public battery RPT observations.

Purpose
-------
Compare the 100-dimensional observation vectors used for experimental MLP
inference against the corresponding chemistry-specific simulated training
observations in ``models/Full_simulations/dataset_<CHEMISTRY>.pt``.

The script uses the same observation construction as
``src/oracle/extract_features_mlp.py``:

    x_i = sqrt((t_i / t_typ)^2 + ((V_i - V_min)/(V_max - V_min))^2)

with 100 uniformly spaced time points.

For each chemistry:
  1. Filter non-finite simulated observations.
  2. Randomly split simulations into a reference set and an independent
     in-distribution control set.
  3. Standardize using reference simulations only.
  4. Fit PCA on reference simulations, retaining 99% explained variance.
  5. Fit 1-nearest-neighbour search on the retained PCA coordinates.
  6. Use control-to-reference NN distances as the empirical in-distribution
     null distribution.
  7. Score every experimental EFC0, EFC50-lower, and EFC50-upper curve by
     its NN-distance percentile relative to the control simulations.

The production preprocessing constants are copied from
``extract_features_mlp.py``. The simulation-training constants are copied from
``generate_simulations.py``. If they differ for a chemistry, the script runs a
second "training_matched" experimental preprocessing variant so the effect of
that mismatch can be quantified directly.

Expected repository layout
--------------------------
Run this file from anywhere, but place it under ``project/src/oracle`` (or edit
PROJECT_ROOT below). The expected simulation files are:

    <PROJECT_ROOT>/models/Full_simulations/dataset_M1A.pt
    <PROJECT_ROOT>/models/Full_simulations/dataset_HG2.pt
    ...

Experimental CSVs are sought under DATA_DIR using both the uploaded-style names
and the older ``*_all.csv`` names.

Outputs
-------
Written to ``reports/observation_ood`` by default:

    observation_ood_summary.csv
    observation_ood_per_curve.csv
    observation_preprocessing_contract.csv
    pca_<chemistry>_<variant>.png
    nn_distance_<chemistry>_<variant>.png

No model checkpoint is needed: this tests the MLP *input distribution*, not its
parameter predictions.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Editable paths / settings
# =============================================================================

# This matches extract_features_mlp.py when this script lives in project/src/oracle.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path("/work3/claho/battery_datasets/processed")
SIM_DIR = PROJECT_ROOT / "models" / "Full_simulations"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "observation_ood"

# The script tries these names in order.
EARLY_CSV_CANDIDATES = [
    DATA_DIR / "public_charging_curve_early.csv",
    DATA_DIR / "public_charging_curve_early_all.csv",
]
LATE_CSV_CANDIDATES = [
    DATA_DIR / "public_charging_curve_late.csv",
    DATA_DIR / "public_charging_curve_late_all.csv",
]
METADATA_CSV_CANDIDATES = [
    DATA_DIR / "public_cell_metadata_cleaned.csv",
]

RANDOM_SEED = 0
REFERENCE_FRACTION = 0.80
PCA_EXPLAINED_VARIANCE = 0.99
NN_NEIGHBORS = 1

# Plotting only; the NN calculation still uses the full reference/control sets.
MAX_REFERENCE_POINTS_IN_PLOT = 5000
MAX_CONTROL_POINTS_IN_PLOT = 2000

# Useful raw-curve warning threshold. It does not exclude any curve.
DURATION_RATIO_WARNING = 3.0


# =============================================================================
# Chemistry contracts
# =============================================================================

# Production experimental preprocessing from extract_features_mlp.py:
# chemistry -> (rated capacity Ah, reference current A, V_min V, V_max V)
EXTRACTION_CONFIG = {
    "25R": (2.5, -1.25, 2.5, 4.2),
    "HG2": (3.0, -1.5, 2.5, 4.2),
    "M1A": (1.1, -0.55, 2.0, 3.6),
    "MJ1": (3.5, 3.5, 2.5, 4.2),
    "PA": (74.273, 24.33, 2.75, 4.2),
    "PBC": (76.0, 25.43, 2.75, 4.2),
    "PD": (83.456, 28.0, 2.5, 4.15),
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
}

# Training-time observation normalization from generate_simulations.py.
TRAINING_CONFIG = {
    "25R": (2.5, -1.25, 2.5, 4.2),
    "HG2": (3.0, -1.5, 2.5, 4.2),
    "M1A": (1.1, -0.55, 2.0, 2.6),
    "MJ1": (3.5, 3.5, 2.5, 4.2),
    "PA": (74.273, 24.33, 2.75, 4.2),
    "PBC": (76.0, 25.43, 2.75, 4.2),
    "PD": (83.456, 28.0, 2.5, 4.15),
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
}

SOURCE_TO_CHEMISTRY = {
    "snl_lfp_": "M1A",
    "snl_nmc_": "HG2",
    "tongji_": "25R",
    "tum_": "VTC5A",
    "cmu_": "VTC6",
    "elt_": "MJ1",
}


# =============================================================================
# Utilities
# =============================================================================

def first_existing(candidates, label, required=True):
    for path in candidates:
        if path.is_file():
            return path
    if required:
        listed = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find {label}. Tried:\n  {listed}")
    return None


def get_chemistry(cell_id):
    s = str(cell_id).lower()
    return next(
        (
            chemistry
            for prefix, chemistry in SOURCE_TO_CHEMISTRY.items()
            if s.startswith(prefix)
        ),
        None,
    )


def config_equal(a, b):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=0)


def make_observation(voltage, time, config):
    """Reproduce the 100-D production observation preprocessing exactly."""
    voltage = np.asarray(voltage, dtype=float)
    time = np.asarray(time, dtype=float)

    mask = np.isfinite(voltage) & np.isfinite(time)
    voltage = voltage[mask]
    time = time[mask]

    order = np.argsort(time)
    voltage = voltage[order]
    time = time[order]

    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]

    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("curve has insufficient valid time points")

    capacity, current, v_min, v_max = config

    # Match extract_features_mlp.py: reset every experimental curve to t=0.
    time = time - time[0]
    t_interp = np.linspace(0.0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)

    t_typ = abs(3600.0 * capacity / current)
    observation = np.sqrt(
        (t_interp / t_typ) ** 2
        + ((v_interp - v_min) / (v_max - v_min)) ** 2
    )

    if observation.shape != (100,) or not np.isfinite(observation).all():
        raise ValueError("observation preprocessing did not produce 100 finite values")

    raw = {
        "n_raw_points": int(len(time)),
        "duration_s": float(time[-1]),
        "t_typ_s": float(t_typ),
        "duration_over_t_typ": float(time[-1] / t_typ),
        "voltage_start_V": float(voltage[0]),
        "voltage_end_V": float(voltage[-1]),
        "voltage_min_V": float(np.min(voltage)),
        "voltage_max_V": float(np.max(voltage)),
    }
    return observation, raw


def load_simulation_x(chemistry):
    path = SIM_DIR / f"dataset_{chemistry}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing simulation dataset for {chemistry}: {path}")

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        _, x = obj
    elif isinstance(obj, dict) and "x" in obj:
        x = obj["x"]
    else:
        raise TypeError(
            f"Unexpected format in {path}; expected (theta, x) or a dict containing 'x'"
        )

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 100:
        raise ValueError(f"{path} has x shape {x.shape}; expected (N, 100)")

    finite = np.isfinite(x).all(axis=1)
    removed = int((~finite).sum())
    x = x[finite]

    if len(x) < 20:
        raise ValueError(f"Only {len(x)} finite simulations remain for {chemistry}")

    return x, path, removed


def empirical_percentile(sorted_control_distances, values):
    """Percentile of each value relative to control distances, in [0, 100]."""
    values = np.asarray(values, dtype=float)
    ranks = np.searchsorted(sorted_control_distances, values, side="right")
    return 100.0 * ranks / len(sorted_control_distances)


def sample_rows(rng, x, max_rows):
    if len(x) <= max_rows:
        return x
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


# =============================================================================
# Experimental observation construction
# =============================================================================

def build_experimental_observations(early, late, metadata=None):
    """Return one row per actual MLP input curve.

    Late lower and upper RPTs are kept separate because the production model sees
    them separately and only interpolates the inferred parameter outputs later.
    """
    records = []
    failures = []

    early_required = {"cell_id", "time_s", "early_curve"}
    late_required = {
        "cell_id",
        "lower_curve_time",
        "upper_curve_time",
        "lower_curve",
        "upper_curve",
    }
    if not early_required.issubset(early.columns):
        raise ValueError(f"Early CSV missing columns: {sorted(early_required - set(early.columns))}")
    if not late_required.issubset(late.columns):
        raise ValueError(f"Late CSV missing columns: {sorted(late_required - set(late.columns))}")

    def add_curve(cell_id, stage, voltage, time, chemistry, variant, config):
        try:
            x, raw = make_observation(voltage, time, config)
        except ValueError as exc:
            failures.append((str(cell_id), chemistry, stage, variant, str(exc)))
            return

        record = {
            "cell_id": str(cell_id),
            "chemistry": chemistry,
            "stage": stage,
            "preprocessing_variant": variant,
            **raw,
            "observation": x,
        }
        records.append(record)

    for cell_id, group in early.groupby("cell_id", sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry not in EXTRACTION_CONFIG:
            failures.append((str(cell_id), chemistry, "efc0", "production", "unknown chemistry"))
            continue

        add_curve(
            cell_id,
            "efc0",
            group["early_curve"].to_numpy(),
            group["time_s"].to_numpy(),
            chemistry,
            "production",
            EXTRACTION_CONFIG[chemistry],
        )

        if not config_equal(EXTRACTION_CONFIG[chemistry], TRAINING_CONFIG[chemistry]):
            add_curve(
                cell_id,
                "efc0",
                group["early_curve"].to_numpy(),
                group["time_s"].to_numpy(),
                chemistry,
                "training_matched",
                TRAINING_CONFIG[chemistry],
            )

    for cell_id, group in late.groupby("cell_id", sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry not in EXTRACTION_CONFIG:
            failures.append((str(cell_id), chemistry, "efc50_lower", "production", "unknown chemistry"))
            continue

        for stage, voltage_col, time_col in [
            ("efc50_lower", "lower_curve", "lower_curve_time"),
            ("efc50_upper", "upper_curve", "upper_curve_time"),
        ]:
            add_curve(
                cell_id,
                stage,
                group[voltage_col].to_numpy(),
                group[time_col].to_numpy(),
                chemistry,
                "production",
                EXTRACTION_CONFIG[chemistry],
            )

            if not config_equal(EXTRACTION_CONFIG[chemistry], TRAINING_CONFIG[chemistry]):
                add_curve(
                    cell_id,
                    stage,
                    group[voltage_col].to_numpy(),
                    group[time_col].to_numpy(),
                    chemistry,
                    "training_matched",
                    TRAINING_CONFIG[chemistry],
                )

    if not records:
        raise RuntimeError("No experimental observations could be constructed")

    exp = pd.DataFrame(records)

    # Optional metadata merge. The observation array itself remains an object column.
    if metadata is not None and "cell_id" in metadata.columns:
        meta = metadata.copy()
        meta["cell_id"] = meta["cell_id"].astype(str)
        if meta["cell_id"].duplicated().any():
            warnings.warn("Metadata has duplicate cell_id rows; metadata merge skipped")
        else:
            keep = [
                c
                for c in [
                    "cell_id",
                    "source_dataset",
                    "temp_C",
                    "charge_c_rate",
                    "discharge_c_rate",
                    "cycle_life",
                    "operational_cluster",
                ]
                if c in meta.columns
            ]
            exp = exp.merge(meta[keep], on="cell_id", how="left", validate="many_to_one")

    return exp, failures


# =============================================================================
# PCA + nearest-neighbour OOD test
# =============================================================================

def run_ood_for_group(chemistry, variant, exp_group, rng):
    x_sim, sim_path, n_invalid = load_simulation_x(chemistry)

    perm = rng.permutation(len(x_sim))
    n_ref = int(round(REFERENCE_FRACTION * len(x_sim)))
    n_ref = min(max(n_ref, 10), len(x_sim) - 10)

    x_ref = x_sim[perm[:n_ref]]
    x_control = x_sim[perm[n_ref:]]

    scaler = StandardScaler()
    z_ref = scaler.fit_transform(x_ref)
    z_control = scaler.transform(x_control)

    x_exp = np.stack(exp_group["observation"].to_numpy())
    z_exp = scaler.transform(x_exp)

    pca = PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full")
    p_ref = pca.fit_transform(z_ref)
    p_control = pca.transform(z_control)
    p_exp = pca.transform(z_exp)

    nn = NearestNeighbors(n_neighbors=NN_NEIGHBORS, algorithm="auto", n_jobs=-1)
    nn.fit(p_ref)

    d_control = nn.kneighbors(p_control, return_distance=True)[0][:, 0]
    d_exp = nn.kneighbors(p_exp, return_distance=True)[0][:, 0]

    d_control_sorted = np.sort(d_control)
    percentiles = empirical_percentile(d_control_sorted, d_exp)

    scored = exp_group.copy().reset_index(drop=True)
    scored["nn_distance"] = d_exp
    scored["nn_control_percentile"] = percentiles
    scored["pca_components"] = pca.n_components_
    scored["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
    scored["sim_reference_n"] = len(x_ref)
    scored["sim_control_n"] = len(x_control)
    scored["sim_invalid_removed"] = n_invalid
    scored["simulation_dataset"] = str(sim_path)

    # Observation arrays are not useful in CSV output and would be huge.
    scored_no_x = scored.drop(columns=["observation"])

    summary_rows = []
    for stage, stage_df in scored_no_x.groupby("stage", sort=False):
        distances = stage_df["nn_distance"].to_numpy(dtype=float)
        pct = stage_df["nn_control_percentile"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "chemistry": chemistry,
                "preprocessing_variant": variant,
                "stage": stage,
                "n_experimental": len(stage_df),
                "n_sim_reference": len(x_ref),
                "n_sim_control": len(x_control),
                "n_sim_invalid_removed": n_invalid,
                "pca_components": int(pca.n_components_),
                "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                "control_nn_p50": float(np.percentile(d_control, 50)),
                "control_nn_p95": float(np.percentile(d_control, 95)),
                "control_nn_p99": float(np.percentile(d_control, 99)),
                "experimental_nn_p50": float(np.percentile(distances, 50)),
                "experimental_nn_p95": float(np.percentile(distances, 95)),
                "experimental_nn_max": float(np.max(distances)),
                "experimental_percentile_p50": float(np.percentile(pct, 50)),
                "experimental_percentile_p95": float(np.percentile(pct, 95)),
                "fraction_above_control_p95": float(np.mean(pct > 95.0)),
                "fraction_above_control_p99": float(np.mean(pct > 99.0)),
                "fraction_above_control_p99_9": float(np.mean(pct > 99.9)),
                "median_duration_over_t_typ": float(np.median(stage_df["duration_over_t_typ"])),
                "max_duration_over_t_typ": float(np.max(stage_df["duration_over_t_typ"])),
            }
        )

    # -------------------- PCA plot --------------------
    # PC1/PC2 is only visualization. NN uses every retained PC above.
    if p_ref.shape[1] >= 2:
        ref_plot = sample_rows(rng, p_ref, MAX_REFERENCE_POINTS_IN_PLOT)
        control_plot = sample_rows(rng, p_control, MAX_CONTROL_POINTS_IN_PLOT)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ref_plot[:, 0], ref_plot[:, 1], s=6, alpha=0.12, label="simulation reference")
        ax.scatter(control_plot[:, 0], control_plot[:, 1], s=8, alpha=0.18, label="simulation control")

        for stage in scored["stage"].unique():
            mask = scored["stage"].to_numpy() == stage
            ax.scatter(p_exp[mask, 0], p_exp[mask, 1], s=35, marker="x", label=stage)

        ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)")
        ax.set_title(f"{chemistry} observation space — {variant}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"pca_{chemistry}_{variant}.png", dpi=180)
        plt.close(fig)

    # -------------------- NN-distance plot --------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d_control, bins=60, density=True, alpha=0.45, label="simulation control")
    for stage in scored_no_x["stage"].unique():
        vals = scored_no_x.loc[scored_no_x["stage"] == stage, "nn_distance"].to_numpy()
        ax.scatter(vals, np.zeros_like(vals), marker="x", s=28, label=stage)
    ax.axvline(np.percentile(d_control, 95), linestyle="--", linewidth=1.5, label="control p95")
    ax.axvline(np.percentile(d_control, 99), linestyle=":", linewidth=1.5, label="control p99")
    ax.set_xlabel("1-NN distance in retained PCA space")
    ax.set_ylabel("control density")
    ax.set_title(f"{chemistry} NN-distance OOD test — {variant}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"nn_distance_{chemistry}_{variant}.png", dpi=180)
    plt.close(fig)

    return scored_no_x, pd.DataFrame(summary_rows)


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    early_path = first_existing(EARLY_CSV_CANDIDATES, "early charging-curve CSV")
    late_path = first_existing(LATE_CSV_CANDIDATES, "late charging-curve CSV")
    metadata_path = first_existing(METADATA_CSV_CANDIDATES, "metadata CSV", required=False)

    print(f"Early curves: {early_path}")
    print(f"Late curves:  {late_path}")
    print(f"Simulation dir: {SIM_DIR}")
    if metadata_path is not None:
        print(f"Metadata:     {metadata_path}")

    early = pd.read_csv(early_path)
    late = pd.read_csv(late_path)
    metadata = pd.read_csv(metadata_path) if metadata_path is not None else None

    # -------------------------------------------------------------------------
    # Preprocessing contract report
    # -------------------------------------------------------------------------
    chemistries_in_data = sorted(
        {
            c
            for c in [get_chemistry(x) for x in pd.concat([early["cell_id"], late["cell_id"]]).unique()]
            if c is not None
        }
    )

    contract_rows = []
    print("\n=== Training vs production preprocessing constants ===")
    for chemistry in chemistries_in_data:
        extraction = EXTRACTION_CONFIG[chemistry]
        training = TRAINING_CONFIG[chemistry]
        matches = config_equal(extraction, training)
        print(
            f"{chemistry:7s} match={str(matches):5s} "
            f"training={training} production={extraction}"
        )
        contract_rows.append(
            {
                "chemistry": chemistry,
                "config_matches": matches,
                "train_capacity_Ah": training[0],
                "train_current_A": training[1],
                "train_vmin_V": training[2],
                "train_vmax_V": training[3],
                "production_capacity_Ah": extraction[0],
                "production_current_A": extraction[1],
                "production_vmin_V": extraction[2],
                "production_vmax_V": extraction[3],
            }
        )

    contract = pd.DataFrame(contract_rows)
    contract.to_csv(OUTPUT_DIR / "observation_preprocessing_contract.csv", index=False)

    # -------------------------------------------------------------------------
    # Build actual experimental MLP inputs
    # -------------------------------------------------------------------------
    exp, failures = build_experimental_observations(early, late, metadata)

    print("\n=== Experimental curves constructed ===")
    print(
        exp.groupby(["chemistry", "preprocessing_variant", "stage"])["cell_id"]
        .nunique()
        .to_string()
    )

    if failures:
        print(f"\nWARNING: {len(failures)} experimental curves failed preprocessing")
        for failure in failures[:20]:
            print("  ", failure)
        pd.DataFrame(
            failures,
            columns=["cell_id", "chemistry", "stage", "preprocessing_variant", "reason"],
        ).to_csv(OUTPUT_DIR / "observation_ood_failures.csv", index=False)

    # Raw duration warnings are deliberately not exclusions.
    suspicious_duration = exp[exp["duration_over_t_typ"] > DURATION_RATIO_WARNING]
    if not suspicious_duration.empty:
        print(
            f"\nWARNING: {len(suspicious_duration)} experimental curves have "
            f"duration/t_typ > {DURATION_RATIO_WARNING:g}."
        )
        print(
            suspicious_duration[
                [
                    "cell_id",
                    "chemistry",
                    "stage",
                    "preprocessing_variant",
                    "duration_s",
                    "t_typ_s",
                    "duration_over_t_typ",
                ]
            ]
            .sort_values("duration_over_t_typ", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # PCA + NN tests
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    all_scored = []
    all_summary = []

    for (chemistry, variant), group in exp.groupby(
        ["chemistry", "preprocessing_variant"], sort=True
    ):
        print(f"\n=== {chemistry} / {variant} ===")
        try:
            scored, summary = run_ood_for_group(chemistry, variant, group, rng)
        except FileNotFoundError as exc:
            print(f"SKIP: {exc}")
            continue

        all_scored.append(scored)
        all_summary.append(summary)

        display_cols = [
            "stage",
            "n_experimental",
            "pca_components",
            "control_nn_p95",
            "control_nn_p99",
            "experimental_nn_p50",
            "experimental_percentile_p50",
            "fraction_above_control_p95",
            "fraction_above_control_p99",
            "max_duration_over_t_typ",
        ]
        print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    if not all_scored:
        raise RuntimeError(
            "No OOD tests ran. Check SIM_DIR and the dataset_<CHEM>.pt files."
        )

    scored = pd.concat(all_scored, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)

    scored_path = OUTPUT_DIR / "observation_ood_per_curve.csv"
    summary_path = OUTPUT_DIR / "observation_ood_summary.csv"
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Overall summary ===")
    print(
        summary[
            [
                "chemistry",
                "preprocessing_variant",
                "stage",
                "n_experimental",
                "experimental_percentile_p50",
                "fraction_above_control_p95",
                "fraction_above_control_p99",
                "max_duration_over_t_typ",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4g}")
    )

    print(f"\nSaved:\n  {summary_path}\n  {scored_path}")
    print(f"Plots and preprocessing contract are in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
