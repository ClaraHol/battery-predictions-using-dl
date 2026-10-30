"""PCA + nearest-neighbour OOD test for the RPT curves used by extract_features_mlpv2.py.

Purpose
-------
Compare the *actual 100-D MLP observations* made from the v2 experimental RPT
parquets against the chemistry-specific simulation observations stored in
models/Full_simulations/dataset_<CHEMISTRY>.pt.

This script deliberately does NOT filter cells by lifetime. The current lifetime
calculation may be wrong, but that should not affect whether the extracted RPT
curves resemble the simulations used to train the inferencer.

Run from anywhere, for example:
    python project/src/oracle/test_observation_ood_mlpv2.py

Outputs
-------
<repo>/reports/observation_ood_mlpv2/
    observation_ood_mlpv2_summary.csv
    observation_ood_mlpv2_per_curve.csv
    observation_ood_mlpv2_simulation_controls.csv
    pca_<CHEMISTRY>.png
    nn_distance_<CHEMISTRY>.png

Interpretation
--------------
For each experimental RPT, ``nn_control_percentile`` is its 1-NN distance
percentile relative to held-out simulations evaluated against an independent
simulation reference set.

    ~50        ordinary in-distribution observation
    >95        unusual relative to simulation controls
    >99        strong OOD evidence
    >99.9      extreme OOD evidence

The three curve types are evaluated separately because extract_features_mlpv2.py
runs the inferencer on efc0, efc50_lower, and efc50_upper separately and only
then interpolates the inferred features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuration copied from extract_features_mlpv2.py
# -----------------------------------------------------------------------------
PROCESSED_DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data")

CHEMISTRY_CONFIG = {
    # chemistry: (rated capacity Ah, reference current magnitude A, Vmin, Vmax)
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
    "MJ1": (3.5, 3.5, 2.5, 4.20),
    "25R": (2.5, 1.25, 2.5, 4.20),
}

CHEMISTRY_DIRECTION = {
    "VTC5A": "discharge",
    "VTC6": "discharge",
    "MJ1": "discharge",
    "25R": "charge",
}

DATA_SOURCES = {
    "vtc5a": {
        "chemistry": "VTC5A",
        "curves": PROCESSED_DATA_DIR / "vtc5a" / "vtc5a_all_curves.parquet",
    },
    "vtc6": {
        "chemistry": "VTC6",
        "curves": PROCESSED_DATA_DIR / "vtc6" / "vtc6_all_curves.parquet",
    },
    "samsung_mj1": {
        "chemistry": "MJ1",
        "curves": PROCESSED_DATA_DIR
        / "samsung_mj1"
        / "samsung_mj1_all_curves.parquet",
    },
    "lg_mj1": {
        "chemistry": "MJ1",
        "curves": PROCESSED_DATA_DIR / "lg_mj1" / "lg_mj1_all_curves.parquet",
    },
    "samsung_25r": {
        "chemistry": "25R",
        "curves": PROCESSED_DATA_DIR
        / "samsung_25r"
        / "samsung_25r_all_curves.parquet",
    },
}

EXPECTED_CURVE_TYPES = ("efc0", "efc50_lower", "efc50_upper")
CURVE_COLUMNS = [
    "cell_id",
    "chemistry",
    "curve_type",
    "time_s",
    "voltage_V",
    "current_A",
    "rpt_efc",
]

RANDOM_SEED = 0
REFERENCE_FRACTION = 0.80
PCA_EXPLAINED_VARIANCE = 0.99
MAX_SIM_POINTS_IN_PCA_PLOT = 5000
MAKE_PLOTS = False


# -----------------------------------------------------------------------------
# Locate models/Full_simulations robustly.
# -----------------------------------------------------------------------------
def find_simulation_dir():
    here = Path(__file__).resolve()
    candidates = []

    # Search upward from script location.
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / "models" / "Full_simulations")
        candidates.append(parent / "project" / "models" / "Full_simulations")

    # Also search upward from current working directory.
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / "models" / "Full_simulations")
        candidates.append(parent / "project" / "models" / "Full_simulations")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return candidate

    tried = "\n".join(f"  {p}" for p in seen)
    raise FileNotFoundError(
        "Could not locate models/Full_simulations. Tried:\n" + tried
    )


SIMULATION_DIR = find_simulation_dir()
REPO_ROOT = SIMULATION_DIR.parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "observation_ood_mlpv2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Simulation directory:", SIMULATION_DIR)
print("Output directory:    ", OUTPUT_DIR)


# -----------------------------------------------------------------------------
# Exact observation preprocessing from extract_features_mlpv2.py.
# -----------------------------------------------------------------------------
def make_observation(voltage, time, chemistry):
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

    capacity, current, v_min, v_max = CHEMISTRY_CONFIG[chemistry]
    time = time - time[0]
    t_interp = np.linspace(0.0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)

    t_typ = abs(3600.0 * capacity / current)
    observation = np.sqrt(
        (t_interp / t_typ) ** 2
        + ((v_interp - v_min) / (v_max - v_min)) ** 2
    )

    if not np.isfinite(observation).all():
        raise ValueError("observation preprocessing produced NaN or Inf")

    diagnostics = {
        "n_raw_points": int(len(time)),
        "duration_s": float(time[-1]),
        "duration_over_ttyp": float(time[-1] / t_typ),
        "start_voltage_V": float(voltage[0]),
        "end_voltage_V": float(voltage[-1]),
        "raw_voltage_min_V": float(np.min(voltage)),
        "raw_voltage_max_V": float(np.max(voltage)),
        "interp_voltage_start_V": float(v_interp[0]),
        "interp_voltage_end_V": float(v_interp[-1]),
    }
    return observation.astype(np.float64), diagnostics


def load_simulation_x(chemistry):
    path = SIMULATION_DIR / f"dataset_{chemistry}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing simulation dataset: {path}")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise ValueError(f"Unexpected simulation dataset structure in {path}")

    theta, x = loaded[0], loaded[1]
    x = torch.as_tensor(x).detach().cpu().numpy().astype(np.float64)

    if x.ndim != 2 or x.shape[1] != 100:
        raise ValueError(
            f"{chemistry}: expected simulation x shape (N, 100), found {x.shape}"
        )

    finite = np.isfinite(x).all(axis=1)
    removed = int((~finite).sum())
    x = x[finite]
    if len(x) < 100:
        raise ValueError(f"{chemistry}: too few finite simulation observations: {len(x)}")

    print(
        f"{chemistry}: loaded {len(x):,} finite simulation observations"
        + (f"; removed {removed:,} non-finite" if removed else "")
    )
    return x


def load_experimental_curves():
    rows = []
    failures = []

    for source_name, source in DATA_SOURCES.items():
        chemistry = source["chemistry"]
        path = source["curves"]
        if not path.is_file():
            print(f"WARNING: missing source {source_name}: {path}")
            continue

        df = pd.read_parquet(path, columns=CURVE_COLUMNS)
        if df.empty:
            print(f"WARNING: empty curve file for {source_name}: {path}")
            continue

        chem_values = set(df["chemistry"].dropna().astype(str).unique())
        if chem_values != {chemistry}:
            raise ValueError(
                f"{source_name}: expected chemistry {chemistry}, found {chem_values}"
            )

        unexpected = sorted(set(df["curve_type"].dropna().astype(str)) - set(EXPECTED_CURVE_TYPES))
        if unexpected:
            raise ValueError(
                f"{source_name}: unexpected curve types {unexpected}; "
                f"expected only {EXPECTED_CURVE_TYPES}"
            )

        print(
            f"{source_name}: {chemistry}, "
            f"{df['cell_id'].nunique()} cells, {len(df):,} raw points"
        )

        for (cell_id, curve_type), group in df.groupby(
            ["cell_id", "curve_type"], sort=False
        ):
            try:
                x, diag = make_observation(
                    group["voltage_V"].to_numpy(),
                    group["time_s"].to_numpy(),
                    chemistry,
                )

                current = pd.to_numeric(group["current_A"], errors="coerce").to_numpy(dtype=float)
                current = current[np.isfinite(current)]
                median_current = float(np.median(current)) if len(current) else np.nan

                rpt = pd.to_numeric(group["rpt_efc"], errors="coerce").to_numpy(dtype=float)
                rpt = rpt[np.isfinite(rpt)]
                rpt_efc = float(np.median(rpt)) if len(rpt) else np.nan

                row = {
                    "source_name": source_name,
                    "chemistry": chemistry,
                    "expected_direction": CHEMISTRY_DIRECTION[chemistry],
                    "cell_id": str(cell_id),
                    "curve_type": str(curve_type),
                    "rpt_efc": rpt_efc,
                    "median_current_A": median_current,
                    **diag,
                    "x": x,
                }
                rows.append(row)
            except ValueError as exc:
                failures.append(
                    {
                        "source_name": source_name,
                        "chemistry": chemistry,
                        "cell_id": str(cell_id),
                        "curve_type": str(curve_type),
                        "reason": str(exc),
                    }
                )

    if not rows:
        raise RuntimeError("No experimental RPT curves were successfully preprocessed")

    if failures:
        print(f"WARNING: {len(failures)} curves failed preprocessing")
        for failure in failures[:20]:
            print("  ", failure)

    return rows, failures


def empirical_percentiles(distances, control_distances):
    control_sorted = np.sort(np.asarray(control_distances, dtype=float))
    distances = np.asarray(distances, dtype=float)

    # Percentage of control distances <= the experimental distance.
    ranks = np.searchsorted(control_sorted, distances, side="right")
    percentile = 100.0 * ranks / len(control_sorted)

    # Conservative empirical upper-tail p-value with +1 correction.
    n_ge = len(control_sorted) - np.searchsorted(
        control_sorted, distances, side="left"
    )
    upper_tail_p = (n_ge + 1.0) / (len(control_sorted) + 1.0)
    return percentile, upper_tail_p


def fit_ood_model(x_sim, chemistry):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(x_sim))

    n_ref = int(round(REFERENCE_FRACTION * len(x_sim)))
    n_ref = min(max(n_ref, 1), len(x_sim) - 1)
    ref_idx = idx[:n_ref]
    control_idx = idx[n_ref:]

    x_ref = x_sim[ref_idx]
    x_control = x_sim[control_idx]

    scaler = StandardScaler()
    z_ref = scaler.fit_transform(x_ref)
    z_control = scaler.transform(x_control)

    pca = PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full")
    p_ref = pca.fit_transform(z_ref)
    p_control = pca.transform(z_control)

    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(p_ref)
    d_control = nn.kneighbors(p_control, return_distance=True)[0][:, 0]

    print(
        f"{chemistry}: PCA kept {p_ref.shape[1]} components "
        f"({pca.explained_variance_ratio_.sum():.4%} variance); "
        f"control NN median={np.median(d_control):.4g}, "
        f"q99={np.quantile(d_control, 0.99):.4g}"
    )

    return {
        "scaler": scaler,
        "pca": pca,
        "nn": nn,
        "p_ref": p_ref,
        "p_control": p_control,
        "d_control": d_control,
        "n_ref": len(x_ref),
        "n_control": len(x_control),
    }


def evaluate_experimental(rows, ood_models):
    output_rows = []

    for chemistry in CHEMISTRY_CONFIG:
        chem_rows = [row for row in rows if row["chemistry"] == chemistry]
        if not chem_rows:
            continue

        model = ood_models[chemistry]
        x_exp = np.stack([row["x"] for row in chem_rows], axis=0)
        z_exp = model["scaler"].transform(x_exp)
        p_exp = model["pca"].transform(z_exp)
        d_exp = model["nn"].kneighbors(p_exp, return_distance=True)[0][:, 0]
        percentiles, upper_tail_p = empirical_percentiles(
            d_exp, model["d_control"]
        )

        for i, row in enumerate(chem_rows):
            clean = {k: v for k, v in row.items() if k != "x"}
            clean.update(
                {
                    "nn_distance": float(d_exp[i]),
                    "nn_control_percentile": float(percentiles[i]),
                    "nn_upper_tail_p": float(upper_tail_p[i]),
                    "ood_gt95": bool(percentiles[i] > 95.0),
                    "ood_gt99": bool(percentiles[i] > 99.0),
                    "ood_gt999": bool(percentiles[i] > 99.9),
                    "pc1": float(p_exp[i, 0]),
                    "pc2": float(p_exp[i, 1]) if p_exp.shape[1] >= 2 else np.nan,
                    "pca_components": int(p_exp.shape[1]),
                    "pca_explained_variance": float(
                        model["pca"].explained_variance_ratio_.sum()
                    ),
                }
            )
            output_rows.append(clean)

    return pd.DataFrame(output_rows)


def make_summary(per_curve, ood_models):
    summary_rows = []

    for (source_name, chemistry, curve_type), group in per_curve.groupby(
        ["source_name", "chemistry", "curve_type"], sort=False
    ):
        d_control = ood_models[chemistry]["d_control"]
        current_expected = CHEMISTRY_CONFIG[chemistry][1]

        summary_rows.append(
            {
                "source_name": source_name,
                "chemistry": chemistry,
                "curve_type": curve_type,
                "n_curves": len(group),
                "simulation_reference_n": ood_models[chemistry]["n_ref"],
                "simulation_control_n": ood_models[chemistry]["n_control"],
                "pca_components": int(group["pca_components"].iloc[0]),
                "pca_explained_variance": float(
                    group["pca_explained_variance"].iloc[0]
                ),
                "control_nn_median": float(np.median(d_control)),
                "control_nn_q95": float(np.quantile(d_control, 0.95)),
                "control_nn_q99": float(np.quantile(d_control, 0.99)),
                "control_nn_q999": float(np.quantile(d_control, 0.999)),
                "experimental_nn_median": float(group["nn_distance"].median()),
                "experimental_nn_q95": float(group["nn_distance"].quantile(0.95)),
                "experimental_nn_max": float(group["nn_distance"].max()),
                "median_control_percentile": float(
                    group["nn_control_percentile"].median()
                ),
                "fraction_gt95": float(group["ood_gt95"].mean()),
                "fraction_gt99": float(group["ood_gt99"].mean()),
                "fraction_gt999": float(group["ood_gt999"].mean()),
                "median_duration_s": float(group["duration_s"].median()),
                "median_duration_over_ttyp": float(
                    group["duration_over_ttyp"].median()
                ),
                "median_start_voltage_V": float(group["start_voltage_V"].median()),
                "median_end_voltage_V": float(group["end_voltage_V"].median()),
                "median_current_A": float(group["median_current_A"].median()),
                "expected_current_A": float(current_expected),
                "median_current_ratio": float(
                    group["median_current_A"].median() / current_expected
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def save_simulation_control_summary(ood_models):
    rows = []
    for chemistry, model in ood_models.items():
        d = model["d_control"]
        rows.append(
            {
                "chemistry": chemistry,
                "n_reference": model["n_ref"],
                "n_control": model["n_control"],
                "pca_components": model["p_ref"].shape[1],
                "pca_explained_variance": model["pca"].explained_variance_ratio_.sum(),
                "nn_min": np.min(d),
                "nn_median": np.median(d),
                "nn_q95": np.quantile(d, 0.95),
                "nn_q99": np.quantile(d, 0.99),
                "nn_q999": np.quantile(d, 0.999),
                "nn_max": np.max(d),
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "observation_ood_mlpv2_simulation_controls.csv", index=False
    )


def plot_pca(chemistry, per_curve, model):
    if model["p_ref"].shape[1] < 2:
        print(f"{chemistry}: only one PCA component; skipping 2-D PCA plot")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    p_ref = model["p_ref"]
    if len(p_ref) > MAX_SIM_POINTS_IN_PCA_PLOT:
        idx = rng.choice(
            len(p_ref), size=MAX_SIM_POINTS_IN_PCA_PLOT, replace=False
        )
        p_ref_plot = p_ref[idx]
    else:
        p_ref_plot = p_ref

    subset = per_curve.loc[per_curve["chemistry"] == chemistry]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        p_ref_plot[:, 0],
        p_ref_plot[:, 1],
        s=6,
        alpha=0.12,
        label="simulation reference",
    )

    markers = {
        "efc0": "o",
        "efc50_lower": "^",
        "efc50_upper": "s",
    }

    for (source_name, curve_type), group in subset.groupby(
        ["source_name", "curve_type"], sort=False
    ):
        ax.scatter(
            group["pc1"],
            group["pc2"],
            s=38,
            marker=markers.get(curve_type, "x"),
            alpha=0.85,
            label=f"{source_name}: {curve_type}",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{chemistry}: simulation vs v2 experimental RPT observations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"pca_{chemistry}.png", dpi=180)
    plt.close(fig)


def plot_nn_distances(chemistry, per_curve, model):
    subset = per_curve.loc[per_curve["chemistry"] == chemistry]
    d_control = model["d_control"]

    labels = ["simulation control"]
    values = [d_control]
    for (source_name, curve_type), group in subset.groupby(
        ["source_name", "curve_type"], sort=False
    ):
        labels.append(f"{source_name}\n{curve_type}")
        values.append(group["nn_distance"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(labels)), 6))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("1-NN distance in standardized PCA space")
    ax.set_title(f"{chemistry}: OOD distance relative to held-out simulations")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"nn_distance_{chemistry}.png", dpi=180)
    plt.close(fig)


def print_key_summary(summary):
    if summary.empty:
        return

    cols = [
        "source_name",
        "chemistry",
        "curve_type",
        "n_curves",
        "median_control_percentile",
        "fraction_gt95",
        "fraction_gt99",
        "fraction_gt999",
        "median_duration_over_ttyp",
        "median_current_ratio",
    ]

    display = summary[cols].copy()
    for column in ["fraction_gt95", "fraction_gt99", "fraction_gt999"]:
        display[column] = 100.0 * display[column]

    print("\n=== Experimental observation OOD summary ===")
    print(
        display.to_string(
            index=False,
            formatters={
                "median_control_percentile": lambda x: f"{x:7.2f}",
                "fraction_gt95": lambda x: f"{x:6.1f}%",
                "fraction_gt99": lambda x: f"{x:6.1f}%",
                "fraction_gt999": lambda x: f"{x:6.1f}%",
                "median_duration_over_ttyp": lambda x: f"{x:8.3f}",
                "median_current_ratio": lambda x: f"{x:8.3f}",
            },
        )
    )

    print("\nInterpretation: percentiles are calibrated against held-out simulations.")
    print("  >95  : unusual")
    print("  >99  : strong OOD evidence")
    print("  >99.9: extreme OOD evidence")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    experimental_rows, failures = load_experimental_curves()
    available_chemistries = sorted({row["chemistry"] for row in experimental_rows})

    ood_models = {}
    for chemistry in available_chemistries:
        x_sim = load_simulation_x(chemistry)
        ood_models[chemistry] = fit_ood_model(x_sim, chemistry)

    per_curve = evaluate_experimental(experimental_rows, ood_models)
    per_curve = per_curve.sort_values(
        ["chemistry", "source_name", "curve_type", "cell_id"]
    ).reset_index(drop=True)

    summary = make_summary(per_curve, ood_models)
    summary = summary.sort_values(
        ["chemistry", "source_name", "curve_type"]
    ).reset_index(drop=True)

    per_curve_path = OUTPUT_DIR / "observation_ood_mlpv2_per_curve.csv"
    summary_path = OUTPUT_DIR / "observation_ood_mlpv2_summary.csv"
    per_curve.to_csv(per_curve_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_simulation_control_summary(ood_models)

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "observation_ood_mlpv2_failures.csv", index=False
        )

    if MAKE_PLOTS:
        for chemistry in available_chemistries:
            plot_pca(chemistry, per_curve, ood_models[chemistry])
            plot_nn_distances(chemistry, per_curve, ood_models[chemistry])

    print_key_summary(summary)
    print("\nSaved:")
    print(" ", summary_path)
    print(" ", per_curve_path)
    print(" ", OUTPUT_DIR / "observation_ood_mlpv2_simulation_controls.csv")
    if MAKE_PLOTS:
        print("  PCA and NN-distance plots in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
