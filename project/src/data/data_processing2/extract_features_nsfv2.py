#!/usr/bin/env python3
"""Extract experimental EFC0/EFC50 features with direct NSF posterior sampling.

This follows the data/curve structure of extract_features_mlpv2.py, but uses
the saved sbi Neural Spline Flow posterior estimator directly. No MCMC and no
rejection sampling are used. This is intentionally a fast diagnostic of the
learned NSF posterior distribution.

For each RPT, the script draws independent samples directly from the learned
flow and uses their mean as the representative 11-parameter estimate. The
sampling is deliberately not truncated to the prior: the fraction of samples
outside the prior is recorded as a diagnostic because the experimental VTC6
observations previously produced essentially zero prior-accepted flow samples.

The paper reports 50,000 posterior samples for its final inference, but its
MCMC implementation details are not available in the supplied code. This
script therefore uses 5,000 direct NSF samples by default for the current
diagnostic experiment.

EFC50 is obtained by inferring the lower and upper bracketing RPTs and
interpolating the posterior means in theta/prior coordinates. The three
dependent stoichiometry features are calculated locally with the
charge/discharge direction explicitly handled, matching the corrected
extract_features_mlpv2.py implementation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi import utils as utils
from sbi.utils.user_input_checks import process_prior


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DATA_DIR = Path("/work3/s214728/battery-predictions/processed_data")
MODEL_DIR = PROJECT_ROOT / "models" / "SBI_models"
OUTPUT_DIR = PROJECT_ROOT / "models" / "inference_results" / "nsf_v2_direct"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------------------------------------------------
# Direct NSF experiment parameters
# -----------------------------------------------------------------------------
# Fast diagnostic experiment: draw directly from the learned flow.
DEFAULT_N_POSTERIOR_SAMPLES = 5_000


# -----------------------------------------------------------------------------
# Same chemistry/source contract as extract_features_mlpv2.py
# -----------------------------------------------------------------------------
EFC50_TARGET = 50.0
LIFETIME_SOH_THRESHOLD = 0.90
EXPECTED_CURVE_TYPES = ("efc0", "efc50_lower", "efc50_upper")

EXPECTED_LOG10_PARAMETERS = {"D_s_n", "D_s_p", "k_n", "k_p", "R_f"}

CHEMISTRY_CONFIG = {
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
        "metadata": PROCESSED_DATA_DIR / "vtc5a" / "vtc5a_all_metadata.parquet",
    },
    "vtc6": {
        "chemistry": "VTC6",
        "curves": PROCESSED_DATA_DIR / "vtc6" / "vtc6_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "vtc6" / "vtc6_all_metadata.parquet",
    },
    "samsung_mj1": {
        "chemistry": "MJ1",
        "curves": PROCESSED_DATA_DIR / "samsung_mj1" / "samsung_mj1_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "samsung_mj1" / "samsung_mj1_all_metadata.parquet",
    },
    "lg_mj1": {
        "chemistry": "MJ1",
        "curves": PROCESSED_DATA_DIR / "lg_mj1" / "lg_mj1_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "lg_mj1" / "lg_mj1_all_metadata.parquet",
    },
    "samsung_25r": {
        "chemistry": "25R",
        "curves": PROCESSED_DATA_DIR / "samsung_25r" / "samsung_25r_all_curves.parquet",
        "metadata": PROCESSED_DATA_DIR / "samsung_25r" / "samsung_25r_all_metadata.parquet",
    },
}

CURVE_COLUMNS = [
    "cell_id",
    "chemistry",
    "curve_type",
    "time_s",
    "voltage_V",
    "current_A",
    "rpt_efc",
]

BASE_METADATA_COLUMNS = [
    "cell_id",
    "chemistry",
    "efc0",
    "efc50_lower",
    "efc50_upper",
    "weight_lower",
    "weight_upper",
    "direction",
]

TARGET_METADATA_COLUMNS = [
    "cycle_life_efc",
    "cycle_life_censored",
    "cycle_life_soh_threshold",
    "temperature_C",
    "charge_c_rate",
    "discharge_c_rate",
]

OPTIONAL_OUTPUT_METADATA_COLUMNS = [
    "lifetime_status",
    "lifetime_censor_reason",
    "temperature_C_raw",
    "ambient_temperature_raw_C",
    "charge_c_rate_raw",
    "discharge_c_rate_raw",
    "charge_voltage_V",
    "charge_cc_c_rate_protocol",
    "source_family",
    "source_dataset",
    "dataset_provenance",
    "source_file",
    "condition",
    "cell_tag",
    "cell_number",
    "test_type",
    "soc_lower_pct",
    "soc_upper_pct",
    "c_rate_method",
    "n_rpts_single_detected_cycle",
    "n_rpts_nonpair",
]


# -----------------------------------------------------------------------------
# Validation and data handling copied in structure from extract_features_mlpv2
# -----------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.map(
        lambda x: x
        if isinstance(x, (bool, np.bool_))
        else {"true": True, "false": False}.get(str(x).strip().lower(), np.nan)
    )
    if mapped.isna().any():
        bad = series.loc[mapped.isna()].astype(str).unique()[:10].tolist()
        raise ValueError(f"{label} contains non-boolean values: {bad}")
    return mapped.astype(bool)


def validate_dataset(curves: pd.DataFrame, metadata: pd.DataFrame, chemistry: str) -> None:
    _require_columns(curves, CURVE_COLUMNS, f"{chemistry} curves")
    _require_columns(
        metadata,
        BASE_METADATA_COLUMNS + TARGET_METADATA_COLUMNS,
        f"{chemistry} metadata",
    )

    if curves.empty or metadata.empty:
        raise ValueError(f"{chemistry} curves/metadata are empty")

    if set(curves["chemistry"].dropna().astype(str).unique()) != {chemistry}:
        raise ValueError(f"Unexpected chemistry values in {chemistry} curves")
    if set(metadata["chemistry"].dropna().astype(str).unique()) != {chemistry}:
        raise ValueError(f"Unexpected chemistry values in {chemistry} metadata")

    if metadata["cell_id"].duplicated().any():
        raise ValueError(f"{chemistry} metadata contains duplicate cell_id values")

    directions = set(metadata["direction"].dropna().astype(str).str.lower().unique())
    expected_direction = CHEMISTRY_DIRECTION[chemistry]
    if directions != {expected_direction}:
        raise ValueError(
            f"{chemistry} must use {expected_direction} RPT curves, found {directions}"
        )

    metadata_ids = set(metadata["cell_id"].astype(str))
    curve_ids = set(curves["cell_id"].astype(str))
    if metadata_ids != curve_ids:
        raise ValueError("Metadata and curve cell IDs do not match")

    actual_curve_types = set(curves["curve_type"].dropna().astype(str).unique())
    if actual_curve_types != set(EXPECTED_CURVE_TYPES):
        raise ValueError(
            f"Curve types are {sorted(actual_curve_types)}, expected {sorted(EXPECTED_CURVE_TYPES)}"
        )

    counts = (
        curves.groupby(["cell_id", "curve_type"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EXPECTED_CURVE_TYPES, fill_value=0)
    )
    if (counts == 0).any(axis=None):
        missing = counts.index[(counts == 0).any(axis=1)].tolist()
        raise ValueError(f"Cells missing RPT curves: {missing[:10]}")

    numeric_meta = ["efc0", "efc50_lower", "efc50_upper", "weight_lower", "weight_upper"]
    if not np.isfinite(metadata[numeric_meta].to_numpy(dtype=float)).all():
        raise ValueError("Metadata contains NaN/Inf in EFC or weight fields")

    if not np.allclose(metadata["efc0"].to_numpy(float), 0.0, atol=1e-8):
        raise ValueError("Metadata efc0 must be zero")

    lower = metadata["efc50_lower"].to_numpy(float)
    upper = metadata["efc50_upper"].to_numpy(float)
    wl = metadata["weight_lower"].to_numpy(float)
    wu = metadata["weight_upper"].to_numpy(float)

    if np.any(upper <= lower):
        raise ValueError("Invalid EFC50 bracket")
    if np.any(lower > EFC50_TARGET) or np.any(upper < EFC50_TARGET):
        raise ValueError("EFC50 bracket does not contain 50")
    if np.any((wl < 0) | (wl > 1)) or np.any((wu < 0) | (wu > 1)):
        raise ValueError("Interpolation weights outside [0, 1]")
    if not np.allclose(wl + wu, 1.0, atol=1e-8):
        raise ValueError("Interpolation weights do not sum to one")

    expected_wu = (EFC50_TARGET - lower) / (upper - lower)
    if not np.allclose(wu, expected_wu, atol=1e-7, rtol=1e-7):
        raise ValueError("weight_upper does not match EFC50 bracket formula")

    curve_efc = (
        curves.groupby(["cell_id", "curve_type"], sort=False)["rpt_efc"]
        .agg(["nunique", "first"])
        .reset_index()
    )
    if (curve_efc["nunique"] != 1).any():
        raise ValueError("Some stored RPT curves have non-constant rpt_efc")

    metadata_indexed = metadata.set_index("cell_id")
    for curve_type, metadata_column in {
        "efc0": "efc0",
        "efc50_lower": "efc50_lower",
        "efc50_upper": "efc50_upper",
    }.items():
        observed = (
            curve_efc.loc[curve_efc["curve_type"] == curve_type]
            .set_index("cell_id")["first"]
        )
        expected = metadata_indexed.loc[observed.index, metadata_column]
        if not np.allclose(observed.to_numpy(float), expected.to_numpy(float), atol=1e-8, rtol=1e-8):
            raise ValueError(f"{curve_type} rpt_efc values disagree with metadata")

    expected_current = CHEMISTRY_CONFIG[chemistry][1]
    current_medians = curves.groupby(["cell_id", "curve_type"])["current_A"].median()
    if not np.allclose(
        current_medians.to_numpy(float),
        expected_current,
        rtol=0.15,
        atol=0.05,
    ):
        raise ValueError(
            f"{chemistry} curve currents do not match expected {expected_current:g} A "
            f"{expected_direction} reference current"
        )


def select_observed_lifetimes(
    curves: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match MLP v2's default selection of uncensored 90% SOH cells."""
    censored = _as_bool_series(metadata["cycle_life_censored"], "cycle_life_censored")
    lifetime = pd.to_numeric(metadata["cycle_life_efc"], errors="coerce")
    threshold = pd.to_numeric(metadata["cycle_life_soh_threshold"], errors="coerce")

    if not np.allclose(threshold.to_numpy(float), LIFETIME_SOH_THRESHOLD, atol=1e-8, rtol=0):
        raise ValueError("Lifetime threshold differs from the v2 90% SOH contract")

    observed = (~censored) & np.isfinite(lifetime.to_numpy(float)) & (lifetime.to_numpy(float) > 0)
    selected_metadata = metadata.loc[observed].copy()
    selected_ids = set(selected_metadata["cell_id"].astype(str))
    selected_curves = curves.loc[curves["cell_id"].astype(str).isin(selected_ids)].copy()
    return selected_curves, selected_metadata


# -----------------------------------------------------------------------------
# Observation preprocessing: same deterministic transform as MLP v2
# -----------------------------------------------------------------------------
def make_observation(voltage: np.ndarray, time: np.ndarray, chemistry: str) -> np.ndarray:
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
        raise ValueError("Curve has insufficient valid time points")

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
        raise ValueError("Observation preprocessing produced NaN/Inf")
    return observation


# -----------------------------------------------------------------------------
# NSF loading and posterior means
# -----------------------------------------------------------------------------
def load_posterior(chemistry: str):
    """Load the trained NSF estimator and the chemistry prior metadata.

    We deliberately use the raw learned posterior estimator rather than
    building an MCMC or prior-rejection posterior wrapper. This is the direct
    flow-sampling diagnostic requested for the current experiment.
    """
    module = importlib.import_module(
        f"src.simulation.parameters.parameter_{chemistry}"
    )
    names = list(module.params_log.keys())
    flags = list(module.params_log_flag.values())

    log_names = {
        name.split(" [", 1)[0]
        for name, flag in zip(names, flags)
        if flag
    }
    if log_names != EXPECTED_LOG10_PARAMETERS:
        raise ValueError(
            f"{chemistry} log-parameter flags do not match expected set: {sorted(log_names)}"
        )

    prior_min = torch.tensor(
        [limits[0] for limits in module.params_log.values()], dtype=torch.float32
    )
    prior_max = torch.tensor(
        [limits[1] for limits in module.params_log.values()], dtype=torch.float32
    )
    prior = utils.BoxUniform(low=prior_min, high=prior_max)
    prior, _, _ = process_prior(prior)

    pkl_path = MODEL_DIR / chemistry / f"inference_{chemistry}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Missing NSF model: {pkl_path}")

    with open(pkl_path, "rb") as handle:
        inference = pickle.load(handle)

    # The Code Ocean-style pickle is an sbi NPE_C inference object.
    # In that representation the trained conditional density estimator is
    # stored as ``_neural_net`` and ``_posterior`` may be None.  Newer
    # DirectPosterior objects instead expose ``posterior_estimator``.
    #
    # For this diagnostic we deliberately want the RAW NSF flow, not a
    # DirectPosterior wrapper, because we want to measure prior leakage
    # rather than silently reject samples outside the prior.
    estimator = getattr(inference, "posterior_estimator", None)
    estimator_source = "posterior_estimator"

    if estimator is None:
        estimator = getattr(inference, "_posterior_estimator", None)
        estimator_source = "_posterior_estimator"

    if estimator is None:
        estimator = getattr(inference, "_neural_net", None)
        estimator_source = "_neural_net"

    if estimator is None:
        raise AttributeError(
            f"Could not find the trained NSF density estimator in {pkl_path}. "
            f"Loaded object type: {type(inference).__module__}."
            f"{type(inference).__name__}. "
            "Expected posterior_estimator, _posterior_estimator, or _neural_net."
        )

    if not hasattr(estimator, "sample"):
        raise TypeError(
            f"Loaded {estimator_source} from {pkl_path}, but it has no sample() "
            f"method. Type: {type(estimator).__module__}.{type(estimator).__name__}"
        )

    if hasattr(estimator, "to"):
        estimator = estimator.to(DEVICE)
    if hasattr(estimator, "eval"):
        estimator.eval()

    print(
        f"Loaded NSF density estimator from {pkl_path.name}: "
        f"{type(inference).__name__}.{estimator_source} -> "
        f"{type(estimator).__name__}"
    )

    return {
        "estimator": estimator,
        "param_names": names,
        "log_flags": flags,
        "prior_min": prior_min,
        "prior_max": prior_max,
        "prior_range": prior_max - prior_min,
    }


def posterior_means_batched(
    estimator,
    observations: torch.Tensor,
    prior_min: torch.Tensor,
    prior_max: torch.Tensor,
    n_samples: int,
    observation_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the learned NSF directly and return means and prior leakage.

    Direct flow sampling is intentionally unconstrained. For every observation
    we record the fraction of samples outside the training prior. This lets us
    distinguish a posterior mean from a prior-truncated mean.
    """
    means = []
    outside_fractions = []
    n_obs = observations.shape[0]
    low = prior_min.to(DEVICE)
    high = prior_max.to(DEVICE)

    # The estimator API is most robust when conditioned on one observation at a
    # time. observation_batch_size therefore controls progress/memory grouping,
    # while individual flow calls remain independent and cheap.
    for start in range(0, n_obs, observation_batch_size):
        end = min(start + observation_batch_size, n_obs)
        for i in range(start, end):
            # ConditionalDensityEstimator.sample expects the condition
            # to carry a batch dimension: (batch_size, condition_dim).
            # For one RPT we therefore pass shape (1, 100).
            x = observations[i].to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                samples = estimator.sample(
                    (n_samples,),
                    condition=x,
                )

            # Raw sbi density estimators return
            # (n_samples, batch_size, n_parameters). Remove the singleton
            # observation batch dimension for the one-RPT case.
            if samples.ndim == 3 and samples.shape[1] == 1:
                samples = samples[:, 0, :]

            if samples.ndim != 2 or samples.shape[0] != n_samples:
                raise RuntimeError(
                    f"Unexpected direct NSF sample shape {tuple(samples.shape)}"
                )

            samples = samples.float()
            outside = ((samples < low) | (samples > high)).any(dim=1)
            outside_fractions.append(outside.float().mean().item())
            means.append(samples.mean(dim=0).cpu())

        print(
            f"\rNSF direct posterior means: {end:5d}/{n_obs}",
            end="",
            flush=True,
        )

    print()
    return torch.stack(means, dim=0), torch.tensor(outside_fractions)


# -----------------------------------------------------------------------------
# Feature formatting
# -----------------------------------------------------------------------------
def result_rows(
    cell_id: str,
    chemistry: str,
    curve_type: str,
    model: dict,
    theta_mean: np.ndarray,
    metadata_row,
    source_name: str,
) -> dict:
    theta = np.asarray(theta_mean, dtype=float)
    low = model["prior_min"].numpy()
    rng = model["prior_range"].numpy()
    normalized = (theta - low) / rng

    physical = np.array(
        [10.0 ** theta[j] if flag else theta[j] for j, flag in enumerate(model["log_flags"])],
        dtype=float,
    )

    row = {
        "cell_id": cell_id,
        "chemistry": chemistry,
        "curve_type": curve_type,
        "cycle_life_efc": float(metadata_row.cycle_life_efc),
        "cycle_life_censored": bool(metadata_row.cycle_life_censored),
        "cycle_life_soh_threshold": float(metadata_row.cycle_life_soh_threshold),
        "temperature_C": float(metadata_row.temperature_C) if pd.notna(metadata_row.temperature_C) else np.nan,
        "charge_c_rate": float(metadata_row.charge_c_rate) if pd.notna(metadata_row.charge_c_rate) else np.nan,
        "discharge_c_rate": float(metadata_row.discharge_c_rate) if pd.notna(metadata_row.discharge_c_rate) else np.nan,
        "source_family": source_name,
    }

    row_dict = metadata_row._asdict()
    for column in OPTIONAL_OUTPUT_METADATA_COLUMNS:
        if column in row_dict:
            row[column] = row_dict[column]

    for name, norm, phys, theta_value in zip(
        model["param_names"], normalized, physical, theta
    ):
        row[f"{name}_normalized"] = float(norm)
        row[f"{name}_physical"] = float(phys)
        # Paper-style feature representation: log10 values for the five
        # log-uniform parameters, native physical scale otherwise.
        row[f"{name}_feature"] = float(theta_value)

    return row


def add_derived_features(results):
    """Calculate the dependent stoichiometry features locally.

    Charge RPTs (25R) infer theta_l^n/theta_l^p directly and derive
    theta_h^n/theta_h^p. Discharge RPTs (VTC5A, VTC6, MJ1) infer theta_h^n/
    theta_h^p directly and derive theta_l^n/theta_l^p. This follows the
    direction-dependent equations in the paper's Supplementary Information.
    """
    required = {
        "chemistry",
        "theta_n_init_physical",
        "theta_p_init_physical",
        "eps_s_n_physical",
        "eps_s_p_physical",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            f"Cannot calculate derived features; missing columns: {missing}"
        )

    results = results.copy()
    F_DEFAULT = 96485.0

    for column in (
        "theta_h_n_physical",
        "theta_l_n_physical",
        "theta_h_p_physical",
        "theta_l_p_physical",
        "theta_off_physical",
    ):
        results[column] = np.nan

    for chemistry in results["chemistry"].dropna().unique():
        mask = results["chemistry"] == chemistry
        direction = CHEMISTRY_DIRECTION[chemistry]
        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )

        def get_parameter(name, *aliases):
            for candidate in (name, *aliases):
                if hasattr(module, candidate):
                    return float(getattr(module, candidate))
            raise AttributeError(
                f"{chemistry}: parameter module has none of {name!r}, {aliases!r}"
            )

        area = get_parameter("area", "electrode_area")
        L_n = get_parameter("L_n", "L_negative")
        L_p = get_parameter("L_p", "L_positive")
        Cs_max_n = get_parameter("Cs_max_n", "c_s_max_n")
        Cs_max_p = get_parameter("Cs_max_p", "c_s_max_p")
        F = get_parameter("F") if hasattr(module, "F") else F_DEFAULT
        Q_As = CHEMISTRY_CONFIG[chemistry][0] * 3600.0

        en = results.loc[mask, "eps_s_n_physical"].to_numpy(dtype=float)
        ep = results.loc[mask, "eps_s_p_physical"].to_numpy(dtype=float)
        theta_n = results.loc[mask, "theta_n_init_physical"].to_numpy(dtype=float)
        theta_p = results.loc[mask, "theta_p_init_physical"].to_numpy(dtype=float)

        if np.any(~np.isfinite(en)) or np.any(~np.isfinite(ep)):
            raise ValueError(f"{chemistry}: non-finite active-material fractions")
        if np.any(en <= 0) or np.any(ep <= 0):
            raise ValueError(f"{chemistry}: active-material fractions must be > 0")

        delta_n = Q_As / (L_n * area * Cs_max_n * en * F)
        delta_p = Q_As / (L_p * area * Cs_max_p * ep * F)

        if direction == "charge":
            theta_l_n = theta_n
            theta_l_p = theta_p
            theta_h_n = theta_l_n + delta_n
            theta_h_p = theta_l_p - delta_p
        elif direction == "discharge":
            theta_h_n = theta_n
            theta_h_p = theta_p
            theta_l_n = theta_h_n - delta_n
            theta_l_p = theta_h_p + delta_p
        else:
            raise ValueError(
                f"Unsupported RPT direction for {chemistry}: {direction}"
            )

        theta_off = theta_l_p + theta_l_n * (
            L_n * Cs_max_n * en / (L_p * Cs_max_p * ep)
        )

        results.loc[mask, "theta_h_n_physical"] = theta_h_n
        results.loc[mask, "theta_l_n_physical"] = theta_l_n
        results.loc[mask, "theta_h_p_physical"] = theta_h_p
        results.loc[mask, "theta_l_p_physical"] = theta_l_p
        results.loc[mask, "theta_off_physical"] = theta_off

    for name in (
        "theta_h_n",
        "theta_l_n",
        "theta_h_p",
        "theta_l_p",
        "theta_off",
    ):
        results[f"{name}_feature"] = results[f"{name}_physical"]

    return results


# -----------------------------------------------------------------------------
# Source processing
# -----------------------------------------------------------------------------
def process_source(
    source_name: str,
    n_posterior_samples: int,
    observation_batch_size: int,
    include_censored: bool,
) -> Path:
    files = DATA_SOURCES[source_name]
    chemistry = files["chemistry"]

    print(f"Source: {source_name}")
    print(f"Chemistry: {chemistry}")
    print(f"Device: {DEVICE}")
    print(f"Posterior samples per RPT: {n_posterior_samples}")
    print("Sampling mode: direct NSF flow (no MCMC, no rejection sampling)")

    curves = pd.read_parquet(files["curves"], columns=CURVE_COLUMNS)
    metadata = pd.read_parquet(files["metadata"])
    validate_dataset(curves, metadata, chemistry)

    if not include_censored:
        curves, metadata = select_observed_lifetimes(curves, metadata)

    print(f"Cells: {len(metadata)}")
    print(f"Curve points: {len(curves):,}")

    model = load_posterior(chemistry)

    grouped_curves = {
        (cell_id, curve_type): group
        for (cell_id, curve_type), group in curves.groupby(
            ["cell_id", "curve_type"], sort=False
        )
    }

    meta_rows = list(metadata.itertuples(index=False))
    observations = []
    keys = []

    # Preserve the v2 curve structure: infer EFC0, lower and upper separately.
    for meta in meta_rows:
        cell_id = meta.cell_id
        for curve_type in EXPECTED_CURVE_TYPES:
            group = grouped_curves[(cell_id, curve_type)]
            obs = make_observation(
                group["voltage_V"].to_numpy(),
                group["time_s"].to_numpy(),
                chemistry,
            )
            observations.append(obs)
            keys.append((cell_id, curve_type, meta))

    observations_t = torch.tensor(np.asarray(observations), dtype=torch.float32)
    theta_means_t, outside_fractions = posterior_means_batched(
        model["estimator"],
        observations_t,
        prior_min=model["prior_min"],
        prior_max=model["prior_max"],
        n_samples=n_posterior_samples,
        observation_batch_size=observation_batch_size,
    )
    theta_means = theta_means_t.numpy()

    # Convert posterior means back into v2-like rows.
    raw_rows = []
    by_cell_curve = {}
    for key, theta_mean, outside_fraction in zip(keys, theta_means, outside_fractions.numpy()):
        cell_id, curve_type, meta = key
        row = result_rows(
            cell_id,
            chemistry,
            curve_type,
            model,
            theta_mean,
            meta,
            source_name,
        )
        row["direct_nsf_prior_outside_fraction"] = float(outside_fraction)
        raw_rows.append(row)
        by_cell_curve[(cell_id, curve_type)] = row

    # Build EFC50 by interpolating posterior means in theta coordinates.
    # Since normalized coordinates are affine in theta, this is equivalent to
    # interpolating normalized outputs as done by extract_features_mlpv2.py.
    final_rows = []
    for meta in meta_rows:
        cell_id = meta.cell_id
        efc0_row = by_cell_curve[(cell_id, "efc0")]
        lower = by_cell_curve[(cell_id, "efc50_lower")]
        upper = by_cell_curve[(cell_id, "efc50_upper")]

        weight_lower = float(meta.weight_lower)
        weight_upper = float(meta.weight_upper)

        row = dict(efc0_row)
        row["curve_type"] = "efc0"
        final_rows.append(row)

        efc50 = dict(upper)
        efc50["curve_type"] = "efc50"
        efc50["direct_nsf_prior_outside_fraction"] = (
            weight_lower * float(lower["direct_nsf_prior_outside_fraction"])
            + weight_upper * float(upper["direct_nsf_prior_outside_fraction"])
        )

        for name in model["param_names"]:
            n0 = float(lower[f"{name}_normalized"])
            n1 = float(upper[f"{name}_normalized"])
            theta50 = weight_lower * n0 + weight_upper * n1
            efc50[f"{name}_normalized"] = theta50

            low = float(model["prior_min"][list(model["param_names"]).index(name)])
            rng = float(model["prior_range"][list(model["param_names"]).index(name)])
            theta_value = low + theta50 * rng
            flag = model["log_flags"][list(model["param_names"]).index(name)]
            physical = 10.0 ** theta_value if flag else theta_value

            efc50[f"{name}_feature"] = theta_value
            efc50[f"{name}_physical"] = physical

        final_rows.append(efc50)

    result = pd.DataFrame(final_rows)

    # Calculate the dependent stoichiometry features locally so NSF and MLP v2
    # use the same explicit charge/discharge convention.
    result = add_derived_features(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"experimental_inference_nsf2_{source_name}.csv"
    result.to_csv(output_path, index=False)

    print(f"Saved {len(result)} rows to {output_path}")
    print(result.groupby("curve_type")["cell_id"].nunique())
    return output_path


def merge_outputs() -> Path:
    files = sorted(OUTPUT_DIR.glob("experimental_inference_nsf2_*.csv"))
    files = [p for p in files if p.name != "experimental_inference_nsf2_all.csv"]
    if not files:
        raise FileNotFoundError(f"No NSF source outputs found in {OUTPUT_DIR}")

    frames = [pd.read_csv(p) for p in files]
    result = pd.concat(frames, ignore_index=True)
    output = OUTPUT_DIR / "experimental_inference_nsf2_all.csv"
    result.to_csv(output, index=False)
    print(f"Merged {len(files)} source files -> {output}")
    print(result.groupby(["source_family", "chemistry", "curve_type"])["cell_id"].nunique())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(DATA_SOURCES), help="One source to process")
    parser.add_argument("--merge", action="store_true", help="Merge completed source CSVs")
    parser.add_argument("--n-posterior-samples", type=int, default=DEFAULT_N_POSTERIOR_SAMPLES)
    parser.add_argument("--observation-batch-size", type=int, default=2)
    parser.add_argument(
        "--include-censored",
        action="store_true",
        help="Do not apply MLP v2's default observed-lifetime/censoring filter",
    )
    args = parser.parse_args()

    if args.merge:
        merge_outputs()
        return

    if args.source is None:
        parser.error("--source is required unless --merge is used")

    process_source(
        source_name=args.source,
        n_posterior_samples=args.n_posterior_samples,
        observation_batch_size=args.observation_batch_size,
        include_censored=args.include_censored,
    )


if __name__ == "__main__":
    main()
