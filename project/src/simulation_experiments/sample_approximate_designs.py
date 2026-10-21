#!/usr/bin/env python3
"""
Sample plausible fixed battery-design parameterizations around a known design.

This script does NOT run a PyBaMM simulation. It only:
  1. samples uncertain fixed design parameters,
  2. checks whether the proposed design can reproduce the known cell capacity
     using physically plausible active-material volume fractions,
  3. checks whether the known OCP functions give plausible empty/full-cell OCVs,
  4. continues until exactly n valid designs have been accepted,
  5. saves the accepted designs to both .pt and .csv.

The OCP functions are assumed to be known exactly and are imported from the
chemistry-specific parameter module.

Typical use:
    python sample_approximate_designs.py \
        --chemistry M1A \
        --accuracy realistic \
        --n 50000

The saved active-material fractions are DERIVED values required to reproduce
the known capacity. They are useful as a plausibility check; they do not have
to replace eps_s_n / eps_s_p as SBI targets in later voltage simulations.
"""

import argparse
import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Make project root importable when this file lives under project/src/...
sys.path.append(str(Path(__file__).resolve().parents[2]))


CHEMISTRY_CONFIG = {
    "25R": "src.simulation.parameters.parameter_25R",
    "HG2": "src.simulation.parameters.parameter_HG2",
    "M1A": "src.simulation.parameters.parameter_M1A",
    "MJ1": "src.simulation.parameters.parameter_MJ1",
    "PA": "src.simulation.parameters.parameter_PA",
    "PBC": "src.simulation.parameters.parameter_PBC",
    "PD": "src.simulation.parameters.parameter_PD",
    "VTC5A": "src.simulation.parameters.parameter_VTC5A",
    "VTC6": "src.simulation.parameters.parameter_VTC6",
}

# Uncertainty around the reference/estimated design.
#
# rel_geometry: uniform +/- fractional error for A, L_n, L_p
# rel_separator: uniform +/- fractional error for L_s
# porosity_abs:  uniform +/- absolute error for eps_e,n and eps_e,p
# particle_factor: log-uniform multiplicative uncertainty for R_n and R_p
# rel_csmax: uniform +/- fractional error for C_s,max
# theta_abs: uniform +/- absolute error for each stoichiometric endpoint
ACCURACY_PROFILES = {
    "good": {
        "rel_geometry": 0.10,
        "rel_separator": 0.10,
        "porosity_abs": 0.03,
        "particle_factor": 1.20,
        "rel_csmax": 0.05,
        "theta_abs": 0.03,
    },
    "realistic": {
        "rel_geometry": 0.25,
        "rel_separator": 0.30,
        "porosity_abs": 0.07,
        "particle_factor": 1.50,
        "rel_csmax": 0.10,
        "theta_abs": 0.07,
    },
    "rough": {
        "rel_geometry": 0.50,
        "rel_separator": 0.50,
        "porosity_abs": 0.12,
        "particle_factor": 2.00,
        "rel_csmax": 0.20,
        "theta_abs": 0.12,
    },
}

SAVE_FIELDS = [
    "area",
    "L_n",
    "L_p",
    "L_s",
    "epse_n",
    "epse_p",
    "epse_s",
    "Rp_n",
    "Rp_p",
    "Cs_max_n",
    "Cs_max_p",
    "theta_n_min",
    "theta_n_max",
    "theta_p_min",
    "theta_p_max",
    "eps_s_n_required",
    "eps_s_p_required",
    "ocv_empty",
    "ocv_full",
]


def sample_relative(rng, value, rel_error):
    """Uniform relative perturbation: value * (1 + U[-rel_error, rel_error])."""
    return value * (1.0 + rng.uniform(-rel_error, rel_error))


def sample_absolute(rng, value, abs_error):
    """Uniform absolute perturbation."""
    return value + rng.uniform(-abs_error, abs_error)


def sample_factor(rng, value, factor):
    """Log-uniform multiplicative perturbation in [value/factor, value*factor]."""
    log_f = np.log(factor)
    return value * np.exp(rng.uniform(-log_f, log_f))


def as_float(value):
    """Convert float / numpy scalar / scalar PyBaMM expression to Python float."""
    if hasattr(value, "evaluate"):
        value = value.evaluate()
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"Expected scalar OCP value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def ocp_value(fn, theta):
    return as_float(fn(float(theta)))


def get_reference_design(module):
    names = [
        "area",
        "L_n",
        "L_p",
        "L_s",
        "epse_n",
        "epse_p",
        "epse_s",
        "Rp_n",
        "Rp_p",
        "Cs_max_n",
        "Cs_max_p",
        "theta_n_min",
        "theta_n_max",
        "theta_p_min",
        "theta_p_max",
    ]
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"{module.__name__} is missing required fixed parameters: {missing}"
        )
    return {name: float(getattr(module, name)) for name in names}


def sample_design(rng, ref, profile):
    return {
        "area": sample_relative(rng, ref["area"], profile["rel_geometry"]),
        "L_n": sample_relative(rng, ref["L_n"], profile["rel_geometry"]),
        "L_p": sample_relative(rng, ref["L_p"], profile["rel_geometry"]),
        "L_s": sample_relative(rng, ref["L_s"], profile["rel_separator"]),
        "epse_n": sample_absolute(rng, ref["epse_n"], profile["porosity_abs"]),
        "epse_p": sample_absolute(rng, ref["epse_p"], profile["porosity_abs"]),
        # Separator porosity is deliberately kept fixed: the historical
        # parameter files mostly treat it as a standard/default quantity.
        "epse_s": ref["epse_s"],
        "Rp_n": sample_factor(rng, ref["Rp_n"], profile["particle_factor"]),
        "Rp_p": sample_factor(rng, ref["Rp_p"], profile["particle_factor"]),
        "Cs_max_n": sample_relative(rng, ref["Cs_max_n"], profile["rel_csmax"]),
        "Cs_max_p": sample_relative(rng, ref["Cs_max_p"], profile["rel_csmax"]),
        "theta_n_min": sample_absolute(rng, ref["theta_n_min"], profile["theta_abs"]),
        "theta_n_max": sample_absolute(rng, ref["theta_n_max"], profile["theta_abs"]),
        "theta_p_min": sample_absolute(rng, ref["theta_p_min"], profile["theta_abs"]),
        "theta_p_max": sample_absolute(rng, ref["theta_p_max"], profile["theta_abs"]),
    }


def validate_design(
    d,
    *,
    capacity_Ah,
    F,
    anode_ocp,
    cathode_ocp,
    V_cut_lb,
    V_cut_ub,
    eps_s_n_bounds,
    eps_s_p_bounds,
    min_inactive_fraction,
    ocv_tol,
):
    """
    Return (valid, reason, enriched_design).

    Capacity consistency:
        Q = F/3600 * A * L * eps_s * Cs_max * Delta(theta)

    We solve this for the active-material fraction eps_s required for the
    proposed geometry to reproduce the known beginning-of-life capacity.

    OCV consistency:
        V_empty = U_p(theta_p,max) - U_n(theta_n,min)
        V_full  = U_p(theta_p,min) - U_n(theta_n,max)
    """
    for name in ("area", "L_n", "L_p", "L_s", "Rp_n", "Rp_p", "Cs_max_n", "Cs_max_p"):
        if not np.isfinite(d[name]) or d[name] <= 0:
            return False, f"invalid_{name}", None

    for name in ("epse_n", "epse_p", "epse_s"):
        if not np.isfinite(d[name]) or not (0.0 < d[name] < 1.0):
            return False, f"invalid_{name}", None

    for name in ("theta_n_min", "theta_n_max", "theta_p_min", "theta_p_max"):
        if not np.isfinite(d[name]) or not (0.0 <= d[name] <= 1.0):
            return False, f"invalid_{name}", None

    delta_n = d["theta_n_max"] - d["theta_n_min"]
    delta_p = d["theta_p_max"] - d["theta_p_min"]
    if delta_n <= 0:
        return False, "invalid_theta_n_window", None
    if delta_p <= 0:
        return False, "invalid_theta_p_window", None

    eps_s_n = (
        3600.0 * capacity_Ah
        / (F * d["area"] * d["L_n"] * d["Cs_max_n"] * delta_n)
    )
    eps_s_p = (
        3600.0 * capacity_Ah
        / (F * d["area"] * d["L_p"] * d["Cs_max_p"] * delta_p)
    )

    if not (eps_s_n_bounds[0] <= eps_s_n <= eps_s_n_bounds[1]):
        return False, "eps_s_n_capacity", None
    if not (eps_s_p_bounds[0] <= eps_s_p <= eps_s_p_bounds[1]):
        return False, "eps_s_p_capacity", None

    # Reserve some electrode volume for binder/conductive additive.
    max_used_fraction = 1.0 - min_inactive_fraction
    if eps_s_n + d["epse_n"] > max_used_fraction:
        return False, "negative_volume_fractions", None
    if eps_s_p + d["epse_p"] > max_used_fraction:
        return False, "positive_volume_fractions", None

    try:
        ocv_empty = (
            ocp_value(cathode_ocp, d["theta_p_max"])
            - ocp_value(anode_ocp, d["theta_n_min"])
        )
        ocv_full = (
            ocp_value(cathode_ocp, d["theta_p_min"])
            - ocp_value(anode_ocp, d["theta_n_max"])
        )
    except Exception:
        return False, "ocp_evaluation", None

    if not np.isfinite(ocv_empty) or not np.isfinite(ocv_full):
        return False, "nonfinite_ocv", None
    if ocv_empty >= ocv_full:
        return False, "reversed_ocv", None

    if abs(ocv_empty - V_cut_lb) > ocv_tol:
        return False, "low_ocv_mismatch", None
    if abs(ocv_full - V_cut_ub) > ocv_tol:
        return False, "high_ocv_mismatch", None

    out = dict(d)
    out["eps_s_n_required"] = eps_s_n
    out["eps_s_p_required"] = eps_s_p
    out["ocv_empty"] = ocv_empty
    out["ocv_full"] = ocv_full
    return True, "accepted", out


def save_results(output_path, designs, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.tensor(
        [[d[name] for name in SAVE_FIELDS] for d in designs],
        dtype=torch.float32,
    )

    torch.save(
        {
            "fields": SAVE_FIELDS,
            "designs": data,
            "metadata": metadata,
        },
        output_path,
    )

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAVE_FIELDS)
        writer.writeheader()
        writer.writerows({name: d[name] for name in SAVE_FIELDS} for d in designs)

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return csv_path, meta_path



# =============================================================================
# Settings
# =============================================================================

N = 50000
ACCURACY = "realistic"  # "good", "realistic", or "rough"
SEED = 0

OCV_TOL = 0.30
MIN_INACTIVE_FRACTION = 0.02
MAX_ATTEMPTS_FACTOR = 200000

CHEMISTRIES = [
    "25R",
    "HG2",
    "M1A",
    "MJ1",
    "PA",
    "PBC",
    "PD",
    "VTC5A",
    "VTC6",
]

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "models" / "Approximate_designs"


def sample_chemistry(chemistry, chemistry_index):
    module_name = CHEMISTRY_CONFIG[chemistry]
    module = importlib.import_module(module_name)

    ref = get_reference_design(module)
    profile = ACCURACY_PROFILES[ACCURACY]

    capacity_Ah = float(
        getattr(module, "capa_used", getattr(module, "capa_nom"))
    )
    V_cut_lb = float(module.V_cut_lb)
    V_cut_ub = float(module.V_cut_ub)
    F = float(getattr(module, "F", 96485.0))

    eps_s_n_bounds = (
        float(getattr(module, "epss_n_lb", 0.4)),
        float(getattr(module, "epss_n_ub", 0.9)),
    )
    eps_s_p_bounds = (
        float(getattr(module, "epss_p_lb", 0.4)),
        float(getattr(module, "epss_p_ub", 0.9)),
    )

    rng = np.random.default_rng(SEED + chemistry_index)

    accepted = []
    rejected = Counter()
    attempts = 0
    max_attempts = MAX_ATTEMPTS_FACTOR * N

    print(f"\n{'=' * 70}")
    print(f"{chemistry} | {ACCURACY} uncertainty | target = {N}")
    print(f"{'=' * 70}")

    while len(accepted) < N and attempts < max_attempts:
        attempts += 1
        proposal = sample_design(rng, ref, profile)

        valid, reason, enriched = validate_design(
            proposal,
            capacity_Ah=capacity_Ah,
            F=F,
            anode_ocp=module.anode_ocp,
            cathode_ocp=module.cathode_ocp,
            V_cut_lb=V_cut_lb,
            V_cut_ub=V_cut_ub,
            eps_s_n_bounds=eps_s_n_bounds,
            eps_s_p_bounds=eps_s_p_bounds,
            min_inactive_fraction=MIN_INACTIVE_FRACTION,
            ocv_tol=OCV_TOL,
        )

        if valid:
            accepted.append(enriched)
            if len(accepted) % 5000 == 0 or len(accepted) == N:
                print(
                    f"Accepted {len(accepted):>7}/{N} "
                    f"after {attempts:>8} proposals "
                    f"({len(accepted) / attempts:.2%})"
                )
        else:.
            rejected[reason] += 1

    if len(accepted) != N:
        print("\nRejection counts:")
        for reason, count in rejected.most_common():
            print(f"  {reason:28s} {count}")
        raise RuntimeError(
            f"{chemistry}: only found {len(accepted)} valid designs "
            f"after {attempts} proposals."
        )

    metadata = {
        "chemistry": chemistry,
        "params_module": module_name,
        "accuracy": ACCURACY,
        "accuracy_profile": profile,
        "n_valid": N,
        "attempts": attempts,
        "acceptance_rate": N / attempts,
        "seed": SEED + chemistry_index,
        "capacity_Ah": capacity_Ah,
        "V_cut_lb": V_cut_lb,
        "V_cut_ub": V_cut_ub,
        "ocv_tolerance_V": OCV_TOL,
        "eps_s_n_bounds": list(eps_s_n_bounds),
        "eps_s_p_bounds": list(eps_s_p_bounds),
        "min_inactive_fraction": MIN_INACTIVE_FRACTION,
        "reference_design": ref,
        "rejection_counts": dict(rejected),
        "note": (
            "No PyBaMM simulation was run. OCP functions were evaluated only at "
            "the proposed stoichiometric endpoints."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"designs_{chemistry}_{ACCURACY}.pt"
    save_results(output, accepted, metadata)

    print(f"Saved {N} valid {chemistry} designs.")
    print(f"Acceptance rate: {N / attempts:.2%}")

    if rejected:
        print("Most common rejections:")
        for reason, count in rejected.most_common(5):
            print(f"  {reason:28s} {count}")


if __name__ == "__main__":
    if ACCURACY not in ACCURACY_PROFILES:
        raise ValueError(
            f"Unknown ACCURACY '{ACCURACY}'. "
            f"Choose from {list(ACCURACY_PROFILES)}."
        )

    for i, chemistry in enumerate(CHEMISTRIES):
        sample_chemistry(chemistry, i)

    print("\nFinished all chemistries.")
