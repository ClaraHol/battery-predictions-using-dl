#!/usr/bin/env python3
"""
Sample capacity- and OCV-constrained approximate battery designs.

Purpose
-------
This sampler is intended for the approximate-design generalization experiment.

For each chemistry it samples uncertain *fixed/nuisance* design quantities
(electrode geometry, porosity, particle size, C_s,max, and stoichiometric
endpoints), then rejects proposals that are inconsistent with:

  1. the known beginning-of-life cell capacity,
  2. physically plausible active-material volume fractions,
  3. electrode-volume constraints, and
  4. the chemistry-specific OCP functions and cell voltage limits.

The important change relative to the previous version is that the accepted
stoichiometric endpoints are now also used to define the two stoichiometry
parameters that should be included in the downstream 11-dimensional SBI
target.

The sampler itself still does NOT run PyBaMM.

Downstream simulation structure
-------------------------------
For each simulation:

  - draw one accepted approximate design from this file;
  - use its nuisance design parameters in PyBaMM;
  - use `theta_n_init` and `theta_p_init` from the same accepted design as
    two of the 11 SBI targets;
  - sample the other 9 SBI targets in the normal way;
  - run the fixed chemistry-specific RPT protocol.

For a charge RPT, the simulation starts from the discharged state:
    theta_n_init = theta_n_min
    theta_p_init = theta_p_max

For a discharge RPT, the simulation starts from the charged state:
    theta_n_init = theta_n_max
    theta_p_init = theta_p_min

The derived `eps_s_n_required` and `eps_s_p_required` are retained as
capacity-consistency diagnostics. They are NOT automatically declared to be
SBI targets by this script.

Typical use
-----------
    python sample_approximate_designs.py \
        --chemistry VTC5A --accuracy realistic --n 100000 \
        --num-workers 8 --resume

    python sample_approximate_designs.py --chemistry all --accuracy realistic
"""

import argparse
import csv
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# Make project root importable when this file lives under project/src/...
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


# =============================================================================
# Chemistry configuration
# =============================================================================

CHEMISTRY_CONFIG = {
    "25R": {
        "params_module": "src.simulation.parameters.parameter_25R",
        "rpt_direction": "charge",
    },
    "HG2": {
        "params_module": "src.simulation.parameters.parameter_HG2",
        "rpt_direction": "charge",
    },
    "M1A": {
        "params_module": "src.simulation.parameters.parameter_M1A",
        "rpt_direction": "charge",
    },
    "MJ1": {
        "params_module": "src.simulation.parameters.parameter_MJ1",
        "rpt_direction": "discharge",
    },
    "PA": {
        "params_module": "src.simulation.parameters.parameter_PA",
        "rpt_direction": "discharge",
    },
    "PBC": {
        "params_module": "src.simulation.parameters.parameter_PBC",
        "rpt_direction": "discharge",
    },
    "PD": {
        "params_module": "src.simulation.parameters.parameter_PD",
        "rpt_direction": "discharge",
    },
    "VTC5A": {
        "params_module": "src.simulation.parameters.parameter_VTC5A",
        "rpt_direction": "discharge",
    },
    "VTC6": {
        "params_module": "src.simulation.parameters.parameter_VTC6",
        "rpt_direction": "discharge",
    },
}


# =============================================================================
# Approximation profiles
# =============================================================================

# Uncertainty around the reference/estimated design.
#
# rel_geometry:
#     uniform +/- fractional error for A, L_n, L_p
#
# rel_separator:
#     uniform +/- fractional error for L_s
#
# porosity_abs:
#     uniform +/- absolute error for eps_e,n and eps_e,p
#
# separator_porosity_abs:
#     uniform +/- absolute error for eps_e,s
#
# particle_factor:
#     log-uniform multiplicative uncertainty for R_n and R_p
#
# rel_csmax:
#     uniform +/- fractional error for C_s,max
#
# theta_abs:
#     uniform +/- absolute error for each stoichiometric endpoint
#
# ocv_tol:
#     tolerated endpoint OCV mismatch expressed as a fraction of the full
#     cell voltage window

ACCURACY_PROFILES = {
    "good": {
        "rel_geometry": 0.10,
        "rel_separator": 0.10,
        "porosity_abs": 0.03,
        "separator_porosity_abs": 0.03,
        "particle_factor": 1.20,
        "rel_csmax": 0.05,
        "theta_abs": 0.03,
        "ocv_tol": 0.25,
    },
    "realistic": {
        "rel_geometry": 0.25,
        "rel_separator": 0.30,
        "porosity_abs": 0.07,
        "separator_porosity_abs": 0.05,
        "particle_factor": 1.50,
        "rel_csmax": 0.10,
        "theta_abs": 0.07,
        "ocv_tol": 0.30,
    },
    "rough": {
        "rel_geometry": 0.50,
        "rel_separator": 0.50,
        "porosity_abs": 0.12,
        "separator_porosity_abs": 0.10,
        "particle_factor": 2.00,
        "rel_csmax": 0.20,
        "theta_abs": 0.12,
        "ocv_tol": 0.40,
    },
}


# =============================================================================
# Output schema
# =============================================================================

# These quantities are uncertain fixed inputs to the forward model.
NUISANCE_FIELDS = [
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
]

# Keep all four endpoints because they are required to represent the sampled
# cell design and are useful for audit/debugging.
STOICH_ENDPOINT_FIELDS = [
    "theta_n_min",
    "theta_n_max",
    "theta_p_min",
    "theta_p_max",
]

# These are the two values that should be appended to the other 9 SBI targets
# in the downstream 11-dimensional inference dataset.
SBI_STOICH_TARGET_FIELDS = [
    "theta_n_init",
    "theta_p_init",
]

# Capacity/OCV diagnostics. eps_s_*_required are not automatically SBI targets.
DERIVED_FIELDS = [
    "eps_s_n_required",
    "eps_s_p_required",
    "ocv_empty",
    "ocv_full",
]

SAVE_FIELDS = (
    NUISANCE_FIELDS
    + STOICH_ENDPOINT_FIELDS
    + SBI_STOICH_TARGET_FIELDS
    + DERIVED_FIELDS
)


DEFAULT_OUTPUT_DIR = ROOT / "models" / "Approximate_designs"


# =============================================================================
# Sampling helpers
# =============================================================================

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


# =============================================================================
# Reference design
# =============================================================================

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
    """
    Sample one approximate battery design around the chemistry reference.

    The four stoichiometric endpoints are sampled jointly with the other design
    quantities. They are later constrained by both capacity and OCV checks.
    """
    return {
        "area": sample_relative(
            rng, ref["area"], profile["rel_geometry"]
        ),
        "L_n": sample_relative(
            rng, ref["L_n"], profile["rel_geometry"]
        ),
        "L_p": sample_relative(
            rng, ref["L_p"], profile["rel_geometry"]
        ),
        "L_s": sample_relative(
            rng, ref["L_s"], profile["rel_separator"]
        ),
        "epse_n": sample_absolute(
            rng, ref["epse_n"], profile["porosity_abs"]
        ),
        "epse_p": sample_absolute(
            rng, ref["epse_p"], profile["porosity_abs"]
        ),
        "epse_s": sample_absolute(
            rng, ref["epse_s"], profile["separator_porosity_abs"]
        ),
        "Rp_n": sample_factor(
            rng, ref["Rp_n"], profile["particle_factor"]
        ),
        "Rp_p": sample_factor(
            rng, ref["Rp_p"], profile["particle_factor"]
        ),
        "Cs_max_n": sample_relative(
            rng, ref["Cs_max_n"], profile["rel_csmax"]
        ),
        "Cs_max_p": sample_relative(
            rng, ref["Cs_max_p"], profile["rel_csmax"]
        ),
        "theta_n_min": sample_absolute(
            rng, ref["theta_n_min"], profile["theta_abs"]
        ),
        "theta_n_max": sample_absolute(
            rng, ref["theta_n_max"], profile["theta_abs"]
        ),
        "theta_p_min": sample_absolute(
            rng, ref["theta_p_min"], profile["theta_abs"]
        ),
        "theta_p_max": sample_absolute(
            rng, ref["theta_p_max"], profile["theta_abs"]
        ),
    }


def select_initial_stoichiometries(design, rpt_direction):
    """
    Select the two stoichiometries used as initial-state SBI targets.

    Convention used by this sampler:
      - theta_n_min / theta_p_max correspond to the discharged/empty state.
      - theta_n_max / theta_p_min correspond to the charged/full state.

    Therefore:
      - charge RPT starts discharged,
      - discharge RPT starts charged.
    """
    if rpt_direction == "charge":
        return design["theta_n_min"], design["theta_p_max"]

    if rpt_direction == "discharge":
        return design["theta_n_max"], design["theta_p_min"]

    raise ValueError(
        f"Unsupported rpt_direction={rpt_direction!r}; "
        "expected 'charge' or 'discharge'."
    )


# =============================================================================
# Physical consistency checks
# =============================================================================

def validate_design(
    design,
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
    rpt_direction,
):
    """
    Return:
        valid, reason, enriched_design

    Capacity consistency
    --------------------
        Q = F/3600 * A * L * eps_s * Cs_max * Delta(theta)

    We solve this equation for the active-material volume fraction eps_s that
    would be required for the sampled geometry + C_s,max + stoichiometric
    window to reproduce the known beginning-of-life cell capacity.

    The proposal is retained only if those required active-material fractions
    lie inside the chemistry's physically allowed eps_s ranges.

    OCV consistency
    ---------------
        V_empty = U_p(theta_p_max) - U_n(theta_n_min)
        V_full  = U_p(theta_p_min) - U_n(theta_n_max)

    The accepted design is enriched with:
        theta_n_init
        theta_p_init

    These two values are the direction-specific stoichiometry targets for the
    downstream 11-dimensional SBI model.
    """

    positive_fields = (
        "area",
        "L_n",
        "L_p",
        "L_s",
        "Rp_n",
        "Rp_p",
        "Cs_max_n",
        "Cs_max_p",
    )

    for name in positive_fields:
        if not np.isfinite(design[name]) or design[name] <= 0:
            return False, f"invalid_{name}", None

    for name in ("epse_n", "epse_p", "epse_s"):
        if not np.isfinite(design[name]) or not (0.0 < design[name] < 1.0):
            return False, f"invalid_{name}", None

    for name in STOICH_ENDPOINT_FIELDS:
        if not np.isfinite(design[name]) or not (0.0 <= design[name] <= 1.0):
            return False, f"invalid_{name}", None

    delta_n = design["theta_n_max"] - design["theta_n_min"]
    delta_p = design["theta_p_max"] - design["theta_p_min"]

    if delta_n <= 0:
        return False, "invalid_theta_n_window", None

    if delta_p <= 0:
        return False, "invalid_theta_p_window", None

    eps_s_n_required = (
        3600.0
        * capacity_Ah
        / (
            F
            * design["area"]
            * design["L_n"]
            * design["Cs_max_n"]
            * delta_n
        )
    )

    eps_s_p_required = (
        3600.0
        * capacity_Ah
        / (
            F
            * design["area"]
            * design["L_p"]
            * design["Cs_max_p"]
            * delta_p
        )
    )

    if not (
        eps_s_n_bounds[0]
        <= eps_s_n_required
        <= eps_s_n_bounds[1]
    ):
        return False, "eps_s_n_capacity", None

    if not (
        eps_s_p_bounds[0]
        <= eps_s_p_required
        <= eps_s_p_bounds[1]
    ):
        return False, "eps_s_p_capacity", None

    # Reserve some electrode volume for binder/conductive additive.
    max_used_fraction = 1.0 - min_inactive_fraction

    if eps_s_n_required + design["epse_n"] > max_used_fraction:
        return False, "negative_volume_fractions", None

    if eps_s_p_required + design["epse_p"] > max_used_fraction:
        return False, "positive_volume_fractions", None

    try:
        ocv_empty = (
            ocp_value(cathode_ocp, design["theta_p_max"])
            - ocp_value(anode_ocp, design["theta_n_min"])
        )

        ocv_full = (
            ocp_value(cathode_ocp, design["theta_p_min"])
            - ocp_value(anode_ocp, design["theta_n_max"])
        )

    except Exception:
        return False, "ocp_evaluation", None

    if not np.isfinite(ocv_empty) or not np.isfinite(ocv_full):
        return False, "nonfinite_ocv", None

    if ocv_empty >= ocv_full:
        return False, "reversed_ocv", None

    voltage_band = V_cut_ub - V_cut_lb

    if voltage_band <= 0:
        raise ValueError(
            f"Invalid voltage limits: V_cut_lb={V_cut_lb}, "
            f"V_cut_ub={V_cut_ub}"
        )

    if abs((ocv_empty - V_cut_lb) / voltage_band) > ocv_tol:
        return False, "low_ocv_mismatch", None

    if abs((ocv_full - V_cut_ub) / voltage_band) > ocv_tol:
        return False, "high_ocv_mismatch", None

    theta_n_init, theta_p_init = select_initial_stoichiometries(
        design,
        rpt_direction,
    )

    out = dict(design)
    out["theta_n_init"] = theta_n_init
    out["theta_p_init"] = theta_p_init
    out["eps_s_n_required"] = eps_s_n_required
    out["eps_s_p_required"] = eps_s_p_required
    out["ocv_empty"] = ocv_empty
    out["ocv_full"] = ocv_full

    return True, "accepted", out


# =============================================================================
# Parallel proposal workers
# =============================================================================

# Read-only state installed once in each worker process. Keeping the chemistry
# module and OCP callables inside the worker avoids trying to pickle PyBaMM
# functions for every proposal.
_WORKER_CONTEXT = None


def get_num_workers(requested=None, default=8):
    """Resolve worker count, preferring an explicit value then the LSF allocation."""
    if requested is not None:
        return requested

    lsf_workers = os.environ.get("LSB_DJOB_NUMPROC")
    if lsf_workers is not None:
        try:
            value = int(lsf_workers)
            if value > 0:
                return value
        except ValueError:
            pass

    return min(default, os.cpu_count() or default)


def proposal_seed(global_seed, proposal_index):
    """
    Deterministic seed for one proposal.

    The seed depends only on the global seed and proposal index, not on worker
    count, task scheduling, or batch size. This makes accepted designs
    reproducible across serial/parallel runs and after checkpoint resume.
    """
    return (
        int(global_seed)
        + 1_000_003 * int(proposal_index)
    ) % (2**63 - 1)


def init_design_worker(
    module_name,
    ref,
    profile,
    validation_kwargs,
    global_seed,
):
    """Install chemistry-specific read-only state in one worker process."""
    global _WORKER_CONTEXT

    module = importlib.import_module(module_name)

    _WORKER_CONTEXT = {
        "ref": ref,
        "profile": profile,
        "validation_kwargs": validation_kwargs,
        "global_seed": int(global_seed),
        "anode_ocp": module.anode_ocp,
        "cathode_ocp": module.cathode_ocp,
    }


def sample_and_validate_one(proposal_index):
    """Generate and validate one independent approximate-design proposal."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Parallel design worker was not initialized")

    rng = np.random.default_rng(
        proposal_seed(
            _WORKER_CONTEXT["global_seed"],
            proposal_index,
        )
    )

    proposal = sample_design(
        rng,
        _WORKER_CONTEXT["ref"],
        _WORKER_CONTEXT["profile"],
    )

    valid, reason, enriched = validate_design(
        proposal,
        anode_ocp=_WORKER_CONTEXT["anode_ocp"],
        cathode_ocp=_WORKER_CONTEXT["cathode_ocp"],
        **_WORKER_CONTEXT["validation_kwargs"],
    )

    return int(proposal_index), bool(valid), reason, enriched


def choose_batch_size(
    *,
    remaining,
    accepted_count,
    effective_attempts,
    max_batch_size,
    num_workers,
):
    """
    Choose a proposal batch size.

    Large batches amortize multiprocessing overhead. Near the target, use the
    observed acceptance rate to reduce unnecessary proposals in the final
    batch while still keeping workers occupied.
    """
    if remaining <= 0:
        return 0

    if accepted_count == 0 or effective_attempts == 0:
        estimated_needed = remaining
    else:
        acceptance_rate = accepted_count / effective_attempts
        estimated_needed = int(
            np.ceil(1.10 * remaining / max(acceptance_rate, 1e-6))
        )

    return min(
        max_batch_size,
        max(num_workers, estimated_needed),
    )


# =============================================================================
# Saving
# =============================================================================

def save_results(output_path, designs, metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.tensor(
        [
            [design[name] for name in SAVE_FIELDS]
            for design in designs
        ],
        dtype=torch.float32,
    )

    payload = {
        "fields": SAVE_FIELDS,
        "designs": data,
        "metadata": metadata,
    }

    # Write completed tensor file atomically so an interrupted job cannot leave
    # a corrupted file that looks finished.
    temp_pt = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temp_pt)
    temp_pt.replace(output_path)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAVE_FIELDS)
        writer.writeheader()

        for design in designs:
            writer.writerow(
                {name: design[name] for name in SAVE_FIELDS}
            )

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as file:
        json.dump(metadata, file, indent=2)

    return csv_path, meta_path


def atomic_save(obj, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    temp_path.replace(path)


# =============================================================================
# Command-line configuration
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sample capacity- and OCV-constrained approximate battery designs "
            "with direction-specific stoichiometry targets."
        )
    )

    parser.add_argument(
        "--chemistry",
        choices=[*CHEMISTRY_CONFIG, "all"],
        default="all",
        help="One chemistry per HPC job, or 'all' (default).",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=50000,
        help="Accepted designs per chemistry.",
    )

    parser.add_argument(
        "--accuracy",
        choices=ACCURACY_PROFILES,
        default="realistic",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Parallel worker processes. Default: LSB_DJOB_NUMPROC on LSF; "
            "otherwise min(8, available CPUs)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help=(
            "Maximum number of independent design proposals submitted per "
            "parallel batch (default: 4096)."
        ),
    )

    parser.add_argument(
        "--min-inactive-fraction",
        type=float,
        default=0.02,
        help="Minimum electrode volume reserved for inactive material.",
    )

    parser.add_argument(
        "--max-attempts-factor",
        type=int,
        default=200000,
        help="Maximum proposed designs = this factor * --n.",
    )

    parser.add_argument(
        "--capacity-source",
        choices=("used", "nominal"),
        default="used",
        help=(
            "Capacity used in the capacity-consistency constraint "
            "(default: used)."
        ),
    )

    parser.add_argument(
        "--capacity-ah",
        type=float,
        default=None,
        help="Override target capacity [Ah]; otherwise use capacity-source.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional filename suffix to keep configurations separate.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help=(
            "Save a restartable checkpoint every N accepted designs; "
            "0 disables."
        ),
    )

    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=300,
        help=(
            "Also checkpoint at least this often during slow rejection loops; "
            "0 disables."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial checkpoint, if any.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing an existing completed output file.",
    )

    # Allow individual approximation-profile entries to be overridden.
    for name in ACCURACY_PROFILES["realistic"]:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=None,
            help=f"Override {name} of the selected accuracy profile.",
        )

    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error("--n must be positive")

    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be positive")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.max_attempts_factor < 1:
        parser.error("--max-attempts-factor must be positive")

    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")

    if args.checkpoint_seconds < 0:
        parser.error("--checkpoint-seconds must be >= 0")

    if not (0 <= args.min_inactive_fraction < 1):
        parser.error("--min-inactive-fraction must lie in [0, 1)")

    if args.capacity_ah is not None and args.capacity_ah <= 0:
        parser.error("--capacity-ah must be > 0")

    if args.tag and not all(
        char.isalnum() or char in "-_"
        for char in args.tag
    ):
        parser.error(
            "--tag may contain only letters, numbers, '-' and '_'"
        )

    args.profile = dict(ACCURACY_PROFILES[args.accuracy])

    for key in args.profile:
        override = getattr(args, key)
        if override is not None:
            args.profile[key] = override

    for key, value in args.profile.items():
        if not np.isfinite(value):
            parser.error(f"Invalid {key}: {value}")

        if value < 0:
            parser.error(f"Invalid {key}: {value}; must be >= 0")

        if key == "particle_factor" and value < 1:
            parser.error(
                f"Invalid particle_factor: {value}; must be >= 1"
            )

    for key in ("rel_geometry", "rel_separator", "rel_csmax"):
        if args.profile[key] >= 1:
            parser.error(
                f"{key} must be below 1 to retain positive samples"
            )

    return args


# =============================================================================
# Chemistry-specific execution
# =============================================================================

def resolve_capacity(module, args):
    if args.capacity_ah is not None:
        return args.capacity_ah

    attr = (
        "capa_used"
        if args.capacity_source == "used"
        else "capa_nom"
    )

    if not hasattr(module, attr):
        raise AttributeError(
            f"{module.__name__} does not define {attr}; "
            "use --capacity-ah to override"
        )

    return float(getattr(module, attr))


def sample_chemistry(chemistry, chemistry_index, args):
    chemistry_config = CHEMISTRY_CONFIG[chemistry]
    module_name = chemistry_config["params_module"]
    rpt_direction = chemistry_config["rpt_direction"]

    module = importlib.import_module(module_name)

    ref = get_reference_design(module)
    capacity_Ah = resolve_capacity(module, args)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""

    output = (
        args.output_dir
        / f"designs_{chemistry}_{args.accuracy}{suffix}.pt"
    )

    checkpoint = output.with_suffix(".partial.pt")

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Completed output already exists: {output} "
            "(use --overwrite or --tag)"
        )

    # Stable chemistry-specific seed in both single-chemistry and all modes.
    seed = args.seed + chemistry_index
    num_workers = get_num_workers(args.num_workers)

    # This is deliberately part of the checkpoint signature because it changes
    # how proposal seeds are generated relative to the old sequential sampler.
    seed_scheme = "global_seed_plus_1000003_times_proposal_index_v1"

    signature = {
        "chemistry": chemistry,
        "params_module": module_name,
        "rpt_direction": rpt_direction,
        "n": args.n,
        "accuracy": args.accuracy,
        "accuracy_profile": args.profile,
        "seed": seed,
        "proposal_seed_scheme": seed_scheme,
        "capacity_Ah": capacity_Ah,
        "capacity_source": (
            "override"
            if args.capacity_ah is not None
            else args.capacity_source
        ),
        "V_cut_lb": V_cut_lb,
        "V_cut_ub": V_cut_ub,
        "ocv_tolerance_fraction": args.profile["ocv_tol"],
        "eps_s_n_bounds": list(eps_s_n_bounds),
        "eps_s_p_bounds": list(eps_s_p_bounds),
        "min_inactive_fraction": args.min_inactive_fraction,
        "max_attempts_factor": args.max_attempts_factor,
        "reference_design": ref,
        "nuisance_fields": NUISANCE_FIELDS,
        "stoich_endpoint_fields": STOICH_ENDPOINT_FIELDS,
        "sbi_stoich_target_fields": SBI_STOICH_TARGET_FIELDS,
        "derived_fields": DERIVED_FIELDS,
    }

    validation_kwargs = {
        "capacity_Ah": capacity_Ah,
        "F": F,
        "V_cut_lb": V_cut_lb,
        "V_cut_ub": V_cut_ub,
        "eps_s_n_bounds": eps_s_n_bounds,
        "eps_s_p_bounds": eps_s_p_bounds,
        "min_inactive_fraction": args.min_inactive_fraction,
        "ocv_tol": args.profile["ocv_tol"],
        "rpt_direction": rpt_direction,
    }

    accepted = []
    rejected = Counter()

    # effective_attempts counts proposals through the Nth retained design.
    # computed_attempts includes any extra proposals that were already running
    # in the final parallel batch when the target was reached.
    effective_attempts = 0
    computed_attempts = 0
    next_proposal_index = 0
    extra_results_after_target = 0
    extra_valid_after_target = 0

    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(
                f"Partial checkpoint already exists: {checkpoint} "
                "(use --resume)"
            )

        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if saved["signature"] != signature:
            raise ValueError(
                f"Checkpoint settings differ from requested run: {checkpoint}. "
                "Old sequential checkpoints are intentionally incompatible "
                "with the deterministic parallel proposal scheme."
            )

        accepted = saved["accepted"]
        rejected = Counter(saved["rejected"])
        effective_attempts = int(saved["effective_attempts"])
        computed_attempts = int(saved["computed_attempts"])
        next_proposal_index = int(saved["next_proposal_index"])

        print(
            f"Resumed {chemistry}: "
            f"{len(accepted)}/{args.n} accepted, "
            f"{effective_attempts} effective proposals; "
            f"workers={num_workers}",
            flush=True,
        )

    else:
        print(
            f"Starting {chemistry}: "
            f"{args.n} designs; "
            f"capacity={capacity_Ah:.6g} Ah; "
            f"profile={args.accuracy}; "
            f"RPT={rpt_direction}; "
            f"workers={num_workers}; "
            f"batch_size={args.batch_size}",
            flush=True,
        )

    max_attempts = args.max_attempts_factor * args.n
    last_checkpoint_time = time.monotonic()

    def save_checkpoint():
        nonlocal last_checkpoint_time

        atomic_save(
            {
                "signature": signature,
                "accepted": [
                    {
                        key: float(value)
                        for key, value in design.items()
                    }
                    for design in accepted
                ],
                "rejected": dict(rejected),
                "effective_attempts": effective_attempts,
                "computed_attempts": computed_attempts,
                "next_proposal_index": next_proposal_index,
            },
            checkpoint,
        )

        last_checkpoint_time = time.monotonic()

    # The worker pool is created once per chemistry. Each worker imports the
    # chemistry parameter module once and then handles many independent
    # proposals, avoiding repeated module/OCP setup overhead.
    with mp.Pool(
        processes=num_workers,
        initializer=init_design_worker,
        initargs=(
            module_name,
            ref,
            args.profile,
            validation_kwargs,
            seed,
        ),
    ) as pool:
        while (
            len(accepted) < args.n
            and effective_attempts < max_attempts
        ):
            remaining = args.n - len(accepted)

            this_batch_size = choose_batch_size(
                remaining=remaining,
                accepted_count=len(accepted),
                effective_attempts=effective_attempts,
                max_batch_size=args.batch_size,
                num_workers=num_workers,
            )

            this_batch_size = min(
                this_batch_size,
                max_attempts - effective_attempts,
            )

            proposal_indices = list(
                range(
                    next_proposal_index,
                    next_proposal_index + this_batch_size,
                )
            )

            chunksize = max(
                1,
                this_batch_size // max(1, num_workers * 8),
            )

            results = pool.map(
                sample_and_validate_one,
                proposal_indices,
                chunksize=chunksize,
            )

            computed_attempts += len(results)
            next_proposal_index += len(results)

            target_reached_at = None

            # pool.map preserves input order, so accepted-design ordering is
            # deterministic even though the work itself executes in parallel.
            for position, (
                proposal_index,
                valid,
                reason,
                enriched,
            ) in enumerate(results):
                effective_attempts += 1

                if valid:
                    accepted.append(enriched)
                else:
                    rejected[reason] += 1

                if len(accepted) == args.n:
                    target_reached_at = position
                    break

            if target_reached_at is not None:
                unused_results = results[target_reached_at + 1 :]
                extra_results_after_target += len(unused_results)
                extra_valid_after_target += sum(
                    1 for _, valid, _, _ in unused_results if valid
                )

            if (
                len(accepted) % 1000 < this_batch_size
                or len(accepted) == args.n
            ):
                print(
                    f"{chemistry}: "
                    f"accepted {len(accepted):>7}/{args.n} "
                    f"after {effective_attempts:>9} proposals "
                    f"({len(accepted) / effective_attempts:.2%}); "
                    f"computed={computed_attempts}",
                    flush=True,
                )

            if len(accepted) >= args.n:
                break

            should_checkpoint_count = (
                args.checkpoint_every
                and len(accepted) > 0
                and (
                    len(accepted) // args.checkpoint_every
                    != (len(accepted) - max(1, this_batch_size))
                    // args.checkpoint_every
                )
            )

            should_checkpoint_time = (
                args.checkpoint_seconds
                and (
                    time.monotonic() - last_checkpoint_time
                    >= args.checkpoint_seconds
                )
            )

            if should_checkpoint_count or should_checkpoint_time:
                save_checkpoint()

                print(
                    f"{chemistry}: checkpoint at "
                    f"{len(accepted)}/{args.n} accepted; "
                    f"{effective_attempts} effective proposals",
                    flush=True,
                )

    if len(accepted) != args.n:
        save_checkpoint()

        print(
            "Rejections:",
            rejected.most_common(),
            flush=True,
        )

        raise RuntimeError(
            f"{chemistry}: found {len(accepted)} designs "
            f"from {effective_attempts} proposals; "
            "raise --max-attempts-factor or revise sampling settings"
        )

    metadata = {
        **signature,
        "n_valid": len(accepted),
        "attempts": effective_attempts,
        "effective_attempts": effective_attempts,
        "computed_attempts": computed_attempts,
        "parallel_extra_results_after_target": extra_results_after_target,
        "parallel_extra_valid_after_target": extra_valid_after_target,
        "acceptance_rate": len(accepted) / effective_attempts,
        "computed_efficiency": len(accepted) / computed_attempts,
        "rejection_counts": dict(rejected),
        "num_workers_used": num_workers,
        "max_batch_size": args.batch_size,
        "notes": [
            "No PyBaMM simulation is run by this sampler.",
            (
                "Design proposals are independent and evaluated in parallel. "
                "Each proposal has a deterministic seed derived from the "
                "global seed and proposal index, so results are independent "
                "of worker count and batch size."
            ),
            (
                "theta_n_init and theta_p_init are direction-specific "
                "stoichiometry values intended as two of the downstream "
                "11 SBI targets."
            ),
            (
                "eps_s_n_required and eps_s_p_required are capacity-"
                "consistency diagnostics and are not automatically used "
                "as SBI targets."
            ),
            (
                "The downstream simulator should use the fixed RPT protocol "
                "for this chemistry rather than random charge/discharge "
                "rates or SOC windows."
            ),
        ],
    }

    save_results(
        output,
        accepted,
        metadata,
    )

    checkpoint.unlink(missing_ok=True)

    print(
        f"Saved {len(accepted)} designs -> {output}",
        flush=True,
    )

    print(
        f"Acceptance: {len(accepted) / effective_attempts:.2%}; "
        f"computed proposals: {computed_attempts}",
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)

    selected = (
        list(CHEMISTRY_CONFIG)
        if args.chemistry == "all"
        else [args.chemistry]
    )

    all_chemistries = list(CHEMISTRY_CONFIG)

    for chemistry in selected:
        chemistry_index = all_chemistries.index(chemistry)

        sample_chemistry(
            chemistry,
            chemistry_index,
            args,
        )

    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
