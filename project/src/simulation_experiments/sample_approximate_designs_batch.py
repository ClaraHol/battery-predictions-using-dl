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
        --chemistry MJ1 --accuracy realistic --n 50000 --resume

    python sample_approximate_designs.py --help

The saved active-material fractions are DERIVED values required to reproduce
the known capacity. They are useful as a plausibility check; they do not have
to replace eps_s_n / eps_s_p as SBI targets in later voltage simulations.
"""

import argparse
import csv
import importlib
import json
import sys
import time
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
# separator_porosity_abs: uniform +/- absolute error for eps_e,s
# particle_factor: log-uniform multiplicative uncertainty for R_n and R_p
# rel_csmax: uniform +/- fractional error for C_s,max
# theta_abs: uniform +/- absolute error for each stoichiometric endpoint
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
        "epse_s": sample_absolute(
            rng, ref["epse_s"], profile["separator_porosity_abs"]
        ),
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

    voltage_band = V_cut_ub - V_cut_lb
    if abs((ocv_empty - V_cut_lb) / voltage_band) > ocv_tol:
        return False, "low_ocv_mismatch", None
    if abs((ocv_full - V_cut_ub) / voltage_band) > ocv_tol:
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

    # Write the completed tensor file atomically so a timed-out job cannot
    # leave a corrupted .pt that looks like a finished dataset.
    temp_pt = output_path.with_suffix(".pt.tmp")
    torch.save({"fields": SAVE_FIELDS, "designs": data, "metadata": metadata}, temp_pt)
    temp_pt.replace(output_path)

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
# Command-line configuration and batch-friendly execution
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "models" / "Approximate_designs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample capacity- and OCV-consistent approximate battery designs."
    )
    parser.add_argument(
        "--chemistry", choices=[*CHEMISTRY_CONFIG, "all"], default="all",
        help="One chemistry per HPC job, or 'all' (default).",
    )
    parser.add_argument("--n", type=int, default=50000, help="Accepted designs per chemistry.")
    parser.add_argument(
        "--accuracy", choices=ACCURACY_PROFILES, default="realistic",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-inactive-fraction", type=float, default=0.02,
        help="Minimum electrode volume reserved for inactive material.",
    )
    parser.add_argument(
        "--max-attempts-factor", type=int, default=200000,
        help="Max proposed designs = this factor * --n.",
    )
    parser.add_argument(
        "--capacity-source", choices=("used", "nominal"), default="used",
        help="Capacity used in the active-material-fraction constraint (default: used).",
    )
    parser.add_argument(
        "--capacity-ah", type=float, default=None,
        help="Override target capacity [Ah]; otherwise use capacity-source.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional filename suffix to keep configurations separate.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000,
        help="Save a restartable checkpoint every N accepted designs; 0 disables.",
    )
    parser.add_argument(
        "--checkpoint-seconds", type=float, default=300,
        help="Also checkpoint at least this often during slow rejection loops; 0 disables.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing partial checkpoint, if any.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Permit replacing an existing completed output file.",
    )

    for name in ACCURACY_PROFILES["realistic"]:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float, default=None,
            help=f"Override {name} of the selected accuracy profile.",
        )

    args = parser.parse_args(argv)
    if args.n < 1 or args.max_attempts_factor < 1:
        parser.error("--n and --max-attempts-factor must both be positive")
    if args.checkpoint_every < 0 or args.checkpoint_seconds < 0:
        parser.error("--checkpoint-every and --checkpoint-seconds must be >= 0")
    if not (0 <= args.min_inactive_fraction < 1):
        parser.error("--min-inactive-fraction in [0,1)")
    if args.capacity_ah is not None and args.capacity_ah <= 0:
        parser.error("--capacity-ah must be > 0")
    if args.tag and not all(c.isalnum() or c in "-_" for c in args.tag):
        parser.error("--tag may contain only letters, numbers, '-' and '_'")
    args.profile = dict(ACCURACY_PROFILES[args.accuracy])
    for key in args.profile:
        override = getattr(args, key)
        if override is not None:
            args.profile[key] = override
    for key, value in args.profile.items():
        if not np.isfinite(value) or value < 0 or (key == "particle_factor" and value < 1):
            parser.error(f"Invalid {key}: {value}; must be >= 0 (particle-factor >= 1)")
    if args.profile["rel_geometry"] >= 1 or args.profile["rel_separator"] >= 1 or args.profile["rel_csmax"] >= 1:
        parser.error("Relative perturbation limits must be below 1 to retain positive samples")
    return args


def resolve_capacity(module, args):
    if args.capacity_ah is not None:
        return args.capacity_ah
    attr = "capa_used" if args.capacity_source == "used" else "capa_nom"
    if not hasattr(module, attr):
        raise AttributeError(f"{module.__name__} does not define {attr}; use --capacity-ah to override")
    return float(getattr(module, attr))


def atomic_save(obj, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def sample_chemistry(chemistry, chemistry_index, args):
    ocv_tol = args.profile["ocv_tol"]
    module_name = CHEMISTRY_CONFIG[chemistry]
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
    output = args.output_dir / f"designs_{chemistry}_{args.accuracy}{suffix}.pt"
    checkpoint = output.with_suffix(".partial.pt")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Completed output already exists: {output} (use --overwrite or --tag)")

    # Stable chemistry-specific seeds in both single-chemistry and all modes.
    seed = args.seed + chemistry_index
    signature = {
        "chemistry": chemistry,
        "n": args.n,
        "accuracy": args.accuracy,
        "accuracy_profile": args.profile,
        "seed": seed,
        "capacity_Ah": capacity_Ah,
        "ocv_tolerance_fraction": ocv_tol,
        "min_inactive_fraction": args.min_inactive_fraction,
        "max_attempts_factor": args.max_attempts_factor,
        "reference_design": ref,
    }
    rng = np.random.default_rng(seed)
    accepted, rejected, attempts = [], Counter(), 0
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"Partial checkpoint already exists: {checkpoint} (use --resume)")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError(f"Checkpoint settings differ from requested run: {checkpoint}")
        accepted = saved["accepted"]
        rejected = Counter(saved["rejected"])
        attempts = saved["attempts"]
        rng.bit_generator.state = saved["rng_state"]
        print(f"Resumed {chemistry}: {len(accepted)}/{args.n} accepted, {attempts} attempts", flush=True)
    else:
        print(f"Starting {chemistry}: {args.n} designs; capacity={capacity_Ah:.6g} Ah; profile={args.accuracy}", flush=True)

    max_attempts = args.max_attempts_factor * args.n
    last_checkpoint_time = time.monotonic()

    def save_checkpoint():
        nonlocal last_checkpoint_time
        atomic_save({
            "signature": signature,
            "accepted": [
                {key: float(value) for key, value in d.items()}
                for d in accepted
            ],
            "rejected": dict(rejected),
            "attempts": attempts,
            "rng_state": rng.bit_generator.state,
        }, checkpoint)
        last_checkpoint_time = time.monotonic()

    while len(accepted) < args.n and attempts < max_attempts:
        attempts += 1
        proposal = sample_design(rng, ref, args.profile)
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
            min_inactive_fraction=args.min_inactive_fraction,
            ocv_tol=args.profile["ocv_tol"],
        )
        if valid:
            accepted.append(enriched)
            if len(accepted) % 1000 == 0 or len(accepted) == args.n:
                print(
                    f"{chemistry}: accepted {len(accepted):>7}/{args.n} "
                    f"after {attempts:>9} proposals ({len(accepted)/attempts:.2%})",
                    flush=True,
                )
            if args.checkpoint_every and len(accepted) % args.checkpoint_every == 0:
                save_checkpoint()
        else:
            rejected[reason] += 1

        if (args.checkpoint_seconds and
                time.monotonic() - last_checkpoint_time >= args.checkpoint_seconds):
            save_checkpoint()
            print(f"{chemistry}: checkpoint at {len(accepted)}/{args.n} accepted; {attempts} attempts", flush=True)

    if len(accepted) != args.n:
        save_checkpoint()
        print("Rejections:", rejected.most_common(), flush=True)
        raise RuntimeError(
            f"{chemistry}: found {len(accepted)} designs from {attempts} proposals; "
            f"raise --max-attempts-factor or revise sampling settings"
        )

    metadata = {
        **signature,
        "params_module": module_name,
        "n_valid": len(accepted),
        "attempts": attempts,
        "acceptance_rate": len(accepted) / attempts,
        "capacity_source": ("override" if args.capacity_ah is not None else args.capacity_source),
        "V_cut_lb": V_cut_lb,
        "V_cut_ub": V_cut_ub,
        "eps_s_n_bounds": list(eps_s_n_bounds),
        "eps_s_p_bounds": list(eps_s_p_bounds),
        "rejection_counts": dict(rejected),
        "note": "No PyBaMM simulation; OCP checked at proposed stoichiometric endpoints.",
    }
    save_results(output, accepted, metadata)
    checkpoint.unlink(missing_ok=True)
    print(f"Saved {len(accepted)} designs -> {output}", flush=True)
    print(f"Acceptance: {len(accepted)/attempts:.2%}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    selected = list(CHEMISTRY_CONFIG) if args.chemistry == "all" else [args.chemistry]
    for chemistry in selected:
        chemistry_index = list(CHEMISTRY_CONFIG).index(chemistry)
        sample_chemistry(chemistry, chemistry_index, args)
    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
