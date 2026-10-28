#!/usr/bin/env python3
"""
Generate a fixed number of successful approximate-design SBI simulations using fixed RPT conditions.

This script matches `sample_approximate_designs_11d.py`.

For each simulation:
    1. choose one accepted approximate design;
    2. sample the first 9 SBI parameters from the chemistry-specific priors;
    3. take theta_n_init and theta_p_init from that SAME accepted design;
    4. temporarily apply the approximate fixed design parameters to the
       chemistry parameter module;
    5. call params_setting() to construct the PyBaMM parameter dictionary;
    6. run the ordinary fixed-condition battery simulator;
    7. construct the 100-dimensional observation used for SBI;
    8. save the 11-dimensional target and observation.

The experimental protocol is not randomized. There is no random:
    - charge/discharge direction,
    - C-rate,
    - SOC start/end window,
    - partial-cycle width.

The two stoichiometry targets are therefore correlated with the same
capacity/OCV-constrained approximate design used to construct the simulation.

Dataset format:
    torch.save((theta, x), dataset_path)

with:
    theta.shape == (N_valid, 11)
    x.shape     == (N_valid, 100)
"""

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d


# =============================================================================
# Project imports
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.simulation.battery_simulator import simulator


# =============================================================================
# Fixed chemistry-specific RPT settings
# =============================================================================
#
# Current sign follows the existing PyBaMM/project convention:
#     negative -> charge
#     positive -> discharge
#
# These currents and rated capacities are retained from the previous
# simulation configuration. Voltage cutoffs are read directly from each
# chemistry parameter module.

CHEMISTRY_CONFIG = {
    "25R": {
        "params_module": "src.simulation.parameters.parameter_25R",
        "Q_rated_Ah": 2.5,
        "current_A": -1.25,
        "rpt_direction": "charge",
        "dod": 1.0,
    },
    "HG2": {
        "params_module": "src.simulation.parameters.parameter_HG2",
        "Q_rated_Ah": 3.0,
        "current_A": -1.5,
        "rpt_direction": "charge",
        "dod": 1.0,
    },
    "M1A": {
        "params_module": "src.simulation.parameters.parameter_M1A",
        "Q_rated_Ah": 1.1,
        "current_A": -0.55,
        "rpt_direction": "charge",
        "dod": 1.0,
    },
    "MJ1": {
        "params_module": "src.simulation.parameters.parameter_MJ1",
        "Q_rated_Ah": 3.5,
        "current_A": 3.5,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
    "PA": {
        "params_module": "src.simulation.parameters.parameter_PA",
        "Q_rated_Ah": 74.273,
        "current_A": 24.33,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
    "PBC": {
        "params_module": "src.simulation.parameters.parameter_PBC",
        "Q_rated_Ah": 76.0,
        "current_A": 25.43,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
    "PD": {
        "params_module": "src.simulation.parameters.parameter_PD",
        "Q_rated_Ah": 83.456,
        "current_A": 28.0,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
    "VTC5A": {
        "params_module": "src.simulation.parameters.parameter_VTC5A",
        "Q_rated_Ah": 2.5,
        "current_A": 2.5,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
    "VTC6": {
        "params_module": "src.simulation.parameters.parameter_VTC6",
        "Q_rated_Ah": 3.0,
        "current_A": 0.6,
        "rpt_direction": "discharge",
        "dod": 1.0,
    },
}


# =============================================================================
# Approximate-design schema
# =============================================================================

# Only these sampled quantities need to override module-level fixed design
# values before params_setting() is called.
#
# The sampled stoichiometric endpoints are NOT module overrides:
# theta_n_init and theta_p_init are passed directly as params[9] and params[10].
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

SBI_STOICH_TARGET_FIELDS = [
    "theta_n_init",
    "theta_p_init",
]

NUM_INDEPENDENT_TARGETS = 9
NUM_SBI_TARGETS = 11
NUM_OBSERVATION_POINTS = 100
VOLTAGE_NOISE_STD_V = 0.005


# =============================================================================
# Worker state
# =============================================================================

_WORKER_MODULE = None
_WORKER_PARAMS_LOG = None
_WORKER_PARAMS_LOG_FLAG = None
_WORKER_PARAMS_SETTING = None
_WORKER_CONFIG = None
_WORKER_V_CUT_LB = None
_WORKER_V_CUT_UB = None


def init_worker(module_name, config):
    """Initialize one multiprocessing worker."""
    global _WORKER_MODULE
    global _WORKER_PARAMS_LOG
    global _WORKER_PARAMS_LOG_FLAG
    global _WORKER_PARAMS_SETTING
    global _WORKER_CONFIG
    global _WORKER_V_CUT_LB
    global _WORKER_V_CUT_UB

    module = importlib.import_module(module_name)

    _WORKER_MODULE = module
    _WORKER_PARAMS_LOG = module.params_log
    _WORKER_PARAMS_LOG_FLAG = module.params_log_flag
    _WORKER_PARAMS_SETTING = module.params_setting
    _WORKER_CONFIG = dict(config)
    _WORKER_V_CUT_LB = float(module.V_cut_lb)
    _WORKER_V_CUT_UB = float(module.V_cut_ub)


# =============================================================================
# Design loading
# =============================================================================

def load_design_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Approximate-design file not found: {path}")

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected approximate-design payload to be a dict, "
            f"got {type(payload).__name__}"
        )

    fields = list(payload["fields"])
    designs = payload["designs"]
    metadata = payload.get("metadata", {})

    if not torch.is_tensor(designs):
        designs = torch.as_tensor(designs, dtype=torch.float32)

    designs = designs.float()

    if designs.ndim != 2:
        raise ValueError(
            f"Expected 2-D design tensor, got shape {tuple(designs.shape)}"
        )

    if designs.shape[1] != len(fields):
        raise ValueError(
            f"Design width {designs.shape[1]} does not match "
            f"{len(fields)} field names"
        )

    required_fields = NUISANCE_FIELDS + SBI_STOICH_TARGET_FIELDS
    missing = [name for name in required_fields if name not in fields]

    if missing:
        raise ValueError(
            f"Approximate-design file is missing required fields: {missing}"
        )

    return fields, designs, metadata


# =============================================================================
# Target sampling and conversion
# =============================================================================

def sample_first_nine_targets(rng, params_log, n_samples):
    """
    Sample the first 9 SBI targets using the same target representation as
    the original simulation scripts.

    Log-scaled parameters remain in log10 space in the saved theta tensor.
    """
    bounds = list(params_log.values())

    if len(bounds) < NUM_SBI_TARGETS:
        raise ValueError(
            f"params_log has {len(bounds)} entries; expected at least 11"
        )

    low = np.asarray(
        [bounds[i][0] for i in range(NUM_INDEPENDENT_TARGETS)],
        dtype=np.float64,
    )
    high = np.asarray(
        [bounds[i][1] for i in range(NUM_INDEPENDENT_TARGETS)],
        dtype=np.float64,
    )

    return rng.uniform(
        low=low,
        high=high,
        size=(n_samples, NUM_INDEPENDENT_TARGETS),
    )


def theta_to_physical(theta_11, params_log_flag):
    """
    Convert the 11-D saved SBI representation to physical values for
    params_setting().
    """
    flags = list(params_log_flag.values())

    if len(flags) < NUM_SBI_TARGETS:
        raise ValueError(
            f"params_log_flag has {len(flags)} entries; expected at least 11"
        )

    physical = []

    for i, value in enumerate(theta_11):
        value = float(value)

        # The first 9 parameters use the legacy log-flag convention.
        # The final 2 parameters are physical stoichiometries supplied by the
        # approximate-design sampler and must never be log-transformed.
        if i < NUM_INDEPENDENT_TARGETS and flags[i] == 1:
            value = 10.0 ** value

        physical.append(value)

    return physical


# =============================================================================
# Approximate-design application
# =============================================================================

def apply_design_for_params_setting(module, design):
    """
    Temporarily override the fixed design values used by params_setting().

    This works because params_setting() looks up area, L_n, L_p, epse_n, etc.
    in the parameter module's global namespace when it constructs params_dict.

    Returns the original values so they can be restored immediately after
    params_setting() has produced the PyBaMM parameter dictionary.
    """
    original = {}

    for name in NUISANCE_FIELDS:
        if not hasattr(module, name):
            raise AttributeError(
                f"{module.__name__} does not define {name!r}"
            )

        original[name] = getattr(module, name)
        setattr(module, name, float(design[name]))

    return original


def restore_design(module, original):
    for name, value in original.items():
        setattr(module, name, value)


# =============================================================================
# Observation construction
# =============================================================================

def build_observation(
    time_s,
    voltage_v,
    *,
    q_rated_ah,
    current_a,
    dod,
    v_cut_lb,
    v_cut_ub,
):
    """
    Paper-style observation:

        y_obs = sqrt(t_norm^2 + V_norm^2)

        t_norm = t / t_typ
        t_typ  = 3600 * Q_rated * DOD / |I|
        V_norm = (V - V_lb) / (V_ub - V_lb)
    """
    t_typ = 3600.0 * q_rated_ah * dod / abs(current_a)

    t_norm = time_s / t_typ
    v_norm = (voltage_v - v_cut_lb) / (v_cut_ub - v_cut_lb)

    return np.sqrt(t_norm**2 + v_norm**2)


# =============================================================================
# Single simulation
# =============================================================================

def simulate_one(task):
    (
        theta9,
        design_index,
        design_values,
        design_fields,
        noise_seed,
    ) = task

    design = {
        name: float(value)
        for name, value in zip(design_fields, design_values)
    }

    theta_11 = np.concatenate(
        [
            np.asarray(theta9, dtype=np.float64),
            np.asarray(
                [
                    design["theta_n_init"],
                    design["theta_p_init"],
                ],
                dtype=np.float64,
            ),
        ]
    )

    try:
        physical_params = theta_to_physical(
            theta_11,
            _WORKER_PARAMS_LOG_FLAG,
        )

        # Only params_setting() needs the module globals to be overridden.
        original_design = apply_design_for_params_setting(
            _WORKER_MODULE,
            design,
        )

        try:
            params = _WORKER_PARAMS_SETTING(physical_params)
        finally:
            restore_design(
                _WORKER_MODULE,
                original_design,
            )

        current_a = float(_WORKER_CONFIG["current_A"])
        q_rated_ah = float(_WORKER_CONFIG["Q_rated_Ah"])
        dod = float(_WORKER_CONFIG["dod"])
        direction = _WORKER_CONFIG["rpt_direction"]

        if direction == "charge" and current_a >= 0:
            raise ValueError(
                "Charge RPT is configured with non-negative current."
            )

        if direction == "discharge" and current_a <= 0:
            raise ValueError(
                "Discharge RPT is configured with non-positive current."
            )

        # Fixed full RPT duration. PyBaMM can terminate earlier when a voltage
        # cutoff is reached.
        t_sim = 3600.0 * q_rated_ah * dod / abs(current_a)

        results = simulator(
            params,
            current_a,
            t_sim,
        )

        if isinstance(results, str):
            raise RuntimeError(results)

        voltage = np.asarray(
            results["Terminal voltage [V]"].entries,
            dtype=np.float64,
        )
        time_s = np.asarray(
            results["Time [s]"].entries,
            dtype=np.float64,
        )

        if (
            len(time_s) < 2
            or len(voltage) < 2
            or not np.isfinite(time_s).all()
            or not np.isfinite(voltage).all()
        ):
            raise RuntimeError("Invalid time/voltage output")

        # Remove duplicate timestamps.
        _, unique_idx = np.unique(time_s, return_index=True)
        unique_idx = np.sort(unique_idx)

        time_s = time_s[unique_idx]
        voltage = voltage[unique_idx]

        if len(time_s) < 2:
            raise RuntimeError("Fewer than two unique timestamps")

        interpolation_time = np.linspace(
            0.0,
            time_s[-1],
            NUM_OBSERVATION_POINTS,
        )

        voltage_interp = interp1d(
            time_s,
            voltage,
            kind="slinear",
        )(interpolation_time)

        noise_rng = np.random.default_rng(int(noise_seed))
        voltage_interp += noise_rng.normal(
            0.0,
            VOLTAGE_NOISE_STD_V,
            size=NUM_OBSERVATION_POINTS,
        )

        y_obs = build_observation(
            interpolation_time,
            voltage_interp,
            q_rated_ah=q_rated_ah,
            current_a=current_a,
            dod=dod,
            v_cut_lb=_WORKER_V_CUT_LB,
            v_cut_ub=_WORKER_V_CUT_UB,
        )

        if not np.isfinite(y_obs).all():
            raise RuntimeError("Non-finite normalized observation")

        return {
            "theta": torch.as_tensor(theta_11, dtype=torch.float32),
            "x": torch.as_tensor(y_obs, dtype=torch.float32),
            "design_index": int(design_index),
            "valid": True,
            "error": "",
        }

    except Exception as exc:
        return {
            "theta": torch.full(
                (NUM_SBI_TARGETS,),
                float("nan"),
                dtype=torch.float32,
            ),
            "x": torch.full(
                (NUM_OBSERVATION_POINTS,),
                float("nan"),
                dtype=torch.float32,
            ),
            "design_index": int(design_index),
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# =============================================================================
# Chunked execution
# =============================================================================

def get_num_workers(default=4):
    n = os.environ.get("LSB_DJOB_NUMPROC")

    if n is not None:
        return int(n)

    return min(default, os.cpu_count() or default)


def atomic_torch_save(obj, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def make_design_attempt_tasks(
    *,
    design_indices,
    attempt_numbers,
    global_seed,
    design_tensor,
    design_fields,
    params_log,
):
    """
    Build one attempt for each requested design.

    The approximate design is fixed. Only the first 9 SBI target parameters
    are redrawn between retries for that design.

    Sampling is deterministic in (global_seed, design_index, attempt_number),
    so an interrupted/restarted run reproduces the same proposal for every
    design attempt.
    """
    tasks = []

    for design_index, attempt_number in zip(
        design_indices,
        attempt_numbers,
    ):
        design_index = int(design_index)
        attempt_number = int(attempt_number)

        # Stable, design-specific RNG stream. The modulo keeps the seed inside
        # NumPy's accepted uint64 range while still separating designs/retries.
        local_seed = (
            int(global_seed)
            + 1_000_003 * design_index
            + 10_000_019 * attempt_number
        ) % (2**63 - 1)

        rng = np.random.default_rng(local_seed)

        theta9 = sample_first_nine_targets(
            rng,
            params_log,
            1,
        )[0]

        noise_seed = int(
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )

        tasks.append(
            (
                theta9,
                design_index,
                design_tensor[design_index].numpy(),
                design_fields,
                noise_seed,
            )
        )

    return tasks


def run_simulations_with_checkpoints(
    *,
    module_name,
    config,
    params_log,
    design_fields,
    design_tensor,
    target_successful,
    num_workers,
    chunk_size,
    checkpoint_dir,
    seed,
    max_attempts_per_design,
):
    """
    Collect exactly `target_successful` valid simulations using at most one
    retained simulation per approximate design.

    Strategy
    --------
    1. Shuffle all approximate designs once using `seed`.
    2. Work through unused designs.
    3. For each active design, draw the first 9 SBI parameters and simulate.
    4. If it fails, keep the SAME design and redraw only those first 9
       parameters.
    5. Once that design produces one successful simulation, retain it and
       never use that design again.
    6. If a design fails `max_attempts_per_design` times, mark it exhausted and
       move on without a retained sample from that design.
    7. Stop when exactly `target_successful` unique-design simulations have
       been collected.

    `chunk_size` controls how many different active designs are attempted in
    parallel in one batch. A design appears at most once in each batch.

    Checkpointing
    -------------
    Each batch is saved as a chunk. Restarting with the same seed/settings
    reproduces the same design order and per-design retry proposals.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_designs = int(design_tensor.shape[0])

    if target_successful > n_designs:
        raise ValueError(
            f"Requested {target_successful} successful simulations but only "
            f"{n_designs} approximate designs are available. With the "
            "one-success-per-design strategy, target_successful cannot exceed "
            "the number of designs."
        )

    rng = np.random.default_rng(seed)
    design_order = rng.permutation(n_designs).astype(np.int64)

    # State for each design.
    attempts_per_design = np.zeros(n_designs, dtype=np.int32)
    completed = np.zeros(n_designs, dtype=bool)
    exhausted = np.zeros(n_designs, dtype=bool)

    theta_success = []
    x_success = []
    design_index_success = []

    total_attempts = 0
    total_failures = 0
    error_counts = Counter()
    chunk_index = 0

    def available_designs():
        """
        Designs that have not succeeded and have not exhausted their retry cap,
        returned in the fixed shuffled design order.
        """
        mask = ~(completed | exhausted)
        return [
            int(idx)
            for idx in design_order
            if mask[int(idx)]
        ]

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(module_name, config),
    ) as pool:

        while len(theta_success) < target_successful:
            candidates = available_designs()

            if not candidates:
                raise RuntimeError(
                    f"Exhausted all {n_designs} approximate designs after "
                    f"{total_attempts} PyBaMM attempts, but obtained only "
                    f"{len(theta_success)}/{target_successful} successful "
                    "unique-design simulations. Generate more approximate "
                    "designs or increase --max-attempts-per-design."
                )

            remaining_needed = target_successful - len(theta_success)

            # We can attempt more designs than remaining successes because
            # failures are expected, but never exceed the configured batch size.
            this_batch_size = min(
                chunk_size,
                len(candidates),
            )

            # Near the end, avoid unnecessary overshoot: there is no benefit in
            # attempting more designs than the number of successes still needed
            # when every attempted design could succeed.
            this_batch_size = min(
                this_batch_size,
                remaining_needed,
            )

            batch_design_indices = candidates[:this_batch_size]
            batch_attempt_numbers = [
                int(attempts_per_design[idx])
                for idx in batch_design_indices
            ]

            chunk_file = (
                checkpoint_dir
                / f"chunk_{chunk_index:04d}.pt"
            )

            if chunk_file.exists():
                print(
                    f"Chunk {chunk_index + 1}: found on disk, loading",
                    flush=True,
                )

                payload = torch.load(
                    chunk_file,
                    map_location="cpu",
                    weights_only=False,
                )

                if payload.get("checkpoint_version") != 3:
                    raise RuntimeError(
                        f"Incompatible checkpoint: {chunk_file}. "
                        "Remove the checkpoint directory before using the "
                        "one-success-per-design strategy."
                    )

                saved_indices = [
                    int(v)
                    for v in payload["attempted_design_indices"]
                ]
                saved_attempts = [
                    int(v)
                    for v in payload["attempt_numbers"]
                ]

                if (
                    saved_indices != batch_design_indices
                    or saved_attempts != batch_attempt_numbers
                ):
                    raise RuntimeError(
                        f"Checkpoint state mismatch in {chunk_file}. "
                        "Remove the checkpoint directory or rerun with exactly "
                        "the same seed, target, retry cap, and chunk size."
                    )

            else:
                print(
                    f"Chunk {chunk_index + 1}: attempting "
                    f"{this_batch_size} designs "
                    f"({len(theta_success)}/{target_successful} successful)",
                    flush=True,
                )

                tasks = make_design_attempt_tasks(
                    design_indices=batch_design_indices,
                    attempt_numbers=batch_attempt_numbers,
                    global_seed=seed,
                    design_tensor=design_tensor,
                    design_fields=design_fields,
                    params_log=params_log,
                )

                results = pool.map(simulate_one, tasks)

                payload = {
                    "checkpoint_version": 3,
                    "attempted_design_indices": batch_design_indices,
                    "attempt_numbers": batch_attempt_numbers,
                    "theta": torch.stack(
                        [result["theta"] for result in results]
                    ),
                    "x": torch.stack(
                        [result["x"] for result in results]
                    ),
                    "valid": torch.as_tensor(
                        [
                            result["valid"]
                            for result in results
                        ],
                        dtype=torch.bool,
                    ),
                    "errors": [
                        result["error"]
                        for result in results
                    ],
                }

                atomic_torch_save(
                    payload,
                    chunk_file,
                )

            valid = payload["valid"]

            n_valid = int(valid.sum().item())
            n_failed = len(valid) - n_valid

            total_attempts += len(valid)
            total_failures += n_failed

            for row, design_index in enumerate(batch_design_indices):
                attempts_per_design[design_index] += 1

                if bool(valid[row]):
                    completed[design_index] = True

                    theta_success.append(
                        payload["theta"][row]
                    )
                    x_success.append(
                        payload["x"][row]
                    )
                    design_index_success.append(
                        design_index
                    )

                else:
                    error = payload["errors"][row]
                    if error:
                        error_counts[error] += 1

                    if (
                        attempts_per_design[design_index]
                        >= max_attempts_per_design
                    ):
                        exhausted[design_index] = True

            print(
                f"Chunk {chunk_index + 1}: "
                f"{n_valid}/{len(valid)} valid; "
                f"cumulative {len(theta_success)}/{target_successful}; "
                f"{int(exhausted.sum())} designs exhausted",
                flush=True,
            )

            chunk_index += 1

    # We deliberately restrict each batch to at most the number of still-needed
    # successes, so no final overshoot should occur.
    theta = torch.stack(theta_success[:target_successful])
    x = torch.stack(x_success[:target_successful])
    design_indices = torch.as_tensor(
        design_index_success[:target_successful],
        dtype=torch.long,
    )

    used_attempt_counts = attempts_per_design[
        design_indices.numpy()
    ]

    return (
        theta,
        x,
        design_indices,
        total_attempts,
        total_failures,
        attempts_per_design,
        completed,
        exhausted,
        error_counts,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chemistry",
        required=True,
        choices=CHEMISTRY_CONFIG.keys(),
    )

    parser.add_argument(
        "--accuracy",
        choices=("good", "realistic", "rough"),
        default="realistic",
    )

    parser.add_argument(
        "--tag",
        default="",
    )

    parser.add_argument(
        "--design-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--num-successful",
        "--num-simulations",
        dest="num_successful",
        type=int,
        default=50000,
        help=(
            "Target number of successful simulations to save. "
            "--num-simulations is retained as a backwards-compatible alias."
        ),
    )

    parser.add_argument(
        "--max-attempts-per-design",
        type=int,
        default=10,
        help=(
            "Maximum number of independent first-9-parameter draws attempted "
            "for one approximate design before that design is abandoned "
            "(default: 10)."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
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
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args(argv)

    if args.num_successful < 1:
        parser.error("--num-successful must be positive")

    if args.max_attempts_per_design < 1:
        parser.error("--max-attempts-per-design must be positive")

    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be positive")

    return args


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    args = parse_args(argv)

    chemistry = args.chemistry
    config = CHEMISTRY_CONFIG[chemistry]
    module_name = config["params_module"]

    params_module = importlib.import_module(module_name)
    params_log = params_module.params_log
    params_log_flag = params_module.params_log_flag

    parameter_names = list(params_log.keys())

    if len(parameter_names) < NUM_SBI_TARGETS:
        raise ValueError(
            f"{module_name}.params_log contains only "
            f"{len(parameter_names)} parameters"
        )

    # Validate fixed current direction.
    if (
        config["rpt_direction"] == "charge"
        and config["current_A"] >= 0
    ):
        raise ValueError("Charge RPT requires negative current")

    if (
        config["rpt_direction"] == "discharge"
        and config["current_A"] <= 0
    ):
        raise ValueError("Discharge RPT requires positive current")

    # -------------------------------------------------------------------------
    # Approximate-design file
    # -------------------------------------------------------------------------

    if args.design_file is None:
        suffix = f"_{args.tag}" if args.tag else ""

        design_file = (
            ROOT
            / "models"
            / "Approximate_designs"
            / f"designs_{chemistry}_{args.accuracy}{suffix}.pt"
        )
    else:
        design_file = args.design_file

    design_fields, design_tensor, design_metadata = load_design_file(
        design_file
    )

    sampler_chemistry = design_metadata.get("chemistry")
    if (
        sampler_chemistry is not None
        and sampler_chemistry != chemistry
    ):
        raise ValueError(
            f"Design file chemistry is {sampler_chemistry}, "
            f"requested chemistry is {chemistry}"
        )

    sampler_direction = design_metadata.get("rpt_direction")
    if (
        sampler_direction is not None
        and sampler_direction != config["rpt_direction"]
    ):
        raise ValueError(
            f"Design sampler direction is {sampler_direction}, "
            f"simulation direction is {config['rpt_direction']}"
        )

    print(
        f"Loaded {design_tensor.shape[0]} approximate designs "
        f"from {design_file}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    suffix = f"_{args.tag}" if args.tag else ""

    checkpoint_dir = (
        ROOT
        / "models"
        / "Approximate_unique_design_chunks"
        / f"{chemistry}_{args.accuracy}{suffix}_seed{args.seed}"
    )

    output_dir = (
        ROOT
        / "models"
        / "Approximate_simulations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (
        output_dir
        / f"dataset_{chemistry}_{args.accuracy}{suffix}.pt"
    )

    design_index_path = (
        output_dir
        / f"design_indices_{chemistry}_{args.accuracy}{suffix}.pt"
    )

    metadata_path = (
        output_dir
        / f"dataset_{chemistry}_{args.accuracy}{suffix}.json"
    )

    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset already exists: {dataset_path}"
        )

    # -------------------------------------------------------------------------
    # Run simulations
    # -------------------------------------------------------------------------

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else get_num_workers()
    )

    print(f"Chemistry: {chemistry}", flush=True)
    print(
        f"Fixed RPT: direction={config['rpt_direction']}, "
        f"current={config['current_A']} A",
        flush=True,
    )
    print(f"Workers: {num_workers}", flush=True)

    (
        theta,
        x,
        design_indices,
        total_attempts,
        total_invalid,
        attempts_per_design,
        completed_designs,
        exhausted_designs,
        error_counts,
    ) = run_simulations_with_checkpoints(
        module_name=module_name,
        config=config,
        params_log=params_log,
        design_fields=design_fields,
        design_tensor=design_tensor,
        target_successful=args.num_successful,
        num_workers=num_workers,
        chunk_size=args.chunk_size,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        max_attempts_per_design=args.max_attempts_per_design,
    )

    # -------------------------------------------------------------------------
    # Final validation and save
    # -------------------------------------------------------------------------

    if theta.shape[1] != NUM_SBI_TARGETS:
        raise RuntimeError(
            f"Unexpected theta shape: {tuple(theta.shape)}"
        )

    if x.shape[1] != NUM_OBSERVATION_POINTS:
        raise RuntimeError(
            f"Unexpected x shape: {tuple(x.shape)}"
        )

    final_valid = (
        torch.isfinite(theta).all(dim=1)
        & torch.isfinite(x).all(dim=1)
    )

    theta = theta[final_valid]
    x = x[final_valid]
    design_indices = design_indices[final_valid]

    # Preserve the original training-data tuple structure.
    atomic_torch_save(
        (theta, x),
        dataset_path,
    )

    atomic_torch_save(
        design_indices,
        design_index_path,
    )

    metadata = {
        "chemistry": chemistry,
        "params_module": module_name,
        "simulator": "src.simulation.battery_simulator.simulator",
        "design_file": str(design_file),
        "accuracy": args.accuracy,
        "requested_successful_simulations": args.num_successful,
        "valid_simulations_saved": int(theta.shape[0]),
        "unique_designs_used": int(theta.shape[0]),
        "available_designs": int(design_tensor.shape[0]),
        "total_attempts": int(total_attempts),
        "invalid_attempts": int(total_invalid),
        "attempt_success_rate": float(theta.shape[0] / total_attempts),
        "max_attempts_per_design": args.max_attempts_per_design,
        "exhausted_designs": int(exhausted_designs.sum()),
        "designs_with_success": int(completed_designs.sum()),
        "mean_attempts_per_successful_design": float(
            attempts_per_design[design_indices.numpy()].mean()
        ),
        "max_attempts_used_by_successful_design": int(
            attempts_per_design[design_indices.numpy()].max()
        ),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "num_workers": num_workers,
        "rpt_direction": config["rpt_direction"],
        "current_A": config["current_A"],
        "Q_rated_Ah": config["Q_rated_Ah"],
        "dod": config["dod"],
        "V_cut_lb": float(params_module.V_cut_lb),
        "V_cut_ub": float(params_module.V_cut_ub),
        "observation_points": NUM_OBSERVATION_POINTS,
        "voltage_noise_std_V": VOLTAGE_NOISE_STD_V,
        "theta_fields": (
            parameter_names[:NUM_INDEPENDENT_TARGETS]
            + SBI_STOICH_TARGET_FIELDS
        ),
        "nuisance_design_fields": NUISANCE_FIELDS,
        "error_counts": dict(error_counts),
        "notes": [
            (
                "The ordinary battery_simulator is used; "
                "initial_soc is not supplied."
            ),
            (
                "Initial electrode concentrations are set by "
                "theta_n_init and theta_p_init in params_setting()."
            ),
            (
                "Experimental conditions are fixed per chemistry; "
                "only physical/design parameters and 5 mV voltage "
                "noise vary."
            ),
            (
                "Each retained training curve comes from a unique approximate "
                "design. On simulation failure, the same design is retried "
                "with a new draw of only the first 9 SBI parameters."
            ),
        ],
    }

    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved dataset -> {dataset_path}", flush=True)
    print(f"theta shape: {tuple(theta.shape)}", flush=True)
    print(f"x shape: {tuple(x.shape)}", flush=True)
    print(
        f"Saved successful simulations: {theta.shape[0]}/{args.num_successful}",
        flush=True,
    )
    print(
        f"Unique approximate designs represented: {len(torch.unique(design_indices))}",
        flush=True,
    )
    print(
        f"Total PyBaMM attempts: {total_attempts}",
        flush=True,
    )
    print(
        f"Attempt success rate: {theta.shape[0] / total_attempts:.2%}",
        flush=True,
    )
    print(
        f"Designs exhausted after {args.max_attempts_per_design} failures: "
        f"{int(exhausted_designs.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
