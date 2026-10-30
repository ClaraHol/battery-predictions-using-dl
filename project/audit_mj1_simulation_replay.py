#!/usr/bin/env python3
"""Replay stored MJ1 training simulations through the current simulator.

Purpose:
    Verify that the current MJ1 parameter module + battery simulator reproduce
    the observations stored in models/Full_simulations/dataset_MJ1.pt.

This is a critical consistency test before interpreting experimental forward
closure. The MLP was trained on the stored dataset. If the current simulator
cannot reproduce that dataset from its stored theta values, then a poor
experimental forward-closure result may simply reflect simulator drift.

Run from anywhere inside the repository. Edit the constants below if needed.
"""

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Editable settings
# -----------------------------------------------------------------------------
N_CASES = 100
RANDOM_SEED = 1234
DATASET_PATH = None  # None -> models/Full_simulations/dataset_MJ1.pt
SAVE_CASES = True

Q_RATED = 3.5
CURRENT = 3.5
DOD = 1.0
V_CUT_LB = 2.5
V_CUT_UB = 4.2
T_SIM = abs(int(3700 * Q_RATED * DOD / CURRENT))
T_TYP = abs(3600 * Q_RATED * DOD / CURRENT)
N_POINTS = 100

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


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError("Could not find project root containing src/ and models/.")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.battery_simulator import simulator  # noqa: E402
from src.simulation.parameters import parameter_MJ1  # noqa: E402


def theta_to_physical(theta: np.ndarray) -> list[float]:
    flags = list(parameter_MJ1.params_log_flag.values())
    if len(flags) != len(theta):
        raise RuntimeError(
            f"Parameter flag length {len(flags)} != theta length {len(theta)}"
        )
    return [10.0 ** float(v) if int(flags[i]) == 1 else float(v)
            for i, v in enumerate(theta)]


def make_observation(time_s: np.ndarray, voltage_v: np.ndarray) -> np.ndarray:
    order = np.argsort(time_s)
    time_s = np.asarray(time_s)[order]
    voltage_v = np.asarray(voltage_v)[order]
    time_s = time_s - time_s[0]

    t_new = np.linspace(0.0, time_s[-1], N_POINTS)
    v_new = interp1d(time_s, voltage_v, kind="slinear")(t_new)

    t_norm = t_new / T_TYP
    v_norm = (v_new - V_CUT_LB) / (V_CUT_UB - V_CUT_LB)
    return np.sqrt(t_norm**2 + v_norm**2)


def simulate_observation(theta: np.ndarray):
    physical = theta_to_physical(theta)
    params = parameter_MJ1.params_setting(physical)
    result = simulator(params, CURRENT, T_SIM)
    if isinstance(result, str):
        raise RuntimeError(result)

    time_s = np.asarray(result["Time [s]"].entries, dtype=float)
    voltage_v = np.asarray(result["Terminal voltage [V]"].entries, dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(voltage_v)
    time_s = time_s[mask]
    voltage_v = voltage_v[mask]

    x = make_observation(time_s, voltage_v)
    return x, time_s, voltage_v


def main():
    dataset_path = (
        Path(DATASET_PATH).expanduser().resolve()
        if DATASET_PATH is not None
        else PROJECT_ROOT / "models" / "Full_simulations" / "dataset_MJ1.pt"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    theta, x_stored = torch.load(dataset_path, map_location="cpu")
    theta = torch.as_tensor(theta).detach().cpu().numpy()
    x_stored = torch.as_tensor(x_stored).detach().cpu().numpy()

    if theta.ndim != 2 or theta.shape[1] != 11:
        raise ValueError(f"Unexpected theta shape: {theta.shape}")
    if x_stored.ndim != 2 or x_stored.shape[1] != 100:
        raise ValueError(f"Unexpected x shape: {x_stored.shape}")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(theta), size=min(N_CASES, len(theta)), replace=False)

    rows = []
    failures = []
    for n, idx in enumerate(indices, 1):
        print(f"{n}/{len(indices)}", end="\r", flush=True)
        try:
            x_replayed, time_s, voltage_v = simulate_observation(theta[idx])
            delta = x_replayed - x_stored[idx]

            rows.append({
                "dataset_index": int(idx),
                "observation_rmse": float(np.sqrt(np.mean(delta**2))),
                "observation_mae": float(np.mean(np.abs(delta))),
                "observation_max_abs": float(np.max(np.abs(delta))),
                "sim_duration_s": float(time_s[-1] - time_s[0]),
                "stored_x_min": float(np.min(x_stored[idx])),
                "stored_x_max": float(np.max(x_stored[idx])),
                "replayed_x_min": float(np.min(x_replayed)),
                "replayed_x_max": float(np.max(x_replayed)),
            })
        except Exception as exc:
            failures.append({
                "dataset_index": int(idx),
                "error": repr(exc),
            })

    results = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)

    print("\n")
    print("=== MJ1 simulation replay audit ===")
    print(f"Dataset: {dataset_path}")
    print(f"Stored cases: {len(theta)}")
    print(f"Tested: {len(indices)}")
    print(f"Successful: {len(results)}")
    print(f"Failures: {len(failures_df)}")

    if len(results):
        print("\nReplay vs stored observation x (stored x includes 5 mV simulation noise):")
        for col in ["observation_rmse", "observation_mae", "observation_max_abs"]:
            print(f"  {col:24s} median={results[col].median():.6g} "
                  f"mean={results[col].mean():.6g} max={results[col].max():.6g}")

    out_dir = PROJECT_ROOT / "models" / "mj1_simulation_replay_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "replay_cases.csv", index=False)
    failures_df.to_csv(out_dir / "replay_failures.csv", index=False)

    # Record the exact prior contract used by the current parameter module.
    prior_rows = []
    for name, bounds, flag in zip(
        parameter_MJ1.params_log.keys(),
        parameter_MJ1.params_log.values(),
        parameter_MJ1.params_log_flag.values(),
    ):
        prior_rows.append({
            "parameter": name,
            "lower": bounds[0],
            "upper": bounds[1],
            "log_flag": flag,
        })
    pd.DataFrame(prior_rows).to_csv(out_dir / "current_mj1_prior_contract.csv", index=False)

    if len(results):
        print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
