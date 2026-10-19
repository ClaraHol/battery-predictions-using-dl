
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.utils.user_input_checks import process_prior

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = Path("/work3/claho/battery_datasets/processed")
MODEL_DIR = PROJECT_ROOT / "models" / "SBI_models"
N_SAMPLES = 5000

# Chemistry: (nominal capacity, reference current, lower voltage, upper voltage)
# Reference conditions from the original simulation configuration.
CHEMISTRY_CONFIG = {
    "25R":   (2.5, -1.25, 2.5, 4.2),
    "HG2":   (3.0, -1.5,  2.5, 4.2),
    "M1A":   (1.1, -0.55, 2.0, 2.6),
    "MJ1":   (3.5,  3.5,  2.5, 4.2),
    "PA":    (74.273, 24.33, 2.75, 4.2),
    "PBC":   (76.0, 25.43, 2.75, 4.2),
    "PD":    (83.456, 28.0, 2.5, 4.15),
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6":  (3.0, 0.6, 2.5, 4.25),
}

SOURCE_TO_CHEMISTRY = {
    "snl_lfp_": "M1A",
    "snl_nmc_": "HG2",
    "tongji_": "25R",
    "tum_": "VTC5A",
    "cmu_": "VTC6",
}


def get_chemistry(cell_id):
    for prefix, chemistry in SOURCE_TO_CHEMISTRY.items():
        if cell_id.startswith(prefix):
            return chemistry
    return None


# ============================================================
# 1. Load all chemistry-specific SBI models
# ============================================================

def load_all_SBI():
    models = {}

    for chemistry, (capacity, current, v_min, v_max) in CHEMISTRY_CONFIG.items():
        path = MODEL_DIR / chemistry / f"inference_{chemistry}.pkl"

        if not path.exists():
            print(f"Missing SBI: {chemistry} ({path})")
            continue

        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )

        params_log = module.params_log

        prior_min = torch.tensor(
            [bounds[0] for bounds in params_log.values()],
            dtype=torch.float32,
        )
        prior_max = torch.tensor(
            [bounds[1] for bounds in params_log.values()],
            dtype=torch.float32,
        )

        prior = utils.BoxUniform(low=prior_min, high=prior_max)
        prior, _, _ = process_prior(prior)

        with open(path, "rb") as handle:
            inference = pickle.load(handle)

        posterior = inference.build_posterior(prior=prior)

        models[chemistry] = {
            "posterior": posterior,
            "param_names": list(params_log.keys()),
            "log_flags": list(module.params_log_flag.values()),
            "prior_min": prior_min,
            "prior_range": prior_max - prior_min,
            "v_min": v_min,
            "v_max": v_max,
            "t_typ": abs(3600 * capacity / current),
        }

        print(f"Loaded SBI: {chemistry}")

    return models


# ============================================================
# 2. Inference on experimental voltage/time curves
# ============================================================

def SBI_inference(model, voltage, time):
    voltage = np.asarray(voltage, dtype=float)
    time = np.asarray(time, dtype=float)

    # Remove invalid data, sort and remove duplicate timestamps.
    valid = np.isfinite(voltage) & np.isfinite(time)
    time, voltage = time[valid], voltage[valid]

    order = np.argsort(time)
    time, voltage = time[order], voltage[order]

    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]

    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("Curve has insufficient valid time points")

    # Reset start time and interpolate to 100 points.
    time = time - time[0]
    t_interp = np.linspace(0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)

    # Match the original simulation's observation encoding.
    t_norm = t_interp / model["t_typ"]
    v_norm = (
        (v_interp - model["v_min"])
        / (model["v_max"] - model["v_min"])
    )

    x = np.sqrt(t_norm**2 + v_norm**2)
    x = torch.tensor(x, dtype=torch.float32)

    # Sample the posterior conditioned on the observation.
    with torch.no_grad():
        samples = model["posterior"].sample(
            (50000,),
            x=x,
            show_progress_bars=False,
            reject_outside_prior=False,
        ).cpu()

    low = model["prior_min"]
    high = low + model["prior_range"]

    inside = ((samples >= low) & (samples <= high)).all(dim=1)

    print(f"Fraction inside prior: {inside.float().mean():.4%}")

    for name, fraction in zip(
        model["param_names"],
        ((samples >= low) & (samples <= high)).float().mean(dim=0),
    ):
        print(f"{name}: {fraction:.2%}")
    # Posterior mean in the original parameter space
    theta_mean = samples.mean(dim=0)

    # Normalize estimates to each chemistry's prior range.
    theta_normalized = (
        (theta_mean - model["prior_min"]) / model["prior_range"]
    )

    # Calculate posterior means in physical units.
    physical_means = []
    for j, log_flag in enumerate(model["log_flags"]):
        values = 10.0 ** samples[:, j] if log_flag else samples[:, j]
        physical_means.append(values.mean().item())

    return theta_normalized.numpy(), np.asarray(physical_means)


# ============================================================
# 3. Load experimental data and evaluate each cell
# ============================================================

def run_inference_efc0(df, models, curve_type):
    results = []
    unique_cell_ids = df["cell_id"].drop_duplicates().tolist()

    for cell_id in unique_cell_ids:
        chemistry = get_chemistry(str(cell_id))

        cell_data = df[df["cell_id"] == cell_id]
        voltage_curve = cell_data["early_curve"].to_numpy()
        time_curve = cell_data["time_s"].to_numpy()
        if chemistry not in models:
            print(f"Skipping {cell_id}: no matching SBI")
            continue

        try:
            normalized, physical = SBI_inference(
                models[chemistry],
                voltage_curve,
                time_curve,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Skipping {cell_id}: {exc}")
            continue

        row = {
            "cell_id": cell_id,
            "chemistry": chemistry,
            "curve_type": curve_type,
        }

        names = models[chemistry]["param_names"]

        for name, norm_value, physical_value in zip(
            names, normalized, physical
        ):
            row[f"{name}_normalized"] = norm_value
            row[f"{name}_physical"] = physical_value

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    torch.manual_seed(0)

    models = load_all_SBI()

    efc0 = pd.read_csv(
        DATA_DIR / "public_charging_curve_early_all.csv"
    )
    efc50 = pd.read_csv(
        DATA_DIR / "public_charging_curve_late_all.csv"
    )
    print("Attempting inference")
    early_results = run_inference_efc0(efc0, models, "efc0")
    late_results = run_inference(efc50, models, "efc50")

    results = pd.concat(
        [early_results, late_results], ignore_index=True
    )

    output = MODEL_DIR / "experimental_inference.csv"
    results.to_csv(output, index=False)

    print(f"\nSaved results: {output}")
    print(results.groupby(["chemistry", "curve_type"]).size())
