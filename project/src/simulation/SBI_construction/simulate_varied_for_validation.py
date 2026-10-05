import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parents[3]))


import os

from src.simulation.battery_simulator_varied import simulator_v2

# ==============================================================================
# CODE ATTRIBUTION & ADAPTATION NOTICE
# ------------------------------------------------------------------------------
# Based on:          Code Ocean Capsule 9957480 (v1)
# Original Paper:    Discovery Learning predicts battery cycle life from minimal experiments
# Source URL:        https://codeocean.com/capsule/9957480/tree/v1
# License:           Refer to capsule source license (typically MIT or CC-BY-4.0)
# Date Accessed:     September 9, 2026
#
# Modifications Made:
#   - Changed solver to IDAKLU
#   - Some general updates to match the new version of SBI, such as changing to torch tensors
# ==============================================================================


print("Python executable:", sys.executable)
print("Python version:", sys.version)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(torch.cuda.is_available())

CHEMISTRY_CONFIG = {
    "25R": {
        "params_module": "src.simulation.parameters.parameter_25R",
        "output_subdir": "25R",
        "Q_rated": 2.5,
        "current": -1.25,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.2,
    },
    "HG2": {
        "params_module": "src.simulation.parameters.parameter_HG2",
        "output_subdir": "HG2",
        "Q_rated": 3,
        "current": -1.5,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.2,
    },
    "M1A": {
        "params_module": "src.simulation.parameters.parameter_M1A",
        "output_subdir": "M1A",
        "Q_rated": 1.1,
        "current": -0.55,
        "dod": 1,
        "V_cut_lb": 2,
        "V_cut_ub": 2.6,
    },
    "MJ1": {
        "params_module": "src.simulation.parameters.parameter_MJ1",
        "output_subdir": "MJ1",
        "Q_rated": 3.5,
        "current": 3.5,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.2,
    },
    "PA": {
        "params_module": "src.simulation.parameters.parameter_PA",
        "output_subdir": "PA",
        "Q_rated": 74.273,
        "current": 24.33,
        "dod": 1,
        "V_cut_lb": 2.75,
        "V_cut_ub": 4.2,
    },
    "PBC": {
        "params_module": "src.simulation.parameters.parameter_PBC",
        "output_subdir": "PBC",
        "Q_rated": 76,
        "current": 25.43,
        "dod": 1,
        "V_cut_lb": 2.75,
        "V_cut_ub": 4.2,
    },
    "PD": {
        "params_module": "src.simulation.parameters.parameter_PD",
        "output_subdir": "PD",
        "Q_rated": 83.456,
        "current": 28,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.15,
    },
    "VTC5A": {
        "params_module": "src.simulation.parameters.parameter_VTC5A",
        "output_subdir": "VTC5A",
        "Q_rated": 2.5,
        "current": 2.5,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.25,
    },
    "VTC6": {
        "params_module": "src.simulation.parameters.parameter_VTC6",
        "output_subdir": "VTC6",
        "Q_rated": 3,
        "current": 0.6,
        "dod": 1,
        "V_cut_lb": 2.5,
        "V_cut_ub": 4.25,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--chemistry", required=True, choices=CHEMISTRY_CONFIG.keys())
args = parser.parse_args()


config = CHEMISTRY_CONFIG[args.chemistry]
print("Chemestry used", args.chemistry)
# Load the specefic parameters for the chemistry
params_module = importlib.import_module(config["params_module"])
params_log = params_module.params_log
params_log_flag = params_module.params_log_flag
params_setting = params_module.params_setting

# Set the save directory for simulations
checkpoint_dir = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "Varied_validation_sim_chunks"
    / config["output_subdir"]
)
dataset_path = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "Varied_validation_simulations"
    / f"dataset_{args.chemistry}.pt"
)

# Set charging conditions and the voltage cut off
Q_rated = config["Q_rated"]

V_cut_lb = config["V_cut_lb"]
V_cut_ub = config["V_cut_ub"]

num_workers = 1

# Define the number of sample trials, 50000 is used in this work
num_simulations = 10000
num_dim = 9


## The low capacity range
l_low = 0.1
l_high = 0.4

## The high capacity range
h_low = 0.6
h_high = 0.9

## Charge C-rate range (negative PyBaMM current)
crp_low = 0.25
crp_high = 1
## Discharge C-rate range (positive PyBaMM current)(note I balance things such that half the dicharges are between low and mid and half between mid and high)
crn_low = 0.25
crn_mid = 1
crn_high = 3


def v_normal(data):
    return (data - V_cut_lb) / (V_cut_ub - V_cut_lb)


def get_num_workers(default=4):
    # LSF sets this to the number of processors allocated via -n
    n = os.environ.get("LSB_DJOB_NUMPROC")
    if n is not None:
        return int(n)
    # Fallback for local/non-HPC runs
    return min(default, os.cpu_count() or default)


prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)


def my_model(params):
    params = np.asarray(params)
    this_param = []
    for j in range(num_dim):
        if list(params_log_flag.values())[j] == 1:
            this_param.append(10 ** params[j])
        else:
            this_param.append(params[j])

    # Parameters needed by params_setting(), but NOT SBI targets
    for j in range(9, 11):
        low, high = list(params_log.values())[j]

        value = np.random.uniform(low, high)

        if list(params_log_flag.values())[j] == 1:
            value = 10 ** value

        this_param.append(value)

    params = params_setting(this_param)

    low_charge_bound = np.random.uniform(low = l_low, high = l_high)
    high_charge_bound = np.random.uniform(low = h_low, high = h_high)

    sign = np.random.choice([-1,1])

    if sign > 0:
        random_decider = np.random.choice([0,1])
        soc_start = high_charge_bound
        soc_end = low_charge_bound
        if random_decider:
            charge_rate = np.random.uniform(crn_low, crn_mid)
        else:
            charge_rate = np.random.uniform(crn_mid,crn_high)
    else:
        soc_start = low_charge_bound
        soc_end = high_charge_bound 
        charge_rate = np.random.uniform(crp_low, crp_high)
    

    soc_window = abs(soc_end - soc_start)

    current = sign * charge_rate * Q_rated
    t_sim = 3600.0 * soc_window / charge_rate



    results = simulator_v2(params, current, t_sim, soc_start)

    num_points = 100
    s = torch.normal(0, 0.005, size=(num_points,))
    if isinstance(results, str):
        ############# abnormal

        return np.full(num_points+8, np.nan)

    Voltage = torch.tensor(results["Terminal voltage [V]"].entries, dtype = torch.float32)

    Time = torch.tensor(results["Time [s]"].entries, dtype = torch.float32)

    time_np = Time.numpy()
    voltage_np = Voltage.numpy()

    if (
        len(time_np) < 2
        or len(voltage_np) < 2
        or not np.isfinite(time_np).all()
        or not np.isfinite(voltage_np).all()
    ):
        return np.full(num_points + 8, np.nan)

    # Remove duplicate timestamps, keeping the first occurrence
    _, unique_idx = np.unique(time_np, return_index=True)
    unique_idx = np.sort(unique_idx)

    time_np = time_np[unique_idx]



    if len(time_np) < 2:
        return np.full(num_points + 8, np.nan)

    voltage_np = voltage_np[unique_idx]
    t_end = time_np[-1]

    this_t = np.linspace(0.0, time_np[-1], num_points)

    f = interp1d(time_np, voltage_np, kind="slinear")
    this_v = torch.tensor(f(this_t), dtype = torch.float32)

    # Add noise
    this_v += s
    # Normalize using voltage bounds
    v_norm = v_normal(this_v)
 

    conditions = torch.tensor([
    charge_rate,
    sign,
    soc_start,
    soc_end,
    Q_rated,
    V_cut_lb,
    V_cut_ub,
    t_end / 3600.0,
    ], dtype = torch.float32)

    y_obs = torch.cat([v_norm, conditions])

    return y_obs


## This fuction was generated using claude:
def run_simulations_with_checkpoints(
    simulator_fn,
    prior,
    num_simulations,
    num_workers,
    chunk_size=2000,
    checkpoint_dir="sim_chunks",
):
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (num_simulations + chunk_size - 1) // chunk_size
    theta_parts, x_parts = [], []

    for i in range(n_chunks):
        this_chunk_size = min(chunk_size, num_simulations - i * chunk_size)
        chunk_file = out_dir / f"chunk_{i:04d}.pt"

        if chunk_file.exists():
            print(f"Chunk {i + 1}/{n_chunks}: found on disk, loading")
            theta_c, x_c = torch.load(chunk_file)
        else:
            print(f"Chunk {i + 1}/{n_chunks}: running {this_chunk_size} simulations")
            theta_c, x_c = simulate_for_sbi(
                simulator_fn,
                proposal=prior,
                num_simulations=this_chunk_size,
                num_workers=num_workers,
            )
            # write to a temp name and rename, so a crash mid-write can't leave a corrupt chunk file
            tmp_file = chunk_file.with_suffix(".tmp")
            torch.save((theta_c, x_c), tmp_file)
            tmp_file.rename(chunk_file)

        theta_parts.append(theta_c)
        x_parts.append(x_c)

    theta = torch.cat(theta_parts, dim=0)
    x = torch.cat(x_parts, dim=0)
    return theta, x


prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator_fn = process_simulator(my_model, prior, prior_returns_numpy)
check_sbi_inputs(simulator_fn, prior)


num_workers = get_num_workers()
print(f"Using {num_workers} workers")
theta, x = run_simulations_with_checkpoints(
    simulator_fn,
    prior,
    num_simulations=num_simulations,
    num_workers=num_workers,
    chunk_size=1000,
    checkpoint_dir=checkpoint_dir,
)


valid = torch.isfinite(theta).all(dim=1) & torch.isfinite(x).all(dim=1)

theta = theta[valid]
x = x[valid]


dataset_path.parent.mkdir(parents=True, exist_ok=True)
torch.save((theta, x), dataset_path)
print(f"Saved dataset to {dataset_path}")
