import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

from sbi import utils as utils
from sbi import analysis as analysis

from sbi.inference import simulate_for_sbi
from sbi.neural_nets import posterior_nn  
from scipy.interpolate import interp1d
import importlib

from sbi.inference import NPE, simulate_for_sbi

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from sbi.utils import BoxUniform

import pickle
from sklearn.model_selection import KFold, cross_validate
import argparse
sys.path.append(str(Path(__file__).resolve().parents[3]))



from scipy.interpolate import interp1d

from src.simulation.battery_simulator import simulator
import os
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
        "Q_rated": 2.5, "current": -1.25, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.2,
    },
    "HG2": {
        "params_module": "src.simulation.parameters.parameter_HG2",
        "output_subdir": "HG2",
        "Q_rated": 3, "current": -1.5, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.2,
    },
    "M1A": {
        "params_module": "src.simulation.parameters.parameter_M1A",
        "output_subdir": "M1A",
        "Q_rated": 1.1, "current": -0.55, "dod": 1,
        "V_cut_lb": 2, "V_cut_ub": 2.6,
    },
    "MJ1": {
        "params_module": "src.simulation.parameters.parameter_MJ1",
        "output_subdir": "MJ1",
        "Q_rated": 3.5, "current": 3.5, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.2,
    },
    "PA": {
        "params_module": "src.simulation.parameters.parameter_PA",
        "output_subdir": "PA",
        "Q_rated": 74.273, "current": 24.33, "dod": 1,
        "V_cut_lb": 2.75, "V_cut_ub": 4.2,
    },
    "PBC": {
        "params_module": "src.simulation.parameters.parameter_PBC",
        "output_subdir": "PBC",
        "Q_rated": 76, "current": 25.43, "dod": 1,
        "V_cut_lb": 2.75, "V_cut_ub": 4.2,
    },
    "PD": {
        "params_module": "src.simulation.parameters.parameter_PD",
        "output_subdir": "PD",
        "Q_rated": 83.456, "current": 28, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.15,
    },
    "VTC5A": {
        "params_module": "src.simulation.parameters.parameter_VTC5A",
        "output_subdir": "VTC5A",
        "Q_rated": 2.5, "current": 2.5, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.25,
    },
    "VTC6": {
        "params_module": "src.simulation.parameters.parameter_VTC6",
        "output_subdir": "VTC6",
        "Q_rated": 3, "current": 0.6, "dod": 1,
        "V_cut_lb": 2.5, "V_cut_ub": 4.25,
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
checkpoint_dir = Path(__file__).resolve().parents[3] / "models" / "validation_chunks" / config["output_subdir"]
dataset_path = (
    Path(__file__).resolve().parents[3] / "models" / "Validation_simulations" / f"dataset_{args.chemistry}.pt"
)

# Set charging conditions and the voltage cut off
Q_rated = config["Q_rated"]
current = config["current"]
dod = config["dod"]
V_cut_lb = config["V_cut_lb"]
V_cut_ub = config["V_cut_ub"]

num_workers = 8

# Define the number of sample trials, 50000 is used in this work
num_simulations= 10000
num_dim = 11


t_typ = np.abs(3600*Q_rated*dod/current)
t_sim = np.abs(int(3700*Q_rated*dod/current))

def t_normal(data):
    return data/t_typ
def v_normal(data):
    return (data-V_cut_lb)/(V_cut_ub-V_cut_lb)

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
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

def my_model(params):
    params = np.asarray(params)
    this_param = []
    for j in range(num_dim):
        if list(params_log_flag.values())[j]==1:
            this_param.append(10**params[j])
        else:
            this_param.append(params[j])

    params = params_setting(this_param)
    results = simulator(params, current, t_sim, V_cut_lb, V_cut_ub)
    num_points=100
    s = torch.normal(0, 0.005, size =(num_points,))
    if isinstance(results, str):
        ############# abnormal

        return np.full(num_points, np.nan)
    else:
        ############# normal
        Voltage = torch.tensor(results["Terminal voltage [V]"].entries)
        Time = torch.tensor(results["Time [s]"].entries)



        this_t = np.linspace(0, Time[-1], num_points)
        f = interp1d(Time, Voltage, kind='slinear') 

        this_v = torch.tensor(f(this_t)) + s

        #############
    v_norm = v_normal(this_v)
    t_norm = t_normal(this_t)
    # This is a very weird way to encode time. Time steps ae uniform so only the stop time is actually relevant.
    # We also destroy information since 0.25 + 0.25 = 0.5 + 0 = 0 + 0.5, a smarter way to do it would be adding the final time
    # as a additional observation (giving 101 points)
    y_obs = torch.sqrt(t_norm**2+v_norm**2)

    return y_obs


## This fuction was generated using claude:
def run_simulations_with_checkpoints(
    simulator_fn, prior, num_simulations, num_workers,
    chunk_size=2000, checkpoint_dir="sim_chunks"
):
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (num_simulations + chunk_size - 1) // chunk_size
    theta_parts, x_parts = [], []

    for i in range(n_chunks):
        this_chunk_size = min(chunk_size, num_simulations - i * chunk_size)
        chunk_file = out_dir / f"chunk_{i:04d}.pt"

        if chunk_file.exists():
            print(f"Chunk {i+1}/{n_chunks}: found on disk, loading")
            theta_c, x_c = torch.load(chunk_file)
        else:
            print(f"Chunk {i+1}/{n_chunks}: running {this_chunk_size} simulations")
            theta_c, x_c = simulate_for_sbi(
                simulator_fn, proposal=prior,
                num_simulations=this_chunk_size, num_workers=num_workers,
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
    simulator_fn, prior, num_simulations=num_simulations,
    num_workers=num_workers, chunk_size= 1000,
    checkpoint_dir=checkpoint_dir,
)


dataset_path.parent.mkdir(parents=True, exist_ok=True)
torch.save((theta, x), dataset_path)
print(f"Saved dataset to {dataset_path}")