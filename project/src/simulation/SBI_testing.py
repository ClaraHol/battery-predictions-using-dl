import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sbi import utils as utils

from sbi.inference import simulate_for_sbi
from sbi.neural_nets import posterior_nn  
from scipy.interpolate import interp1d

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from battery_simulator import simulator
from parameters.parameter_25R import params_log, params_log_flag, params_setting
from sbi.analysis import plot_summary
import pickle
import warnings
import pybamm

# Suppress standard Python warnings (like DeprecationWarnings)
warnings.filterwarnings("ignore")

# Set the PyBaMM logger level to ERROR to hide warnings and below
pybamm.set_logging_level("ERROR")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =======================================================
# Sample from the prior and simulate a single battery
Q_rated = 2.5 # Ah
current = -1.25 # A
dod = 1  # DOD range
V_cut_lb = 2.5
V_cut_ub = 4.2


num_dim = 11


t_typ = np.abs(3600*Q_rated*dod/current)
t_sim = np.abs(int(3700*Q_rated*dod/current))

def t_normal(data):
    return data/t_typ
def v_normal(data):
    return (data-V_cut_lb)/(V_cut_ub-V_cut_lb)

prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))


def generate_voltage_curve(prior):  
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    samples = prior.sample((1,))
    params = np.asarray(samples)[0,:]

    this_param = []
    for j in range(num_dim):
        if list(params_log_flag.values())[j]==1:
            this_param.append(10**params[j])
        else:
            this_param.append(params[j])

    params = params_setting(this_param)
    results = simulator(params, current, t_sim, V_cut_lb, V_cut_ub)
    if isinstance(results, str):
        return None, None, 1
    num_points=100
    s = torch.normal(0, 0.005, size =(num_points,))

    Voltage = torch.tensor(results["Terminal voltage [V]"].entries)
    Time = torch.tensor(results["Time [s]"].entries)

    this_t = np.linspace(0, Time[-1], num_points)
    f = interp1d(Time, Voltage, kind='slinear') 

    this_v = torch.tensor(f(this_t)) + s


    v_norm = v_normal(this_v)
    t_norm = t_normal(this_t)

    y_obs = torch.sqrt(t_norm**2+v_norm**2)

    
    return samples, y_obs, 0

def run_multiple_checks(prior, n_checks=30, max_attempts=400):
    all_errors = []
    n_extreme_skipped = 0
    attempts = 0

    while len(all_errors) < n_checks and attempts < max_attempts:
        attempts += 1
        true_features, y_obs, was_extreme = generate_voltage_curve(prior)  # have it return a flag
        if was_extreme:
            n_extreme_skipped += 1
            continue

        posterior_samples = posterior.sample((50000,), x=y_obs, show_progress_bars=False)
        posterior_means = posterior_samples.mean(dim=0)
        normalized_error = ((posterior_means - true_features.squeeze(0)) / prior_range)**2
        all_errors.append(normalized_error)

    print(f"Shape: {posterior_samples.T.shape}")
    correlation = torch.corrcoef(posterior_samples.T)
    print(f"Skipped {n_extreme_skipped} extreme-case draws out of {attempts} attempts")
    return torch.stack(all_errors), correlation

# =============================================
## Load the model

if __name__ == '__main__':
    model_dir = Path(__file__).resolve().parents[2] / "models" / "SBI_models"

    with open(model_dir / "inference_25R_old_encoding.pkl", "rb") as handle:
        inference = pickle.load(handle)


    posterior = inference.build_posterior(prior = prior)

    prior_min_t = torch.tensor(prior_min)
    prior_max_t = torch.tensor(prior_max)
    prior_range = prior_max_t - prior_min_t

    all_errors, correlation = run_multiple_checks(prior, n_checks=300)
    mean_errors = all_errors.mean(dim=0)


    param_names = list(params_log.keys())
    for name, err in zip(param_names, mean_errors):
        print(f"{name:40s} normalized MSE: {err.item():.4f}")

    print(f"\nOverall normalized MSE: {mean_errors.mean().item():.4f}")
    print("\nFull correlation array:")


    # print(correlation.tolist())



    