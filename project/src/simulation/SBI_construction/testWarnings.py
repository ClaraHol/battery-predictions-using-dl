import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.utils.user_input_checks import (
    process_prior,
)

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.simulation.battery_simulator import simulator
from src.simulation.parameters.parameter_25R import (
    params_log,
    params_log_flag,
    params_setting,
)

# This was generated using Claude to test determine the rate of occurance for each error type.

# Define some basic parameters for this public from Tongjij
Q_rated = 2.5  # Ah
current = -1.25  # A
dod = 1  # DOD range
V_cut_lb = 2.5
V_cut_ub = 4.2
num_workers = 8

# Define the number of sample trials, 50000 is used in this work
num_simulations = 1000
num_dim = 11

# I am uncertain about the exact logic of why t_typ and t_sim are different. It is t_typ which is used for normilization
# but I don't understand why we don't normalize with the total time scale
t_typ = np.abs(3600 * Q_rated * dod / current)
t_sim = np.abs(int(3700 * Q_rated * dod / current))


def t_normal(data):
    return data / t_typ


def v_normal(data):
    return (data - V_cut_lb) / (V_cut_ub - V_cut_lb)


prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
prior, num_parameters, prior_returns_numpy = process_prior(prior)


def run_diagnostic(n_samples=500):
    counter = Counter()
    examples = defaultdict(list)

    for _ in range(n_samples):
        raw = prior.sample((1,)).numpy().flatten()
        this_param = [
            10 ** raw[j] if list(params_log_flag.values())[j] == 1 else raw[j]
            for j in range(num_dim)
        ]
        params = params_setting(this_param)
        result = simulator(params, current, t_sim, V_cut_lb, V_cut_ub)

        if isinstance(result, str):
            m = re.search(r"\[(.*?)\]", result)
            category = m.group(1) if m else "unknown"
        else:
            category = "success"

        counter[category] += 1
        if len(examples[category]) < 5:
            examples[category].append(
                {
                    "theta_n_init": this_param[9],
                    "theta_p_init": this_param[10],
                    "k_n": this_param[2],
                    "k_p": this_param[3],
                }
            )

    return counter, examples


if __name__ == "__main__":
    tally, examples = run_diagnostic(500)
    total = sum(tally.values())
    for category, count in tally.most_common():
        print(f"\n{category:25s} {count:5d}  ({100 * count / total:.1f}%)")
        for ex in examples[category]:
            print("   ", ex)
