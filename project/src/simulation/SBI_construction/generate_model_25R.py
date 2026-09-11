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


from sbi.inference import NPE, simulate_for_sbi

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from sbi.utils import BoxUniform

import pickle
from sklearn.model_selection import KFold, cross_validate

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.simulation.parameters.parameter_25R import params_log, params_log_flag, params_setting


from scipy.interpolate import interp1d

from src.simulation.battery_simulator import simulator


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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(torch.cuda.is_available())


# Define some basic parameters for this public from Tongjij
Q_rated = 2.5 # Ah
current = -1.25 # A
dod = 1  # DOD range
V_cut_lb = 2.5
V_cut_ub = 4.2
num_workers = 8

# Define the number of sample trials, 50000 is used in this work
num_simulations=50000
num_dim = 11

# I am uncertain about the exact logic of why t_typ and t_sim are different. It is t_typ which is used for normilization
# but I don't understand why we don't normalize with the total time scale
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

        this_t = torch.linspace(0, t_sim, num_points)
        
        this_v = torch.ones(num_points)*V_cut_lb + s
        #############
    else:
        ############# normal
        Voltage = torch.tensor(results["Terminal voltage [V]"].entries)
        Time = torch.tensor(results["Time [s]"].entries)

        # import matplotlib.pyplot as plt
        # plt.plot(Time, Voltage)
        # plt.show()
       

        this_t = np.linspace(0, Time[-1], num_points)
        f = interp1d(Time, Voltage, kind='slinear') 

        this_v = torch.tensor(f(this_t)) + s

        #############
    v_norm = v_normal(this_v)
    t_norm = t_normal(this_t)
    # This is a very weird way to encode time. Time steps ae uniform so only the stop time is actually relevant.
    # We also destroy information since 0.25 + 0.25 = 0.5 + 0 = 0 + 0.5, a smarter way to do it would be adding the final time
    # as a additional observation (giving 101 points)
    y_obs = np.sqrt(np.asarray(t_norm)**2 + np.asarray(v_norm)**2)

    return y_obs


# 1. Implement the data generate
# Data generated are used for the model training


prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator_fn = process_simulator(my_model, prior, prior_returns_numpy)
check_sbi_inputs(simulator_fn, prior)

num_simulations = 10
num_workers = 4
theta, x = simulate_for_sbi(
    simulator_fn, proposal=prior, num_simulations=num_simulations, num_workers=num_workers
)

### 2.Load the dataset
theta_tra = theta.clone().detach()
x_tra = x.clone().detach()

### 3.Set the hyper-parameters
### max epoch number set to be 200 in this work, to aviod overfitting
### spline flow
flow_step=9
bin=9
tail_bound = 3.0
### resnet
res_block = 9
hidden_feature = int(2**8)
dropout = 0.05
use_batch_norm = True
### training-related
max_num_epochs = 200
batch_size = 256
lr = 0.0005
clip = 21.0
#############################

### 4.Create the model
nde_nsf = posterior_nn(model="nsf" ,z_score_x='structured')

inference = NPE(show_progress_bars=True,prior=prior,density_estimator=nde_nsf, device=device)
inference = inference.append_simulations(theta_tra, x_tra, proposal=prior, data_device=device)




if __name__ == '__main__':
    print("Starting")
    import time
    start = time.time()
    ### Train the model, validation deactivated
    inference.train(show_train_summary=True, max_num_epochs=max_num_epochs,  
                    learning_rate =lr, training_batch_size=batch_size ,validation_fraction=0.4, clip_max_norm=clip,
                    )
    end = time.time()
    print("time used:",end-start)
    
    ### Save the trained model
    output_dir = Path(__file__).resolve().parents[3] / "models" / "SBI_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "inference_25R.pkl", "wb") as handle:
        pickle.dump(inference, handle)