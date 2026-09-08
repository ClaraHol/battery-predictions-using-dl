import numpy as np
import pybamm
import torch
from sbi import utils as utils
from sbi.inference import simulate_for_sbi
from sbi.neural_nets import posterior_nn  # new location
from scipy.interpolate import interp1d

from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

Q_rated = 3      # Ah, matches Chen2020 nominal cell capacity
current = -1.5   # A
V_cut_lb = 2.5
V_cut_ub = 4.2
t_typ = np.abs(3600 * Q_rated / current)
t_sim = np.abs(int(3700 * Q_rated / current))

def t_normal(data): return data / t_typ
def v_normal(data): return (data - V_cut_lb) / (V_cut_ub - V_cut_lb)

# Pick 2 well-known, physically meaningful parameters to vary for the test
param_names = [
    "Negative electrode diffusivity [m2.s-1]",
    "Positive electrode diffusivity [m2.s-1]",
]
num_dim = len(param_names)

# log10-uniform priors spanning a couple orders of magnitude around Chen2020 defaults
prior_min = torch.tensor([-15.0, -15.0])
prior_max = torch.tensor([-12.0, -12.0])
prior = utils.BoxUniform(low=prior_min, high=prior_max)

base_params = pybamm.ParameterValues("Chen2020")

def simulator(theta):
    
    parameter_values = base_params.copy()
    for name, val in zip(param_names, theta):
        parameter_values[name] = 10 ** val   # undo log10

    model = pybamm.lithium_ion.SPMe()  # fast model for the test

    model.events.append(
        pybamm.Event(
            "Minimum voltage limit",
            model.variables["Terminal voltage [V]"] - V_cut_lb,
            pybamm.EventType.TERMINATION,
        )
    )
    experiment = None
    try:
        print("simulating")
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, t_sim])
    except Exception:
        print("entered exception")
        num_points = 100
        this_t = torch.linspace(0, t_sim, num_points)
        this_v = torch.ones(num_points) * V_cut_lb + torch.normal(0, 0.005, size=(num_points,))
        v_norm = v_normal(this_v)
        t_norm = t_normal(this_t)
        return torch.concatenate([t_norm, v_norm])

    Voltage = torch.Tensor(np.array(sol["Terminal voltage [V]"].entries))
    Time = torch.Tensor(np.array(sol["Time [s]"].entries))
    num_points = 100
    this_t = np.linspace(0, Time[-1], num_points)
    f = interp1d(Time, Voltage, kind="slinear")
    this_v = torch.Tensor(f(this_t)) + torch.normal(0, 0.005, size=(num_points,))

    v_norm = v_normal(this_v)
    t_norm = t_normal(this_t)

    last_time = torch.tensor([Time[-1].item()], dtype=torch.float32)
    return torch.concatenate([last_time, v_norm])  # length-200, avoids the norm-collapse issue too

def my_model(params):
    return simulator(params)

prior = BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator_fn = process_simulator(my_model, prior, prior_returns_numpy)
check_sbi_inputs(simulator_fn, prior)

num_simulations = 10
num_workers = 6
theta, x = simulate_for_sbi(
    simulator_fn, proposal=prior, num_simulations=num_simulations, num_workers=num_workers
)

theta_tra = theta.clone().detach()
x_tra = x.clone().detach()

nde_nsf = posterior_nn(model="nsf", z_score_x='structured')
inference = NPE(prior=prior, density_estimator=nde_nsf, device=device)
print(type(theta_tra))
print(type(x_tra))
inference = inference.append_simulations(theta_tra, x_tra, proposal=prior, data_device=device)

if __name__ == '__main__':
    max_num_epochs = 100
    batch_size = 20
    lr = 1e-3
    clip = 5

    density_estimator = inference.train(
        show_train_summary=True, max_num_epochs=max_num_epochs,
        learning_rate=lr, training_batch_size=batch_size,
        validation_fraction=0.5, clip_max_norm=clip,
    )