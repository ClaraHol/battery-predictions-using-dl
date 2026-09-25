import torch
import numpy as np
import sys
import time
import argparse
import importlib
import pickle
from pathlib import Path

from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
from sbi.analysis import plot_summary

sys.path.append(str(Path(__file__).resolve().parents[2]))

####### This file was entiraly generated be claude, based in prevouis implimentations which did training/simulation sepperate for each model

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
#   - Split into a standalone training step, separate from simulation
#   - Parameterized by --chemistry to match the simulation script
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
print("CUDA available:", torch.cuda.is_available())

# Keep this in sync with the simulation script's CHEMISTRY_CONFIG.
# Only params_module and output_subdir are actually needed here (to build the
# prior and locate the saved dataset); the simulation-only fields (Q_rated,
# current, dod, V_cut_lb, V_cut_ub) aren't needed at train time.
CHEMISTRY_CONFIG = {
    "25R": {"params_module": "src.simulation.parameters.parameter_25R", "output_subdir": "25R"},
    "HG2": {"params_module": "src.simulation.parameters.parameter_HG2", "output_subdir": "HG2"},
    "M1A": {"params_module": "src.simulation.parameters.parameter_M1A", "output_subdir": "M1A"},
    "MJ1": {"params_module": "src.simulation.parameters.parameter_MJ1", "output_subdir": "MJ1"},
    "PA": {"params_module": "src.simulation.parameters.parameter_PA", "output_subdir": "PA"},
    "PBC": {"params_module": "src.simulation.parameters.parameter_PBC", "output_subdir": "PBC"},
    "PD": {"params_module": "src.simulation.parameters.parameter_PD", "output_subdir": "PD"},
    "VTC5A": {"params_module": "src.simulation.parameters.parameter_VTC5A", "output_subdir": "VTC5A"},
    "VTC6": {"params_module": "src.simulation.parameters.parameter_VTC6", "output_subdir": "VTC6"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--chemistry", required=True, choices=CHEMISTRY_CONFIG.keys())
parser.add_argument("--max-epochs", type=int, default=5000)
args = parser.parse_args()

config = CHEMISTRY_CONFIG[args.chemistry]
print("Chemistry used:", args.chemistry)

params_module = importlib.import_module(config["params_module"])
params_log = params_module.params_log

num_dim = 11

# ------------------------------------------------------------------------
# 1. Rebuild the prior (must match what the simulation script used, since
#    it defines the parameter space the density estimator is trained over)
# ------------------------------------------------------------------------
prior_min = []
prior_max = []
for i in range(num_dim):
    prior_min.append(list(params_log.values())[i][0])
    prior_max.append(list(params_log.values())[i][1])
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# ------------------------------------------------------------------------
# 2. Load the simulated dataset produced by the simulation script
# ------------------------------------------------------------------------
dataset_path = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "Full_simulations"
    / f"dataset_{args.chemistry}.pt"
)
if not dataset_path.exists():
    raise FileNotFoundError(
        f"No dataset found at {dataset_path}. Run the simulation script for "
        f"--chemistry {args.chemistry} first."
    )

theta, x = torch.load(dataset_path)
print(f"Loaded dataset: theta {theta.shape}, x {x.shape} from {dataset_path}")

theta_tra = theta.clone().detach()
x_tra = x.clone().detach()

# ------------------------------------------------------------------------
# 3. Hyper-parameters (unchanged from the combined script)
# ------------------------------------------------------------------------
# spline flow
flow_step = 9
bin = 9
tail_bound = 3.0
# resnet
res_block = 9
hidden_feature = int(2**8)
dropout = 0.05
use_batch_norm = True
# training-related
max_num_epochs = args.max_epochs
batch_size = 256
lr = 0.00005
clip = 21.0

# ------------------------------------------------------------------------
# 4. Create and train the model
# ------------------------------------------------------------------------
nde_nsf = posterior_nn(model="nsf", z_score_x="structured")

inference = NPE(show_progress_bars=True, prior=prior, density_estimator=nde_nsf, device=device)
inference = inference.append_simulations(theta_tra, x_tra, proposal=prior, data_device=device)

output_dir = Path(__file__).resolve().parents[2] / "models" / "SBI_models" / config["output_subdir"]
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Starting training")
    start = time.time()
    inference.train(
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
        learning_rate=lr,
        training_batch_size=batch_size,
        validation_fraction=0.1,
        clip_max_norm=clip,
    )
    end = time.time()
    print("time used:", end - start)

    with open(output_dir / f"inference_{args.chemistry}.pkl", "wb") as handle:
        pickle.dump(inference, handle)

    fig, axes = plot_summary(inference, tags=["training_loss", "validation_loss"], figsize=(10, 4))
    fig.savefig(output_dir / f"training_summary_{args.chemistry}.png", dpi=150, bbox_inches="tight")
    print(f"Saved model and training summary to {output_dir}")