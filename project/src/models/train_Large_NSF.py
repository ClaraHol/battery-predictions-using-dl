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

print("Python executable:", sys.executable)
print("Python version:", sys.version)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())


# ==============================================================================
# Chemistry configuration
# ==============================================================================

CHEMISTRY_CONFIG = {
    "25R": {"params_module": "src.simulation.parameters.parameter_25R"},
    "HG2": {"params_module": "src.simulation.parameters.parameter_HG2"},
    "M1A": {"params_module": "src.simulation.parameters.parameter_M1A"},
    "MJ1": {"params_module": "src.simulation.parameters.parameter_MJ1"},
    "PA": {"params_module": "src.simulation.parameters.parameter_PA"},
    "PBC": {"params_module": "src.simulation.parameters.parameter_PBC"},
    "PD": {"params_module": "src.simulation.parameters.parameter_PD"},
    "VTC5A": {"params_module": "src.simulation.parameters.parameter_VTC5A"},
    "VTC6": {"params_module": "src.simulation.parameters.parameter_VTC6"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

num_dim = 11


# ==============================================================================
# 1. Construct a common prior covering ALL chemistries
# ==============================================================================

all_prior_min = []
all_prior_max = []

for chemistry, config in CHEMISTRY_CONFIG.items():

    params_module = importlib.import_module(config["params_module"])
    params_log = params_module.params_log

    chemistry_min = [
        list(params_log.values())[i][0]
        for i in range(num_dim)
    ]

    chemistry_max = [
        list(params_log.values())[i][1]
        for i in range(num_dim)
    ]

    all_prior_min.append(chemistry_min)
    all_prior_max.append(chemistry_max)


all_prior_min = np.asarray(all_prior_min)
all_prior_max = np.asarray(all_prior_max)

# Union of all chemistry-specific parameter ranges
prior_min = all_prior_min.min(axis=0)
prior_max = all_prior_max.max(axis=0)

print("\nCombined prior:")
for i, (low, high) in enumerate(zip(prior_min, prior_max)):
    print(f"Parameter {i:2d}: [{low:.6g}, {high:.6g}]")


prior = utils.BoxUniform(
    low=torch.as_tensor(prior_min, dtype=torch.float32),
    high=torch.as_tensor(prior_max, dtype=torch.float32),
)

prior, num_parameters, prior_returns_numpy = process_prior(prior)


# ==============================================================================
# 2. Load ALL datasets
# ==============================================================================

dataset_dir = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "Full_simulations"
)

theta_list = []
x_list = []

print("\nLoading datasets:")

for chemistry in CHEMISTRY_CONFIG:

    dataset_path = dataset_dir / f"dataset_{chemistry}.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No dataset found for {chemistry} at {dataset_path}"
        )

    theta, x = torch.load(dataset_path)

    print(
        f"{chemistry:6s}: "
        f"theta {tuple(theta.shape)}, "
        f"x {tuple(x.shape)}"
    )

    theta_list.append(theta)
    x_list.append(x)


# ==============================================================================
# 3. Combine all datasets
# ==============================================================================

theta_tra = torch.cat(theta_list, dim=0)
x_tra = torch.cat(x_list, dim=0)

print("\nCombined dataset:")
print("theta:", theta_tra.shape)
print("x:    ", x_tra.shape)


# ==============================================================================
# 4. Randomly shuffle simulations
# ==============================================================================

generator = torch.Generator()
generator.manual_seed(args.seed)

permutation = torch.randperm(
    theta_tra.shape[0],
    generator=generator
)

theta_tra = theta_tra[permutation]
x_tra = x_tra[permutation]

print(f"\nShuffled {theta_tra.shape[0]} simulations.")
print(f"Random seed: {args.seed}")


# ==============================================================================
# 5. Hyperparameters
# ==============================================================================

# spline flow
flow_step = 9
bin = 9
tail_bound = 3.0

# resnet
res_block = 9
hidden_feature = int(2**8)
dropout = 0.05
use_batch_norm = True

# training
max_num_epochs = args.max_epochs
batch_size = 256*4
lr = 0.00005
clip = 21.0


# ==============================================================================
# 6. Create SBI model
# ==============================================================================

nde_nsf = posterior_nn(
    model="nsf",
    z_score_x="structured",
)

inference = NPE(
    show_progress_bars=True,
    prior=prior,
    density_estimator=nde_nsf,
    device=device,
)

inference = inference.append_simulations(
    theta_tra,
    x_tra,
    proposal=prior,
    data_device=device,
)


# ==============================================================================
# 7. Output directory
# ==============================================================================

output_dir = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "SBI_models"
    / "Combined"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 8. Train
# ==============================================================================

if __name__ == "__main__":

    print("\nStarting combined training")
    print(f"Total simulations: {len(theta_tra):,}")

    start = time.time()

    inference.train(
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
        learning_rate=lr,
        training_batch_size=batch_size,
        validation_fraction=0.2,
        clip_max_norm=clip,
    )

    end = time.time()

    print(f"\nTraining time: {end - start:.2f} seconds")


    # --------------------------------------------------------------------------
    # Save inference object
    # --------------------------------------------------------------------------

    model_path = output_dir / "inference_combined.pkl"

    with open(model_path, "wb") as handle:
        pickle.dump(inference, handle)

    print(f"Saved model to {model_path}")


    # --------------------------------------------------------------------------
    # Save training curves
    # --------------------------------------------------------------------------

    fig, axes = plot_summary(
        inference,
        tags=["training_loss", "validation_loss"],
        figsize=(10, 4),
    )

    summary_path = output_dir / "training_summary_combined.png"

    fig.savefig(
        summary_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved training summary to {summary_path}")