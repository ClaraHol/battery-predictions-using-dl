import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sbi import utils as utils
from sbi.utils.user_input_checks import process_prior


# ============================================================================
# Setup
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


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
parser.add_argument(
    "--chemistry",
    required=True,
    choices=CHEMISTRY_CONFIG.keys(),
)
parser.add_argument(
    "--n-posterior-samples",
    type=int,
    default=1000,
    help="Number of NSF posterior samples per voltage curve.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
)
args = parser.parse_args()

# ============================================================================
# Batched NSF posterior-mean evaluation
# ============================================================================

def evaluate_nsf_batched(
    posterior,
    x,
    n_samples=1000,
    batch_size=256,
):
    """
    Estimate E[theta | x] for every observation in x using batched
    posterior sampling.

    Parameters
    ----------
    posterior
        SBI posterior returned by inference.build_posterior().
    x : torch.Tensor
        Shape (N, x_dim).
    n_samples : int
        Number of posterior samples used to estimate each posterior mean.
    batch_size : int
        Number of observations evaluated simultaneously.

    Returns
    -------
    torch.Tensor
        Posterior means with shape (N, theta_dim).
    """

    posterior_means = []

    n_observations = x.shape[0]

    # Access the trained conditional density estimator.
    estimator = posterior.posterior_estimator
    estimator.eval()

    for start in range(0, n_observations, batch_size):

        end = min(
            start + batch_size,
            n_observations,
        )

        x_batch = x[start:end].to(device)

        current_batch_size = x_batch.shape[0]

        with torch.no_grad():

            samples = estimator.sample(
                sample_shape=torch.Size([n_samples]),
                condition=x_batch,
            )

        # Depending on SBI version this should normally be
        #
        # (n_samples, batch_size, theta_dim)
        #
        # but check explicitly.

        if start == 0:
            print(
                "NSF batched sample shape:",
                samples.shape,
            )

        # Expected:
        # samples.shape =
        # (n_samples, batch_size, theta_dim)

        if samples.shape[0] == n_samples:

            batch_mean = samples.mean(dim=0)

        # Some versions/backends may return
        # (batch_size, n_samples, theta_dim)
        elif samples.shape[1] == n_samples:

            batch_mean = samples.mean(dim=1)

        else:
            raise RuntimeError(
                "Unexpected posterior sample shape: "
                f"{samples.shape}"
            )

        assert batch_mean.shape[0] == current_batch_size

        posterior_means.append(
            batch_mean.cpu()
        )

        print(
            f"\rNSF: {end:5d}/{n_observations}",
            end="",
            flush=True,
        )

    print()

    return torch.cat(
        posterior_means,
        dim=0,
    )


# ============================================================================
# MLP architecture
# Must be identical to the architecture used during training.
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_features,
        dropout_probability,
        use_batch_norm,
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn0 = nn.BatchNorm1d(hidden_features, eps=1e-3)
            self.bn1 = nn.BatchNorm1d(hidden_features, eps=1e-3)

        self.linear0 = nn.Linear(hidden_features, hidden_features)
        self.linear1 = nn.Linear(hidden_features, hidden_features)

        self.dropout = nn.Dropout(p=dropout_probability)
        self.activation = nn.ReLU()

    def forward(self, inputs):

        temps = inputs

        if self.use_batch_norm:
            temps = self.bn0(temps)

        temps = self.activation(temps)
        temps = self.linear0(temps)

        if self.use_batch_norm:
            temps = self.bn1(temps)

        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear1(temps)

        return inputs + temps


class ResidualNetRegressor(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        num_blocks,
        dropout_probability,
        use_batch_norm,
    ):
        super().__init__()

        self.initial_layer = nn.Linear(
            in_features,
            hidden_features,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_features,
                    dropout_probability,
                    use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = nn.Linear(
            hidden_features,
            out_features,
        )

    def forward(self, x):

        h = self.initial_layer(x)

        for block in self.blocks:
            h = block(h)

        return self.final_layer(h)


# ============================================================================
# Load validation data
# ============================================================================

validation_path = (
    PROJECT_ROOT
    / "Validation_simulations"
    / "dataset.pt"
)

print(f"\nLoading validation data from:\n{validation_path}")

theta_true, x_val = torch.load(
    validation_path,
    map_location="cpu",
)

theta_true = theta_true.float()
x_val = x_val.float()

print("theta:", theta_true.shape)
print("x:", x_val.shape)

num_validation = theta_true.shape[0]
theta_dim = theta_true.shape[1]
x_dim = x_val.shape[1]


# ============================================================================
# Construct prior
# ============================================================================

config = CHEMISTRY_CONFIG[args.chemistry]

params_module = importlib.import_module(
    config["params_module"]
)

params_log = params_module.params_log

param_names = list(params_log.keys())

prior_min = torch.tensor(
    [bounds[0] for bounds in params_log.values()],
    dtype=torch.float32,
)

prior_max = torch.tensor(
    [bounds[1] for bounds in params_log.values()],
    dtype=torch.float32,
)

prior_range = prior_max - prior_min

prior = utils.BoxUniform(
    low=prior_min,
    high=prior_max,
)

prior, _, _ = process_prior(prior)


# ============================================================================
# Load NSF
# ============================================================================

sbi_path = (
    PROJECT_ROOT
    / "models"
    / "SBI_models"
    / args.chemistry
    / f"inference_{args.chemistry}.pkl"
)

print(f"\nLoading NSF:\n{sbi_path}")

with open(sbi_path, "rb") as handle:
    inference = pickle.load(handle)

posterior = inference.build_posterior(
    prior=prior
)


# ============================================================================
# Load MLP
# ============================================================================

mlp_path = (
    PROJECT_ROOT
    / "models"
    / "MLP_models"
    / args.chemistry
    / f"mlp_{args.chemistry}.pt"
)

print(f"\nLoading MLP:\n{mlp_path}")


# Same hyperparameters as training
hidden_feature = 2**8
res_block = 9
dropout = 0.05
use_batch_norm = True


model = ResidualNetRegressor(
    in_features=x_dim,
    out_features=theta_dim,
    hidden_features=hidden_feature,
    num_blocks=res_block,
    dropout_probability=dropout,
    use_batch_norm=use_batch_norm,
).to(device)


checkpoint = torch.load(
    mlp_path,
    map_location=device,
)


# Support both:
#
# old:
# torch.save(model.state_dict(), ...)
#
# and new:
# torch.save({
#     "model_state_dict": ...,
#     "prior_min": ...,
#     ...
# })

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)


model.eval()


# ============================================================================
# Evaluate MLP
# ============================================================================

print("\nEvaluating MLP...")

mlp_predictions_normalized = []

with torch.no_grad():

    for start in range(
        0,
        num_validation,
        args.batch_size,
    ):

        end = min(
            start + args.batch_size,
            num_validation,
        )

        xb = x_val[start:end].to(device)

        pred = model(xb)

        mlp_predictions_normalized.append(
            pred.cpu()
        )


mlp_predictions_normalized = torch.cat(
    mlp_predictions_normalized,
    dim=0,
)


# The new MLP predicts theta normalized to [0,1].
#
# Convert back into the original parameter coordinates.

mlp_predictions = (
    mlp_predictions_normalized
    * prior_range
    + prior_min
)

# ============================================================================
# Evaluate NSF
# ============================================================================

print("\nEvaluating NSF...")

nsf_predictions = evaluate_nsf_batched(
    posterior=posterior,
    x=x_val,
    n_samples=args.n_posterior_samples,
    batch_size=args.batch_size,
)

print(
    "NSF predictions:",
    nsf_predictions.shape,
)

# ============================================================================
# Normalized errors
# ============================================================================

#
# Error definition:
#
#     ((prediction - truth) / prior_range)^2
#
# This means every parameter is measured relative to the width
# of its own prior.
#

mlp_errors = (
    (mlp_predictions - theta_true)
    / prior_range
) ** 2


nsf_errors = (
    (nsf_predictions - theta_true)
    / prior_range
) ** 2


# Mean over the 10,000 validation curves
mlp_feature_mse = mlp_errors.mean(dim=0)
nsf_feature_mse = nsf_errors.mean(dim=0)


# ============================================================================
# Print results
# ============================================================================

print("\n")
print("=" * 90)
print(f"Results: {args.chemistry}")
print("=" * 90)

print(
    f"{'Parameter':40s}"
    f"{'NSF normalized MSE':>22s}"
    f"{'MLP normalized MSE':>22s}"
)

print("-" * 90)

for name, nsf_err, mlp_err in zip(
    param_names,
    nsf_feature_mse,
    mlp_feature_mse,
):

    print(
        f"{name:40s}"
        f"{nsf_err.item():22.6f}"
        f"{mlp_err.item():22.6f}"
    )


print("-" * 90)

print(
    f"{'Overall':40s}"
    f"{nsf_feature_mse.mean().item():22.6f}"
    f"{mlp_feature_mse.mean().item():22.6f}"
)


# ============================================================================
# Save results
# ============================================================================

output_dir = (
    PROJECT_ROOT
    / "models"
    / "Validation_results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


results = {
    "chemistry": args.chemistry,
    "parameter_names": param_names,

    "theta_true": theta_true,

    "nsf_predictions": nsf_predictions,
    "mlp_predictions": mlp_predictions,

    "nsf_errors": nsf_errors,
    "mlp_errors": mlp_errors,

    "nsf_feature_mse": nsf_feature_mse,
    "mlp_feature_mse": mlp_feature_mse,

    "nsf_overall_mse": nsf_feature_mse.mean(),
    "mlp_overall_mse": mlp_feature_mse.mean(),

    "prior_min": prior_min,
    "prior_max": prior_max,
}


output_path = (
    output_dir
    / f"validation_{args.chemistry}.pt"
)

torch.save(
    results,
    output_path,
)

print(f"\nSaved results to:\n{output_path}")