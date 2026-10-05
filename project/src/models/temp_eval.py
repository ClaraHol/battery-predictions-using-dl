import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CHEMISTRIES = ["25R", "HG2", "M1A", "MJ1", "PA", "PBC", "PD", "VTC5A", "VTC6"]


# ----------------------------------------------------------------------
# MLP architecture — must match training
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, hidden_features, dropout_probability, use_batch_norm):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.bn0 = nn.BatchNorm1d(hidden_features, eps=1e-3)
            self.bn1 = nn.BatchNorm1d(hidden_features, eps=1e-3)

        self.linear0 = nn.Linear(hidden_features, hidden_features)
        self.linear1 = nn.Linear(hidden_features, hidden_features)

        self.dropout = nn.Dropout(dropout_probability)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = x

        if self.use_batch_norm:
            h = self.bn0(h)

        h = self.activation(h)
        h = self.linear0(h)

        if self.use_batch_norm:
            h = self.bn1(h)

        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear1(h)

        return x + h


class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.initial_layer = nn.Linear(in_features, 256)

        self.blocks = nn.ModuleList([
            ResidualBlock(256, 0.2, True)
            for _ in range(9)
        ])

        self.final_layer = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.initial_layer(x)

        for block in self.blocks:
            x = block(x)

        return self.final_layer(x)


# ----------------------------------------------------------------------
# Load combined MLP
# ----------------------------------------------------------------------

model_path = (
    PROJECT_ROOT
    / "models"
    / "MLP_models"
    / "Combined"
    / "mlp_combined.pt"
)

checkpoint = torch.load(model_path, map_location=device)

prior_min = checkpoint["prior_min"].to(device)
prior_max = checkpoint["prior_max"].to(device)
prior_range = prior_max - prior_min
parameter_names = checkpoint["parameter_names"]

model = ResidualNetRegressor(
    in_features=100,
    out_features=len(parameter_names),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded: {model_path}")


# ----------------------------------------------------------------------
# Evaluate every chemistry
# ----------------------------------------------------------------------

all_errors = []

for chemistry in CHEMISTRIES:

    path = (
        PROJECT_ROOT
        / "models"
        / "Validation_simulations"
        / f"dataset_{chemistry}.pt"
    )

    theta, x = torch.load(path, map_location=device)

    theta = theta.float()
    x = x.float()

    # Remove failed simulations
    finite = (
        torch.isfinite(x).all(dim=1)
        & torch.isfinite(theta).all(dim=1)
    )

    theta = theta[finite]
    x = x[finite]

    # Predict normalized theta
    with torch.no_grad():
        pred_normalized = model(x)

    # Return predictions to physical parameter space
    pred = pred_normalized * prior_range + prior_min

    # Normalized absolute error
    errors = torch.abs(pred - theta) / prior_range

    all_errors.append(errors.cpu())

    print(
        f"{chemistry:6s} | "
        f"N = {len(theta):5d} | "
        f"MAE = {errors.mean().item():.6f}"
    )


# ----------------------------------------------------------------------
# Combined results
# ----------------------------------------------------------------------

all_errors = torch.cat(all_errors)

feature_mae = all_errors.mean(dim=0)

print("\n" + "=" * 60)
print("Combined validation results")
print("=" * 60)

for name, error in zip(parameter_names, feature_mae):
    print(f"{name:30s} {error.item():.6f}")

print("-" * 60)
print(f"{'Overall':30s} {feature_mae.mean().item():.6f}")