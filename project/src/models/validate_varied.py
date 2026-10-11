
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CHEMISTRIES = [
    "25R", "HG2", "M1A", "MJ1", "PA",
    "PBC", "PD", "VTC5A", "VTC6"
]

# ------------------------------------------------------------
# Model architecture (must match training)
# ------------------------------------------------------------

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
        h = self.bn0(x) if self.use_batch_norm else x
        h = self.activation(h)
        h = self.linear0(h)
        h = self.bn1(h) if self.use_batch_norm else h
        h = self.activation(h)
        h = self.dropout(h)
        return x + self.linear1(h)


class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features=108, out_features=9):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, 256)
        self.blocks = nn.ModuleList([
            ResidualBlock(256, 0.2, True) for _ in range(9)
        ])
        self.final_layer = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)


# ------------------------------------------------------------
# Load model and normalization
# ------------------------------------------------------------

model_path = (
    PROJECT_ROOT / "models" / "MLP_models"
    / "Combined" / "mlp_varied_combined.pt"
)

checkpoint = torch.load(
    model_path, map_location="cpu", weights_only=True
)

prior_min = checkpoint["prior_min"].float().to(device)
prior_max = checkpoint["prior_max"].float().to(device)
prior_range = prior_max - prior_min
parameter_names = checkpoint["parameter_names"]

model = ResidualNetRegressor(
    in_features=108,
    out_features=9
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Only standardize if the model was trained with this scaling.
has_scaling = "x_mean" in checkpoint and "x_std" in checkpoint

if has_scaling:
    x_mean = checkpoint["x_mean"].float().to(device)
    x_std = checkpoint["x_std"].float().to(device)
    x_std = x_std.clamp(min=1e-6)

print(f"Loaded: {model_path}")
print(f"Input scaling: {'enabled' if has_scaling else 'disabled'}")

# ------------------------------------------------------------
# Evaluate each chemistry
# ------------------------------------------------------------

dataset_dir = (
    PROJECT_ROOT / "models" / "Varied_validation_simulations"
)

all_errors = []

for chemistry in CHEMISTRIES:
    theta, x = torch.load(
        dataset_dir / f"dataset_{chemistry}.pt",
        map_location="cpu",
        weights_only=True
    )

    theta = theta.float()
    x = x.float()

    assert theta.shape[1] == 9
    assert x.shape[1] == 108

    valid = (
        torch.isfinite(theta).all(dim=1)
        & torch.isfinite(x).all(dim=1)
    )

    theta = theta[valid]
    x = x[valid]

    if len(theta) == 0:
        print(f"{chemistry}: no valid simulations")
        continue

    errors_parts = []

    with torch.no_grad():
        for xb, tb in zip(
            x.split(4096),
            theta.split(4096)
        ):
            xb = xb.to(device)
            tb = tb.to(device)

            if has_scaling:
                xb = (xb - x_mean) / x_std

            pred_normalized = model(xb)
            target_normalized = (tb - prior_min) / prior_range

            errors_parts.append(
                (pred_normalized - target_normalized).abs().cpu()
            )

    errors = torch.cat(errors_parts)
    all_errors.append(errors)

    print(
        f"{chemistry:6s} | "
        f"N = {len(theta):5d} | "
        f"MAE = {errors.mean().item():.6f}"
    )

# ------------------------------------------------------------
# Combined results
# ------------------------------------------------------------

all_errors = torch.cat(all_errors, dim=0)
feature_mae = all_errors.mean(dim=0)

print("\n" + "=" * 60)
print("Combined validation results")
print("=" * 60)

for name, error in zip(parameter_names, feature_mae):
    print(f"{name:30s} {error.item():.6f}")

print("-" * 60)
print(f"{'Overall':30s} {feature_mae.mean().item():.6f}")
