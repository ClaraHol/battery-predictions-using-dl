import torch
import torch.nn as nn
import numpy as np
import sys
import time
import argparse
import importlib
import matplotlib.pyplot as plt
from pathlib import Path

# This script was generated using claude and ChatGPT to provide an comparison the NSFs

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
print("CUDA available:", torch.cuda.is_available())

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
parser.add_argument("--max-epochs", type=int, default=1000)
parser.add_argument(
    "--patience",
    type=int,
    default=30,
    help="Early stopping patience (epochs with no val improvement)",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

num_dim = 9


# ------------------------------------------------------------------------
# 1. Build combined parameter bounds across ALL chemistries
# ------------------------------------------------------------------------

all_prior_min = []
all_prior_max = []
parameter_names = None

print("Loading parameter bounds:")
for chemistry, config in CHEMISTRY_CONFIG.items():

    params_module = importlib.import_module(config["params_module"])
    params_log = params_module.params_log

    # Only the first 9 parameters are SBI/MLP targets
    current_parameter_names = list(params_log.keys())[:num_dim]

    if parameter_names is None:
        parameter_names = current_parameter_names
    elif current_parameter_names != parameter_names:
        raise ValueError(
            f"Parameter ordering for {chemistry} does not match the other chemistries."
        )

    chemistry_min = torch.tensor(
        [bounds[0] for bounds in list(params_log.values())[:num_dim]],
        dtype=torch.float32,
    )

    chemistry_max = torch.tensor(
        [bounds[1] for bounds in list(params_log.values())[:num_dim]],
        dtype=torch.float32,
    )

    all_prior_min.append(chemistry_min)
    all_prior_max.append(chemistry_max)


all_prior_min = torch.stack(all_prior_min)
all_prior_max = torch.stack(all_prior_max)

# Bounding box covering the parameter ranges of every chemistry
prior_min = all_prior_min.min(dim=0).values
prior_max = all_prior_max.max(dim=0).values

prior_range = prior_max - prior_min

if torch.any(prior_range <= 0):
    raise ValueError("Every combined prior must have upper bound > lower bound.")


print("\nCombined parameter bounds:")

for name, low, high in zip(parameter_names, prior_min, prior_max):
    print(f"{name:20s}: [{low:.6g}, {high:.6g}]")


def normalize_theta(theta):
    return (theta - prior_min) / prior_range


def denormalize_theta(theta_normalized):
    return theta_normalized * prior_range + prior_min


# ------------------------------------------------------------------------
# 2. Load ALL simulated datasets
# ------------------------------------------------------------------------

dataset_dir = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "Varied_simulations"
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

    theta = theta.float()
    x = x.float()

    print(
        f"\n{chemistry}: "
        f"theta {tuple(theta.shape)}, "
        f"x {tuple(x.shape)}"
    )

    # ------------------------------------------------------------
    # Remove NaN/Inf simulations from THIS chemistry
    # ------------------------------------------------------------

    valid_mask = (
        torch.isfinite(x).all(dim=1)
        & torch.isfinite(theta).all(dim=1)
    )

    n_total = len(valid_mask)
    n_valid = valid_mask.sum().item()
    n_invalid = n_total - n_valid

    print(
        f"  Valid:   {n_valid}/{n_total} "
        f"({100 * n_valid / n_total:.2f}%)"
    )

    print(
        f"  Removed: {n_invalid}/{n_total} "
        f"({100 * n_invalid / n_total:.2f}%)"
    )

    theta = theta[valid_mask]
    x = x[valid_mask]

    theta_list.append(theta)
    x_list.append(x)


# ------------------------------------------------------------------------
# 3. Combine all chemistries
# ------------------------------------------------------------------------

theta = torch.cat(theta_list, dim=0)
x = torch.cat(x_list, dim=0)



assert torch.isfinite(theta).all()
assert torch.isfinite(x).all()

if theta.shape[1] != 9 or x.shape[1] != 108:
    raise ValueError(
        f"{chemistry}: expected theta [N,9] and x [N,108], "
        f"got {tuple(theta.shape)} and {tuple(x.shape)}"
    )

print("\nCombined dataset:")
print("theta:", theta.shape)
print("x:    ", x.shape)

x_dim = x.shape[1]
if x_dim != 108:
    raise ValueError(f"Expected x dimension 108, got {x_dim}")
theta_dim = theta.shape[1]

if theta_dim != len(prior_min):
    raise ValueError(
        f"Dataset has {theta_dim} target parameters, "
        f"but {len(prior_min)} parameter bounds were defined."
    )


# ------------------------------------------------------------------------
# 4. Normalize all targets using the SAME combined parameter bounds
# ------------------------------------------------------------------------

theta_normalized = normalize_theta(theta)

print(
    "\nNormalized theta using combined bounds "
    "(each target maps to the global parameter range [0, 1])."
)


# ------------------------------------------------------------------------
# 3. Hyper-parameters, matched to the spline-flow's internal conditioner
#    network (see train_model.py) so the two architectures are comparable
# ------------------------------------------------------------------------
res_block = 9
hidden_feature = int(2**8)
dropout = 0.2
use_batch_norm = True

max_num_epochs = args.max_epochs
batch_size = 256*8
lr = 1*1e-4
clip = 21.0
validation_fraction = 0.2


# ------------------------------------------------------------------------
# 4. Model: a ResidualNet regressor, architecturally the same "block" used
#    inside each coupling layer of the NSF (res_block residual blocks of
#    hidden_feature units), but here mapping x -> theta directly instead
#    of x -> spline parameters.
# ------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_features, dropout_probability, use_batch_norm):
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
    def __init__(self, in_features, out_features, hidden_features, num_blocks, dropout_probability, use_batch_norm):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_features, dropout_probability, use_batch_norm)
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.final_layer(h)


model = ResidualNetRegressor(
    in_features=x_dim,
    out_features=theta_dim,
    hidden_features=hidden_feature,
    num_blocks=res_block,
    dropout_probability=dropout,
    use_batch_norm=use_batch_norm,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MLP regressor parameter count: {num_params:,}")
print("(Compare against your saved spline-flow model's parameter count; "
      "the flow stacks several such blocks across flow_step coupling layers, "
      "so its total is expected to be larger — scale hidden_feature/res_block "
      "up here if you want a closer match.)")

# ------------------------------------------------------------------------
# 5. Train/validation split (same validation_fraction as the flow training)
# ------------------------------------------------------------------------
n = theta_normalized.shape[0]
n_val = max(1, int(n * validation_fraction))
perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
val_idx = perm[:n_val]
train_idx = perm[n_val:]

theta_train, x_train = theta_normalized[train_idx].to(device), x[train_idx].to(device)
theta_val, x_val = theta_normalized[val_idx].to(device), x[val_idx].to(device)

x_mean = x[train_idx].mean(dim=0)
x_std = x[train_idx].std(dim=0)
x_std = torch.clamp(x_std, min=1e-6)

x_train = (x[train_idx] - x_mean) / x_std
x_val = (x[val_idx] - x_mean) / x_std

train_dataset = torch.utils.data.TensorDataset(x_train, theta_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,      # multiply LR by 0.5
    patience=3,      # wait 3 epochs without improvement
    min_lr=1e-8,
)

output_dir = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "MLP_models"
    / "Combined"
)

output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Starting training")
    start = time.time()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(max_num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, thetab in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, thetab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, theta_val).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train: {epoch_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

    
        if epochs_since_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    end = time.time()
    print("time used:", end - start)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prior_min": prior_min,
            "prior_max": prior_max,
            "parameter_names": parameter_names,
            "x_mean": x_mean,
            "x_std": x_std,
        },
        output_dir / "mlp_varied_combined.pt",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="training_loss")
    ax.plot(val_losses, label="validation_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (prior-normalized parameters)")
    ax.set_title(f"MLP training summary: s")
    ax.legend()
    fig.savefig(
        output_dir / "mlp_training_summary_combined.png",
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved model and training summary to {output_dir}")