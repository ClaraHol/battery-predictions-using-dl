limport torch
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
parser.add_argument("--chemistry", required=True, choices=CHEMISTRY_CONFIG.keys())
parser.add_argument("--max-epochs", type=int, default=5000)
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs with no val improvement)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print("Chemistry used:", args.chemistry)

# ------------------------------------------------------------------------
# 1. Rebuild the parameter priors used by SBI.
#    The MLP is trained on parameters scaled to [0, 1] according to these
#    prior bounds. Consequently, an MSE of the same size in any parameter
#    corresponds to the same fraction of that parameter's prior range.
# ------------------------------------------------------------------------
params_module = importlib.import_module(CHEMISTRY_CONFIG[args.chemistry]["params_module"])
params_log = params_module.params_log

prior_min = torch.tensor(
    [bounds[0] for bounds in params_log.values()], dtype=torch.float32
)
prior_max = torch.tensor(
    [bounds[1] for bounds in params_log.values()], dtype=torch.float32
)
prior_range = prior_max - prior_min

if torch.any(prior_range <= 0):
    raise ValueError("Every prior must have upper bound > lower bound.")

def normalize_theta(theta):
    return (theta - prior_min) / prior_range

def denormalize_theta(theta_normalized):
    return theta_normalized * prior_range + prior_min

# ------------------------------------------------------------------------
# 2. Load the same simulated dataset used to train the spline-flow model
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
theta = theta.float()
x = x.float()
print(f"Loaded dataset: theta {theta.shape}, x {x.shape} from {dataset_path}")

x_dim = x.shape[1]
theta_dim = theta.shape[1]

if theta_dim != len(prior_min):
    raise ValueError(
        f"Dataset has {theta_dim} target parameters, but {args.chemistry} "
        f"defines {len(prior_min)} priors."
    )

# Normalize targets using the prior support, not dataset statistics.
# This makes all target dimensions contribute equally to the MSE.
theta_normalized = normalize_theta(theta)
print("Normalized theta using prior bounds (each target maps to prior range [0, 1]).")

# ------------------------------------------------------------------------
# 3. Hyper-parameters, matched to the spline-flow's internal conditioner
#    network (see train_model.py) so the two architectures are comparable
# ------------------------------------------------------------------------
res_block = 9
hidden_feature = int(2**8)
dropout = 0.05
use_batch_norm = True

max_num_epochs = args.max_epochs
batch_size = 256
lr = 0.00005
clip = 21.0
validation_fraction = 0.1


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

train_dataset = torch.utils.data.TensorDataset(x_train, theta_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

output_dir = Path(__file__).resolve().parents[2] / "models" / "MLP_models" / args.chemistry
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 50 == 0 or epoch == max_num_epochs - 1:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

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
            "parameter_names": list(params_log.keys()),
        },
        output_dir / f"mlp_{args.chemistry}.pt",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="training_loss")
    ax.plot(val_losses, label="validation_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (prior-normalized parameters)")
    ax.set_title(f"MLP training summary: {args.chemistry}")
    ax.legend()
    fig.savefig(output_dir / f"mlp_training_summary_{args.chemistry}.png", dpi=150, bbox_inches="tight")

    print(f"Saved model and training summary to {output_dir}")
