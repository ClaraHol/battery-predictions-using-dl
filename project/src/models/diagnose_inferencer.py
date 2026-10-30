from pathlib import Path
import importlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sbi import utils as sbi_utils
from sbi.utils.user_input_checks import process_prior

# --------------------------- editable constants ---------------------------
CHEMISTRY = "MJ1"
N_POSTERIOR_SAMPLES = 2000
N_POSTERIOR_CASES = 200   # prior-constrained sampling is slower; start small
SEED = 0
# ------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

PARAM_MODULE = f"src.simulation.parameters.parameter_{CHEMISTRY}"
params_module = importlib.import_module(PARAM_MODULE)
params_log = params_module.params_log
current_names = list(params_log.keys())
current_min = torch.tensor([b[0] for b in params_log.values()], dtype=torch.float32)
current_max = torch.tensor([b[1] for b in params_log.values()], dtype=torch.float32)

val_path = PROJECT_ROOT / "models" / "Validation_simulations" / f"dataset_{CHEMISTRY}.pt"
mlp_path = PROJECT_ROOT / "models" / "MLP_models" / CHEMISTRY / f"mlp_{CHEMISTRY}.pt"
sbi_path = PROJECT_ROOT / "models" / "SBI_models" / CHEMISTRY / f"inference_{CHEMISTRY}.pkl"

theta_true, x_val = torch.load(val_path, map_location="cpu")
theta_true = theta_true.float()
x_val = x_val.float()
finite = torch.isfinite(x_val).all(dim=1) & torch.isfinite(theta_true).all(dim=1)
theta_true = theta_true[finite]
x_val = x_val[finite]

checkpoint = torch.load(mlp_path, map_location="cpu")
if "model_state_dict" not in checkpoint:
    raise RuntimeError("Expected the newer checkpoint dictionary format.")

ckpt_names = list(checkpoint["parameter_names"])
ckpt_min = checkpoint["prior_min"].float().cpu()
ckpt_max = checkpoint["prior_max"].float().cpu()

print("\n=== Checkpoint contract ===")
print("names identical:", ckpt_names == current_names)
print("prior_min identical:", torch.allclose(ckpt_min, current_min, rtol=0, atol=0))
print("prior_max identical:", torch.allclose(ckpt_max, current_max, rtol=0, atol=0))
if ckpt_names != current_names:
    print("checkpoint names:", ckpt_names)
    print("current names:   ", current_names)
if not torch.allclose(ckpt_min, current_min) or not torch.allclose(ckpt_max, current_max):
    print("checkpoint prior_min:", ckpt_min.tolist())
    print("current prior_min:   ", current_min.tolist())
    print("checkpoint prior_max:", ckpt_max.tolist())
    print("current prior_max:   ", current_max.tolist())

# Always use the checkpoint metadata to interpret this checkpoint.
prior_min = ckpt_min
prior_max = ckpt_max
prior_range = prior_max - prior_min

# Dataset-vs-prior order/support diagnostic.
theta_norm = (theta_true - prior_min) / prior_range
print("\n=== Validation theta support ===")
for j, name in enumerate(ckpt_names):
    col = theta_norm[:, j]
    outside = ((col < -1e-6) | (col > 1 + 1e-6)).float().mean().item()
    print(f"{name:28s} min={col.min().item(): .4f} max={col.max().item(): .4f} outside={100*outside:6.2f}%")

class ResidualBlock(nn.Module):
    def __init__(self, hidden_features, dropout_probability=0.2, use_batch_norm=True):
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
        y = x
        if self.use_batch_norm:
            y = self.bn0(y)
        y = self.activation(y)
        y = self.linear0(y)
        if self.use_batch_norm:
            y = self.bn1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear1(y)
        return x + y

class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=256, num_blocks=9):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_features) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.final_layer(h)

model = ResidualNetRegressor(x_val.shape[1], theta_true.shape[1]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    mlp_norm = model(x_val.to(device)).cpu()
mlp = mlp_norm * prior_range + prior_min

print("\n=== MLP output support ===")
for j, name in enumerate(ckpt_names):
    z = mlp_norm[:, j]
    outside = ((z < 0) | (z > 1)).float().mean().item()
    print(f"{name:28s} pred_min={z.min().item(): .4f} pred_max={z.max().item(): .4f} outside={100*outside:6.2f}%")

# Midpoint baseline under normalized prior coordinates.
midpoint = torch.full_like(theta_norm, 0.5)
mlp_mae = (mlp_norm - theta_norm).abs().mean(dim=0)
base_mae = (midpoint - theta_norm).abs().mean(dim=0)

print("\n=== MLP vs truth, normalized MAE ===")
for j, name in enumerate(ckpt_names):
    print(f"{name:28s} baseline={base_mae[j].item():.5f} mlp={mlp_mae[j].item():.5f}")
print("overall baseline:", base_mae.mean().item())
print("overall MLP:     ", mlp_mae.mean().item())

# Load NSF and use the DirectPosterior for support-constrained samples.
prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)
prior, _, _ = process_prior(prior)
with open(sbi_path, "rb") as f:
    inference = pickle.load(f)
posterior = inference.build_posterior(prior=prior)

n_cases = min(N_POSTERIOR_CASES, len(x_val))
nsf_means = []
nsf_vars_norm = []
direct_flow_means = []
direct_flow_oob = []

print(f"\nSampling constrained NSF posterior for {n_cases} validation cases...")
for i in range(n_cases):
    x_i = x_val[i].to(device)

    # Prior-constrained posterior samples.
    s = posterior.sample(
        (N_POSTERIOR_SAMPLES,),
        x=x_i,
        show_progress_bars=False,
    ).detach().cpu()
    nsf_means.append(s.mean(dim=0))
    nsf_vars_norm.append((s / prior_range).var(dim=0, unbiased=False))

    # Raw density-estimator samples, for measuring support leakage only.
    with torch.no_grad():
        raw = posterior.posterior_estimator.sample(
            sample_shape=torch.Size([N_POSTERIOR_SAMPLES]),
            condition=x_i.unsqueeze(0),
        ).detach().cpu()
    raw = raw.reshape(-1, theta_true.shape[1])
    direct_flow_means.append(raw.mean(dim=0))
    direct_flow_oob.append(((raw < prior_min) | (raw > prior_max)).any(dim=1).float().mean())

    if (i + 1) % 10 == 0 or i + 1 == n_cases:
        print(f"  {i+1}/{n_cases}")

nsf_mean = torch.stack(nsf_means)
nsf_norm = (nsf_mean - prior_min) / prior_range
raw_mean = torch.stack(direct_flow_means)
raw_norm = (raw_mean - prior_min) / prior_range
post_var_norm = torch.stack(nsf_vars_norm)
flow_oob = torch.stack(direct_flow_oob)
truth_sub = theta_norm[:n_cases]
mlp_sub = mlp_norm[:n_cases]

print("\n=== Decisive comparison ===")
print("mean normalized MAE: MLP vs truth:          ", (mlp_sub - truth_sub).abs().mean().item())
print("mean normalized MAE: constrained NSF vs truth:", (nsf_norm - truth_sub).abs().mean().item())
print("mean normalized MAE: MLP vs constrained NSF: ", (mlp_sub - nsf_norm).abs().mean().item())
print("mean posterior std / prior width:            ", post_var_norm.sqrt().mean().item())
print("raw-flow sample fraction outside prior:      ", flow_oob.mean().item())
print("raw-flow mean vs constrained mean MAE:       ", (raw_norm - nsf_norm).abs().mean().item())

print("\nPer parameter: MLP vs constrained NSF mean")
for j, name in enumerate(ckpt_names):
    d = (mlp_sub[:, j] - nsf_norm[:, j]).abs().mean().item()
    pstd = post_var_norm[:, j].sqrt().mean().item()
    print(f"{name:28s} mean_abs_delta={d:.5f} posterior_std={pstd:.5f}")
