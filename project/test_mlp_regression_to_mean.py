"""Quantify regression-to-the-mean/shrinkage of the trained MLP on held-out simulations.

Run from the repository root or from this directory. Edit CHEMISTRY if needed.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CHEMISTRY = "VTC6"
DATASET = ROOT / "models" / "Full_simulations" / f"dataset_{CHEMISTRY}.pt"
CHECKPOINT = ROOT / "models" / "MLP_models"/ CHEMISTRY / f"mlp_{CHEMISTRY}.pt"
OUT = ROOT / f"mlp_shrinkage_{CHEMISTRY}.csv"

# Same architecture as train_MLPs.py.
HIDDEN = 256
BLOCKS = 9
DROPOUT = 0.2
USE_BATCH_NORM = True

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
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_features, dropout_probability, use_batch_norm)
            for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.final_layer(h)

obj = torch.load(DATASET, map_location="cpu", weights_only=False)
theta, x = obj[:2]
theta = theta.float()
x = x.float()
valid = torch.isfinite(theta).all(1) & torch.isfinite(x).all(1)
theta = theta[valid]
x = x[valid]

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
prior_min = torch.as_tensor(ckpt["prior_min"]).float()
prior_max = torch.as_tensor(ckpt["prior_max"]).float()
names = ckpt["parameter_names"]

model = ResidualNetRegressor(x.shape[1], theta.shape[1], HIDDEN, BLOCKS, DROPOUT, USE_BATCH_NORM)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

preds = []
with torch.no_grad():
    for i in range(0, len(x), 1024):
        preds.append(model(x[i:i+1024]))
pred = torch.cat(preds)

scale = prior_max - prior_min

# IMPORTANT: the MLP was trained against theta_normalized, so its
# output is ALREADY in prior-normalized coordinates. Do not normalize
# the prediction a second time.
true_n = ((theta - prior_min) / scale).numpy()
pred_n = pred.numpy()

if not np.isfinite(pred_n).all():
    raise ValueError("MLP predictions contain NaN/Inf.")

print(
    "Normalized target range:",
    float(true_n.min()), float(true_n.max())
)
print(
    "MLP output range:",
    float(pred_n.min()), float(pred_n.max())
)

rows = []
for j, name in enumerate(names):
    y = true_n[:, j]
    yhat = pred_n[:, j]
    slope, intercept = np.polyfit(y, yhat, 1)
    r = np.corrcoef(y, yhat)[0, 1]
    true_std = np.std(y)
    pred_std = np.std(yhat)
    rows.append({
        "parameter": name,
        "slope_pred_vs_true": slope,
        "intercept": intercept,
        "r2": r*r,
        "true_std": true_std,
        "pred_std": pred_std,
        "std_ratio_pred_over_true": pred_std / max(true_std, 1e-12),
        "normalized_mae": np.mean(np.abs(yhat-y)),
        "normalized_bias": np.mean(yhat-y),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(f"\nWrote {OUT}")
