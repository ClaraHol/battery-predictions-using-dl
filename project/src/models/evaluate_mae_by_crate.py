
# Created purely using chatGPT
"""Evaluate the combined varied-protocol MLP by C-rate and direction.

Run from anywhere: python project/src/models/evaluate_mae_by_crate.py
Requires the same project directory structure as train_Large_varied_MLP.py.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CHEMISTRIES = ['25R', 'HG2', 'M1A', 'MJ1', 'PA', 'PBC', 'PD', 'VTC5A', 'VTC6']


class ResidualBlock(nn.Module):
    def __init__(self, hidden_features=256, dropout_probability=0.2, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn0 = nn.BatchNorm1d(hidden_features, eps=1e-3)
            self.bn1 = nn.BatchNorm1d(hidden_features, eps=1e-3)
        self.linear0 = nn.Linear(hidden_features, hidden_features)
        self.linear1 = nn.Linear(hidden_features, hidden_features)
        self.dropout = nn.Dropout(dropout_probability)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        h = self.bn0(inputs) if self.use_batch_norm else inputs
        h = self.activation(h)
        h = self.linear0(h)
        h = self.bn1(h) if self.use_batch_norm else h
        h = self.activation(h)
        h = self.dropout(h)
        return inputs + self.linear1(h)


class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features=108, out_features=9):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, 256)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(9)])
        self.final_layer = nn.Linear(256, out_features)

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.final_layer(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/MLP_models/Combined/mlp_varied_combined.pt')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'models/Varied_validation_simulations')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/mae_by_crate')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--bins', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    args = parser.parse_args()
    if len(args.bins) < 2 or any(b >= c for b, c in zip(args.bins, args.bins[1:])):
        parser.error('--bins must be strictly increasing')

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    names = ckpt['parameter_names']
    low = ckpt['prior_min'].float().to(device)
    width = (ckpt['prior_max'] - ckpt['prior_min']).float().to(device)
    model = ResidualNetRegressor(108, len(names)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    x_mean = ckpt.get('x_mean')
    x_std = ckpt.get('x_std')
    if (x_mean is None) != (x_std is None):
        raise ValueError('Checkpoint has only one of x_mean/x_std')
    if x_mean is not None:
        x_mean = x_mean.float().to(device)
        x_std = x_std.float().clamp(min=1e-6).to(device)

    bins = np.asarray(args.bins)
    # [chemistry, direction, C-rate bin, parameter]
    sums = np.zeros((len(CHEMISTRIES) + 1, 2, len(bins) - 1, len(names)), dtype=np.float64)
    counts = np.zeros(sums.shape[:-1], dtype=np.int64)
    for chem_idx, chemistry in enumerate(CHEMISTRIES):
        theta, x = torch.load(args.data_dir / f'dataset_{chemistry}.pt', map_location='cpu', weights_only=True)
        if theta.ndim != 2 or theta.shape[1] != len(names) or x.ndim != 2 or x.shape[1] != 108:
            raise ValueError(f'{chemistry}: expected theta [N,{len(names)}] and x [N,108], got {theta.shape}, {x.shape}')
        good = torch.isfinite(theta).all(dim=1) & torch.isfinite(x).all(dim=1)
        theta, x = theta[good].float(), x[good].float()
        print(f'{chemistry}: {len(x)} valid simulations')
        for xb, tb in zip(x.split(args.batch_size), theta.split(args.batch_size)):
            with torch.no_grad():
                inp = xb.to(device)
                if x_mean is not None:
                    inp = (inp - x_mean) / x_std
                pred = model(inp)
                truth = (tb.to(device) - low) / width
                err = (pred - truth).abs().cpu().numpy()
            rates = xb[:, 100].numpy()
            signs = xb[:, 101].numpy()
            for direction_idx, is_direction in enumerate((signs < 0, signs > 0)):
                for j in range(len(bins) - 1):
                    in_bin = (rates >= bins[j]) & (rates < bins[j + 1] if j < len(bins) - 2 else rates <= bins[j + 1])
                    selected = is_direction & in_bin
                    if not selected.any():
                        continue
                    n = int(selected.sum())
                    total = err[selected].sum(axis=0)
                    for idx in (chem_idx, len(CHEMISTRIES)):
                        counts[idx, direction_idx, j] += n
                        sums[idx, direction_idx, j] += total

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'mae_by_crate.csv'
    with csv_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['chemistry', 'direction', 'crate_low', 'crate_high', 'n', 'overall_mae', *names])
        for i, chem in enumerate([*CHEMISTRIES, 'ALL']):
            for d, direction in enumerate(['charge', 'discharge']):
                for j in range(len(bins) - 1):
                    n = counts[i, d, j]
                    maes = sums[i, d, j] / n if n else np.full(len(names), np.nan)
                    writer.writerow([chem, direction, bins[j], bins[j + 1], n, float(maes.mean()), *maes])
                    if chem == 'ALL':
                        print(f'{direction:9s} {bins[j]:.2f}-{bins[j+1]:.2f}C | N={n:6d} | MAE={maes.mean():.4f}')

    centers = (bins[:-1] + bins[1:]) / 2
    for d, direction in enumerate(['charge', 'discharge']):
        n = counts[-1, d]
        maes = np.divide(sums[-1, d], n[:, None], out=np.full_like(sums[-1, d], np.nan), where=n[:, None] > 0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(centers, maes.mean(axis=1), marker='o', label='Overall MAE')
        ax.set(xlabel='Applied C-rate', ylabel='Global-prior-normalized MAE', title=f'{direction.capitalize()}: prediction error vs C-rate')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f'{direction}_overall_mae.png', dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for p, name in enumerate(names):
            ax.plot(centers, maes[:, p], marker='.', label=name)
        ax.set(xlabel='Applied C-rate', ylabel='Global-prior-normalized MAE', title=f'{direction.capitalize()}: parameter MAE vs C-rate')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(args.output_dir / f'{direction}_parameter_mae.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(f'Wrote {csv_path} and four PNG plots')


if __name__ == '__main__':
    main()
