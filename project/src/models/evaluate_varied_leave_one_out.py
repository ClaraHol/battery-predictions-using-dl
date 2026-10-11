#!/usr/bin/env python3
"""Evaluate all leave-one-chemistry-out models on independent varied validation datasets.

For each model, report (1) all nine chemistries and (2) only its excluded
chemistry. MAE/MSE use common widest configured prior bounds across all nine chemistries.
"""
import argparse
import csv
import importlib
import sys
from pathlib import Path

import torch
from torch import nn

CHEMISTRIES = ('25R', 'HG2', 'M1A', 'MJ1', 'PA', 'PBC', 'PD', 'VTC5A', 'VTC6')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def global_bounds():
    lows, highs = [], []
    names = None
    for chem in CHEMISTRIES:
        module = importlib.import_module(f'src.simulation.parameters.parameter_{chem}')
        params = module.params_log
        current_names = list(params)[:9]
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError(f'Parameter ordering mismatch: {chem}')
        bounds = list(params.values())[:9]
        lows.append(torch.tensor([v[0] for v in bounds], dtype=torch.float64))
        highs.append(torch.tensor([v[1] for v in bounds], dtype=torch.float64))
    lower = torch.stack(lows).min(0).values
    upper = torch.stack(highs).max(0).values
    if ((upper - lower) <= 0).any():
        raise ValueError('Invalid global bounds')
    return names, (upper - lower).float()


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(256, eps=1e-3)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-3)
        self.linear0 = nn.Linear(256, 256)
        self.linear1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = self.bn0(x)
        h = self.linear0(self.activation(h))
        h = self.bn1(h)
        return x + self.linear1(self.dropout(self.activation(h)))


class ResidualNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_layer = nn.Linear(108, 256)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(9)])
        self.final_layer = nn.Linear(256, 9)

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)


def load_validation_data(folder):
    data = {}
    for chem in CHEMISTRIES:
        theta, x = torch.load(folder / f'dataset_{chem}.pt', map_location='cpu', weights_only=True)
        theta, x = theta.float(), x.float()
        if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != 9 or x.shape[1] != 108 or len(theta) != len(x):
            raise ValueError(f'{chem}: expected theta [N,9] and x [N,108]')
        valid = torch.isfinite(theta).all(1) & torch.isfinite(x).all(1)
        data[chem] = (theta[valid], x[valid])
        print(f'{chem:6s}: {valid.sum().item():5d} / {len(valid):5d} valid')
    return data


def evaluate(model, theta, x, prior_min, prior_range, global_prior_range, x_mean, x_std, device, batch_size):
    abs_sum = torch.zeros(9, dtype=torch.float64)
    sq_sum = torch.zeros(9, dtype=torch.float64)
    n = len(theta)
    if n == 0:
        raise ValueError('Validation dataset has no valid samples')
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size].to(device)
            tb = theta[i:i + batch_size].to(device)
            prediction = model((xb - x_mean) / x_std)
            pred_physical = prediction * prior_range + prior_min
            error = ((pred_physical - tb) / global_prior_range).cpu().double()
            abs_sum += error.abs().sum(0)
            sq_sum += error.square().sum(0)
    return n, abs_sum, sq_sum


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models/MLP_models/LeaveOneOut')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'models/Varied_validation_simulations')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    global_names, global_prior_range = global_bounds()
    global_prior_range = global_prior_range.to(device)
    data = load_validation_data(args.data_dir)
    summary, details = [], []

    for excluded in CHEMISTRIES:
        path = args.models_dir / f'excluded_{excluded}' / 'model.pt'
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if ckpt.get('excluded_chemistry') != excluded:
            raise ValueError(f'{path}: excluded chemistry mismatch')
        model = ResidualNetRegressor().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        pmin = ckpt['prior_min'].float().to(device)
        prange = (ckpt['prior_max'] - ckpt['prior_min']).float().to(device)
        xmean = ckpt['x_mean'].float().to(device)
        xstd = ckpt['x_std'].float().to(device).clamp_min(1e-6)
        names = ckpt['parameter_names']
        # Remove units from global parameter names
        global_names_clean = [name.split(" [")[0] for name in global_names]

        if list(names) != global_names_clean:
            raise ValueError(
                f"{path}: parameter ordering does not match global bounds"
            )

        all_n = 0
        all_abs = torch.zeros(9, dtype=torch.float64)
        all_sq = torch.zeros(9, dtype=torch.float64)
        excluded_metrics = None
        print(f'\n===== Model excluded: {excluded} =====')
        for chem in CHEMISTRIES:
            n, absolute, squared = evaluate(model, *data[chem], pmin, prange, global_prior_range, xmean, xstd, device, args.batch_size)
            all_n += n
            all_abs += absolute
            all_sq += squared
            mae = (absolute / n).mean().item()
            mse = (squared / n).mean().item()
            # print(f'  {chem:6s} | N={n:5d} | MAE={mae:.6f} | MSE={mse:.6f}' + ('  <-- excluded' if chem == excluded else ''))
            for j, name in enumerate(names):
                details.append({'excluded_model': excluded, 'evaluated_on': chem,
                                'parameter': name, 'n': n,
                                'mae': (absolute[j] / n).item(), 'mse': (squared[j] / n).item()})
            if chem == excluded:
                excluded_metrics = (n, absolute, squared)

        for scope, (n, absolute, squared) in (
            ('all', (all_n, all_abs, all_sq)),
            ('excluded_only', excluded_metrics),
        ):
            row = {'excluded_model': excluded, 'scope': scope, 'n': n,
                   'mae': (absolute / n).mean().item(),
                   'mse': (squared / n).mean().item()}
            summary.append(row)
            print(f'  {scope:13s} | N={n:6d} | MAE={row["mae"]:.6f} | MSE={row["mse"]:.6f}')
            for j, name in enumerate(names):
                details.append({'excluded_model': excluded, 'evaluated_on': scope,
                                'parameter': name, 'n': n,
                                'mae': (absolute[j] / n).item(), 'mse': (squared[j] / n).item()})

    write_csv(args.models_dir / 'validation_summary.csv', summary,
              ['excluded_model', 'scope', 'n', 'mae', 'mse'])
    write_csv(args.models_dir / 'validation_per_parameter.csv', details,
              ['excluded_model', 'evaluated_on', 'parameter', 'n', 'mae', 'mse'])
    print(f'\nSaved validation_summary.csv and validation_per_parameter.csv to {args.models_dir}')
    print('All normalized errors use the same widest configured bounds across all nine chemistries.')


if __name__ == '__main__':
    main()
