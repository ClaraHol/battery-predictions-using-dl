#!/usr/bin/env python3
"""Leave-one-chemistry-out training for varied battery simulations.

Run from anywhere: python project/src/models/train_varied_leave_one_out.py --all
Or: python project/src/models/train_varied_leave_one_out.py --exclude HG2
"""
import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CHEMISTRIES = ('25R', 'HG2', 'M1A', 'MJ1', 'PA', 'PBC', 'PD', 'VTC5A', 'VTC6')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_NAMES = ('D_s_n', 'D_s_p', 'k_n', 'k_p', 'D_e', 'sigma_e', 'R_f', 'eps_s_n', 'eps_s_p')


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

    def forward(self, x):
        h = self.bn0(x) if self.use_batch_norm else x
        h = self.linear0(self.activation(h))
        h = self.bn1(h) if self.use_batch_norm else h
        h = self.linear1(self.dropout(self.activation(h)))
        return x + h


class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features=108, out_features=9):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, 256)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(9)])
        self.final_layer = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.initial_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)


def load_data(directory, chemistry):
    theta, x = torch.load(directory / f'dataset_{chemistry}.pt', map_location='cpu', weights_only=True)
    theta, x = theta.float(), x.float()
    if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != 9 or x.shape[1] != 108 or len(theta) != len(x):
        raise ValueError(f'{chemistry}: expected theta [N,9], x [N,108]; got {theta.shape}, {x.shape}')
    valid = torch.isfinite(theta).all(1) & torch.isfinite(x).all(1)
    print(f'{chemistry}: {valid.sum().item()}/{len(valid)} valid', flush=True)
    if not valid.any():
        raise ValueError(f'No valid rows for {chemistry}')
    return theta[valid], x[valid]


def save_atomic(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def evaluate(model, x, theta, loss_fn, device, batch_size):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = theta[start:start + batch_size].to(device)
            total += loss_fn(model(xb), yb).item() * len(xb)
            count += len(xb)
    return total / count


def run_fold(excluded, args):
    print(f'\n===== Excluding {excluded} =====', flush=True)
    out = args.output_dir / f'excluded_{excluded}'
    out.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out / 'resume.pt', out / 'model.pt'
    if not args.resume and (latest_path.exists() or best_path.exists()):
        raise FileExistsError(f'{out} already has checkpoints; use --resume or move the folder')

    included = [c for c in CHEMISTRIES if c != excluded]
    gen = torch.Generator().manual_seed(args.seed)
    train_t, train_x, val_t, val_x = [], [], [], []
    for chemistry in included:
        theta, x = load_data(args.dataset_dir, chemistry)
        perm = torch.randperm(len(theta), generator=gen)
        n_val = max(1, int(len(theta) * args.val_fraction))
        vi, ti = perm[:n_val], perm[n_val:]
        train_t.append(theta[ti]); train_x.append(x[ti])
        val_t.append(theta[vi]); val_x.append(x[vi])

    train_t, train_x = torch.cat(train_t), torch.cat(train_x)
    val_t, val_x = torch.cat(val_t), torch.cat(val_x)

    # Estimate normalization only on training rows of the 8 included chemistries.
    # Since simulation targets are in log-space where flagged, preserve those coordinates.
    prior_min = train_t.amin(0)
    prior_max = train_t.amax(0)
    prior_range = prior_max - prior_min
    if (prior_range <= 0).any():
        raise ValueError('Degenerate target range')
    x_mean = train_x.mean(0)
    x_std = train_x.std(0).clamp_min(1e-6)
    train_y = (train_t - prior_min) / prior_range
    val_y = (val_t - prior_min) / prior_range
    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std

    device = torch.device(args.device)
    model = ResidualNetRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-7
    )
    loss_fn = nn.MSELoss()
    start_epoch, best_val, no_improvement = 0, float('inf'), 0
    train_losses, val_losses = [], []

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
        ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
        if ckpt['included_chemistries'] != included or ckpt['seed'] != args.seed or ckpt['val_fraction'] != args.val_fraction:
            raise ValueError('Resume settings differ from original fold')
        for key, value in [('prior_min', prior_min), ('prior_max', prior_max), ('x_mean', x_mean), ('x_std', x_std)]:
            if not torch.allclose(ckpt[key].cpu(), value, rtol=1e-5, atol=1e-6):
                raise ValueError(f'Resumed {key} differs: data or preprocessing changed')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Optimizer tensors saved on CPU need to move to the training device.
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val, no_improvement = ckpt['best_val_loss'], ckpt['epochs_since_improvement']
        train_losses, val_losses = ckpt['train_losses'], ckpt['val_losses']
        torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        np.random.set_state(ckpt['numpy_rng_state'])
        random.setstate(ckpt['python_rng_state'])
        if device.type == 'cuda' and 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        print(f'Resuming at epoch {start_epoch + 1}', flush=True)

    loader_gen = torch.Generator().manual_seed(args.seed + 1000)
    if args.resume and 'loader_rng_state' in ckpt:
        loader_gen.set_state(ckpt['loader_rng_state'])
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, generator=loader_gen, num_workers=0)
    print(f'Train {len(train_x)} | Val {len(val_x)} | Device {device}', flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        total, n = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        train_loss = total / n
        val_loss = evaluate(model, val_x, val_y, loss_fn, device, args.batch_size)
        scheduler.step(val_loss)
        train_losses.append(train_loss); val_losses.append(val_loss)
        improved = val_loss < best_val
        if improved:
            best_val, no_improvement = val_loss, 0
            save_atomic({
                'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'prior_min': prior_min, 'prior_max': prior_max,
                'parameter_names': list(PARAMETER_NAMES),
                'x_mean': x_mean, 'x_std': x_std,
                'excluded_chemistry': excluded, 'included_chemistries': included,
                'epoch': epoch + 1, 'best_val_loss': best_val,
                'train_losses': train_losses.copy(), 'val_losses': val_losses.copy(),
            }, best_path)
        else:
            no_improvement += 1

        resume = {
            'epoch': epoch + 1,
            'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val,
            'epochs_since_improvement': no_improvement,
            'train_losses': train_losses.copy(), 'val_losses': val_losses.copy(),
            'prior_min': prior_min, 'prior_max': prior_max,
            'x_mean': x_mean, 'x_std': x_std,
            'included_chemistries': included, 'excluded_chemistry': excluded,
            'seed': args.seed, 'val_fraction': args.val_fraction,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'loader_rng_state': loader_gen.get_state(),
        }
        if device.type == 'cuda':
            resume['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        save_atomic(resume, latest_path)
        with (out / 'losses.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('epoch', 'train_mse', 'val_mse'))
            writer.writerows((i + 1, tr, va) for i, (tr, va) in enumerate(zip(train_losses, val_losses)))
        print(f'{excluded:5s} Epoch {epoch+1:4d} | Train {train_loss:.6f} | Val {val_loss:.6f} | Best {best_val:.6f} | LR {optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        if no_improvement >= args.early_stopping_patience:
            print('Early stopping', flush=True)
            break
    print(f'{excluded}: best validation MSE {best_val:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exclude', choices=CHEMISTRIES)
    group.add_argument('--all', action='store_true')
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'models' / 'Varied_simulations')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'models' / 'MLP_models' / 'LeaveOneOut')
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=21.0)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for excluded in CHEMISTRIES if args.all else (args.exclude,):
        run_fold(excluded, args)


if __name__ == '__main__':
    main()
