import argparse
import importlib
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

CHEMISTRIES = ['25R', 'HG2', 'M1A', 'MJ1', 'PA', 'PBC', 'PD', 'VTC5A', 'VTC6']
NUM_DIM = 9
X_DIM = 108
HIDDEN = 256
BLOCKS = 9
DROPOUT = 0.2
BATCH_SIZE = 2048
LR = 1e-4
CLIP = 21.0
VAL_FRACTION = 0.2


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
        h = self.bn0(inputs) if self.use_batch_norm else inputs
        h = self.activation(h)
        h = self.linear0(h)
        h = self.bn1(h) if self.use_batch_norm else h
        h = self.activation(h)
        h = self.dropout(h)
        return inputs + self.linear1(h)


class ResidualNetRegressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_blocks,
                 dropout_probability, use_batch_norm):
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


def atomic_save(obj, path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def plot_history(train_losses, val_losses, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (global-prior-normalized targets)')
    ax.set_yscale('log')
    ax.set_title('Combined varied MLP training')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'mlp_training_summary_combined.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=1000,
                        help='Total epoch limit, including completed epochs on resume')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    all_min, all_max = [], []
    parameter_names = None
    for chemistry in CHEMISTRIES:
        module = importlib.import_module(f'src.simulation.parameters.parameter_{chemistry}')
        params = list(module.params_log.items())[:NUM_DIM]
        names = [name for name, _ in params]
        if parameter_names is None:
            parameter_names = names
        elif names != parameter_names:
            raise ValueError(f'Parameter ordering differs for {chemistry}: {names}')
        all_min.append(torch.tensor([bounds[0] for _, bounds in params], dtype=torch.float32))
        all_max.append(torch.tensor([bounds[1] for _, bounds in params], dtype=torch.float32))

    prior_min = torch.stack(all_min).min(dim=0).values
    prior_max = torch.stack(all_max).max(dim=0).values
    prior_range = prior_max - prior_min
    if not torch.all(prior_range > 0):
        raise ValueError('Invalid global parameter bounds')

    dataset_dir = PROJECT_ROOT / 'models' / 'Varied_simulations'
    theta_list, x_list = [], []
    for chemistry in CHEMISTRIES:
        theta, x = torch.load(dataset_dir / f'dataset_{chemistry}.pt',
                              map_location='cpu', weights_only=True)
        theta, x = theta.float(), x.float()
        if theta.ndim != 2 or x.ndim != 2 or theta.shape[1] != NUM_DIM or x.shape[1] != X_DIM:
            raise ValueError(f'{chemistry}: expected theta [N,9] and x [N,108], '
                             f'got {tuple(theta.shape)}, {tuple(x.shape)}')
        valid = torch.isfinite(theta).all(dim=1) & torch.isfinite(x).all(dim=1)
        print(f'{chemistry}: {int(valid.sum())}/{len(valid)} valid', flush=True)
        theta_list.append(theta[valid])
        x_list.append(x[valid])

    theta = torch.cat(theta_list)
    x = torch.cat(x_list)
    theta_normalized = (theta - prior_min) / prior_range

    n = len(theta)
    n_val = max(1, int(n * VAL_FRACTION))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_mean = x[train_idx].mean(dim=0)
    x_std = x[train_idx].std(dim=0).clamp(min=1e-6)
    x_train = (x[train_idx] - x_mean) / x_std
    x_val = (x[val_idx] - x_mean) / x_std
    theta_train = theta_normalized[train_idx]
    theta_val = theta_normalized[val_idx]

    # Keep the full dataset on CPU and transfer individual batches to the accelerator.
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, theta_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, theta_val),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )

    model = ResidualNetRegressor(X_DIM, NUM_DIM, HIDDEN, BLOCKS, DROPOUT, True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-8
    )
    loss_fn = nn.MSELoss()

    output_dir = PROJECT_ROOT / 'models' / 'MLP_models' / 'Combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / 'mlp_varied_combined.pt'
    resume_path = output_dir / 'mlp_varied_resume.pt'

    start_epoch = 0
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improvement = 0
    train_losses, val_losses = [], []

    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        for name, current in [('prior_min', prior_min), ('prior_max', prior_max),
                              ('x_mean', x_mean), ('x_std', x_std)]:
            if not torch.allclose(current, ckpt[name], atol=1e-7, rtol=1e-6):
                raise ValueError(f'{name} differs from resume checkpoint; check data and seed')
        if ckpt['parameter_names'] != parameter_names or ckpt['seed'] != args.seed:
            raise ValueError('Parameter order or random seed differs from checkpoint')
        if ckpt['train_count'] != len(train_idx) or ckpt['val_count'] != len(val_idx):
            raise ValueError('Training/validation dataset sizes differ from checkpoint')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_state = ckpt['best_state']
        epochs_since_improvement = ckpt['epochs_since_improvement']
        train_losses = ckpt['train_losses']
        val_losses = ckpt['val_losses']
        torch.set_rng_state(ckpt['torch_rng_state'].cpu())
        np.random.set_state(ckpt['numpy_rng_state'])
        random.setstate(ckpt['python_rng_state'])
        if device.type == 'cuda' and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        print(f'Resuming at epoch {start_epoch + 1}, best val {best_val_loss:.6f}', flush=True)
    elif resume_path.exists():
        raise FileExistsError(
            f'{resume_path} already exists. Use --resume, or move/delete old checkpoints '
            'before starting a new experiment.'
        )

    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        train_sum = 0.0
        for xb, tb in train_loader:
            xb, tb = xb.to(device), tb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_sum += loss.item() * len(xb)
        train_loss = train_sum / len(train_idx)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, tb in val_loader:
                xb, tb = xb.to(device), tb.to(device)
                val_sum += loss_fn(model(xb), tb).item() * len(xb)
        val_loss = val_sum / len(val_idx)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = cpu_state(model)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f'Epoch {epoch + 1:4d} | Train: {train_loss:.6f} | '
              f'Val: {val_loss:.6f} | Best: {best_val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}', flush=True)

        metadata = {
            'prior_min': prior_min, 'prior_max': prior_max,
            'parameter_names': parameter_names,
            'x_mean': x_mean, 'x_std': x_std,
        }
        if improved:
            atomic_save({'model_state_dict': best_state, **metadata}, best_path)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': cpu_state(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_state': best_state,
            'best_val_loss': best_val_loss,
            'epochs_since_improvement': epochs_since_improvement,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'seed': args.seed,
            'train_count': len(train_idx),
            'val_count': len(val_idx),
            **metadata,
        }
        atomic_save(checkpoint, resume_path)
        plot_history(train_losses, val_losses, output_dir)

        if epochs_since_improvement >= args.patience:
            print(f'Early stopping after epoch {epoch + 1}', flush=True)
            break

    print(f'Time used: {time.time() - start_time:.1f}s', flush=True)
    print(f'Best validation MSE: {best_val_loss:.6f}', flush=True)
    print(f'Best model: {best_path}', flush=True)
    print(f'Resume checkpoint: {resume_path}', flush=True)


if __name__ == '__main__':
    main()
