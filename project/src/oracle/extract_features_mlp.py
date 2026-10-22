"""Extract experimental EFC0/EFC50 features using chemistry-specific MLP regressors.

Run from any working directory:
    python project/src/oracle/extract_features_mlp.py

Uses the same 100-dimensional observation preprocessing as the original
Full_simulations MLP training data. Does NOT use the combined MLP.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from derived_features import add_derived_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("/work3/claho/battery_datasets/processed")
MODEL_DIR = PROJECT_ROOT / "models" / "MLP_models"
OUTPUT = PROJECT_ROOT / "models" / "inference_results" / "experimental_inference_mlp.csv"
DEVICE = torch.device("cpu")

# Chemistry: rated capacity (Ah), reference current (A), voltage limits (V)
CHEMISTRY_CONFIG = {
    "25R": (2.5, -1.25, 2.5, 4.2),
    "HG2": (3.0, -1.5, 2.5, 4.2),
    "M1A": (1.1, -0.55, 2.0, 3.6),
    "MJ1": (3.5, 3.5, 2.5, 4.2),
    "PA": (74.273, 24.33, 2.75, 4.2),
    "PBC": (76.0, 25.43, 2.75, 4.2),
    "PD": (83.456, 28.0, 2.5, 4.15),
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
}

SOURCE_TO_CHEMISTRY = {
    "snl_lfp_": "M1A",
    "snl_nmc_": "HG2",
    "tongji_": "25R",
    "tum_": "VTC5A",
    "cmu_": "VTC6",
    "elt_": "MJ1",
}


def get_chemistry(cell_id):
    s = str(cell_id).lower()
    return next(
        (chemistry for prefix, chemistry in SOURCE_TO_CHEMISTRY.items()
         if s.startswith(prefix)),
        None,
    )


# Exactly the same residual network as in the supplied train_mlp.py.
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
    def __init__(self, in_features, out_features, hidden_features,
                 num_blocks, dropout_probability, use_batch_norm):
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


def load_all_mlps():
    models = {}
    for chemistry in CHEMISTRY_CONFIG:
        path = MODEL_DIR / chemistry / f"mlp_{chemistry}.pt"
        if not path.is_file():
            print(f"Missing MLP for {chemistry}: {path}")
            continue

        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )
        names = list(module.params_log.keys())
        flags = list(module.params_log_flag.values())
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)

        # Prefer saved training bounds; otherwise reconstruct from parameter module.
        module_low = torch.tensor(
            [limits[0] for limits in module.params_log.values()],
            dtype=torch.float32,
        )
        module_high = torch.tensor(
            [limits[1] for limits in module.params_log.values()],
            dtype=torch.float32,
        )
        low = torch.as_tensor(checkpoint.get("prior_min", module_low),
                              dtype=torch.float32)
        high = torch.as_tensor(checkpoint.get("prior_max", module_high),
                               dtype=torch.float32)
        saved_names = checkpoint.get("parameter_names", names)
        if list(saved_names) != names:
            raise ValueError(f"Parameter ordering mismatch for {chemistry}")
        if len(names) != 11 or not torch.all(high > low):
            raise ValueError(f"Invalid parameters/prior for {chemistry}")

        # Training architecture: x=100, theta=11, 9 residual blocks, width 256.
        net = ResidualNetRegressor(
            in_features=100,
            out_features=len(names),
            hidden_features=256,
            num_blocks=9,
            dropout_probability=0.2,
            use_batch_norm=True,
        ).to(DEVICE)
        net.load_state_dict(state, strict=True)
        net.eval()

        models[chemistry] = {
            "net": net,
            "param_names": names,
            "log_flags": flags,
            "prior_min": low,
            "prior_range": high - low,
        }
        print(f"Loaded MLP: {chemistry}")
    return models


def mlp_inference(model, voltage, time, chemistry):
    voltage = np.asarray(voltage, dtype=float)
    time = np.asarray(time, dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(time)
    voltage, time = voltage[mask], time[mask]

    order = np.argsort(time)
    voltage, time = voltage[order], time[order]
    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]
    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("Curve has insufficient valid time points")

    capacity, current, v_min, v_max = CHEMISTRY_CONFIG[chemistry]
    time = time - time[0]
    t_interp = np.linspace(0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)
    t_typ = abs(3600 * capacity / current)
    x = np.sqrt(
        (t_interp / t_typ)**2 +
        ((v_interp - v_min) / (v_max - v_min))**2
    )
    x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        normalized = model["net"](x).squeeze(0).cpu()

    if not torch.isfinite(normalized).all():
        raise ValueError("MLP output contains NaN or Inf")

    # Model outputs theta normalized to each chemistry's prior bounds.
    theta = normalized * model["prior_range"] + model["prior_min"]
    physical = torch.stack([
        10.0 ** theta[j] if flag else theta[j]
        for j, flag in enumerate(model["log_flags"])
    ])
    return normalized.numpy(), physical.numpy()


def result_row(cell_id, chemistry, curve_type, model, normalized, physical):
    row = {"cell_id": cell_id, "chemistry": chemistry, "curve_type": curve_type}
    for name, norm, phys in zip(model["param_names"], normalized, physical):
        row[f"{name}_normalized"] = float(norm)
        row[f"{name}_physical"] = float(phys)
    return row


def run_inference_efc0(df, models):
    rows = []
    for cell_id, cell_data in df.groupby("cell_id", sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry not in models:
            print(f"Skipping EFC0 {cell_id}: no matching MLP")
            continue
        try:
            normalized, physical = mlp_inference(
                models[chemistry],
                cell_data["early_curve"].to_numpy(),
                cell_data["time_s"].to_numpy(),
                chemistry,
            )
        except ValueError as exc:
            print(f"Skipping EFC0 {cell_id}: {exc}")
            continue
        rows.append(result_row(
            cell_id, chemistry, "efc0", models[chemistry],
            normalized, physical,
        ))
    return pd.DataFrame(rows)


def run_inference_efc50(df, models):
    rows = []
    for cell_id, cell_data in df.groupby("cell_id", sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry not in models:
            print(f"Skipping EFC50 {cell_id}: no matching MLP")
            continue
        try:
            weights = cell_data["weight"].dropna().unique()
            if len(weights) != 1:
                raise ValueError("Expected one consistent interpolation weight")
            weight = float(weights[0])
            if not 0 <= weight <= 1:
                raise ValueError("Weight must be between 0 and 1")

            lower_norm, lower_phys = mlp_inference(
                models[chemistry],
                cell_data["lower_curve"].to_numpy(),
                cell_data["lower_curve_time"].to_numpy(),
                chemistry,
            )
            upper_norm, upper_phys = mlp_inference(
                models[chemistry],
                cell_data["upper_curve"].to_numpy(),
                cell_data["upper_curve_time"].to_numpy(),
                chemistry,
            )
            if weight == 0:
                normalized, physical = lower_norm, lower_phys
            elif weight == 1:
                normalized, physical = upper_norm, upper_phys
            else:
                normalized = (1 - weight) * lower_norm + weight * upper_norm
                physical = (1 - weight) * lower_phys + weight * upper_phys
        except ValueError as exc:
            print(f"Skipping EFC50 {cell_id}: {exc}")
            continue

        rows.append(result_row(
            cell_id, chemistry, "efc50", models[chemistry],
            normalized, physical,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = load_all_mlps()
    if not models:
        raise RuntimeError("No chemistry-specific MLP checkpoints found")

    efc0 = pd.read_csv(DATA_DIR / "public_charging_curve_early_all.csv")
    efc50 = pd.read_csv(DATA_DIR / "public_charging_curve_late_all.csv")

    early = run_inference_efc0(efc0, models)
    print("EFC0 inference completed")
    late = run_inference_efc50(efc50, models)


    results = pd.concat([early, late], ignore_index=True)
    results = add_derived_features(results)


    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(f"Saved {len(results)} rows to {OUTPUT}")


    if not results.empty:
        print(results.groupby(["chemistry", "curve_type"])["cell_id"].nunique())
    early_ids = set(early["cell_id"]) if not early.empty else set()
    late_ids = set(late["cell_id"]) if not late.empty else set()
    print(f"Unique cells: {len(early_ids | late_ids)}")
    print(f"Paired EFC0/EFC50: {len(early_ids & late_ids)}")
    print(f"Missing EFC0: {len(late_ids - early_ids)}")
    print(f"Missing EFC50: {len(early_ids - late_ids)}")
