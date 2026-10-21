"""Extract experimental EFC0/EFC50 parameters using one combined NSF.

Assumes a combined NSF trained on 100-dimensional Full_simulations observations.
Edit COMBINED_MODEL_PATH to the location of your saved combined NPE pickle.
"""
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi import utils
from sbi.utils.user_input_checks import process_prior

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
DATA_DIR = Path('/work3/claho/battery_datasets/processed')
MODEL_DIR = PROJECT_ROOT / 'models' / 'SBI_models'
# Set this to the actual path used when you saved the trained combined NSF.
COMBINED_MODEL_PATH = MODEL_DIR / 'Combined' / 'inference_combined.pkl'
N_SAMPLES = 50000

# Chemistry: rated capacity (Ah), reference current (A), voltage lower/upper (V)
CHEMISTRY_CONFIG = {
    '25R': (2.5, -1.25, 2.5, 4.2),
    'HG2': (3.0, -1.5, 2.5, 4.2),
    'M1A': (1.1, -0.55, 2.0, 2.6),
    'MJ1': (3.5, 3.5, 2.5, 4.2),
    'PA': (74.273, 24.33, 2.75, 4.2),
    'PBC': (76.0, 25.43, 2.75, 4.2),
    'PD': (83.456, 28.0, 2.5, 4.15),
    'VTC5A': (2.5, 2.5, 2.5, 4.25),
    'VTC6': (3.0, 0.6, 2.5, 4.25),
}
SOURCE_TO_CHEMISTRY = {
    'tongji_': '25R', 'tum_': 'VTC5A', 'cmu_': 'VTC6',
    'snl_lfp_': 'M1A', 'snl_nmc_': 'HG2',
    'snl_18650_lfp_': 'M1A', 'snl_18650_nmc_': 'HG2',
    'elt_': 'MJ1',
}


def get_chemistry(cell_id):
    s = str(cell_id).lower()
    return next((v for prefix, v in SOURCE_TO_CHEMISTRY.items() if s.startswith(prefix)), None)


def load_combined_SBI():
    mins, maxs = [], []
    names, log_flags = None, None
    for chemistry in CHEMISTRY_CONFIG:
        module = importlib.import_module(f'src.simulation.parameters.parameter_{chemistry}')
        current_names = list(module.params_log)
        current_flags = list(module.params_log_flag.values())
        if names is None:
            names, log_flags = current_names, current_flags
        elif current_names != names or current_flags != log_flags:
            raise ValueError(f'Parameter names/order or log flags differ for {chemistry}')
        mins.append([pair[0] for pair in module.params_log.values()])
        maxs.append([pair[1] for pair in module.params_log.values()])

    low = torch.tensor(np.min(mins, axis=0), dtype=torch.float32)
    high = torch.tensor(np.max(maxs, axis=0), dtype=torch.float32)
    if not torch.all(high > low):
        raise ValueError('Invalid combined prior bounds')
    prior, _, _ = process_prior(utils.BoxUniform(low=low, high=high))

    if not COMBINED_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f'Combined NSF not found: {COMBINED_MODEL_PATH}\n'
            'Update COMBINED_MODEL_PATH to match the model saved by your training script.'
        )
    with COMBINED_MODEL_PATH.open('rb') as f:
        inference = pickle.load(f)  # Only unpickle your own trusted model.
    posterior = inference.build_posterior(prior=prior)
    print(f'Loaded combined NSF: {COMBINED_MODEL_PATH}')
    return {'posterior': posterior, 'param_names': names, 'log_flags': log_flags,
            'prior_min': low, 'prior_range': high - low}


def SBI_inference(model, voltage, time, chemistry, n_samples=N_SAMPLES):
    voltage = np.asarray(voltage, dtype=float)
    time = np.asarray(time, dtype=float)
    good = np.isfinite(voltage) & np.isfinite(time)
    voltage, time = voltage[good], time[good]
    order = np.argsort(time)
    voltage, time = voltage[order], time[order]
    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]
    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError('Curve has insufficient valid time points')

    capacity, current, v_min, v_max = CHEMISTRY_CONFIG[chemistry]
    time = time - time[0]
    t_interp = np.linspace(0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)
    t_typ = abs(3600 * capacity / current)
    x = np.sqrt((t_interp / t_typ)**2 + ((v_interp - v_min) / (v_max - v_min))**2)
    x = torch.tensor(x, dtype=torch.float32)

    # The model used here MUST expect the 100 features defined above.
    samples = None
    for attempt in range(5):
        try:
            with torch.no_grad():
                samples = model['posterior'].sample(
                    (n_samples,), x=x, show_progress_bars=False,
                    reject_outside_prior=False,
                ).cpu()
            break
        except AssertionError:
            print(f'NSF numerical failure, retry {attempt + 1}/5')
    if samples is None:
        raise RuntimeError('NSF sampling failed after 5 attempts')
    if not torch.isfinite(samples).all():
        raise ValueError('Posterior samples contain NaN or Inf')

    low, span = model['prior_min'], model['prior_range']
    acceptance = (((samples >= low) & (samples <= low + span)).all(dim=1)
                  .float().mean().item())
    normalized = ((samples.mean(0) - low) / span).numpy()
    physical = np.array([
        (10.0 ** samples[:, j] if is_log else samples[:, j]).mean().item()
        for j, is_log in enumerate(model['log_flags'])
    ])
    return normalized, physical, acceptance


def output_row(cell_id, chemistry, curve_type, model, normalized, physical, acceptance):
    row = {'cell_id': cell_id, 'chemistry': chemistry,
           'curve_type': curve_type, 'prior_acceptance': acceptance}
    for name, norm, phys in zip(model['param_names'], normalized, physical):
        row[f'{name}_normalized'] = norm
        row[f'{name}_physical'] = phys
    return row


def run_inference_efc0(df, model):
    rows = []
    for cell_id, cell_data in df.groupby('cell_id', sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry is None:
            print(f'Skipping {cell_id}: unrecognized chemistry')
            continue
        try:
            normalized, physical, acceptance = SBI_inference(
                model, cell_data['early_curve'].to_numpy(),
                cell_data['time_s'].to_numpy(), chemistry,
            )
        except (ValueError, RuntimeError) as exc:
            print(f'Skipping EFC0 {cell_id}: {exc}')
            continue
        rows.append(output_row(cell_id, chemistry, 'efc0', model,
                               normalized, physical, acceptance))
    return pd.DataFrame(rows)


def run_inference_efc50(df, model):
    rows = []
    for cell_id, cell_data in df.groupby('cell_id', sort=False):
        chemistry = get_chemistry(cell_id)
        if chemistry is None:
            print(f'Skipping {cell_id}: unrecognized chemistry')
            continue
        try:
            weight = float(cell_data['weight'].iloc[0])
            if not np.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError('Weight must be finite and between 0 and 1')
            lower = SBI_inference(model,
                cell_data['lower_curve'].to_numpy(),
                cell_data['lower_curve_time'].to_numpy(), chemistry)
            upper = SBI_inference(model,
                cell_data['upper_curve'].to_numpy(),
                cell_data['upper_curve_time'].to_numpy(), chemistry)
            normalized = (1 - weight) * lower[0] + weight * upper[0]
            physical = (1 - weight) * lower[1] + weight * upper[1]
            acceptance = (1 - weight) * lower[2] + weight * upper[2]
        except (ValueError, RuntimeError) as exc:
            print(f'Skipping EFC50 {cell_id}: {exc}')
            continue
        rows.append(output_row(cell_id, chemistry, 'efc50', model,
                               normalized, physical, acceptance))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    torch.manual_seed(1)
    model = load_combined_SBI()
    efc0 = pd.read_csv(DATA_DIR / 'public_charging_curve_early_all.csv')
    efc50 = pd.read_csv(DATA_DIR / 'public_charging_curve_late_all.csv')
    early = run_inference_efc0(efc0, model)
    print('EFC0 inference completed')
    late = run_inference_efc50(efc50, model)
    results = pd.concat([early, late], ignore_index=True)
    output = MODEL_DIR / 'experimental_inference_combined.csv'
    results.to_csv(output, index=False)
    print(f'Saved results to {output}')
    print(results.groupby(['chemistry', 'curve_type'])['cell_id'].nunique())
    early_ids = set(early['cell_id']) if not early.empty else set()
    late_ids = set(late['cell_id']) if not late.empty else set()
    print(f'Paired cells: {len(early_ids & late_ids)}')
    print(f'Missing EFC0: {len(late_ids - early_ids)}')
    print(f'Missing EFC50: {len(early_ids - late_ids)}')
