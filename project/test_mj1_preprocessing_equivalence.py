from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Run this script from inside the battery-predictions-using-dl repository.
# It imports the production MJ1 parameter module and battery simulator, then
# sends ONE identical raw simulated voltage/time curve through the two
# preprocessing implementations.
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
# If you copy this file into the repository root, this is already correct.
# Otherwise replace PROJECT_ROOT with the repository root, e.g.
# PROJECT_ROOT = Path('/zhome/.../battery-predictions-using-dl')
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.battery_simulator import simulator
from src.simulation.parameters import parameter_MJ1 as p

# Must match generate_simulations.py and extract_features_mlpv2.py.
Q_RATED = 3.5
CURRENT_A = 3.5
DOD = 1.0
V_MIN = 2.5
V_MAX = 4.2
N_POINTS = 100

T_TYP = abs(3600.0 * Q_RATED * DOD / CURRENT_A)
T_SIM = abs(int(3700.0 * Q_RATED * DOD / CURRENT_A))

ATOL = 1e-10
RTOL = 1e-10


def midpoint_theta_prior_coordinates():
    """Midpoint of each MJ1 prior, in the same coordinates stored in theta."""
    return np.array(
        [(lo + hi) / 2.0 for lo, hi in p.params_log.values()],
        dtype=float,
    )


def prior_theta_to_physical(theta):
    """Apply the same log10 -> physical conversion as generate_simulations.py."""
    flags = list(p.params_log_flag.values())
    physical = []
    for value, is_log10 in zip(theta, flags):
        physical.append(10.0 ** value if is_log10 else value)
    return physical


def training_preprocess_no_noise(time_s, voltage_v):
    """Exact deterministic core of generate_simulations.py::my_model()."""
    time_s = np.asarray(time_s, dtype=float)
    voltage_v = np.asarray(voltage_v, dtype=float)

    # generate_simulations.py does NOT subtract time_s[0].
    t_interp = np.linspace(0.0, time_s[-1], N_POINTS)
    f = interp1d(time_s, voltage_v, kind="slinear")
    v_interp = np.asarray(f(t_interp), dtype=float)

    t_norm = t_interp / T_TYP
    v_norm = (v_interp - V_MIN) / (V_MAX - V_MIN)
    x = np.sqrt(t_norm**2 + v_norm**2)
    return x, t_interp, v_interp


def experimental_preprocess(time_s, voltage_v):
    """Exact logic of extract_features_mlpv2.py::make_observation()."""
    voltage = np.asarray(voltage_v, dtype=float)
    time = np.asarray(time_s, dtype=float)

    mask = np.isfinite(voltage) & np.isfinite(time)
    voltage = voltage[mask]
    time = time[mask]

    order = np.argsort(time)
    voltage = voltage[order]
    time = time[order]

    time, unique_idx = np.unique(time, return_index=True)
    voltage = voltage[unique_idx]

    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("curve has insufficient valid time points")

    time = time - time[0]
    t_interp = np.linspace(0.0, time[-1], N_POINTS)
    v_interp = np.interp(t_interp, time, voltage)

    x = np.sqrt(
        (t_interp / T_TYP) ** 2
        + ((v_interp - V_MIN) / (V_MAX - V_MIN)) ** 2
    )
    return x, t_interp, v_interp


def summarize(name, a, b):
    delta = np.asarray(a) - np.asarray(b)
    print(f"\n{name}")
    print(f"  max_abs_diff = {np.max(np.abs(delta)):.12e}")
    print(f"  mean_abs_diff = {np.mean(np.abs(delta)):.12e}")
    print(f"  rmse = {np.sqrt(np.mean(delta**2)):.12e}")
    print(f"  allclose(atol={ATOL:g}, rtol={RTOL:g}) = {np.allclose(a, b, atol=ATOL, rtol=RTOL)}")


def main():
    theta = midpoint_theta_prior_coordinates()
    physical = prior_theta_to_physical(theta)
    params = p.params_setting(physical)

    result = simulator(params, CURRENT_A, T_SIM)
    if isinstance(result, str):
        raise RuntimeError(f"MJ1 simulator failed: {result}")

    raw_time = np.asarray(result["Time [s]"].entries, dtype=float)
    raw_voltage = np.asarray(result["Terminal voltage [V]"].entries, dtype=float)

    if raw_time.ndim != 1 or raw_voltage.ndim != 1:
        raise RuntimeError("Expected one-dimensional simulator time/voltage arrays")
    if len(raw_time) != len(raw_voltage):
        raise RuntimeError("Simulator time/voltage lengths differ")

    print("=== Raw simulated MJ1 curve ===")
    print(f"points:       {len(raw_time)}")
    print(f"time start:   {raw_time[0]:.12g} s")
    print(f"time end:     {raw_time[-1]:.12g} s")
    print(f"duration:     {raw_time[-1] - raw_time[0]:.12g} s")
    print(f"voltage start:{raw_voltage[0]:.12g} V")
    print(f"voltage end:  {raw_voltage[-1]:.12g} V")
    print(f"t_typ:        {T_TYP:.12g} s")

    x_train, t_train, v_train = training_preprocess_no_noise(raw_time, raw_voltage)
    x_exp, t_exp, v_exp = experimental_preprocess(raw_time, raw_voltage)

    summarize("Interpolated time grid: training vs experimental", t_train, t_exp)
    summarize("Interpolated voltage: training vs experimental", v_train, v_exp)
    summarize("Final 100-D observation x: training vs experimental", x_train, x_exp)

    # If there is a discrepancy, diagnose whether it comes only from the fact that
    # experimental preprocessing resets time to zero while simulation preprocessing
    # assumes simulator time already starts at zero.
    shifted_time = raw_time - raw_time[0]
    x_train_reset, _, _ = training_preprocess_no_noise(shifted_time, raw_voltage)
    summarize(
        "Final x after manually resetting simulator time: training-reset vs experimental",
        x_train_reset,
        x_exp,
    )

    max_x = np.max(np.abs(x_train - x_exp))
    if max_x <= ATOL:
        print("\nPASS: deterministic preprocessing paths are numerically equivalent.")
    elif np.max(np.abs(x_train_reset - x_exp)) <= ATOL:
        print("\nDIAGNOSIS: mismatch is caused by time-origin handling only.")
        print("Training assumes simulator Time[0] == 0; experimental code explicitly resets it.")
    else:
        print("\nFAIL: preprocessing paths differ beyond numerical precision.")
        print("Inspect interpolation, duplicate timestamps, voltage bounds, and time normalization.")

    # Save inputs/outputs so they can be inspected or plotted without rerunning PyBaMM.
    out = np.column_stack([t_train, v_train, x_train, t_exp, v_exp, x_exp])
    out_path = Path("mj1_preprocessing_equivalence.csv")
    np.savetxt(
        out_path,
        out,
        delimiter=",",
        header="t_train,v_train,x_train,t_exp,v_exp,x_exp",
        comments="",
    )
    print(f"Saved comparison to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
