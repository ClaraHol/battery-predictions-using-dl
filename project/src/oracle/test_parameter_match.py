import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("/work3/claho/battery_datasets/processed")
MODEL_DIR = PROJECT_ROOT / "models" / "MLP_models"
OUTPUT = MODEL_DIR / "experimental_inference_mlp.csv"
DEVICE = torch.device("cpu")

CHEMISTRY_CONFIG = {
    "25R": (2.5, -1.25, 2.5, 4.2),
    "HG2": (3.0, -1.5, 2.5, 4.2),
    "M1A": (1.1, -0.55, 2.0, 2.6),
    "MJ1": (3.5, 3.5, 2.5, 4.2),
    "PA": (74.273, 24.33, 2.75, 4.2),
    "PBC": (76.0, 25.43, 2.75, 4.2),
    "PD": (83.456, 28.0, 2.5, 4.15),
    "VTC5A": (2.5, 2.5, 2.5, 4.25),
    "VTC6": (3.0, 0.6, 2.5, 4.25),
}
def check_parameter_consistency():
    print("\n===== MLP PARAMETER CONSISTENCY =====")

    for chemistry, (capacity, current, v_min, v_max) in CHEMISTRY_CONFIG.items():
        print(f"\n{'=' * 50}")
        print(f"Chemistry: {chemistry}")

        module = importlib.import_module(
            f"src.simulation.parameters.parameter_{chemistry}"
        )

        checkpoint_path = (
            MODEL_DIR / chemistry / f"mlp_{chemistry}.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        # Compare parameter ordering
        names_file = list(module.params_log.keys())
        names_checkpoint = checkpoint["parameter_names"]

        if names_file == names_checkpoint:
            print("Parameter ordering: PASS")
        else:
            print("Parameter ordering: FAIL")
            print("File:      ", names_file)
            print("Checkpoint:", names_checkpoint)

        # Compare parameter bounds
        file_min = np.array([
            bounds[0] for bounds in module.params_log.values()
        ])
        file_max = np.array([
            bounds[1] for bounds in module.params_log.values()
        ])

        checkpoint_min = checkpoint["prior_min"].numpy()
        checkpoint_max = checkpoint["prior_max"].numpy()

        print("\nParameter bounds:")

        for i, name in enumerate(names_file):
            same = (
                np.isclose(file_min[i], checkpoint_min[i])
                and np.isclose(file_max[i], checkpoint_max[i])
            )

            print(
                f"{'PASS' if same else 'FAIL'} "
                f"{name}: "
                f"file=[{file_min[i]:.4g}, {file_max[i]:.4g}], "
                f"checkpoint=[{checkpoint_min[i]:.4g}, "
                f"{checkpoint_max[i]:.4g}]"
            )

        # Observation preprocessing configuration
        t_typ = abs(3600 * capacity / current)

        print("\nObservation normalization:")
        print(f"Capacity:          {capacity} Ah")
        print(f"Reference current: {current} A")
        print(f"Voltage minimum:   {v_min} V")
        print(f"Voltage maximum:   {v_max} V")
        print(f"Characteristic t:  {t_typ:.2f} s")




SOURCE_TO_CHEMISTRY = {
    "snl_18650_lfp_": "M1A",
    "snl_18650_nmc_": "HG2",
    "tongji_": "25R",
    "tum_": "VTC5A",
    "cmu_": "VTC6",
    "elt_": "MJ1",
}


def get_chemistry(cell_id):
    cell_id = str(cell_id).lower()
    for prefix, chemistry in SOURCE_TO_CHEMISTRY.items():
        if cell_id.startswith(prefix):
            return chemistry
    return None


def curve_duration(time):
    time = np.asarray(time, dtype=float)
    time = time[np.isfinite(time)]

    if len(time) < 2:
        return np.nan

    return time.max() - time.min()


def analyze(df, curve_type):
    rows = []

    for cell_id, cell in df.groupby("cell_id"):
        chemistry = get_chemistry(cell_id)

        if chemistry not in CHEMISTRY_CONFIG:
            continue

        capacity, current, v_min, v_max = CHEMISTRY_CONFIG[chemistry]
        t_typ = abs(3600 * capacity / current)

        if curve_type == "efc0":
            durations = {
                "early": curve_duration(cell["time_s"])
            }
        else:
            durations = {
                "lower": curve_duration(cell["lower_curve_time"]),
                "upper": curve_duration(cell["upper_curve_time"]),
            }

        for segment, duration in durations.items():
            rows.append({
                "cell_id": cell_id,
                "chemistry": chemistry,
                "curve_type": curve_type,
                "segment": segment,
                "duration_s": duration,
                "t_typ_s": t_typ,
                "time_ratio": duration / t_typ,
            })

    return rows

if __name__ == "__main__":
    # check_parameter_consistency()


    efc0 = pd.read_csv(DATA_DIR / "public_charging_curve_early_all.csv")
    efc50 = pd.read_csv(DATA_DIR / "public_charging_curve_late_all.csv")

    results = pd.DataFrame(
        analyze(efc0, "efc0") +
        analyze(efc50, "efc50")
    )

    print("\n===== Normalized duration by chemistry =====")
    print(
        results.groupby(["chemistry", "curve_type", "segment"])["time_ratio"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n===== Fraction of curves with t_end > t_typ =====")
    print(
        results.assign(exceeds=results["time_ratio"] > 1)
        .groupby(["chemistry", "curve_type", "segment"])["exceeds"]
        .mean()
        .mul(100)
        .round(1)
        .to_string()
    )

    print("\n===== 20 largest time ratios =====")
    print(
        results.nlargest(20, "time_ratio")
        .to_string(index=False)
    )

    # cell = efc50[
    # efc50["cell_id"] == "tongji_CY25-05_1-#2"
    # ]

    # print(cell[
    #     ["lower_curve_time", "upper_curve_time"]
    # ].describe())

    # print(
    #     np.array_equal(
    #         cell["lower_curve_time"].to_numpy(),
    #         cell["upper_curve_time"].to_numpy()
    #     )
    # )