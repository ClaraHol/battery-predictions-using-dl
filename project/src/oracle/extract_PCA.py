from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path

import importlib
import sys
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("/work3/claho/battery_datasets/processed")
MODEL_DIR = PROJECT_ROOT / "models" / "MLP_models"
OUTPUT = MODEL_DIR / "experimental_inference_mlp.csv"


SOURCE_TO_CHEMISTRY = {
    "snl_lfp_": "M1A",
    "snl_nmc_": "HG2",
    "tongji_": "25R",
    "tum_": "VTC5A",
    "cmu_": "VTC6",
    "elt_": "MJ1",
}

def get_chemistry(cell_id):
    for prefix, chemistry in SOURCE_TO_CHEMISTRY.items():
        if cell_id.startswith(prefix):
            return chemistry
    raise ValueError(f"Unknown chemistry: {cell_id}")

def encode_curve(curve, time_col, voltage_col):
    time = curve[time_col].to_numpy(dtype=float)
    voltage = curve[voltage_col].to_numpy(dtype=float)

    mask = np.isfinite(time) & np.isfinite(voltage)
    time, voltage = time[mask], voltage[mask]

    order = np.argsort(time)
    time, voltage = time[order], voltage[order]

    time, idx = np.unique(time, return_index=True)
    voltage = voltage[idx]

    if len(time) < 2:
        raise ValueError("Insufficient time points")

    time = time - time[0]

    # Interpolate voltage to 100 equally spaced points
    t_interp = np.linspace(0, time[-1], 100)
    v_interp = np.interp(t_interp, time, voltage)

    return v_interp.astype(np.float32)


if __name__ == "__main__":
    efc0 = pd.read_csv(DATA_DIR / "public_charging_curve_early_all.csv")
    efc50 = pd.read_csv(DATA_DIR / "public_charging_curve_late_all.csv")

    datasets = [
        ("efc0", efc0, "time_s", "early_curve"),
        ("efc50", efc50, "lower_curve_time", "lower_curve"),
    ]

    encoded = []
    metadata = []

    for efc, df, time_col, voltage_col in datasets:
        for cell_id, curve in df.groupby("cell_id"):
            chemistry = get_chemistry(cell_id)

            try:
                x = encode_curve(curve, time_col, voltage_col)
            except ValueError as e:
                print(f"Skipping {cell_id}: {e}")
                continue

            encoded.append(x)
            metadata.append({
                "cell_id": cell_id,
                "chemistry": chemistry,
                "curve_type": efc,
            })

    X = np.stack(encoded)
    metadata = pd.DataFrame(metadata)

    print("Encoded shape:", X.shape)

    # PCA on the combined EFC0 and EFC50 representations
    pca = PCA(n_components=11)
    Z = pca.fit_transform(X)

    print("Explained variance:", pca.explained_variance_ratio_)
    print("Total variance:", pca.explained_variance_ratio_.sum())

    for i in range(Z.shape[1]):
        metadata[f"PC{i+1}"] = Z[:, i]

    OUTPUT = PROJECT_ROOT / "models" / "inference_results" / "experimental_inference_PCA.csv"
    metadata.to_csv(OUTPUT, index=False)



    # Plot PCA, distinguishing EFC0 and EFC50
    plt.figure(figsize=(10, 7))

    for (chemistry, efc), group in metadata.groupby(["chemistry", "curve_type"]):
        plt.scatter(
            group["PC1"],
            group["PC2"],
            label=f"{chemistry} {efc}",
            marker="o" if efc == "efc0" else "x",
            alpha=0.7,
            s=25,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA of experimental charging curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig( PROJECT_ROOT / "models" / "experimental_pca.png", dpi=200)
    plt.show()