from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

results_dir = PROJECT_ROOT / "models" / "Validation_results"
plot_dir = PROJECT_ROOT / "reports" / "Validation_plots"

plot_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Find every available validation result
# ============================================================

result_files = sorted(results_dir.glob("validation_*.pt"))

print(f"Found {len(result_files)} validation files:")

for file in result_files:
    print(f"  {file.name}")


# ============================================================
# Plot each chemistry
# ============================================================

all_nsf_errors = []
all_mlp_errors = []
all_param_names = None

for result_file in result_files:

    results = torch.load(
        result_file,
        map_location="cpu",
        weights_only=False,
    )

    chemistry = results["chemistry"]
    param_names = results["parameter_names"]

    # Shape:
    # (num_validation_samples, 11)
    nsf_errors = np.sqrt(results["nsf_errors"].cpu().numpy())
    mlp_errors = np.sqrt(results["mlp_errors"].cpu().numpy())

    all_nsf_errors.append(nsf_errors)
    all_mlp_errors.append(mlp_errors)
    all_param_names = param_names


    n_params = len(param_names)

    fig, ax = plt.subplots(figsize=(16, 7))

    # Central position of each parameter
    positions = np.arange(n_params)

    offset = 0.18
    width = 0.30

    # --------------------------------------------------------
    # NSF
    # --------------------------------------------------------

    bp_nsf = ax.boxplot(
        [nsf_errors[:, i] for i in range(n_params)],
        positions=positions - offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    bp_mlp = ax.boxplot(
        [mlp_errors[:, i] for i in range(n_params)],
        positions=positions + offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    # Colors
    for box in bp_nsf["boxes"]:
        box.set_facecolor("tab:blue")
        box.set_alpha(0.7)

    for box in bp_mlp["boxes"]:
        box.set_facecolor("tab:orange")
        box.set_alpha(0.7)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xticks(positions)
    ax.set_xticklabels(
        param_names,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Normalized error")
    ax.set_title(f"{chemistry} — Validation errors")

    ax.grid(
        axis="y",
        alpha=0.3,
    )
    # ax.set_yscale("log")


    # Dummy artists for legend
    ax.plot([], [], color="tab:blue", linewidth=8, label="NSF")
    ax.plot([], [], color="tab:orange", linewidth=8, label="MLP")

    ax.legend()

    plt.tight_layout()

    output_path = plot_dir / f"validation_boxplot_{chemistry}.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


if all_nsf_errors:
    combined_nsf_errors = np.concatenate(all_nsf_errors, axis=0)
    combined_mlp_errors = np.concatenate(all_mlp_errors, axis=0)

    # Box plot across all validation errors and chemistries.
    n_params = len(all_param_names)
    positions = np.arange(n_params)
    offset = 0.18
    width = 0.30

    fig, ax = plt.subplots(figsize=(16, 7))

    bp_nsf = ax.boxplot(
        [combined_nsf_errors[:, i] for i in range(n_params)],
        positions=positions - offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    bp_mlp = ax.boxplot(
        [combined_mlp_errors[:, i] for i in range(n_params)],
        positions=positions + offset,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    for box in bp_nsf["boxes"]:
        box.set_facecolor("tab:blue")
        box.set_alpha(0.7)

    for box in bp_mlp["boxes"]:
        box.set_facecolor("tab:orange")
        box.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_param_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized error")
    ax.set_title("All chemistries — Validation errors")
    ax.grid(axis="y", alpha=0.3)
    ax.plot([], [], color="tab:blue", linewidth=8, label="NSF")
    ax.plot([], [], color="tab:orange", linewidth=8, label="MLP")
    ax.legend()

    plt.tight_layout()
    aggregate_output_path = plot_dir / "validation_boxplot_all_chemistries.png"
    plt.savefig(aggregate_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {aggregate_output_path}")

    mean_nsf_errors = np.mean(combined_nsf_errors, axis=0)
    mean_mlp_errors = np.mean(combined_mlp_errors, axis=0)

    print("\nMean errors across all chemistries:")
    for parameter_name, nsf_mean, mlp_mean in zip(
        all_param_names,
        mean_nsf_errors,
        mean_mlp_errors,
    ):
        print(
            f"  {parameter_name}: NSF={nsf_mean:.6g}, "
            f"MLP={mlp_mean:.6g}"
        )

    