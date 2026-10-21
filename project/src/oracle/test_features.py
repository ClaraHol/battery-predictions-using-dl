import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# csv_path = PROJECT_ROOT / "models" / "MLP_models" / "experimental_inference_mlp.csv"
csv_path = PROJECT_ROOT / "models" / "MLP_models" / "Combined" / "experimental_inference_mlp_combined.csv"
# csv_path = PROJECT_ROOT / "models" / "SBI_models" / "experimental_inference.csv"
df = pd.read_csv(csv_path)

# Identify normalized parameter columns
features = [col for col in df.columns if col.endswith("_normalized")]
features_physical = [col for col in df.columns if col.endswith("_physical")]

# Number of unique cells
print("\n===== Cell counts =====")
print(f"Total unique cells: {df['cell_id'].nunique()}")

print("\nUnique cells per chemistry:")
print(df.groupby("chemistry")["cell_id"].nunique().to_string())

print("\nUnique cells per chemistry and EFC:")
print(
    df.groupby(["chemistry", "curve_type"])["cell_id"]
    .nunique()
    .to_string()
)

# Average normalized parameter values
print("\n===== Average normalized parameters (all cells) =====")
print(df[features].mean().round(4).to_string())

print("\n===== Average normalized parameters by chemistry =====")
print(
    df.groupby("chemistry")[features]
    .mean()
    .round(4)
    .T
    .to_string()
)

print("\n===== Average normalized parameters by chemistry and EFC =====")
print(
    df.groupby(["chemistry", "curve_type"])[features]
    .mean()
    .round(4)
    .T
    .to_string()
)


# ============================================================
# Check that every cell has both EFC0 and EFC50
# ============================================================

early_ids = set(df.loc[df["curve_type"] == "efc0", "cell_id"])
late_ids = set(df.loc[df["curve_type"] == "efc50", "cell_id"])



# print(list(df))
only_early = early_ids - late_ids
only_late = late_ids - early_ids
matched = early_ids & late_ids

print("\n===== EFC0 / EFC50 consistency =====")
print(f"Cells with both EFC0 and EFC50: {len(matched)}")
print(f"Cells only in EFC0: {len(only_early)}")
print(f"Cells only in EFC50: {len(only_late)}")

if not only_early and not only_late:
    print("PASS: All cells have both EFC0 and EFC50.")
else:
    print("FAIL: Some cells are missing an EFC0 or EFC50 inference.")

    for label, ids in [
        ("Missing EFC50", only_early),
        ("Missing EFC0", only_late),
    ]:
        if ids:
            print(f"\n{label}:")
            for cell_id in sorted(ids):
                print(f"  {cell_id}")

# Check for duplicate inference results
duplicates = df.duplicated(subset=["cell_id", "curve_type"], keep=False)

if duplicates.any():
    print("\nWARNING: Duplicate cell_id / curve_type results:")
    print(
        df.loc[duplicates, ["cell_id", "chemistry", "curve_type"]]
        .sort_values(["cell_id", "curve_type"])
        .to_string(index=False)
    )

# print(df.loc[df["curve_type"] == "efc0"][features_physical])

