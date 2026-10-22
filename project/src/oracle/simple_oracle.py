
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import NearestNeighbors

import ast
import numpy as np
import matplotlib.pyplot as plt


from pathlib import Path
import pandas as pd




INTERPRETER = "SBI"
DATA_ROOT = Path(f"/work3/s214728/battery-predictions/models/{INTERPRETER}_models/Combined")
METADATA_ROOT = Path("/work3/claho/battery_datasets/processed")
OUT_DIR = Path(f"work3/claho/battery_models/oracle/{INTERPRETER.lower()}/")


def load_metadata(
    filename: str = "public_cell_metadata_all_groups.csv",
    filename2: str = "public_cell_metadata_cleaned.csv"
):
    # Load metadata
    df_all = pd.read_csv(METADATA_ROOT / filename)
    df_cleaned = pd.read_csv(METADATA_ROOT / filename2)  

    # Generate consistent group IDs
    df_all["group_id"] = pd.factorize(df_all["operational_cluster"])[0]

    # Split into training and testing
    train_df = df_all[df_all["cell_id"].isin(df_cleaned["cell_id"])].copy()
    test_df = df_all[~df_all["cell_id"].isin(df_cleaned["cell_id"])].copy()

    # Convert to desired format
    def prepare_metadata(df):
        conditions = df["operational_cluster"].apply(ast.literal_eval)

        out = df[["cell_id", "group_id", "cycle_life"]].copy()
        out[["temp", "Chrg-rate", "Dchrg-rate"]] = pd.DataFrame(conditions.tolist(),index=df.index)

        return out.reset_index(drop=True)

    train = prepare_metadata(train_df)
    test = prepare_metadata(test_df)
    all = prepare_metadata(df_all)

    return train, test, all

def load_feature_data(filename: str = "experimental_inference.csv"):
    df = pd.read_csv(DATA_ROOT / filename)
    
    efc0 = df[df["curve_type"] == "efc0"].set_index("cell_id")
    efc50 = df[df["curve_type"] == "efc50"].set_index("cell_id")

    # Assert that each cell only has 1 efc0 and 1 efc50
    assert efc0.index.is_unique, "Duplicate cell IDs at EFC 0"
    assert efc50.index.is_unique, "Duplicate cell IDs at EFC 50"

    # Only extract the physical features not normalized features
    feature_columns = [f for f in efc0.columns if "physical" in f]

    # Keep only numerical features
    features = efc0[feature_columns].select_dtypes(include="number").columns

    # Keep only cells with both curves
    common_ids = efc0.index.intersection(efc50.index)

    # EFC 0 features
    out = efc0.loc[common_ids, features].copy()

    # Differences
    diff = (
        efc50.loc[common_ids, features]
        - efc0.loc[common_ids, features]
    ).add_suffix("_diff")

    # Combine
    out = pd.concat([out, diff], axis=1).reset_index()
    return out

    

def main():
    mlp_filename = "experimental_inference_mlp.csv"

    # Load only once; the original metadata_test is not used for CV.
    metadata_train, _, metadata_all = load_metadata()
    feature_data = load_feature_data(
        "experimental_inference.csv" if INTERPRETER == "SBI" else mlp_filename
    )


    data = metadata_train.merge(
        feature_data, on="cell_id", how="inner", validate="one_to_one"
    ).reset_index(drop=True)

    meta_columns = [
        "cell_id", "group_id", "temp", "Chrg-rate", "Dchrg-rate", "cycle_life"
    ]
    feature_columns = [col for col in feature_data.columns if col != "cell_id"]

    X = data[
        feature_columns +
        ["temp", "Chrg-rate", "Dchrg-rate"]
    ].to_numpy()

    y = data["cycle_life"].to_numpy()

    loo = LeaveOneOut()
    predictions = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=10,
            l2_regularization=1.0,
            random_state=1
        )

        model.fit(X[train_idx], np.log(y[train_idx]))

        predictions[test_idx] = np.exp(
            model.predict(X[test_idx])
        )

    data["prediction"] = predictions

    # Calculate Cell-Level Metrics
    df_eval = data[["cell_id", "group_id", "cycle_life", "prediction"]].copy()
    df_eval['loss'] = np.abs(df_eval['cycle_life'] - df_eval['prediction']) / df_eval['cycle_life']
    mape = df_eval['loss'].mean() * 100

    # Calculate Group-Level Metrics via pandas groupby
    df_group = df_eval.groupby("group_id")[['cycle_life', 'prediction']].mean()
    df_group['group_loss'] = np.abs(df_group['cycle_life'] - df_group['prediction']) / df_group['cycle_life']
    group_mape = df_group['group_loss'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cell-level error
    axes[0].plot(range(len(df_eval)), df_eval['loss'], color='blue', alpha=0.6)
    axes[0].scatter(range(len(df_eval)), df_eval['loss'], s=5, c='r')
    axes[0].set_title(f'Cell Loss (MAPE = {mape:.4f}%)')
    axes[0].set_ylabel('Absolute Relative Error')
    axes[0].set_xlabel('Sample Index')

    # Plot 2: Cell-level Scatter (Predicted vs True)
    ref = np.linspace(0, max(df_eval['cycle_life'].max(), df_eval['prediction'].max()), 100)
    axes[1].plot(ref, ref, 'k--')
    axes[1].scatter(df_eval['prediction'], df_eval['cycle_life'], c='r', alpha=0.5, s=10)
    axes[1].set_title('Observed vs Predicted EFC')
    axes[1].set_xlabel('Predicted EFC')
    axes[1].set_ylabel('Observed EFC')

    plt.tight_layout()

    # Ensure directory exists before saving
    save_path = Path(f"project/reports/figures/simple_oracle/simple_oracle_{INTERPRETER}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)


    data["ape"] = (
        100 * np.abs(data["cycle_life"] - data["prediction"])
        / data["cycle_life"]
    )

    group_mape = (
        data.groupby("group_id")["ape"]
        .mean()
    )
    plt.figure()
    plt.plot(group_mape)
    plt.title("Group MAPE")
    plt.xlabel("Group number")
    plt.ylabel("MAPE")
    plt.tight_layout()
    plt.savefig(f"project/reports/figures/simple_oracle/group_mape_{INTERPRETER}.png", dpi = 300)

    print(group_mape)
    print(f"Mean: {group_mape.mean()}")

    overall_mape = data["ape"].mean()

    group1_mape = data.loc[
        data["group_id"] == 1, "ape"
    ].mean()

    without_group1 = data.loc[
        data["group_id"] != 1, "ape"
    ].mean()

    group1 =  data.loc[data["group_id"] == 1]
    group1_features = group1[feature_columns]
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(group1_features)
    distance_mat, neighbours_mat = knn.kneighbors(group1_features)
    print(f"{distance_mat=}")
    print(f"{neighbours_mat=}")


    rows = []

    for i in range(len(group1)):
        j = neighbours_mat[i, 1]  # Skip self-neighbor

        life_i = group1.iloc[i]["cycle_life"]
        life_j = group1.iloc[j]["cycle_life"]

        rows.append({
            "cell_id": group1.iloc[i]["cell_id"],
            "cycle_life": life_i,
            "neighbor_id": group1.iloc[j]["cell_id"],
            "neighbor_life": life_j,
            "feature_distance": distance_mat[i, 1],
            "life_ratio": max(life_i, life_j) / min(life_i, life_j),
            "life_difference": abs(life_i - life_j)
        })

    neighbor_analysis = pd.DataFrame(rows)

    print(
        neighbor_analysis.sort_values(
            "feature_distance"
        ).to_string(index=False)
    )


    print(
        group1[
            ["cell_id", "cycle_life", "prediction", "ape"]
        ].sort_values("ape", ascending=False)
    )

    print(f"Overall MAPE: {overall_mape:.2f}%")
    print(f"Group 1 MAPE: {group1_mape:.2f}%")
    print(f"Without Group 1: {without_group1:.2f}%")
    return 0

if __name__ == "__main__":
    main()