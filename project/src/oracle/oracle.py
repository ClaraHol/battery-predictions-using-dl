import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import pandas as pd
import joblib


INTERPRETER = "mlp"
DATA_ROOT = Path(f"/work3/s214728/battery-predictions/models/inference_results/")
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

    return train, test

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

def train_oracle(metadata_train: pd.DataFrame, 
                 feature_data_train: pd.DataFrame,  
                 metadata_test: pd.DataFrame,  
                 feature_data_test: pd.DataFrame,
                 remove_outliers = False,
                 alpha_sbi = 0.8,
                 C_svr = 1.4,
                 epsilon_f_ope = 0.2,
                 l1_ratio = 0.8,
                 
    ):
    """
    Two-stage meta-learning pipeline for battery life prediction.

    Assumes `feature_data_train` and `feature_data_test` are DataFrames
    containing a 'cell_id' column and feature columns.

    Assumes 'meta_data_test/train' contains columns 
    ["cell_id", "group_id", "temp", "Chrg-rate", "Dchrg-rate", "cycle_life"]
    
    """
    #-------------------------------------------------------------------------------------------------
    # Stage 1: Train an ElasticNet for each cluster (Temp, Chrg-Rate, Dchrg-Rate)
    #-------------------------------------------------------------------------------------------------
    train_data = metadata_train.merge(
    feature_data_train,
    on="cell_id",
    how="inner",
    validate="one_to_one"
    )

    test_data = metadata_test.merge(
    feature_data_test,
    on="cell_id",
    how="inner",
    validate="one_to_one"
    )
    
    feature_columns = feature_data_train.columns.drop("cell_id").tolist()


    scaler_features = StandardScaler()
    scaler_features.fit(train_data[feature_columns])

    train_data[feature_columns] = scaler_features.transform(train_data[feature_columns])
    test_data[feature_columns] = scaler_features.transform(test_data[feature_columns])

    model_parameters_train = []
    meta_inputs_train = []

    for group_id, group in train_data.groupby("group_id"):

        train_features = group[feature_columns].to_numpy(dtype=float)
        train_target = np.log(group["cycle_life"].to_numpy(dtype=float))

        # Feature selection
        k_value = (
            10 if train_features.shape[1] > 10
            else 5 if train_features.shape[1] > 6
            else 3 if train_features.shape[1] > 3
            else 1
        )

        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=k_value
        )

        train_features = selector.fit_transform(
            train_features, train_target
        )


        # Important: retain original feature dimensions
        keep_mask = selector.get_support()
        X = group[feature_columns].to_numpy(dtype=float).copy()
        X[:, ~keep_mask] = 0

        # ElasticNet
        elasticNet = linear_model.ElasticNet(
            alpha=alpha_sbi,
            l1_ratio=l1_ratio,
            random_state=0
        )

        elasticNet.fit(X, train_target)

        parameters = np.insert(elasticNet.coef_, 0, elasticNet.intercept_)
        model_parameters_train.append(parameters)

        # One operating-condition vector per group
        meta_inputs_train.append(
            group[["temp", "Chrg-rate", "Dchrg-rate"]]
            .iloc[0]
            .to_numpy(dtype=float)
        )
   

    #-------------------------------------------------------------------------------------------------
    # Stage 2: Train meta predictor
    #-------------------------------------------------------------------------------------------------
    # Step 1: Convert training and target data to numpy arrays 
        # One coefficient vector per operational group
    
    train_target_svr = np.asarray(model_parameters_train, dtype=float)
    
    # One row per operational group
    train_input = np.asarray(meta_inputs_train, dtype=float)

    
    # Step 2: Scale and pass to SVR
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(train_input) #shape: (n_cells, 3)
    Y_train = scaler_Y.fit_transform(train_target_svr)
    #print(f"{X_train.shape=}")
    #print(f"{Y_train.shape=}")


    # Step 3: Fit SVR meta predictor
    svr = SVR(kernel='rbf', C=C_svr, epsilon=epsilon_f_ope)
    meta_predictor = MultiOutputRegressor(svr)
    meta_predictor.fit(X_train, Y_train)

    # One input row per test cell
    meta_features = ["temp", "Chrg-rate", "Dchrg-rate"]
    test_input = metadata_test[meta_features].to_numpy(dtype=float)
    X_test = scaler_X.transform(test_input)

    # Predict parameters using the meta_predictor
    y_pred = meta_predictor.predict(X_test)
    model_parameters_test = scaler_Y.inverse_transform(y_pred)

    # ---------------------------------------------------------------------------
    # Stage 3: Vectorized Evaluation on Test Data
    # ---------------------------------------------------------------------------

    test_input = test_data[meta_features].to_numpy(dtype=float)
    X_test = scaler_X.transform(test_input)
    model_parameters_test = scaler_Y.inverse_transform(meta_predictor.predict(X_test))
    X_test_features = test_data[feature_columns].to_numpy(dtype=float)

    intercepts = model_parameters_test[:, 0]
    coefficients = model_parameters_test[:, 1:]

    assert X_test_features.shape[1] == coefficients.shape[1]

    predicted_cycle_life = (
        np.sum(X_test_features * coefficients, axis=1)
        + intercepts
    )
    predicted_cycle_life = np.exp(predicted_cycle_life)

    df_eval = test_data[["cell_id", "group_id", "cycle_life"]].copy()
    df_eval["predicted_cycle_life"] = predicted_cycle_life

    # Calculate Cell-Level Metrics
    df_eval['loss'] = np.abs(df_eval['cycle_life'] - df_eval['predicted_cycle_life']) / df_eval['cycle_life']
    if remove_outliers:
        df_eval = df_eval[df_eval['loss'] < 15]
    outliers = df_eval[df_eval['loss'] > 0.5]
    print(f"Outliers:\n {outliers}")
    mape = df_eval['loss'].mean() * 100

    # Calculate Group-Level Metrics via pandas groupby
    df_group = df_eval.groupby("group_id")[['cycle_life', 'predicted_cycle_life']].mean()
    df_group['group_loss'] = np.abs(df_group['cycle_life'] - df_group['predicted_cycle_life']) / df_group['cycle_life']
    group_mape = df_group['group_loss'].mean() * 100


    # Save models
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, OUT_DIR/"scaler_X.pkl") 
    joblib.dump(scaler_Y, OUT_DIR/"scaler_Y.pkl") 
    joblib.dump(meta_predictor, OUT_DIR/"meta_predictor.pkl")

    # ---------------------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cell-level error
    axes[0].plot(range(len(df_eval)), df_eval['loss'], color='blue', alpha=0.6)
    axes[0].scatter(range(len(df_eval)), df_eval['loss'], s=5, c='r')
    axes[0].set_title(f'Cell Loss (MAPE = {mape:.4f}%)')
    axes[0].set_ylabel('Absolute Relative Error')
    axes[0].set_xlabel('Sample Index')

    # Plot 2: Cell-level Scatter (Predicted vs True)
    ref = np.linspace(0, max(df_eval['cycle_life'].max(), df_eval['predicted_cycle_life'].max()), 100)
    axes[1].plot(ref, ref, 'k--')
    axes[1].scatter(df_eval['predicted_cycle_life'], df_eval['cycle_life'], c='r', alpha=0.5, s=10)
    axes[1].set_title('Observed vs Predicted EFC')
    axes[1].set_xlabel('Predicted EFC')
    axes[1].set_ylabel('Observed EFC')

    plt.tight_layout()

    # Ensure directory exists before saving
    save_path = Path(f"project/reports/figures/oracle/oracle_training_{INTERPRETER.lower()}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    return mape, group_mape

def run_oracle(meta_data, features, num_features = 28):
    """
    Takes one data point at a time.
    Assumes that meta_data = (temp, Chrg-rate, Dchrg-rate) and 
    features containes the 28 features from the SBI (so no cell_ids)

    """
    # Load models
    meta_predictor = joblib.load(OUT_DIR/"meta_predictor.pkl")
    scaler_X = joblib.load(OUT_DIR/"scaler_X.pkl")
    scaler_Y = joblib.load(OUT_DIR/"scaler_Y.pkl")

    X = scaler_X.transform(meta_data.to_numpy())

    Y = meta_predictor.predict(X)
    parameters = scaler_Y.inverse_transform(Y)

    # Extract predicted weights [intercept, coefficients]
    intercept = parameters[0, 0]
    coef = parameters[0, 1:]

    # Direct matrix multiplication (Vectorized ElasticNet Evaluation)
    X_cell = features[:, :num_features].to_numpy()
    y_pred = np.abs(X_cell @ coef + intercept)

    return y_pred

def main():
    """
    Training loop for the oracle
    """

    
    mlp_filname = "experimental_inference_mlp.csv"





    # 1. Load the data
    metadata_train, metadata_test = load_metadata()
    if INTERPRETER == "SBI":
        feature_data = load_feature_data()
    else:
        feature_data = load_feature_data(mlp_filname)



    # Match metadata and features by cell_id
    train = metadata_train.merge(
        feature_data,
        on="cell_id",
        how="inner",
        validate="one_to_one"
    )
    test_new = metadata_test.merge(
        feature_data,
        on="cell_id",
        how="inner",
        validate="one_to_one"
    )

    # Select one test cell per operational group
    test = train.groupby("group_id").sample(
        n=2,
        random_state=0
    )

    # Remaining cells for training
    train = train.drop(test.index)

    # Separate metadata and features
    meta_columns = [
        "cell_id", "group_id", "temp",
        "Chrg-rate", "Dchrg-rate", "cycle_life"
    ]

    metadata_train = train[meta_columns].copy()
    metadata_test = test[meta_columns].copy()

    feature_columns = [
        col for col in feature_data.columns
        if col != "cell_id"
    ]

    feature_data_train = train[["cell_id"] + feature_columns].copy()
    feature_data_test = test[["cell_id"] + feature_columns].copy()

    #print(f"{metadata_train.shape=}")
    #print(f"{metadata_test.shape=}")

    #print(f"{feature_data_train.shape=}")
    #print(f"{feature_data_test.shape=}")
    


    # 2. Run the training loop
    train_oracle(metadata_train,
                feature_data_train, 
                metadata_test, 
                feature_data_test
                )


    return 

if __name__ == "__main__":
    main()