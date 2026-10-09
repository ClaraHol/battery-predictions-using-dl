import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import pandas as pd
import pickle
import joblib




DATA_ROOT = Path("work3/claho/battery_data/processed")
OUT_DIR = Path("work3/claho/battery_models/oracle/")

def load_metadata():
    return

def load_feature_data():
    return

def train_oracle(metadata_train: pd.DataFrame, 
                 feature_data_train: pd.DataFrame,  
                 metadata_test: pd.DataFrame,  
                 feature_data_test: pd.DataFrame,  
                 num_features: int,
                 alpha_sbi = 0.8,
                 C_svr = 1.4,
                 epsilon_f_ope = 0.2,
                 l1_ratio = 0.8
    ):
    """
    Two-stage meta-learning pipeline for battery life prediction.

    Assumes `feature_data_train` and `feature_data_test` are DataFrames
    containing a 'cell_id' column, feature columns, and target columns.

    Assumes 'meta_data_test/train' contains columns 
    ["cell_id", "group_id", "temp", "Chrg-rate", "Dchrg-rate]
    
    """
    model_parameters_train = []
    cell_ids_train = metadata_train['cell_id'].values
    #-------------------------------------------------------------------------------------------------
    # Stage 1: Train an ElasticNet for each cluster (Temp, Chrg-Rate, Dchrg-Rate)
    #-------------------------------------------------------------------------------------------------

    for cell_id in cell_ids_train:
        # Extract cell feature rows using cell_id
        feature_data = feature_data_train[feature_data_train['cell_id'] == cell_id]

        # Extract features (first `num_features` columns) and target (last column)
        train_features = feature_data.iloc[:, :num_features].to_numpy()
        train_target = feature_data.iloc[:, -1].to_numpy()  # EFC cycle life target in last column

        #---------------------------------------------------------------------------------------------
        # Remove uninformative features by setting them to zero
        #---------------------------------------------------------------------------------------------
        # 1. Number of features to keep
        k_value = next(k for cap, k in [(10, 10), (6, 5), (3, 3)] if num_features > cap) or 1
        
        # 2. Fit selector & identify kept/dropped features
        selector = SelectKBest(score_func=mutual_info_regression, k=k_value).fit(train_features, train_target)
        keep_mask = selector.get_support()

        # 3. Create cleaned array (zeros out non-selected features cleanly)
        train_feature_clean = np.where(keep_mask, train_features, 0)

        #---------------------------------------------------------------------------------------------
        # Train ElasticNet
        #---------------------------------------------------------------------------------------------
        elasticNet = linear_model.ElasticNet(alpha=alpha_sbi, l1_ratio=l1_ratio, random_state=0)
        elasticNet.fit(train_feature_clean, train_target)

        # Save model parameters
        coefficients = elasticNet.coef_
        intercept = elasticNet.intercept_
        parameter = np.insert(coefficients, 0, intercept)   # [Intercept, coef_1, coef_2, ...]
        model_parameters_train.append(parameter)

    #-------------------------------------------------------------------------------------------------
    # Stage 2: Train meta predictor
    #-------------------------------------------------------------------------------------------------
    # Step 1: Convert training and target data to numpy arrays 
    train_target_svr = np.array(model_parameters_train)
    
    # Exclude cell_ids and group_id from metadata
    train_input = metadata_train.drop(columns=["cell_id", "group_id"]).to_numpy() 
    test_input = metadata_test.drop(columns=["cell_id", "group_id"]).to_numpy()
    
    # Step 2: Scale and pass to SVR
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(train_input) #shape: (n_cells, 3)
    X_test = scaler_X.transform(test_input)
    Y_train = scaler_Y.fit_transform(train_target_svr)


    # Step 3: Fit SVR meta predictor
    svr = SVR(kernel='rbf', C=C_svr, epsilon=epsilon_f_ope)
    meta_predictor = MultiOutputRegressor(svr)
    meta_predictor.fit(X_train, Y_train)


    # Predict parameters using the meta_predictor
    y_pred = meta_predictor.predict(X_test)
    model_parameters_test = scaler_Y.inverse_transform(y_pred)

    # ---------------------------------------------------------------------------
    # Stage 3: Vectorized Evaluation on Test Data
    # ---------------------------------------------------------------------------
    test_records = []

    test_cell_ids = feature_data_test["cell_id"].values()
    for idx, cell_id in enumerate(test_cell_ids):
        test_features = feature_data_test.loc[[cell_id]].copy()
        
        # Extract predicted weights [intercept, coefficients]
        intercept = model_parameters_test[idx, 0]
        coef = model_parameters_test[idx, 1:]

        # Direct matrix multiplication (Vectorized ElasticNet Evaluation)
        X_test_cell = test_features[:, :num_features].to_numpy()
        y_pred = np.abs(X_test_cell @ coef + intercept)

        test_features['predicted_cycle_life'] = y_pred
        test_features['true_cycle_life'] = test_features.iloc[:, -1].values
       
        test_records.append(test_features)

    # Combine all evaluated test data
    df_eval = pd.concat(test_records, ignore_index=True)

    # Merge absed on group_id
    df_eval = df_eval.merge(
        metadata_test[['cell_id', 'group_id']], on='cell_id', how='left'
    )

    # Calculate Cell-Level Metrics
    df_eval['loss'] = np.abs(df_eval['true_cycle_life'] - df_eval['predicted_cycle_life']) / df_eval['true_cycle_life']
    mape = df_eval['loss'].mean()

    # Calculate Group-Level Metrics via pandas groupby
    df_group = df_eval.groupby("group_id")[['true_cycle_life', 'predicted_cycle_life']].mean()
    df_group['group_loss'] = np.abs(df_group['true_cycle_life'] - df_group['predicted_cycle_life']) / df_group['true_cycle_life']
    group_mape = df_group['group_loss'].mean()


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
    axes[0].set_title(f'Cell Loss (MAPE = {mape:.4f})')
    axes[0].set_ylabel('Absolute Relative Error')
    axes[0].set_xlabel('Sample Index')

    # Plot 2: Cell-level Scatter (Predicted vs True)
    ref = np.linspace(0, max(df_eval['true_cycle_life'].max(), df_eval['predicted_cycle_life'].max()), 100)
    axes[1].plot(ref, ref, 'k--')
    axes[1].scatter(df_eval['predicted_cycle_life'], df_eval['true_cycle_life'], c='r', alpha=0.5, s=10)
    axes[1].set_title('Observed vs Predicted EFC')
    axes[1].set_xlabel('Predicted EFC')
    axes[1].set_ylabel('Observed EFC')

    plt.tight_layout()

    # Ensure directory exists before saving
    save_path = Path("project/reports/figures/oracle_training.png")
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

    # 1. Load the data



    # 2. Run the training loop


    return 
