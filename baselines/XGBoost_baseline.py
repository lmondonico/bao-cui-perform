"""Train and evaluate the XGBoost baseline."""

DATASET = "diffmix"
TARGET = "LIC"

RANDOM_SEED = 42

import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Baseline_Utils.PNAS_featurizer import PNASFeaturizer
from Baseline_Utils.GNN_regr_utils import load_split_datasets
from core.utils import get_unique_results_dir
from Baseline_Utils.ML_utils import (
    setup_logging,
    stop_logging,
    save_predictions_to_csv,
    plot_standardized_parity,
)
import warnings

warnings.filterwarnings("ignore")

RESULTS_DIR = get_unique_results_dir(f"baselines/XGBoost_{DATASET}")
os.makedirs(RESULTS_DIR, exist_ok=True)

logger, file_handler = setup_logging(RESULTS_DIR, "xgboost_terminal_output.log")

train_dataset, val_dataset, test_dataset = load_split_datasets(DATASET)

print(f"\nDataset splits:")
print(f"Train: {len(train_dataset)} samples")
print(f"Val:   {len(val_dataset)} samples")
print(f"Test:  {len(test_dataset)} samples")

featurizer = PNASFeaturizer()


def extract_features(dataset):
    """Extract 5 features: 4 PNAS atomic ratios + normalized temperature"""
    features_list = []
    for _, sample in dataset.iterrows():
        features = featurizer.calculate_features(sample, include_temperature=True)
        features_list.append(
            [
                features["sO"],
                features["FO"],
                features["InOr"],
                features["aC"],
                features["Temp_norm"],
            ]
        )
    return np.array(features_list)


print(f"\n{'/' * 50}")
print(f"XGBOOST FOR IONIC CONDUCTIVITY PREDICTION")
print(f"{'/' * 50}\n")

X_train = extract_features(train_dataset)
y_train = train_dataset[TARGET].values

X_val = extract_features(val_dataset)
y_val = val_dataset[TARGET].values

X_test = extract_features(test_dataset)
y_test = test_dataset[TARGET].values

xgb_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

model = XGBRegressor(**xgb_params)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)
train_pearson = stats.pearsonr(y_train, y_train_pred)[0]

val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val, y_val_pred)
val_pearson = stats.pearsonr(y_val, y_val_pred)[0]

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)
test_pearson = stats.pearsonr(y_test, y_test_pred)[0]

print("\nResults:")
print(
    f"Training   - MSE: {train_mse:.3f}, RMSE: {train_rmse:.3f}, R2: {train_r2:.3f}, Pearson r: {train_pearson:.3f}"
)
print(
    f"Validation - MSE: {val_mse:.3f}, RMSE: {val_rmse:.3f}, R2: {val_r2:.3f}, Pearson r: {val_pearson:.3f}"
)
print(
    f"Test       - MSE: {test_mse:.3f}, RMSE: {test_rmse:.3f}, R2: {test_r2:.3f}, Pearson r: {test_pearson:.3f}"
)

print("\nFeature Importances:")
feature_names = ["sO", "FO", "InOr", "aC", "Temp_norm"]
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

save_predictions_to_csv(
    train_true=y_train,
    train_pred=y_train_pred,
    val_true=y_val,
    val_pred=y_val_pred,
    test_true=y_test,
    test_pred=y_test_pred,
    save_path=RESULTS_DIR,
)

plot_path = os.path.join(RESULTS_DIR, "xgboost_parity_plot.png")
plot_standardized_parity(
    train_true=y_train,
    train_pred=y_train_pred,
    val_true=y_val,
    val_pred=y_val_pred,
    test_true=y_test,
    test_pred=y_test_pred,
    save_path=plot_path,
    model_name="XGBoost",
)

print("\nXGBoost training complete!")

stop_logging(logger, file_handler)
