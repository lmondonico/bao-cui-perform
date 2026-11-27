"""
Utility functions for traditional ML baselines (LinReg, RF, XGBoost, SVM)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def setup_logging(results_dir, filename="terminal_output.log"):
    """
    Set up logging to capture terminal output to a file in the results directory.
    """
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(results_dir, filename)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    class StreamToLogger:
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ""

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())
            sys.__stdout__.write(buf)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = StreamToLogger(logger, logging.INFO)

    print(f"Logging initialized. Output saved to: {log_file}")

    return logger, file_handler


def stop_logging(logger, file_handler):
    """
    Stop logging and restore normal stdout.
    """
    sys.stdout = sys.__stdout__

    logger.removeHandler(file_handler)
    file_handler.close()

    print(f"Logging stopped and file closed")


def save_predictions_to_csv(
    train_true, train_pred, val_true, val_pred, test_true, test_pred, save_path
):
    """
    Save predictions to CSV in standardized format matching train_kfold.py.
    Columns: Task, Truth, Prediction (compatible with plot_parity_from_csv.py)
    """
    train_df = pd.DataFrame(
        {
            "Task": "Training",
            "Truth": train_true,
            "Prediction": train_pred,
        }
    )

    val_df = pd.DataFrame(
        {
            "Task": "Validation",
            "Truth": val_true,
            "Prediction": val_pred,
        }
    )

    test_df = pd.DataFrame(
        {
            "Task": "Test",
            "Truth": test_true,
            "Prediction": test_pred,
        }
    )

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    csv_path = os.path.join(save_path, "regr_predictions.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}\n")

    return csv_path


def plot_standardized_parity(
    train_true,
    train_pred,
    val_true,
    val_pred,
    test_true,
    test_pred,
    save_path,
    model_name="Model",
):
    """
    Create standardized parity plot matching plot_parity_from_csv.py format.
    - Metrics in title (R² and RMSE)
    - DPI 900
    - No histogram inset
    - Simple legend
    - Colors: orange (train), blue (val), green (test)
    """
    train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
    train_r2 = r2_score(train_true, train_pred)

    val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    val_r2 = r2_score(val_true, val_pred)

    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_r2 = r2_score(test_true, test_pred)

    axis_tick_size = 14
    axis_title_size = 16
    plot_title_size = 18

    plt.figure(figsize=(12, 10))

    plt.scatter(
        train_true,
        train_pred,
        color="orange",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Training",
    )
    plt.scatter(
        val_true,
        val_pred,
        color="blue",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Validation",
    )
    plt.scatter(
        test_true,
        test_pred,
        color="green",
        alpha=0.5,
        s=50,
        edgecolors="none",
        label="Test",
    )

    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)

    title_line1 = f"{model_name} Parity Plot (Actual vs Predicted)"
    title_line2 = (
        f"Train: R²={train_r2:.3f} RMSE={train_rmse:.3f} | "
        f"Val: R²={val_r2:.3f} RMSE={val_rmse:.3f} | "
        f"Test: R²={test_r2:.3f} RMSE={test_rmse:.3f}"
    )
    plt.title(
        f"{title_line1}\n{title_line2}", fontsize=plot_title_size, fontweight="bold"
    )

    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)

    plt.legend(frameon=False, fontsize=16)
    plt.grid(False)

    plt.tight_layout()

    plt.savefig(save_path, dpi=900, bbox_inches="tight")
    plt.close()

    print(f"Standardized parity plot saved to: {save_path}")
