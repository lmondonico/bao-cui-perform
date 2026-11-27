"""
Finetune SMITED-IC model (https://doi.org/10.48550/arXiv.2307.03811)
"""

import torch
import torch.nn as nn
from torch import optim

import pandas as pd
import numpy as np

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from utils.SMITED_trainers import TrainerRegressor
from utils.SMITED_utils import RMSELoss, get_optim_groups
from utils.SMITED_load import load_smi_ted


GPU_ID = 0

N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
MAX_LEN = 202
NUM_FEATS = 32

BATCH_SIZE = 32
LEARNING_RATE = 3e-6
LR_MULTIPLIER = 1
MAX_EPOCHS = 600
DROPOUT = 0.1
D_DROPOUT = 0.1

LOSS_FUNCTION = "rmse"
TARGET_METRIC = "rmse"

DATASET = "smited"

DATA_ROOT = f"./splits-{DATASET}"
DATASET_NAME = DATASET
MEASURE_NAME = "ionic"

SMI_TED_VERSION = "v1"
MODEL_PATH = "."
CKPT_FILENAME = "smi-ted-Light_40.pt"

CHECKPOINTS_FOLDER = "./checkpoints_ic"
RESULTS_FOLDER = f"./smited-ic-results-{DATASET}"
SAVE_CKPT = True
SAVE_EVERY_EPOCH = False
START_SEED = 42
TRAIN_DECODER = True


class Config:
    """Configuration object to mimic argparse namespace."""

    def __init__(self):
        self.n_layer = N_LAYER
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        self.max_len = MAX_LEN
        self.num_feats = NUM_FEATS

        self.n_batch = BATCH_SIZE
        self.lr_start = LEARNING_RATE
        self.lr_multiplier = LR_MULTIPLIER
        self.max_epochs = MAX_EPOCHS
        self.dropout = DROPOUT
        self.d_dropout = D_DROPOUT

        self.loss_fn = LOSS_FUNCTION
        self.target_metric = TARGET_METRIC

        self.data_root = DATA_ROOT
        self.dataset_name = DATASET_NAME
        self.measure_name = MEASURE_NAME

        self.smi_ted_version = SMI_TED_VERSION
        self.model_path = MODEL_PATH
        self.ckpt_filename = CKPT_FILENAME

        self.checkpoints_folder = CHECKPOINTS_FOLDER
        self.results_folder = RESULTS_FOLDER
        self.save_ckpt = SAVE_CKPT
        self.save_every_epoch = SAVE_EVERY_EPOCH
        self.start_seed = START_SEED
        self.train_decoder = TRAIN_DECODER

        self.n_output = 1


def get_device():
    """Select GPU or CPU. Uses GPU_ID if specified, otherwise selects least used GPU."""
    if not torch.cuda.is_available():
        return "cpu"

    if GPU_ID is not None:
        num_gpus = torch.cuda.device_count()
        if GPU_ID >= num_gpus:
            print(
                f"Warning: GPU_ID={GPU_ID} not available. Only {num_gpus} GPU(s) detected."
            )
            print(f"Falling back to GPU 0")
            return "cuda:0"
        print(f"Using manually specified GPU {GPU_ID}")
        return f"cuda:{GPU_ID}"

    num_gpus = torch.cuda.device_count()
    best_gpu = 0
    min_memory = float("inf")

    for i in range(num_gpus):
        memory_used = torch.cuda.memory_allocated(i)
        if memory_used < min_memory:
            min_memory = memory_used
            best_gpu = i

    print(f"Auto-selected GPU {best_gpu} (least used)")
    return f"cuda:{best_gpu}"


def initialize_weights(m):
    """Initialize weights for finetuning."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def main():
    start_time = time.time()

    config = Config()

    os.makedirs(config.results_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.results_folder, f"training_log_{timestamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("main")
    log.setLevel(logging.INFO)

    console_log = logging.getLogger("console")
    console_log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_log.addHandler(console_handler)
    console_log.propagate = False

    log.info("=" * 80)
    log.info("SMI-TED-IC Finetuning - Ionic Conductivity Prediction")
    log.info("=" * 80)
    log.info(f"\nResults folder: {config.results_folder}")
    log.info(f"Log file: {log_file}")

    log.info(f"\nConfiguration:")
    log.info(
        f"  Model: SMI-TED-{config.smi_ted_version} ({config.n_layer} layers, {config.n_head} heads)"
    )
    log.info(f"  Batch size: {config.n_batch}")
    log.info(f"  Learning rate: {config.lr_start}")
    log.info(f"  Max epochs: {config.max_epochs}")
    log.info(f"  Loss function: {config.loss_fn}")
    log.info(f"  Dataset: {config.dataset_name}")
    log.info(f"  Target: {config.measure_name}")

    device = get_device()
    log.info(f"\nUsing device: {device}")

    log.info(f"\nLoading datasets from: {config.data_root}")
    df_train = pd.read_csv(f"{config.data_root}/train_data.csv")
    df_valid = pd.read_csv(f"{config.data_root}/validation_data.csv")
    df_test = pd.read_csv(f"{config.data_root}/test_data.csv")

    log.info(f"  Train: {len(df_train)} samples")
    log.info(f"  Valid: {len(df_valid)} samples")
    log.info(f"  Test:  {len(df_test)} samples")

    log.info(
        f"\nLoading SMI-TED model from: {config.model_path}/{config.ckpt_filename}"
    )

    ckpt_path = os.path.join(config.model_path, config.ckpt_filename)
    if not os.path.exists(ckpt_path):
        log.error(f"\n{'=' * 80}")
        log.error("ERROR: Pretrained checkpoint not found!")
        log.error(f"{'=' * 80}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    model = load_smi_ted(
        folder=config.model_path,
        ckpt_filename=config.ckpt_filename,
        vocab_filename="bert_vocab_curated.txt",
        eval=False,
        vocab_path=script_dir,
        device=device,
    )

    log.info("\nReinitializing prediction head weights...")
    model.net.apply(initialize_weights)
    log.info(f"Prediction head architecture:")
    log.info(model.net)

    lr = config.lr_start * config.lr_multiplier
    optim_groups = get_optim_groups(model, keep_decoder=bool(config.train_decoder))

    if config.loss_fn == "rmse":
        loss_function = RMSELoss()
    elif config.loss_fn == "mae":
        loss_function = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss_fn}")

    log.info(f"\nInitializing trainer...")

    trainer = TrainerRegressor(
        raw_data=(df_train, df_valid, df_test),
        dataset_name=config.dataset_name,
        target=config.measure_name,
        batch_size=config.n_batch,
        hparams=config,
        target_metric=config.target_metric,
        seed=config.start_seed,
        smi_ted_version=config.smi_ted_version,
        checkpoints_folder=config.checkpoints_folder,
        results_folder=config.results_folder,
        restart_filename=None,
        device=device,
        save_every_epoch=bool(config.save_every_epoch),
        save_ckpt=bool(config.save_ckpt),
        console_logger=console_log,
    )

    trainer.compile(
        model=model,
        optimizer=optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99)),
        loss_fn=loss_function,
    )

    log.info(f"\n{'=' * 80}")
    log.info("Starting fine-tuning...")
    log.info(f"{'=' * 80}\n")

    trainer.fit(max_epochs=config.max_epochs)

    log.info(f"\n{'=' * 80}")
    log.info("Calculating final metrics for all datasets...")
    log.info(f"{'=' * 80}\n")

    import utils.SMITED_load as load_module

    model_for_eval = load_module.load_smi_ted(
        folder=config.checkpoints_folder,
        ckpt_filename=trainer.last_filename,
        vocab_filename="bert_vocab_curated.txt",
        eval=True,
        vocab_path=script_dir,
        device=device,
    ).to(device)
    model_for_eval.eval()

    log.info("Evaluating on training set...")
    _, train_loss, train_metrics = trainer._validate_one_epoch(
        trainer.train_loader, model_for_eval
    )

    log.info("Evaluating on validation set...")
    _, val_loss, val_metrics = trainer._validate_one_epoch(
        trainer.valid_loader, model_for_eval
    )

    log.info("Evaluating on test set...")
    _, test_loss, test_metrics = trainer._validate_one_epoch(
        trainer.test_loader, model_for_eval
    )

    log.info(f"\nFinal Training Metrics:")
    log.info(f"  R² Score: {train_metrics['r2']:.4f}")
    log.info(f"  RMSE: {train_metrics['rmse']:.4f}")
    log.info(f"  MSE: {train_metrics['mse']:.4f}")
    log.info(f"  MAE: {train_metrics['mae']:.4f}")
    log.info(f"  Pearson R: {train_metrics['pearson_r']:.4f}")

    log.info(f"\nFinal Validation Metrics:")
    log.info(f"  R² Score: {val_metrics['r2']:.4f}")
    log.info(f"  RMSE: {val_metrics['rmse']:.4f}")
    log.info(f"  MSE: {val_metrics['mse']:.4f}")
    log.info(f"  MAE: {val_metrics['mae']:.4f}")
    log.info(f"  Pearson R: {val_metrics['pearson_r']:.4f}")

    log.info(f"\nFinal Test Metrics:")
    log.info(f"  R² Score: {test_metrics['r2']:.4f}")
    log.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    log.info(f"  MSE: {test_metrics['mse']:.4f}")
    log.info(f"  MAE: {test_metrics['mae']:.4f}")
    log.info(f"  Pearson R: {test_metrics['pearson_r']:.4f}")

    metrics_dict = {
        "dataset": ["training", "validation", "test"],
        "r2_score": [
            round(train_metrics["r2"], 4),
            round(val_metrics["r2"], 4),
            round(test_metrics["r2"], 4),
        ],
        "rmse": [
            round(train_metrics["rmse"], 4),
            round(val_metrics["rmse"], 4),
            round(test_metrics["rmse"], 4),
        ],
        "mse": [
            round(train_metrics["mse"], 4),
            round(val_metrics["mse"], 4),
            round(test_metrics["mse"], 4),
        ],
        "mae": [
            round(train_metrics["mae"], 4),
            round(val_metrics["mae"], 4),
            round(test_metrics["mae"], 4),
        ],
        "pearson_r": [
            round(train_metrics["pearson_r"], 4),
            round(val_metrics["pearson_r"], 4),
            round(test_metrics["pearson_r"], 4),
        ],
        "loss": [round(train_loss, 4), round(val_loss, 4), round(test_loss, 4)],
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = os.path.join(
        config.results_folder, f"{config.dataset_name}_metrics.csv"
    )
    metrics_df.to_csv(metrics_csv_path, index=False)
    log.info(f"\nMetrics saved to: {metrics_csv_path}")

    log.info(f"\n{'=' * 80}")
    log.info("Generating plots and saving results...")
    log.info(f"{'=' * 80}\n")

    axis_tick_size = 14
    axis_title_size = 16
    plot_title_size = 18

    history_df = pd.DataFrame(
        {
            "epoch": range(1, len(trainer.history["train_loss"]) + 1),
            "training_loss": trainer.history["train_loss"],
            "validation_loss": trainer.history["val_loss"],
        }
    )
    history_csv_path = os.path.join(
        config.results_folder, f"{config.dataset_name}_training_history.csv"
    )
    history_df.to_csv(history_csv_path, index=False)
    log.info(f"Training history saved to: {history_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(trainer.history["train_loss"], label="Train Loss", color="orange")
    plt.plot(trainer.history["val_loss"], label="Validation Loss", color="blue")
    plt.legend(frameon=False, fontsize=axis_tick_size)
    plt.xlabel("Epoch", fontsize=axis_title_size)
    plt.ylabel("Loss", fontsize=axis_title_size)
    plt.title(
        "SMI-TED-IC: Training and Validation Loss Curves", fontsize=plot_title_size
    )
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    plt.grid(False)
    plt.tight_layout()
    loss_plot_path = os.path.join(
        config.results_folder, f"{config.dataset_name}_loss.png"
    )
    plt.savefig(loss_plot_path, dpi=900)
    plt.close()
    log.info(f"Loss plot saved to: {loss_plot_path}")

    log.info("\nGenerating parity plot predictions...")

    import utils.SMITED_load as load_module

    model_for_parity = load_module.load_smi_ted(
        folder=config.checkpoints_folder,
        ckpt_filename=trainer.last_filename,
        vocab_filename="bert_vocab_curated.txt",
        eval=True,
        vocab_path=script_dir,
        device=device,
    ).to(device)
    model_for_parity.eval()

    train_preds_list = []
    train_actuals_list = []
    with torch.no_grad():
        for data in trainer.train_loader:
            smiles, targets = data
            targets = targets.to(device)
            embeddings = model_for_parity.extract_embeddings(smiles).to(device)
            predictions = model_for_parity.net(embeddings).squeeze()
            train_preds_list.append(predictions.cpu().numpy())
            train_actuals_list.append(targets.cpu().numpy())

    train_preds = np.concatenate(train_preds_list)
    train_actuals = np.concatenate(train_actuals_list)

    val_preds_list = []
    val_actuals_list = []
    with torch.no_grad():
        for data in trainer.valid_loader:
            smiles, targets = data
            targets = targets.to(device)
            embeddings = model_for_parity.extract_embeddings(smiles).to(device)
            predictions = model_for_parity.net(embeddings).squeeze()
            val_preds_list.append(predictions.cpu().numpy())
            val_actuals_list.append(targets.cpu().numpy())

    val_preds = np.concatenate(val_preds_list)
    val_actuals = np.concatenate(val_actuals_list)

    test_preds_list = []
    test_actuals_list = []
    with torch.no_grad():
        for data in trainer.test_loader:
            smiles, targets = data
            targets = targets.to(device)
            embeddings = model_for_parity.extract_embeddings(smiles).to(device)
            predictions = model_for_parity.net(embeddings).squeeze()
            test_preds_list.append(predictions.cpu().numpy())
            test_actuals_list.append(targets.cpu().numpy())

    test_preds = np.concatenate(test_preds_list)
    test_actuals = np.concatenate(test_actuals_list)

    plt.figure(figsize=(12, 10))

    plt.scatter(
        train_actuals,
        train_preds,
        alpha=0.5,
        s=50,
        c="orange",
        edgecolors="none",
        label="Training",
    )

    plt.scatter(
        val_actuals,
        val_preds,
        alpha=0.5,
        s=50,
        c="blue",
        edgecolors="none",
        label="Validation",
    )

    plt.scatter(
        test_actuals,
        test_preds,
        alpha=0.5,
        s=50,
        c="green",
        edgecolors="none",
        label="Test",
    )

    all_values = np.concatenate([train_actuals, val_actuals, test_actuals])
    min_val, max_val = all_values.min(), all_values.max()
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)
    plt.title("SMI-TED-IC: Parity Plot", fontsize=plot_title_size)
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    plt.legend(frameon=False, fontsize=16, loc="best")
    plt.tight_layout()

    parity_plot_path = os.path.join(
        config.results_folder, f"{config.dataset_name}_parity_plot.png"
    )
    plt.savefig(parity_plot_path, dpi=900)
    plt.close()
    log.info(f"Parity plot saved to: {parity_plot_path}")

    log.info("\nSaving predictions to CSV...")
    predictions_df = pd.DataFrame(
        {
            "split": (
                ["train"] * len(train_actuals)
                + ["validation"] * len(val_actuals)
                + ["test"] * len(test_actuals)
            ),
            "actual": np.concatenate([train_actuals, val_actuals, test_actuals]),
            "predicted": np.concatenate([train_preds, val_preds, test_preds]),
        }
    )
    predictions_csv_path = os.path.join(
        config.results_folder, f"{config.dataset_name}_predictions.csv"
    )
    predictions_df.to_csv(predictions_csv_path, index=False)
    log.info(f"Predictions saved to: {predictions_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    log.info(f"\n{'=' * 80}")
    log.info("Fine-tuning complete!")
    log.info(f"{'=' * 80}\n")
    log.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    log.info(f"Checkpoints saved to: {config.checkpoints_folder}")
    log.info(f"Results saved to: {config.results_folder}")
    log.info(f"\nTo use the finetuned model:")
    log.info(f"  from load import load_smi_ted")
    log.info(f"  model = load_smi_ted('{config.checkpoints_folder}', 'best_model.pt')")


if __name__ == "__main__":
    os.chdir(script_dir)
    main()
