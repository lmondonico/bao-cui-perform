"""Train and evaluate the MM-MoLFormer baseline."""

DATASET = "smited"
TARGET = "LIC"
GPU_ID = 0

RANDOM_SEED = 42
batch_size = 256
num_epochs = 200

import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Baseline_Utils.GNN_regr_utils import *
from Baseline_Utils.GNN_modules import MolformerMoleculeEncoder
from core.utils import get_unique_results_dir
import warnings

warnings.filterwarnings("ignore")

RESULTS_DIR = get_unique_results_dir(f"baselines/Molformer_{DATASET}")
GNN_TYPE = "Molformer"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    logger, file_handler = setup_logging(
        RESULTS_DIR, f"{GNN_TYPE.lower()}_terminal_output.log"
    )

    start_time = time.time()

    if torch.cuda.is_available() and GPU_ID >= 0:
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"\nUsing device: cuda:{GPU_ID}\n")
    elif GPU_ID == -1:
        device = torch.device("cpu")
        print(f"\nUsing device: CPU (GPU_ID set to -1)\n")
    else:
        device = torch.device("cpu")
        print(f"\nCUDA not available. Using device: CPU\n")

    print(f"\n{'/' * 50}")
    print(f"{GNN_TYPE} FOR IONIC CONDUCTIVITY PREDICTION")
    print(f"{'/' * 50}\n")

    print("Loading Molformer encoder...")
    molformer_encoder = MolformerMoleculeEncoder(max_length=202)

    train_data, val_data, test_data = load_split_datasets(DATASET)

    processed_datasets = create_gnn_datasets(
        train_data,
        val_data,
        test_data,
        target=TARGET,
        molecule_encoder=molformer_encoder,
    )
    train_dataset, val_dataset, test_dataset = processed_datasets

    train_loader, val_loader, test_loader = create_gnn_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    model = initialize_gnn_model(GNN_TYPE, device, molformer_encoder=molformer_encoder)

    weights_path = os.path.join(RESULTS_DIR, f"{GNN_TYPE.lower()}_model.pth")
    plot_path = os.path.join(RESULTS_DIR, f"{GNN_TYPE.lower()}_training_loss.png")

    print(f"\nTraining {GNN_TYPE} model...")
    train_results, val_results = train_gnn_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        device,
        weights_path,
        plot_save_path=plot_path,
        show_epoch_info=True,
        kfold_train=False,
        patience=50,
    )

    print(f"\nEvaluating {GNN_TYPE} model on test set...")
    test_results = evaluate_gnn_model(
        model, test_loader, device, task="Test", weights_path=weights_path
    )

    print("\nFinal Results:")
    metrics = calculate_metrics(
        train_results,
        val_results,
        test_results,
        show_results=True,
        show_prediction_ranges=True,
    )

    plot_gnn_results_standardized(
        train_results,
        val_results,
        test_results,
        metrics,
        RESULTS_DIR,
        gnn_type=GNN_TYPE,
    )
    save_gnn_predictions_simple_csv(
        train_results,
        val_results,
        test_results,
        save_path=RESULTS_DIR,
        gnn_type=GNN_TYPE,
    )

    get_running_time(start_time)
    print(f"\n{GNN_TYPE} training complete! Results saved to {RESULTS_DIR}\n")

    stop_logging(logger, file_handler)


if __name__ == "__main__":
    main()
