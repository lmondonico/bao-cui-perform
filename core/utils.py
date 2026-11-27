"""Shared training and evaluation utilities."""

import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.imports import *
from modules.tokenizer.tokenizer_perform import Tokenizer
from modules.perform.checkpoint_compat import (
    normalize_motif_architecture,
    normalize_motif_state_dict,
)
from modules.dataset_processing import *
from modules.moiety_features import *
from tasks.training.train_parm import TARGET_LABEL
from config.datasets import get_dataset_config
from collections import defaultdict
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU selection will use PyTorch memory only.")


def get_gpu_utilization(device_id):
    """Get GPU compute utilization percentage using pynvml."""
    if not PYNVML_AVAILABLE:
        return 0

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception:
        return 0


def get_gpu_memory_usage(device_id):
    """Get GPU memory utilization percentage and details."""
    if not PYNVML_AVAILABLE:
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory
            usage_percent = (allocated / total) * 100 if total > 0 else 0
            return usage_percent, allocated, total
        except Exception:
            return 0, 0, 0

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage_percent = (memory_info.used / memory_info.total) * 100
        return memory_usage_percent, memory_info.used, memory_info.total
    except Exception:
        return 0, 0, 0


def calculate_gpu_score(device_id, memory_weight=0.4, utilization_weight=0.6):
    """Calculate combined GPU usage score (lower is better)."""
    memory_usage, _, _ = get_gpu_memory_usage(device_id)
    gpu_utilization = get_gpu_utilization(device_id)

    score = (memory_weight * memory_usage) + (utilization_weight * gpu_utilization)
    return score


def print_gpu_status(show_details=True):
    """Print current GPU status for debugging."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    num_gpus = torch.cuda.device_count()

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
        except Exception:
            print("NVML initialization failed")
            return

    print(f"\nGPU Status ({num_gpus} GPUs):")
    if show_details:
        print("GPU | Util% | Mem% | MemUsed/Total(GB) | Score")
        print("-" * 55)

    for i in range(num_gpus):
        try:
            memory_usage, mem_used, mem_total = get_gpu_memory_usage(i)
            gpu_util = get_gpu_utilization(i)
            score = calculate_gpu_score(i)

            if show_details:
                print(
                    f" {i}  | {gpu_util:4.1f} | {memory_usage:4.1f} | {mem_used / 1e9:5.2f}/{mem_total / 1e9:5.2f} | {score:5.1f}"
                )
            else:
                print(
                    f"GPU {i}: {gpu_util:4.1f}% util, {memory_usage:4.1f}% mem (score: {score:4.1f})"
                )

        except Exception as e:
            print(f"GPU {i}: Error - {e}")

    if show_details:
        print("-" * 55)


def get_least_used_gpu():
    """Find GPU with lowest combined memory + utilization score."""
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return 0

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
        except Exception:
            print("NVML initialization failed, falling back to memory-only selection")
            return get_least_used_gpu_fallback()

    best_gpu = 0
    best_score = float("inf")

    for i in range(num_gpus):
        try:
            if PYNVML_AVAILABLE:
                score = calculate_gpu_score(i)
            else:
                memory_used = torch.cuda.memory_allocated(i)
                score = memory_used

            if score < best_score:
                best_score = score
                best_gpu = i

        except Exception:
            continue

    return best_gpu


def get_least_used_gpu_fallback():
    """Fallback method using only PyTorch memory allocation."""
    num_gpus = torch.cuda.device_count()
    min_memory_used = float("inf")
    best_gpu = 0

    for i in range(num_gpus):
        try:
            memory_used = torch.cuda.memory_allocated(i)
            if memory_used < min_memory_used:
                min_memory_used = memory_used
                best_gpu = i
        except Exception:
            continue

    return best_gpu


def get_device(show_device=True, show_gpu_details=False):
    """Check if GPU is available and select least used one."""

    print("\n")
    print("+" * 60)
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                if show_gpu_details:
                    print_gpu_status(show_details=True)

                device_id = get_least_used_gpu()
                device = torch.device(f"cuda:{device_id}")

                if show_device:
                    memory_usage, _, _ = get_gpu_memory_usage(device_id)
                    gpu_util = get_gpu_utilization(device_id)
                    score = calculate_gpu_score(device_id)
                    print(
                        f"Selected GPU {device_id}: {gpu_util:.1f}% util, {memory_usage:.1f}% mem (score: {score:.1f})"
                    )
                    print(f"Using device: {device}")
            else:
                device = torch.device("cuda")
                if show_device:
                    print(f"\nUsing device: {device} (single GPU)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            if show_device:
                print(f"\nUsing device: {device}")
        else:
            device = torch.device("cpu")
            if show_device:
                print(f"\nUsing device: {device}")

        print("+" * 60, "\n")
        return device
    except Exception as e:
        print(f"Error detecting device: {e}")
        sys.exit(1)


def get_running_time(start_time):
    """Calculate the running time of a script."""

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    print("\n" + "/" * 30)
    print(f"PROCESS COMPLETED in {elapsed_minutes:.2f} minutes")
    print("/" * 30 + "\n")

    return


def get_unique_results_dir(base_name):
    """
    Return a unique run directory without replacing an existing result.
    """
    from config.paths import RESULTS_DIR

    pid = os.getpid()
    return os.path.join(RESULTS_DIR, f"{base_name}_{pid}")


def copy_parameter_files(results_dir, dataset_type=None, script_type=None):
    """
    Copy the contents of current parameter files into a single txt file.

    Args:
        results_dir: Directory to save the regr_parameters.txt file
        dataset_type: Type of dataset ("conductivity" for IC, "ablation" for ablation studies)
        script_type: Unused, kept for backward compatibility
    """
    from config.paths import PROJECT_ROOT

    if dataset_type == "conductivity":
        param_files = [
            os.path.join(PROJECT_ROOT, "tasks/training/train_parm.py"),
            os.path.join(PROJECT_ROOT, "tasks/training/train_loss_function.py"),
            os.path.join(PROJECT_ROOT, "modules/perform/perform_parm.py"),
        ]
    elif dataset_type == "ablation":
        param_files = [
            os.path.join(PROJECT_ROOT, "tasks/ablation/ablation_parm.py"),
            os.path.join(PROJECT_ROOT, "tasks/training/train_loss_function.py"),
        ]
    else:
        param_files = [
            os.path.join(PROJECT_ROOT, "tasks/training/train_parm.py"),
        ]

    if dataset_type == "ablation":
        output_path = os.path.join(results_dir, "ablation_parameters.txt")
    else:
        output_path = os.path.join(results_dir, "regr_parameters.txt")

    try:
        with open(output_path, "w") as output_file:
            for file_path in param_files:
                output_file.write(f"{'=' * 80}\n")
                output_file.write(f"FILE: {file_path}\n")
                output_file.write(f"{'=' * 80}\n\n")

                with open(file_path, "r") as input_file:
                    output_file.write(input_file.read())
                output_file.write("\n\n")

        print(f"Parameter files copied to: {output_path}")
    except Exception as e:
        print(f"Error copying parameter files: {e}")


def save_preprocessed_file(source_path, results_dir):
    """Move preprocessed file to results directory with original filename."""
    import shutil

    original_filename = os.path.basename(source_path)
    dest_path = os.path.join(results_dir, original_filename)
    shutil.move(source_path, dest_path)
    print(f"Saved preprocessed file to: {dest_path}")
    return dest_path


def load_dataset(data_path, dataset_type="conductivity"):
    """Load dataset with flexible column handling for conductivity datasets."""
    from config.datasets import get_dataset_config

    config = get_dataset_config(dataset_type)
    if not config:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dtypes = {}

    max_solvents = config["max_solvents"]
    max_salts = config["max_salts"]

    for i in range(1, max_solvents + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_salts + 1):
        dtypes[f"Salt_{i}"] = str

    if dataset_type == "conductivity":
        dtypes["IC"] = float
        dtypes["LIC"] = float
        dtypes["Temperature"] = float

    dtypes["Reference"] = str
    dtypes["Electrolyte"] = str

    import csv

    dataset = pd.read_csv(
        data_path,
        header=0,
        dtype=dtypes,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
        on_bad_lines="skip",
    )

    return dataset


def get_tokenizer():
    """Get the PerForm tokenizer instance."""
    return Tokenizer()


def load_preprocessed_data(tokenizer=None, custom_path=None):
    """Load dataset and preprocessed data."""

    try:
        if custom_path:
            preprocess_path = custom_path
        else:
            if tokenizer is None:
                raise ValueError("Either tokenizer or custom_path must be provided")
            preprocess_path = f"data/preprocessed_mol_info_{tokenizer}.npy"

        preprocessed_data = np.load(preprocess_path, allow_pickle=True).item()

        if "molecules" in preprocessed_data:
            map_dict = preprocessed_data["molecules"]
            input_vocab_size = preprocessed_data.get("vocab_size", 1)
            moiety_key_map = preprocessed_data.get("moiety_key_map", {})
            key_moiety_map = preprocessed_data.get("key_moiety_map", {})
        else:
            map_dict = preprocessed_data
            input_vocab_size = 1 + max(
                max(data["moiety_ids"])
                for data in map_dict.values()
                if "moiety_ids" in data
            )
            moiety_key_map = {}
            key_moiety_map = {}

        return map_dict, input_vocab_size, moiety_key_map, key_moiety_map
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        sys.exit(1)


def feature_engineering(map_dict, molar_ratio_noise, dataset_type, *database):
    """Perform feature engineering on datasets."""
    from config.datasets import get_dataset_config

    config = get_dataset_config(dataset_type)
    if not config:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    tokenizer = get_tokenizer()
    processed_datasets = []

    training_mode = molar_ratio_noise > 0

    for data in database:
        processed_dataset = FeatureEngineering(
            data,
            tokenizer,
            map_dict,
            label_field=TARGET_LABEL,
            dataset_type=dataset_type,
            molar_ratio_noise=molar_ratio_noise,
            training=training_mode,
        )
        processed_datasets.append(processed_dataset)

    return processed_datasets


def create_data_loaders(
    *datasets,
    max_num_molecules,
    batch_size,
    shuffle_default=False,
    sampler_default=None,
    drop_last=True,
):
    """Create data loaders for training, validation, and test sets."""

    collate_function = CollateFn(max_num_mols=max_num_molecules)
    processed_dataloaders = []

    for dataset in datasets:
        processed_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_default,
            sampler=sampler_default,
            collate_fn=collate_function,
            num_workers=0,
            pin_memory=False,
            drop_last=drop_last,
        )
        processed_dataloaders.append(processed_dataset)

    return processed_dataloaders


def evaluate_model(
    model,
    loader,
    device,
    setup_criterion,
    task="Test",
    weights_path=None,
    kfold_val=False,
):
    """Evaluate the model on the given dataset."""

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(normalize_motif_state_dict(checkpoint["model_state_dict"]))

    criterion = setup_criterion()

    model.eval()
    total_loss = 0
    test_results = defaultdict(list)
    test_results["task"] = task

    with torch.no_grad():
        batch_test_predictions = []
        batch_train_truths = []
        batch_train_row_ids = []

        if not kfold_val:
            max_batch = tqdm(loader, desc="Evaluation", leave=False)
        else:
            max_batch = loader

        for batch in max_batch:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            y = batch["y"].reshape(-1)
            row_ids = batch["row_id"] if "row_id" in batch else torch.zeros_like(y)

            outputs = model(batch)

            loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            batch_test_predictions.extend(outputs.detach().cpu().numpy())
            batch_train_truths.extend(y.detach().cpu().numpy())
            batch_train_row_ids.extend(row_ids.detach().cpu().numpy())

    test_loss = total_loss / len(loader.dataset)
    test_results["loss"] = test_loss
    test_results["predictions"] = np.array(batch_test_predictions)
    test_results["truths"] = np.array(batch_train_truths)
    test_results["row_ids"] = np.array(batch_train_row_ids)

    return test_results


def run_inference(model, loader, device, weights_path=None):
    """Run inference and return predictions and Row_ID."""

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(normalize_motif_state_dict(checkpoint["model_state_dict"]))

    model.eval()
    inference_results = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch)

            inference_results["predictions"].extend(outputs.detach().cpu().numpy())
            inference_results["row_ids"].extend(batch["row_id"].cpu().numpy())

    inference_results["predictions"] = np.array(inference_results["predictions"])
    inference_results["row_ids"] = np.array(inference_results["row_ids"])

    return inference_results


def calculate_metrics(*results, show_results=False, show_prediction_ranges=False):
    """Calculate RMSE and R² metrics for a fold."""

    metrics = {}
    print(f"METRICS RESULTS:") if show_results else None

    for result in results:
        task = result["task"]
        y_true = result["truths"]
        y_pred = result["predictions"]

        mse = mean_squared_error(y_true, y_pred)
        task_metrics = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_true, y_pred),
            "pearson_r": pearsonr(y_true, y_pred)[0],
        }
        metrics[task] = task_metrics

        if show_results:
            print(f"\n{task}")
            print(
                f"\tLoss: {result['loss']:.4f}, MSE: {task_metrics['mse']:.3f}, RMSE: {task_metrics['rmse']:.3f}, R2: {task_metrics['r2']:.3f}, Pearson r: {task_metrics['pearson_r']:.3f}"
            )

        if show_prediction_ranges:
            print(f"\tPrediction ranges:\t[{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
            print(f"\tTarget ranges:\t\t[{np.min(y_true):.2f}, {np.max(y_true):.2f}]")

    return metrics


def save_predictions_to_csv(*results, save_path=None):
    """Save predictions and actual values to a CSV file."""

    save_predictions_path = os.path.join(save_path, "regr_predictions.csv")
    combined_df = pd.DataFrame()

    for result in results:
        task = result["task"]
        row_ids = result["row_ids"]
        truths = result["truths"]
        predictions = result["predictions"]

        df = pd.DataFrame(
            {
                "Task": task,
                "Row_ID": row_ids,
                "Truth": truths,
                "Prediction": predictions,
            }
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df["Absolute_Error"] = abs(
        combined_df["Truth"] - combined_df["Prediction"]
    )
    combined_df["Percent_Error"] = (
        combined_df["Absolute_Error"] / combined_df["Truth"]
    ) * 100

    combined_df.to_csv(save_predictions_path, index=False)
    print(f"Prediction data saved to {save_predictions_path}\n")


def save_inference_predictions(results, save_path=None):
    """Save inference predictions to CSV."""

    save_predictions_path = os.path.join(save_path, "inference_predictions.csv")
    save_high_ic_path = os.path.join(save_path, "high_ic_predictions.csv")

    row_ids = results["row_ids"]
    predictions = results["predictions"]
    predictions_array = np.array(predictions)
    ic_values = 10**predictions_array

    df = pd.DataFrame(
        {
            "Row_ID": row_ids,
            "Predicted LIC": predictions_array.round(5),
            "Predicted IC (mS/cm)": ic_values.round(5),
        }
    )

    df = df.sort_values(by="Predicted LIC", ascending=False)
    df_high_ic = df[df["Predicted IC (mS/cm)"] >= 10.0]

    df.to_csv(save_predictions_path, index=False)
    df_high_ic.to_csv(save_high_ic_path, index=False)

    print(f"Inference prediction data saved to:\n\t{save_predictions_path}")
    print(f"High IC predictions (>=10 mS/cm) saved to:\n\t{save_high_ic_path}")
    print(f"Number of formulations: {len(df)}\n")


def setup_logging(results_dir, filename="task_terminal_output.log"):
    """
    Set up logging to capture terminal output to a file in the results directory.
    """
    import logging
    import sys

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
            self.terminal = sys.stdout

        def write(self, buf):
            self.terminal.write(buf)
            self.linebuf += buf
            if buf.endswith("\n"):
                self.logger.log(self.log_level, self.linebuf.rstrip())
                self.linebuf = ""

        def flush(self):
            if self.linebuf:
                self.logger.log(self.log_level, self.linebuf)
                self.linebuf = ""
            self.terminal.flush()

    sys.stdout = StreamToLogger(logger)

    print(f"Logging experiment output to: {log_file}")

    return logger, file_handler


def stop_logging(logger, file_handler):
    """
    Stop logging and restore normal stdout.
    """
    import sys

    sys.stdout = sys.__stdout__

    logger.removeHandler(file_handler)
    file_handler.close()

    print(f"Logging stopped and file closed")


def extract_training_mappings_from_checkpoint(checkpoint_path):
    """Extract moiety mappings from enhanced model checkpoint"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint = dict(checkpoint)
    checkpoint["architecture"] = normalize_motif_architecture(
        checkpoint["architecture"]
    )
    moiety_key_map = checkpoint["moiety_key_map"]
    key_moiety_map = checkpoint["key_moiety_map"]
    vocab_size = checkpoint["vocab_size"]
    return moiety_key_map, key_moiety_map, vocab_size, checkpoint


def process_single_smiles_worker(args):
    """
    Worker function for parallel SMILES processing.

    Args:
        args: Tuple of (smiles, moiety_key_map, key_moiety_map)

    Returns:
        Tuple of (smiles, processed_data_dict) or (smiles, None) if error
    """
    smiles, moiety_key_map, key_moiety_map = args

    try:
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.utils import get_tokenizer
        from modules.moiety_features import (
            map_atom_to_moiety,
            build_moiety_edges,
            get_dist_matrix,
            get_adj_matrix,
        )

        tokenizer = get_tokenizer()
        tokenizer.moiety_key_map = moiety_key_map
        tokenizer.key_moiety_map = key_moiety_map

        result = tokenizer.get_moieties(smiles)
        clique_atom_ids, edges, moiety_ids = result[0], result[1], result[2]

        atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
        moiety_edges = build_moiety_edges(edges, atom_to_moiety)

        node_list = list(range(len(clique_atom_ids)))
        moiety_edges = list(moiety_edges)

        dist_matrix = get_dist_matrix(node_list, moiety_edges)
        adj_matrix = get_adj_matrix(node_list, moiety_edges)

        processed_data = {
            "adj_matrix": adj_matrix,
            "dist_matrix": dist_matrix,
            "clique_atom_ids": clique_atom_ids,
            "edges": edges,
            "moiety_ids": moiety_ids,
        }

        return smiles, processed_data

    except Exception as e:
        import numpy as np

        error_data = {
            "adj_matrix": np.array([[]], dtype=np.float32),
            "dist_matrix": np.array([[]], dtype=np.float32),
            "clique_atom_ids": [],
            "edges": [],
            "moiety_ids": [],
        }
        print(f"Error processing mol {smiles}: {e}")
        return smiles, error_data


def create_inference_preprocessing(
    inference_data_path,
    moiety_key_map,
    key_moiety_map,
    vocab_size,
    dataset_type,
    results_dir,
    process_id,
):
    """Create inference NPY using training moiety mappings"""
    print("Creating inference preprocessing with training mappings...")

    config = get_dataset_config(dataset_type)
    max_num_solvent_molecules = config["max_solvents"]
    max_num_salt_molecules = config["max_salts"]

    dtypes = {}
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    dtypes["Row_ID"] = int

    df = pd.read_csv(inference_data_path, dtype=dtypes)

    unique_SMILES_in_dataset = set()
    for col in [f"Solvent_{i}" for i in range(1, max_num_solvent_molecules + 1)] + [
        f"Salt_{i}" for i in range(1, max_num_salt_molecules + 1)
    ]:
        smiles_values = df[col].dropna().astype(str)
        smiles_values = smiles_values[smiles_values != "nan"]
        for smiles_string in smiles_values:
            if ":" in smiles_string:
                smiles_part = smiles_string.split(":")[0].strip()
                unique_SMILES_in_dataset.add(smiles_part)

    all_mols_dict = {
        smi: {
            "adj_matrix": None,
            "dist_matrix": None,
            "clique_atom_ids": None,
            "edges": None,
            "moiety_ids": None,
        }
        for smi in unique_SMILES_in_dataset
    }

    unique_smiles_list = list(unique_SMILES_in_dataset)
    args_list = [
        (smiles, moiety_key_map, key_moiety_map) for smiles in unique_smiles_list
    ]

    num_cores = min(90, mp.cpu_count() - 6, len(unique_smiles_list))
    print(f"Using {num_cores} cores for parallel SMILES processing...")

    try:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_smiles_worker, args_list),
                    total=len(args_list),
                    desc=f"Preprocessing {len(args_list)} molecules (parallel)",
                )
            )

        for smiles, processed_data in results:
            if processed_data is not None:
                all_mols_dict[smiles] = processed_data

        print(
            f"Successfully processed {len(results)} molecules using {num_cores} cores"
        )

    except Exception as e:
        print(f"Multiprocessing failed, falling back to sequential processing: {e}")
        tokenizer = get_tokenizer()
        tokenizer.moiety_key_map = moiety_key_map
        tokenizer.key_moiety_map = key_moiety_map

        for smiles in tqdm(
            unique_SMILES_in_dataset,
            desc="Preprocessing inference molecules (sequential fallback)",
        ):
            try:
                result = tokenizer.get_moieties(smiles)
                clique_atom_ids, edges, moiety_ids = result[0], result[1], result[2]

                atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
                moiety_edges = build_moiety_edges(edges, atom_to_moiety)

                node_list = list(range(len(clique_atom_ids)))
                moiety_edges = list(moiety_edges)

                dist_matrix = get_dist_matrix(node_list, moiety_edges)
                adj_matrix = get_adj_matrix(node_list, moiety_edges)

                all_mols_dict[smiles]["adj_matrix"] = adj_matrix
                all_mols_dict[smiles]["dist_matrix"] = dist_matrix
                all_mols_dict[smiles]["clique_atom_ids"] = clique_atom_ids
                all_mols_dict[smiles]["edges"] = edges
                all_mols_dict[smiles]["moiety_ids"] = moiety_ids

            except Exception as e:
                print(f"Error processing mol {smiles}: {e}")
                all_mols_dict[smiles]["adj_matrix"] = np.array([[]], dtype=np.float32)
                all_mols_dict[smiles]["dist_matrix"] = np.array([[]], dtype=np.float32)
                all_mols_dict[smiles]["clique_atom_ids"] = []
                all_mols_dict[smiles]["edges"] = []
                all_mols_dict[smiles]["moiety_ids"] = []

    preprocessed_data = {
        "molecules": all_mols_dict,
        "moiety_key_map": moiety_key_map,
        "key_moiety_map": key_moiety_map,
        "vocab_size": vocab_size,
    }

    output_path = os.path.join(
        results_dir, f"preprocessed_mol_info_inference_{process_id}.npy"
    )
    np.save(output_path, preprocessed_data)
    print(f"Inference preprocessing saved to: {output_path}")
    return output_path


def load_pretrained_ic_model(model, pretrained_path):
    """Load pretrained IC model weights."""
    checkpoint = torch.load(pretrained_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(normalize_motif_state_dict(state_dict), strict=True)
    print(f"Loaded pretrained weights from: {pretrained_path}\n")
    return model
