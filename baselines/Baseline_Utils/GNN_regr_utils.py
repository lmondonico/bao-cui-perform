"""
This module contains utility functions for GNN-based regression tasks.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from config.paths import DATA_DIR

from .GNN_modules import MoleculeEncoder, BaselineFormulationModel


repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from tasks.training.train_parm import (
    max_num_solvent_molecules,
    max_num_salt_molecules,
)
from tasks.training.train_loss_function import ConductivityMSELoss


GNN_TYPES = ["AttentiveFP", "Molformer"]
DEFAULT_GNN_TYPE = "AttentiveFP"
NODE_FEATURES_DIM = 6
EDGE_FEATURES_DIM = 3
GNN_HIDDEN_DIM = 128
GNN_NUM_LAYERS = 3
GAT_HEADS = 4


def get_device(show_device=True):
    """Check if GPU is available."""
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if show_device:
            print(f"\nUsing device: {device}\n")
        return device
    except Exception as e:
        print(f"Error detecting device: {e}")
        sys.exit(1)


def get_running_time(start_time):
    """Calculate the running time of a script."""
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60

    print("/" * 30)
    print(f"PROCESS COMPLETED in {elapsed_minutes:.2f} minutes")
    print("/" * 30 + "\n")


def load_dataset(data_path):
    """Load dataset for GNN models."""
    dtypes = {}

    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Solvent_{i}"] = str
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Salt_{i}"] = str
    for i in range(1, max_num_solvent_molecules + 1):
        dtypes[f"Conc_Solvent_{i}"] = float
    for i in range(1, max_num_salt_molecules + 1):
        dtypes[f"Conc_Salt_{i}"] = float

    dtypes["Row_ID"] = int
    dtypes["IC"] = float
    dtypes["LIC"] = float
    dtypes["Temperature"] = float

    dataset = pd.read_csv(data_path, header=0, dtype=dtypes)
    no_formulations = len(dataset.dropna(how="all"))
    print(f"Number of rows in the dataset: {no_formulations}")

    return dataset, no_formulations


def _parse_molecule_column(value):
    """
    Parse a single molecule column value in "SMILES:ratio" format.

    Args:
        value: String in format "SMILES:ratio" or NaN/empty

    Returns:
        tuple: (smiles_str, concentration_float) or (None, None) for empty values
    """
    if pd.isna(value) or value == "" or str(value).strip() == "":
        return None, None

    value_str = str(value).strip()
    parts = value_str.rsplit(":", 1)

    if len(parts) == 2:
        smiles = parts[0].strip()
        try:
            concentration = float(parts[1].strip())
            return smiles, concentration
        except ValueError:
            return value_str, None
    else:
        return value_str, None


def parse_smiles_ratio_format(df, max_solvents=5, max_salts=3):
    """
    Parse DataFrame with "SMILES:ratio" format into separate SMILES and concentration columns.

    Args:
        df: DataFrame with columns like "Solvent_1" containing "SMILES:ratio" strings
        max_solvents: Maximum number of solvent columns to process
        max_salts: Maximum number of salt columns to process

    Returns:
        DataFrame with separate Solvent_i, Conc_Solvent_i, Salt_i, Conc_Salt_i columns
    """
    parsed_df = df.copy()

    for i in range(1, max_solvents + 1):
        solvent_col = f"Solvent_{i}"
        conc_col = f"Conc_Solvent_{i}"

        if solvent_col in parsed_df.columns:
            smiles_list = []
            conc_list = []

            for value in parsed_df[solvent_col]:
                smiles, conc = _parse_molecule_column(value)
                smiles_list.append(smiles)
                conc_list.append(conc if conc is not None else np.nan)

            parsed_df[solvent_col] = smiles_list
            parsed_df[conc_col] = conc_list

    for i in range(1, max_salts + 1):
        salt_col = f"Salt_{i}"
        conc_col = f"Conc_Salt_{i}"

        if salt_col in parsed_df.columns:
            smiles_list = []
            conc_list = []

            for value in parsed_df[salt_col]:
                smiles, conc = _parse_molecule_column(value)
                smiles_list.append(smiles)
                conc_list.append(conc if conc is not None else np.nan)

            parsed_df[salt_col] = smiles_list
            parsed_df[conc_col] = conc_list

    return parsed_df


def load_split_datasets(dataset_name, splits_base_dir=None):
    """
    Load train, validation, and test CSV files for a given dataset and parse SMILES:ratio format.

    Args:
        dataset_name: Name of the dataset (e.g., "calisol")
        splits_base_dir: Base directory containing splits-* folders (defaults to DATA_DIR from config)

    Returns:
        tuple: (train_df, val_df, test_df) with parsed separate columns
    """
    if splits_base_dir is None:
        splits_base_dir = DATA_DIR

    splits_dir = os.path.join(splits_base_dir, f"splits-{dataset_name}")

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    print(f"\nLoading dataset splits from: {splits_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    train_df = parse_smiles_ratio_format(train_df)
    val_df = parse_smiles_ratio_format(val_df)
    test_df = parse_smiles_ratio_format(test_df)

    return train_df, val_df, test_df


class GNNFormulationDataset(Dataset):
    """Dataset for GNN-based formulation property prediction."""

    def __init__(
        self,
        data_df,
        molecule_encoder,
        label_field,
        max_num_solvent_molecules=5,
        max_num_salt_molecules=3,
        concentration_noise=0.03,
    ):
        """
        Args:
            data_df: DataFrame containing formulation data
            molecule_encoder: MoleculeEncoder instance
            label_field: Target property column name
            max_num_solvent_molecules: Maximum number of solvent molecules
            max_num_salt_molecules: Maximum number of salt molecules
            concentration_noise: Amount of noise to add to concentrations during training
        """
        self.data_df = data_df.copy()
        self.encoder = molecule_encoder
        self.label_field = label_field
        self.max_num_solvent_molecules = max_num_solvent_molecules
        self.max_num_salt_molecules = max_num_salt_molecules
        self.max_num_molecules = max_num_solvent_molecules + max_num_salt_molecules
        self.concentration_noise = concentration_noise

        self.processed_data = self._process_data()

    def _add_concentration_noise(self, concentration, concentration_noise_range=None):
        """Add noise to concentration value."""
        if concentration_noise_range is None:
            concentration_noise_range = self.concentration_noise
        if concentration <= 0 or concentration_noise_range == 0:
            return concentration

        min_allowed = concentration * (1 - concentration_noise_range)
        max_allowed = concentration * (1 + concentration_noise_range)
        return np.random.uniform(min_allowed, max_allowed)

    def _process_data(self):
        """Process and store data tuples with molecule SMILES and concentrations."""
        processed_data = []

        for idx, row in self.data_df.iterrows():
            smiles_list = []
            concentrations = []

            for i in range(1, self.max_num_solvent_molecules + 1):
                mol_col = f"Solvent_{i}"
                conc_col = f"Conc_Solvent_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    smiles_list.append(row[mol_col])
                    if conc_col in row and pd.notna(row[conc_col]):
                        concentrations.append(float(row[conc_col]))
                    else:
                        concentrations.append(0.0)
                else:
                    smiles_list.append(None)
                    concentrations.append(0.0)

            for i in range(1, self.max_num_salt_molecules + 1):
                mol_col = f"Salt_{i}"
                conc_col = f"Conc_Salt_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    smiles_list.append(row[mol_col])
                    if conc_col in row and pd.notna(row[conc_col]):
                        concentrations.append(float(row[conc_col]))
                    else:
                        concentrations.append(0.0)
                else:
                    smiles_list.append(None)
                    concentrations.append(0.0)

            label = row[self.label_field]
            row_id = row["Row_ID"]
            temperature = row["Temperature"]

            processed_data.append(
                {
                    "smiles": smiles_list,
                    "concentrations": concentrations,
                    "label": label,
                    "elf_code": row_id,
                    "temperature": temperature,
                }
            )

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        noisy_concentrations = [
            self._add_concentration_noise(c) for c in item["concentrations"]
        ]

        mol_graphs = []
        for i, smiles in enumerate(item["smiles"]):
            if smiles is not None:
                if hasattr(self.encoder, "encode_smiles"):
                    mol_graphs.append((i, smiles))
                else:
                    graph = self.encoder.smiles_to_graph(smiles)
                    if graph is None:
                        raise ValueError(f"Invalid molecular graph: row={item['elf_code']}, smiles={smiles}")
                    mol_graphs.append((i, graph))

        return {
            "smiles": item["smiles"],
            "concentrations": noisy_concentrations,
            "mol_graphs": mol_graphs,
            "label": item["label"],
            "elf_code": item["elf_code"],
            "temperature": item["temperature"],
        }


def collate_gnn_formulations(batch):
    """Collate function for batching GNN formulation data."""
    batch_smiles = [item["smiles"] for item in batch]
    batch_mol_graphs = [item["mol_graphs"] for item in batch]
    batch_labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    batch_elf_codes = torch.tensor(
        [item["elf_code"] for item in batch], dtype=torch.long
    )
    batch_temperatures = torch.tensor(
        [item["temperature"] for item in batch], dtype=torch.float32
    )

    processed_mol_graphs = []
    aligned_concentrations = []
    for item, formulation_graphs in zip(batch, batch_mol_graphs):
        sorted_graphs = sorted(formulation_graphs, key=lambda x: x[0])
        indices = [index for index, _ in sorted_graphs]
        expected = {i for i, value in enumerate(item["concentrations"]) if value > 0}
        if not expected.issubset(indices):
            raise ValueError(f"Missing positive-amount graph for row {item['elf_code']}")
        values = [item["concentrations"][i] for i in indices]
        values += [0.0] * (len(item["concentrations"]) - len(values))
        aligned_concentrations.append(values)
        processed_mol_graphs.append([graph for _, graph in sorted_graphs])

    return {
        "smiles": batch_smiles,
        "concentrations": torch.tensor(aligned_concentrations, dtype=torch.float32),
        "mol_graphs": processed_mol_graphs,
        "labels": batch_labels,
        "elf_codes": batch_elf_codes,
        "temperature": batch_temperatures,
    }


def dataset_partition(dataset, train_size, val_size, random_seed=42):
    """Partition dataset into train, validation, and test sets."""
    if random_seed:
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )
    else:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    train_end = int(train_size * len(dataset))
    val_end = train_end + int(val_size * len(dataset))

    train_df = dataset.iloc[:train_end]
    val_df = dataset.iloc[train_end:val_end]
    test_df = dataset.iloc[val_end:]

    return train_df, val_df, test_df


def dataset_partition_kfold(dataset, kfold_test_size, random_seed=42):
    """Partition dataset into k-folds with held-out test set."""
    if random_seed:
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )
    else:
        dataset = dataset.sample(frac=1).reset_index(drop=True)

    train_val_idx = int((1 - kfold_test_size) * len(dataset))

    train_val_data = dataset.iloc[:train_val_idx]
    test_data = dataset.iloc[train_val_idx:]

    print(f"\nTotal dataset size:\t{len(dataset)}")
    print(f"Train+Val size:\t\t{len(train_val_data)}")
    print(f"Test size:\t\t{len(test_data)}")

    return train_val_data, test_data


def create_gnn_datasets(*dataframes, target, molecule_encoder=None):
    """Create GNN datasets from dataframes."""
    if molecule_encoder is None:
        molecule_encoder = MoleculeEncoder()

    datasets = []
    for df in dataframes:
        dataset = GNNFormulationDataset(
            df,
            molecule_encoder,
            label_field=target,
            max_num_solvent_molecules=max_num_solvent_molecules,
            max_num_salt_molecules=max_num_salt_molecules,
            concentration_noise=0.03,
        )
        datasets.append(dataset)

    return datasets


def create_gnn_dataloaders(
    *datasets, batch_size=32, shuffle_default=False, sampler_default=None
):
    """Create data loaders for GNN datasets."""
    loaders = []

    for dataset in datasets:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_default,
            sampler=sampler_default,
            collate_fn=collate_gnn_formulations,
        )
        loaders.append(loader)

    return loaders


def initialize_gnn_model(
    gnn_type=DEFAULT_GNN_TYPE, device=None, molformer_encoder=None
):
    """Initialize a GNN-based formulation model."""
    if gnn_type not in GNN_TYPES:
        print(
            f"Warning: GNN type {gnn_type} not recognized. Using {DEFAULT_GNN_TYPE} instead."
        )
        gnn_type = DEFAULT_GNN_TYPE

    model = BaselineFormulationModel(
        gnn_type=gnn_type,
        node_features_dim=NODE_FEATURES_DIM,
        edge_features_dim=EDGE_FEATURES_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=1,
        num_layers=GNN_NUM_LAYERS,
        heads=GAT_HEADS,
        max_molecules=max_num_solvent_molecules + max_num_salt_molecules,
        molformer_encoder=molformer_encoder,
    )

    if device:
        model = model.to(device)

    return model


def create_baseline_model(
    gnn_type=DEFAULT_GNN_TYPE,
    hidden_dim=GNN_HIDDEN_DIM,
    num_layers=GNN_NUM_LAYERS,
    heads=GAT_HEADS,
    molformer_encoder=None,
):
    """Create a baseline formulation model with specified parameters."""
    model = BaselineFormulationModel(
        gnn_type=gnn_type,
        node_features_dim=NODE_FEATURES_DIM,
        edge_features_dim=EDGE_FEATURES_DIM,
        hidden_dim=hidden_dim,
        out_dim=1,
        num_layers=num_layers,
        heads=heads,
        max_molecules=max_num_solvent_molecules + max_num_salt_molecules,
        molformer_encoder=molformer_encoder,
    )
    return model


def setup_gnn_optimizer(model):
    """Set up optimizer for GNN model."""
    learning_rate = 1e-4
    weight_decay = 0.001
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def setup_gnn_scheduler(optimizer, train_loader, num_epochs, max_lr=1e-3):
    """Set up learning rate scheduler for GNN model."""
    total_steps = len(train_loader) * num_epochs
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=100.0,
    )


def train_gnn_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    save_weight_file,
    plot_save_path=None,
    show_epoch_info=False,
    kfold_train=False,
    patience=10,
):
    """Train a GNN-based model with early stopping."""
    criterion = ConductivityMSELoss()
    optimizer = setup_gnn_optimizer(model)
    scheduler = setup_gnn_scheduler(optimizer, train_loader, num_epochs)

    train_results = defaultdict(list)
    train_results["task"] = "Training"
    val_results = defaultdict(list)
    val_results["task"] = "Validation"
    best_val_loss = float("inf")

    patience_counter = 0
    early_stop = False

    if kfold_train:
        max_epochs = tqdm(range(num_epochs), desc="Training", leave=False)
    else:
        print("\nStarting GNN training...")
        max_epochs = range(num_epochs)

    loss_history = []
    for epoch in max_epochs:
        model.train()

        total_loss = 0
        epoch_train_predictions = []
        epoch_train_truths = []
        epoch_train_elf_codes = []

        if not kfold_train:
            print(f"\n---Epoch {epoch + 1}/{num_epochs}")
            max_batch = tqdm(train_loader, desc="Training", leave=False)
        else:
            max_batch = train_loader

        for batch in max_batch:
            labels = (
                batch["labels"].to(device).view(-1, 1)
            )
            elf_codes = batch["elf_codes"]
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]
            temperature = batch["temperature"].to(device)

            optimizer.zero_grad()
            outputs = model(mol_graphs, concentrations, temperature)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_train_predictions.extend(
                outputs.detach().cpu().numpy().flatten()
            )
            epoch_train_truths.extend(
                labels.detach().cpu().numpy().flatten()
            )
            epoch_train_elf_codes.extend(elf_codes.detach().cpu().numpy())

        epoch_train_loss = total_loss / len(train_loader.dataset)

        epoch_val_results = evaluate_gnn_model(
            model, val_loader, device, kfold_val=kfold_train
        )
        (
            epoch_val_predictions,
            epoch_val_targets,
            epoch_val_elf_codes,
            epoch_val_loss,
        ) = (
            epoch_val_results["predictions"],
            epoch_val_results["truths"],
            epoch_val_results["elf_codes"],
            epoch_val_results["loss"],
        )

        if show_epoch_info:
            print(
                f"Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, LR: {current_lr:.2e}"
            )

        loss_history.append({"Epoch": epoch + 1, "Training": epoch_train_loss, "Validation": epoch_val_loss})
        pd.DataFrame(loss_history).to_csv(os.path.join(os.path.dirname(save_weight_file), "loss_history.csv"), index=False)
        if not np.isfinite(epoch_train_loss) or not np.isfinite(epoch_val_loss):
            raise ValueError("Nonfinite baseline loss")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0

            train_results["loss"] = epoch_train_loss
            train_results["predictions"] = np.array(epoch_train_predictions)
            train_results["truths"] = np.array(epoch_train_truths)
            train_results["elf_codes"] = np.array(epoch_train_elf_codes)
            val_results["loss"] = best_val_loss
            val_results["predictions"] = epoch_val_predictions
            val_results["truths"] = epoch_val_targets
            val_results["elf_codes"] = epoch_val_elf_codes

            torch.save(model.state_dict(), save_weight_file)

            if show_epoch_info:
                print(f"Validation loss improved, model weights saved")
        else:
            patience_counter += 1
            if show_epoch_info:
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                if show_epoch_info:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                early_stop = True
                break

    if plot_save_path:
        history = pd.DataFrame(loss_history)
        history.plot(x="Epoch", y=["Training", "Validation"])
        plt.ylabel("MSE")
        plt.savefig(plot_save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return train_results, val_results


def evaluate_gnn_model(
    model, loader, device, task="Test", weights_path=None, kfold_val=False
):
    """Evaluate a GNN-based model on the given dataset."""
    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )

    criterion = ConductivityMSELoss()

    model.eval()
    total_loss = 0
    test_results = defaultdict(list)
    test_results["task"] = task

    with torch.no_grad():
        batch_test_predictions = []
        batch_test_truths = []
        batch_test_elf_codes = []

        if not kfold_val:
            max_batch = tqdm(loader, desc=f"Evaluating ({task})", leave=False)
        else:
            max_batch = loader

        for batch in max_batch:
            labels = (
                batch["labels"].to(device).view(-1, 1)
            )
            elf_codes = batch["elf_codes"]
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]
            temperature = batch["temperature"].to(device)

            outputs = model(mol_graphs, concentrations, temperature)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            batch_test_predictions.extend(
                outputs.detach().cpu().numpy().flatten()
            )
            batch_test_truths.extend(
                labels.detach().cpu().numpy().flatten()
            )
            batch_test_elf_codes.extend(elf_codes.detach().cpu().numpy())

    test_loss = total_loss / len(loader.dataset)
    test_results["loss"] = test_loss
    test_results["predictions"] = np.array(batch_test_predictions)
    test_results["truths"] = np.array(batch_test_truths)
    test_results["elf_codes"] = np.array(batch_test_elf_codes)

    return test_results


def run_gnn_inference(model, loader, device, weights_path=None):
    """Run inference with a GNN-based model."""
    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )

    model.eval()
    inference_results = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            concentrations = batch["concentrations"].to(device)
            mol_graphs = batch["mol_graphs"]
            elf_codes = batch["elf_codes"]
            temperature = batch["temperature"].to(device)

            outputs = model(mol_graphs, concentrations, temperature)

            inference_results["predictions"].extend(
                outputs.detach().cpu().numpy().flatten()
            )
            inference_results["elf_codes"].extend(elf_codes.cpu().numpy())

    inference_results["predictions"] = np.array(inference_results["predictions"])
    inference_results["elf_codes"] = np.array(inference_results["elf_codes"])

    return inference_results


def calculate_metrics(*results, show_results=False, show_prediction_ranges=False):
    """Calculate RMSE and R² metrics for model evaluation."""
    metrics = {}
    print(f"METRICS RESULTS:") if show_results else None

    for result in results:
        task = result["task"]
        y_true = result["truths"]
        y_pred = result["predictions"]

        mse = mean_squared_error(y_true, y_pred)
        task_metrics = {
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_true, y_pred),
            "pearson_r": stats.pearsonr(y_true, y_pred)[0],
        }
        metrics[task] = task_metrics

        if show_results:
            print(f"\n{task}")
            print(
                f"\tLoss: {result['loss']:.4f}, MSE: {mse:.3f}, RMSE: {task_metrics['rmse']:.3f}, "
                f"R2: {task_metrics['r2']:.3f}, Pearson r: {task_metrics['pearson_r']:.3f}"
            )

        if show_prediction_ranges:
            print(f"\tPrediction ranges:\t[{np.min(y_pred):.3f}, {np.max(y_pred):.3f}]")
            print(f"\tTarget ranges:\t\t[{np.min(y_true):.3f}, {np.max(y_true):.3f}]")

    return metrics


def save_gnn_predictions_to_csv(*results, save_path=None, gnn_type=DEFAULT_GNN_TYPE):
    """Save GNN model predictions to CSV."""
    if save_path is None:
        save_path = os.getcwd()

    save_predictions_path = os.path.join(
        save_path, f"{gnn_type.lower()}_predictions.csv"
    )
    combined_df = pd.DataFrame()

    for result in results:
        task = result["task"]
        elf_codes = result["elf_codes"]
        truths = result["truths"]
        predictions = result["predictions"]

        df = pd.DataFrame(
            {
                "Task": task,
                "Elf_Code": elf_codes,
                "Truth": truths,
                "Prediction": predictions,
            }
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df["Absolute_Error"] = abs(
        combined_df["Truth"] - combined_df["Prediction"]
    )
    combined_df["Percent_Error"] = (
        combined_df["Absolute_Error"] / combined_df["Truth"].clip(lower=1e-5)
    ) * 100

    combined_df.to_csv(save_predictions_path, index=False)
    print(f"GNN prediction data saved to {save_predictions_path}\n")

    return combined_df


def save_gnn_predictions_simple_csv(
    *results, save_path=None, gnn_type=DEFAULT_GNN_TYPE
):
    """
    Save GNN model predictions to CSV in simplified format matching train_kfold.py.
    Columns: Task, Truth, Prediction (compatible with plot_parity_from_csv.py)
    """
    if save_path is None:
        save_path = os.getcwd()

    save_predictions_path = os.path.join(save_path, "regr_predictions.csv")
    combined_df = pd.DataFrame()

    for result in results:
        task = result["task"]
        truths = result["truths"]
        predictions = result["predictions"]

        df = pd.DataFrame(
            {
                "Task": task,
                "Truth": truths,
                "Prediction": predictions,
            }
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(save_predictions_path, index=False)
    print(f"Predictions saved to {save_predictions_path}\n")

    return combined_df


def plot_gnn_results(
    train_results,
    val_results,
    test_results,
    metrics,
    save_path,
    gnn_type=DEFAULT_GNN_TYPE,
):
    """Create GNN evaluation plots with metrics."""
    train_pred = train_results["predictions"]
    train_true = train_results["truths"]
    val_pred = val_results["predictions"]
    val_true = val_results["truths"]
    test_pred = test_results["predictions"]
    test_true = test_results["truths"]

    train_rmse = metrics["Training"]["rmse"]
    train_r2 = metrics["Training"]["r2"]
    val_rmse = metrics["Validation"]["rmse"]
    val_r2 = metrics["Validation"]["r2"]
    test_rmse = metrics["Test"]["rmse"]
    test_r2 = metrics["Test"]["r2"]

    metrics_text_pos = (0.85, 0.2)
    histogram_pos = [0.137, 0.74, 0.25, 0.2]
    axis_tick_size = 14
    axis_title_size = 18
    plot_title_size = 20
    histogram_tick_size = 12
    plt.figure(figsize=(12, 10))
    plt.scatter(train_true, train_pred, color="orange", alpha=0.5, label="Training")
    plt.scatter(val_true, val_pred, color="blue", alpha=0.5, label="Validation")
    plt.scatter(test_true, test_pred, color="green", alpha=0.5, label="Test")

    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    metrics_text = (
        f"Training:\n RMSE: {train_rmse:.3f}\n R2: {train_r2:.3f}\n\n"
        f"Validation:\n RMSE: {val_rmse:.3f}\n R2: {val_r2:.3f}\n\n"
        f"Test:\n RMSE: {test_rmse:.3f}\n R2: {test_r2:.3f}"
    )
    plt.gcf().text(
        metrics_text_pos[0],
        metrics_text_pos[1],
        metrics_text,
        fontsize=axis_tick_size,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    dot_positions = [(0.11, "orange"), (0.023, "blue"), (-0.065, "green")]
    for y_offset, color in dot_positions:
        plt.gcf().text(
            metrics_text_pos[0] - 0.02,
            metrics_text_pos[1] + y_offset,
            "●",
            color=color,
            fontsize=axis_tick_size,
            va="center",
            ha="center",
        )

    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)
    plt.title(
        f"{gnn_type} -- Actual vs Predicted Values with Metrics",
        fontsize=plot_title_size,
    )
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)

    inset_axes = plt.axes(histogram_pos)
    inset_axes.hist(all_pred, bins=50, alpha=0.75, color="gray")
    inset_axes.tick_params(axis="both", labelsize=histogram_tick_size)
    inset_axes.set_xlabel("Predictions", fontsize=histogram_tick_size)
    inset_axes.set_ylabel("Frequency", fontsize=histogram_tick_size)

    plt.tight_layout()
    gnn_plot_file_name = f"{gnn_type.lower()}_parity_plot.png"
    gnn_plot_full_save_path = os.path.join(save_path, gnn_plot_file_name)
    plt.savefig(gnn_plot_full_save_path)
    plt.close()

    print(f"GNN results plot saved to: {gnn_plot_full_save_path}")


def plot_gnn_results_standardized(
    train_results,
    val_results,
    test_results,
    metrics,
    save_path,
    gnn_type=DEFAULT_GNN_TYPE,
):
    """
    Create standardized parity plot matching plot_parity_from_csv.py format.
    - Metrics in title (R² and RMSE)
    - DPI 900
    - No histogram inset
    - Simple legend
    - Colors: orange (train), blue (val), green (test)
    """
    train_pred = train_results["predictions"]
    train_true = train_results["truths"]
    val_pred = val_results["predictions"]
    val_true = val_results["truths"]
    test_pred = test_results["predictions"]
    test_true = test_results["truths"]

    train_rmse = metrics["Training"]["rmse"]
    train_r2 = metrics["Training"]["r2"]
    val_rmse = metrics["Validation"]["rmse"]
    val_r2 = metrics["Validation"]["r2"]
    test_rmse = metrics["Test"]["rmse"]
    test_r2 = metrics["Test"]["r2"]

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

    title_line1 = f"{gnn_type} Parity Plot (Actual vs Predicted)"
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

    gnn_plot_file_name = f"{gnn_type.lower()}_parity_plot.png"
    gnn_plot_full_save_path = os.path.join(save_path, gnn_plot_file_name)
    plt.savefig(gnn_plot_full_save_path, dpi=900, bbox_inches="tight")
    plt.close()

    print(f"Standardized parity plot saved to: {gnn_plot_full_save_path}")


def plot_kfold_cross_results(fold_metrics, save_path, gnn_type=DEFAULT_GNN_TYPE):
    """Plot cross-validation results focusing on RMSE and R2."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_to_plot = ["rmse", "r2"]
    data_to_plot = [
        [fold_metrics[fold]["metrics"][metric] for fold in fold_metrics.keys()]
        for metric in metrics_to_plot
    ]

    ax1.boxplot(data_to_plot, labels=["RMSE", "R²"])
    ax1.set_title(f"{gnn_type} - Distribution of Metrics Across Folds")
    ax1.grid(True)

    all_true = []
    all_pred = []
    for fold in fold_metrics.keys():
        all_true.extend(fold_metrics[fold]["truths"])
        all_pred.extend(fold_metrics[fold]["predictions"])

    ax2.scatter(all_true, all_pred, alpha=0.5)
    ax2.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], "r--")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title(f"{gnn_type} - Predicted vs Actual Values (All Folds)")
    ax2.grid(True)

    plt.tight_layout()

    kfold_plot_filename = f"{gnn_type.lower()}_kfold_cross_val_results.png"
    plot_save_path = os.path.join(save_path, kfold_plot_filename)
    plt.savefig(plot_save_path)
    plt.close()
    print(f"\nK-fold cross-validation plot saved to: {plot_save_path}")


def kfold_cross_validation(
    device,
    train_val_dataset,
    test_dataset,
    kfold_n_splits,
    gnn_type=DEFAULT_GNN_TYPE,
    kfold_weights_dir="kfold_weights",
    kfold_patience=10,
    molformer_encoder=None,
):
    """Perform k-fold cross-validation for GNN model."""
    os.makedirs(kfold_weights_dir, exist_ok=True)
    best_model_path = os.path.join(
        kfold_weights_dir, f"{gnn_type.lower()}_best_model.pth"
    )

    def calculate_average_kfold_metrics(all_kfold_metrics):
        """Calculate average metrics across all folds."""
        print("\n")
        print("+" * 50, f"\n{gnn_type} K-Fold Cross-Validation COMPLETE.")
        print("\nAverage Results Across Folds:")
        avg_kfold_metrics = {}
        avg_metrics = {}
        std_metrics = {}
        for metric in ["rmse", "r2"]:
            values = [
                all_kfold_metrics[fold]["metrics"][metric]
                for fold in all_kfold_metrics.keys()
            ]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            print(
                f"\t{metric.upper()}:\t{avg_metrics[metric]:.3f} ± {std_metrics[metric]:.3f}"
            )
        print("+" * 50)
        avg_kfold_metrics["kfold_avg"] = avg_metrics
        avg_kfold_metrics["kfold_std"] = std_metrics

        return avg_kfold_metrics

    all_kfold_metrics = {}
    best_overall_rmse = float("inf")

    kfold = KFold(n_splits=kfold_n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(range(len(train_val_dataset)))
    ):
        print(f"\nFold {fold + 1}/{kfold_n_splits}")

        fold_weights_path = os.path.join(
            kfold_weights_dir, f"{gnn_type.lower()}_fold_{fold + 1}_model.pth"
        )

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = create_gnn_dataloaders(
            train_val_dataset, sampler_default=train_sampler
        )[0]
        val_loader = create_gnn_dataloaders(
            train_val_dataset, sampler_default=val_sampler
        )[0]

        model = initialize_gnn_model(gnn_type, device, molformer_encoder)

        _, fold_val_results = train_gnn_model(
            model,
            train_loader,
            val_loader,
            num_epochs,
            device,
            fold_weights_path,
            kfold_train=True,
            patience=kfold_patience,
        )

        fold_metrics_dict = calculate_metrics(fold_val_results)

        all_kfold_metrics[fold] = {
            "metrics": fold_metrics_dict["Validation"],
            "predictions": fold_val_results["predictions"],
            "truths": fold_val_results["truths"],
        }
        current_rmse = fold_metrics_dict["Validation"]["rmse"]

        print(f"Fold {fold + 1} model saved to {fold_weights_path}")

        if current_rmse < best_overall_rmse:
            best_overall_rmse = current_rmse
            best_train_loader = train_loader
            best_val_loader = val_loader
            torch.save(model.state_dict(), best_model_path)
            print(f"New best overall model saved to {best_model_path}")

        print(f"Fold {fold + 1} Results:")
        print(f"\tRMSE:\t{current_rmse:.3f}")
        print(f"\tR2:\t{fold_metrics_dict['Validation']['r2']:.3f}")

    avg_kfold_metrics = calculate_average_kfold_metrics(all_kfold_metrics)

    test_loader = create_gnn_dataloaders(test_dataset)[0]
    model = initialize_gnn_model(gnn_type, device, molformer_encoder)
    train_results = evaluate_gnn_model(
        model, best_train_loader, device, task="Training", weights_path=best_model_path
    )
    val_results = evaluate_gnn_model(
        model, best_val_loader, device, task="Validation", weights_path=best_model_path
    )
    test_results = evaluate_gnn_model(
        model, test_loader, device, task="Test", weights_path=best_model_path
    )

    return (
        train_results,
        val_results,
        test_results,
        all_kfold_metrics,
        avg_kfold_metrics,
    )


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
