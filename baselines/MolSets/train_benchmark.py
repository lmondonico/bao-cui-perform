"""Train and evaluate MolSets-SAGE using supplied splits."""

DATASET = "smited"
GPU_ID = 0

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from models import MolSets
from data_utils import GraphSetDataset, graph_set_collate

hyperpars = {
    "hidden_dim": 64,
    "emb_dim": 32,
    "att_dim": 16,
    "n_conv_layers": 3,
    "conv": "SAGEConv",
    "after_readout": "tanh",
    "max_ep": 10000,
    "es_patience": 10,
    "max_ep_wo_improv": 20,
    "lr": 0.001,
    "lrsch_patience": 10,
    "lrsch_factor": 0.5,
    "weight_decay": 0.0001,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if GPU_ID >= 0 and torch.cuda.is_available():
    device = torch.device(f"cuda:{GPU_ID}")
    print(f"Using GPU: {GPU_ID}")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"\n{'=' * 60}")
print(f"MolSets-SAGE Training - Dataset: {DATASET}")
print(f"{'=' * 60}\n")

print("Loading datasets...")
train_dataset = GraphSetDataset(f"splits-{DATASET}/data_train.pkl")
train_stats = train_dataset.get_stats()
val_dataset = GraphSetDataset(
    f"splits-{DATASET}/data_val.pkl", train_stats=train_stats
)
test_dataset = GraphSetDataset(
    f"splits-{DATASET}/data_test.pkl", train_stats=train_stats
)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")
print(f"  Test:  {len(test_dataset)} samples")
print(
    f"  Target normalization: mean={train_stats['target_mean']:.4f}, std={train_stats['target_std']:.4f}"
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_loader.collate_fn = graph_set_collate
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
val_loader.collate_fn = graph_set_collate
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loader.collate_fn = graph_set_collate

model = MolSets(
    n_node_features=13,
    hidden_dim=hyperpars["hidden_dim"],
    emb_dim=hyperpars["emb_dim"],
    att_dim=hyperpars["att_dim"],
    output_dim=1,
    conv=hyperpars["conv"],
    n_conv_layers=hyperpars["n_conv_layers"],
    after_readout=hyperpars["after_readout"],
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=hyperpars["lr"], weight_decay=hyperpars["weight_decay"]
)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=hyperpars["lrsch_factor"],
    patience=hyperpars["lrsch_patience"],
)
loss_fn = torch.nn.MSELoss()

print(f"\nModel parameters: {sum(param.numel() for param in model.parameters())}")
print(
    f"Hidden dim: {hyperpars['hidden_dim']}, Emb dim: {hyperpars['emb_dim']}, Att dim: {hyperpars['att_dim']}"
)


def train(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for sample in pbar:
        inputs, mws, fracs, salt_mols, salt_graphs, targets, temps = sample
        sample_size = len(targets)
        outs = torch.empty((sample_size, 1)).to(device)
        targets = torch.tensor(targets).unsqueeze(1).to(device)
        salt_mols = torch.tensor(salt_mols).to(device)
        temps = torch.tensor(temps).to(device)

        for j in range(sample_size):
            graph_set = inputs[j].to(device)
            salt_graph = salt_graphs[j].to(device)
            frac = torch.tensor(fracs[j]).to(device)
            mw = torch.tensor(mws[j]).to(device)
            optimizer.zero_grad()
            outs[j] = model(graph_set, mw, frac, salt_mols[j], salt_graph, temps[j])

        loss = criterion(outs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return train_loss / len(loader)


def evaluate(model, loader, criterion):
    """Evaluate on a dataset."""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for sample in pbar:
            inputs, mws, fracs, salt_mols, salt_graphs, targets, temps = sample
            sample_size = len(targets)
            outs = torch.empty((sample_size, 1)).to(device)
            targets = torch.tensor(targets).unsqueeze(1).to(device)
            salt_mols = torch.tensor(salt_mols).to(device)
            temps = torch.tensor(temps).to(device)

            for j in range(sample_size):
                graph_set = inputs[j].to(device)
                salt_graph = salt_graphs[j].to(device)
                frac = torch.tensor(fracs[j]).to(device)
                mw = torch.tensor(mws[j]).to(device)
                outs[j] = model(graph_set, mw, frac, salt_mols[j], salt_graph, temps[j])

            loss = criterion(outs, targets)
            val_loss += loss.item()
    return val_loss / len(loader)


print(f"\n{'=' * 60}")
print("Training...")
print(f"{'=' * 60}\n")

best_val_loss = np.inf
best_model = None
epochs_wo_improv = 0

pbar_epoch = tqdm(range(hyperpars["max_ep"]), desc="Epochs")
for epoch in pbar_epoch:
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, val_loader, loss_fn)
    scheduler.step(val_loss)

    if epoch > hyperpars["es_patience"]:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_wo_improv = 0
        else:
            epochs_wo_improv += 1
        if epochs_wo_improv >= hyperpars["max_ep_wo_improv"]:
            tqdm.write(f"\nEarly stopping at epoch {epoch + 1}")
            break

    pbar_epoch.set_postfix(
        {"train_loss": f"{train_loss:.5f}", "val_loss": f"{val_loss:.5f}"}
    )

    if (epoch + 1) % 10 == 0 or epoch < 5:
        tqdm.write(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}"
        )

if best_model is not None:
    model.load_state_dict(best_model)
    print(f"\nLoaded best model (val loss: {best_val_loss:.5f})")

    model_save_path = f"model_{DATASET}_hidden{hyperpars['hidden_dim']}.pt"
    torch.save(best_model, model_save_path)
    print(f"Model weights saved to: {model_save_path}")


def evaluate_split(model, loader, dataset, split_name):
    """
    Evaluate on a split and return predictions.

    Args:
        model: Trained model
        loader: Data loader
        dataset: Dataset object (for denormalization)
        split_name: Name of split for display

    Returns:
        Tuple of (truths, predictions) in original scale
    """
    model.eval()
    predictions = []
    truths = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {split_name}", leave=False)
        for sample in pbar:
            inputs, mws, fracs, salt_mols, salt_graphs, targets, temps = sample
            sample_size = len(targets)
            salt_mols = torch.tensor(salt_mols).to(device)
            temps = torch.tensor(temps).to(device)

            for j in range(sample_size):
                graph_set = inputs[j].to(device)
                salt_graph = salt_graphs[j].to(device)
                frac = torch.tensor(fracs[j]).to(device)
                mw = torch.tensor(mws[j]).to(device)

                out = model(graph_set, mw, frac, salt_mols[j], salt_graph, temps[j])

                predictions.append(out.cpu().numpy())
                truths.append(targets[j])

    predictions = np.stack(predictions).squeeze()
    truths = np.array(truths)

    predictions_orig = dataset.get_orig(predictions)
    truths_orig = dataset.get_orig(truths)

    return truths_orig, predictions_orig


print(f"\n{'=' * 60}")
print("Final Evaluation")
print(f"{'=' * 60}\n")

results = {}

for split_name, loader, dataset in [
    ("Training", train_loader, train_dataset),
    ("Validation", val_loader, val_dataset),
    ("Test", test_loader, test_dataset),
]:
    truths, predictions = evaluate_split(model, loader, dataset, split_name)

    mse = mean_squared_error(truths, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(truths, predictions)
    pearson_r = pearsonr(truths, predictions)[0]

    print(f"{split_name}:")
    print(f"  MSE:       {mse:.3f}")
    print(f"  RMSE:      {rmse:.3f}")
    print(f"  R2:        {r2:.3f}")
    print(f"  Pearson r: {pearson_r:.3f}")
    print()

    results[split_name] = (truths, predictions)


print(f"{'=' * 60}")
print("Saving predictions...")
print(f"{'=' * 60}\n")

rows = []
for split_name, (truths, preds) in results.items():
    for truth, pred in zip(truths, preds):
        rows.append(
            {"Task": split_name, "Truth": float(truth), "Prediction": float(pred)}
        )

df_predictions = pd.DataFrame(rows)
output_file = f"regr_predictions_{DATASET}.csv"
df_predictions.to_csv(output_file, index=False)
print(f"Predictions saved to: {output_file}")
print(f"  Training:   {len(results['Training'][0])} samples")
print(f"  Validation: {len(results['Validation'][0])} samples")
print(f"  Test:       {len(results['Test'][0])} samples")
print(f"  Total:      {len(rows)} samples")

print(f"\n{'=' * 60}")
print("Training complete!")
print(f"{'=' * 60}")
