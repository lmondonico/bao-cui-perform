"""Train and evaluate GeoMix using supplied split CSVs."""

import os
import sys
import time
import json
import math
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*TorchScript.*")
warnings.filterwarnings("ignore", message=".*tight_layout.*")

import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim
import torch.utils
from torch.utils.data import DataLoader

GEOMIX_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GEOMIX_DIR)

from utils.seed import fix_seed
from dataset.CALiSol.ConductivityDataset import ConductivityDataset, custom_collate_fn as custom_collate_CALiSol
from dataset.DiffMixData.DiffMixData import DiffMixData, custom_collate_fn as custom_collate_DiffMix
from models.EGNN_linear import EGNN_linear
from models.TFN_linear import TFN_linear
from models.TFN_atten import TFN_atten
from models.EGNN_atten import EGNN_atten
from models.GeoMix import GeoMix
from models.GeoMix_ctrf import GeoMix as GeoMix_ctrf


parser = argparse.ArgumentParser(description='GeoMix Benchmark with Pre-Split Data')

parser.add_argument('--exp_name', type=str, default='benchmark')
parser.add_argument('--model', type=str, default='GeoMix')
parser.add_argument('--encoder', type=str, default="TFN")
parser.add_argument('--num_conv_layer', type=int, default=4)
parser.add_argument('--num_gil_layer', type=int, default=3)
parser.add_argument('--num_mlp_layer', type=int, default=3)
parser.add_argument('--num_atten_layer', type=int, default=3)
parser.add_argument('--num_atten_head', type=int, default=4)
parser.add_argument('--num_channels', type=int, default=8)
parser.add_argument('--graph_hidden_dim', type=int, default=64)
parser.add_argument('--if_salt_embd', type=str, default="True")
parser.add_argument('--no_R', action='store_false', dest='if_R')
parser.add_argument('--no_t', action='store_false', dest='if_t')
parser.add_argument('--att_dropout', type=float, default=0)
parser.add_argument('--att_temp', type=float, default=2.0)

parser.add_argument('--dataset', type=str, default='CALiSol')
parser.add_argument('--train_csv', type=str, required=True)
parser.add_argument('--val_csv', type=str, required=True)
parser.add_argument('--test_csv', type=str, required=True)
parser.add_argument('--npz_dir', type=str, default='./data/CoordData/coords-solvent-pubchem-radius.npz')
parser.add_argument('--salt_npz_dir', type=str, default='./data/CoordData/coords-salts-radius.npz')
parser.add_argument('--max_samples', type=int, default=100000)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=501)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=1e-12)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (epochs without val improvement)')
parser.add_argument('--min_epochs', type=int, default=50, help='Minimum epochs before early stopping kicks in')

parser.add_argument('--multi_gpu', action='store_true', help='Use all available GPUs via DataParallel')
parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU IDs (e.g., "0,1,2,3")')

parser.add_argument('--test_interval', type=int, default=5)
parser.add_argument('--results_dir', type=str, default='./benchmark_results')
parser.add_argument(
    '--run_name',
    type=str,
    default=None,
    help='Explicit output-directory/file prefix for collision-free seeded runs',
)

args = parser.parse_args()

if not torch.cuda.is_available():
    args.device = 'cpu'
else:
    args.device = 'cuda'


def plot_loss_curves(train_losses, val_losses, val_epochs, save_path):
    """
    Plot training and validation loss curves.
    Matches core/plotting.py PlotLossesLR style.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(range(1, len(train_losses) + 1), train_losses,
            label="Train Loss", color="orange", linewidth=2)
    ax.plot(val_epochs, val_losses,
            label="Validation Loss", color="blue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss over Epochs")
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"  Loss plot saved to {save_path}")


def plot_parity(split_results, save_path, model_name="", dataset_name=""):
    """
    Parity plot: Actual vs Predicted for train/val/test splits.
    Matches core/plotting.py plot_results() style exactly.
    """
    axis_tick_size = 14
    axis_title_size = 16
    plot_title_size = 18

    train_true, train_pred = split_results["Training"]
    val_true, val_pred = split_results["Validation"]
    test_true, test_pred = split_results["Test"]

    train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
    train_r2 = r2_score(train_true, train_pred)
    val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    val_r2 = r2_score(val_true, val_pred)
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_r2 = r2_score(test_true, test_pred)

    plt.figure(figsize=(12, 10))
    plt.scatter(train_true, train_pred, color="orange", alpha=0.5, s=50, edgecolors="none", label="Training")
    plt.scatter(val_true, val_pred, color="blue", alpha=0.5, s=50, edgecolors="none", label="Validation")
    plt.scatter(test_true, test_pred, color="green", alpha=0.5, s=50, edgecolors="none", label="Test")

    all_true = np.concatenate([train_true, val_true, test_true])
    all_pred = np.concatenate([train_pred, val_pred, test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    plt.xlabel("Actual Values", fontsize=axis_title_size)
    plt.ylabel("Predicted Values", fontsize=axis_title_size)
    title_line1 = "Parity Plot (Actual vs Predicted)"
    title_line2 = (
        f"Train: R\u00b2={train_r2:.3f} RMSE={train_rmse:.3f} | "
        f"Val: R\u00b2={val_r2:.3f} RMSE={val_rmse:.3f} | "
        f"Test: R\u00b2={test_r2:.3f} RMSE={test_rmse:.3f}"
    )
    plt.title(f"{title_line1}\n{title_line2}", fontsize=plot_title_size, fontweight="bold")
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
        plt.tight_layout()
    plt.savefig(save_path, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"  Parity plot saved to {save_path}")


def load_dataset(csv_path, dataset_type, npz_dir, salt_npz_dir, max_samples, device):
    if dataset_type == 'CALiSol':
        return ConductivityDataset(
            dataset_path=csv_path, npz_path=npz_dir,
            salt_npz_path=salt_npz_dir, max_samples=max_samples, device=device
        )
    elif dataset_type == 'DiffMixData':
        return DiffMixData(
            dataset_path=csv_path, npz_path=npz_dir,
            salt_npz_path=salt_npz_dir, max_samples=max_samples, device=device
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_epoch(model, optimizer, loader, args, backprop=True):
    """Run one epoch of training or evaluation."""
    if backprop:
        model.train()
    else:
        model.eval()

    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    res = {'loss': 0.0, 'counter': 0, 'MAE': 0.0}
    all_preds = []
    all_targets = []

    context = torch.no_grad() if not backprop else torch.enable_grad()
    with context:
        for batch in loader:
            for key, value in batch.items():
                batch[key] = value.to(args.device)

            n2g, n2b, g2b = batch['n2g'], batch['n2b'], batch['g2b']
            model_inputs = {
                'batch': batch['graphs'],
                'salt_batch': batch['salt_graphs'],
                'n2g_indices': n2g,
                'n2b_indices': n2b,
                'g2b_indices': g2b,
                'T': batch['T'],
                'c': batch['c'],
                'salt_one_hot': batch['salt_one_hot']
            }

            optimizer.zero_grad()
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            reg = 0.0

            if args.model in ['EGNN_linear', 'TFN_linear', 'EGNN_atten', 'TFN_atten']:
                pred = raw_model(**model_inputs)
            elif args.model in ['GeoMix', 'GeoMix_ctrf']:
                pred, reg = raw_model(**model_inputs)

            valid_mask = ~(pred.isnan() | pred.isinf()).squeeze()
            if valid_mask.any():
                valid_pred = pred[valid_mask]
                valid_target = batch['k'][valid_mask]

                loss = loss_fn(valid_pred, valid_target) + reg

                if not (loss.isnan() or loss.isinf()):
                    if backprop:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                    all_preds.extend(valid_pred.detach().cpu().view(-1).tolist())
                    all_targets.extend(valid_target.detach().cpu().view(-1).tolist())
                    res['loss'] += loss.item() * len(valid_target)
                    res['counter'] += len(valid_target)

    avg_loss = res['loss'] / res['counter'] if res['counter'] > 0 else float('inf')
    return avg_loss, np.array(all_preds), np.array(all_targets)


def compute_metrics(preds, targets):
    """Compute metrics on LIC (log10 IC) scale — the unified scale for all models."""
    m = {}
    m['mse'] = float(mean_squared_error(targets, preds))
    m['rmse'] = float(np.sqrt(m['mse']))
    m['r2'] = float(r2_score(targets, preds))
    m['pearson_r'] = float(pearsonr(preds, targets)[0])
    m['mae'] = float(np.mean(np.abs(targets - preds)))
    rho, _ = spearmanr(preds, targets)
    m['spearman_rho'] = float(rho)
    return m


def main():
    fix_seed(seed=args.seed)

    ds_name = Path(args.train_csv).parent.name
    run_name = args.run_name or f"{args.model}_{ds_name}"

    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}, Encoder: {args.encoder}, Dataset: {ds_name}")
    print(f"Train: {args.train_csv}")
    print(f"Val:   {args.val_csv}")
    print(f"Test:  {args.test_csv}")

    if args.multi_gpu and torch.cuda.device_count() > 1:
        if args.gpu_ids:
            gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        effective_batch_size = args.batch_size * len(gpu_ids)
    else:
        gpu_ids = None
        effective_batch_size = args.batch_size
        if torch.cuda.is_available():
            print(f"Using single GPU: cuda:0")

    print(f"Effective batch size: {effective_batch_size}")
    print(f"{'='*60}\n")

    load_device = args.device
    print("Loading training data...")
    dataset_train = load_dataset(args.train_csv, args.dataset, args.npz_dir, args.salt_npz_dir, args.max_samples, load_device)
    print("Loading validation data...")
    dataset_val = load_dataset(args.val_csv, args.dataset, args.npz_dir, args.salt_npz_dir, args.max_samples, load_device)
    print("Loading test data...")
    dataset_test = load_dataset(args.test_csv, args.dataset, args.npz_dir, args.salt_npz_dir, args.max_samples, load_device)

    if args.dataset == 'CALiSol':
        collate_fn = partial(custom_collate_CALiSol, device=args.device)
    else:
        collate_fn = partial(custom_collate_DiffMix, device=args.device)

    loader_train = DataLoader(dataset_train, batch_size=effective_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=effective_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)
    loader_test = DataLoader(dataset_test, batch_size=effective_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)

    common_params = {
        'num_layer': args.num_conv_layer, 'node_input_dim': 14, 'edge_attr_dim': 1,
        'hidden_dim': args.graph_hidden_dim, 'num_mlp_layer': args.num_mlp_layer,
        'device': args.device,
    }
    atten_params = {'num_atten_layer': args.num_atten_layer, 'num_atten_head': args.num_atten_head}
    salt_embedding = {'if_salt_embed': args.if_salt_embd, 'salt_embed_dim': 14}

    if args.model == 'GeoMix':
        model = GeoMix(**common_params, **atten_params, **salt_embedding, num_gil_layer=args.num_gil_layer, encoder=args.encoder)
    elif args.model == 'GeoMix_ctrf':
        model = GeoMix_ctrf(**common_params, **atten_params, **salt_embedding, num_gil_layer=args.num_gil_layer, encoder=args.encoder, att_dropout=args.att_dropout, att_temp=args.att_temp)
    elif args.model == 'TFN_atten':
        model = TFN_atten(**common_params, **atten_params)
    elif args.model == 'EGNN_atten':
        model = EGNN_atten(**common_params, **atten_params)
    elif args.model == 'TFN_linear':
        model = TFN_linear(**common_params, **salt_embedding)
    elif args.model == 'EGNN_linear':
        model = EGNN_linear(**common_params, **salt_embedding)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(args.device)

    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_losses = []
    val_losses = []
    best = {'epoch': 0, 'val_loss': float('inf'), 'model_state': None}
    epochs_without_improvement = 0

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={args.patience}, min_epochs={args.min_epochs})...")
    start_time = time.time()

    warmup_epochs = 5

    pbar = tqdm(range(args.epochs), desc="Training", unit="ep")
    for epoch in pbar:
        train_loss, _, _ = run_epoch(model, optimizer, loader_train, args, backprop=True)
        train_losses.append(train_loss)

        if epoch % args.test_interval == 0 and epoch >= warmup_epochs:
            val_loss, _, _ = run_epoch(model, optimizer, loader_val, args, backprop=False)
            val_losses.append(val_loss)

            if val_loss < best['val_loss']:
                best['epoch'] = epoch
                best['val_loss'] = val_loss
                raw = model.module if isinstance(model, nn.DataParallel) else model
                best['model_state'] = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += args.test_interval

            pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                             best=f"{best['val_loss']:.4f}", best_ep=best['epoch'],
                             no_improv=epochs_without_improvement)

            if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
                tqdm.write(f"\n  Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs)")
                break

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min. Best epoch: {best['epoch']}, stopped at epoch: {epoch}")

    raw = model.module if isinstance(model, nn.DataParallel) else model
    if best['model_state'] is None:
        print("WARNING: No valid model state saved (all validation was NaN). Using final model state.")
        best['model_state'] = {k: v.cpu().clone() for k, v in raw.state_dict().items()}
    raw.load_state_dict(best['model_state'])
    model.eval()

    ckpt_path = run_dir / f"{run_name}_best.pth"
    torch.save(best['model_state'], ckpt_path)
    print(f"  Model saved to {ckpt_path}")

    print(f"\n{'='*60}")
    print("Final Evaluation on Best Model")
    print(f"{'='*60}\n")

    split_results = {}

    for split_name, loader in [("Training", loader_train), ("Validation", loader_val), ("Test", loader_test)]:
        _, preds, targets = run_epoch(model, optimizer, loader, args, backprop=False)
        split_results[split_name] = (targets, preds)
        m = compute_metrics(preds, targets)

        print(f"{split_name} (LIC scale):")
        print(f"  MSE={m['mse']:.6f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  Pearson r={m['pearson_r']:.4f}")
        print()

    test_targets, test_preds = split_results['Test']
    test_metrics = compute_metrics(test_preds, test_targets)

    metrics_rows = []
    for split_name in ["Training", "Validation", "Test"]:
        targets, preds = split_results[split_name]
        m = compute_metrics(preds, targets)
        m['split'] = split_name
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = run_dir / f"{run_name}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Metrics saved to {metrics_csv}")

    pred_rows = []
    for split_name in ["Training", "Validation", "Test"]:
        targets, preds = split_results[split_name]
        for t, p in zip(targets, preds):
            pred_rows.append({'split': split_name, 'actual_lic': float(t), 'predicted_lic': float(p)})
    pred_df = pd.DataFrame(pred_rows)
    preds_csv = run_dir / f"{run_name}_predictions.csv"
    pred_df.to_csv(preds_csv, index=False)
    print(f"  Predictions saved to {preds_csv}")

    history_df = pd.DataFrame({
        'epoch': list(range(len(train_losses))),
        'train_loss': train_losses,
    })
    val_history = pd.DataFrame({
        'epoch': [i * args.test_interval for i in range(len(val_losses))],
        'val_loss': val_losses,
    })
    history_csv = run_dir / f"{run_name}_training_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"  Training history saved to {history_csv}")

    result_json = run_dir / f"{run_name}_results.json"
    with open(result_json, 'w') as f:
        json.dump({
            'model': args.model, 'encoder': args.encoder, 'dataset': ds_name,
            'seed': args.seed, 'best_epoch': best['epoch'],
            'test_metrics': test_metrics,
            'hyperparameters': {
                'epochs': args.epochs, 'batch_size': args.batch_size,
                'effective_batch_size': effective_batch_size,
                'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                'num_conv_layer': args.num_conv_layer, 'num_gil_layer': args.num_gil_layer,
                'graph_hidden_dim': args.graph_hidden_dim,
                'multi_gpu': args.multi_gpu,
                'num_gpus': len(gpu_ids) if gpu_ids else 1,
            },
            'training_time_min': round(elapsed / 60, 1),
        }, f, indent=2)
    print(f"  Results JSON saved to {result_json}")

    val_epoch_list = list(range(warmup_epochs, warmup_epochs + len(val_losses) * args.test_interval, args.test_interval))
    plot_loss_curves(train_losses, val_losses, val_epoch_list, run_dir / f"{run_name}_loss.png")

    plot_parity(split_results, run_dir / f"{run_name}_parity.png",
                model_name=args.model, dataset_name=ds_name)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {run_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
