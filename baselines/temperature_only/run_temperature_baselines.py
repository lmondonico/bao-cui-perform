"""Fit temperature-only controls on supplied train, validation and test splits."""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--split-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.study import local_path, require_training_enabled
from data.conversion_scripts.prepare_splits import split_filename
require_training_enabled()
SPLIT_DIR = local_path(args.split_dir, ROOT / "data")
OUT_DIR = local_path(args.output, ROOT / "experiments/results")
if OUT_DIR.exists():
    raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
OUT_DIR.mkdir(parents=True)
predictions, metric_rows, histories = [], [], []
split_frames = {name: pd.read_csv(SPLIT_DIR / split_filename('calisol', name)) for name in ("train", "validation", "test")}


def save_result(name, seed, prediction_by_split):
    for split, prediction in prediction_by_split.items():
        frame = split_frames[split]
        truth = frame["LIC"].to_numpy(dtype=float)
        if len(prediction) != len(frame):
            raise ValueError(f"Prediction membership changed for {name}/{split}")
        metric_rows.append({"model": name, "seed": seed, "split": split,
                            "n": len(frame), "unique_compositions": frame.Composition_ID.nunique(),
                            "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
                            "MAE": float(mean_absolute_error(truth, prediction)),
                            "R2": float(r2_score(truth, prediction))})
        predictions.extend({"model": name, "seed": seed, "split": split,
                            "Row_ID": int(row_id), "Truth": float(actual), "Prediction": float(estimate)}
                           for row_id, actual, estimate in zip(frame.Row_ID, truth, prediction))

def load_split(name):
    df = split_frames[name]
    T_celsius = df["Temperature"].values.astype(np.float64)
    T_kelvin = T_celsius + 273.15
    LIC = df["LIC"].values.astype(np.float64)
    return T_kelvin, LIC


T_train, y_train = load_split("train")
T_val, y_val = load_split("validation")
T_test, y_test = load_split("test")

print(f"Split sizes  —  train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
print(f"Temperature range (K): {T_train.min():.1f} – {T_train.max():.1f}")
print(f"LIC range: {y_train.min():.3f} – {y_train.max():.3f}")
print()

X_train_arr = (1.0 / T_train).reshape(-1, 1)
X_val_arr = (1.0 / T_val).reshape(-1, 1)
X_test_arr = (1.0 / T_test).reshape(-1, 1)

lr = LinearRegression().fit(X_train_arr, y_train)
pred_arr = lr.predict(X_test_arr)
rmse_arr = np.sqrt(mean_squared_error(y_test, pred_arr))
r2_arr = r2_score(y_test, pred_arr)
print(f"[Arrhenius Linear]  RMSE={rmse_arr:.4f}  R²={r2_arr:.4f}")
save_result("arrhenius", args.seed, {"train": lr.predict(X_train_arr), "validation": lr.predict(X_val_arr), "test": pred_arr})
(OUT_DIR / "arrhenius.json").write_text(json.dumps({"coefficient": lr.coef_.tolist(), "intercept": float(lr.intercept_)}) + "\n")

try:
    from xgboost import XGBRegressor
except ImportError:
    sys.exit("xgboost not installed – run: pip install xgboost")

X_train_xgb = np.column_stack([T_train, 1.0 / T_train])
X_val_xgb = np.column_stack([T_val, 1.0 / T_val])
X_test_xgb = np.column_stack([T_test, 1.0 / T_test])

xgb = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    early_stopping_rounds=50,
    random_state=args.seed,
    verbosity=0,
)
xgb.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
pred_xgb = xgb.predict(X_test_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
r2_xgb = r2_score(y_test, pred_xgb)
print(f"[XGBoost]           RMSE={rmse_xgb:.4f}  R²={r2_xgb:.4f}")
save_result("xgboost_temperature", args.seed, {"train": xgb.predict(X_train_xgb), "validation": xgb.predict(X_val_xgb), "test": pred_xgb})
xgb.save_model(OUT_DIR / "xgboost.json")

device = torch.device(args.device)


class TempMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_features(T):
    return np.column_stack([T, 1.0 / T])


X_raw_train = make_features(T_train)
X_raw_val = make_features(T_val)
X_raw_test = make_features(T_test)

mu = X_raw_train.mean(axis=0)
sigma = X_raw_train.std(axis=0)


def to_tensor(X, y=None):
    Xt = torch.tensor((X - mu) / sigma, dtype=torch.float32, device=device)
    if y is not None:
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        return Xt, yt
    return Xt


Xt_train, yt_train = to_tensor(X_raw_train, y_train)
Xt_val, yt_val = to_tensor(X_raw_val, y_val)
Xt_test, yt_test = to_tensor(X_raw_test, y_test)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
model = TempMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn = nn.MSELoss()

best_val = float("inf")
patience_ctr = 0
best_state = None
train_losses_epoch = []
val_losses_epoch = []

for epoch in range(1, 1001):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xt_train), yt_train)
    loss.backward()
    opt.step()

    train_losses_epoch.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xt_val), yt_val).item()
    val_losses_epoch.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= 100:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(Xt_test).cpu().numpy()
    prediction_by_split = {"train": model(Xt_train).cpu().numpy(),
                           "validation": model(Xt_val).cpu().numpy(), "test": pred}
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"[MLP seed={seed}]      RMSE={rmse:.4f}  R²={r2:.4f}  epochs={len(train_losses_epoch)}")
save_result("temperature_mlp", seed, prediction_by_split)
torch.save({"model_state_dict": best_state, "seed": seed, "feature_mean": mu.tolist(),
            "feature_std": sigma.tolist(), "selection": "minimum_validation_MSE"}, OUT_DIR / f"mlp_seed_{seed}.pth")
histories.extend({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                 for epoch, (train_loss, val_loss) in enumerate(zip(train_losses_epoch, val_losses_epoch), 1))
best_mlp_val_rmse = np.sqrt(best_val)
best_mlp_seed = seed
best_mlp_pred = pred
best_train_losses = train_losses_epoch
best_val_losses = val_losses_epoch

print("\n" + "=" * 78)
print("TEMPERATURE-ONLY BASELINE RESULTS (Test Set)")
print("=" * 78)
header = f"{'Model':<30} {'Input':<18} {'RMSE ↓':>10} {'R² ↑':>12}"
print(header)
print("-" * 78)
print(f"{'Arrhenius Linear':<30} {'1/T':<18} {rmse_arr:>10.4f} {r2_arr:>12.4f}")
print(f"{'XGBoost':<30} {'T, 1/T':<18} {rmse_xgb:>10.4f} {r2_xgb:>12.4f}")
print(f"{'MLP (256-256-128)':<30} {'T, 1/T':<18} {rmse:>10.4f} {r2:>12.4f}")
print("-" * 78)
print("=" * 78)

best_name, best_validation_rmse, best_pred = "MLP", best_mlp_val_rmse, best_mlp_pred
best_rmse_val = np.sqrt(mean_squared_error(y_test, best_pred))
best_r2_val = r2_score(y_test, best_pred)
pd.DataFrame(predictions).to_csv(OUT_DIR / "predictions.csv", index=False)
pd.DataFrame(metric_rows).to_csv(OUT_DIR / "metrics.csv", index=False)
pd.DataFrame(histories).to_csv(OUT_DIR / "training_history.csv", index=False)
(OUT_DIR / "selection.json").write_text(json.dumps({"criterion": "Prespecified MLP control and supplied seed; best epoch by validation MSE", "displayed_model": best_name,
    "displayed_model_validation_rmse": float(best_validation_rmse), "mlp_seed": best_mlp_seed,
    "mlp_validation_rmse": float(best_mlp_val_rmse)}, indent=2) + "\n")

axis_tick_size = 14
axis_title_size = 16
plot_title_size = 18

fig, ax = plt.subplots(figsize=(12, 10))
sc = ax.scatter(y_test, best_pred, c=T_test, cmap="coolwarm", s=50, alpha=0.5, edgecolors="none")

all_vals = np.concatenate([y_test, best_pred])
min_val = np.min(all_vals)
max_val = np.max(all_vals)
ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

ax.set_xlabel("Actual Values", fontsize=axis_title_size)
ax.set_ylabel("Predicted Values", fontsize=axis_title_size)

title_line1 = f"{best_name} — Temperature-Only Parity Plot (Actual vs Predicted)"
title_line2 = f"Test: R²={best_r2_val:.3f} RMSE={best_rmse_val:.3f}"
ax.set_title(f"{title_line1}\n{title_line2}", fontsize=plot_title_size, fontweight="bold")

ax.tick_params(axis="both", labelsize=axis_tick_size)
ax.grid(False)

cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Temperature (K)", fontsize=axis_title_size)
cbar.ax.tick_params(labelsize=axis_tick_size)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "temperature_only_scatter.png")
fig.savefig(plot_path, dpi=900, bbox_inches="tight")
plt.close()
print(f"\nScatter plot saved to: {plot_path}")

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

epochs_range = range(1, len(best_train_losses) + 1)
ax1.plot(epochs_range, best_train_losses, label="Train Loss", color="orange", linewidth=2)
ax1.plot(epochs_range, best_val_losses, label="Validation Loss", color="blue", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss over Epochs")
ax1.legend(frameon=False)
ax1.grid(False)

plt.tight_layout()
loss_plot_path = os.path.join(OUT_DIR, "regr_loss_plot.png")
fig.savefig(loss_plot_path, dpi=900)
plt.close()
print(f"Loss curve plot saved to: {loss_plot_path}")
