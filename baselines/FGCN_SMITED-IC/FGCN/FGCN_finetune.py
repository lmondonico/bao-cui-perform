"""
Finetune F-GCN model (https://doi.org/10.48550/arXiv.2307.03811)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
import pickle
import time
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.compat.v1.keras.backend as K
import pydot

warnings.filterwarnings("ignore")

DATASET = "smited"
GPU_ID = 0

start_time = time.time()

script_dir = Path(__file__).parent

results_dir = script_dir / f"fgcn-results-{DATASET}"
results_dir.mkdir(parents=True, exist_ok=True)

os.makedirs(results_dir, exist_ok=True)

import logging

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(results_dir, f"training_log_preprocessed_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()
log.setLevel(logging.INFO)

log.info(f"Results will be saved to: {results_dir}")
log.info(f"Log file: {log_file}")


def configure_gpu():
    """Configure TensorFlow to use specified GPU."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        log.info("No GPUs available, using CPU")
        return "CPU"

    if GPU_ID is None:
        selected_gpu = gpus[0]
        gpu_id = 0
        log.info(f"Auto-selected GPU 0")
    elif GPU_ID >= len(gpus):
        log.warning(f"GPU {GPU_ID} not available. Only {len(gpus)} GPU(s) detected.")
        log.warning(f"Falling back to GPU 0")
        selected_gpu = gpus[0]
        gpu_id = 0
    else:
        selected_gpu = gpus[GPU_ID]
        gpu_id = GPU_ID
        log.info(f"Using manually specified GPU {GPU_ID}")

    try:
        tf.config.set_visible_devices(selected_gpu, "GPU")
        tf.config.experimental.set_memory_growth(selected_gpu, True)
        log.info(f"GPU {gpu_id} configured successfully")
    except RuntimeError as e:
        log.error(f"Error configuring GPU: {e}")

    return f"cuda:{gpu_id}"


device = configure_gpu()
log.info(f"Using device: {device}")

modelname = "FGCN"

n_node_features = 100
n_molecules = 6

preprocessed_dir = script_dir / f"preprocessed-{DATASET}"
train_preprocessed = preprocessed_dir / "train_data_preprocessed.pkl"
val_preprocessed = preprocessed_dir / "validation_data_preprocessed.pkl"
test_preprocessed = preprocessed_dir / "test_data_preprocessed.pkl"

log.info(f"Loading preprocessed training data from: {train_preprocessed}")
log.info(f"Loading preprocessed validation data from: {val_preprocessed}")
log.info(f"Loading preprocessed test data from: {test_preprocessed}")


with open(train_preprocessed, "rb") as f:
    train_data = pickle.load(f)

with open(val_preprocessed, "rb") as f:
    val_data = pickle.load(f)

with open(test_preprocessed, "rb") as f:
    test_data = pickle.load(f)

log.info(f"Training samples: {len(train_data)}")
log.info(f"Validation samples: {len(val_data)}")
log.info(f"Test samples: {len(test_data)}")


class GCNLayer(tf.keras.layers.Layer):
    """Implementation of GCN as layer"""

    def __init__(self, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        nodes, adj = inputs
        degree = tf.reduce_sum(adj, axis=-1)
        degree = tf.maximum(degree, 1e-6)
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


class GRLayer(tf.keras.layers.Layer):
    """A GNN layer that computes average over all node features"""

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction

    def compute_output_shape(self, input_shape):
        node_shape, adj_shape = input_shape
        return (node_shape[0], node_shape[2])


def gcn_mols(n_node_features, max_atoms=None):
    """Create GCN model for molecules - keep None for weight compatibility"""
    ninput = tf.keras.Input(
        (
            None,
            n_node_features,
        )
    )
    ainput = tf.keras.Input(
        (
            None,
            None,
        )
    )

    layer1 = GCNLayer("tanh")([ninput, ainput])
    layer2 = GCNLayer("tanh")(layer1)
    layer3 = GCNLayer("tanh")(layer2)
    layer4 = GCNLayer("tanh")(layer3)
    layer5 = GRLayer()(layer4)
    output = tf.keras.layers.Dense(1)(layer5)

    return keras.Model(inputs=(ninput, ainput), outputs=layer5)


def list_to_tf_dataset(data):
    """
    Convert list of preprocessed samples to TensorFlow dataset
    Eliminates Python generator bottleneck by converting to tensors upfront
    """
    tensors = []
    labels = []

    for sample in data:
        interleaved = []
        for node, adj in sample["graphs"]:
            interleaved.append(node)
            interleaved.append(adj)

        concentrations = sample["concentrations"]
        loading = sample["loading"]
        y = sample["properties"]

        tensors.append((*interleaved, concentrations, loading))
        labels.append(y)

    tensor_data = tuple(
        tf.convert_to_tensor(np.array([x[i] for x in tensors]))
        for i in range(len(tensors[0]))
    )
    tensor_labels = tf.convert_to_tensor(np.array(labels))

    return tf.data.Dataset.from_tensor_slices((tensor_data, tensor_labels))


max_atoms = 70
output_signature_list = []

for i in range(n_molecules):
    output_signature_list.append(
        tf.TensorSpec(shape=(max_atoms, n_node_features), dtype=tf.float32)
    )
    output_signature_list.append(
        tf.TensorSpec(shape=(max_atoms, max_atoms), dtype=tf.float32)
    )

output_signature_list = (
    output_signature_list
    + [tf.TensorSpec(shape=(n_molecules,), dtype=tf.float32)]
    + [tf.TensorSpec(shape=(1,), dtype=tf.float32)]
)

log.info("\nOutput shapes\n{}".format(output_signature_list))


log.info("Converting preprocessed data to TensorFlow tensors...")
dataset = list_to_tf_dataset(train_data)
val_dataset = list_to_tf_dataset(val_data)
test_dataset = list_to_tf_dataset(test_data)

log.info(f"Training dataset: {dataset}")
log.info(f"Validation dataset: {val_dataset}")
log.info(f"Test dataset: {test_dataset}")


formulation_inputs = {}
model_molecule = {}

log.info(f"N node features: {n_node_features}")

for i in range(n_molecules):
    formulation_inputs[f"node{i}"] = tf.keras.Input(
        (
            max_atoms,
            n_node_features,
        )
    )
    formulation_inputs[f"adj{i}"] = tf.keras.Input(
        (
            max_atoms,
            max_atoms,
        )
    )

    model_molecule[f"modelmolecule{i}"] = gcn_mols(n_node_features, max_atoms)
    weights_path = script_dir / "pretrained_weights" / "my_checkpoint"
    model_molecule[f"modelmolecule{i}"].load_weights(weights_path)
    model_molecule[f"modelmolecule{i}"].trainable = False


wt = tf.keras.Input((n_molecules,))
cath = tf.keras.Input((1,))

log.debug(f"\nFormulation inputs\n{formulation_inputs}\n")
log.debug(f"wt\n{wt}")
log.debug(f"cath\n{cath}")

gcn_out = {}
gcn_wt = {}

for i in range(n_molecules):
    gcn_out[f"gcn_out{i}"] = model_molecule[f"modelmolecule{i}"](
        [formulation_inputs[f"node{i}"], formulation_inputs[f"adj{i}"]]
    )
    gcn_wt[f"gcn_wt{i}"] = layers.Multiply()(
        [gcn_out[f"gcn_out{i}"], tf.expand_dims(wt[:, i], -1)]
    )

log.debug(f"\ngcn out {gcn_out[f'gcn_out0']}\n")
log.debug(f"\ngcn wt {gcn_wt[f'gcn_wt0']}\n")

log.debug("\nLayers to add\n{}\n".format([v for _, v in gcn_wt.items()]))

x1 = layers.concatenate([v for _, v in gcn_wt.items()])
x3 = layers.concatenate([x1, cath])
x = layers.Dense(200, activation="sigmoid")(x3)
x = layers.Dense(100, activation="sigmoid")(x)
performance = layers.Dense(1)(x)

inps = []
for ith in range(n_molecules):
    inps.append(formulation_inputs[f"node{ith}"])
    inps.append(formulation_inputs[f"adj{ith}"])

inps = inps + [wt] + [cath]

log.debug(f"inps\n{inps}\n")
log.info(f"inputs {inps}\n")

electrolyte_model = keras.Model(inputs=inps, outputs=performance)
electrolyte_model.summary()

keras.utils.plot_model(
    electrolyte_model,
    os.path.join(results_dir, modelname + "_struct.png"),
    show_shapes=True,
)


opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
electrolyte_model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())

epochs = 600
save_period = 2
batch_size = 32

log.info(f"\nStarting training for {epochs} epochs...")
log.info(f"Batch size: {batch_size}")
log.info(f"Optimizer: Adam (lr=0.0001)")

result = electrolyte_model.fit(
    dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
    epochs=epochs,
    verbose=2,
    validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
)

electrolyte_model.save(os.path.join(results_dir, modelname))
log.info(f"Model saved to: {os.path.join(results_dir, modelname)}")


log.info("\n" + "=" * 60)
log.info("Calculating final metrics...")
log.info("=" * 60)

train_predictions = electrolyte_model.predict(dataset.batch(batch_size), verbose=0)
val_predictions = electrolyte_model.predict(val_dataset.batch(batch_size), verbose=0)
test_predictions = electrolyte_model.predict(test_dataset.batch(batch_size), verbose=0)

train_actuals = np.concatenate([y.numpy() for _, y in dataset.batch(batch_size)])
val_actuals = np.concatenate([y.numpy() for _, y in val_dataset.batch(batch_size)])
test_actuals = np.concatenate([y.numpy() for _, y in test_dataset.batch(batch_size)])

train_r2 = r2_score(train_actuals, train_predictions)
train_rmse = np.sqrt(mean_squared_error(train_actuals, train_predictions))
train_mse = mean_squared_error(train_actuals, train_predictions)
train_pearson_r, _ = pearsonr(train_actuals.flatten(), train_predictions.flatten())

val_r2 = r2_score(val_actuals, val_predictions)
val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
val_mse = mean_squared_error(val_actuals, val_predictions)
val_pearson_r, _ = pearsonr(val_actuals.flatten(), val_predictions.flatten())

test_r2 = r2_score(test_actuals, test_predictions)
test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
test_mse = mean_squared_error(test_actuals, test_predictions)
test_pearson_r, _ = pearsonr(test_actuals.flatten(), test_predictions.flatten())

log.info(f"\nFinal Training Metrics:")
log.info(f"  R² Score: {train_r2:.4f}")
log.info(f"  RMSE: {train_rmse:.4f}")
log.info(f"  MSE: {train_mse:.4f}")
log.info(f"  Pearson R: {train_pearson_r:.4f}")

log.info(f"\nFinal Validation Metrics:")
log.info(f"  R² Score: {val_r2:.4f}")
log.info(f"  RMSE: {val_rmse:.4f}")
log.info(f"  MSE: {val_mse:.4f}")
log.info(f"  Pearson R: {val_pearson_r:.4f}")

log.info(f"\nFinal Test Metrics (Held-out):")
log.info(f"  R² Score: {test_r2:.4f}")
log.info(f"  RMSE: {test_rmse:.4f}")
log.info(f"  MSE: {test_mse:.4f}")
log.info(f"  Pearson R: {test_pearson_r:.4f}")

metrics_dict = {
    "dataset": ["training", "validation", "test"],
    "r2_score": [train_r2, val_r2, test_r2],
    "rmse": [train_rmse, val_rmse, test_rmse],
    "mse": [train_mse, val_mse, test_mse],
    "pearson_r": [train_pearson_r, val_pearson_r, test_pearson_r],
    "final_loss": [result.history["loss"][-1], result.history["val_loss"][-1], "N/A"],
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(os.path.join(results_dir, modelname + "_metrics.csv"), index=False)
log.info(f"\nMetrics saved to: {os.path.join(results_dir, modelname + '_metrics.csv')}")


axis_tick_size = 14
axis_title_size = 16
plot_title_size = 18

plt.figure(figsize=(10, 6))
plt.plot(result.history["loss"], label="Train Loss", color="orange")
plt.plot(result.history["val_loss"], label="Validation Loss", color="blue")
plt.legend(frameon=False, fontsize=axis_tick_size)
plt.xlabel("Epoch", fontsize=axis_title_size)
plt.ylabel("Loss", fontsize=axis_title_size)
plt.title("F-GCN: Training and Validation Loss Curves", fontsize=plot_title_size)
plt.xticks(fontsize=axis_tick_size)
plt.yticks(fontsize=axis_tick_size)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, modelname + "_loss.png"), dpi=900)
plt.close()
log.info(f"Loss plot saved to: {os.path.join(results_dir, modelname + '_loss.png')}")

plt.figure(figsize=(12, 10))

plt.scatter(
    train_actuals,
    train_predictions,
    alpha=0.5,
    s=50,
    c="orange",
    edgecolors="none",
    label="Training",
)

plt.scatter(
    val_actuals,
    val_predictions,
    alpha=0.5,
    s=50,
    c="blue",
    edgecolors="none",
    label="Validation",
)

plt.scatter(
    test_actuals,
    test_predictions,
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
plt.title("F-GCN: Parity Plot", fontsize=plot_title_size)
plt.xticks(fontsize=axis_tick_size)
plt.yticks(fontsize=axis_tick_size)
plt.legend(frameon=False, fontsize=16, loc="best")
plt.tight_layout()

plt.savefig(os.path.join(results_dir, modelname + "_parity_plot.png"), dpi=900)
plt.close()
log.info(
    f"Parity plot saved to: {os.path.join(results_dir, modelname + '_parity_plot.png')}"
)

history_df = pd.DataFrame(
    {
        "epoch": range(1, len(result.history["loss"]) + 1),
        "training_loss": result.history["loss"],
        "validation_loss": result.history["val_loss"],
    }
)
history_df.to_csv(
    os.path.join(results_dir, modelname + "_training_history.csv"), index=False
)
log.info(
    f"Training history saved to {os.path.join(results_dir, modelname + '_training_history.csv')}"
)

end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

log.info(f"\n{'=' * 50}")
log.info(f"Training completed successfully!")
log.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
log.info(f"All results saved to: {results_dir}")
log.info(f"{'=' * 50}")
