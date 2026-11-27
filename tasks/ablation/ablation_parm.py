import torch
import sys
import os


sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tasks.training.train_loss_function import *

ABLATION_ARCHITECTURE = {
    "d_model": 516,
    "motif_num_layers": 4,
    "motif_num_heads": 6,
    "motif_dropout": 0.1,
    "main_att_num_heads": 6,
    "main_att_dropout": 0.1,
    "final_dropout": 0.1,
}

max_num_solvent_molecules = 5
max_num_salt_molecules = 3
molar_ratio_noise_range = 0.03

TARGET_LABEL = "LIC"

kfold_test_seed = 42
kfold_n_splits = 5
kfold_test_size = 0.1

loader_batch_size = 64
num_epochs = 600
max_learning_rate = 7e-5

DATASET_FRACTION = 1


def setup_criterion():
    return ConductivityMSELoss()


def setup_optimizer(model, learning_rate=max_learning_rate):
    weight_decay = 0.01
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def setup_scheduler(optimizer, train_loader, num_epochs, max_lr=max_learning_rate):
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
