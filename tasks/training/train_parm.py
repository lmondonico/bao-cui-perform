"""PerForm training settings."""

import torch
import sys
import os


sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tasks.training.train_loss_function import *

from modules.perform.perform_parm import ARCHITECTURE

max_num_solvent_molecules = 5
max_num_salt_molecules = 3
molar_ratio_noise_range = 0.03

TARGET_LABEL = "LIC"

kfold_test_seed = 42
kfold_n_splits = 5
kfold_test_size = 0.1

loader_batch_size = 64
num_epochs = 600
PATIENCE = num_epochs
max_learning_rate = 7e-5

DATASET_FRACTION = 1


def setup_criterion():
    """Initialize and return the loss function."""
    return ConductivityMSELoss()


def setup_optimizer(model, learning_rate=max_learning_rate):
    """Initialize and return the optimizer."""
    weight_decay = 0.01
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def setup_scheduler(optimizer, train_loader, num_epochs, max_lr=max_learning_rate):
    """Initialize and return the learning rate scheduler."""
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
