"""
Loss function for PerForm model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConductivityMSELoss(nn.Module):
    def __init__(self):
        """MSE loss for conductivity prediction."""
        super().__init__()

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        return mse_loss
