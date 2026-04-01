"""
Loss functions for FMT reconstruction.

Follows the reference Critic.py from MS_GDUN_for_MICCAI2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Soft Dice coefficient (no binarization).

    pred, label: (B, N, 1) or (B, N)
    Returns: scalar tensor
    """
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
        label = label.squeeze(-1)
    intersection = (pred * label).sum()
    return (2 * intersection + eps) / (pred.sum() + label.sum() + eps)


def location_error(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Location error: Euclidean distance between centroids.

    centroid = sum(val * coord) / sum(val)

    pred, label: (B, N, 1)
    nodes: (N, 3)
    Returns: scalar mean over batch
    """
    # Mass
    pred_mass = pred.squeeze(-1).sum(dim=1, keepdim=True) + eps  # [B, 1]
    label_mass = label.squeeze(-1).sum(dim=1, keepdim=True) + eps

    # Weighted centroid — keep (B,1) shape for broadcasting with (B,3) result
    pred_centroid = torch.matmul(pred.squeeze(-1), nodes) / pred_mass  # [B, 3]
    label_centroid = torch.matmul(label.squeeze(-1), nodes) / label_mass  # [B, 3]

    return torch.norm(pred_centroid - label_centroid, dim=-1).mean()


def criterion(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
    weight_dice: float = 0.7,
    weight_le: float = 0.2,
    weight_mse: float = 0.1,
) -> torch.Tensor:
    """Combined loss: 0.7 * (1-dice) + 0.2 * le + 0.1 * mse.

    pred, label: (B, N, 1)
    nodes: (N, 3)
    """
    dice = dice_coeff(pred, label)
    le = location_error(pred, label, nodes)
    mse = F.mse_loss(pred, label)
    return weight_dice * (1 - dice) + weight_le * le + weight_mse * mse
