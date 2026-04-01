"""
Evaluation metrics for FMT reconstruction.
"""

import torch
from train.losses.criterion import dice_coeff, location_error


def evaluate_batch(
    pred: torch.Tensor,
    label: torch.Tensor,
    nodes: torch.Tensor,
) -> dict:
    """Compute all metrics for a batch.

    pred, label: (B, N, 1)
    nodes: (N, 3)
    """
    dice = dice_coeff(pred, label).item()
    le = location_error(pred, label, nodes).item()
    mse = torch.nn.functional.mse_loss(pred, label).item()
    return {
        "dice": dice,
        "location_error": le,
        "mse": mse,
    }


def summarize_metrics(metric_dicts: list[dict]) -> dict:
    """Average metrics over a list of batch results."""
    keys = metric_dicts[0].keys()
    return {k: sum(d[k] for d in metric_dicts) / len(metric_dicts) for k in keys}
