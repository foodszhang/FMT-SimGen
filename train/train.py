#!/usr/bin/env python3
"""
MS-GDUN training script for FMT-SimGen.

Usage:
    python train/train.py
"""

import sys
from pathlib import Path

# project root is two levels up from train/train.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train.models.ms_gdun import GCAIN_full
from train.data.dataset import FMTSimGenDataset
from train.losses.criterion import criterion
from train.metrics.evaluation import evaluate_batch, summarize_metrics

# Ensure train/ subpackages are importable
import train.models  # noqa: F401
import train.data    # noqa: F401
import train.losses  # noqa: F401
import train.metrics # noqa: F401


def train():
    # Load config
    config_path = Path(__file__).parent / "config" / "train_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    loss_cfg = cfg["loss"]

    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])
    checkpoint_dir = Path(paths_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "train.txt",
    )
    val_set = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / "val.txt",
    )

    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], shuffle=False)

    # Use actual dimensions from loaded mesh (not config)
    n_nodes = train_set.nodes.shape[0]
    n_surface = train_set.A.shape[0]
    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")
    print(f"Shared assets: {n_nodes} nodes, {n_surface} surface nodes")

    # Verify config matches
    cfg_n_nodes = model_cfg.get("n_nodes", n_nodes)
    if cfg_n_nodes != n_nodes:
        print(f"WARNING: config n_nodes={cfg_n_nodes} != actual {n_nodes}, using actual")
    cfg_n_surface = model_cfg.get("n_surface", n_surface)
    if cfg_n_surface != n_surface:
        print(f"WARNING: config n_surface={cfg_n_surface} != actual {n_surface}, using actual")

    # Move shared assets to GPU
    A = train_set.A.cuda()           # [S, N]
    L = train_set.L.cuda()           # [N, N]
    L0 = train_set.L0.cuda()
    L1 = train_set.L1.cuda()
    L2 = train_set.L2.cuda()
    L3 = train_set.L3.cuda()
    knn_idx = train_set.knn_idx.cuda()
    sens_w = train_set.sens_w.cuda()
    nodes = train_set.nodes.cuda()

    print(f"A GPU memory: {A.element_size() * A.nelement() / 1e6:.1f} MB")
    print(f"L (dense) GPU memory: {L.element_size() * L.nelement() / 1e6:.1f} MB")

    # Model
    model = GCAIN_full(
        L=L,
        A=A,
        L0=L0,
        L1=L1,
        L2=L2,
        L3=L3,
        knn_idx=knn_idx,
        sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg["scheduler_patience"],
        factor=0.5,
    )

    best_val_loss = float("inf")

    for epoch in range(train_cfg["max_epochs"]):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            b = batch["b"].cuda()    # [B, S, 1]
            gt = batch["gt"].cuda() # [B, N, 1]

            X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
            pred = model(X0, b)  # [B, N, 1]

            loss = criterion(
                pred, gt, nodes,
                weight_dice=loss_cfg["weight_dice"],
                weight_le=loss_cfg["weight_le"],
                weight_mse=loss_cfg["weight_mse"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss_mean = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_metrics = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                b = batch["b"].cuda()
                gt = batch["gt"].cuda()
                X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
                pred = model(X0, b)
                loss = criterion(
                    pred, gt, nodes,
                    weight_dice=loss_cfg["weight_dice"],
                    weight_le=loss_cfg["weight_le"],
                    weight_mse=loss_cfg["weight_mse"],
                )
                val_losses.append(loss.item())
                val_metrics.append(evaluate_batch(pred, gt, nodes))

        val_loss_mean = sum(val_losses) / len(val_losses)
        val_summary = summarize_metrics(val_metrics)

        scheduler.step(val_loss_mean)

        print(
            f"Epoch {epoch+1:03d}/{train_cfg['max_epochs']} | "
            f"Train: {train_loss_mean:.4f} | "
            f"Val: {val_loss_mean:.4f} | "
            f"Dice: {val_summary['dice']:.4f} | "
            f"LE: {val_summary['location_error']:.4f} | "
            f"MSE: {val_summary['mse']:.6f}"
        )

        # Save best
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            ckpt_path = checkpoint_dir / "best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best model to {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
