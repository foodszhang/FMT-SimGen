#!/usr/bin/env python3
"""
Evaluate trained MS-GDUN model on FMT-SimGen dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from train.models.ms_gdun import GCAIN_full
from train.data.dataset import FMTSimGenDataset
from train.metrics.evaluation import evaluate_batch, summarize_metrics


def evaluate(checkpoint_path: str | Path, split: str = "val"):
    config_path = Path(__file__).parent / "config" / "train_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])

    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=splits_dir / f"{split}.txt",
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    A = dataset.A.cuda()
    L = dataset.L.cuda()
    L0 = dataset.L0.cuda()
    L1 = dataset.L1.cuda()
    L2 = dataset.L2.cuda()
    L3 = dataset.L3.cuda()
    knn_idx = dataset.knn_idx.cuda()
    sens_w = dataset.sens_w.cuda()
    nodes = dataset.nodes.cuda()

    model = GCAIN_full(
        L=L, A=A,
        L0=L0, L1=L1, L2=L2, L3=L3,
        knn_idx=knn_idx,
        sens_w=sens_w,
        num_layer=model_cfg["num_layer"],
        feat_dim=model_cfg["feat_dim"],
    ).cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint)
    model.eval()

    n_nodes = dataset.nodes.shape[0]
    all_metrics = []
    with torch.no_grad():
        for batch in loader:
            b = batch["b"].cuda()
            gt = batch["gt"].cuda()
            X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
            pred = model(X0, b)
            all_metrics.append(evaluate_batch(pred, gt, nodes))

    summary = summarize_metrics(all_metrics)
    print(f"\n=== {split.upper()} Results ===")
    print(f"Dice:            {summary['dice']:.4f}")
    print(f"Location Error:  {summary['location_error']:.4f} mm")
    print(f"MSE:             {summary['mse']:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()
    evaluate(args.ckpt, args.split)
