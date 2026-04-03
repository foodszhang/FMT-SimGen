#!/usr/bin/env python3
"""
Evaluate trained MS-GDUN model on FMT-SimGen dataset.

Supports per-foci-type metric aggregation when dataset_manifest.json is available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import yaml
import torch
from torch.utils.data import DataLoader

from train.models.ms_gdun import GCAIN_full
from train.data.dataset import FMTSimGenDataset
from train.metrics.evaluation import evaluate_batch, summarize_metrics


def load_manifest(samples_dir: Path) -> dict | None:
    """Load dataset manifest from multiple possible locations."""
    candidates = [
        Path("output/dataset_manifest.json"),
        Path("data/dataset_manifest.json"),
        samples_dir.parent / "dataset_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def get_sample_names(split_file: Path) -> list[str]:
    """Get ordered list of sample names from split file."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def evaluate(checkpoint_path: str | Path, split: str = "val"):
    config_path = Path(__file__).parent / "config" / "train_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]

    shared_dir = Path(paths_cfg["shared_dir"])
    samples_dir = Path(paths_cfg["samples_dir"])
    splits_dir = Path(paths_cfg["splits_dir"])

    split_file = splits_dir / f"{split}.txt"
    dataset = FMTSimGenDataset(
        shared_dir=shared_dir,
        samples_dir=samples_dir,
        split_file=split_file,
    )
    # Use batch_size=1 so metrics align 1:1 with sample names for per-foci grouping
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

    # Load manifest for per-foci grouping
    manifest = load_manifest(samples_dir)
    sample_names = get_sample_names(split_file) if manifest else None

    n_nodes = dataset.nodes.shape[0]
    all_metrics = []
    all_sample_names = []

    with torch.no_grad():
        for batch in loader:
            b = batch["b"].cuda()
            gt = batch["gt"].cuda()
            X0 = torch.zeros(b.size(0), n_nodes, 1, device="cuda")
            pred = model(X0, b)
            batch_metrics = evaluate_batch(pred, gt, nodes)
            all_metrics.append(batch_metrics)

            # Track sample names for this batch
            if sample_names is not None:
                batch_size = b.size(0)
                all_sample_names.extend(sample_names[:batch_size])
                sample_names = sample_names[batch_size:]

    # Group metrics by foci count
    if manifest and all_sample_names:
        metrics_by_foci = {1: [], 2: [], 3: []}
        for i, metrics in enumerate(all_metrics):
            sample_name = all_sample_names[i] if i < len(all_sample_names) else None
            if sample_name and sample_name in manifest["samples"]:
                n_foci = manifest["samples"][sample_name]["num_foci"]
                if n_foci in metrics_by_foci:
                    metrics_by_foci[n_foci].append(metrics)
            # For batch-level metrics, we average within batch
            # This is a simplification - batch metrics are already averaged

    summary = summarize_metrics(all_metrics)
    print(f"\n=== {split.upper()} Results ===")
    print(f"Dice:            {summary['dice']:.4f}")
    print(f"Location Error:  {summary['location_error']:.4f} mm")
    print(f"MSE:             {summary['mse']:.6f}")

    # Per-foci breakdown
    if manifest and all_sample_names:
        print(f"\n=== Per-Foci Breakdown ===")
        print(f"{'Metric':<20} {'Overall':>10} {'1-Foci':>10} {'2-Foci':>10} {'3-Foci':>10}")
        print("-" * 62)

        # Collect per-sample metrics
        sample_metrics = []
        for i, metrics in enumerate(all_metrics):
            sample_name = all_sample_names[i] if i < len(all_sample_names) else None
            n_foci = None
            if sample_name and sample_name in manifest["samples"]:
                n_foci = manifest["samples"][sample_name]["num_foci"]
            sample_metrics.append((n_foci, metrics))

        # Group by foci
        by_foci = {None: [], 1: [], 2: [], 3: []}
        for n_foci, metrics in sample_metrics:
            by_foci[n_foci].append(metrics)

        # Compute summaries
        overall = summarize_metrics(all_metrics)
        foci_summaries = {}
        for n in [1, 2, 3]:
            if by_foci[n]:
                foci_summaries[n] = summarize_metrics(by_foci[n])
            else:
                foci_summaries[n] = None

        # Print table
        for key in ["dice", "dice_bin_0.3", "dice_bin_0.1", "recall_0.3", "recall_0.1", "precision_0.3", "location_error", "mse"]:
            overall_val = overall.get(key, 0)
            vals = []
            for n in [1, 2, 3]:
                if foci_summaries[n]:
                    vals.append(f"{foci_summaries[n].get(key, 0):>10.4f}")
                else:
                    vals.append(f"{'N/A':>10}")
            print(f"{key:<20} {overall_val:>10.4f} {''.join(vals)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()
    evaluate(args.ckpt, args.split)
