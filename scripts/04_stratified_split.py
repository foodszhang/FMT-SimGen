#!/usr/bin/env python3
"""
Create stratified train/val split based on foci counts and depth tiers.

Split rules (1000 samples):
- Total: 1000 samples, 80% train / 20% val
- 1-foci: 300 total → train 240, val 60
- 2-foci: 350 total → train 280, val 70
- 3-foci: 350 total → train 280, val 70

Outputs:
- train/splits/train.txt (800 lines, sample_XXXX only)
- train/splits/val.txt (200 lines, sample_XXXX only)
- train/splits/train_with_foci.txt (tab-separated: sample_XXXX\tnum_foci\tdepth_tier)
- train/splits/val_with_foci.txt (tab-separated: sample_XXXX\tnum_foci\tdepth_tier)
- output/dataset_manifest.json (full metadata with cross-tabulation)

Usage:
    python scripts/04_stratified_split.py
    python scripts/04_stratified_split.py --n_total 1000
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def main():
    samples_dir = Path("data/samples")
    splits_dir = Path("train/splits")
    manifest_path = Path("output/dataset_manifest.json")

    # Ensure directories exist
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all samples with their foci counts
    samples = {}
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.name.startswith("sample_"):
            continue
        sample_name = sample_dir.name

        params_path = sample_dir / "tumor_params.json"
        if not params_path.exists():
            print(f"WARNING: {params_path} not found, skipping {sample_name}")
            continue

        with open(params_path) as f:
            params = json.load(f)

        num_foci = params.get("num_foci", len(params.get("foci", [])))
        samples[sample_name] = {
            "num_foci": num_foci,
            "foci_details": params.get("foci", []),
        }

    if len(samples) != 1000:
        print(f"WARNING: Found {len(samples)} samples, expected 1000")

    # Group by foci count
    by_foci = {1: [], 2: [], 3: []}
    for name, info in samples.items():
        n = info["num_foci"]
        if n in by_foci:
            by_foci[n].append(name)
        else:
            print(f"WARNING: Unknown foci count {n} for {name}")

    # Shuffle within each group with seed
    rng = np.random.default_rng(42)
    for n in by_foci:
        rng.shuffle(by_foci[n])

    # Stratified split for 1000 samples: 80/20
    train_counts = {1: 240, 2: 280, 3: 280}
    val_counts = {1: 60, 2: 70, 3: 70}

    train_samples = []
    val_samples = []

    for n_foci, count in train_counts.items():
        train_samples.extend(by_foci[n_foci][:count])

    for n_foci, count in val_counts.items():
        val_samples.extend(by_foci[n_foci][:count])

    # Verify counts
    train_by_foci = {1: 0, 2: 0, 3: 0}
    val_by_foci = {1: 0, 2: 0, 3: 0}
    for name in train_samples:
        train_by_foci[samples[name]["num_foci"]] += 1
    for name in val_samples:
        val_by_foci[samples[name]["num_foci"]] += 1

    print(f"Train samples: {len(train_samples)} ({train_by_foci})")
    print(f"Val samples: {len(val_samples)} ({val_by_foci})")

    # Write train.txt and val.txt
    with open(splits_dir / "train.txt", "w") as f:
        for name in sorted(train_samples):
            f.write(name + "\n")

    with open(splits_dir / "val.txt", "w") as f:
        for name in sorted(val_samples):
            f.write(name + "\n")

    # Write with_foci versions
    with open(splits_dir / "train_with_foci.txt", "w") as f:
        for name in sorted(train_samples):
            n = samples[name]["num_foci"]
            f.write(f"{name}\t{n}\n")

    with open(splits_dir / "val_with_foci.txt", "w") as f:
        for name in sorted(val_samples):
            n = samples[name]["num_foci"]
            f.write(f"{name}\t{n}\n")

    print(f"\nSaved to {splits_dir}/")

    # Build manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "seed": 42,
        "foci_distribution": {
            "1": 300,
            "2": 350,
            "3": 350,
        },
        "split": {
            "train": 800,
            "val": 200,
            "train_by_foci": train_by_foci,
            "val_by_foci": val_by_foci,
        },
        "samples": {},
    }

    # Add sample info with split assignment
    train_set = set(train_samples)
    for name, info in sorted(samples.items()):
        split = "train" if name in train_set else "val"
        manifest["samples"][name] = {
            "num_foci": info["num_foci"],
            "split": split,
            "foci_details": info["foci_details"],
        }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path}")

    # Verify val coverage
    val_foci = {1: 0, 2: 0, 3: 0}
    for name in val_samples:
        val_foci[samples[name]["num_foci"]] += 1
    print(f"\nVal set foci coverage: {val_foci}")
    assert val_foci == val_counts, f"Val coverage mismatch! Expected {val_counts}, got {val_foci}"

    print("\nDone!")


if __name__ == "__main__":
    main()
