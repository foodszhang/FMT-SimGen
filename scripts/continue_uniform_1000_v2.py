#!/usr/bin/env python3
"""Continue generating remaining samples for uniform_1000_v2 experiment."""
import sys
sys.path.insert(0, "/home/foods/pro/FMT-SimGen")

import json
import logging
from pathlib import Path
from fmt_simgen.dataset.builder import DatasetBuilder
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT = "uniform_1000_v2"
CONFIG_PATH = Path("/home/foods/pro/FMT-SimGen/config/uniform_1000_v2.yaml")
DATA_DIR = Path(f"/home/foods/pro/FMT-SimGen/data/{EXPERIMENT}")
SAMPLES_DIR = DATA_DIR / "samples"
TARGET = 1000
START_FROM = 171  # we already have 0-170 (171 samples)

# Load config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Count existing samples
existing = sorted([int(d.name.split("_")[1]) for d in SAMPLES_DIR.iterdir() if d.is_dir()])
logger.info(f"Existing samples: {len(existing)}, range {existing[0]}-{existing[-1] if existing else 'N/A'}")

# Build dataset (reuses existing shared assets)
builder = DatasetBuilder(config)
builder.setup()  # will skip regeneration of shared assets

# We need to call _generate_sample once at a time to fill remaining slots
# The builder's generate() method has the full loop; let's use it with a trick:
# Patch the manifest to start from sample 171
import numpy as np
from fmt_simgen.dataset.manifest import Manifest

manifest_path = DATA_DIR / "manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    logger.info(f"Manifest has {len(manifest['samples'])} entries")
else:
    manifest = None
    logger.warning("No manifest found, starting fresh")

# Generate remaining samples using the internal generate flow
# We replicate the per-sample loop from builder.generate()
from fmt_simgen.tumor.tumor_generator import TumorGenerator
from fmt_simgen.sampling.dual_sampler import DualSampler
from fmt_simgen.fem.fem_solver import FEMSolver

tumor_gen = TumorGenerator(config)
dual_sampler = DualSampler(config, builder.mesh_data, builder.fem_solver)
fem_solver = builder.fem_solver

num_foci_dist = config["tumor"]["num_foci_distribution"]
foci_vals = list(num_foci_dist.keys())
foci_probs = list(num_foci_dist.values())

depth_config = config["tumor"]["depth_distribution"]
tiers = list(depth_config.keys())
weights = [depth_config[t]["weight"] for t in tiers]
depth_ranges = {t: depth_config[t]["range"] for t in tiers}

# Pre-compute remaining plan
remaining = TARGET - len(existing)
plan = []
for i in range(remaining):
    tier = np.random.choice(tiers, p=weights)
    n_foci = int(np.random.choice(foci_vals, p=foci_probs))
    lo, hi = depth_ranges[tier]
    depth_mm = float(np.random.uniform(lo, hi))
    plan.append((n_foci, depth_mm, tier))

logger.info(f"Generating {remaining} remaining samples...")

saved = 0
quality_filter = config.get("quality_filter", {})
filter_enabled = quality_filter.get("enabled", True)
min_b_max = quality_filter.get("min_b_max", 0.001)
min_gt_frac = quality_filter.get("min_gt_nonzero_frac", 0.001)
min_gt_nonzero_count = quality_filter.get("min_gt_nonzero_count", 20)
max_retries = quality_filter.get("max_retries", 10)

for idx, (n_foci, depth_mm, depth_tier) in enumerate(plan):
    sample_id = existing[-1] + 1 + idx if idx > 0 else existing[-1] + 1
    logger.info(f"  Generating sample {saved + 1}/{remaining} (global #{sample_id})...")

    organ_failed_all_attempts = True
    quality_passed = False

    for attempt in range(max_retries + 1):
        tumor_sample = tumor_gen.generate_sample(
            num_foci=n_foci, depth_mm=depth_mm, depth_tier=depth_tier
        )

        gt_nodes = dual_sampler.sample_to_nodes(tumor_sample)
        measurement_b = fem_solver.forward(gt_nodes)

        b_max = float(np.max(np.abs(measurement_b)))
        gt_nonzero_frac = float(np.count_nonzero(gt_nodes)) / len(gt_nodes)
        gt_nonzero_count = int(np.count_nonzero(gt_nodes))

        b_ok = b_max >= min_b_max
        frac_ok = gt_nonzero_frac >= min_gt_frac
        count_ok = gt_nonzero_count >= min_gt_nonzero_count
        organ_ok = getattr(tumor_sample, "_organ_constraint_passed", True)

        quality_this_attempt = b_ok and frac_ok and count_ok
        organ_failed_all_attempts = organ_failed_all_attempts and not organ_ok

        if (not filter_enabled or quality_this_attempt) and organ_ok:
            gt_voxels = dual_sampler.sample_to_voxels(tumor_sample)
            quality_passed = True

            # Save
            out_dir = SAMPLES_DIR / f"sample_{sample_id:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "tumor_params.json", "w") as f:
                json.dump(tumor_sample.to_json(), f, indent=2)
            np.save(out_dir / "measurement_b.npy", measurement_b)
            np.save(out_dir / "gt_nodes.npy", gt_nodes)
            np.save(out_dir / "gt_voxels.npy", gt_voxels)

            logger.info(f"    Saved to {out_dir.name}")
            saved += 1
            break
        else:
            msg = "organ constraint" if not organ_ok else "quality filter"
            logger.warning(f"    Attempt {attempt+1}: failed ({msg})")

    if not quality_passed:
        logger.warning(f"    Sample {sample_id}: all attempts failed, skipping")

logger.info(f"Done. Generated {saved}/{remaining} additional samples")
logger.info(f"Total samples: {len(existing) + saved}/{TARGET}")
