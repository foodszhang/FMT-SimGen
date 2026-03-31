#!/usr/bin/env python3
"""
Step 0: Generate mesh and system matrix.

This script generates the tetrahedral FEM mesh and assembles the system matrix.
It should be run once before generating samples, as these are shared assets.

Usage:
    python scripts/01_generate_mesh.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from fmt_simgen.dataset.builder import DatasetBuilder


def main():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    builder = DatasetBuilder(config)

    print("=" * 60)
    print("Step 0: Generating shared assets (mesh + system matrix)")
    print("=" * 60)

    assets = builder.build_shared_assets(force_regenerate=True)

    print("\nAsset generation complete!")
    print(f"  Mesh file: {assets['mesh']}")
    print(f"  System matrix prefix: {assets['matrix']}")


if __name__ == "__main__":
    main()
