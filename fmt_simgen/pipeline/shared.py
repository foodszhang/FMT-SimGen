"""Shared utilities for FMT-SimGen pipeline entry points."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config_with_inheritance(config_path: str) -> dict:
    """Load configuration file with _base_ inheritance support.

    Parameters
    ----------
    config_path : str
        Path to the config file.

    Returns
    -------
    dict
        Merged configuration dictionary with all _base_ keys resolved.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_name = cfg.pop("_base_", None)
    if base_name:
        base_path = config_path.parent / base_name
        base_cfg = load_config_with_inheritance(str(base_path))
        # Deep merge: base_cfg gets merged with cfg overrides
        _deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg


def derive_samples_dir(config: dict) -> Path:
    """Derive the samples directory path from config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary with ``dataset`` section.

    Returns
    -------
    Path
        ``{output_path}/{experiment_name}/samples/``
    """
    dataset_cfg = config.get("dataset", {})
    experiment_name = dataset_cfg.get("experiment_name", "default")
    base_output = Path(dataset_cfg.get("output_path", "data/"))
    return base_output / experiment_name / "samples"
