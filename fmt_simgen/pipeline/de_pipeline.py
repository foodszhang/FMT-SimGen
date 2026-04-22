"""DE pipeline entry point for FMT-SimGen.

Handles Steps 1-4: mesh + system matrix generation, then N sample generation.
"""

from __future__ import annotations

import logging
from typing import Optional

from fmt_simgen.dataset.builder import DatasetBuilder

logger = logging.getLogger(__name__)


def run_de_pipeline(
    config: dict,
    num_samples: Optional[int] = None,
    start_index: int = 0,
) -> None:
    """Run the DE (Diffuse Optical) pipeline.

    Parameters
    ----------
    config : dict
        Already-loaded configuration dictionary. Must contain keys:
        ``dataset``, ``tumor``, ``physics``, ``atlas``, ``mesh``.
        NOT a file path — caller is responsible for loading.
    num_samples : int, optional
        Number of samples to generate. Uses
        ``config["dataset"]["num_samples"]`` if not specified.
    start_index : int, default 0
        Starting sample index. Complete samples at or after start_index
        are skipped (supports batched resumption).
    """
    dataset_cfg = config.get("dataset", {})
    experiment_name = dataset_cfg.get("experiment_name", "default")

    logger.info("=" * 60)
    logger.info(
        "Phase DE: Generating %s samples for experiment '%s'",
        "N" if num_samples is None else num_samples,
        experiment_name,
    )
    logger.info("=" * 60)

    builder = DatasetBuilder(config)
    builder.build_samples(num_samples=num_samples, start_index=start_index)

    logger.info("DE phase complete: samples in data/%s/samples/", experiment_name)
