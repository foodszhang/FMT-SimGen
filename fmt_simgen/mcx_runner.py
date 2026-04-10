"""
MCX simulation runner for FMT-SimGen.

Handles MCX CLI invocation for 3D fluence volume simulation.
Supports GPU (mcx) and OpenCL/CPU (mcxcl) backends with automatic detection.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_mcx_executable() -> tuple[str, bool]:
    """Detect available MCX executable and whether it supports GPU.

    Returns
    -------
    tuple[str, bool]
        (executable_name, gpu_supported)

    Raises
    ------
    RuntimeError
        If neither mcx nor mcxcl is available.
    """
    import jdata as jd

    # Try GPU version first (mcx)
    for name in ("mcx", "mcx.exe"):
        path = shutil.which(name)
        if path is not None:
            try:
                result = subprocess.run(
                    [name, "-L"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                out = (result.stdout + result.stderr).lower()
                if "gpu" in out or "nvidia" in out:
                    logger.info(f"Detected GPU-capable MCX: {name}")
                    return name, True
                else:
                    logger.warning(f"Detected %s but no GPU capability", name)
            except subprocess.TimeoutExpired:
                logger.warning("Timeout checking %s GPU capability", name)
            except Exception as e:
                logger.warning("Error checking %s GPU capability: %s", name, e)

    # Fall back to CPU version (mcxcl)
    for name in ("mcxcl", "mcxcl.exe"):
        path = shutil.which(name)
        if path is not None:
            logger.info("Using CPU OpenCL MCX: %s", name)
            return name, False

    raise RuntimeError(
        "MCX not found in PATH. Install MCX (GPU) or MCXCL (OpenCL/CPU) and ensure "
        "it is accessible via the 'mcx' or 'mcxcl' command. "
        "See: https://github.com/fangcc/mcx"
    )


def run_mcx_single(
    sample_dir: str | Path,
    mcx_exec: Optional[str] = None,
    gpu_supported: Optional[bool] = None,
) -> str:
    """Run MCX simulation for a single sample.

    Parameters
    ----------
    sample_dir : str | Path
        Path to sample directory containing the MCX JSON config.
    mcx_exec : str | None
        MCX executable name (e.g. "mcx"). Auto-detected if None.
    gpu_supported : bool | None
        Whether GPU mode is supported. Auto-detected if None.

    Returns
    -------
    str
        Absolute path to the generated .jnii file.

    Raises
    ------
    RuntimeError
        If MCX execution fails or output file is not produced.
    """
    sample_dir = Path(sample_dir)

    # Auto-detect MCX if not provided
    if mcx_exec is None:
        mcx_exec, gpu_supported = detect_mcx_executable()

    # Find JSON config (session ID from Session.ID field)
    json_files = list(sample_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON config found in {sample_dir}")

    # Use the JSON with Session.ID matching directory name pattern
    # The JSON is named {sample_id}.json where sample_id matches directory name
    sample_name = sample_dir.name
    json_path = sample_dir / f"{sample_name}.json"
    if not json_path.exists():
        raise RuntimeError(f"Config {json_path} not found in {sample_dir}")

    # Read session ID from JSON to determine output filename
    import json

    with open(json_path) as f:
        config = json.load(f)
    session_id = config.get("Session", {}).get("ID", sample_name)
    output_jnii = sample_dir / f"{session_id}.jnii"

    # Skip if already computed (idempotent)
    if output_jnii.exists():
        logger.info("Skipping %s: %s already exists", sample_dir.name, output_jnii.name)
        return str(output_jnii)

    # Run MCX
    logger.info("Running %s -f %s -a 1 in %s", mcx_exec, json_path.name, sample_dir)
    try:
        result = subprocess.run(
            [mcx_exec, "-f", json_path.name, "-a", "1"],
            cwd=sample_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.stdout:
            logger.debug("MCX stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.debug("MCX stderr: %s", result.stderr.strip())
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MCX timed out after 600s for {sample_dir.name}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"MCX failed for {sample_dir.name}: {e.stderr or e.stdout}"
        )

    # Validate output
    if not output_jnii.exists():
        raise RuntimeError(
            f"MCX completed but output not found: {output_jnii}. "
            f"Check VolumeFile path in JSON config."
        )

    # Verify fluence is non-zero
    try:
        import jdata as jd

        data = jd.loadjd(str(output_jnii))
        nifti_data = data["NIFTIData"] if isinstance(data, dict) else data
        if nifti_data.sum() == 0:
            logger.warning(
                "MCX output for %s is all-zero fluence. "
                "Possible cause: source pattern outside volume or zero pattern.",
                sample_dir.name,
            )
    except Exception as e:
        logger.warning("Could not verify fluence non-zero for %s: %s", output_jnii, e)

    logger.info("MCX completed: %s", output_jnii)
    return str(output_jnii)


def run_mcx_batch(
    samples_dir: str | Path,
    max_workers: int = 1,
    mcx_exec: Optional[str] = None,
) -> list[str]:
    """Run MCX simulations for all samples in a directory.

    Parameters
    ----------
    samples_dir : str | Path
        Root directory containing sample subdirectories.
    max_workers : int
        Number of parallel workers (default 1). GPU simulations
        are typically not parallelized.
    mcx_exec : str | None
        MCX executable name. Auto-detected if None.

    Returns
    -------
    list[str]
        List of successfully generated .jnii paths.
    """
    samples_dir = Path(samples_dir)

    # Detect MCX
    if mcx_exec is None:
        mcx_exec, gpu_supported = detect_mcx_executable()
    else:
        gpu_supported = None

    # Find all sample directories
    sample_dirs = sorted(
        d for d in samples_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {samples_dir}")

    logger.info(
        "Found %d samples, using executable '%s', max_workers=%d",
        len(sample_dirs),
        mcx_exec,
        max_workers,
    )

    # Validate volume file exists (referenced by all JSON configs).
    # The VolumeFile in JSON is relative to sample_dir: ../../../../output/shared/.
    # Compute the absolute path from project_root/output/shared/.
    project_root = Path(__file__).parent.parent
    volume_path = project_root / "output" / "shared" / "mcx_volume_trunk.bin"
    if not volume_path.exists():
        logger.warning(
            "Volume file not found at %s (referenced by MCX configs). "
            "MCX may fail if the path is incorrect.",
            volume_path,
        )

    results: list[str] = []
    errors: list[str] = []

    for sample_dir in sample_dirs:
        try:
            output = run_mcx_single(
                sample_dir,
                mcx_exec=mcx_exec,
                gpu_supported=gpu_supported,
            )
            results.append(output)
        except Exception as e:
            logger.error("Failed %s: %s", sample_dir.name, e)
            errors.append(f"{sample_dir.name}: {e}")
            # Continue batch processing
            continue

    logger.info(
        "Batch complete: %d succeeded, %d failed", len(results), len(errors)
    )
    if errors:
        logger.warning("Failures: %s", "; ".join(errors))

    return results
