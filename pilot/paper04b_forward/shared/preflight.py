"""Pre-flight checks for MCX simulation consistency.

Provides safety checks to prevent voxel_size mismatches between
config, MCX JSON, and closed-form calculations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def assert_voxel_consistency(
    mcx_json_path: Union[str, Path],
    config_voxel_mm: float,
    tolerance: float = 1e-9,
) -> float:
    """Assert that MCX JSON LengthUnit matches config voxel_size.

    This is the "0.4 bug never recurs" insurance check.

    Parameters
    ----------
    mcx_json_path : str or Path
        Path to MCX JSON config file.
    config_voxel_mm : float
        Expected voxel size from config (mm).
    tolerance : float
        Allowed difference (default 1e-9 for exact match).

    Returns
    -------
    float
        The MCX JSON LengthUnit value.

    Raises
    ------
    AssertionError
        If voxel sizes don't match.
    FileNotFoundError
        If MCX JSON file doesn't exist.
    KeyError
        If Domain.LengthUnit not found in JSON.
    """
    mcx_json_path = Path(mcx_json_path)
    if not mcx_json_path.exists():
        raise FileNotFoundError(f"MCX JSON not found: {mcx_json_path}")

    with open(mcx_json_path) as f:
        cfg = json.load(f)

    lu = cfg["Domain"]["LengthUnit"]

    diff = abs(lu - config_voxel_mm)
    if diff > tolerance:
        raise AssertionError(
            f"VOXEL SIZE MISMATCH: config={config_voxel_mm}mm, "
            f"MCX JSON LengthUnit={lu}mm, diff={diff:.2e}mm\n"
            f"This is the '0.4 bug' - check your configuration!"
        )

    logger.debug(f"Voxel consistency OK: config={config_voxel_mm}mm, MCX={lu}mm")
    return lu


def assert_volume_shape_consistency(
    volume_shape: tuple,
    voxel_size_mm: float,
    expected_phys_size_mm: tuple = None,
    tolerance_mm: float = 0.1,
) -> None:
    """Assert volume physical dimensions match expectations.

    Parameters
    ----------
    volume_shape : tuple
        Volume shape (NX, NY, NZ).
    voxel_size_mm : float
        Voxel size in mm.
    expected_phys_size_mm : tuple, optional
        Expected physical size (X_mm, Y_mm, Z_mm).
    tolerance_mm : float
        Allowed difference in mm.
    """
    if expected_phys_size_mm is None:
        return

    nx, ny, nz = volume_shape
    phys_x = nx * voxel_size_mm
    phys_y = ny * voxel_size_mm
    phys_z = nz * voxel_size_mm

    exp_x, exp_y, exp_z = expected_phys_size_mm

    for dim, actual, expected in [
        ("X", phys_x, exp_x),
        ("Y", phys_y, exp_y),
        ("Z", phys_z, exp_z),
    ]:
        diff = abs(actual - expected)
        if diff > tolerance_mm:
            raise AssertionError(
                f"Physical size mismatch in {dim}: "
                f"actual={actual:.1f}mm, expected={expected:.1f}mm, diff={diff:.1f}mm"
            )

    logger.debug(
        f"Volume shape OK: {volume_shape} @ {voxel_size_mm}mm = "
        f"({phys_x:.1f}, {phys_y:.1f}, {phys_z:.1f})mm"
    )


def preflight_check(
    mcx_json_path: Union[str, Path],
    config_voxel_mm: float,
    volume_shape: tuple = None,
    expected_phys_size_mm: tuple = None,
) -> dict:
    """Run all pre-flight checks before MCX simulation.

    Parameters
    ----------
    mcx_json_path : str or Path
        Path to MCX JSON config.
    config_voxel_mm : float
        Config voxel size in mm.
    volume_shape : tuple, optional
        Volume shape for physical size check.
    expected_phys_size_mm : tuple, optional
        Expected physical dimensions.

    Returns
    -------
    dict
        Check results with keys: voxel_mm, passed, messages.
    """
    results = {
        "voxel_mm": None,
        "passed": True,
        "messages": [],
    }

    try:
        lu = assert_voxel_consistency(mcx_json_path, config_voxel_mm)
        results["voxel_mm"] = lu
        results["messages"].append(f"Voxel consistency: OK ({lu}mm)")
    except (AssertionError, FileNotFoundError, KeyError) as e:
        results["passed"] = False
        results["messages"].append(f"Voxel consistency: FAILED - {e}")

    if volume_shape is not None and expected_phys_size_mm is not None:
        try:
            assert_volume_shape_consistency(
                volume_shape, config_voxel_mm, expected_phys_size_mm
            )
            results["messages"].append("Physical size: OK")
        except AssertionError as e:
            results["passed"] = False
            results["messages"].append(f"Physical size: FAILED - {e}")

    if results["passed"]:
        logger.info("Pre-flight check PASSED")
    else:
        logger.error("Pre-flight check FAILED: %s", results["messages"])

    return results
