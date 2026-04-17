#!/usr/bin/env python3
"""Stage 2: Uniform Volume Source Validation.

Validates whether multi-point cubature can accurately approximate
voxelized uniform volume source surface projections.

MCX: Voxelized uniform spherical/ellipsoidal source (all voxels emit)
Analytic: SR-6 / grid-27 / stratified-33 multi-point sampling + surface-aware Green
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fmt_simgen.mcx_projection import project_volume_reference
from surface_projection import (
    compute_ncc,
    compute_rmse,
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)
from source_quadrature import sample_uniform

logger = logging.getLogger(__name__)

# MCX executable
MCX_EXE = "/mnt/f/win-pro/bin/mcx.exe"

# Default tissue parameters (homogeneous)
DEFAULT_TISSUE_PARAMS = {
    "mua_mm": 0.0236,  # 1/mm
    "mus_prime_mm": 0.89,  # 1/mm
    "g": 0.9,
    "n": 1.37,
}

# Camera parameters - match multiposition test (Stage 1.5) for direct comparison
# NOTE: These differ from the original Stage 2 proposal (25/22/113) to enable
# direct comparison with multiposition results. The larger FOV (50mm vs 22mm)
# and higher resolution (256 vs 113) provide better angular coverage for
# multi-position evaluation.
CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)
VOXEL_SIZE_MM = 0.2

# Atlas volume parameters (from Stage 1.5)
# Volume is in centered coordinates (origin at volume center)
ATLAS_VOLUME_SHAPE = (190, 200, 104)  # XYZ
# Y center is at slice 100 (ny/2), physical Y=0 corresponds to that slice
# For trunk region, Y=2.4mm is the center slice


def create_voxelized_uniform_sphere(
    center_mm: np.ndarray,
    radius_mm: float,
    shape: Tuple[int, int, int] = ATLAS_VOLUME_SHAPE,
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Create a voxelized uniform spherical source mask.

    Parameters
    ----------
    center_mm : np.ndarray
        Sphere center in mm (centered coordinates).
    radius_mm : float
        Sphere radius in mm.
    shape : tuple
        Output volume shape (X, Y, Z).
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    source_mask : np.ndarray
        Binary mask (X, Y, Z) where 1 indicates source voxel.
    """
    nx, ny, nz = shape

    # Create coordinate grids
    x = (np.arange(nx) - nx / 2 + 0.5) * voxel_size_mm
    y = (np.arange(ny) - ny / 2 + 0.5) * voxel_size_mm
    z = (np.arange(nz) - nz / 2 + 0.5) * voxel_size_mm

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Distance from sphere center
    r = np.sqrt(
        (xx - center_mm[0]) ** 2 + (yy - center_mm[1]) ** 2 + (zz - center_mm[2]) ** 2
    )

    # Source mask
    source_mask = (r <= radius_mm).astype(np.uint8)

    return source_mask


def create_voxelized_uniform_ellipsoid(
    center_mm: np.ndarray,
    axes_mm: np.ndarray,
    shape: Tuple[int, int, int] = ATLAS_VOLUME_SHAPE,
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Create a voxelized uniform ellipsoidal source mask.

    Parameters
    ----------
    center_mm : np.ndarray
        Ellipsoid center in mm (centered coordinates).
    axes_mm : np.ndarray
        Semi-axis lengths [a, b, c] in mm.
    shape : tuple
        Output volume shape (X, Y, Z).
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    source_mask : np.ndarray
        Binary mask (X, Y, Z) where 1 indicates source voxel.
    """
    nx, ny, nz = shape

    # Create coordinate grids
    x = (np.arange(nx) - nx / 2 + 0.5) * voxel_size_mm
    y = (np.arange(ny) - ny / 2 + 0.5) * voxel_size_mm
    z = (np.arange(nz) - nz / 2 + 0.5) * voxel_size_mm

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Normalized distance (ellipsoid equation)
    dist_sq = (
        ((xx - center_mm[0]) / axes_mm[0]) ** 2
        + ((yy - center_mm[1]) / axes_mm[1]) ** 2
        + ((zz - center_mm[2]) / axes_mm[2]) ** 2
    )

    # Source mask
    source_mask = (dist_sq <= 1.0).astype(np.uint8)

    return source_mask


def create_atlas_with_source(
    source_mask: np.ndarray,
    shape: Tuple[int, int, int] = ATLAS_VOLUME_SHAPE,
) -> np.ndarray:
    """Create atlas volume with embedded source.

    Label 1 = tissue, Label 2 = source (same optical params)

    Parameters
    ----------
    source_mask : np.ndarray
        Binary source mask (X, Y, Z).
    shape : tuple
        Volume shape.

    Returns
    -------
    volume : np.ndarray
        Label volume with source.
    """
    volume = np.ones(shape, dtype=np.uint8)
    volume[source_mask > 0] = 2
    return volume


def save_volume_to_file(volume: np.ndarray, filepath: Path):
    """Save volume to binary file for MCX."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    volume.astype(np.uint8).tofile(filepath)
    logger.info("Saved volume to %s (shape=%s)", filepath, volume.shape)


def create_mcx_config(
    volume_file: Path,
    output_dir: Path,
    source_mask: np.ndarray,
    tissue_params: dict,
    n_photons: int = 1e8,
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> dict:
    """Create MCX configuration for voxelized uniform source.

    Uses pattern3d source type for volume source.

    Parameters
    ----------
    volume_file : Path
        Path to volume binary file.
    output_dir : Path
        Output directory for MCX results.
    source_mask : np.ndarray
        Source mask for pattern3d.
    tissue_params : dict
        Tissue optical parameters.
    n_photons : int
        Number of photons to simulate.
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    config : dict
        MCX configuration dictionary.
    """
    nx, ny, nz = ATLAS_VOLUME_SHAPE

    # Extract source voxels positions
    source_indices = np.argwhere(source_mask > 0)  # [N, 3] in [ix, iy, iz]

    if len(source_indices) == 0:
        raise ValueError("Empty source mask")

    logger.info("Source has %d voxels", len(source_indices))

    # For uniform source, each voxel emits equally
    # We'll use multiple point sources approach for simplicity
    # (pattern3d can be added later if needed)

    config = {
        "Version": "Octave/Matlab/JSON",
        "Session": {
            "ID": output_dir.name,
            "Photons": int(n_photons),
            "DoAutoThread": 1,
            "DoSaveVolume": 0,
            "DoSaveTrajectory": 0,
        },
        "Forward": {"T0": 0.0, "T1": 5e-08, "Dt": 5e-08},
        "Domain": {
            "OriginType": 1,
            "VolumeFile": str(volume_file),
            "Dim": [int(nx), int(ny), int(nz)],
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},  # Label 0: air
                {
                    "mua": tissue_params["mua_mm"],
                    "mus": tissue_params["mus_prime_mm"] / (1 - tissue_params["g"]),
                    "g": tissue_params["g"],
                    "n": tissue_params["n"],
                },  # Label 1: tissue
                {
                    "mua": tissue_params["mua_mm"],
                    "mus": tissue_params["mus_prime_mm"] / (1 - tissue_params["g"]),
                    "g": tissue_params["g"],
                    "n": tissue_params["n"],
                },  # Label 2: source (same params)
            ],
            "MediaFormat": "uint8",
        },
        "Optode": {
            "Source": {
                "Type": "pencil",
                "Pos": [float(nx / 2), float(ny / 2), float(nz / 2)],
                "Dir": [0, 0, 1],
            }
        },
    }

    return config


def run_mcx_volume_source(
    volume_file: Path,
    output_dir: Path,
    source_mask: np.ndarray,
    tissue_params: dict,
    n_photons: int = 1e8,
    voxel_size_mm: float = VOXEL_SIZE_MM,
) -> np.ndarray:
    """Run MCX with voxelized uniform volume source (single simulation).

    Uses MCX pattern3d source type to specify per-voxel emission intensity.
    All source voxels emit uniformly.

    Parameters
    ----------
    volume_file : Path
        Path to volume binary file.
    output_dir : Path
        Output directory.
    source_mask : np.ndarray
        Source region mask (binary) in XYZ order.
    tissue_params : dict
        Tissue optical parameters.
    n_photons : int
        Total number of photons to simulate.
    voxel_size_mm : float
        Voxel size in mm.

    Returns
    -------
    fluence : np.ndarray
        Fluence volume (X, Y, Z).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = ATLAS_VOLUME_SHAPE  # XYZ order

    # Get source voxel indices
    source_indices = np.argwhere(source_mask > 0)
    n_source_voxels = len(source_indices)

    if n_source_voxels == 0:
        raise ValueError("Empty source mask")

    logger.info("Volume source: %d voxels, %d photons", n_source_voxels, n_photons)

    # Create uniform tissue volume (label 1)
    tissue_volume = np.ones(ATLAS_VOLUME_SHAPE, dtype=np.uint8)
    temp_volume_file = output_dir / "tissue_volume.bin"
    tissue_volume.tofile(temp_volume_file)

    # Create source pattern (float32, ZYX order for MCX binary)
    # pattern3d expects intensity per voxel
    source_pattern = source_mask.astype(np.float32)
    # Transpose from XYZ to ZYX for MCX
    source_pattern_zyx = source_pattern.transpose(2, 1, 0)
    pattern_file = output_dir / "source_pattern.bin"
    source_pattern_zyx.tofile(pattern_file)

    # Find bounding box of source for pattern dimensions
    x_min, x_max = source_indices[:, 0].min(), source_indices[:, 0].max()
    y_min, y_max = source_indices[:, 1].min(), source_indices[:, 1].max()
    z_min, z_max = source_indices[:, 2].min(), source_indices[:, 2].max()

    # Pattern dimensions (add 1 to include the max index)
    pattern_nx = x_max - x_min + 1
    pattern_ny = y_max - y_min + 1
    pattern_nz = z_max - z_min + 1

    logger.info(
        "Pattern bbox: X[%d:%d], Y[%d:%d], Z[%d:%d]",
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    )
    logger.info("Pattern dimensions: [%d, %d, %d]", pattern_nx, pattern_ny, pattern_nz)

    # Extract pattern for bounding box only (more efficient)
    pattern_crop = source_mask[
        x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
    ].astype(np.float32)
    pattern_crop_zyx = pattern_crop.transpose(2, 1, 0)
    pattern_file = output_dir / "source_pattern.bin"
    pattern_crop_zyx.tofile(pattern_file)

    # MCX config with pattern3d source
    # CRITICAL: Dim must be [X, Y, Z] = [nx, ny, nz] to match multiposition format
    # Pos is also in [X, Y, Z] order for pattern3d
    # VolumeFile must be relative path since MCX runs from output_dir
    config = {
        "Domain": {
            "VolumeFile": "tissue_volume.bin",  # Relative path
            "Dim": [nx, ny, nz],  # XYZ order (matches multiposition)
            "OriginType": 1,
            "LengthUnit": float(voxel_size_mm),
            "Media": [
                {"mua": 0.0, "mus": 0.0, "g": 1.0, "n": 1.0},  # Label 0: air
                {
                    "mua": tissue_params["mua_mm"],
                    "mus": tissue_params["mus_prime_mm"] / (1 - tissue_params["g"]),
                    "g": tissue_params["g"],
                    "n": tissue_params["n"],
                },  # Label 1: tissue
            ],
        },
        "Session": {
            "ID": "volume_source",
            "Photons": int(n_photons),
            "RNGSeed": 42,
        },
        "Forward": {
            "T0": 0.0,
            "T1": 5.0e-08,
            "DT": 5.0e-08,
        },
        "Optode": {
            "Source": {
                "Type": "pattern3d",
                "Pos": [
                    int(x_min),
                    int(y_min),
                    int(z_min),
                ],  # XYZ order (matches Dim)
                "Dir": [0, 0, 1, "_NaN_"],
                "Pattern": {
                    "Nx": int(pattern_nx),
                    "Ny": int(pattern_ny),
                    "Nz": int(pattern_nz),
                    "Data": str(pattern_file.name),
                },
                "Param1": [int(pattern_nx), int(pattern_ny), int(pattern_nz)],
            }
        },
    }

    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("MCX config saved to %s", config_file)

    # Run MCX (single simulation)
    cmd = [MCX_EXE, "-f", str(config_file.name)]  # Run from output_dir
    logger.info("Running MCX: %s (cwd=%s)", " ".join(cmd), output_dir)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, cwd=output_dir
        )
        if result.returncode != 0:
            logger.error("MCX stdout: %s", result.stdout)
            logger.error("MCX stderr: %s", result.stderr)
            raise RuntimeError(f"MCX failed with code {result.returncode}")
    except Exception as e:
        logger.error("MCX error: %s", e)
        raise

    # Load fluence
    jnii_file = output_dir / "volume_source.jnii"
    if not jnii_file.exists():
        raise FileNotFoundError(f"MCX output not found: {jnii_file}")

    import jdata as jd

    data = jd.loadjd(str(jnii_file))
    nifti = data["NIFTIData"] if isinstance(data, dict) else data
    if nifti.ndim == 5:
        nifti = nifti[:, :, :, 0, 0]
    # JNII is already in XYZ order - NO transpose needed (confirmed from multiposition)
    fluence = nifti.astype(np.float32)

    # Save fluence
    fluence_file = output_dir / "fluence.npy"
    np.save(fluence_file, fluence)
    logger.info("Saved fluence to %s (shape=%s)", fluence_file, fluence.shape)

    return fluence


def render_green_uniform_source_projection(
    source_center_mm: np.ndarray,
    source_radius_mm: float,
    atlas_binary_xyz: np.ndarray,
    angle_deg: float,
    camera_distance_mm: float,
    fov_mm: float,
    detector_resolution: Tuple[int, int],
    tissue_params: dict,
    voxel_size_mm: float = VOXEL_SIZE_MM,
    sampling_scheme: str = "stratified-33",
) -> np.ndarray:
    """Render uniform source projection using multi-point cubature.

    Parameters
    ----------
    source_center_mm : np.ndarray
        Source center [3] in mm.
    source_radius_mm : float
        Source radius (for sphere) or axes (for ellipsoid).
    atlas_binary_xyz : np.ndarray
        Binary atlas mask (X, Y, Z).
    angle_deg : float
        Viewing angle.
    camera_distance_mm : float
        Camera distance.
    fov_mm : float
        Field of view.
    detector_resolution : tuple
        (width, height) in pixels.
    tissue_params : dict
        Tissue optical parameters.
    voxel_size_mm : float
        Voxel size in mm.
    sampling_scheme : str
        Cubature scheme ("sr-6", "grid-27", "stratified-33", etc.).

    Returns
    -------
    projection : np.ndarray
        (H, W) 2D projection image.
    """
    # Step 1: Sample source volume using cubature
    if isinstance(source_radius_mm, (list, tuple, np.ndarray)):
        # Ellipsoid
        axes = np.array(source_radius_mm)
        points, weights = sample_uniform(
            center=source_center_mm,
            axes=axes,
            alpha=1.0,
            scheme=sampling_scheme,
        )
    else:
        # Sphere
        axes = np.array([source_radius_mm, source_radius_mm, source_radius_mm])
        points, weights = sample_uniform(
            center=source_center_mm,
            axes=axes,
            alpha=1.0,
            scheme=sampling_scheme,
        )

    logger.info("Using %s scheme: %d points", sampling_scheme, len(points))

    # Step 2: Get surface coordinates (only once per angle)
    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary_xyz,
        angle_deg,
        camera_distance_mm,
        fov_mm,
        detector_resolution,
        voxel_size_mm,
    )

    # Step 3: Compute Green's function for each sample point and sum
    projection = np.zeros(detector_resolution[::-1], dtype=np.float32)

    for pt, w in zip(points, weights):
        proj_i = green_infinite_point_source_on_surface(
            pt, surface_coords, valid_mask, tissue_params
        )
        projection += w * proj_i

    return projection


def run_single_config(
    config_id: str,
    source_center_mm: np.ndarray,
    source_radius_mm: float,
    angle_deg: float,
    sampling_scheme: str,
    output_base_dir: Path,
    tissue_params: dict = DEFAULT_TISSUE_PARAMS,
    n_photons: int = 1e8,
    skip_mcx: bool = False,
    source_axes_mm: np.ndarray = None,
) -> dict:
    """Run single Stage 2 configuration.

    Parameters
    ----------
    config_id : str
        Configuration identifier.
    source_center_mm : np.ndarray
        Source center [3] in mm.
    source_radius_mm : float
        Source radius in mm (for sphere).
    angle_deg : float
        Viewing angle.
    sampling_scheme : str
        Cubature scheme.
    output_base_dir : Path
        Base output directory.
    tissue_params : dict
        Tissue optical parameters.
    n_photons : int
        Number of photons for MCX.
    skip_mcx : bool
        If True, skip MCX and use existing results.
    source_axes_mm : np.ndarray, optional
        Ellipsoid semi-axes [a, b, c] in mm. If provided, overrides radius.

    Returns
    -------
    results : dict
        Results dictionary with NCC, RMSE, etc.
    """
    logger.info("=" * 60)
    logger.info("Running config: %s", config_id)

    # Determine if sphere or ellipsoid
    if source_axes_mm is not None:
        logger.info("Source: center=%s, axes=%s mm", source_center_mm, source_axes_mm)
        is_ellipsoid = True
    else:
        logger.info(
            "Source: center=%s, radius=%.1fmm", source_center_mm, source_radius_mm
        )
        is_ellipsoid = False

    logger.info("Angle: %.0f°, Scheme: %s", angle_deg, sampling_scheme)
    logger.info("=" * 60)

    output_dir = output_base_dir / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create atlas binary mask
    atlas_binary = np.ones(ATLAS_VOLUME_SHAPE, dtype=np.uint8)

    # Create source mask
    if is_ellipsoid:
        source_mask = create_voxelized_uniform_ellipsoid(
            source_center_mm, source_axes_mm
        )
        source_desc = f"axes={source_axes_mm}"
    else:
        source_mask = create_voxelized_uniform_sphere(
            source_center_mm, source_radius_mm
        )
        source_desc = f"radius={source_radius_mm}mm"

    n_source_voxels = np.sum(source_mask)
    logger.info("Source voxels: %d (%s)", n_source_voxels, source_desc)

    # Run MCX (single simulation for volume source)
    fluence_file = output_dir / "fluence.npy"

    if not skip_mcx or not fluence_file.exists():
        logger.info("Running MCX volume source simulation...")
        fluence = run_mcx_volume_source(
            volume_file=output_dir / "tissue_volume.bin",
            output_dir=output_dir / "mcx_run",
            source_mask=source_mask,
            tissue_params=tissue_params,
            n_photons=n_photons,
        )
    else:
        logger.info("Loading existing MCX fluence...")
        fluence = np.load(fluence_file)

    # Project MCX fluence to 2D
    logger.info("Projecting MCX fluence (angle=%.0f°)...", angle_deg)
    mcx_proj, _ = project_volume_reference(
        fluence,
        angle_deg,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    # Compute analytic projection
    logger.info("Computing analytic projection (%s)...", sampling_scheme)

    if is_ellipsoid:
        green_proj = render_green_uniform_source_projection(
            source_center_mm,
            source_axes_mm,  # Pass axes for ellipsoid
            atlas_binary,
            angle_deg,
            CAMERA_DISTANCE_MM,
            FOV_MM,
            DETECTOR_RESOLUTION,
            tissue_params,
            VOXEL_SIZE_MM,
            sampling_scheme,
        )
    else:
        green_proj = render_green_uniform_source_projection(
            source_center_mm,
            source_radius_mm,
            atlas_binary,
            angle_deg,
            CAMERA_DISTANCE_MM,
            FOV_MM,
            DETECTOR_RESOLUTION,
            tissue_params,
            VOXEL_SIZE_MM,
            sampling_scheme,
        )

    # Compute metrics
    ncc = compute_ncc(mcx_proj, green_proj)
    rmse = compute_rmse(mcx_proj, green_proj)

    # Peak ratio (for intensity comparison)
    mcx_peak = mcx_proj.max()
    green_peak = green_proj.max()
    peak_ratio = green_peak / mcx_peak if mcx_peak > 0 else 0

    logger.info("Results: NCC=%.4f, RMSE=%.4f, Peak Ratio=%.3f", ncc, rmse, peak_ratio)

    # Save results
    results = {
        "config_id": config_id,
        "source_center_mm": source_center_mm.tolist(),
        "angle_deg": float(angle_deg),
        "sampling_scheme": sampling_scheme,
        "n_source_voxels": int(n_source_voxels),
        "ncc": float(ncc),
        "rmse": float(rmse),
        "mcx_peak": float(mcx_peak),
        "green_peak": float(green_peak),
        "peak_ratio": float(peak_ratio),
    }

    if is_ellipsoid:
        results["source_axes_mm"] = source_axes_mm.tolist()
    else:
        results["source_radius_mm"] = float(source_radius_mm)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save projections
    np.save(output_dir / "mcx_proj.npy", mcx_proj)
    np.save(output_dir / "green_proj.npy", green_proj)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Output directory
    output_base_dir = Path(__file__).parent / "results" / "stage2"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Source center: 4mm depth from dorsal surface (from Stage 1.5)
    # Using centered coordinates (origin at volume center)
    # Y=2.4mm is the trunk center slice (from multiposition tests)
    # Z=4mm is 4mm below dorsal surface (dorsal is around +8mm)
    source_center_mm = np.array([0.0, 2.4, 4.0])

    # Experiment configurations
    configs = [
        # S2-U1: Small sphere r=1mm, SR-6
        {
            "id": "S2-U1-r1mm-sr6",
            "radius": 1.0,
            "angle": 0.0,
            "scheme": "7-point",  # SR-6 for uniform = 7-point
        },
        # S2-U2: Medium sphere r=2mm, SR-6
        {
            "id": "S2-U2-r2mm-sr6",
            "radius": 2.0,
            "angle": 0.0,
            "scheme": "7-point",
        },
        # S2-U3: Medium sphere r=2mm, grid-27
        {
            "id": "S2-U3-r2mm-grid27",
            "radius": 2.0,
            "angle": 0.0,
            "scheme": "grid-27",
        },
        # S2-U4: Medium sphere r=2mm, stratified-33
        {
            "id": "S2-U4-r2mm-strat33",
            "radius": 2.0,
            "angle": 0.0,
            "scheme": "stratified-33",
        },
        # S2-U5: Ellipsoid axes=[2,2,1]mm, SR-6
        {
            "id": "S2-U5-ellipsoid-sr6",
            "radius": None,  # Special handling for ellipsoid
            "axes": [2.0, 2.0, 1.0],
            "angle": 0.0,
            "scheme": "7-point",
        },
    ]

    all_results = []

    for config in configs:
        try:
            # Check if ellipsoid
            if "axes" in config:
                result = run_single_config(
                    config_id=config["id"],
                    source_center_mm=source_center_mm,
                    source_radius_mm=2.0,  # Dummy value
                    angle_deg=config["angle"],
                    sampling_scheme=config["scheme"],
                    output_base_dir=output_base_dir,
                    skip_mcx=False,
                    source_axes_mm=np.array(config["axes"]),
                )
            else:
                result = run_single_config(
                    config_id=config["id"],
                    source_center_mm=source_center_mm,
                    source_radius_mm=config.get("radius", 2.0),
                    angle_deg=config["angle"],
                    sampling_scheme=config["scheme"],
                    output_base_dir=output_base_dir,
                    skip_mcx=False,
                )
            all_results.append(result)
        except Exception as e:
            logger.error("Config %s failed: %s", config["id"], e)
            import traceback

            traceback.print_exc()

    # Save summary
    with open(output_base_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Stage 2 Complete!")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(
            "%s: NCC=%.4f, Scheme=%s",
            r["config_id"],
            r["ncc"],
            r["sampling_scheme"],
        )


if __name__ == "__main__":
    main()
