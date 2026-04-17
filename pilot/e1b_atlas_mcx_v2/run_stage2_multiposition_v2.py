#!/usr/bin/env python3
"""Stage 2 Multi-Position (v2): Volume Source Validation - FIXED.

Root causes of v1 failure:
1. VolumeFile was all-1 box, not real atlas
2. Byte order wrong (X slow axis vs MCX expecting X fast axis)
3. Source Z position hardcoded instead of computed from dorsal_z

Fix: Align completely with fmt_simgen/mcx_config.py + mcx_source.py pattern.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from surface_projection import compute_ncc, compute_rmse, project_get_surface_coords
from fmt_simgen.mcx_projection import project_volume_reference
from source_quadrature import sample_uniform

logger = logging.getLogger(__name__)

MCX_EXE = "/mnt/f/win-pro/bin/mcx.exe"

DEFAULT_TISSUE_PARAMS = {
    "mua_mm": 0.08697,
    "mus_prime_mm": 4.2907,
    "g": 0.9,
    "n": 1.37,
}

CAMERA_DISTANCE_MM = 200.0
FOV_MM = 50.0
DETECTOR_RESOLUTION = (256, 256)
VOXEL_SIZE_MM = 0.2

BEST_ANGLES = {
    "P1-dorsal": 0.0,
    "P2-left": 90.0,
    "P3-right": -90.0,
    "P4-dorsal-lat": -30.0,
    "P5-ventral": 60.0,
}


def create_atlas_homogeneous_volume(atlas_bin_path: Path) -> np.ndarray:
    """Load atlas and create homogeneous volume (XYZ order)."""
    if not atlas_bin_path.exists():
        raise FileNotFoundError(f"Atlas bin not found: {atlas_bin_path}")

    volume = np.fromfile(atlas_bin_path, dtype=np.uint8)
    volume = volume.reshape((104, 200, 190))
    volume_xyz = volume.transpose(2, 1, 0)
    homogeneous = np.where(volume_xyz > 0, 1, 0).astype(np.uint8)

    n_tissue = np.sum(homogeneous > 0)
    logger.info(f"Atlas homogeneous volume: {homogeneous.shape} (XYZ)")
    logger.info(
        f"  Tissue: {n_tissue}/{homogeneous.size} ({100 * n_tissue / homogeneous.size:.1f}%)"
    )

    return homogeneous


def get_atlas_center(
    atlas_binary_xyz: np.ndarray, voxel_size_mm: float
) -> Tuple[float, float, float]:
    """Find center of mass and dorsal surface (same as Stage 1.5)."""
    tissue_mask = atlas_binary_xyz > 0
    tissue_voxels = np.argwhere(tissue_mask)

    if len(tissue_voxels) == 0:
        raise ValueError("No tissue found in atlas")

    nx, ny, nz = atlas_binary_xyz.shape
    center = np.array([nx / 2, ny / 2, nz / 2])

    tissue_mm = (tissue_voxels - center + 0.5) * voxel_size_mm

    center_x = float(tissue_mm[:, 0].mean())
    center_y = float(tissue_mm[:, 1].mean())
    dorsal_z = float(tissue_mm[:, 2].max())

    return center_x, center_y, dorsal_z


def get_surface_positions(atlas_binary_xyz: np.ndarray, voxel_size_mm: float) -> dict:
    """Find surface positions at Y=center slice."""
    ny = atlas_binary_xyz.shape[1]
    y_center = ny // 2

    slice_xz = atlas_binary_xyz[:, y_center, :]
    tissue_x, tissue_z = np.where(slice_xz > 0)

    if len(tissue_x) == 0:
        raise ValueError("No tissue found")

    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2

    tissue_x_mm = (tissue_x - x_center + 0.5) * voxel_size_mm
    tissue_z_mm = (tissue_z - z_center + 0.5) * voxel_size_mm

    return {
        "dorsal_z": tissue_z_mm.max(),
        "ventral_z": tissue_z_mm.min(),
        "left_x": tissue_x_mm.min(),
        "right_x": tissue_x_mm.max(),
        "center_x": (tissue_x_mm.min() + tissue_x_mm.max()) / 2,
        "center_z": (tissue_z_mm.min() + tissue_z_mm.max()) / 2,
    }


def create_uniform_source_pattern(
    source_center_mm: np.ndarray,
    source_radius_mm: float,
    atlas_binary_xyz: np.ndarray,
    voxel_size_mm: float = 0.2,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Generate pattern3d source.

    Parameters
    ----------
    source_center_mm : np.ndarray
        Source center in CENTERED coordinates (mm) - origin at volume center
    source_radius_mm : float
        Source radius (mm)
    atlas_binary_xyz : np.ndarray
        Atlas binary mask (XYZ order)
    voxel_size_mm : float
        Voxel size (mm)

    Returns
    -------
    pattern : np.ndarray
        float32 (nx_pat, ny_pat, nz_pat) XYZ order
    origin : (x0, y0, z0)
        Pattern bbox origin in volume voxel coords (XYZ)
    """
    nx, ny, nz = atlas_binary_xyz.shape

    center_vox = source_center_mm / voxel_size_mm
    center_vox[0] += nx / 2
    center_vox[1] += ny / 2
    center_vox[2] += nz / 2

    radius_vox = source_radius_mm / voxel_size_mm

    pad = 2
    x0 = max(0, int(np.floor(center_vox[0] - radius_vox)) - pad)
    y0 = max(0, int(np.floor(center_vox[1] - radius_vox)) - pad)
    z0 = max(0, int(np.floor(center_vox[2] - radius_vox)) - pad)
    x1 = min(nx, int(np.ceil(center_vox[0] + radius_vox)) + pad)
    y1 = min(ny, int(np.ceil(center_vox[1] + radius_vox)) + pad)
    z1 = min(nz, int(np.ceil(center_vox[2] + radius_vox)) + pad)

    logger.info(
        f"Center voxel: [{center_vox[0]:.1f}, {center_vox[1]:.1f}, {center_vox[2]:.1f}]"
    )
    logger.info(f"Pattern bbox: [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]")

    ix = np.arange(x0, x1)
    iy = np.arange(y0, y1)
    iz = np.arange(z0, z1)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")

    px = (gx - nx / 2 + 0.5) * voxel_size_mm
    py = (gy - ny / 2 + 0.5) * voxel_size_mm
    pz = (gz - nz / 2 + 0.5) * voxel_size_mm
    r = np.sqrt(
        (px - source_center_mm[0]) ** 2
        + (py - source_center_mm[1]) ** 2
        + (pz - source_center_mm[2]) ** 2
    )
    pattern = (r <= source_radius_mm).astype(np.float32)

    atlas_crop = atlas_binary_xyz[x0:x1, y0:y1, z0:z1]
    pattern = pattern * (atlas_crop > 0)

    logger.info(f"Pattern shape: {pattern.shape}, nonzero: {np.sum(pattern > 0)}")

    return pattern, (x0, y0, z0)


def write_mcx_pattern3d_config(
    sample_id: str,
    pattern: np.ndarray,
    origin: Tuple[int, int, int],
    output_dir: Path,
    volume_file_rel: str,
    volume_shape: Tuple[int, int, int],
    material_yaml_path: Path,
    n_photons: int = int(1e8),
) -> Path:
    """Write MCX config following fmt_simgen/mcx_config.py pattern exactly.

    Parameters
    ----------
    sample_id : str
        Sample ID
    pattern : np.ndarray
        Pattern (nx, ny, nz) XYZ order
    origin : (x0, y0, z0)
        Pattern origin in volume voxel coords
    output_dir : Path
        Output directory
    volume_file_rel : str
        Relative path to mcx_volume_trunk.bin
    volume_shape : (nx, ny, nz)
        Volume shape XYZ
    material_yaml_path : Path
        Path to mcx_material.yaml
    n_photons : int
        Number of photons

    Returns
    -------
    json_path : Path
        Path to generated JSON config
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_zyx = pattern.transpose(2, 1, 0)
    pattern_zyx.astype(np.float32).tofile(output_dir / f"source-{sample_id}.bin")

    nx_pat, ny_pat, nz_pat = pattern.shape
    x0, y0, z0 = origin
    nx, ny, nz = volume_shape

    with open(material_yaml_path, "r") as f:
        media = yaml.safe_load(f)

    config = {
        "Domain": {
            "VolumeFile": volume_file_rel,
            "Dim": [int(nx), int(ny), int(nz)],
            "OriginType": 1,
            "LengthUnit": 0.2,
            "Media": media,
        },
        "Session": {
            "Photons": int(n_photons),
            "RNGSeed": 42,
            "ID": sample_id,
        },
        "Forward": {"T0": 0.0, "T1": 5.0e-08, "DT": 5.0e-08},
        "Optode": {
            "Source": {
                "Pos": [int(x0), int(y0), int(z0)],
                "Dir": [0, 0, 1, "_NaN_"],
                "Type": "pattern3d",
                "Pattern": {
                    "Nx": int(nx_pat),
                    "Ny": int(ny_pat),
                    "Nz": int(nz_pat),
                    "Data": f"source-{sample_id}.bin",
                },
                "Param1": [int(nx_pat), int(ny_pat), int(nz_pat)],
            }
        },
    }

    json_path = output_dir / f"{sample_id}.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Wrote MCX config: {json_path}")
    return json_path


def run_mcx_simulation(
    json_path: Path, mcx_exec: str = MCX_EXE, timeout: int = 1800
) -> Path:
    """Run MCX and return jnii path."""
    work_dir = json_path.parent
    session_id = json_path.stem
    output_jnii = work_dir / f"{session_id}.jnii"

    if output_jnii.exists():
        logger.info(f"Skipping MCX: {output_jnii} already exists")
        return output_jnii

    logger.info(f"Running MCX: {mcx_exec} -f {json_path.name}")
    try:
        subprocess.run(
            [mcx_exec, "-f", json_path.name],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MCX timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MCX failed: {e.stderr or e.stdout}")

    if not output_jnii.exists():
        raise FileNotFoundError(f"MCX output not found: {output_jnii}")

    logger.info(f"MCX complete: {output_jnii}")
    return output_jnii


def load_mcx_fluence(jnii_path: Path) -> np.ndarray:
    """Load MCX JNII output (XYZ order)."""
    import jdata as jd

    data = jd.loadjd(str(jnii_path))
    fluence = np.array(data["NIFTIData"], dtype=np.float32)

    if fluence.ndim > 3:
        fluence = fluence[..., 0, 0]

    logger.info(f"Loaded MCX fluence: {fluence.shape}, max={fluence.max():.6e}")
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
    voxel_size_mm: float,
    sampling_scheme: str = "7-point",
) -> np.ndarray:
    """Render uniform source projection using multi-point cubature."""
    from surface_projection import green_infinite_point_source_on_surface

    axes = np.array([source_radius_mm, source_radius_mm, source_radius_mm])
    points, weights = sample_uniform(
        center=source_center_mm,
        axes=axes,
        alpha=1.0,
        scheme=sampling_scheme,
    )

    logger.info(f"Using {sampling_scheme} scheme: {len(points)} points")

    surface_coords, valid_mask = project_get_surface_coords(
        atlas_binary_xyz,
        angle_deg,
        camera_distance_mm,
        fov_mm,
        detector_resolution,
        voxel_size_mm,
    )

    projection = np.zeros(detector_resolution[::-1], dtype=np.float32)

    for pt, w in zip(points, weights):
        proj_i = green_infinite_point_source_on_surface(
            pt, surface_coords, valid_mask, tissue_params
        )
        projection += w * proj_i

    return projection


def generate_volume_source_configs(
    surface_positions: dict,
    source_radius_mm: float = 2.0,
    depth_mm: float = 4.0,
    sampling_scheme: str = "7-point",
) -> List[Dict]:
    """Generate volume source configs for P1-P5 positions."""
    cx = surface_positions["center_x"]
    cz = surface_positions["center_z"]
    y_center = 2.4

    configs = []

    p1_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P1-dorsal-r{source_radius_mm}",
            "source_pos": [cx, y_center, p1_z],
            "best_angle": BEST_ANGLES["P1-dorsal"],
            "description": f"Dorsal, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    p2_x = surface_positions["left_x"] + depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P2-left-r{source_radius_mm}",
            "source_pos": [p2_x, y_center, cz],
            "best_angle": BEST_ANGLES["P2-left"],
            "description": f"Left side, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    p3_x = surface_positions["right_x"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P3-right-r{source_radius_mm}",
            "source_pos": [p3_x, y_center, cz],
            "best_angle": BEST_ANGLES["P3-right"],
            "description": f"Right side, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    left_offset = (surface_positions["center_x"] - surface_positions["left_x"]) * 0.5
    p4_x = cx - left_offset
    p4_z = surface_positions["dorsal_z"] - depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P4-dorsal-lat-r{source_radius_mm}",
            "source_pos": [p4_x, y_center, p4_z],
            "best_angle": BEST_ANGLES["P4-dorsal-lat"],
            "description": f"Dorsal-lateral, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    p5_z = surface_positions["ventral_z"] + depth_mm
    configs.append(
        {
            "id": f"S2-Vol-P5-ventral-r{source_radius_mm}",
            "source_pos": [cx, y_center, p5_z],
            "best_angle": BEST_ANGLES["P5-ventral"],
            "description": f"Ventral, r={source_radius_mm}mm, {sampling_scheme}",
            "radius": source_radius_mm,
            "scheme": sampling_scheme,
        }
    )

    return configs


def run_single_position(
    config: Dict,
    atlas_binary_xyz: np.ndarray,
    output_base_dir: Path,
    volume_file_rel: str,
    volume_shape: Tuple[int, int, int],
    material_yaml_path: Path,
    tissue_params: dict = DEFAULT_TISSUE_PARAMS,
    n_photons: int = int(1e8),
) -> Dict:
    """Run single position with pattern3d source."""
    config_id = config["id"]
    source_pos = np.array(config["source_pos"])
    source_radius = config["radius"]
    scheme = config["scheme"]
    best_angle = config["best_angle"]

    output_dir = output_base_dir / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Position: %s", config_id)
    logger.info("Source: %s mm, r=%.1fmm", source_pos, source_radius)
    logger.info("Scheme: %s, Angle: %.0f°", scheme, best_angle)
    logger.info("=" * 70)

    pattern, origin = create_uniform_source_pattern(
        source_pos,
        source_radius,
        atlas_binary_xyz,
        VOXEL_SIZE_MM,
    )

    n_voxels = np.sum(pattern > 0)
    logger.info("Source voxels: %d", n_voxels)

    json_path = write_mcx_pattern3d_config(
        config_id,
        pattern,
        origin,
        output_dir,
        volume_file_rel,
        volume_shape,
        material_yaml_path,
        n_photons,
    )

    jnii_path = run_mcx_simulation(json_path)
    fluence = load_mcx_fluence(jnii_path)

    logger.info("Projecting at angle %.0f°...", best_angle)

    mcx_proj, _ = project_volume_reference(
        fluence,
        best_angle,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        VOXEL_SIZE_MM,
    )

    green_proj = render_green_uniform_source_projection(
        source_pos,
        source_radius,
        atlas_binary_xyz,
        best_angle,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        tissue_params,
        VOXEL_SIZE_MM,
        scheme,
    )

    ncc = compute_ncc(mcx_proj, green_proj)
    rmse = compute_rmse(mcx_proj, green_proj)

    logger.info("NCC=%.4f, RMSE=%.4f", ncc, rmse)

    np.save(output_dir / f"mcx_a{int(best_angle)}.npy", mcx_proj)
    np.save(output_dir / f"green_a{int(best_angle)}.npy", green_proj)
    np.save(output_dir / "fluence.npy", fluence)

    result = {
        "config_id": config_id,
        "source_pos": source_pos.tolist(),
        "source_radius_mm": source_radius,
        "sampling_scheme": scheme,
        "best_angle": best_angle,
        "ncc": ncc,
        "rmse": rmse,
        "n_source_voxels": int(n_voxels),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_base_dir = Path(__file__).parent / "results" / "stage2_multiposition_v2"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    atlas_bin_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_volume_trunk.bin"
    )
    atlas_binary_xyz = create_atlas_homogeneous_volume(atlas_bin_path)

    material_yaml_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_material.yaml"
    )
    volume_shape = atlas_binary_xyz.shape

    volume_file_rel = str(atlas_bin_path.resolve())

    surface_pos = get_surface_positions(atlas_binary_xyz, VOXEL_SIZE_MM)
    logger.info("Surface positions:")
    logger.info("  Dorsal Z: %.1f mm", surface_pos["dorsal_z"])
    logger.info("  Ventral Z: %.1f mm", surface_pos["ventral_z"])
    logger.info("  Left X: %.1f mm", surface_pos["left_x"])
    logger.info("  Right X: %.1f mm", surface_pos["right_x"])

    configs = generate_volume_source_configs(
        surface_pos,
        source_radius_mm=2.0,
        depth_mm=4.0,
        sampling_scheme="7-point",
    )

    all_results = []
    for config in configs:
        try:
            result = run_single_position(
                config,
                atlas_binary_xyz,
                output_base_dir,
                str(volume_file_rel),
                volume_shape,
                material_yaml_path,
                n_photons=int(1e8),
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Config %s failed: %s", config["id"], e)
            import traceback

            traceback.print_exc()

    with open(output_base_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("Stage 2 Multi-Position Volume Source Summary (v2)")
    logger.info("=" * 70)
    logger.info(f"{'Position':<25} {'Angle':<12} {'NCC':<8} {'Status':<10}")
    logger.info("-" * 70)
    for r in all_results:
        status = "✅ PASS" if r["ncc"] >= 0.95 else "❌ FAIL"
        logger.info(
            f"{r['config_id']:<25} {r['best_angle']:>6.0f}°      {r['ncc']:>6.3f}   {status}"
        )
    logger.info("=" * 70)

    nccs = [r["ncc"] for r in all_results]
    logger.info("NCC Statistics:")
    logger.info("  Mean: %.4f", np.mean(nccs))
    logger.info("  Min:  %.4f", np.min(nccs))
    logger.info("  Max:  %.4f", np.max(nccs))


if __name__ == "__main__":
    main()
