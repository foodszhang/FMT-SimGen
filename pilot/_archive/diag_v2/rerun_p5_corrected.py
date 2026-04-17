#!/usr/bin/env python3
"""Re-run P5 with corrected Y position.

Root cause of P5 failure:
- Y=2.4mm is near anterior where liver is only 1.2mm below ventral surface
- 4mm depth from ventral puts source INSIDE liver
- This explains the 50% liver path and low NCC

Fix:
- Use Y=10.0mm where liver is deeper and 4mm depth lands in soft_tissue
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

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
VOXEL_SIZE_MM = 0.4
DOWNSAMPLE_FACTOR = 2

BEST_ANGLES = {
    "P1-dorsal": 0.0,
    "P2-left": 90.0,
    "P3-right": -90.0,
    "P4-dorsal-lat": -30.0,
    "P5-ventral": 60.0,
}


def load_downsampled_volume() -> Tuple[np.ndarray, Path]:
    """Load or create downsampled volume."""
    volume_path = (
        Path(__file__).parent.parent
        / "results"
        / "stage2_multiposition_v2"
        / "mcx_volume_downsampled_2x.bin"
    )

    if not volume_path.exists():
        raise FileNotFoundError(f"Downsampled volume not found: {volume_path}")

    volume = np.fromfile(volume_path, dtype=np.uint8)
    volume = volume.reshape((52, 100, 95))
    volume_xyz = volume.transpose(2, 1, 0)

    return volume_xyz, volume_path


def get_surface_positions(volume_xyz: np.ndarray, voxel_size_mm: float) -> dict:
    """Get surface positions at Y=center slice."""
    ny = volume_xyz.shape[1]
    y_center = ny // 2

    slice_xz = volume_xyz[:, y_center, :]
    tissue_x, tissue_z = np.where(slice_xz > 0)

    if len(tissue_x) == 0:
        raise ValueError("No tissue found")

    nx, nz = slice_xz.shape
    x_center, z_center = nx / 2, nz / 2

    tissue_x_mm = (tissue_x - x_center + 0.5) * voxel_size_mm
    tissue_z_mm = (tissue_z - z_center + 0.5) * voxel_size_mm

    return {
        "dorsal_z": float(tissue_z_mm.max()),
        "ventral_z": float(tissue_z_mm.min()),
        "left_x": float(tissue_x_mm.min()),
        "right_x": float(tissue_x_mm.max()),
        "center_x": float((tissue_x_mm.min() + tissue_x_mm.max()) / 2),
        "center_z": float((tissue_z_mm.min() + tissue_z_mm.max()) / 2),
    }


def create_uniform_source_pattern(
    source_center_mm: np.ndarray,
    source_radius_mm: float,
    atlas_binary_xyz: np.ndarray,
    voxel_size_mm: float = 0.4,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Generate pattern3d source."""
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
    volume_file_abs: str,
    volume_shape: Tuple[int, int, int],
    material_yaml_path: Path,
    n_photons: int = int(1e8),
) -> Path:
    """Write MCX config."""
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
            "VolumeFile": volume_file_abs,
            "Dim": [int(nx), int(ny), int(nz)],
            "OriginType": 1,
            "LengthUnit": VOXEL_SIZE_MM,
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


def main():
    """Run P5 with corrected Y position."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("P5 CORRECTED RUN - Y=10.0mm")
    print("=" * 80)
    print("\nRoot cause of original P5 failure:")
    print("  Y=2.4mm is near anterior where liver is only 1.2mm below ventral")
    print("  4mm depth from ventral puts source INSIDE liver")
    print("\nFix: Use Y=10.0mm where liver is deeper")

    atlas_binary_xyz, volume_path = load_downsampled_volume()

    surface_pos = get_surface_positions(atlas_binary_xyz, VOXEL_SIZE_MM)

    material_yaml_path = Path(
        "/home/foods/pro/FMT-SimGen/output/shared/mcx_material.yaml"
    )
    volume_shape = atlas_binary_xyz.shape

    # CORRECTED P5: Y=10.0mm instead of 2.4mm
    cx = surface_pos["center_x"]
    y_corrected = 10.0  # Changed from 2.4
    p5_z = surface_pos["ventral_z"] + 4.0

    source_pos = np.array([cx, y_corrected, p5_z])

    print(f"\nCorrected P5 source position: {source_pos} mm")
    print(f"  Ventral Z: {surface_pos['ventral_z']:.2f} mm")
    print(f"  P5 Z (ventral + 4): {p5_z:.2f} mm")

    # Verify source is in soft_tissue
    ix = int(round(source_pos[0] / VOXEL_SIZE_MM + volume_shape[0] / 2 - 0.5))
    iy = int(round(source_pos[1] / VOXEL_SIZE_MM + volume_shape[1] / 2 - 0.5))
    iz = int(round(source_pos[2] / VOXEL_SIZE_MM + volume_shape[2] / 2 - 0.5))

    print(f"\nSource voxel (XYZ): ({ix}, {iy}, {iz})")

    if (
        0 <= ix < volume_shape[0]
        and 0 <= iy < volume_shape[1]
        and 0 <= iz < volume_shape[2]
    ):
        label = atlas_binary_xyz[ix, iy, iz]
        names = {
            0: "background",
            1: "soft_tissue",
            2: "bone",
            3: "brain",
            4: "heart",
            5: "stomach",
            6: "abdominal",
            7: "liver",
            8: "kidney",
            9: "lung",
        }
        print(f"Label at source: {label} ({names.get(label, 'unknown')})")

        if label == 7:
            print("❌ Source is still in liver! Need deeper Y position.")
            return
        elif label == 1:
            print("✓ Source is in soft_tissue (correct!)")
        else:
            print(f"⚠️ Source is in {names.get(label, 'unknown')}")

    output_dir = Path(__file__).parent / "results" / "P5_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_id = "S2-Vol-P5-ventral-corrected"

    pattern, origin = create_uniform_source_pattern(
        source_pos,
        2.0,
        atlas_binary_xyz,
        VOXEL_SIZE_MM,
    )

    json_path = write_mcx_pattern3d_config(
        config_id,
        pattern,
        origin,
        output_dir,
        str(volume_path.resolve()),
        volume_shape,
        material_yaml_path,
        n_photons=int(1e8),
    )

    jnii_path = run_mcx_simulation(json_path)
    fluence = load_mcx_fluence(jnii_path)

    best_angle = BEST_ANGLES["P5-ventral"]

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
        2.0,
        atlas_binary_xyz,
        best_angle,
        CAMERA_DISTANCE_MM,
        FOV_MM,
        DETECTOR_RESOLUTION,
        DEFAULT_TISSUE_PARAMS,
        VOXEL_SIZE_MM,
        "7-point",
    )

    ncc = compute_ncc(mcx_proj, green_proj)
    rmse = compute_rmse(mcx_proj, green_proj)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  NCC: {ncc:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  k = MCX_sum / Green_sum: {mcx_proj.sum() / green_proj.sum():.4e}")

    if ncc >= 0.9:
        print("\n✓ PASS: NCC ≥ 0.9")
        print("  P5 geometry issue FIXED!")
    elif ncc >= 0.8:
        print("\n~ ACCEPTABLE: NCC ≥ 0.8")
        print("  P5 improved but not fully recovered")
    else:
        print("\n✗ FAIL: NCC < 0.8")
        print("  Need further investigation")

    np.save(output_dir / f"mcx_a{int(best_angle)}.npy", mcx_proj)
    np.save(output_dir / f"green_a{int(best_angle)}.npy", green_proj)

    results = {
        "config_id": config_id,
        "source_pos": source_pos.tolist(),
        "y_corrected_mm": y_corrected,
        "source_radius_mm": 2.0,
        "sampling_scheme": "7-point",
        "best_angle": best_angle,
        "ncc": ncc,
        "rmse": rmse,
        "k_sum": float(mcx_proj.sum() / green_proj.sum()),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
