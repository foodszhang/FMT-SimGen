"""MCX volume preparation utilities for FMT-SimGen.

This module prepares MCX simulation inputs by cropping the torso region from
the Digimouse atlas and downsampling 2x.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fmt_simgen.frame_contract import VOXEL_SIZE_MM

logger = logging.getLogger(__name__)


def mus_from_mus_prime(mus_prime: float, g: float) -> float:
    """Convert reduced scattering (mus') to total scattering (mus).

    mus = mus_prime / (1 - g)

    For background (g=1.0 or mus_prime=0), return mus=0.
    """
    if mus_prime == 0.0 or g >= 1.0:
        return 0.0
    return mus_prime / (1.0 - g)


def build_material_list_from_config(
    mcx_config: dict[str, Any],
    physics_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build MCX material list from config.

    Reads tissue mapping and optical parameters from config, ensuring
    consistency with DE pipeline.

    Args:
        mcx_config: MCX config section with tissue_mapping and tissue_classes.
        physics_config: Physics config section with tissues optical parameters.

    Returns:
        List of material dicts with mua, mus, g, n, tag, name.
    """
    tissue_classes = mcx_config.get("tissue_classes", {})
    tissues_params = physics_config.get("tissues", {})

    materials = []
    for label_str, class_info in tissue_classes.items():
        label = int(label_str)
        name = class_info.get("name", f"tissue_{label}")
        param_source = class_info.get("param_source", name)

        if param_source not in tissues_params:
            logger.warning(
                f"MCX tissue {label} ({name}): param_source '{param_source}' "
                f"not found in physics.tissues, using defaults"
            )
            params = {"mu_a": 0.0, "mu_s_prime": 0.0, "g": 0.9, "n": 1.37}
        else:
            params = tissues_params[param_source]

        mu_a = params.get("mu_a", 0.0)
        mu_s_prime = params.get("mu_s_prime", params.get("mu_sp", 0.0))
        g = params.get("g", 0.9)
        n = params.get("n", 1.37)

        mus = mus_from_mus_prime(mu_s_prime, g)

        materials.append(
            {
                "mua": mu_a,
                "mus": mus,
                "g": g,
                "n": n,
                "tag": label,
                "name": name,
            }
        )

        logger.debug(
            f"MCX material {label} ({name}): "
            f"mu_a={mu_a:.5f}, mu_s'={mu_s_prime:.5f}, mu_s={mus:.5f}, g={g}, n={n}"
        )

    return materials


def get_tissue_mapping(mcx_config: dict[str, Any]) -> dict[int, int]:
    """Get tissue mapping from config.

    Args:
        mcx_config: MCX config section with tissue_mapping.

    Returns:
        Dict mapping original label to MCX label.
    """
    mapping = mcx_config.get("tissue_mapping", {})
    return {int(k): int(v) for k, v in mapping.items()}


def load_atlas_labels(atlas_path: str | Path) -> tuple[np.ndarray, float]:
    """Load tissue labels from atlas npz file.

    Automatically detects the key name for the label volume.

    Args:
        atlas_path: Path to atlas npz file.

    Returns:
        Tuple of (tissue_labels, voxel_size_mm).

    Raises:
        FileNotFoundError: If atlas file doesn't exist.
        ValueError: If no valid label key found.
    """
    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    atlas_data = np.load(atlas_path, allow_pickle=True)
    logger.info(f"Atlas npz keys: {list(atlas_data.keys())}")

    # Find the label volume key (check original_labels first since tissue_labels may not exist)
    label_key = None
    for key in ["original_labels", "tissue_labels", "labels", "data", "volume"]:
        if key in atlas_data:
            label_key = key
            break

    if label_key is None:
        available = list(atlas_data.keys())
        raise ValueError(f"No label key found in atlas. Available keys: {available}")

    labels = atlas_data[label_key]
    voxel_size = float(atlas_data["voxel_size"])

    logger.info(
        f"Using label key '{label_key}', shape: {labels.shape}, dtype: {labels.dtype}"
    )
    logger.info(f"Voxel size: {voxel_size} mm")

    return labels, voxel_size


def crop_and_downsample(
    labels: np.ndarray,
    y_start: int,
    y_end: int,
    downsample_factor: int = 2,
) -> np.ndarray:
    """Crop torso region and downsample.

    Args:
        labels: Original atlas labels, shape (X, Y, Z).
        y_start: Start voxel index for Y crop (anterior→posterior).
        y_end: End voxel index for Y crop (non-inclusive).
        downsample_factor: Downsampling factor (default 2).

    Returns:
        Downsampled volume, shape (X/df, Y/df, Z/df) in XYZ order.
    """
    # Crop Y dimension (columns in numpy indexing)
    cropped = labels[:, y_start:y_end, :]
    logger.info(f"Cropped shape (XYZ): {cropped.shape}")

    # Majority-vote 2x2x2 block downsample
    from scipy.stats import mode

    bs = downsample_factor
    shape = cropped.shape
    new_shape = (shape[0] // bs, bs, shape[1] // bs, bs, shape[2] // bs, bs)
    vol_blocks = cropped.reshape(new_shape).transpose(0, 2, 4, 1, 3, 5)
    blocks_flat = vol_blocks.reshape(vol_blocks.shape[:3] + (bs**3,))
    mode_result, _ = mode(blocks_flat, axis=-1, keepdims=False)
    ds = mode_result.astype(np.uint8)
    logger.info(f"Downsampled shape (XYZ): {ds.shape}")

    return ds


def prepare_mcx_volume(
    atlas_path: str | Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Prepare MCX volume from atlas.

    1. Load atlas and detect label key
    2. Map original labels to MCX classes (config-defined)
    3. Crop torso region (Y ∈ [y_start, y_end] from config)
    4. Downsample 2x
    5. Transpose to ZYX order for MCX
    6. Generate material list from physics config

    Args:
        atlas_path: Path to output/shared/atlas_full.npz.
        config: Full config dict with mcx and physics sections.

    Returns:
        Tuple of (volume_zyx, material_list).
        volume_zyx is uint8 with shape (Z, Y, X).
        material_list is list of dicts with mua, mus, g, n, tag, name.
    """
    mcx_config = config.get("mcx", {})
    physics_config = config.get("physics", {})
    downsample_factor = mcx_config.get("downsample_factor", 2)

    labels, voxel_size = load_atlas_labels(atlas_path)

    # Y crop parameters from config (pixel indices at original voxel size)
    # For Digimouse: y_start=340, y_end=740 at 0.1mm → physical Y=[34,74]mm
    y_start = int(mcx_config.get("trunk_crop_y_start", 340))
    y_end = int(mcx_config.get("trunk_crop_y_end", 740))
    original_shape = labels.shape
    logger.info(f"Original atlas shape: {original_shape}")

    tissue_mapping = get_tissue_mapping(mcx_config)
    num_tissues = mcx_config.get("num_tissues", 10)

    labels_mcx = np.zeros_like(labels)
    for orig_label, mcx_label in tissue_mapping.items():
        labels_mcx[labels == orig_label] = mcx_label

    unique_orig = np.unique(labels)
    unique_mcx = np.unique(labels_mcx)
    logger.info(f"Original label range: {unique_orig.min()} - {unique_orig.max()}")
    logger.info(
        f"MCX class range after mapping: {unique_mcx.min()} - {unique_mcx.max()}"
    )

    volume_xyz = crop_and_downsample(labels_mcx, y_start, y_end, downsample_factor)

    volume_zyx = volume_xyz.transpose(2, 1, 0)
    logger.info(f"MCX volume shape (ZYX): {volume_zyx.shape}")

    unique_labels = np.unique(volume_zyx)
    max_label = int(unique_labels.max())
    logger.info(f"Label range in final volume: {unique_labels.min()} - {max_label}")

    tissue_classes = mcx_config.get("tissue_classes", {})
    if max_label >= num_tissues:
        logger.error(
            f"Found label {max_label} which exceeds expected range 0-{num_tissues - 1}. "
            f"Material list only has entries for 0-{num_tissues - 1}."
        )
        raise ValueError(
            f"Label {max_label} out of range 0-{num_tissues - 1} after mapping"
        )

    for label in unique_labels:
        count = int((volume_zyx == label).sum())
        if count < 100:
            class_info = tissue_classes.get(str(label), tissue_classes.get(label, {}))
            tissue_name = class_info.get("name", f"tissue_{label}")
            logger.warning(
                f"Label {label} ({tissue_name}) has only {count} voxels after downsampling. "
                f"It may have largely disappeared."
            )

    material_list = build_material_list_from_config(mcx_config, physics_config)

    return volume_zyx, material_list


def save_mcx_volume(
    volume_zyx: np.ndarray,
    material_list: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save MCX volume binary and material YAML.

    Args:
        volume_zyx: Volume array shape (Z, Y, X), dtype uint8.
        material_list: List of material parameter dicts.
        output_dir: Directory to save output files.

    Returns:
        Tuple of (volume_bin_path, material_yaml_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_bin_path = output_dir / "mcx_volume_trunk.bin"
    material_yaml_path = output_dir / "mcx_material.yaml"

    # Save binary volume
    volume_zyx.astype(np.uint8).tofile(volume_bin_path)
    logger.info(f"Saved MCX volume: {volume_bin_path} ({volume_zyx.nbytes:,} bytes)")

    # Save material YAML
    with open(material_yaml_path, "w") as f:
        yaml.dump(material_list, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved material YAML: {material_yaml_path}")

    return volume_bin_path, material_yaml_path


def print_volume_statistics(
    volume_zyx: np.ndarray,
    voxel_size: float,
    original_shape: tuple[int, int, int],
    y_start: int,
    y_end: int,
    downsample_factor: int,
    material_list: list[dict[str, Any]],
) -> None:
    """Print detailed volume statistics.

    Args:
        volume_zyx: Volume array shape (Z, Y, X).
        voxel_size: Original voxel size in mm.
        original_shape: Original atlas shape (X, Y, Z).
        y_start: Y crop start voxel index.
        y_end: Y crop end voxel index.
        downsample_factor: Downsampling factor.
        material_list: Material parameter list.
    """
    ds_voxel_size = voxel_size * downsample_factor
    zyx_shape = volume_zyx.shape

    logger.info("=" * 60)
    logger.info("MCX VOLUME STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Original atlas shape (XYZ): {original_shape}")
    logger.info(f"Original voxel size: {voxel_size} mm")
    logger.info(
        f"Y crop: [{y_start}, {y_end}) -> [{y_start * voxel_size}, {y_end * voxel_size}) mm"
    )
    logger.info(
        f"Downsample factor: {downsample_factor} -> voxel size {ds_voxel_size} mm"
    )
    logger.info(
        f"Downsampled shape (XYZ): [{zyx_shape[2]}, {zyx_shape[1]}, {zyx_shape[0]}]"
    )
    logger.info(f"MCX Dim (ZYX): {zyx_shape}")
    logger.info(
        f"Physical dimensions: X={zyx_shape[2] * ds_voxel_size}mm, "
        f"Y={zyx_shape[1] * ds_voxel_size}mm, Z={zyx_shape[0] * ds_voxel_size}mm"
    )
    logger.info(f"Trunk offset mm: [0, {y_start * voxel_size}, 0]")
    logger.info(f"Total voxels: {volume_zyx.size:,}")
    logger.info(f"Valid (non-zero) voxels: {(volume_zyx > 0).sum():,}")
    logger.info(f"Valid voxel fraction: {(volume_zyx > 0).mean() * 100:.2f}%")
    logger.info("")

    # Per-label statistics
    logger.info("Per-label voxel counts:")
    unique_labels = np.unique(volume_zyx)
    label_to_name = {m["tag"]: m["name"] for m in material_list}

    for label in unique_labels:
        count = int((volume_zyx == label).sum())
        fraction = count / volume_zyx.size * 100
        name = label_to_name.get(label, "unknown")
        logger.info(
            f"  Label {label:2d} ({name:12s}): {count:8,} voxels ({fraction:6.2f}%)"
        )

    logger.info("=" * 60)
