"""MCX volume preparation utilities for FMT-SimGen.

This module prepares MCX simulation inputs by cropping the torso region from
the Digimouse atlas and downsampling 2x.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fmt_simgen.frame_contract import TRUNK_OFFSET_ATLAS_MM, TRUNK_SIZE_MM

logger = logging.getLogger(__name__)


# Optical parameters from CC Task Packet (10 classes, labels 0-9)
# mua: absorption coefficient [1/mm]
# mus_prime: reduced scattering coefficient [1/mm]
# g: anisotropy factor [-]
# n: refractive index [-]
TISSUE_PARAMETERS = [
    {"label": 0, "name": "background",  "mua": 0.0,      "mus_prime": 0.0,       "g": 1.00, "n": 1.0},
    {"label": 1, "name": "soft_tissue", "mua": 0.08697,  "mus_prime": 4.29071,   "g": 0.90, "n": 1.37},
    {"label": 2, "name": "bone",        "mua": 0.04,     "mus_prime": 20.0,       "g": 0.90, "n": 1.37},
    {"label": 3, "name": "brain",       "mua": 0.0648,   "mus_prime": 2.392,      "g": 0.90, "n": 1.37},
    {"label": 4, "name": "heart",       "mua": 0.05881,  "mus_prime": 6.42581,    "g": 0.85, "n": 1.37},
    {"label": 5, "name": "stomach",    "mua": 0.01304,  "mus_prime": 17.9615,    "g": 0.92, "n": 1.37},
    {"label": 6, "name": "abdominal",  "mua": 0.06597,  "mus_prime": 16.09293,  "g": 0.85, "n": 1.37},
    {"label": 7, "name": "liver",       "mua": 0.35182,  "mus_prime": 6.78066,   "g": 0.90, "n": 1.37},
    {"label": 8, "name": "kidney",      "mua": 0.06597,  "mus_prime": 16.09293,  "g": 0.85, "n": 1.37},
    {"label": 9, "name": "lung",       "mua": 0.19639,  "mus_prime": 36.52133,  "g": 0.94, "n": 1.37},
]

# Mapping from original Digimouse atlas labels (0-21) to CC Task Packet classes (0-9).
# Original label -> CC class:
#   0 -> 0 (background)
#   1 -> 1 (skin -> soft_tissue)
#   2 -> 2 (skeleton -> bone)
#   3,4,5,6,7,8,10 -> 3 (brain regions -> brain)
#   9 -> 4 (heart)
#   11,12,13,14 -> 1 (muscle/fat/cartilage/tongue -> soft_tissue) [NOTE: merged with skin]
#   15 -> 5 (stomach)
#   16,17 -> 6 (spleen/pancreas -> abdominal)
#   18 -> 7 (liver)
#   19,20 -> 8 (kidney/adrenal -> kidney)
#   21 -> 9 (lung)
ORIGINAL_TO_CC_LABEL = {
    0: 0,
    1: 1,   # skin -> soft_tissue
    2: 2,   # bone
    3: 3,   # brain
    4: 3,   # medulla -> brain
    5: 3,   # cerebellum -> brain
    6: 3,   # olfactory_bulb -> brain
    7: 3,   # external_brain -> brain
    8: 3,   # striatum -> brain
    9: 4,   # heart
    10: 3,  # brain_other -> brain
    11: 1,  # muscle -> soft_tissue
    12: 1,  # fat -> soft_tissue
    13: 1,  # cartilage -> soft_tissue
    14: 1,  # tongue -> soft_tissue
    15: 5,  # stomach
    16: 6,  # spleen -> abdominal
    17: 6,  # pancreas -> abdominal
    18: 7,  # liver
    19: 8,  # kidney
    20: 8,  # adrenal -> kidney
    21: 9,  # lung
}


def mus_from_mus_prime(mus_prime: float, g: float) -> float:
    """Convert reduced scattering (mus') to total scattering (mus).

    mus = mus_prime / (1 - g)

    For background (g=1.0 or mus_prime=0), return mus=0.
    """
    if mus_prime == 0.0 or g >= 1.0:
        return 0.0
    return mus_prime / (1.0 - g)


def build_material_list() -> list[dict[str, Any]]:
    """Build MCX material list from CC Task Packet parameters.

    Converts mus_prime to mus using mus = mus_prime / (1 - g).

    Returns:
        List of material dicts with mua, mus, g, n, tag, name.
    """
    materials = []
    for tissue in TISSUE_PARAMETERS:
        mus = mus_from_mus_prime(tissue["mus_prime"], tissue["g"])
        materials.append({
            "mua": tissue["mua"],
            "mus": mus,
            "g": tissue["g"],
            "n": tissue["n"],
            "tag": tissue["label"],
            "name": tissue["name"],
        })
    return materials


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

    logger.info(f"Using label key '{label_key}', shape: {labels.shape}, dtype: {labels.dtype}")
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
    """Prepare MCX volume from Digimouse atlas.

    1. Load atlas and detect label key
    2. Map original 22 labels to CC Task Packet 10 classes (0-9)
    3. Crop torso region (Y ∈ [y_start, y_end])
    4. Downsample 2x
    5. Transpose to ZYX order for MCX
    6. Generate material list

    Args:
        atlas_path: Path to output/shared/atlas_full.npz.
        config: Config dict with mcx section (trunk_crop, downsample_factor).

    Returns:
        Tuple of (volume_zyx, material_list).
        volume_zyx is uint8 with shape (Z, Y, X).
        material_list is list of dicts with mua, mus, g, n, tag, name.
    """
    mcx_config = config.get("mcx", {})
    downsample_factor = mcx_config.get("downsample_factor", 2)

    # Load atlas
    labels, voxel_size = load_atlas_labels(atlas_path)

    # Y crop from TRUNK_OFFSET_ATLAS_MM (single source of truth)
    y_start = int(np.round(TRUNK_OFFSET_ATLAS_MM[1] / voxel_size))
    y_end = y_start + int(np.round(TRUNK_SIZE_MM[1] / voxel_size))
    original_shape = labels.shape
    logger.info(f"Original atlas shape: {original_shape}")

    # Map original labels to CC Task Packet classes (0-9)
    labels_cc = np.zeros_like(labels)
    for orig_label, cc_label in ORIGINAL_TO_CC_LABEL.items():
        labels_cc[labels == orig_label] = cc_label

    unique_orig = np.unique(labels)
    unique_cc = np.unique(labels_cc)
    logger.info(f"Original label range: {unique_orig.min()} - {unique_orig.max()}")
    logger.info(f"CC class range after mapping: {unique_cc.min()} - {unique_cc.max()}")

    # Crop and downsample
    volume_xyz = crop_and_downsample(labels_cc, y_start, y_end, downsample_factor)

    # Transpose to ZYX order for MCX
    volume_zyx = volume_xyz.transpose(2, 1, 0)
    logger.info(f"MCX volume shape (ZYX): {volume_zyx.shape}")

    # Verify label range
    unique_labels = np.unique(volume_zyx)
    max_label = int(unique_labels.max())
    logger.info(f"Label range in final volume: {unique_labels.min()} - {max_label}")

    if max_label > 9:
        logger.error(
            f"Found label {max_label} which exceeds expected range 0-9. "
            f"Material list only has entries for 0-9."
        )
        raise ValueError(f"Label {max_label} out of range 0-9 after mapping")

    # Warn about small labels
    for label in unique_labels:
        count = int((volume_zyx == label).sum())
        if count < 100:
            tissue_name = next(
                (t["name"] for t in TISSUE_PARAMETERS if t["label"] == label),
                "unknown"
            )
            logger.warning(
                f"Label {label} ({tissue_name}) has only {count} voxels after downsampling. "
                f"It may have largely disappeared."
            )

    # Build material list
    material_list = build_material_list()

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
    logger.info(f"Y crop: [{y_start}, {y_end}) -> [{y_start * voxel_size}, {y_end * voxel_size}) mm")
    logger.info(f"Downsample factor: {downsample_factor} -> voxel size {ds_voxel_size} mm")
    logger.info(f"Downsampled shape (XYZ): [{zyx_shape[2]}, {zyx_shape[1]}, {zyx_shape[0]}]")
    logger.info(f"MCX Dim (ZYX): {zyx_shape}")
    logger.info(f"Physical dimensions: X={zyx_shape[2] * ds_voxel_size}mm, "
                f"Y={zyx_shape[1] * ds_voxel_size}mm, Z={zyx_shape[0] * ds_voxel_size}mm")
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
        logger.info(f"  Label {label:2d} ({name:12s}): {count:8,} voxels ({fraction:6.2f}%)")

    logger.info("=" * 60)
