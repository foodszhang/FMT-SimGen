"""
DigimouseAtlas: Load and process Digimouse whole-body atlas data.

This module handles:
- Loading Digimouse .hdr/.img atlas files via nibabel
- Tissue label merging (original ~22 types → configurable classes)
- Surface and subcutaneous region extraction
- Whole-body processing (not cropped like P1 brain region)
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class LabelStats:
    """Statistics for a single tissue label."""

    label: int
    name: str
    voxel_count: int
    centroid: Tuple[float, float, float]
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]]


@dataclass
class AtlasInfo:
    """Container for atlas metadata."""

    shape: Tuple[int, int, int]
    voxel_size: float
    affine: np.ndarray
    unique_labels: np.ndarray
    label_stats: List[LabelStats]


class DigimouseAtlas:
    """Class for loading and processing Digimouse atlas data."""

    DEFAULT_TAG_MAPPING = {
        0: "background",
        1: "skin",
        2: "skeleton",
        3: "brain",
        4: "medulla",
        5: "cerebellum",
        6: "olfactory_bulb",
        7: "external_brain",
        8: "striatum",
        9: "heart",
        10: "brain_other",
        11: "muscle",
        12: "fat",
        13: "cartilage",
        14: "tongue",
        15: "stomach",
        16: "spleen",
        17: "pancreas",
        18: "liver",
        19: "kidney",
        20: "adrenal",
        21: "lung",
    }

    def __init__(self, path: str):
        """Initialize DigimouseAtlas loader.

        Parameters
        ----------
        path : str
            Path to Digimouse .hdr atlas file.
        """
        self.path = Path(path)
        self._volume: Optional[np.ndarray] = None
        self._header: Optional[nib.Nifti1Header] = None
        self._affine: Optional[np.ndarray] = None
        self._info: Optional[AtlasInfo] = None
        self._label_stats: Dict[int, LabelStats] = {}
        # ── Caches (lazy-initialized) ──
        self._edt: Optional[np.ndarray] = None  # Euclidean distance transform
        self._subq_cache: Dict[tuple, np.ndarray] = {}  # subcutaneous mask cache
        self._subq_coords_cache: Dict[tuple, np.ndarray] = {}  # physical coords cache

    def load(self) -> "DigimouseAtlas":
        """Load the atlas file.

        Returns
        -------
        DigimouseAtlas
            Self for method chaining.

        Raises
        ------
        FileNotFoundError
            If atlas file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Atlas file not found: {self.path}\n"
                f"Please check the path or download the Digimouse atlas."
            )

        logger.info(f"Loading atlas from: {self.path}")
        img = nib.load(str(self.path))

        self._volume = img.get_fdata().astype(np.uint8)
        self._header = img.header
        self._affine = img.affine

        if len(self._volume.shape) == 4:
            logger.info("Detected 4D volume, taking first volume")
            self._volume = self._volume[:, :, :, 0]

        logger.info(f"Atlas loaded: shape={self._volume.shape}")

        self._compute_label_stats()
        self._print_header_info()

        return self

    def _print_header_info(self) -> None:
        """Print atlas header information for coordinate system understanding."""
        logger.info("=== Atlas Header Info ===")
        logger.info(f"Shape: {self._volume.shape}")
        logger.info(f"Voxel size (pixdim): {self._header.get_zooms()}")
        logger.info(f"Affine matrix:\n{self._affine}")

        pixdim = self._header.get_zooms()
        logger.info(
            f"Coordinate interpretation: "
            f"X={pixdim[0]:.3f}mm, Y={pixdim[1]:.3f}mm, Z={pixdim[2]:.3f}mm"
        )

        extent_mm = tuple(s * p for s, p in zip(self._volume.shape[:3], pixdim[:3]))
        logger.info(
            f"Physical extent: X={extent_mm[0]:.1f}mm, Y={extent_mm[1]:.1f}mm, Z={extent_mm[2]:.1f}mm"
        )

    def _compute_label_stats(self) -> None:
        """Compute statistics for each unique label."""
        unique_labels = np.unique(self._volume)
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")

        self._label_stats = {}
        all_stats = []

        for label in unique_labels:
            mask = self._volume == label
            voxel_count = int(np.sum(mask))

            if voxel_count == 0:
                continue

            centroid_voxel = np.array(np.where(mask)).mean(axis=1)

            pixdim = self._header.get_zooms()[:3]
            centroid_mm = tuple(c * p for c, p in zip(centroid_voxel, pixdim))

            nonzero_coords = np.where(mask)
            bbox_min = tuple(int(np.min(arr)) for arr in nonzero_coords)
            bbox_max = tuple(int(np.max(arr)) for arr in nonzero_coords)
            bbox = (bbox_min, bbox_max)

            name = self.DEFAULT_TAG_MAPPING.get(label, f"unknown_label_{label}")

            stats = LabelStats(
                label=label,
                name=name,
                voxel_count=voxel_count,
                centroid=centroid_mm,
                bounding_box=bbox,
            )
            self._label_stats[label] = stats
            all_stats.append(stats)

            logger.info(
                f"  Label {label:2d} ({name:15s}): "
                f"{voxel_count:8d} voxels, "
                f"centroid=({centroid_mm[0]:7.1f}, {centroid_mm[1]:7.1f}, {centroid_mm[2]:7.1f})mm, "
                f"bbox={bbox}"
            )

        self._info = AtlasInfo(
            shape=self._volume.shape,
            voxel_size=self._header.get_zooms()[0],
            affine=self._affine,
            unique_labels=unique_labels,
            label_stats=all_stats,
        )

    @property
    def volume(self) -> np.ndarray:
        """Get the full atlas volume [X×Y×Z].

        Returns
        -------
        np.ndarray
            3D array of tissue labels.
        """
        if self._volume is None:
            raise RuntimeError("Atlas not loaded. Call load() first.")
        return self._volume

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get atlas volume shape (X, Y, Z).

        Returns
        -------
        Tuple[int, int, int]
            Atlas dimensions.
        """
        return self.volume.shape

    @property
    def voxel_size(self) -> float:
        """Get voxel size in mm.

        Returns
        -------
        float
            Voxel size in millimeters.
        """
        if self._header is None:
            raise RuntimeError("Atlas not loaded. Call load() first.")
        return float(self._header.get_zooms()[0])

    @property
    def info(self) -> AtlasInfo:
        """Get atlas info.

        Returns
        -------
        AtlasInfo
            Atlas metadata and label statistics.
        """
        if self._info is None:
            raise RuntimeError("Atlas not loaded. Call load() first.")
        return self._info

    def get_label_stats(self, label: int) -> LabelStats:
        """Get statistics for a specific label.

        Parameters
        ----------
        label : int
            Tissue label.

        Returns
        -------
        LabelStats
            Statistics for the label.
        """
        if label not in self._label_stats:
            raise ValueError(f"Unknown label: {label}")
        return self._label_stats[label]

    def merge_tissues(
        self,
        merge_rules: Dict[int, int],
        inplace: bool = False,
        new_names: Optional[Dict[int, str]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply tissue label merging rules.

        Parameters
        ----------
        merge_rules : Dict[int, int]
            Mapping from original label to merged label.
            Example: {1: 1, 3: 1, 2: 2} merges labels 1,3 to 1, and 2 to 2.
        inplace : bool, default False
            If True, modify the internal volume directly.
        new_names : Dict[int, str], optional
            Names for merged labels.

        Returns
        -------
        Tuple[np.ndarray, Dict]
            (merged volume, reverse mapping from new label to list of original labels)
        """
        if self._volume is None:
            raise RuntimeError("Atlas not loaded. Call load() first.")

        if inplace:
            merged = self._volume
        else:
            merged = self._volume.copy()

        reverse_mapping: Dict[int, List[int]] = {}

        for old_label, new_label in merge_rules.items():
            mask = merged == old_label
            merged[mask] = new_label
            if new_label not in reverse_mapping:
                reverse_mapping[new_label] = []
            reverse_mapping[new_label].append(old_label)

        unmapped_labels = set(np.unique(merged)) - set(merge_rules.values()) - {0}
        for label in unmapped_labels:
            if label not in reverse_mapping:
                reverse_mapping[label] = [label]

        logger.info("Tissue merge applied:")
        for new_label, old_labels in sorted(reverse_mapping.items()):
            logger.info(f"  Merged labels {old_labels} -> {new_label}")

        return merged, reverse_mapping

    def get_tissue_mask(self, label: int) -> np.ndarray:
        """Get boolean mask for a specific tissue label.

        Parameters
        ----------
        label : int
            Target tissue label.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates voxels of the target tissue.
        """
        return self.volume == label

    def get_surface_mask(self, margin: int = 1) -> np.ndarray:
        """Extract surface voxels using morphological operations.

        Surface voxels are those adjacent to background (label 0).

        Parameters
        ----------
        margin : int, default 1
            Number of voxel layers to consider as "surface".

        Returns
        -------
        np.ndarray
            Boolean mask of surface voxels.
        """
        vol = self.volume
        background = (vol == 0).astype(np.uint8)

        structure = np.ones((3, 3, 3), dtype=np.uint8)

        eroded = ndimage.binary_erosion(
            background, structure=structure, iterations=margin
        )

        surface = eroded & ~ndimage.binary_erosion(
            eroded, structure=structure, iterations=1
        )

        return surface

    def get_body_mask(self) -> np.ndarray:
        """Get mask of the entire body (non-background).

        Returns
        -------
        np.ndarray
            Boolean mask of body voxels.
        """
        return self.volume != 0

    def _ensure_edt(self) -> np.ndarray:
        """Compute EDT lazily (only once), then cache it.

        Returns
        -------
        np.ndarray
            Euclidean distance transform of body mask.
        """
        if self._edt is None:
            logger.info("  Computing EDT (one-time, ~5-10s)...")
            body_mask = self.volume != 0
            self._edt = ndimage.distance_transform_edt(body_mask).astype(np.float32)
            logger.info(f"  EDT done. Shape={self._edt.shape}, "
                        f"max_dist={self._edt.max():.1f} voxels")
        return self._edt

    def get_subcutaneous_region(
        self,
        depth_range_mm: Tuple[float, float] = (1.0, 3.0),
        regions: List[str] = None,
        exclude_labels: List[int] = None,
        torso_only: bool = True,
    ) -> np.ndarray:
        """Get subcutaneous region mask at specified depth range.

        Coordinate system (from affine matrix analysis):
        - X: Left (-19mm at voxel 0) to Right (+19mm at voxel 379)
        - Y: Anterior (-50mm at voxel 0) to Posterior (+50mm at voxel 991)
        - Z: Inferior (-10mm at voxel 0) to Superior (+11mm at voxel 207)
        -> Dorsal (back) is +Z, Ventral (belly) is -Z

        Parameters
        ----------
        depth_range_mm : Tuple[float, float], default (1.0, 3.0)
            Depth range from surface in millimeters.
        regions : List[str], optional
            Body regions: "dorsal" (back, +Z), "ventral" (belly, -Z), "lateral" (sides).
            If None, include all regions.
        exclude_labels : List[int], optional
            Tissue labels to exclude from subcutaneous region
            (e.g., organ labels where tumors shouldn't be placed).
        torso_only : bool, default True
            If True, exclude head (anterior Y<30mm) and tail (posterior Y>85mm) regions.

        Returns
        -------
        np.ndarray
            Boolean mask of subcutaneous region.
        """
        if regions is None:
            regions = ["dorsal", "lateral"]

        # Build cache key from all filtering parameters
        cache_key = (
            tuple(depth_range_mm),
            tuple(sorted(regions)),
            tuple(sorted(exclude_labels)) if exclude_labels else (),
            torso_only,
        )

        if cache_key in self._subq_cache:
            return self._subq_cache[cache_key]

        voxel_size = self.voxel_size
        depth_min_vox = max(1, int(depth_range_mm[0] / voxel_size))
        depth_max_vox = max(2, int(depth_range_mm[1] / voxel_size))

        logger.info(
            f"Computing subcutaneous region: depth={depth_range_mm}mm, "
            f"voxels={depth_min_vox}-{depth_max_vox}, regions={regions}"
        )

        vol = self.volume
        x_dim, y_dim, z_dim = vol.shape

        body_mask = vol != 0

        # Use cached EDT instead of recomputing
        dist = self._ensure_edt()

        subq_mask = body_mask & (dist >= depth_min_vox) & (dist <= depth_max_vox)

        if exclude_labels is not None:
            for label in exclude_labels:
                subq_mask = subq_mask & (vol != label)

        if (
            "dorsal" in regions
            and "ventral" not in regions
            and "lateral" not in regions
        ):
            logger.info("  Filtering to dorsal region (+Z / superior)")
            z_center = z_dim // 2
            subq_mask[:, :, :z_center] = False
        elif (
            "ventral" in regions
            and "dorsal" not in regions
            and "lateral" not in regions
        ):
            logger.info("  Filtering to ventral region (-Z / inferior)")
            z_center = z_dim // 2
            subq_mask[:, :, z_center:] = False
        elif (
            "lateral" in regions
            and "dorsal" not in regions
            and "ventral" not in regions
        ):
            logger.info("  Filtering to lateral region (sides)")
            x_center = x_dim // 2
            x_margin = x_dim // 4
            subq_mask[:x_margin, :, :] = False
            subq_mask[x_dim - x_margin :, :, :] = False
        elif "dorsal" in regions and "lateral" in regions:
            logger.info("  Filtering to dorsal+lateral regions")
            z_center = z_dim // 2
            x_margin = x_dim // 4
            subq_mask[:, :, :z_center] = False
            subq_mask[:x_margin, :, :] = False
            subq_mask[x_dim - x_margin :, :, :] = False

        if torso_only:
            y_head_threshold = int(30 / voxel_size)
            y_tail_threshold = y_dim - int(30 / voxel_size)
            logger.info(
                f"  Filtering torso: Y in [{y_head_threshold}, {y_tail_threshold}] (voxel)"
            )
            subq_mask[:, :y_head_threshold, :] = False
            subq_mask[:, y_tail_threshold:, :] = False

        voxel_count = int(np.sum(subq_mask))
        logger.info(
            f"Subcutaneous region: {voxel_count} voxels "
            f"({voxel_count * voxel_size**3:.2f} mm³)"
        )

        self._subq_cache[cache_key] = subq_mask
        return subq_mask

    def get_subcutaneous_coords(
        self,
        depth_range_mm: Tuple[float, float],
        regions: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Return pre-computed physical coordinates for fast random sampling.

        Caches the result so repeated calls with same params are O(1).

        Parameters
        ----------
        depth_range_mm : Tuple[float, float]
            Depth range from surface in millimeters.
        regions : List[str]
            Body regions to include.
        **kwargs : dict
            Forwarded to get_subcutaneous_region (exclude_labels, torso_only).

        Returns
        -------
        np.ndarray
            Physical coordinates [M, 3] in mm.
        """
        cache_key = (tuple(depth_range_mm), tuple(sorted(regions)))

        if cache_key not in self._subq_coords_cache:
            mask = self.get_subcutaneous_region(depth_range_mm, regions, **kwargs)
            voxel_coords = np.argwhere(mask)  # [M, 3] voxel indices
            # Convert to physical coordinates
            physical_coords = voxel_coords.astype(np.float64) * self.voxel_size
            self._subq_coords_cache[cache_key] = physical_coords
            logger.info(
                f"  Cached {len(physical_coords)} subcutaneous coords "
                f"for depth={depth_range_mm}, regions={regions}"
            )

        return self._subq_coords_cache[cache_key]

    def get_torso_slice(
        self, axis: str = "z", position: float = 0.5
    ) -> Tuple[np.ndarray, int]:
        """Get a slice from the torso region.

        Parameters
        ----------
        axis : str, default "z"
            Axis to slice: "x" (sagittal), "y" (coronal), "z" (axial)
        position : float, default 0.5
            Position along the axis (0.0 to 1.0).

        Returns
        -------
        Tuple[np.ndarray, int]
            (slice data, slice index)
        """
        vol = self.volume
        if axis == "z":
            idx = int(vol.shape[2] * position)
            slc = vol[:, :, idx]
        elif axis == "y":
            idx = int(vol.shape[1] * position)
            slc = vol[:, idx, :]
        elif axis == "x":
            idx = int(vol.shape[0] * position)
            slc = vol[idx, :, :]
        else:
            raise ValueError(f"Unknown axis: {axis}")

        return slc, idx

    def save(
        self,
        filepath: str,
        merged_volume: Optional[np.ndarray] = None,
        tumor_region_mask: Optional[np.ndarray] = None,
        merge_info: Optional[Dict] = None,
    ) -> Path:
        """Save atlas data to .npz file.

        Parameters
        ----------
        filepath : str
            Output file path.
        merged_volume : np.ndarray, optional
            Merged tissue labels volume.
        tumor_region_mask : np.ndarray, optional
            Boolean mask of tumor placement region.
        merge_info : Dict, optional
            Information about tissue merge mapping.

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "original_labels": self.volume,
            "voxel_size": self.voxel_size,
            "shape": np.array(self.shape),
        }

        if merged_volume is not None:
            save_dict["tissue_labels"] = merged_volume

        if tumor_region_mask is not None:
            save_dict["tumor_region_mask"] = tumor_region_mask

        if merge_info is not None:
            save_dict["merge_info"] = merge_info

        if self._info is not None:
            label_stats_list = []
            for stats in self._info.label_stats:
                label_stats_list.append(
                    {
                        "label": stats.label,
                        "name": stats.name,
                        "voxel_count": stats.voxel_count,
                        "centroid": stats.centroid,
                        "bounding_box": stats.bounding_box,
                    }
                )
            save_dict["label_stats"] = label_stats_list

        np.savez(filepath, **save_dict)
        logger.info(f"Atlas saved to: {filepath}")

        return filepath

    @staticmethod
    def load_saved(filepath: str) -> Dict:
        """Load saved atlas data from .npz file.

        Parameters
        ----------
        filepath : str
            Path to .npz file.

        Returns
        -------
        Dict
            Dictionary with atlas data.
        """
        data = np.load(filepath, allow_pickle=True)
        result = {k: data[k] for k in data.files}
        if "merge_info" in result and isinstance(result["merge_info"], np.ndarray):
            result["merge_info"] = result["merge_info"].item()
        return result
