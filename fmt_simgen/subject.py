"""Subject geometry and label contracts.

This module is the runtime geometry source of truth for subject-specific
volumes.  The legacy Digimouse constants remain available through
``fmt_simgen.frame_contract`` for compatibility, but pipeline code should use
``SubjectManifest`` so other CT/segmentation inputs can define their own frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class VolumeSpec:
    """Voxel volume geometry in trunk-local coordinates."""

    shape_xyz: tuple[int, int, int]
    voxel_size_mm: float
    origin_world_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Return volume shape in MCX ZYX order."""
        x, y, z = self.shape_xyz
        return (z, y, x)

    @property
    def extent_mm(self) -> np.ndarray:
        """Return physical extent in XYZ millimeters."""
        return np.asarray(self.shape_xyz, dtype=np.float64) * float(self.voxel_size_mm)

    @property
    def center_world_mm(self) -> np.ndarray:
        """Return geometric center in trunk-local millimeters."""
        return np.asarray(self.origin_world_mm, dtype=np.float64) + self.extent_mm / 2.0


@dataclass
class LabelRoleSpec:
    """Semantic label roles used by tumor placement and validation."""

    background_labels: tuple[int, ...] = (0,)
    allowed_tumor_labels: tuple[int, ...] = (1,)
    forbidden_tumor_labels: tuple[int, ...] = (0, 2)


@dataclass
class SubjectManifest:
    """Serializable subject-level geometry, label, and artifact contract."""

    subject_id: str
    world_frame: str
    output_dir: str
    mcx_volume: VolumeSpec
    atlas_to_world_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    crop_bbox_mm: Optional[dict[str, list[float]]] = None
    segmentation_path: Optional[str] = None
    segmentation_format: Optional[str] = None
    label_key: Optional[str] = None
    label_mapping: dict[int, int] = field(default_factory=dict)
    label_roles: LabelRoleSpec = field(default_factory=LabelRoleSpec)

    @property
    def volume_center_world_mm(self) -> np.ndarray:
        """Return volume rotation center in trunk-local millimeters."""
        return self.mcx_volume.center_world_mm

    @property
    def volume_extents_mm(self) -> np.ndarray:
        """Return MCX/GT volume extent in millimeters."""
        return self.mcx_volume.extent_mm

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        """Return MCX/GT volume shape in XYZ order."""
        return self.mcx_volume.shape_xyz

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Return MCX volume shape in ZYX order."""
        return self.mcx_volume.shape_zyx

    @property
    def voxel_size_mm(self) -> float:
        """Return MCX/GT voxel size in millimeters."""
        return self.mcx_volume.voxel_size_mm

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        data = asdict(self)
        data["mcx_volume"]["shape_xyz"] = list(self.mcx_volume.shape_xyz)
        data["mcx_volume"]["origin_world_mm"] = list(self.mcx_volume.origin_world_mm)
        data["mcx_volume"]["shape_zyx"] = list(self.shape_zyx)
        data["mcx_volume"]["extent_mm"] = self.volume_extents_mm.tolist()
        data["volume_center_world_mm"] = self.volume_center_world_mm.tolist()
        data["atlas_to_world_offset_mm"] = list(self.atlas_to_world_offset_mm)
        data["label_mapping"] = {str(k): int(v) for k, v in self.label_mapping.items()}
        data["label_roles"] = {
            "background_labels": list(self.label_roles.background_labels),
            "allowed_tumor_labels": list(self.label_roles.allowed_tumor_labels),
            "forbidden_tumor_labels": list(self.label_roles.forbidden_tumor_labels),
        }
        return data

    def save(self, path: str | Path) -> Path:
        """Save the manifest as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubjectManifest":
        """Create a manifest from a serialized dictionary."""
        mcx_data = data.get("mcx_volume", {})
        shape_xyz = tuple(int(v) for v in mcx_data["shape_xyz"])
        origin = tuple(float(v) for v in mcx_data.get("origin_world_mm", (0.0, 0.0, 0.0)))
        roles = data.get("label_roles", {})
        return cls(
            subject_id=str(data.get("subject_id", "subject")),
            world_frame=str(data.get("world_frame", "mcx_trunk_local_mm")),
            output_dir=str(data.get("output_dir", "output/shared")),
            mcx_volume=VolumeSpec(
                shape_xyz=shape_xyz,
                voxel_size_mm=float(mcx_data["voxel_size_mm"]),
                origin_world_mm=origin,
            ),
            atlas_to_world_offset_mm=tuple(
                float(v) for v in data.get("atlas_to_world_offset_mm", (0.0, 0.0, 0.0))
            ),
            crop_bbox_mm=data.get("crop_bbox_mm"),
            segmentation_path=data.get("segmentation_path"),
            segmentation_format=data.get("segmentation_format"),
            label_key=data.get("label_key"),
            label_mapping={int(k): int(v) for k, v in data.get("label_mapping", {}).items()},
            label_roles=LabelRoleSpec(
                background_labels=tuple(int(v) for v in roles.get("background_labels", (0,))),
                allowed_tumor_labels=tuple(int(v) for v in roles.get("allowed_tumor_labels", (1,))),
                forbidden_tumor_labels=tuple(int(v) for v in roles.get("forbidden_tumor_labels", (0, 2))),
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SubjectManifest":
        """Load a manifest JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def _shape_xyz_from_mcx_config(mcx_config: dict[str, Any]) -> tuple[int, int, int]:
    """Read MCX ZYX shape from config and return XYZ shape."""
    shape_zyx = tuple(int(v) for v in mcx_config.get("volume_shape", [104, 200, 190]))
    return (shape_zyx[2], shape_zyx[1], shape_zyx[0])


def _label_roles_from_config(config: dict[str, Any]) -> LabelRoleSpec:
    """Read semantic label roles from subject/tumor config with legacy defaults."""
    subject_roles = config.get("subject", {}).get("label_roles", {})
    tumor_cfg = config.get("tumor", {})
    return LabelRoleSpec(
        background_labels=tuple(int(v) for v in subject_roles.get("background_labels", [0])),
        allowed_tumor_labels=tuple(
            int(v) for v in subject_roles.get("allowed_tumor_labels", tumor_cfg.get("allowed_tumor_labels", [1]))
        ),
        forbidden_tumor_labels=tuple(
            int(v) for v in subject_roles.get("forbidden_tumor_labels", tumor_cfg.get("forbidden_tumor_labels", [0, 2]))
        ),
    )


def subject_manifest_from_config(config: dict[str, Any]) -> SubjectManifest:
    """Build a subject manifest from config, preserving legacy Digimouse defaults."""
    subject_cfg = config.get("subject", {})
    manifest_path = subject_cfg.get("manifest_path")
    if manifest_path:
        return SubjectManifest.load(manifest_path)

    mcx_cfg = config.get("mcx", {})
    mesh_cfg = config.get("mesh", {})
    output_dir = subject_cfg.get("output_dir") or mesh_cfg.get("output_path", "output/shared")
    voxel_size = float(subject_cfg.get("target_voxel_size_mm", mcx_cfg.get("voxel_size_mm", 0.2)))

    if "volume_shape_xyz" in subject_cfg:
        shape_xyz = tuple(int(v) for v in subject_cfg["volume_shape_xyz"])
    elif subject_cfg.get("crop_bbox_mm") is not None:
        bbox = subject_cfg["crop_bbox_mm"]
        shape_xyz = tuple(
            int(round((float(bbox[axis][1]) - float(bbox[axis][0])) / voxel_size))
            for axis in ("x", "y", "z")
        )
    else:
        shape_xyz = _shape_xyz_from_mcx_config(mcx_cfg)

    trunk_offset = tuple(
        float(v) for v in subject_cfg.get(
            "atlas_to_world_offset_mm",
            mcx_cfg.get("trunk_offset_mm", [0.0, 34.0, 0.0]),
        )
    )

    label_mapping = subject_cfg.get("label_mapping", mcx_cfg.get("tissue_mapping", {}))
    return SubjectManifest(
        subject_id=str(subject_cfg.get("id", "digimouse_legacy")),
        world_frame=str(subject_cfg.get("world_frame", "mcx_trunk_local_mm")),
        output_dir=str(output_dir),
        mcx_volume=VolumeSpec(
            shape_xyz=shape_xyz,
            voxel_size_mm=voxel_size,
            origin_world_mm=tuple(float(v) for v in subject_cfg.get("origin_world_mm", [0.0, 0.0, 0.0])),
        ),
        atlas_to_world_offset_mm=trunk_offset,
        crop_bbox_mm=subject_cfg.get("crop_bbox_mm"),
        segmentation_path=subject_cfg.get("segmentation_path", config.get("atlas", {}).get("path")),
        segmentation_format=subject_cfg.get("format", "npz"),
        label_key=subject_cfg.get("label_key"),
        label_mapping={int(k): int(v) for k, v in label_mapping.items()},
        label_roles=_label_roles_from_config(config),
    )


def load_subject_manifest(config: dict[str, Any], shared_dir: str | Path | None = None) -> SubjectManifest:
    """Load manifest from shared directory if present, otherwise derive from config."""
    subject_cfg = config.get("subject", {})
    if subject_cfg:
        return subject_manifest_from_config(config)

    if shared_dir is not None:
        manifest_path = Path(shared_dir) / "frame_manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            if "mcx_volume" in data and "shape_xyz" in data["mcx_volume"]:
                return SubjectManifest.from_dict(data)
    return subject_manifest_from_config(config)


def load_segmentation_labels(
    path: str | Path,
    label_key: str | None = None,
) -> tuple[np.ndarray, float]:
    """Load segmentation labels from NPZ or NIfTI.

    Returns labels in XYZ array order and an isotropic voxel size in mm.  NIfTI
    inputs are expected to be pre-oriented into the project canonical axes.
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        key = label_key
        if key is None:
            for candidate in ("original_labels", "tissue_labels", "labels", "data", "volume"):
                if candidate in data:
                    key = candidate
                    break
        if key is None:
            raise ValueError(f"No label key found in {path}; keys={list(data.keys())}")
        voxel_size = float(data["voxel_size"]) if "voxel_size" in data else 1.0
        return np.asarray(data[key]), voxel_size

    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        import nibabel as nib

        img = nib.load(str(path))
        labels = np.asarray(img.get_fdata(), dtype=np.int32)
        zooms = img.header.get_zooms()[:3]
        if max(zooms) - min(zooms) > 1e-6:
            raise ValueError(f"Anisotropic NIfTI voxels are not supported yet: {zooms}")
        return labels, float(zooms[0])

    raise ValueError(f"Unsupported segmentation format: {path}")
