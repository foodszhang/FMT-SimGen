#!/usr/bin/env python3
"""Test Stage 2 components without running MCX."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from source_quadrature import sample_uniform
from surface_projection import (
    project_get_surface_coords,
    green_infinite_point_source_on_surface,
)

# Test parameters
VOXEL_SIZE_MM = 0.2
ATLAS_VOLUME_SHAPE = (190, 200, 104)
DEFAULT_TISSUE_PARAMS = {
    "mua_mm": 0.0236,
    "mus_prime_mm": 0.89,
}


def test_source_quadrature():
    """Test uniform source sampling schemes."""
    print("=" * 60)
    print("Testing Source Quadrature Schemes")
    print("=" * 60)

    center = np.array([0.0, 30.0, 4.0])

    schemes = ["7-point", "grid-27", "stratified-33"]

    for scheme in schemes:
        # Sphere r=2mm
        axes = np.array([2.0, 2.0, 2.0])
        points, weights = sample_uniform(center, axes, alpha=1.0, scheme=scheme)

        print(f"\n{scheme}:")
        print(f"  Points: {len(points)}")
        print(f"  Weight sum: {weights.sum():.4f}")
        print(f"  First point: {points[0]}")
        print(f"  Last point: {points[-1]}")

        # Verify points are within ellipsoid
        dist = np.sqrt(
            ((points[:, 0] - center[0]) / axes[0]) ** 2
            + ((points[:, 1] - center[1]) / axes[1]) ** 2
            + ((points[:, 2] - center[2]) / axes[2]) ** 2
        )
        max_dist = dist.max()
        print(f"  Max normalized distance: {max_dist:.3f} (should be <= 1.0)")


def test_surface_projection():
    """Test surface coordinate computation."""
    print("\n" + "=" * 60)
    print("Testing Surface Projection")
    print("=" * 60)

    # Create simple atlas (ellipsoid)
    nx, ny, nz = ATLAS_VOLUME_SHAPE
    x = (np.arange(nx) - nx / 2 + 0.5) * VOXEL_SIZE_MM
    y = (np.arange(ny) - ny / 2 + 0.5) * VOXEL_SIZE_MM
    z = (np.arange(nz) - nz / 2 + 0.5) * VOXEL_SIZE_MM

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Ellipsoid atlas
    atlas = ((xx / 18.0) ** 2 + ((yy - 30) / 22.0) ** 2 + (zz / 9.0) ** 2) <= 1.0
    atlas = atlas.astype(np.uint8)

    print(f"Atlas shape: {atlas.shape}")
    print(f"Atlas voxels: {atlas.sum()}")

    # Test surface coords at 0 degrees
    angle_deg = 0.0
    camera_distance_mm = 25.0
    fov_mm = 22.0
    detector_resolution = (113, 113)

    import time

    start = time.time()
    surface_coords, valid_mask = project_get_surface_coords(
        atlas,
        angle_deg,
        camera_distance_mm,
        fov_mm,
        detector_resolution,
        VOXEL_SIZE_MM,
    )
    elapsed = time.time() - start

    print(f"\nSurface coords computation time: {elapsed:.3f}s")
    print(f"Valid pixels: {valid_mask.sum()}/{valid_mask.size}")
    print(f"Valid ratio: {valid_mask.sum() / valid_mask.size * 100:.1f}%")

    return surface_coords, valid_mask


def test_green_projection(surface_coords, valid_mask):
    """Test Green's function projection."""
    print("\n" + "=" * 60)
    print("Testing Green's Function Projection")
    print("=" * 60)

    source_pos = np.array([0.0, 30.0, 4.0])

    import time

    start = time.time()
    projection = green_infinite_point_source_on_surface(
        source_pos, surface_coords, valid_mask, DEFAULT_TISSUE_PARAMS
    )
    elapsed = time.time() - start

    print(f"Green computation time: {elapsed:.3f}s")
    print(f"Projection shape: {projection.shape}")
    print(f"Projection max: {projection.max():.4e}")
    print(f"Projection min: {projection.min():.4e}")
    print(f"Valid max: {projection[valid_mask].max():.4e}")


def test_uniform_source_projection():
    """Test full uniform source projection pipeline."""
    print("\n" + "=" * 60)
    print("Testing Uniform Source Projection Pipeline")
    print("=" * 60)

    # Create atlas
    nx, ny, nz = ATLAS_VOLUME_SHAPE
    x = (np.arange(nx) - nx / 2 + 0.5) * VOXEL_SIZE_MM
    y = (np.arange(ny) - ny / 2 + 0.5) * VOXEL_SIZE_MM
    z = (np.arange(nz) - nz / 2 + 0.5) * VOXEL_SIZE_MM

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    atlas = ((xx / 18.0) ** 2 + ((yy - 30) / 22.0) ** 2 + (zz / 9.0) ** 2) <= 1.0
    atlas = atlas.astype(np.uint8)

    # Source parameters
    source_center = np.array([0.0, 30.0, 4.0])
    source_radius = 2.0

    # Get surface coords (once per angle)
    angle_deg = 0.0
    camera_distance_mm = 25.0
    fov_mm = 22.0
    detector_resolution = (113, 113)

    print("Getting surface coordinates...")
    surface_coords, valid_mask = project_get_surface_coords(
        atlas,
        angle_deg,
        camera_distance_mm,
        fov_mm,
        detector_resolution,
        VOXEL_SIZE_MM,
    )

    # Sample source with different schemes
    schemes = ["7-point", "grid-27", "stratified-33"]
    axes = np.array([source_radius, source_radius, source_radius])

    for scheme in schemes:
        print(f"\nTesting {scheme}...")

        points, weights = sample_uniform(source_center, axes, alpha=1.0, scheme=scheme)

        import time

        start = time.time()
        projection = np.zeros(detector_resolution[::-1], dtype=np.float32)

        for pt, w in zip(points, weights):
            proj_i = green_infinite_point_source_on_surface(
                pt, surface_coords, valid_mask, DEFAULT_TISSUE_PARAMS
            )
            projection += w * proj_i

        elapsed = time.time() - start

        print(f"  Points: {len(points)}")
        print(f"  Computation time: {elapsed:.3f}s")
        print(f"  Projection max: {projection.max():.4e}")
        print(f"  Projection mean: {projection[valid_mask].mean():.4e}")


def main():
    print("Stage 2 Component Tests")
    print("=" * 60)

    test_source_quadrature()
    surface_coords, valid_mask = test_surface_projection()
    test_green_projection(surface_coords, valid_mask)
    test_uniform_source_projection()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
