#!/usr/bin/env python3
"""Coordinate sanity test for E1c.

Verifies coordinate mapping from config → world → MCX → surface image.
"""

import json
from pathlib import Path
import numpy as np
import yaml


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def config_to_world(source_center: list) -> dict:
    """Convert config source_center to world coordinates.

    Args:
        source_center: [x_mm, y_mm, depth_from_dorsal_mm]

    Returns:
        dict with world coordinates
    """
    x_mm, y_mm, depth = source_center
    z_world = 10.0 - depth
    return {
        "x_world": x_mm,
        "y_world": y_mm,
        "z_world": z_world,
        "depth_from_dorsal_mm": depth,
    }


def world_to_mcx(world_coords: dict) -> dict:
    """Convert world coordinates to MCX box coordinates.

    Args:
        world_coords: dict with x_world, y_world, z_world

    Returns:
        dict with MCX coordinates
    """
    return {
        "x_mcx_mm": world_coords["x_world"] + 15.0,
        "y_mcx_mm": world_coords["y_world"] + 15.0,
        "z_mcx_mm": world_coords["depth_from_dorsal_mm"],
    }


def world_to_surface_pixel(
    x_world: float,
    y_world: float,
    image_size: int,
    pixel_size_mm: float,
) -> dict:
    """Convert world position to surface image pixel coordinates.

    Args:
        x_world, y_world: world coordinates in mm
        image_size: image dimension (pixels)
        pixel_size_mm: pixel size in mm

    Returns:
        dict with col, row, u_mm, v_mm
    """
    fov_mm = image_size * pixel_size_mm
    u_mm = x_world
    v_mm = y_world

    col = (u_mm + fov_mm / 2) / pixel_size_mm
    row = (v_mm + fov_mm / 2) / pixel_size_mm

    return {
        "u_mm": u_mm,
        "v_mm": v_mm,
        "col": col,
        "row": row,
    }


def verify_sanity():
    """Run sanity checks on all test configs."""
    config = load_config()
    image_size = config["image"]["size"]
    pixel_size_mm = config["image"]["pixel_size_mm"]
    fov_mm = config["image"]["fov_mm"]

    results = []

    for config_id, cfg in config["configs"].items():
        source_center = cfg["source_center"]

        world_coords = config_to_world(source_center)
        mcx_coords = world_to_mcx(world_coords)
        pixel_coords = world_to_surface_pixel(
            world_coords["x_world"],
            world_coords["y_world"],
            image_size,
            pixel_size_mm,
        )

        result = {
            "config_id": config_id,
            "x_mm": source_center[0],
            "y_mm": source_center[1],
            "depth_from_dorsal_mm": source_center[2],
            "x_world": world_coords["x_world"],
            "y_world": world_coords["y_world"],
            "z_world": world_coords["z_world"],
            "x_mcx_mm": mcx_coords["x_mcx_mm"],
            "y_mcx_mm": mcx_coords["y_mcx_mm"],
            "z_mcx_mm": mcx_coords["z_mcx_mm"],
            "expected_peak_u_mm": pixel_coords["u_mm"],
            "expected_peak_v_mm": pixel_coords["v_mm"],
            "expected_peak_col": pixel_coords["col"],
            "expected_peak_row": pixel_coords["row"],
            "image_size": image_size,
            "pixel_size_mm": pixel_size_mm,
            "fov_mm": fov_mm,
        }
        results.append(result)

    return results


def check_constraints(results: list) -> list:
    """Verify coordinate constraints."""
    errors = []

    for r in results:
        config_id = r["config_id"]

        if r["x_mcx_mm"] < 0 or r["x_mcx_mm"] > 30:
            errors.append(f"{config_id}: x_mcx_mm={r['x_mcx_mm']} out of [0, 30]")
        if r["y_mcx_mm"] < 0 or r["y_mcx_mm"] > 30:
            errors.append(f"{config_id}: y_mcx_mm={r['y_mcx_mm']} out of [0, 30]")
        if r["z_mcx_mm"] < 0 or r["z_mcx_mm"] > 20:
            errors.append(f"{config_id}: z_mcx_mm={r['z_mcx_mm']} out of [0, 20]")

        if r["expected_peak_col"] < 0 or r["expected_peak_col"] > r["image_size"]:
            errors.append(
                f"{config_id}: col={r['expected_peak_col']} out of image bounds"
            )
        if r["expected_peak_row"] < 0 or r["expected_peak_row"] > r["image_size"]:
            errors.append(
                f"{config_id}: row={r['expected_peak_row']} out of image bounds"
            )

    for r in results:
        config_id = r["config_id"]

        if config_id == "M01":
            if abs(r["expected_peak_col"] - r["image_size"] / 2) > 1:
                errors.append(
                    f"M01: peak col {r['expected_peak_col']} should be at center {r['image_size'] / 2}"
                )
            if abs(r["expected_peak_row"] - r["image_size"] / 2) > 1:
                errors.append(
                    f"M01: peak row {r['expected_peak_row']} should be at center {r['image_size'] / 2}"
                )

        if config_id == "M03":
            if r["expected_peak_col"] <= r["image_size"] / 2:
                errors.append(
                    f"M03: peak col {r['expected_peak_col']} should be > center (right of center)"
                )
            if r["expected_peak_row"] <= r["image_size"] / 2:
                errors.append(
                    f"M03: peak row {r['expected_peak_row']} should be > center (bottom of center)"
                )

    return errors


def format_table(results: list) -> str:
    """Format results as text table."""
    lines = []
    lines.append("=" * 120)
    lines.append("E1c Coordinate Sanity Check")
    lines.append("=" * 120)
    lines.append("")

    header = (
        f"{'Config':<6} "
        f"{'x_mm':>6} {'y_mm':>6} {'depth':>6} | "
        f"{'x_world':>8} {'y_world':>8} {'z_world':>8} | "
        f"{'x_mcx':>7} {'y_mcx':>7} {'z_mcx':>6} | "
        f"{'peak_col':>9} {'peak_row':>9}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for r in results:
        line = (
            f"{r['config_id']:<6} "
            f"{r['x_mm']:>6.1f} {r['y_mm']:>6.1f} {r['depth_from_dorsal_mm']:>6.1f} | "
            f"{r['x_world']:>8.1f} {r['y_world']:>8.1f} {r['z_world']:>8.1f} | "
            f"{r['x_mcx_mm']:>7.1f} {r['y_mcx_mm']:>7.1f} {r['z_mcx_mm']:>6.1f} | "
            f"{r['expected_peak_col']:>9.1f} {r['expected_peak_row']:>9.1f}"
        )
        lines.append(line)

    lines.append("")
    lines.append(f"Image size: {results[0]['image_size']} x {results[0]['image_size']}")
    lines.append(f"Pixel size: {results[0]['pixel_size_mm']} mm")
    lines.append(f"FOV: {results[0]['fov_mm']} mm")
    lines.append("")

    return "\n".join(lines)


def main():
    config = load_config()
    output_dir = Path(__file__).parent / config["output"]["sanity_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running coordinate sanity check...")

    results = verify_sanity()
    errors = check_constraints(results)

    table_str = format_table(results)
    print(table_str)

    if errors:
        print("\n" + "=" * 80)
        print("ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        print("=" * 80)
        sanity_passed = False
    else:
        print("\n" + "=" * 80)
        print("ALL SANITY CHECKS PASSED")
        print("=" * 80)
        sanity_passed = True

    with open(output_dir / "coordinate_sanity.json", "w") as f:
        json.dump(
            {
                "sanity_passed": sanity_passed,
                "errors": errors,
                "results": results,
            },
            f,
            indent=2,
        )

    with open(output_dir / "coordinate_sanity.txt", "w") as f:
        f.write(table_str)
        if errors:
            f.write("\n\nERRORS:\n")
            for e in errors:
                f.write(f"  - {e}\n")
        else:
            f.write("\n\nALL CHECKS PASSED\n")

    print(f"\nResults saved to {output_dir}")

    return sanity_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
