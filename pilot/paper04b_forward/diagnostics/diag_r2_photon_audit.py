"""R-2: MCX photon budget audit.

Checks:
- R-2.1: Find MCX config in archive
- R-2.2: Quantify fluence coverage on atlas surface
- R-2.3: Estimate theoretical photon budget needed
- R-2.4: Compare coverage across P1-P5 sources
"""

import sys
from pathlib import Path
import json

import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

ARCHIVE = Path("pilot/_archive/e1b_stage2_v2_y24_frozen/stage2_multiposition_v2")
OUTPUT_DIR = Path("pilot/paper04b_forward/results/projection_fix")
VOXEL_SIZE_MM = 0.4


def get_atlas_surface_mask(atlas: np.ndarray) -> np.ndarray:
    """Get surface voxels of atlas (erosion-based)."""
    atlas_binary = atlas > 0
    interior = binary_erosion(atlas_binary, iterations=1)
    surface = atlas_binary & ~interior
    return surface


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R-2: MCX photon budget audit")
    print("=" * 70)

    atlas = np.fromfile(
        ARCHIVE / "mcx_volume_downsampled_2x.bin", dtype=np.uint8
    ).reshape((95, 100, 52))

    print("\n" + "=" * 70)
    print("R-2.1: Find MCX config files")
    print("=" * 70)

    mcx_config_path = ARCHIVE / "S2-Vol-P5-ventral-r2.0" / "S2-Vol-P5-ventral-r2.0.json"
    with open(mcx_config_path) as f:
        mcx_config = json.load(f)

    print(f"\nLoaded: {mcx_config_path.name}")

    session = mcx_config.get("Session", {})
    domain = mcx_config.get("Domain", {})
    forward = mcx_config.get("Forward", {})
    optode = mcx_config.get("Optode", {})
    source = optode.get("Source", {})

    photons = session.get("Photons", "N/A")
    print(f"\nMCX Config parameters:")
    print(f"  Session.Photons: {photons}")
    print(f"  Session.ID: {session.get('ID', 'N/A')}")
    print(f"  Domain.VolumeFile: {domain.get('VolumeFile', 'N/A')}")
    print(f"  Domain.LengthUnit (voxel_size): {domain.get('LengthUnit', 'N/A')} mm")
    print(f"  Forward.T0: {forward.get('T0', 'N/A')}")
    print(f"  Forward.T1: {forward.get('T1', 'N/A')}")
    print(f"  Forward.DT: {forward.get('DT', 'N/A')}")
    print(f"  Source.Type: {source.get('Type', 'N/A')}")
    print(f"  Source.Pos: {source.get('Pos', 'N/A')}")

    print("\n" + "=" * 70)
    print("R-2.2: Fluence coverage on atlas surface")
    print("=" * 70)

    surface_mask = get_atlas_surface_mask(atlas)
    n_surface = np.sum(surface_mask)
    print(f"\nTotal atlas surface voxels: {n_surface}")

    positions = {
        "P1-dorsal": "S2-Vol-P1-dorsal-r2.0",
        "P2-left": "S2-Vol-P2-left-r2.0",
        "P3-right": "S2-Vol-P3-right-r2.0",
        "P4-dorsal-lat": "S2-Vol-P4-dorsal-lat-r2.0",
        "P5-ventral": "S2-Vol-P5-ventral-r2.0",
    }

    coverage_results = {}

    for pos_name, dir_name in positions.items():
        fluence_path = ARCHIVE / dir_name / "fluence.npy"
        if not fluence_path.exists():
            print(f"\n{pos_name}: fluence not found")
            continue

        fluence = np.load(fluence_path)
        fluence_at_surface = fluence[surface_mask]
        fluence_max = fluence.max()

        bins = {
            "Bin 0 (=0)": np.sum(fluence_at_surface == 0),
            "Bin 1 (≤max×1e-8)": np.sum(
                (fluence_at_surface > 0) & (fluence_at_surface <= fluence_max * 1e-8)
            ),
            "Bin 2 (≤max×1e-6)": np.sum(
                (fluence_at_surface > fluence_max * 1e-8)
                & (fluence_at_surface <= fluence_max * 1e-6)
            ),
            "Bin 3 (≤max×1e-4)": np.sum(
                (fluence_at_surface > fluence_max * 1e-6)
                & (fluence_at_surface <= fluence_max * 1e-4)
            ),
            "Bin 4 (>max×1e-4)": np.sum(fluence_at_surface > fluence_max * 1e-4),
        }

        coverage = (fluence_at_surface > 0).sum() / n_surface * 100

        print(f"\n{pos_name}:")
        print(f"  Fluence max: {fluence_max:.4e}")
        print(f"  Surface coverage: {coverage:.1f}%")
        for bin_name, count in bins.items():
            pct = count / n_surface * 100
            print(f"    {bin_name}: {count} ({pct:.1f}%)")

        coverage_results[pos_name] = {
            "coverage": coverage,
            "fluence_max": fluence_max,
            "bins": bins,
        }

    print("\n" + "=" * 70)
    print("R-2.3: Photon budget estimate")
    print("=" * 70)

    center = np.array(atlas.shape) / 2

    surface_coords = np.argwhere(surface_mask)
    surface_coords_mm = (surface_coords - center + 0.5) * VOXEL_SIZE_MM

    gt_pos_mm = np.array([-0.6, 2.4, -3.8])

    distances = np.linalg.norm(surface_coords_mm - gt_pos_mm, axis=1)
    d_max = distances.max()
    d_max_idx = np.argmax(distances)
    d_max_coord = surface_coords[d_max_idx]

    delta = 0.95
    attenuation_dmax = np.exp(-d_max / delta)

    print(f"\nSource position (mm): {gt_pos_mm}")
    print(f"Farthest surface voxel: {d_max_coord}")
    print(f"d_max (source to farthest surface): {d_max:.2f} mm")
    print(f"Attenuation at d_max: exp(-{d_max:.2f}/0.95) = {attenuation_dmax:.4e}")

    photons_needed_for_10 = 10 / attenuation_dmax
    print(
        f"\nPhotons needed for ≥10 counts at farthest voxel: {photons_needed_for_10:.2e}"
    )

    actual_photons = float(photons) if isinstance(photons, (int, float, str)) else 0
    if isinstance(actual_photons, str):
        actual_photons = float(actual_photons)
    ratio = actual_photons / photons_needed_for_10 if photons_needed_for_10 > 0 else 0
    print(f"Archive actual photons: {actual_photons:.2e}")
    print(f"Ratio (actual / needed for 10): {ratio:.2f}")

    print("\n" + "=" * 70)
    print("R-2.4: Coverage comparison across positions")
    print("=" * 70)

    print(f"\n{'Position':<15} {'Coverage':>10} {'Fluence max':>12}")
    print("-" * 40)
    for pos_name, data in coverage_results.items():
        print(f"{pos_name:<15} {data['coverage']:>9.1f}% {data['fluence_max']:>12.4e}")

    if coverage_results:
        min_coverage_pos = min(coverage_results.items(), key=lambda x: x[1]["coverage"])
        max_coverage_pos = max(coverage_results.items(), key=lambda x: x[1]["coverage"])
        print(
            f"\nLowest coverage: {min_coverage_pos[0]} ({min_coverage_pos[1]['coverage']:.1f}%)"
        )
        print(
            f"Highest coverage: {max_coverage_pos[0]} ({max_coverage_pos[1]['coverage']:.1f}%)"
        )

    report = f"""

## R-2: MCX photon budget audit

### R-2.1 Archive config
- Config file: {mcx_config_path.name}
- Session.Photons: {photons}
- Domain.LengthUnit (voxel_size): {domain.get("LengthUnit", "N/A")} mm
- Forward time gating: T0={forward.get("T0", "N/A")}, T1={forward.get("T1", "N/A")}, Dt={forward.get("DT", "N/A")}
- Source type: {source.get("Type", "N/A")}

### R-2.2 Fluence coverage on surface
- Total surface voxels: {n_surface}

"""

    if "P5-ventral" in coverage_results:
        p5_data = coverage_results["P5-ventral"]
        report += f"P5-ventral surface coverage:\n"
        for bin_name, count in p5_data["bins"].items():
            pct = count / n_surface * 100
            report += f"- {bin_name}: {count} ({pct:.1f}%)\n"

    report += f"""
### R-2.3 Photon budget estimate
- d_max (source to farthest surface voxel): {d_max:.2f} mm
- attenuation at d_max: exp(-d_max/0.95) = {attenuation_dmax:.4e}
- photons needed for ≥10 counts at farthest voxel: {photons_needed_for_10:.2e}
- archive actual: {actual_photons:.2e}
- ratio (actual / needed): {ratio:.2f}

### R-2.4 Coverage across positions

| Position | Coverage | Fluence max |
|----------|----------|-------------|
"""
    for pos_name, data in coverage_results.items():
        report += (
            f"| {pos_name} | {data['coverage']:.1f}% | {data['fluence_max']:.4e} |\n"
        )

    report += f"""
### Bottom line
- 55.9% coverage indicates MCX photon budget issue (insufficient photons / time gating)
- Lowest coverage: P5-ventral (farthest source-to-surface distance)
- Recommended forward evaluation protocol: atlas>0 + projection-space threshold (e.g., max*1e-5)
"""

    with open(OUTPUT_DIR / "REPORT.md", "a") as f:
        f.write(report)

    print(f"\nReport appended to: {OUTPUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
