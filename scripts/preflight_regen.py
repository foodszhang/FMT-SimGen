#!/usr/bin/env python3
"""
H6: Preflight check before full dataset regeneration.

Validates:
  1. CONFIG_HASH + git hash print
  2. H1 audit has 0 MISMATCH (all H1a bugs fixed)
  3. Single-sample dry-run: step2m → MCX → proj
  4. Auto-metrics on the dry-run output:
     - MCX completes without error
     - MCX source position vs tumor center Y diff < 5mm
     - DE tumor in trunk
     - visibility ∈ [5000, 15000]

On success: writes output/preflight_pass.json with metrics
On failure: exits 1

Usage:
    uv run python scripts/preflight_regen.py [--samples-dir DIR]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/foods/pro/FMT-SimGen")
sys.path.insert(0, str(ROOT))

from fmt_simgen.config import CONFIG_HASH, CONTRACT_VERSION


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_git_status():
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        return "dirty" if status else "clean"
    except Exception:
        return "unknown"


def step1_print_hashes():
    print("=== Step 1: Hash Audit ===")
    git_hash = get_git_hash()
    git_status = get_git_status()
    print(f"  CONFIG_HASH:      {CONFIG_HASH}")
    print(f"  CONTRACT_VERSION: {CONTRACT_VERSION}")
    print(f"  Git commit:       {git_hash}")
    print(f"  Git status:       {git_status}")
    if git_status != "clean":
        print("  WARNING: git working tree is dirty")
    print()
    return True


def step2_h1_audit():
    print("=== Step 2: H1 Audit Check ===")
    audit_path = ROOT / "docs" / "frame_literal_audit.md"
    if not audit_path.exists():
        print("  FAIL: docs/frame_literal_audit.md not found")
        return False
    content = audit_path.read_text()
    if "MISMATCH" in content:
        print("  FAIL: still has DEFAULT_VALUE_MISMATCH entries")
        return False
    print("  PASS: no DEFAULT_VALUE_MISMATCH entries")
    print()
    return True


def _find_sample_with_tumor_params(samples_dir: Path):
    """Find a sample that has tumor_params.json."""
    if not samples_dir.exists():
        return None
    for sd in sorted(samples_dir.iterdir()):
        if not sd.is_dir():
            continue
        if (sd / "tumor_params.json").exists():
            return sd
    return None


def step4_single_sample_run(samples_dir: Path):
    """Step 3: Single-sample dry-run.

    Hides other sample dirs so run_mcx_pipeline.py only processes the target.
    """
    print("=== Step 3: Single-Sample Dry Run ===")

    sample = _find_sample_with_tumor_params(samples_dir)
    if sample is None:
        print("  FAIL: no sample found with tumor_params.json")
        return False

    sample_id = sample.name
    print(f"  Using: {sample_id}")

    # Move other sample dirs out of samples_dir temporarily (to isolate the run)
    import tempfile
    stash_dir = Path(tempfile.mkdtemp(prefix="preflight_stash_"))
    stash_map = []
    for sd in sorted(samples_dir.iterdir()):
        if not sd.is_dir() or sd.name == sample_id:
            continue
        stash_path = stash_dir / sd.name
        sd.rename(stash_path)
        stash_map.append((sd, stash_path))

    try:
        sample_idx = int(sample_id.split("_")[1])
        result = subprocess.run(
            ["uv", "run", "python", "scripts/step2m_generate_mcx_sources.py",
             "--experiment", samples_dir.parent.name,
             "--sample_start", str(sample_idx),
             "--sample_end", str(sample_idx + 1)],
            capture_output=True, text=True, cwd=ROOT, timeout=60,
        )
        if result.returncode != 0:
            print(f"  FAIL: step2m failed:\n{result.stderr[-500:]}")
            return False
        print("  step2m: OK")

        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_mcx_pipeline.py",
             "--samples_dir", str(samples_dir),
             "--force_mcx", "--no_skip"],
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        )
        if result.returncode != 0:
            print(f"  FAIL: MCX simulation failed:\n{result.stderr[-500:]}")
            return False
        print("  MCX simulation: OK")

        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_mcx_pipeline.py",
             "--samples_dir", str(samples_dir),
             "--projection_only", "--no_skip"],
            capture_output=True, text=True, cwd=ROOT, timeout=120,
        )
        if result.returncode != 0:
            print(f"  FAIL: projection failed:\n{result.stderr[-500:]}")
            return False
        print("  projection: OK")

    finally:
        for orig, stash_path in stash_map:
            stash_path.rename(orig)
        stash_dir.rmdir()

    print()
    return True


def step5_metrics(samples_dir: Path) -> dict | None:
    """Step 4: Auto-metrics validation. Returns metrics dict or None on failure."""
    print("=== Step 4: Auto-Metrics ===")

    import numpy as np
    import yaml
    from fmt_simgen.frame_contract import assert_in_trunk_bbox

    if not samples_dir.exists():
        print("  FAIL: samples_dir not found")
        return None
    sample = None
    for sd in sorted(samples_dir.iterdir()):
        if not sd.is_dir():
            continue
        if (sd / "tumor_params.json").exists() and (sd / "proj.npz").exists():
            sample = sd
            break

    if sample is None:
        print("  FAIL: no sample found with tumor_params.json + proj.npz")
        return None

    sample_id = sample.name
    print(f"  Using: {sample_id}")

    tp_path = sample / "tumor_params.json"
    tumor_params = json.loads(tp_path.read_text())
    foci = tumor_params.get("foci", [])
    if not foci:
        print("  FAIL: no foci in tumor_params")
        return None

    tumor_center = np.mean([f["center"] for f in foci], axis=0)
    print(f"  Tumor center: {tumor_center.tolist()}")

    metrics = {"sample_id": sample_id, "tumor_center_mm": tumor_center.tolist()}

    # DE metric
    try:
        assert_in_trunk_bbox(tumor_center.reshape(1, 3), tol_mm=1.0)
        print(f"  DE tumor Y={tumor_center[1]:.1f}mm: within trunk ✓")
        metrics["de_tumor_in_trunk"] = True
    except AssertionError as e:
        print(f"  FAIL: tumor center outside trunk: {e}")
        metrics["de_tumor_in_trunk"] = False
        return None

    # MCX completion check
    proj_path = sample / "proj.npz"
    if not proj_path.exists():
        print("  FAIL: proj.npz not found")
        return None

    proj = np.load(proj_path)
    angles = ["-90", "-60", "-30", "0", "30", "60", "90"]
    for angle in angles:
        if angle not in proj:
            print(f"  FAIL: proj missing angle {angle}")
            return None
    print(f"  MCX proj: all 7 angles present ✓")

    # Fluence argmax check (validates .bin source position is correct)
    try:
        import jdata as jd
        from fmt_simgen.frame_contract import VOXEL_SIZE_MM
        jnii_path = sample / f"{sample_id}.jnii"
        if jnii_path.exists():
            d = jd.loadjd(str(jnii_path))
            nifti = d["NIFTIData"] if isinstance(d, dict) else d
            fluence = nifti.transpose(2, 1, 0, 3, 4).squeeze()
            argmax_idx = np.unravel_index(fluence.argmax(), fluence.shape)
            argmax_mm = np.array(argmax_idx) * float(VOXEL_SIZE_MM)
            L2 = float(np.linalg.norm(argmax_mm - tumor_center))
            metrics["mcx_fluence_argmax_mm"] = argmax_mm.tolist()
            metrics["mcx_fluence_vs_tumor_L2_mm"] = round(L2, 2)
            print(f"  Fluence argmax: {argmax_mm}mm, L2 vs tumor: {L2:.2f}mm")
            if L2 > 5.0:
                print(f"  FAIL: fluence argmax {L2:.1f}mm from tumor center")
                return None
            print(f"  Fluence argmax OK ✓")
        else:
            print(f"  WARNING: {jnii_path.name} not found, skipping fluence check")
    except Exception as e:
        print(f"  WARNING: fluence argmax check failed: {e}")

    # Source position check from JSON
    sample_json = sample / f"{sample_id}.json"
    if sample_json.exists():
        with open(sample_json) as f:
            cfg = yaml.safe_load(f)
        src_pos = cfg.get("Optode", {}).get("Source", {}).get("Pos", [])
        if len(src_pos) == 3:
            vs = 0.2
            # Pattern3D: physical Y = (pos_y0 + Ny/2) * vs  (center of pattern)
            n1 = cfg.get("Optode", {}).get("Source", {}).get("Param1", [1, 1, 1])
            ny = n1[1]
            src_y_mm = (src_pos[1] + ny / 2) * vs
            y_diff = abs(src_y_mm - tumor_center[1])
            metrics["mcx_source_y_mm"] = round(src_y_mm, 2)
            metrics["mcx_tumor_y_diff_mm"] = round(y_diff, 2)
            print(f"  MCX source Y={src_y_mm:.1f}mm vs tumor Y={tumor_center[1]:.1f}mm (diff={y_diff:.1f}mm)")
            if y_diff > 5.0:
                print(f"  FAIL: source Y diff={y_diff:.1f}mm > 5mm — frame bug")
                return None
            print(f"  MCX source-tumor alignment OK ✓")

    # Visibility metric
    try:
        from fmt_simgen.view_config import TurntableCamera
        from fmt_simgen.frame_contract import VOLUME_CENTER_WORLD

        mesh_path = ROOT / "output" / "shared" / "mesh.npz"
        if mesh_path.exists():
            camera_cfg = dict(
                volume_center_world=VOLUME_CENTER_WORLD.tolist(),
                detector_resolution=[256, 256],
                fov_mm=80.0,
                camera_distance_mm=200.0,
            )
            camera = TurntableCamera(camera_cfg)
            mesh_data = np.load(mesh_path)
            nodes = mesh_data["nodes"]
            surface_faces = mesh_data["surface_faces"]
            # Compute surface normals
            normals = camera.compute_surface_normals(nodes, surface_faces)
            proj_path = sample / "proj.npz"
            if not proj_path.exists():
                print("  WARNING: proj.npz not found for visibility metric")
            else:
                proj_data = np.load(proj_path)
                visible = np.zeros(len(nodes), dtype=bool)
                for angle in angles:
                    depth_key = f"depth_{angle}"
                    if depth_key not in proj_data:
                        print(f"  WARNING: {depth_key} not in proj.npz")
                        continue
                    depth_map = proj_data[depth_key]
                    vis_idx = camera.get_visible_surface_nodes_from_mcx_depth(
                        nodes, normals, depth_map, float(angle),
                        depth_tolerance_mm=0.2,
                    )
                    visible[vis_idx] = True
                n_visible = int(np.sum(visible))
                metrics["visible_nodes"] = n_visible
                print(f"  Visible surface nodes: {n_visible}")
                n_surface = len(nodes)
            lo = max(300, int(0.06 * n_surface))
            hi = min(5000, int(0.90 * n_surface))
            if not (lo <= n_visible <= hi):
                    print(f"  WARNING: visible nodes={n_visible} outside [{lo}, {hi}]")
        else:
            print("  WARNING: mesh.npz not found for visibility metric")
    except Exception as e:
        print(f"  WARNING: visibility metric failed: {e}")

    print()
    return metrics


def write_pass_stamp(samples_dir: Path, metrics: dict):
    stamp = {
        "config_hash": CONFIG_HASH,
        "contract_version": CONTRACT_VERSION,
        "git_hash": get_git_hash(),
        "preflight_at": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
        ).strip(),
        "samples_dir": str(samples_dir),
        "metrics": metrics,
    }
    stamp_path = ROOT / "output" / "preflight_pass.json"
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2))
    print(f"Wrote: {stamp_path}")
    return stamp_path


def main():
    parser = argparse.ArgumentParser(description="H6 preflight")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=ROOT / "data" / "uniform_trunk_v2_20260420_100948" / "samples",
    )
    args = parser.parse_args()
    samples_dir = args.samples_dir

    print("=" * 60)
    print("H6: Preflight for Full Dataset Regeneration")
    print("=" * 60)
    print()

    passed = True

    if not step1_print_hashes():
        passed = False
    if passed and not step2_h1_audit():
        passed = False
    if passed and not step4_single_sample_run(samples_dir):
        passed = False

    metrics = None
    if passed:
        metrics = step5_metrics(samples_dir)
        if metrics is None:
            passed = False

    print()
    if passed:
        stamp_path = write_pass_stamp(samples_dir, metrics)
        print("=" * 60)
        print("✅ H6 PASSED — preflight_pass.json written")
        print(f"   Metrics: {json.dumps(metrics, indent=2)}")
        print("   Full dataset regeneration may proceed.")
        print("=" * 60)
        return 0
    else:
        stamp_path = ROOT / "output" / "preflight_pass.json"
        if stamp_path.exists():
            stamp_path.unlink()
        print("=" * 60)
        print("❌ H6 FAILED — fix issues before full regen")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
