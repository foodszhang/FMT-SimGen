#!/usr/bin/env python3
"""
H4 backfill: walk existing output directories, write provenance sidecars.

For each artifact (trunk_volume.npz, mesh.npz, etc.) in output/shared/,
writes a .provenance.json sidecar with the CURRENT config_hash.

Also migrates frame_manifest.json to include config_hash and contract_version.

Usage:
    uv run python scripts/backfill_provenance.py [--dry-run]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/foods/pro/FMT-SimGen")

# Ensure fmt_simgen is importable regardless of CWD
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def backfill_sharedArtifacts(dry_run: bool = False) -> None:
    """Backfill provenance for output/shared/ artifacts."""
    shared_dir = ROOT / "output" / "shared"
    if not shared_dir.exists():
        print("No output/shared/ directory found, skipping")
        return

    from fmt_simgen.config import CONFIG_HASH, CONTRACT_VERSION
    from fmt_simgen.io import write_provenance, migrate_manifest

    git_hash = "unknown"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    # Artifacts in output/shared/ that get provenance sidecars
    artifact_configs = [
        ("trunk_volume.npz", "scripts/step0a_build_trunk_canonical.py"),
        ("mcx_volume_trunk.bin", "scripts/step0f_mcx_volume.py"),
        ("mesh.npz", "scripts/step0b_generate_mesh.py"),
    ]

    written = []
    skipped = []

    for artifact_name, generator in artifact_configs:
        artifact_path = shared_dir / artifact_name
        if not artifact_path.exists():
            skipped.append(f"{artifact_name} (not found)")
            continue

        prov_path = Path(f"{artifact_path}.provenance.json")
        if prov_path.exists():
            existing = json.loads(prov_path.read_text())
            if existing.get("config_hash") == CONFIG_HASH:
                skipped.append(f"{artifact_name} (already current)")
                continue
            else:
                print(f"  {artifact_name}: stale provenance, will overwrite")

        if dry_run:
            print(f"  [dry-run] would write provenance for {artifact_name}")
        else:
            write_provenance(artifact_path, generator)
            print(f"  wrote: {prov_path.name}")
        written.append(artifact_name)

    # Migrate frame_manifest.json
    manifest_path = shared_dir / "frame_manifest.json"
    if manifest_path.exists():
        migrated = migrate_manifest(manifest_path)
        if dry_run:
            if migrated.get("config_hash") != json.loads(manifest_path.read_text()).get("config_hash"):
                print(f"  [dry-run] would migrate frame_manifest.json")
            else:
                print(f"  frame_manifest.json already current, no change needed")
        else:
            if migrated.get("config_hash") != json.loads(manifest_path.read_text()).get("config_hash"):
                manifest_path.write_text(json.dumps(migrated, indent=2))
                print(f"  migrated: frame_manifest.json")
    else:
        print("  frame_manifest.json not found, skipping")

    return written, skipped


def backfill_sampleArtifacts(dry_run: bool = False) -> None:
    """Backfill provenance for per-sample artifacts (tumor_params.json sources)."""
    samples_root = ROOT / "data"
    if not samples_root.exists():
        print("No data/ directory, skipping sample backfill")
        return

    from fmt_simgen.io import write_provenance

    written = 0
    for exp_dir in sorted(samples_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        samples_dir = exp_dir / "samples"
        if not samples_dir.is_dir():
            continue

        for sample_dir in sorted(samples_dir.iterdir())[:5]:  # first 5 per experiment
            if not sample_dir.is_dir():
                continue
            tumor_params = sample_dir / "tumor_params.json"
            if not tumor_params.exists():
                continue

            prov_path = Path(f"{tumor_params}.provenance.json")
            if prov_path.exists():
                continue

            if dry_run:
                print(f"  [dry-run] would write provenance for {tumor_params}")
            else:
                write_provenance(tumor_params, "scripts/02_generate_dataset.py (pre-H4 sample)")
                print(f"  wrote: {prov_path.name} for {sample_dir.name}")
            written += 1

    if written > 0 and not dry_run:
        print(f"  wrote {written} sample provenance files")
    elif written == 0 and not dry_run:
        print("  no new sample provenance files needed")


def main():
    from fmt_simgen.config import CONFIG_HASH

    parser = argparse.ArgumentParser(description="H4 provenance backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    print("=== H4 Provenance Backfill ===")
    print(f"Current CONFIG_HASH: {CONFIG_HASH}")
    print()

    print("[1/2] output/shared/ artifacts:")
    w, s = backfill_sharedArtifacts(dry_run=args.dry_run)
    print(f"  written: {len(w)}, skipped: {len(s)}")

    print()
    print("[2/2] Per-sample artifacts (first 5 per experiment):")
    backfill_sampleArtifacts(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run complete — rerun without --dry-run to apply]")


if __name__ == "__main__":
    main()
