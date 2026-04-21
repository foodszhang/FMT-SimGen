"""
Provenance tracking for fmt_simgen artifacts (H4).

Every generated artifact can carry a provenance sidecar (.provenance.json)
that records the config_hash, contract_version, git_hash, generator script,
and input hashes at the time of generation.

This enables StaleCacheError to be raised when artifacts generated with
old constants are accidentally loaded — preventing silent frame-bug propagation.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Re-export the exception for callers
from fmt_simgen.config import CONFIG_HASH, CONTRACT_VERSION


class StaleCacheError(RuntimeError):
    """Raised when loading an artifact whose config_hash != current CONFIG_HASH."""
    pass


def get_git_hash() -> str:
    """Get the current git commit hash (short form)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_of_file(path: Path) -> str:
    """Return sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_provenance(
    artifact_path: Path,
    generator: str,
    inputs: Optional[dict[str, str]] = None,
) -> Path:
    """Write a provenance sidecar for an artifact.

    Parameters
    ----------
    artifact_path : Path
        Path to the artifact (e.g. output/shared/trunk_volume.npz)
    generator : str
        Script that generated this artifact (e.g. "scripts/step0a_build_trunk_canonical.py")
    inputs : dict[str, str], optional
        Mapping of input name → sha256 of input file

    Returns
    -------
    Path
        Path to the written sidecar file
    """
    prov: dict = {
        "config_hash": CONFIG_HASH,
        "contract_version": CONTRACT_VERSION,
        "git_hash": get_git_hash(),
        "generator": generator,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs or {},
    }
    prov_path = Path(f"{artifact_path}.provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov_path


def load_with_provenance(artifact_path: Path):
    """Load an artifact, verifying config_hash matches current CONFIG_HASH.

    Parameters
    ----------
    artifact_path : Path
        Path to the artifact (.npz, .bin, etc.)

    Returns
    -------
    The loaded artifact (via np.load for .npz, or raw bytes for .bin)

    Raises
    ------
    StaleCacheError
        If the artifact's provenance config_hash != current CONFIG_HASH
    """
    import numpy as np

    prov_path = Path(f"{artifact_path}.provenance.json")
    if not prov_path.exists():
        raise StaleCacheError(
            f"No provenance sidecar for {artifact_path}. "
            f"Artifact may be pre-H4 and must be regenerated."
        )

    prov = json.loads(prov_path.read_text())
    prov_hash = prov.get("config_hash", "")

    if prov_hash != CONFIG_HASH:
        raise StaleCacheError(
            f"{artifact_path} was generated with config_hash={prov_hash} "
            f"({prov.get('contract_version', '?')}, {prov.get('generated_at', '?')}). "
            f"Current CONFIG_HASH={CONFIG_HASH}. "
            f"Regenerate with: {prov.get('generator', 'unknown script')}"
        )

    # Load the actual artifact
    suffix = Path(artifact_path).suffix.lower()
    if suffix == ".npz":
        return np.load(artifact_path)
    elif suffix == ".bin":
        return Path(artifact_path).read_bytes()
    else:
        raise NotImplementedError(f"load_with_provenance: unsupported suffix {suffix}")


def migrate_manifest(manifest_path: Path) -> dict:
    """Migrate frame_manifest.json to include config_hash and contract_version.

    If the manifest already has a config_hash, leave it unchanged.
    If not, add config_hash and contract_version fields.

    Parameters
    ----------
    manifest_path : Path
        Path to frame_manifest.json

    Returns
    -------
    dict
        The (possibly updated) manifest dict
    """
    import copy
    from fmt_simgen.config import CONFIG_HASH, CONTRACT_VERSION

    manifest = json.loads(manifest_path.read_text())

    # Only migrate if frame_contract_version is old (pre-H4)
    old_version = manifest.get("frame_contract_version", "")
    needs_migration = (
        "config_hash" not in manifest
        or manifest.get("config_hash") != CONFIG_HASH
    )

    if needs_migration:
        migrated = copy.deepcopy(manifest)
        migrated["config_hash"] = CONFIG_HASH
        migrated["contract_version"] = CONTRACT_VERSION
        if "migrated_from" not in migrated:
            migrated["migrated_from"] = old_version or "pre-H4"
        return migrated
    return manifest
