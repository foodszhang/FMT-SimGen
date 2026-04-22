"""Tetrahedral mesh generation with adaptive refinement."""

from typing import Dict

from fmt_simgen.mesh.mesh_generator import MeshData, MeshGenerator, MeshQualityReport


def make_mesh_generator(config: Dict):
    """Factory: pick backend via config['mesh_backend'].

    - 'amira'    (recommended): VTK SurfaceNets3D + TetGen. Guarantees
                                shared-node multi-material FEM mesh.
    - 'iso2mesh' (legacy):       iso2mesh cgalmesh path.
    """
    backend = (config.get("mesh_backend") or "iso2mesh").lower()
    if backend == "amira":
        from fmt_simgen.mesh.amira_mesh_generator import AmiraMeshGenerator

        return AmiraMeshGenerator(config)
    if backend == "iso2mesh":
        return MeshGenerator(config)
    raise ValueError(f"Unknown mesh_backend={backend!r}")


__all__ = ["MeshGenerator", "MeshData", "MeshQualityReport", "make_mesh_generator"]
