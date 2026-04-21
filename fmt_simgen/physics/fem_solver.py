"""
FEMSolver: Finite Element Method solver for the Diffusion Equation.

Implements:
- System matrix assembly (stiffness K + mass C + Robin boundary B)
- Source mass matrix assembly (F)
- Forward matrix computation: A = M^{-1} * F, restricted to surface nodes
- Surface measurement extraction

Mathematical formulation (referencing MS-GDUN fem_matrices.py):
- Diffusion equation: -∇·(D∇Φ) + μ_a·Φ = S (steady-state)
- Robin BC: D·∂Φ/∂n + Φ/(2·An) = 0 on boundary
- System: M @ Φ = F, where M = K + C + B
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

logger = logging.getLogger(__name__)


class FEMMatrices(NamedTuple):
    """Container for FEM matrices."""

    M: sp.spmatrix
    K: sp.spmatrix
    C: sp.spmatrix
    B: sp.spmatrix
    F: sp.spmatrix
    surface_index: np.ndarray


class FEMSolver:
    """FEM solver for steady-state diffusion equation with Robin BC."""

    ORIGINAL_TO_MERGED_LABEL = {
        0: 0,  # background
        1: 1,  # skin
        2: 2,  # skeleton/bone
        3: 3,  # brain
        4: 3,  # medulla
        5: 3,  # cerebellum
        6: 3,  # olfactory_bulb
        7: 3,  # external_brain
        8: 3,  # striatum
        9: 5,  # heart
        10: 3,  # brain_other
        11: 4,  # muscle
        12: 4,  # fat
        13: 4,  # cartilage
        14: 4,  # tongue
        15: 6,  # stomach
        16: 7,  # spleen
        17: 8,  # pancreas
        18: 9,  # liver
        19: 10,  # kidney
        20: 10,  # adrenal
        21: 11,  # lung
    }

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        surface_faces: np.ndarray,
        tissue_labels: np.ndarray,
        opt_params_manager,
    ):
        """Initialize FEM solver.

        Parameters
        ----------
        nodes : np.ndarray
            Node coordinates [N×3] in mm.
        elements : np.ndarray
            Tetrahedron node indices [M×4] (0-based).
        surface_faces : np.ndarray
            Surface triangle node indices [F×3].
        tissue_labels : np.ndarray
            Tissue label for each element [M].
            These are the ORIGINAL Digimouse labels (1-21), not merged labels.
        opt_params_manager : OpticalParameterManager
            Optical parameter manager.
        """
        self.nodes = nodes
        self.elements = elements
        self.surface_faces = surface_faces
        self.tissue_labels = tissue_labels
        self.opt_manager = opt_params_manager
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]

        self._matrices: Optional[FEMMatrices] = None
        self._forward_matrix: Optional[np.ndarray] = None
        self._surface_index: Optional[np.ndarray] = None
        self._lu: Optional["scipy.sparse.linalg.SuperLU"] = None  # cached LU of M

    def map_labels_to_merged(self) -> np.ndarray:
        """Map original Digimouse labels to merged labels.

        Returns
        -------
        np.ndarray
            Merged tissue labels for each element [M].
        """
        merged_labels = np.array(
            [self.ORIGINAL_TO_MERGED_LABEL.get(l, 0) for l in self.tissue_labels]
        )
        return merged_labels

    def assign_optical_params(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Assign optical parameters to each element.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            (D_array, mu_a_array, An) where:
            - D_array: diffusion coefficient for each element [M]
            - mu_a_array: absorption coefficient for each element [M]
            - An: Robin boundary coefficient
        """
        merged_labels = self.map_labels_to_merged()

        params = self.opt_manager.get_multi_params(merged_labels)
        D_array = params["D"]
        mu_a_array = params["mu_a"]

        R, An = self.opt_manager.compute_ro_and_an(self.opt_manager.n)

        logger.info(
            f"Optical params assigned: D range=[{D_array.min():.4f}, {D_array.max():.4f}]"
        )
        logger.info(f"  mu_a range=[{mu_a_array.min():.4f}, {mu_a_array.max():.4f}]")
        logger.info(f"  An = {An:.4f}")

        return D_array, mu_a_array, An

    @staticmethod
    def _tet_geom(nodes4: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute tet determinant and gradient terms.

        Ported from MS-GDUN fem_matrices.py:_tet_geom.

        Parameters
        ----------
        nodes4 : np.ndarray
            4 node coordinates [4×3].

        Returns
        -------
        Tuple[float, np.ndarray]
            (d0, g) where:
            - d0: 6 * tet volume
            - g: gradient matrix [4×4]
        """
        a = np.vstack([np.ones((1, 4), dtype=np.float64), nodes4.T])
        d0 = abs(np.linalg.det(a))

        av1 = -np.linalg.det(a[np.ix_([0, 2, 3], [1, 2, 3])])
        bv1 = np.linalg.det(a[np.ix_([0, 1, 3], [1, 2, 3])])
        cv1 = -np.linalg.det(a[np.ix_([0, 1, 2], [1, 2, 3])])

        av2 = np.linalg.det(a[np.ix_([0, 2, 3], [0, 2, 3])])
        bv2 = -np.linalg.det(a[np.ix_([0, 1, 3], [0, 2, 3])])
        cv2 = np.linalg.det(a[np.ix_([0, 1, 2], [0, 2, 3])])

        av3 = -np.linalg.det(a[np.ix_([0, 2, 3], [0, 1, 3])])
        bv3 = np.linalg.det(a[np.ix_([0, 1, 3], [0, 1, 3])])
        cv3 = -np.linalg.det(a[np.ix_([0, 1, 2], [0, 1, 3])])

        av4 = np.linalg.det(a[np.ix_([0, 2, 3], [0, 1, 2])])
        bv4 = -np.linalg.det(a[np.ix_([0, 1, 3], [0, 1, 2])])
        cv4 = np.linalg.det(a[np.ix_([0, 1, 2], [0, 1, 2])])

        g = np.array(
            [
                [
                    av1 * av1 + bv1 * bv1 + cv1 * cv1,
                    av1 * av2 + bv1 * bv2 + cv1 * cv2,
                    av1 * av3 + bv1 * bv3 + cv1 * cv3,
                    av1 * av4 + bv1 * bv4 + cv1 * cv4,
                ],
                [
                    av2 * av1 + bv2 * bv1 + cv2 * cv1,
                    av2 * av2 + bv2 * bv2 + cv2 * cv2,
                    av2 * av3 + bv2 * bv3 + cv2 * cv3,
                    av2 * av4 + bv2 * bv4 + cv2 * cv4,
                ],
                [
                    av3 * av1 + bv3 * bv1 + cv3 * cv1,
                    av3 * av2 + bv3 * bv2 + cv3 * cv2,
                    av3 * av3 + bv3 * bv3 + cv3 * cv3,
                    av3 * av4 + bv3 * bv4 + cv3 * cv4,
                ],
                [
                    av4 * av1 + bv4 * bv1 + cv4 * cv1,
                    av4 * av2 + bv4 * bv2 + cv4 * cv2,
                    av4 * av3 + bv4 * bv3 + cv4 * cv3,
                    av4 * av4 + bv4 * bv4 + cv4 * cv4,
                ],
            ],
            dtype=np.float64,
        )
        return d0, g

    def assemble_stiffness(self, D: np.ndarray) -> sp.spmatrix:
        """Assemble global stiffness matrix K.

        Parameters
        ----------
        D : np.ndarray
            Diffusion coefficient for each element [M].

        Returns
        -------
        sp.spmatrix
            Stiffness matrix [N×N].
        """
        n = self.n_nodes
        row_inds = []
        col_inds = []
        data = []

        for e in range(self.n_elements):
            tet = self.elements[e]
            pts = self.nodes[tet]
            d0, g = self._tet_geom(pts)

            if d0 <= 1e-15:
                continue

            k_local = (D[e] / (6.0 * d0)) * g

            for i in range(4):
                for j in range(4):
                    row_inds.append(tet[i])
                    col_inds.append(tet[j])
                    data.append(k_local[i, j])

        K = sp.csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
        logger.info(f"Stiffness matrix K assembled: {K.nnz} non-zeros")
        return K

    def assemble_mass(self, D: np.ndarray, mu_a: np.ndarray) -> sp.spmatrix:
        """Assemble global mass matrix C (absorption weighted).

        Parameters
        ----------
        D : np.ndarray
            Diffusion coefficient for each element [M] (not used here, kept for API).
        mu_a : np.ndarray
            Absorption coefficient for each element [M].

        Returns
        -------
        sp.spmatrix
            Mass matrix [N×N].
        """
        n = self.n_nodes
        row_inds = []
        col_inds = []
        data = []

        for e in range(self.n_elements):
            tet = self.elements[e]
            pts = self.nodes[tet]
            d0, _ = self._tet_geom(pts)

            if d0 <= 1e-15:
                continue

            c_local = (d0 / 120.0) * np.ones((4, 4), dtype=np.float64)
            c_local += np.diag(np.diag(c_local))
            c_local = mu_a[e] * c_local

            for i in range(4):
                for j in range(4):
                    row_inds.append(tet[i])
                    col_inds.append(tet[j])
                    data.append(c_local[i, j])

        C = sp.csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
        logger.info(f"Mass matrix C assembled: {C.nnz} non-zeros")
        return C

    def assemble_source_mass(self) -> sp.spmatrix:
        """Assemble source mass matrix F (not absorption weighted).

        Returns
        -------
        sp.spmatrix
            Source mass matrix [N×N].
        """
        n = self.n_nodes
        row_inds = []
        col_inds = []
        data = []

        for e in range(self.n_elements):
            tet = self.elements[e]
            pts = self.nodes[tet]
            d0, _ = self._tet_geom(pts)

            if d0 <= 1e-15:
                continue

            f_local = (d0 / 120.0) * np.ones((4, 4), dtype=np.float64)
            f_local += np.diag(np.diag(f_local))

            for i in range(4):
                for j in range(4):
                    row_inds.append(tet[i])
                    col_inds.append(tet[j])
                    data.append(f_local[i, j])

        F = sp.csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
        logger.info(f"Source mass matrix F assembled: {F.nnz} non-zeros")
        return F

    def assemble_boundary(self, An: float) -> sp.spmatrix:
        """Assemble Robin boundary matrix B.

        Parameters
        ----------
        An : float
            Robin boundary coefficient.

        Returns
        -------
        sp.spmatrix
            Boundary matrix [N×N].
        """
        n = self.n_nodes
        row_inds = []
        col_inds = []
        data = []

        for f in range(self.surface_faces.shape[0]):
            tri = self.surface_faces[f]
            x = self.nodes[tri]

            ax = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [x[0, 0], x[1, 0], x[2, 0]],
                    [x[0, 1], x[1, 1], x[2, 1]],
                ],
                dtype=np.float64,
            )
            ay = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [x[0, 0], x[1, 0], x[2, 0]],
                    [x[0, 2], x[1, 2], x[2, 2]],
                ],
                dtype=np.float64,
            )
            az = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [x[0, 1], x[1, 1], x[2, 1]],
                    [x[0, 2], x[1, 2], x[2, 2]],
                ],
                dtype=np.float64,
            )
            d0 = float(
                np.sqrt(
                    np.linalg.det(ax) ** 2
                    + np.linalg.det(ay) ** 2
                    + np.linalg.det(az) ** 2
                )
            )

            b_local = (d0 / (48.0 * An)) * np.ones((3, 3), dtype=np.float64)
            b_local += np.diag(np.diag(b_local))

            for i in range(3):
                for j in range(3):
                    row_inds.append(tri[i])
                    col_inds.append(tri[j])
                    data.append(b_local[i, j])

        B = sp.csr_matrix((data, (row_inds, col_inds)), shape=(n, n), dtype=np.float64)
        logger.info(f"Boundary matrix B assembled: {B.nnz} non-zeros")
        return B

    def assemble_system_matrix(self) -> FEMMatrices:
        """Assemble all FEM matrices.

        Returns
        -------
        FEMMatrices
            Named tuple containing M, K, C, B, F matrices and surface_index.
        """
        D, mu_a, An = self.assign_optical_params()

        logger.info("Assembling stiffness matrix K...")
        K = self.assemble_stiffness(D)

        logger.info("Assembling mass matrix C...")
        C = self.assemble_mass(D, mu_a)

        logger.info("Assembling source mass matrix F...")
        F = self.assemble_source_mass()

        logger.info("Assembling boundary matrix B...")
        B = self.assemble_boundary(An)

        M = K + C + B

        unique_surface_nodes = np.unique(self.surface_faces)
        surface_index = unique_surface_nodes.astype(np.int32)

        self._matrices = FEMMatrices(
            M=M, K=K, C=C, B=B, F=F, surface_index=surface_index
        )
        self._surface_index = surface_index

        logger.info(f"System matrix M assembled: {M.nnz} non-zeros")
        logger.info(f"Sparsity: {M.nnz / (self.n_nodes * self.n_nodes) * 100:.4f}%")

        return self._matrices

    def compute_forward_matrix(self) -> np.ndarray:
        """Compute forward matrix A = M^{-1} * F, restricted to surface nodes.

        This gives the surface fluence when unit source is at each node.

        Returns
        -------
        np.ndarray
            Forward matrix [S×N] where S = surface nodes, N = total nodes.
        """
        if self._matrices is None:
            self.assemble_system_matrix()

        M = self._matrices.M
        F = self._matrices.F
        surface_index = self._surface_index

        n = self.n_nodes
        n_surf = len(surface_index)

        logger.info(
            f"Computing forward matrix: M={M.shape}, F={F.shape}, n_surf={n_surf}"
        )

        logger.info("Performing LU decomposition of M...")
        lu = splu(M.tocsc())

        A_cols = []
        chunk_size = 500

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            logger.info(f"  Solving columns {start}:{end}...")

            for j in range(start, end):
                f_j = F[:, j].toarray().ravel()
                x_j = lu.solve(f_j)
                A_cols.append(x_j[surface_index])

        A_surface = np.array(A_cols).T

        logger.info(f"Forward matrix computed: {A_surface.shape}")
        self._forward_matrix = A_surface

        return A_surface

    def forward(self, d_gt_nodes: np.ndarray) -> np.ndarray:
        """Compute forward measurement: b = A @ d_gt.

        Uses sparse matvec + 1 LU solve (not dense A matrix).
        LU factorization is cached after first call.

        Parameters
        ----------
        d_gt_nodes : np.ndarray
            Ground truth fluorescence at nodes [N].

        Returns
        -------
        np.ndarray
            Forward measurement at surface nodes [S].
        """
        if self._matrices is None:
            self.assemble_system_matrix()

        if self._lu is None:
            import time
            t0 = time.time()
            self._lu = splu(self._matrices.M.tocsc())
            logger.info(
                f"LU factored once: {time.time()-t0:.1f}s, "
                f"nnz_L+U={self._lu.L.nnz + self._lu.U.nnz:,}"
            )

        f = self._matrices.F @ d_gt_nodes          # sparse matvec, ms
        x = self._lu.solve(f)                       # 1 solve
        return np.maximum(x[self._surface_index], 0.0)

    def validate(self, A: Optional[np.ndarray] = None) -> Dict:
        """Validate forward matrix by placing a point source.

        Parameters
        ----------
        A : np.ndarray, optional
            Forward matrix. If None, computes it.

        Returns
        -------
        Dict
            Validation results including max location and value.
        """
        if A is None:
            A = self.compute_forward_matrix()

        n_surf = A.shape[0]

        center_node = np.argmin(
            np.sum((self.nodes - self.nodes.mean(axis=0)) ** 2, axis=1)
        )

        d_point = np.zeros(self.n_nodes)
        d_point[center_node] = 1.0

        b_point = A @ d_point
        b_point = np.maximum(b_point, 0.0)

        max_idx = np.argmax(b_point)
        max_val = b_point[max_idx]

        surface_node_coords = self.nodes[self._surface_index]
        max_coord = surface_node_coords[max_idx]

        results = {
            "source_node": center_node,
            "source_coord": self.nodes[center_node],
            "max_response_node": max_idx,
            "max_response_coord": max_coord,
            "max_response_value": float(max_val),
            "total_surface_nodes": n_surf,
            "A_shape": A.shape,
        }

        logger.info("Validation results:")
        logger.info(f"  Source node: {center_node} at {self.nodes[center_node]}")
        logger.info(f"  Max response at surface node {max_idx}: value={max_val:.6f}")
        logger.info(f"  Max response location: {max_coord}")

        return results

    def get_surface_measurement_matrix(self) -> sp.spmatrix:
        """Get surface measurement matrix that extracts surface node values.

        Returns
        -------
        sp.spmatrix
            Sparse matrix [S×N] that extracts surface measurements.
        """
        if self._surface_index is None:
            if self._matrices is None:
                self.assemble_system_matrix()

        n_surf = len(self._surface_index)
        rows = np.arange(n_surf)
        cols = self._surface_index
        data = np.ones(n_surf)

        return sp.csr_matrix((data, (rows, cols)), shape=(n_surf, self.n_nodes))

    def get_surface_nodes(self) -> np.ndarray:
        """Get coordinates of surface nodes.

        Returns
        -------
        np.ndarray
            Surface node coordinates [S×3].
        """
        if self._surface_index is None:
            if self._matrices is None:
                self.assemble_system_matrix()
        return self.nodes[self._surface_index]

    def save_system_matrix(self, filepath: str) -> Path:
        """Save system matrix to .npz file.

        Parameters
        ----------
        filepath : str
            Output file path (without extension).

        Returns
        -------
        Path
            Path to saved file.
        """
        if self._matrices is None:
            self.assemble_system_matrix()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        sp.save_npz(str(filepath.with_suffix(".M.npz")), self._matrices.M)
        sp.save_npz(str(filepath.with_suffix(".K.npz")), self._matrices.K)
        sp.save_npz(str(filepath.with_suffix(".C.npz")), self._matrices.C)
        sp.save_npz(str(filepath.with_suffix(".B.npz")), self._matrices.B)
        sp.save_npz(str(filepath.with_suffix(".F.npz")), self._matrices.F)

        np.savez(
            filepath.with_suffix(".index.npz"),
            surface_index=self._surface_index,
            nodes=self.nodes,
            elements=self.elements,
            tissue_labels=self.tissue_labels,
        )

        if self._forward_matrix is not None:
            np.savez(
                filepath.with_suffix(".A.npz"),
                forward_matrix=self._forward_matrix,
            )

        logger.info(f"System matrix saved to: {filepath}.*")
        return filepath

    def load_system_matrix(self, filepath: str) -> "FEMSolver":
        """Load system matrix from .npz file.

        Parameters
        ----------
        filepath : str
            Path to system matrix file (without extension).

        Returns
        -------
        FEMSolver
            Self for method chaining.
        """
        filepath = Path(filepath)

        M = sp.load_npz(str(filepath.with_suffix(".M.npz")))
        K = sp.load_npz(str(filepath.with_suffix(".K.npz")))
        C = sp.load_npz(str(filepath.with_suffix(".C.npz")))
        B = sp.load_npz(str(filepath.with_suffix(".B.npz")))
        F = sp.load_npz(str(filepath.with_suffix(".F.npz")))

        index_data = np.load(str(filepath.with_suffix(".index.npz")))
        surface_index = index_data["surface_index"]
        self.nodes = index_data["nodes"]
        self.elements = index_data["elements"]
        self.tissue_labels = index_data["tissue_labels"]
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

        self._matrices = FEMMatrices(
            M=M, K=K, C=C, B=B, F=F, surface_index=surface_index
        )
        self._surface_index = surface_index

        A_file = filepath.with_suffix(".A.npz")
        if A_file.exists():
            A_data = np.load(A_file)
            if "forward_matrix" in A_data:
                self._forward_matrix = A_data["forward_matrix"]
            elif "indices" in A_data and "indptr" in A_data:
                # CSR sparse format: reconstruct the forward matrix
                from scipy.sparse import csr_matrix

                self._forward_matrix = csr_matrix(
                    (A_data["data"], A_data["indices"], A_data["indptr"]),
                    shape=tuple(A_data["shape"]),
                )
                logger.info(
                    f"  Forward matrix loaded as CSR: {self._forward_matrix.shape}"
                )

        logger.info(f"System matrix loaded from: {filepath}.*")
        return self
