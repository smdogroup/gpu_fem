import numpy as np
import scipy.sparse as sp


def svd_thresholded_inverse(A: np.ndarray, alpha: float = 1e-12) -> np.ndarray:
    """
    Return a dense inverse-like operator for A using SVD singular-value flooring.

    If A = U diag(s) V^T, we replace s_i <- max(s_i, alpha*s_1) then invert.
    This damps/regularizes near-null (often constraint-like) directions on each patch.
    """
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    sigma1 = s[0] if s.size > 0 else 1.0
    s_thresh = np.maximum(s, alpha * sigma1)
    s_inv = 1.0 / s_thresh
    # invA = V diag(s_inv) U^T
    return (VT.T * s_inv) @ U.T


class TwodimSVDAddSchwarz:
    """
    2D Additive Schwarz smoother for node-block matrices.

    Optional feature:
      - per-patch SVD thresholded inverse to "soften" hard constraint directions
        that can make Schwarz overly stiff / thickness-sensitive.

    API:
      - solve(rhs): one smoother application starting from soln=0
      - smooth_defect(soln, defect): in-place MG smoother semantics:
          soln += dsoln
          defect -= K @ dsoln
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int = 2,
        coupled_size: int = 2,
        omega: float = 0.7,
        iters: int = 1,
        # --- new knobs ---
        use_svd_threshold: bool = False,
        svd_alpha: float = 1e-12,
    ):
        assert sp.isspmatrix_csr(K) or sp.isspmatrix_bsr(K), "K must be CSR or BSR"
        self.K = K.tocsr()  # we rely on indptr/indices
        self.N = self.K.shape[0]
        self.block_dim = int(block_dim)
        assert self.N % self.block_dim == 0
        self.nnodes = self.N // self.block_dim

        self.nx = int(nx)
        self.ny = int(ny)
        assert self.nx * self.ny == self.nnodes, (
            f"nx*ny must equal nnodes: {self.nx}*{self.ny} != {self.nnodes}"
        )

        self.coupled_size = int(coupled_size)
        assert self.coupled_size >= 1

        self.omega = float(omega)
        self.iters = int(iters)

        # SVD thresholding options
        self.use_svd_threshold = bool(use_svd_threshold)
        self.svd_alpha = float(svd_alpha)

        # derived
        self._nnx = self.nx - (self.coupled_size - 1)
        self._nny = self.ny - (self.coupled_size - 1)
        assert self._nnx > 0 and self._nny > 0

        self._num_blocks = self._nnx * self._nny
        self._sblock_nodes = self.coupled_size * self.coupled_size
        self._sblock_size = self._sblock_nodes * self.block_dim

        # cache
        self._sch_nodes = []     # list of arrays of node indices, length _num_blocks
        self._sch_dofs = []      # list of arrays of dof indices for fast gather/scatter
        self._invK = None        # list of dense patch inverse operators

        self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(
        cls,
        assembler,
        omega: float = 0.7,
        iters: int = 1,
        coupled_size: int = 2,
        use_svd_threshold: bool = False,
        svd_alpha: float = 1e-12,
    ):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=assembler.nny if hasattr(assembler, "nny") else assembler.nnx,
            block_dim=assembler.dof_per_node,
            coupled_size=coupled_size,
            omega=omega,
            iters=iters,
            use_svd_threshold=use_svd_threshold,
            svd_alpha=svd_alpha,
        )

    # -------------------------
    # patch assembly / inverses
    # -------------------------
    def _find_block(self, i_node: int, j_node: int) -> np.ndarray:
        """
        Return the (bs,bs) dense block K_{i_node,j_node} if present, else zeros.
        Assumes node-block ordering in the global matrix.
        """
        bs = self.block_dim
        K = self.K

        r0 = bs * i_node
        c0 = bs * j_node

        blk = np.zeros((bs, bs), dtype=float)
        for rr in range(bs):
            gr = r0 + rr
            for p in range(K.indptr[gr], K.indptr[gr + 1]):
                gc = K.indices[p]
                if c0 <= gc < c0 + bs:
                    blk[rr, gc - c0] = K.data[p]
        return blk

    def _invert_patch(self, Kloc: np.ndarray) -> np.ndarray:
        """
        Build a dense patch inverse operator.
        - default: exact dense inverse (fastest apply, can be brittle if Kloc is near-singular)
        - SVD threshold: floors singular values to alpha*sigma1 before inverting (more robust)
        """
        if self.use_svd_threshold:
            return svd_thresholded_inverse(Kloc, alpha=self.svd_alpha)
        else:
            return np.linalg.inv(Kloc)

    def rebuild_patch_inverses(self):
        """
        Precompute patch node lists, dof lists, and inverted patch matrices.
        Stores inverses in self._invK as a list of dense arrays.
        """
        bs = self.block_dim

        self._sch_nodes = []
        self._sch_dofs = []
        invKs = []

        for iblock in range(self._num_blocks):
            ix0 = iblock % self._nnx
            iy0 = iblock // self._nnx

            # nodes in this coupled_size x coupled_size patch (lexicographic)
            nodes = []
            for j in range(self.coupled_size):
                for i in range(self.coupled_size):
                    nodes.append((ix0 + i) + self.nx * (iy0 + j))
            nodes = np.array(nodes, dtype=int)
            self._sch_nodes.append(nodes)

            # dof indices for fast gather/scatter
            dofs = np.empty(self._sblock_size, dtype=int)
            for a, n in enumerate(nodes):
                dofs[bs * a : bs * (a + 1)] = np.arange(bs * n, bs * (n + 1), dtype=int)
            self._sch_dofs.append(dofs)

            # assemble dense patch matrix
            Kloc = np.zeros((self._sblock_size, self._sblock_size), dtype=float)
            for a, na in enumerate(nodes):
                ia0 = bs * a
                for b, nb in enumerate(nodes):
                    ib0 = bs * b
                    Kloc[ia0 : ia0 + bs, ib0 : ib0 + bs] = self._find_block(na, nb)

            invKs.append(self._invert_patch(Kloc))

        self._invK = invKs

    # -------------------------
    # smoother / preconditioner API
    # -------------------------
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs)
        assert rhs.shape == (self.N,)
        soln = np.zeros_like(rhs)
        defect = rhs.copy()
        self.smooth_defect(soln, defect)
        return soln

    def smooth_defect(self, soln: np.ndarray, defect: np.ndarray):
        soln = np.asarray(soln)
        defect = np.asarray(defect)
        assert soln.shape == (self.N,)
        assert defect.shape == (self.N,)

        if self._invK is None:
            self.rebuild_patch_inverses()

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            # additive Schwarz: sum of patch corrections computed from current defect
            for pid in range(self._num_blocks):
                invKp = self._invK[pid]
                dofs = self._sch_dofs[pid]

                # gather patch defect
                dloc = defect[dofs]  # shape (patch_dofs,)

                # local solve (or SVD-thresholded inverse apply)
                uloc = invKp @ dloc

                # scatter-add correction
                dsoln[dofs] += self.omega * uloc

            # MG-smoother semantics: update soln and defect incrementally
            soln += dsoln
            defect -= self.K.dot(dsoln)

        return
