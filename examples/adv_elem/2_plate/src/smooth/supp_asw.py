import numpy as np
import scipy.sparse as sp


class TwodimSupportAddSchwarz:
    """
    2D "support ASW" smoother:
      - one patch per node i
      - patch = support(w_i) in a structured Q1 grid sense:
          nodes with |dx| <= r and |dy| <= r around i,
        where r = coupled_size//2
      - interior gives (2r+1)x(2r+1) = coupled_size x coupled_size patches
      - edge/corner nodes automatically get smaller patches

    Notes:
      * This matches your "loop over all nodes (not just interior)" request.
      * For Q1 4-node elements, coupled_size=3 corresponds to the natural
        3x3 node support in the interior.

    Performance:
      - Precomputes dof lists and dense patch inverses once.
      - Applies additive Schwarz updates with MG-smoother semantics.

    Assumptions:
      - Global matrix is dof-level CSR/BSR with node-block ordering.
      - Nodes are laid out lexicographically: node = ix + nx*iy.
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int = 2,
        coupled_size: int = 3,
        omega: float = 0.7,
        iters: int = 1,
        drop_tol: float = 0.0,
    ):
        assert sp.isspmatrix_csr(K) or sp.isspmatrix_bsr(K), "K must be CSR or BSR"
        self.K = K.tocsr()
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
        # natural "support radius" in nodes
        self.r = self.coupled_size // 2  # coupled_size=3 -> r=1
        assert self.r >= 0

        self.omega = float(omega)
        self.iters = int(iters)

        # Optional: if you want to damp very-weak couplings when assembling Kloc
        # (sometimes helps if K has tiny fill due to constraints/penalties)
        self.drop_tol = float(drop_tol)

        # caches
        self._patch_nodes: list[np.ndarray] = []
        self._patch_dofs: list[np.ndarray] = []
        self._invK: list[np.ndarray] = []

        self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(
        cls,
        assembler,
        omega: float = 0.7,
        iters: int = 1,
        coupled_size: int = 3,
        drop_tol: float = 0.0,
    ):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=assembler.nny if hasattr(assembler, "nny") else assembler.nnx,
            block_dim=assembler.dof_per_node,
            coupled_size=coupled_size,
            omega=omega,
            iters=iters,
            drop_tol=drop_tol,
        )

    # -------------------------
    # helpers
    # -------------------------
    def _node_id(self, ix: int, iy: int) -> int:
        return ix + self.nx * iy

    def _find_block(self, i_node: int, j_node: int) -> np.ndarray:
        """
        Return the (bs,bs) dense block K_{i_node,j_node} if present, else zeros.
        Works on dof-level CSR (robust for small bs).
        """
        bs = self.block_dim
        K = self.K
        r0 = bs * i_node
        c0 = bs * j_node

        blk = np.zeros((bs, bs), dtype=float)
        for rr in range(bs):
            gr = r0 + rr
            row_start = K.indptr[gr]
            row_end = K.indptr[gr + 1]
            for p in range(row_start, row_end):
                gc = K.indices[p]
                if c0 <= gc < c0 + bs:
                    val = K.data[p]
                    if self.drop_tol > 0.0 and abs(val) < self.drop_tol:
                        continue
                    blk[rr, gc - c0] = val
        return blk

    def _patch_nodes_for_center(self, center_node: int) -> np.ndarray:
        """
        Return node indices in the structured support patch around center_node.
        Edge/corner patches are smaller automatically.
        """
        ix = center_node % self.nx
        iy = center_node // self.nx

        xs = range(max(0, ix - self.r), min(self.nx - 1, ix + self.r) + 1)
        ys = range(max(0, iy - self.r), min(self.ny - 1, iy + self.r) + 1)

        nodes = [self._node_id(x, y) for y in ys for x in xs]  # lexicographic
        return np.array(nodes, dtype=int)

    # -------------------------
    # patch assembly / inverses
    # -------------------------
    def rebuild_patch_inverses(self):
        """
        Build:
          - one patch per node
          - dense Kloc for that patch
          - store inv(Kloc) for apply
        """
        bs = self.block_dim
        self._patch_nodes = []
        self._patch_dofs = []
        self._invK = []

        for center in range(self.nnodes):
            nodes = self._patch_nodes_for_center(center)
            self._patch_nodes.append(nodes)

            # dof list for quick gather/scatter
            dofs = np.empty(bs * len(nodes), dtype=int)
            for a, n in enumerate(nodes):
                dofs[bs * a : bs * (a + 1)] = np.arange(bs * n, bs * (n + 1), dtype=int)
            self._patch_dofs.append(dofs)

            # assemble dense patch matrix
            m = bs * len(nodes)
            Kloc = np.zeros((m, m), dtype=float)
            for a, na in enumerate(nodes):
                ia0 = bs * a
                for b, nb in enumerate(nodes):
                    ib0 = bs * b
                    Kloc[ia0 : ia0 + bs, ib0 : ib0 + bs] = self._find_block(na, nb)

            # invert; if singular-ish, fall back to a tiny Tikhonov
            try:
                invK = np.linalg.inv(Kloc)
            except np.linalg.LinAlgError:
                # minimal regularization; keeps code from crashing on constrained patches
                reg = 1e-14 * np.linalg.norm(Kloc, ord=2) if np.any(Kloc) else 1e-14
                invK = np.linalg.inv(Kloc + reg * np.eye(m))
            self._invK.append(invK)

    # -------------------------
    # smoother API
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

        if not self._invK:
            self.rebuild_patch_inverses()

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            # one patch per node
            for pid in range(self.nnodes):
                invKp = self._invK[pid]
                dofs = self._patch_dofs[pid]

                dloc = defect[dofs]
                uloc = invKp @ dloc

                dsoln[dofs] += self.omega * uloc

            # MG semantics
            soln += dsoln
            defect -= self.K.dot(dsoln)

        return
