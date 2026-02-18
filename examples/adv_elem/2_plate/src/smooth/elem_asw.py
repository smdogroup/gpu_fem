import numpy as np
import scipy.sparse as sp

class TwodimElementAddSchwarz:
    """
    2D Additive Schwarz smoother for node-block matrices.

    Change from prior version:
      - For coupled_size=2 (and in general), we create ONE patch PER NODE (nx*ny patches).
      - At each anchor node (ix,iy), we TRY to build a coupled_size x coupled_size patch
        anchored there. If it doesn't fit fully in-bounds, we FALL BACK to a 1-node patch.
        (So boundaries do NOT get rectangular partial patches; they get 1-node subdomains.)
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

        self.omega = float(omega)
        self.iters = int(iters)

        # cache
        self._sch_nodes = []  # list[np.ndarray], variable length per patch
        self._invK = None     # list[np.ndarray], inverse per patch (variable size)

        self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(cls, assembler, omega: float = 0.7, iters: int = 1, coupled_size: int = 2):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=assembler.nny if hasattr(assembler, "nny") else assembler.nnx,
            block_dim=assembler.dof_per_node,
            coupled_size=coupled_size,
            omega=omega,
            iters=iters,
        )

    def rebuild_patch_inverses(self):
        """
        Precompute patch node lists and inverted patch matrices.
        One patch per node (nx*ny). If a full coupled_size x coupled_size patch
        doesn't fit, use a 1-node patch at that anchor.
        """
        bs = self.block_dim
        K = self.K

        self._sch_nodes = []
        invKs = []

        def _find_block(i_node: int, j_node: int):
            """Return (bs,bs) dense block K_{i_node,j_node} if present else zeros."""
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

        # one patch per node (anchor)
        for iy in range(self.ny):
            for ix in range(self.nx):
                anchor = ix + self.nx * iy

                # can we fit a full coupled_size x coupled_size patch anchored at (ix,iy)?
                full_fits = (
                    (ix + self.coupled_size <= self.nx) and
                    (iy + self.coupled_size <= self.ny)
                )

                if full_fits:
                    nodes = []
                    for j in range(self.coupled_size):
                        for i in range(self.coupled_size):
                            nodes.append((ix + i) + self.nx * (iy + j))
                    nodes = np.array(nodes, dtype=int)
                else:
                    # boundary fallback: 1-node subdomain
                    nodes = np.array([anchor], dtype=int)

                self._sch_nodes.append(nodes)

                # assemble dense patch matrix (size depends on patch node count)
                m = len(nodes) * bs
                Kloc = np.zeros((m, m), dtype=float)
                for a, na in enumerate(nodes):
                    ia0 = bs * a
                    for b, nb in enumerate(nodes):
                        ib0 = bs * b
                        Kloc[ia0:ia0 + bs, ib0:ib0 + bs] = _find_block(na, nb)

                # invert
                invKs.append(np.linalg.inv(Kloc))

        self._invK = invKs

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

        bs = self.block_dim

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            for pid, nodes in enumerate(self._sch_nodes):
                invKp = self._invK[pid]

                # gather patch defect
                dloc = np.concatenate([defect[bs*n:bs*(n+1)] for n in nodes])

                # local solve
                uloc = invKp @ dloc

                # scatter-add correction
                for a, n in enumerate(nodes):
                    dsoln[bs*n:bs*(n+1)] += self.omega * uloc[bs*a:bs*(a+1)]

            soln += dsoln
            defect -= self.K.dot(dsoln)

        return
