import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class OnedimAddSchwarz:
    def __init__(self, K:sp.bsr_matrix, F:np.ndarray, block_dim:int=2, coupled_size:int=2, omega:float=0.7, iters:int=1):
        assert sp.isspmatrix_bsr(K) or sp.isspmatrix_csr(K)

        self.K = K
        self.F = F
        self.N = K.shape[0]
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim

        self.coupled_size = coupled_size
        self.omega = omega
        self.iters = iters # number of times we apply the solver per iteration

    def solve(self, rhs:np.ndarray):
        bs = self.block_dim
        soln = np.zeros_like(rhs)
        defect = rhs.copy()

        for iter in range(self.iters):

            # loop over each subspace of 1D
            for ind in range(self.nnodes - (self.coupled_size - 1)):
                # extract small dense matrix..
                c_dof = self.coupled_size * self.block_dim
                Kc = np.zeros((c_dof, c_dof))
                # extract these blocks into Kc
                for row in range(ind, ind + self.coupled_size):
                    for jp in range(self.K.indptr[row], self.K.indptr[row+1]):
                        j = self.K.indices[jp]
                        if j in list(range(ind, ind + self.coupled_size)):
                            inode = row - ind; jnode = j - ind
                            Kc[bs * inode : bs * (inode + 1), bs * jnode : bs * (jnode+1)] = self.K.data[jp] * 1.0
                
                Fc = defect[bs * ind : bs * (ind + self.coupled_size)].copy()

                uc = np.linalg.solve(Kc, Fc)
                soln[bs * ind : bs * (ind + self.coupled_size)] += self.omega * uc

            # compute new defect
            defect = rhs - self.K.dot(soln)

        return soln

import numpy as np
import scipy.sparse as sp


class TwodimAddSchwarz:
    """
    2D Additive Schwarz smoother for node-block matrices.

    Key properties:
      - solve(rhs) == apply smoother to (soln=0, defect=rhs) once
      - smooth_defect(soln, defect) updates defect in-place via defect -= K @ dsoln
        (this is what makes it a real MG smoother)
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

        # derived
        self._nnx = self.nx - (self.coupled_size - 1)
        self._nny = self.ny - (self.coupled_size - 1)
        assert self._nnx > 0 and self._nny > 0

        self._num_blocks = self._nnx * self._nny
        self._sblock_nodes = self.coupled_size * self.coupled_size
        self._sblock_size = self._sblock_nodes * self.block_dim

        # cache
        self._sch_nodes = []   # list of arrays of node indices, length _num_blocks
        self._invK = None      # list/array of inverted patch matrices

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

    # -------------------------
    # patch assembly / inverses
    # -------------------------
    def rebuild_patch_inverses(self):
        """
        Precompute patch node lists and inverted patch matrices.
        Stores inverses in self._invK as a list of dense arrays.
        """
        bs = self.block_dim
        K = self.K

        self._sch_nodes = []
        invKs = []

        # helper: find block in CSR row of node i to node j
        def _find_block(i_node: int, j_node: int):
            """
            Return the (bs,bs) dense block K_{i_node,j_node} if present, else zeros.
            Assumes node-block ordering in the global matrix.
            """
            # global dof row range for node i_node
            r0 = bs * i_node
            # We search in *node graph* sense: scan CSR rows for the first dof row of node i_node
            # and look for columns in the dof-range of node j_node.
            c0 = bs * j_node

            # Because K is assembled as true dof-level CSR, we need to gather bsxbs by scanning
            # each of the bs rows. This is robust (and still cheap for small bs).
            blk = np.zeros((bs, bs), dtype=float)
            for rr in range(bs):
                gr = r0 + rr
                for p in range(K.indptr[gr], K.indptr[gr + 1]):
                    gc = K.indices[p]
                    # column belongs to node j_node iff c0 <= gc < c0+bs
                    if c0 <= gc < c0 + bs:
                        blk[rr, gc - c0] = K.data[p]
            return blk

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

            # assemble dense patch matrix
            Kloc = np.zeros((self._sblock_size, self._sblock_size), dtype=float)
            for a, na in enumerate(nodes):
                for b, nb in enumerate(nodes):
                    blk = _find_block(na, nb)
                    ia0 = bs * a
                    ib0 = bs * b
                    Kloc[ia0:ia0 + bs, ib0:ib0 + bs] = blk

            # invert (or factor)
            invKs.append(np.linalg.inv(Kloc))

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

        bs = self.block_dim

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            # additive Schwarz: sum of patch corrections computed from current defect
            for pid, nodes in enumerate(self._sch_nodes):
                invKp = self._invK[pid]

                # gather patch defect (stack node dofs)
                # shape: (bs*nodes_in_patch,)
                dloc = np.concatenate([defect[bs*n:bs*(n+1)] for n in nodes])

                # local solve
                uloc = invKp @ dloc

                # scatter-add correction
                for a, n in enumerate(nodes):
                    dsoln[bs*n:bs*(n+1)] += self.omega * uloc[bs*a:bs*(a+1)]

            # update soln and defect INCREMENTALLY (critical for MG smoothing semantics)
            soln += dsoln
            defect -= self.K.dot(dsoln)

        return
