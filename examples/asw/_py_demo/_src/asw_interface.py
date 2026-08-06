import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class TwodimInterfaceAdditiveSchwarz:
    """
    2D interface-overlap additive Schwarz for node-block matrices.

    Partitions are defined by ELEMENT ranges, not node ranges.

    For nx x ny nodes:
        nxe = nx - 1
        nye = ny - 1

    The element grid is split into num_px x num_py rectangular pieces.
    Uneven partitions are allowed.

    Each patch owns the nodes touched by its owned elements, then expands
    by `overlap` nodes. overlap=0 still shares interface nodes naturally
    because adjacent element partitions both touch the interface node line.

    Corrections on shared nodes are weighted by 1 / multiplicity.
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int = 2,
        num_px: int = 2,
        num_py: int = 2,
        overlap: int = 0,
        omega: float = 1.0,
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

        self.nxe = self.nx - 1
        self.nye = self.ny - 1

        self.num_px = int(num_px)
        self.num_py = int(num_py)

        assert self.num_px >= 1
        assert self.num_py >= 1
        assert self.num_px <= self.nxe, "num_px cannot exceed number of x-elements"
        assert self.num_py <= self.nye, "num_py cannot exceed number of y-elements"

        self.overlap = int(overlap)
        assert self.overlap >= 0

        self.omega = float(omega)
        self.iters = int(iters)

        self.num_patches = self.num_px * self.num_py

        self._patch_nodes = []
        self._owned_nodes = []
        self._patch_weights = []
        self._invK = None

        self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(
        cls,
        assembler,
        omega: float = 1.0,
        iters: int = 1,
        num_px: int = 2,
        num_py: int = 2,
        overlap: int = 0,
    ):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=assembler.nny if hasattr(assembler, "nny") else assembler.nnx,
            block_dim=assembler.dof_per_node,
            num_px=num_px,
            num_py=num_py,
            overlap=overlap,
            omega=omega,
            iters=iters,
        )

    def _node_index(self, ix: int, iy: int) -> int:
        return ix + self.nx * iy

    def _split_range(self, nitems: int, nparts: int, ipart: int):
        """
        Split [0, nitems) into possibly uneven contiguous chunks.

        Returns:
            start, end_exclusive
        """
        start = (ipart * nitems) // nparts
        end = ((ipart + 1) * nitems) // nparts
        return start, end

    def _find_block(self, i_node: int, j_node: int):
        """
        Return dense bs x bs block K_{i_node,j_node}.
        Works for true DOF-level CSR.
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

    def _build_patch_node_lists(self):
        """
        Build patch nodes from element partitions.

        Owned element range:
            ex in [ex0, ex1)
            ey in [ey0, ey1)

        Owned nodes touched by those elements:
            ix in [ex0, ex1]
            iy in [ey0, ey1]

        Patch nodes:
            owned node box expanded by `overlap` nodes, clipped to domain.

        Important:
            overlap=0 still gives interface-node sharing between adjacent
            element partitions, because both element partitions touch the
            common node line.
        """
        self._patch_nodes = []
        self._owned_nodes = []

        multiplicity = np.zeros(self.nnodes, dtype=int)

        for py in range(self.num_py):
            ey0, ey1 = self._split_range(self.nye, self.num_py, py)

            for px in range(self.num_px):
                ex0, ex1 = self._split_range(self.nxe, self.num_px, px)

                # owned element block must be nonempty
                assert ex1 > ex0
                assert ey1 > ey0

                # nodes touched by owned elements
                ox0 = ex0
                ox1 = ex1
                oy0 = ey0
                oy1 = ey1

                # overlap-expanded node box
                px0 = max(0, ox0 - self.overlap)
                px1 = min(self.nx - 1, ox1 + self.overlap)

                py0 = max(0, oy0 - self.overlap)
                py1 = min(self.ny - 1, oy1 + self.overlap)

                owned = []
                for iy in range(oy0, oy1 + 1):
                    for ix in range(ox0, ox1 + 1):
                        owned.append(self._node_index(ix, iy))

                patch = []
                for iy in range(py0, py1 + 1):
                    for ix in range(px0, px1 + 1):
                        node = self._node_index(ix, iy)
                        patch.append(node)
                        multiplicity[node] += 1

                self._owned_nodes.append(np.array(owned, dtype=int))
                self._patch_nodes.append(np.array(patch, dtype=int))

        assert np.all(multiplicity > 0)

        self._patch_weights = []
        for nodes in self._patch_nodes:
            weights = np.array([1.0 / multiplicity[n] for n in nodes], dtype=float)
            self._patch_weights.append(weights)

    def rebuild_patch_inverses(self):
        """
        Precompute overlapping patch matrices and dense inverses.

        For real GPU use:
            replace np.linalg.inv(Kloc) with sparse/dense factor handles.
        """
        self._build_patch_node_lists()

        bs = self.block_dim
        invKs = []

        for nodes in self._patch_nodes:
            nloc = len(nodes)
            Kloc = np.zeros((bs * nloc, bs * nloc), dtype=float)

            for a, na in enumerate(nodes):
                for b, nb in enumerate(nodes):
                    blk = self._find_block(na, nb)

                    ia0 = bs * a
                    ib0 = bs * b

                    Kloc[ia0:ia0 + bs, ib0:ib0 + bs] = blk

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

            for pid, nodes in enumerate(self._patch_nodes):
                invKp = self._invK[pid]
                weights = self._patch_weights[pid]

                dloc = np.concatenate(
                    [defect[bs * n:bs * (n + 1)] for n in nodes]
                )

                uloc = invKp @ dloc

                for a, n in enumerate(nodes):
                    w = self.omega * weights[a]
                    dsoln[bs * n:bs * (n + 1)] += (
                        w * uloc[bs * a:bs * (a + 1)]
                    )

            soln += dsoln
            defect -= self.K.dot(dsoln)

        return