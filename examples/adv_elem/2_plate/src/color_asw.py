import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


class TwodimAddSchwarzColored22:
    """
    2D Additive Schwarz smoother with 2x2 colored ordering (4 colors), LU patch solves,
    and optional diagonal nugget stabilization.

    - Patch = (coupled_size x coupled_size) nodes (default 2x2), block_dim DOF per node
    - Coloring is on the PATCH-ANCHOR grid:
        anchor at (ix0,iy0) with ix0 in [0..nx-cs], iy0 in [0..ny-cs]
        color = (ix0 % 2) + 2*(iy0 % 2)  -> 0..3

    Semantics:
      - solve(rhs) applies one smoothing application to (soln=0, defect=rhs).
      - smooth_defect(soln, defect) updates defect in-place: defect -= K @ dsoln
        (true MG smoother behavior).

    Notes:
      - Uses LU factors per patch (like your GPU LU approach).
      - Optional nugget: Kloc += (nugget_rel * ||Kloc||_inf + nugget_abs) * I
        to stabilize near-singular patches (thin limit).
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int,
        coupled_size: int = 2,
        omega: float = 0.7,
        iters: int = 1,
        # nugget stabilization
        nugget_rel: float = 0.0,     # try 1e-12 .. 1e-10 for thin shells
        nugget_abs: float = 0.0,     # optional absolute nugget
        # ordering / behavior
        colored: bool = True,        # 4-color sweeps over 2x2 anchor parity
        rebuild_on_init: bool = True,
    ):
        assert sp.isspmatrix_csr(K) or sp.isspmatrix_bsr(K), "K must be CSR or BSR"
        self.K = K.tocsr()
        self.N = int(self.K.shape[0])

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

        self.nugget_rel = float(nugget_rel)
        self.nugget_abs = float(nugget_abs)

        self.colored = bool(colored)

        # anchor grid (patch bottom-left corners)
        self._nnx = self.nx - (self.coupled_size - 1)
        self._nny = self.ny - (self.coupled_size - 1)
        assert self._nnx > 0 and self._nny > 0

        self._num_patches = self._nnx * self._nny
        self._patch_nodes = self.coupled_size * self.coupled_size
        self._patch_size = self._patch_nodes * self.block_dim

        # caches
        self._patch_dofs = []   # list[np.ndarray] global dof indices for each patch
        self._lu = None         # list[(lu, piv)] per patch

        # 4-color schedule: list of patch-id arrays
        self._color_pids = None

        if rebuild_on_init:
            self.rebuild_patch_factors()

    @classmethod
    def from_assembler(cls, assembler, **kwargs):
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=getattr(assembler, "nny", assembler.nnx),
            block_dim=assembler.dof_per_node,
            **kwargs,
        )

    # -------------------------
    # patch assembly / LU factors
    # -------------------------
    def rebuild_patch_factors(self):
        bs = self.block_dim
        K = self.K

        self._patch_dofs = []
        lu_list = []

        # Helper: dense (bs x bs) node block extraction from dof-level CSR
        def _find_block(i_node: int, j_node: int) -> np.ndarray:
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

        # Build patches in anchor order (ix0 fastest)
        for iy0 in range(self._nny):
            for ix0 in range(self._nnx):
                # patch nodes (lexicographic in local i,j)
                nodes = []
                for j in range(self.coupled_size):
                    for i in range(self.coupled_size):
                        nodes.append((ix0 + i) + self.nx * (iy0 + j))
                nodes = np.asarray(nodes, dtype=int)

                # global dof list for this patch
                dofs = np.concatenate([np.arange(bs * n, bs * (n + 1), dtype=int) for n in nodes])
                self._patch_dofs.append(dofs)

                # assemble dense patch matrix
                Kloc = np.zeros((self._patch_size, self._patch_size), dtype=float)
                for a, na in enumerate(nodes):
                    ia0 = bs * a
                    for b, nb in enumerate(nodes):
                        ib0 = bs * b
                        Kloc[ia0:ia0 + bs, ib0:ib0 + bs] = _find_block(na, nb)

                # optional nugget stabilization
                if self.nugget_rel != 0.0 or self.nugget_abs != 0.0:
                    scale = np.linalg.norm(Kloc, ord=np.inf)
                    tau = self.nugget_abs + self.nugget_rel * scale
                    if tau != 0.0:
                        Kloc = Kloc + (tau + 1e-30) * np.eye(Kloc.shape[0])

                # LU factorization (store factors, solve later)
                lu_list.append(la.lu_factor(Kloc))

        self._lu = lu_list

        # Build color groups (4 colors based on anchor parity)
        if self.colored and self.coupled_size == 2:
            # only meaningful for 2x2 patch anchoring; still works for larger cs but parity logic differs.
            colors = [[] for _ in range(4)]
            pid = 0
            for iy0 in range(self._nny):
                for ix0 in range(self._nnx):
                    c = (ix0 & 1) + 2 * (iy0 & 1)
                    colors[c].append(pid)
                    pid += 1
            self._color_pids = [np.asarray(lst, dtype=int) for lst in colors]
        elif self.colored:
            # fallback: still color by anchor parity even for cs>2 (reasonable heuristic)
            colors = [[] for _ in range(4)]
            pid = 0
            for iy0 in range(self._nny):
                for ix0 in range(self._nnx):
                    c = (ix0 & 1) + 2 * (iy0 & 1)
                    colors[c].append(pid)
                    pid += 1
            self._color_pids = [np.asarray(lst, dtype=int) for lst in colors]
        else:
            self._color_pids = None

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

        if self._lu is None:
            self.rebuild_patch_factors()

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            if self.colored and self._color_pids is not None:
                # 4-color additive: accumulate contributions color-by-color
                # (reduces destructive interference vs fully additive, still parallel per color)
                for pids in self._color_pids:
                    if pids.size == 0:
                        continue

                    dcolor = np.zeros_like(soln)
                    for pid in pids:
                        I = self._patch_dofs[pid]
                        lu_piv = self._lu[pid]

                        dloc = defect[I]
                        uloc = la.lu_solve(lu_piv, dloc)

                        dcolor[I] += self.omega * uloc

                    dsoln += dcolor
                    # update defect incrementally per-color (more GS-like stability)
                    soln += dcolor
                    defect -= self.K.dot(dcolor)

            else:
                # plain additive Schwarz (one global add then one global defect update)
                for pid, I in enumerate(self._patch_dofs):
                    lu_piv = self._lu[pid]
                    dloc = defect[I]
                    uloc = la.lu_solve(lu_piv, dloc)
                    dsoln[I] += self.omega * uloc

                soln += dsoln
                defect -= self.K.dot(dsoln)

        return
