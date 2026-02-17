import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


class TwodimAddSchwarzColored22_BC:
    """
    2D Schwarz smoother with:
      - 2x2 patches (coupled_size=2)
      - boundary-aware patch anchoring (clamped anchors so boundary nodes get same coverage quality)
      - 4-color sweep on patch anchors
      - LU factorization per patch
      - optional nugget
      - OPTIONAL Dirichlet DOF elimination inside each patch via fixed_dofs mask

    Why this helps near boundaries:
      - boundary nodes participate in a *full set of patches* (via clamped anchors)
      - fixed DOFs do not pollute patch solves (eliminate or pin)
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx: int,
        ny: int,
        block_dim: int,
        omega: float = 0.7,
        iters: int = 1,
        nugget_rel: float = 0.0,
        nugget_abs: float = 0.0,
        colored: bool = True,
        # BC handling:
        fixed_dofs: np.ndarray = None,     # bool array length N (True => Dirichlet/fixed)
        eliminate_fixed: bool = True,             # eliminate fixed from patch solve if True, else pin
        pin_value: float = 1.0,                   # used if eliminate_fixed=False (diag set to pin_value*scale)
    ):
        assert sp.isspmatrix_csr(K) or sp.isspmatrix_bsr(K)
        self.K = K.tocsr()
        self.N = int(self.K.shape[0])

        self.nx = int(nx)
        self.ny = int(ny)
        self.block_dim = int(block_dim)
        assert self.N % self.block_dim == 0
        self.nnodes = self.N // self.block_dim
        assert self.nnodes == self.nx * self.ny

        self.omega = float(omega)
        self.iters = int(iters)

        self.nugget_rel = float(nugget_rel)
        self.nugget_abs = float(nugget_abs)

        self.colored = bool(colored)

        if fixed_dofs is not None:
            fixed_dofs = np.asarray(fixed_dofs, dtype=bool)
            assert fixed_dofs.shape == (self.N,)
        self.fixed_dofs = fixed_dofs
        self.eliminate_fixed = bool(eliminate_fixed)
        self.pin_value = float(pin_value)

        # caches
        self._patch_I = []          # global dof indices for each patch (length 4*bs)
        self._patch_free = []       # indices (local) of free dofs in the patch
        self._lu = []               # LU factors for Kff per patch (or full if no elimination)

        self._color_pids = None     # 4-color lists of patch ids
        self.rebuild()

    @staticmethod
    def fixed_mask_for_plate_w_only(nx: int, ny: int, block_dim: int, clamped: bool = False) -> np.ndarray:
        """
        Convenience helper if you *don't* already have a fixed DOF mask.
        For your RM plate ordering [w, thx, thy] per node:
          simply-supported: w fixed on boundary
          clamped: w, thx, thy fixed on boundary
        """
        assert block_dim in (3, 2, 1)
        N = nx * ny * block_dim
        fixed = np.zeros(N, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                on = (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)
                if not on:
                    continue
                node = i + nx * j
                base = block_dim * node
                if clamped:
                    fixed[base:base + block_dim] = True
                else:
                    fixed[base + 0] = True  # w
        return fixed
    
    @classmethod
    def from_assembler(cls, assembler, **kwargs):
        fixed_dofs = TwodimAddSchwarzColored22_BC.fixed_mask_for_plate_w_only(
            nx=assembler.nnx, ny=assembler.nnx, block_dim=3, clamped=False
        )
        return cls(
            assembler.kmat,
            nx=assembler.nnx,
            ny=getattr(assembler, "nny", assembler.nnx),
            block_dim=assembler.dof_per_node,
            fixed_dofs = fixed_dofs,
            **kwargs,
        )

    def rebuild(self):
        bs = self.block_dim
        K = self.K
        fixed = self.fixed_dofs

        self._patch_I.clear()
        self._patch_free.clear()
        self._lu.clear()

        # Dense (bs x bs) block extraction for node i->j from dof-level CSR
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

        # --- boundary-aware 2x2 patches via clamped anchors ---
        # Make a patch "per node" by choosing the 2x2 anchor that contains that node,
        # clamped to valid anchor range [0..nx-2], [0..ny-2].
        #
        # This gives uniform coverage including boundaries/corners.
        anchors = []
        for j in range(self.ny):
            for i in range(self.nx):
                ax = min(max(i - 1, 0), self.nx - 2)
                ay = min(max(j - 1, 0), self.ny - 2)
                anchors.append((ax, ay))
        # unique anchors, keep stable order
        seen = set()
        uniq = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                uniq.append(a)

        # build patches from anchors
        for (ax, ay) in uniq:
            n00 = (ax + 0) + self.nx * (ay + 0)
            n10 = (ax + 1) + self.nx * (ay + 0)
            n01 = (ax + 0) + self.nx * (ay + 1)
            n11 = (ax + 1) + self.nx * (ay + 1)
            nodes = np.array([n00, n10, n01, n11], dtype=int)  # 2x2

            I = np.concatenate([np.arange(bs*n, bs*(n+1), dtype=int) for n in nodes])
            self._patch_I.append(I)

            # assemble dense patch matrix
            Kloc = np.zeros((4*bs, 4*bs), dtype=float)
            for a, na in enumerate(nodes):
                ia0 = bs * a
                for b, nb in enumerate(nodes):
                    ib0 = bs * b
                    Kloc[ia0:ia0+bs, ib0:ib0+bs] = _find_block(na, nb)

            # nugget
            if self.nugget_rel != 0.0 or self.nugget_abs != 0.0:
                scale = np.linalg.norm(Kloc, ord=np.inf)
                tau = self.nugget_abs + self.nugget_rel * scale
                if tau != 0.0:
                    Kloc = Kloc + (tau + 1e-30) * np.eye(Kloc.shape[0])

            # handle fixed dofs
            if fixed is None:
                free = np.arange(Kloc.shape[0], dtype=int)
                self._patch_free.append(free)
                self._lu.append(la.lu_factor(Kloc))
            else:
                fixed_loc = fixed[I]  # bool length 4*bs
                if self.eliminate_fixed:
                    free = np.where(~fixed_loc)[0]
                    self._patch_free.append(free)
                    if free.size == 0:
                        # fully fixed patch: store None
                        self._lu.append(None)
                    else:
                        Kff = Kloc[np.ix_(free, free)]
                        self._lu.append(la.lu_factor(Kff))
                else:
                    # pin fixed dofs strongly inside patch
                    free = np.arange(Kloc.shape[0], dtype=int)
                    self._patch_free.append(free)
                    scale = np.linalg.norm(Kloc, ord=np.inf) + 1e-30
                    for ii in np.where(fixed_loc)[0]:
                        Kloc[ii, :] = 0.0
                        Kloc[:, ii] = 0.0
                        Kloc[ii, ii] = self.pin_value * scale
                    self._lu.append(la.lu_factor(Kloc))

        # --- 4-color ordering on anchors ---
        # Color by anchor parity. We need each patch's anchor; reuse uniq list order.
        if self.colored:
            colors = [[] for _ in range(4)]
            for pid, (ax, ay) in enumerate(uniq):
                c = (ax & 1) + 2 * (ay & 1)
                colors[c].append(pid)
            self._color_pids = [np.asarray(lst, dtype=int) for lst in colors]
        else:
            self._color_pids = None

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

        for _ in range(self.iters):
            if self.colored and self._color_pids is not None:
                # GS-like per-color updates (much better boundary behavior than fully additive)
                for pids in self._color_pids:
                    if pids.size == 0:
                        continue
                    dcolor = np.zeros_like(soln)

                    for pid in pids:
                        I = self._patch_I[pid]
                        free = self._patch_free[pid]
                        lu = self._lu[pid]
                        if lu is None:
                            continue

                        dloc = defect[I]
                        if self.fixed_dofs is None or (not self.eliminate_fixed):
                            uloc = la.lu_solve(lu, dloc)
                            dcolor[I] += self.omega * uloc
                        else:
                            # eliminate fixed dofs: solve only on free entries, scatter into full patch vector
                            rhs_f = dloc[free]
                            uf = la.lu_solve(lu, rhs_f)
                            uloc = np.zeros_like(dloc)
                            uloc[free] = uf
                            dcolor[I] += self.omega * uloc

                    soln += dcolor
                    defect -= self.K.dot(dcolor)
            else:
                # plain additive
                dsoln = np.zeros_like(soln)
                for pid in range(len(self._patch_I)):
                    I = self._patch_I[pid]
                    free = self._patch_free[pid]
                    lu = self._lu[pid]
                    if lu is None:
                        continue

                    dloc = defect[I]
                    if self.fixed_dofs is None or (not self.eliminate_fixed):
                        uloc = la.lu_solve(lu, dloc)
                        dsoln[I] += self.omega * uloc
                    else:
                        uf = la.lu_solve(lu, dloc[free])
                        uloc = np.zeros_like(dloc)
                        uloc[free] = uf
                        dsoln[I] += self.omega * uloc

                soln += dsoln
                defect -= self.K.dot(dsoln)

        return
