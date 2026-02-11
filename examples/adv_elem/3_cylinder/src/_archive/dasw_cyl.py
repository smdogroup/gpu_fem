import numpy as np
import scipy.sparse as sp


class TwoDimAddSchwarzDeRhamCylinderVertexEdges:
    """
    2D additive Schwarz smoother for De Rham IGA CYLINDER block system:

        K = sp.bmat([
            [Kww,   Kwu,   Kwv,   Kwthx,   Kwthy],
            [Kuw,   Kuu,   Kuv,   Kuthx,   Kuthy],
            [Kvw,   Kvu,   Kvv,   Kvthx,   Kvthy],
            [Kthxw, Kthxu, Kthxv, Kthxthx, Kthxthy],
            [Kthyw, Kthyu, Kthyv, Kthythx, Kthythy],
        ], format="csr")

    Global ordering assumed:
        u = [ w(0..nw-1), u(0..nu-1), v(0..nv-1), thx(0..nthx-1), thy(0..nthy-1) ]

    De Rham grids (match your cylinder assembler):
        w   grid: (nx_w,   ny_w)    with nx_w   = nxe+2, ny_w   = nye+2
        u   grid: (nx_thy, ny_thy)  with nx_thy = nxe+1, ny_thy = nye+2   (same as thy)
        v   grid: (nx_thx, ny_thx)  with nx_thx = nxe+2, ny_thx = nye+1   (same as thx)
        thx grid: (nx_thx, ny_thx)
        thy grid: (nx_thy, ny_thy)

    Patch type
    ----------
    - "vertex_edges" (default): patch anchored at each w-vertex (iw,jw):
        { w(iw,jw),
          u(iw,jw), u(iw, jw-1),
          v(iw,jw), v(iw-1, jw),
          thx(iw,jw), thx(iw-1, jw),
          thy(iw,jw), thy(iw, jw-1) }

      All with bounds checks (and your K already includes BC rows/cols as identity where constrained).

    Notes
    -----
    - Additive Schwarz: dsoln += omega * inv(K_I) * defect_I for each patch I.
    - Use as smoother (iters=1..3). omega ~ 0.6-0.9 typical.
    - Call rebuild_patch_inverses() whenever K changes (new assembly / BC changes / nonlinear update).
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx_w: int, ny_w: int,
        nx_thy: int, ny_thy: int,
        nx_thx: int, ny_thx: int,
        omega: float = 0.7,
        iters: int = 1,
        build_inverses: bool = True,
        patch_type: str = "vertex_edges",
        use_pinv_fallback: bool = True,
    ):
        assert sp.isspmatrix(K), "K must be a scipy sparse matrix."
        self.K = K.tocsr()

        # grids
        self.nx_w, self.ny_w = int(nx_w), int(ny_w)
        self.nx_thy, self.ny_thy = int(nx_thy), int(ny_thy)  # u and thy live here
        self.nx_thx, self.ny_thx = int(nx_thx), int(ny_thx)  # v and thx live here

        # dof counts
        self.nw = self.nx_w * self.ny_w
        self.nu = self.nx_thy * self.ny_thy
        self.nv = self.nx_thx * self.ny_thx
        self.nthx = self.nx_thx * self.ny_thx
        self.nthy = self.nx_thy * self.ny_thy
        self.N = self.nw + self.nu + self.nv + self.nthx + self.nthy

        assert self.K.shape == (self.N, self.N), f"K shape {self.K.shape} != {(self.N, self.N)}"

        # offsets
        self.off_w = 0
        self.off_u = self.off_w + self.nw
        self.off_v = self.off_u + self.nu
        self.off_thx = self.off_v + self.nv
        self.off_thy = self.off_thx + self.nthx

        self.omega = float(omega)
        self.iters = int(iters)
        self.patch_type = str(patch_type)
        self.use_pinv_fallback = bool(use_pinv_fallback)

        self.patches = self._build_patches()
        self._invK = None
        if build_inverses:
            self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(
        cls,
        assembler,
        omega: float = 0.7,
        iters: int = 1,
        build_inverses: bool = True,
        patch_type: str = "vertex_edges",
        use_pinv_fallback: bool = True,
    ):
        """
        Expects your DeRhamIGACylinderAssembler fields:
            assembler.kmat
            assembler.nx_w, assembler.ny_w
            assembler.nx_thy, assembler.ny_thy
            assembler.nx_thx, assembler.ny_thx
        """
        return cls(
            assembler.kmat,
            nx_w=assembler.nx_w, ny_w=assembler.ny_w,
            nx_thy=assembler.nx_thy, ny_thy=assembler.ny_thy,
            nx_thx=assembler.nx_thx, ny_thx=assembler.ny_thx,
            omega=omega,
            iters=iters,
            build_inverses=build_inverses,
            patch_type=patch_type,
            use_pinv_fallback=use_pinv_fallback,
        )

    # -------------------------
    # indexing helpers
    # -------------------------
    @staticmethod
    def _node(i: int, j: int, nx: int) -> int:
        return i + nx * j

    @staticmethod
    def _in_bounds(i: int, j: int, nx: int, ny: int) -> bool:
        return (0 <= i < nx) and (0 <= j < ny)

    # -------------------------
    # patch construction
    # -------------------------
    def _build_patches(self):
        if self.patch_type == "vertex_edges":
            return self._build_patches_vertex_edges()
        if self.patch_type == "wblock_vertex_edges":
            return self._build_patches_wblock_vertex_edges(bw=2, bh=2)
        raise ValueError(f"Unknown patch_type='{self.patch_type}'.")

    def _build_patches_vertex_edges(self):
        """
        For each w-vertex (iw,jw), gather:
          - w(iw,jw)
          - u neighbors on (nx_thy, ny_thy): (iw,jw), (iw, jw-1)
          - thy neighbors on same grid:      (iw,jw), (iw, jw-1)
          - v neighbors on (nx_thx, ny_thx): (iw,jw), (iw-1, jw)
          - thx neighbors on same grid:      (iw,jw), (iw-1, jw)
        with bounds checks.
        """
        patches = []

        for jw in range(self.ny_w):
            for iw in range(self.nx_w):
                dofs = []

                # --- w center ---
                dofs.append(self.off_w + self._node(iw, jw, self.nx_w))

                # --- u + thy on (nx_thy, ny_thy) ---
                # Note: u-grid has nx_thy = nxe+1, so iw==nx_w-1 may be out-of-bounds -> auto-truncated.
                uthy_cands = [(iw, jw), (iw, jw - 1)]
                for (i, j) in uthy_cands:
                    if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                        gid = self._node(i, j, self.nx_thy)
                        dofs.append(self.off_u + gid)    # u
                        dofs.append(self.off_thy + gid)  # thy

                # --- v + thx on (nx_thx, ny_thx) ---
                vthx_cands = [(iw, jw), (iw - 1, jw)]
                for (i, j) in vthx_cands:
                    if self._in_bounds(i, j, self.nx_thx, self.ny_thx):
                        gid = self._node(i, j, self.nx_thx)
                        dofs.append(self.off_v + gid)    # v
                        dofs.append(self.off_thx + gid)  # thx

                patches.append(np.array(sorted(set(dofs)), dtype=int))

        return patches
    
    def _build_patches_wblock_vertex_edges(self, bw: int = 2, bh: int = 2):
        """
        Patch is a bw x bh block of w-vertices (default 2x2 -> 4 w nodes),
        plus ALL u/v/thx/thy DOFs that touch ANY vertex in the block, using
        the same local candidate rules as _build_patches_vertex_edges():

          u/thy candidates per vertex: (iw, jw), (iw, jw-1)
          v/thx candidates per vertex: (iw, jw), (iw-1, jw)

        We slide the block by 1 in both directions for overlap (good for AS).
        Boundary truncation is automatic via bounds checks.
        """
        patches = []

        for jw0 in range(self.ny_w):
            for iw0 in range(self.nx_w):
                dofs = []
                w_vertices = []

                # --- gather w vertices in the bw x bh block ---
                for dj in range(bh):
                    for di in range(bw):
                        iw = iw0 + di
                        jw = jw0 + dj
                        if self._in_bounds(iw, jw, self.nx_w, self.ny_w):
                            gid_w = self._node(iw, jw, self.nx_w)
                            dofs.append(self.off_w + gid_w)
                            w_vertices.append((iw, jw))

                if not w_vertices:
                    continue

                # --- union of u/thy and v/thx touching all vertices ---
                for (iw, jw) in w_vertices:
                    # u + thy (same grid nx_thy x ny_thy)
                    for (i, j) in ((iw, jw), (iw, jw - 1)):
                        if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                            gid = self._node(i, j, self.nx_thy)
                            dofs.append(self.off_u + gid)
                            dofs.append(self.off_thy + gid)

                    # v + thx (same grid nx_thx x ny_thx)
                    for (i, j) in ((iw, jw), (iw - 1, jw)):
                        if self._in_bounds(i, j, self.nx_thx, self.ny_thx):
                            gid = self._node(i, j, self.nx_thx)
                            dofs.append(self.off_v + gid)
                            dofs.append(self.off_thx + gid)

                patches.append(np.array(sorted(set(dofs)), dtype=int))

        return patches

    # -------------------------
    # patch inverses
    # -------------------------
    def rebuild_patch_inverses(self):
        """
        Recompute dense inverses for each patch K[I,I].
        Call whenever K changes (new assembly, new BCs, nonlinear update, etc).
        """
        invs = []
        for I in self.patches:
            KI = self.K[I[:, None], I].toarray()
            try:
                invs.append(np.linalg.inv(KI))
            except np.linalg.LinAlgError:
                if self.use_pinv_fallback:
                    invs.append(np.linalg.pinv(KI))
                else:
                    raise
        self._invK = invs

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

            for pid, I in enumerate(self.patches):
                invKI = self._invK[pid]
                uloc = invKI @ defect[I]
                dsoln[I] += self.omega * uloc

            soln += dsoln
            defect -= self.K.dot(dsoln)

        return
