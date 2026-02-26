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
                uthy_cands = [(iw, jw), (iw - 1, jw)]
                for (i, j) in uthy_cands:
                    if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                        gid = self._node(i, j, self.nx_thy)
                        dofs.append(self.off_u + gid)    # u
                        dofs.append(self.off_thy + gid)  # thy

                # --- v + thx on (nx_thx, ny_thx) ---
                vthx_cands = [(iw, jw), (iw, jw - 1)]
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
                    for (i, j) in ((iw, jw), (iw - 1, jw)):
                        if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                            gid = self._node(i, j, self.nx_thy)
                            dofs.append(self.off_u + gid)
                            dofs.append(self.off_thy + gid)

                    # v + thx (same grid nx_thx x ny_thx)
                    for (i, j) in ((iw, jw), (iw, jw - 1)):
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


# ==========================================================


class MixedTwoDimAddSchwarzDeRhamCylinderVertexEdges:
    """
    2D additive Schwarz smoother for the MIXED DeRham cylinder block system:

        unknown ordering (global):
            u = [ w, w2, u, v, thx, thy ]

    Mixed De Rham grids (match your Mixed assembler / element docstring):
        w   : (nx_w,   ny_w)    = (nxe+3, nye+3)   (p3,p3)
        w2  : (nx_w2,  ny_w2)   = (nxe+3, nye+1)   (p3,p1)   "vertex -> edge-edges" in y
        u   : (nx_u,   ny_u)    = (nxe+2, nye+3)   (p2,p3)
        v   : (nx_v,   ny_v)    = (nxe+3, nye+2)   (p3,p2)
        thx : (nx_thx, ny_thx)  = (nxe+3, nye+2)   (p3,p2)   same as v-grid
        thy : (nx_thy, ny_thy)  = (nxe+2, nye+3)   (p2,p3)   same as u-grid

    Patch types
    ----------
    - "vertex_edges" (default): 1-vertex patch anchored at each w-vertex (iw,jw)
    - "wblock_vertex_edges":    2-vertex case via a bw x bh block of w-vertices (default 2x2)

    Notes on the new w2 coupling
    ----------------------------
    Your constraint row puts *w -> w2* coupling that is like "vertex to edge-edges"
    (two orders lower in y). For smoothing, we include a small local set of w2 DOFs
    associated with the same (iw,jw) anchor:
        w2(iw, jw2) with jw2 in {jw-1, jw} on the w2 grid (ny_w2 = nye+1),
    and also include neighbors in x as needed by bounds truncation.

    This is intentionally local + robust; it doesn't assume periodic hoop (your assembler uses open hoop).
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx_w: int, ny_w: int,
        nx_w2: int, ny_w2: int,
        nx_u: int, ny_u: int,
        nx_v: int, ny_v: int,
        omega: float = 0.7,
        iters: int = 1,
        build_inverses: bool = True,
        patch_type: str = "vertex_edges",
        use_pinv_fallback: bool = True,
        bw: int = 2, bh: int = 2,  # only used for wblock patches
    ):
        assert sp.isspmatrix(K), "K must be a scipy sparse matrix."
        self.K = K.tocsr()

        # grids
        self.nx_w,  self.ny_w  = int(nx_w),  int(ny_w)
        self.nx_w2, self.ny_w2 = int(nx_w2), int(ny_w2)
        self.nx_u,  self.ny_u  = int(nx_u),  int(ny_u)
        self.nx_v,  self.ny_v  = int(nx_v),  int(ny_v)
        self.nx_thx, self.ny_thx = self.nx_v, self.ny_v  # same grid
        self.nx_thy, self.ny_thy = self.nx_u, self.ny_u  # same grid

        # dof counts
        self.nw   = self.nx_w  * self.ny_w
        self.nw2  = self.nx_w2 * self.ny_w2
        self.nu   = self.nx_u  * self.ny_u
        self.nv   = self.nx_v  * self.ny_v
        self.nthx = self.nx_thx * self.ny_thx
        self.nthy = self.nx_thy * self.ny_thy
        self.N = self.nw + self.nw2 + self.nu + self.nv + self.nthx + self.nthy

        assert self.K.shape == (self.N, self.N), f"K shape {self.K.shape} != {(self.N, self.N)}"

        # offsets
        self.off_w   = 0
        self.off_w2  = self.off_w   + self.nw
        self.off_u   = self.off_w2  + self.nw2
        self.off_v   = self.off_u   + self.nu
        self.off_thx = self.off_v   + self.nv
        self.off_thy = self.off_thx + self.nthx

        self.omega = float(omega)
        self.iters = int(iters)
        self.patch_type = str(patch_type)
        self.use_pinv_fallback = bool(use_pinv_fallback)

        self.bw = int(bw)
        self.bh = int(bh)

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
        bw: int = 2,
        bh: int = 2,
    ):
        """
        Expects your MixedDeRhamIGACylinderAssembler (or similar) fields:
            assembler.kmat
            assembler.nx_w,  assembler.ny_w
            assembler.nx_w2, assembler.ny_w2
            assembler.nx_u,  assembler.ny_u
            assembler.nx_v,  assembler.ny_v
        """
        return cls(
            assembler.kmat,
            nx_w=assembler.nx_w, ny_w=assembler.ny_w,
            nx_w2=assembler.nx_w2, ny_w2=assembler.ny_w2,
            nx_u=assembler.nx_u, ny_u=assembler.ny_u,
            nx_v=assembler.nx_v, ny_v=assembler.ny_v,
            omega=omega,
            iters=iters,
            build_inverses=build_inverses,
            patch_type=patch_type,
            use_pinv_fallback=use_pinv_fallback,
            bw=bw, bh=bh,
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
            return self._build_patches_wblock_vertex_edges(bw=self.bw, bh=self.bh)
        raise ValueError(f"Unknown patch_type='{self.patch_type}'.")

    def _add_w_center(self, dofs, iw, jw):
        dofs.append(self.off_w + self._node(iw, jw, self.nx_w))

    def _add_u_thy_neighbors(self, dofs, iw, jw):
        """
        u/thy live on (nx_u, ny_u) = (nxe+2, nye+3) which is "one lower in x" than w,
        and same in y as w. A robust vertex-edge rule (consistent with your old intent):
            include (iw, jw) and (iw-1, jw) on the u/thy grid.
        """
        for (i, j) in ((iw, jw), (iw - 1, jw)):
            if self._in_bounds(i, j, self.nx_u, self.ny_u):
                gid = self._node(i, j, self.nx_u)
                dofs.append(self.off_u + gid)
                dofs.append(self.off_thy + gid)

    def _add_v_thx_neighbors(self, dofs, iw, jw):
        """
        v/thx live on (nx_v, ny_v) = (nxe+3, nye+2) which matches w in x and is one lower in y.
        Robust vertex-edge rule:
            include (iw, jw) and (iw, jw-1) on the v/thx grid.
        """
        for (i, j) in ((iw, jw), (iw, jw - 1)):
            if self._in_bounds(i, j, self.nx_v, self.ny_v):
                gid = self._node(i, j, self.nx_v)
                dofs.append(self.off_v + gid)
                dofs.append(self.off_thx + gid)

    def _add_w2_neighbors(self, dofs, iw, jw):
        """
        w2 lives on (nx_w2, ny_w2) = (nxe+3, nye+1) : same x as w, two lower in y than w (ny_w = nye+3).
        Treat this like "vertex -> edge-edges" in y.

        Practical local choice (good smoother):
            include w2(iw, jw2) for jw2 in {jw-1, jw} (mapped into ny_w2 bounds),
        plus a tiny x-neighborhood {iw, iw-1} to match how other vertex-edge maps behave near seams.
        """
        for ii in (iw, iw - 1):
            for jj in (jw, jw - 1):
                if self._in_bounds(ii, jj, self.nx_w2, self.ny_w2):
                    gid = self._node(ii, jj, self.nx_w2)
                    dofs.append(self.off_w2 + gid)

    def _build_patches_vertex_edges(self):
        """
        1-vertex patch anchored at each w-vertex (iw,jw):
          - w(iw,jw)
          - w2 local 'edge-edges' in y: (iw or iw-1) x (jw or jw-1)
          - u,thy on u-grid: (iw,jw) and (iw-1,jw)
          - v,thx on v-grid: (iw,jw) and (iw,jw-1)
        with bounds truncation.
        """
        patches = []
        for jw in range(self.ny_w):
            for iw in range(self.nx_w):
                dofs = []
                self._add_w_center(dofs, iw, jw)
                self._add_w2_neighbors(dofs, iw, jw)
                self._add_u_thy_neighbors(dofs, iw, jw)
                self._add_v_thx_neighbors(dofs, iw, jw)
                patches.append(np.array(sorted(set(dofs)), dtype=int))
        return patches

    def _build_patches_wblock_vertex_edges(self, bw: int = 2, bh: int = 2):
        """
        2-vertex (and more) case: patch is a bw x bh block of w-vertices (default 2x2),
        plus union of all touching w2/u/v/thx/thy DOFs using the same local candidate rules
        as the 1-vertex patch.

        Slide by 1 in i/j for overlap (as written).
        """
        patches = []
        for jw0 in range(self.ny_w):
            for iw0 in range(self.nx_w):
                dofs = []
                w_vertices = []

                for dj in range(bh):
                    for di in range(bw):
                        iw = iw0 + di
                        jw = jw0 + dj
                        if self._in_bounds(iw, jw, self.nx_w, self.ny_w):
                            w_vertices.append((iw, jw))
                            dofs.append(self.off_w + self._node(iw, jw, self.nx_w))

                if not w_vertices:
                    continue

                for (iw, jw) in w_vertices:
                    self._add_w2_neighbors(dofs, iw, jw)
                    self._add_u_thy_neighbors(dofs, iw, jw)
                    self._add_v_thx_neighbors(dofs, iw, jw)

                patches.append(np.array(sorted(set(dofs)), dtype=int))
        return patches

    # -------------------------
    # patch inverses
    # -------------------------
    def rebuild_patch_inverses(self):
        """
        Recompute dense inverses for each patch K[I,I].
        Call whenever K changes (new assembly / BC changes / nonlinear update / etc).
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
