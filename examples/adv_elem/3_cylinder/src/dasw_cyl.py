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

    Grids (match your UPDATED assembler):
        w   : (nx_w,   ny_w)   = (nxe+2, nye+2)   (p2,p2)  2x2 IGA
        u   : (nx_u,   ny_u)   = (nxe+3, nye+2)   (p3,p2)  3x2 IGA
        v   : (nx_v,   ny_v)   = (nxe+2, nye+3)   (p2,p3)  2x3 IGA
        thx : (nx_thx, ny_thx) = (nxe+1, nye+2)   (p1,p2)  1x2 IGA   (x-edge-ish)
        thy : (nx_thy, ny_thy) = (nxe+2, nye+1)   (p2,p1)  2x1 IGA   (y-edge-ish)

    Patch type
    ----------
    - "vertex_edges" (default): patch anchored at each w-vertex (iw,jw):
        includes a tiny, consistent local set of dofs from each grid.

    Notes
    -----
    - Additive Schwarz: dsoln += omega * inv(K_I) * defect_I for each patch I.
    - Call rebuild_patch_inverses() whenever K changes.
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nx_w: int, ny_w: int,
        nx_u: int, ny_u: int,
        nx_v: int, ny_v: int,
        nx_thx: int, ny_thx: int,
        nx_thy: int, ny_thy: int,
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
        self.nx_u, self.ny_u = int(nx_u), int(ny_u)
        self.nx_v, self.ny_v = int(nx_v), int(ny_v)
        self.nx_thx, self.ny_thx = int(nx_thx), int(ny_thx)
        self.nx_thy, self.ny_thy = int(nx_thy), int(ny_thy)

        # dof counts
        self.nw = self.nx_w * self.ny_w
        self.nu = self.nx_u * self.ny_u
        self.nv = self.nx_v * self.ny_v
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
        Expects UPDATED assembler fields:
            assembler.kmat
            assembler.nx_w, assembler.ny_w
            assembler.nx_u, assembler.ny_u
            assembler.nx_v, assembler.ny_v
            assembler.nx_thx, assembler.ny_thx
            assembler.nx_thy, assembler.ny_thy
        """
        return cls(
            assembler.kmat,
            nx_w=assembler.nx_w, ny_w=assembler.ny_w,
            nx_u=assembler.nx_u, ny_u=assembler.ny_u,
            nx_v=assembler.nx_v, ny_v=assembler.ny_v,
            nx_thx=assembler.nx_thx, ny_thx=assembler.ny_thx,
            nx_thy=assembler.nx_thy, ny_thy=assembler.ny_thy,
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

          - u (p3,p2): x is richer -> include (iw,jw) and (iw+1,jw) on u-grid
          - v (p2,p3): y is richer -> include (iw,jw) and (iw,jw+1) on v-grid

          - thx (p1,p2): x-edge-ish -> include (iw-1,jw) and (iw,jw) on thx-grid
          - thy (p2,p1): y-edge-ish -> include (iw,jw-1) and (iw,jw) on thy-grid

        All with bounds truncation.
        """
        patches = []
        for jw in range(self.ny_w):
            for iw in range(self.nx_w):
                dofs = []

                # --- w center ---
                dofs.append(self.off_w + self._node(iw, jw, self.nx_w))

                # --- u neighbors (nx_u = nxe+3, ny_u = nye+2) ---
                for (i, j) in ((iw, jw), (iw + 1, jw)):
                    if self._in_bounds(i, j, self.nx_u, self.ny_u):
                        dofs.append(self.off_u + self._node(i, j, self.nx_u))

                # --- v neighbors (nx_v = nxe+2, ny_v = nye+3) ---
                for (i, j) in ((iw, jw), (iw, jw + 1)):
                    if self._in_bounds(i, j, self.nx_v, self.ny_v):
                        dofs.append(self.off_v + self._node(i, j, self.nx_v))

                # --- thx neighbors (nx_thx = nxe+1, ny_thx = nye+2) ---
                for (i, j) in ((iw - 1, jw), (iw, jw)):
                    if self._in_bounds(i, j, self.nx_thx, self.ny_thx):
                        dofs.append(self.off_thx + self._node(i, j, self.nx_thx))

                # --- thy neighbors (nx_thy = nxe+2, ny_thy = nye+1) ---
                for (i, j) in ((iw, jw - 1), (iw, jw)):
                    if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                        dofs.append(self.off_thy + self._node(i, j, self.nx_thy))

                patches.append(np.array(sorted(set(dofs)), dtype=int))

        return patches

    def _build_patches_wblock_vertex_edges(self, bw: int = 2, bh: int = 2):
        """
        Patch is a bw x bh block of w-vertices (default 2x2),
        plus union of u/v/thx/thy DOFs that touch ANY vertex in the block, using
        the same local candidate rules as _build_patches_vertex_edges().

        Slide by 1 in both directions for overlap.
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
                            w_vertices.append((iw, jw))
                            dofs.append(self.off_w + self._node(iw, jw, self.nx_w))

                if not w_vertices:
                    continue

                for (iw, jw) in w_vertices:
                    # u
                    for (i, j) in ((iw, jw), (iw + 1, jw)):
                        if self._in_bounds(i, j, self.nx_u, self.ny_u):
                            dofs.append(self.off_u + self._node(i, j, self.nx_u))

                    # v
                    for (i, j) in ((iw, jw), (iw, jw + 1)):
                        if self._in_bounds(i, j, self.nx_v, self.ny_v):
                            dofs.append(self.off_v + self._node(i, j, self.nx_v))

                    # thx
                    for (i, j) in ((iw - 1, jw), (iw, jw)):
                        if self._in_bounds(i, j, self.nx_thx, self.ny_thx):
                            dofs.append(self.off_thx + self._node(i, j, self.nx_thx))

                    # thy
                    for (i, j) in ((iw, jw - 1), (iw, jw)):
                        if self._in_bounds(i, j, self.nx_thy, self.ny_thy):
                            dofs.append(self.off_thy + self._node(i, j, self.nx_thy))

                patches.append(np.array(sorted(set(dofs)), dtype=int))

        return patches

    # -------------------------
    # patch inverses
    # -------------------------
    def rebuild_patch_inverses(self):
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
                uloc = self._invK[pid] @ defect[I]
                dsoln[I] += self.omega * uloc

            soln += dsoln
            defect -= self.K.dot(dsoln)

        return