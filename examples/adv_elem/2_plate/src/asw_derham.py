import numpy as np
import scipy.sparse as sp


class TwoDimAddSchwarzDeRhamVertexEdges:
    """
    2D additive Schwarz smoother for De Rham IGA plate block system:

        K = sp.bmat([[Kww,   Kwtx,  Kwty],
                     [Ktxw,  Ktxtx, Ktxty],
                     [Ktyw,  Ktytx, Ktyty]], format="csr")

    Global ordering assumed:
        u = [ w(0..nw-1), thx(0..ntx-1), thy(0..nty-1) ]

    De Rham grids (same as the plate assembler I wrote you):
        w   grid: (nxw,  nyw)  with nxw = nxe+2, nyw = nye+2
        thx grid: (nxtx, nytx) with nxtx = nxe+1, nytx = nye+2
        thy grid: (nxty, nyty) with nxty = nxe+2, nyty = nye+1

    Patch types
    ----------
    - "vertex_edges" (default): patch anchored at each w-vertex (iw,jw):
        { w(iw,jw),
          thx(iw-1,jw), thx(iw,jw),  thx(iw-1,jw-1), thx(iw,jw-1),
          thy(iw,jw-1), thy(iw,jw),  thy(iw-1,jw-1), thy(iw-1,jw) }

      i.e. all nearby thx/thy edges that touch the vertex (up to 4 of each component),
      with boundary truncation automatically.

    Notes
    -----
    - This is additive Schwarz: dsoln += omega * inv(K_I) * defect_I for each patch.
    - Best used as a smoother (iters=1..3). For SPD-ish systems, omega ~ 0.6-0.9.
    - Rebuild inverses if K changes (new assembly / BC changes / nonlinear update).
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nxw: int,
        nyw: int,
        nxtx: int,
        nytx: int,
        nxty: int,
        nyty: int,
        omega: float = 0.7,
        iters: int = 1,
        build_inverses: bool = True,
        patch_type: str = "vertex_edges",
        use_pinv_fallback: bool = True,
    ):
        assert sp.isspmatrix(K), "K must be a scipy sparse matrix."
        self.K = K.tocsr()

        self.nxw, self.nyw = int(nxw), int(nyw)
        self.nxtx, self.nytx = int(nxtx), int(nytx)
        self.nxty, self.nyty = int(nxty), int(nyty)

        self.nw = self.nxw * self.nyw
        self.ntx = self.nxtx * self.nytx
        self.nty = self.nxty * self.nyty
        self.N = self.nw + self.ntx + self.nty

        assert self.K.shape == (self.N, self.N), f"K shape {self.K.shape} != {(self.N, self.N)}"

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
        # Expecting the DeRhamIGAPlateAssembler-like fields
        return cls(
            assembler.kmat,
            nxw=assembler.nxw, nyw=assembler.nyw,
            nxtx=assembler.nxtx, nytx=assembler.nytx,
            nxty=assembler.nxty, nyty=assembler.nyty,
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

    def _in_bounds(self, i: int, j: int, nx: int, ny: int) -> bool:
        return (0 <= i < nx) and (0 <= j < ny)

    # -------------------------
    # patch construction
    # -------------------------
    def _build_patches(self):
        if self.patch_type == "vertex_edges":
            return self._build_patches_vertex_edges()
        raise ValueError(f"Unknown patch_type='{self.patch_type}'.")

    def _build_patches_vertex_edges(self):
        """
        For each w-vertex (iw,jw), gather:
          - itself
          - thx dofs adjacent to the vertex:
                thx(iw, jw), thx(iw-1, jw), thx(iw, jw-1), thx(iw-1, jw-1)
          - thy dofs adjacent to the vertex:
                thy(iw, jw), thy(iw, jw-1), thy(iw-1, jw), thy(iw-1, jw-1)
        with proper bounds checks.
        """
        patches = []
        off_tx = self.nw
        off_ty = self.nw + self.ntx

        for jw in range(self.nyw):
            for iw in range(self.nxw):
                dofs = []

                # w center
                w_gid = self._node(iw, jw, self.nxw)
                dofs.append(w_gid)

                # thx neighbors (grid: nxtx x nytx)
                # candidate (i,j) pairs on thx grid
                thx_cands = [
                    (iw,   jw),
                    (iw-1, jw),
                    (iw,   jw-1),
                    (iw-1, jw-1),
                ]
                for (i, j) in thx_cands:
                    if self._in_bounds(i, j, self.nxtx, self.nytx):
                        dofs.append(off_tx + self._node(i, j, self.nxtx))

                # thy neighbors (grid: nxty x nyty)
                thy_cands = [
                    (iw,   jw),
                    (iw,   jw-1),
                    (iw-1, jw),
                    (iw-1, jw-1),
                ]
                for (i, j) in thy_cands:
                    if self._in_bounds(i, j, self.nxty, self.nyty):
                        dofs.append(off_ty + self._node(i, j, self.nxty))

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
