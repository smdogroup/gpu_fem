import numpy as np
import scipy.sparse as sp

class OneDimAddSchwarzVertex2Edges:
    """
    Additive Schwarz smoother for global K built as:
        K = sp.bmat([[Kww, Kwth],
                    [Kthw, Kthth]], format="csr")

    Global ordering assumed: [w0..w_{nw-1}, th0..th_{nth-1}]

    patch_type options
    ------------------
    1) "vertex2edges" (default):
        patch i: { w_i, th_{i-1}, th_i } (with boundary truncation)

    2) "2nodes3edges":
        patch i (anchored on edge i): { w_i, w_{i+1}, th_{i-1}, th_i, th_{i+1} }
        with boundary truncation.
        - i ranges over edges: i = 0..nth-1
        - This is the natural 1D analogue of a 2-node / 3-edge patch.

    Notes:
      - This class is 1D only; no nx/ny parameters.
      - You already pass nw and nth, so nothing is inferred from the matrix.
    """

    def __init__(
        self,
        K: sp.spmatrix,
        nw: int,
        nth: int,
        omega: float = 0.7,
        iters: int = 1,
        build_inverses: bool = True,
        patch_type: str = "vertex2edges",
    ):
        assert sp.isspmatrix(K), "K must be a scipy sparse matrix (e.g., result of sp.bmat)."
        self.K = K.tocsr()

        self.nw = int(nw)
        self.nth = int(nth)
        self.N = self.nw + self.nth
        assert self.K.shape == (self.N, self.N), f"K shape {self.K.shape} != {(self.N, self.N)}"

        self.omega = float(omega)
        self.iters = int(iters)
        self.patch_type = str(patch_type)

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
        patch_type: str = "vertex2edges",
    ):
        return cls(
            assembler.kmat,
            nw=assembler.nw,
            nth=assembler.nth,
            omega=omega,
            iters=iters,
            build_inverses=build_inverses,
            patch_type=patch_type,
        )

    # -------------------------
    # Patch construction
    # -------------------------
    def _build_patches(self):
        if self.patch_type == "vertex2edges":
            return self._build_patches_vertex2edges()
        if self.patch_type == "2nodes3edges":
            return self._build_patches_2nodes3edges()
        raise ValueError(f"Unknown patch_type='{self.patch_type}'. Use 'vertex2edges' or '2nodes3edges'.")

    def _build_patches_vertex2edges(self):
        patches = []
        for iw in range(self.nw):
            dofs = [iw]  # w_i
            th_left = iw - 1
            th_right = iw
            if 0 <= th_left < self.nth:
                dofs.append(self.nw + th_left)
            if 0 <= th_right < self.nth:
                dofs.append(self.nw + th_right)
            patches.append(np.array(sorted(set(dofs)), dtype=int))
        return patches

    def _build_patches_2nodes3edges(self):
        """
        Patch anchored on edge i (i=0..nth-1):
          w nodes: i, i+1   (if in range)
          th edges: i-1, i, i+1  (if in range)
        """
        patches = []
        for iedge in range(self.nth):
            dofs = []

            # 2 nodes (w)
            w0 = iedge
            w1 = iedge + 1
            if 0 <= w0 < self.nw:
                dofs.append(w0)
            if 0 <= w1 < self.nw:
                dofs.append(w1)

            # 3 edges (th)
            for j in (iedge - 1, iedge, iedge + 1):
                if 0 <= j < self.nth:
                    dofs.append(self.nw + j)

            patches.append(np.array(sorted(set(dofs)), dtype=int))
        return patches

    # -------------------------
    # Patch inverses
    # -------------------------
    def rebuild_patch_inverses(self):
        """
        Recompute dense inverses for each patch K[I,I].
        Call whenever K changes (new assembly, different BCs, nonlinear update, etc).
        """
        invs = []
        for I in self.patches:
            KI = self.K[I[:, None], I].toarray()
            try:
                invs.append(np.linalg.inv(KI))
            except np.linalg.LinAlgError:
                invs.append(np.linalg.pinv(KI))
        self._invK = invs

    # -------------------------
    # Smoother / preconditioner
    # -------------------------
    def solve(self, rhs: np.ndarray):
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

            for patch_id, I in enumerate(self.patches):
                invKI = self._invK[patch_id]
                uloc = invKI @ defect[I]
                dsoln[I] += self.omega * uloc

            soln += dsoln
            defect -= self.K.dot(dsoln)
        return
