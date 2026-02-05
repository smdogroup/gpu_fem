import numpy as np
import scipy.sparse as sp

class OneDimAddSchwarzVertex2Edges:
    """
    Additive Schwarz smoother for a *single* global matrix K that was built like:
        K = sp.bmat([[Kww, Kwth],
                    [Kthw, Kthth]], format="csr")

    Global ordering assumed: [w0..w_{nw-1}, th0..th_{nth-1}]

    Patch i contains:
        - w_i
        - theta_{i-1}, theta_i   (two adjacent edges)
    with boundary truncation:
        i=0      -> {w0, theta0}
        i=nw-1   -> {w_{nw-1}, theta_{nth-1}}
    """

    def __init__(self, K: sp.spmatrix, nw: int, nth: int, omega: float = 0.7, iters: int = 1,
                 build_inverses: bool = True):
        assert sp.isspmatrix(K), "K must be a scipy sparse matrix (e.g., result of sp.bmat)."
        self.K = K.tocsr()

        self.nw = int(nw)
        self.nth = int(nth)
        self.N = self.nw + self.nth
        assert self.K.shape == (self.N, self.N), f"K shape {self.K.shape} != {(self.N, self.N)}"

        self.omega = float(omega)
        self.iters = int(iters)

        self.patches = self._build_patches()
        self._invK = None
        if build_inverses:
            self.rebuild_patch_inverses()

    @classmethod
    def from_assembler(cls, assembler, omega: float = 0.7, iters: int = 1, build_inverses: bool = True):
        # assembler must have kmat (global CSR), nw, nth
        return cls(assembler.kmat, nw=assembler.nw, nth=assembler.nth,
                   omega=omega, iters=iters, build_inverses=build_inverses)

    def _build_patches(self):
        patches = []
        for iw in range(self.nw):
            dofs = [iw]  # w_i in global indexing

            th_left = iw - 1
            th_right = iw

            if 0 <= th_left < self.nth:
                dofs.append(self.nw + th_left)
            if 0 <= th_right < self.nth:
                dofs.append(self.nw + th_right)

            patches.append(np.array(dofs, dtype=int))
        return patches

    def rebuild_patch_inverses(self):
        """
        Recompute dense inverses for each patch K[I,I].
        Call whenever K changes (new assembly, different BCs, nonlinear update, etc).
        """
        invs = []
        for I in self.patches:
            KI = self.K[I[:, None], I].toarray()

            # Guard against accidental singular patch (e.g., if BC rows/cols are identity-only)
            # This is *usually* fine, but if it trips, switch to lstsq.
            try:
                invs.append(np.linalg.inv(KI))
            except np.linalg.LinAlgError:
                # fallback: pseudo-inverse (robust for singular patches)
                invs.append(np.linalg.pinv(KI))

        self._invK = invs

    def solve(self, rhs: np.ndarray):
        """
        Apply Schwarz iterations starting from zero.
        Useful as a standalone smoother or as a preconditioner action z = M^{-1} r.
        """
        rhs = np.asarray(rhs)
        assert rhs.shape == (self.N,)
        soln = np.zeros_like(rhs)
        defect = rhs.copy()
        self.smooth_defect(soln, defect)
        return soln

    def smooth_defect(self, soln: np.ndarray, defect: np.ndarray):
        """
        Additive Schwarz smoothing:
            dsoln[I] += omega * inv(K[I,I]) * defect[I]
            soln += dsoln
            defect -= K * dsoln
        """
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
                dloc = defect[I]
                uloc = invKI @ dloc
                dsoln[I] += self.omega * uloc

            soln += dsoln
            defect -= self.K.dot(dsoln)
        return
