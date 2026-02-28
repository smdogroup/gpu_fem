import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from std_assembler import StandardBeamAssembler


class SchurComplementBeamAssembler(StandardBeamAssembler):
    """
    Global Schur-complement assembler.

    Uses element-provided DOF counts:
      - element.dof_per_node        : condensed DOFs per node (MG-visible), e.g. 2 for [w, th]
      - element.full_dof_per_node   : full DOFs per node for global condensation, e.g. 4 for [w, th, xiw, xith]

    Full-node ordering assumed (per node):
        [w, th, xiw, xith]  (or generally: first dpn are the "kept" DOFs, remaining are auxiliary)
    Condensed-node ordering (per node):
        first `dof_per_node` entries (e.g., [w, th])

    This class:
      - assembles FULL global K,f with full_dof_per_node (no BCs on aux)
      - globally eliminates aux DOFs
      - exposes ONLY the condensed (kept) system through self.kmat/self.force and solves only that
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- pull DOF sizes from element ---
        if not hasattr(self.element, "full_dof_per_node"):
            raise AttributeError(
                "Element must define `full_dof_per_node` for SchurComplementBeamAssembler."
            )

        self.full_dof_per_node = int(self.element.full_dof_per_node)
        self.dof_per_node = int(self.element.dof_per_node)  # condensed (MG-visible)

        # expose condensed size to outside world
        self.N = self.dof_per_node * self.nnodes

        # BCs only on condensed DOFs (same convention as StandardBeamAssembler)
        dpn = self.dof_per_node
        if self.element.clamped:
            # fully clamped on kept dofs only
            self.bcs = list(range(dpn)) + list(range(dpn * (self.nnodes - 1), dpn * self.nnodes))
        else:
            # simply supported (w only) assumed: idof=0 at ends
            # if dpn>1, only constrain w at ends like your StandardBeamAssembler does
            self.bcs = [0, dpn * (self.nnodes - 1)]

        # internal full-system holders (optional for debugging)
        self._kmat_full = None
        self._force_full = None

    @property
    def dof_conn_full(self):
        """
        Element DOF connectivity for full system (2 nodes/elem, full_dof_per_node dofs each),
        with per-node ordering [w, th, xiw, xith, ...].
        """
        dpn_full = self.full_dof_per_node
        return [[dpn_full * ix + j for j in range(2 * dpn_full)] for ix in range(self.nxe)]

    def _apply_bcs_condensed(self, S: sp.spmatrix, f: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
        """Apply BCs ONLY to condensed system."""
        S = S.tolil()
        for bc in self.bcs:
            S[bc, :] = 0.0
            S[:, bc] = 0.0
            S[bc, bc] = 1.0
            f[bc] = 0.0
        return S.tocsr(), f
    
    def _build_nofill_mask_kept(self, nnodes: int, dpn_keep: int) -> sp.csr_matrix:
        """
        Mask for a 1D nodal nearest-neighbor stencil on the KEPT unknowns.
        Keeps block entries between node i and nodes in {i-1,i,i+1}.
        """
        rows, cols = [], []
        for i in range(nnodes):
            for j in (i - 1, i, i + 1):
                if 0 <= j < nnodes:
                    # keep the entire dpn_keep x dpn_keep block between node i and node j
                    for a in range(dpn_keep):
                        r = dpn_keep * i + a
                        for b in range(dpn_keep):
                            c = dpn_keep * j + b
                            rows.append(r)
                            cols.append(c)

        data = np.ones(len(rows), dtype=float)
        return sp.coo_matrix((data, (rows, cols)),
                            shape=(dpn_keep * nnodes, dpn_keep * nnodes)).tocsr()

    def _assemble_system(self):
        # ---- Assemble FULL global K,f with dpn_full ----
        dpn_full = self.full_dof_per_node
        Nfull = dpn_full * self.nnodes

        rows = []
        cols = []
        vals = []
        ffull = np.zeros(Nfull, dtype=float)

        # element-level values
        kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length)
        unit_felem = self.element.get_felem(mag=1.0, elem_length=self.elem_length)

        # element loads at midpoints
        x_vals = [(ielem + 0.5) * self.elem_length for ielem in range(self.nxe)]
        load_vals = [self.load_fcn(x) for x in x_vals]

        # print(f"{kelem.shape=} {unit_felem.shape=}")

        # --- sanity checks on element sizes ---
        if kelem.shape != (2 * dpn_full, 2 * dpn_full):
            raise ValueError(
                f"Element kelem has shape {kelem.shape}, expected {(2*dpn_full, 2*dpn_full)} "
                f"for full_dof_per_node={dpn_full}."
            )
        if unit_felem.shape[0] != 2 * dpn_full:
            raise ValueError(
                f"Element felem has length {unit_felem.shape[0]}, expected {2*dpn_full} "
                f"for full_dof_per_node={dpn_full}."
            )

        # IMPORTANT:
        # This assembler assumes kelem/felem local ordering is NODE-INTERLEAVED, i.e.:
        #   [node0(dpn_full dofs), node1(dpn_full dofs)]
        # with per-node order [kept..., aux...].
        #
        # If your element returns a different local ordering, permute kelem and unit_felem here.

        for ielem in range(self.nxe):
            edofs = np.array(self.dof_conn_full[ielem], dtype=int)

            ii, jj = np.meshgrid(edofs, edofs, indexing="ij")
            rows.append(ii.ravel())
            cols.append(jj.ravel())
            vals.append(kelem.ravel())

            fe = unit_felem * load_vals[ielem]
            np.add.at(ffull, edofs, fe)

        Kfull = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(Nfull, Nfull),
        ).tocsr()

        self._kmat_full = Kfull
        self._force_full = ffull

        # ---- GLOBAL Schur complement eliminating auxiliary DOFs ----
        dpn_keep = self.dof_per_node
        dpn_aux = dpn_full - dpn_keep
        if dpn_aux <= 0:
            raise ValueError(
                f"full_dof_per_node ({dpn_full}) must be > dof_per_node ({dpn_keep})."
            )

        # indices: keep are first dpn_keep per node, aux are remaining per node
        iu = np.array([dpn_full * i + j for i in range(self.nnodes) for j in range(dpn_keep)], dtype=int)
        ib = np.array([dpn_full * i + j for i in range(self.nnodes) for j in range(dpn_keep, dpn_full)], dtype=int)

        Kuu = Kfull[iu, :][:, iu]
        Kub = Kfull[iu, :][:, ib]
        Kbu = Kfull[ib, :][:, iu]
        Kbb = Kfull[ib, :][:, ib]

        fu = ffull[iu].copy()
        fb = ffull[ib].copy()

        # Factorize Kbb once
        Kbb_fac = spla.splu(Kbb.tocsc())

        # X = Kbb^{-1} Kbu ; y = Kbb^{-1} fb
        X = Kbb_fac.solve(Kbu.toarray())
        y = Kbb_fac.solve(fb)

        # Exact Schur complement and rhs
        S_exact = Kuu - Kub @ sp.csr_matrix(X)
        fS = fu - Kub @ y


        # -----------------------------
        # Enforce NO-FILL pattern:
        # keep only nearest-neighbor nodal coupling in the KEPT space
        # -----------------------------
        # tried but this doesn't work..
        # mask = self._build_nofill_mask_kept(self.nnodes, dpn_keep)

        # # Drop off-pattern entries
        # S = S_exact.tocsr().multiply(mask)
        S = S_exact.tocsr()
        
        plt.spy(S_exact)
        plt.show()

        # (optional) clean tiny numerical trash introduced by solve/multiply
        S.eliminate_zeros()

        # Apply BCs ONLY to condensed system
        S, fS = self._apply_bcs_condensed(S, np.asarray(fS).ravel())

        # Store as BSR with blocksize=(dpn_keep, dpn_keep) for MG smoothers
        self.kmat = S.tobsr(blocksize=(dpn_keep, dpn_keep))
        self.force = np.asarray(fS).ravel()

    def direct_solve(self):
        self._assemble_system()
        self.u = spla.spsolve(self.kmat, self.force)
        return self.u

    @property
    def xvec(self) -> list:
        return [i * self.elem_length for i in range(self.nnodes)]

    def plot_disp(self, idof: int = 0):
        xvec = self.xvec
        dpn = self.dof_per_node
        w = self.u[idof::dpn]
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)" if idof == 0 else "th(x)")
        plt.show()