import numpy as np
import sys
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
import scipy.sparse as sp

class TimoshenkoElement_OptProlong:
    """fully integrated timoshenko beam element (experiences locking"""
    def __init__(self, reduced_integrated:bool=False, locking_aware_prolong:bool=True, prolong_mode:str='energy', lam:float=1e-2):              
        self.dof_per_node = 2
        self.nodes_per_elem = 2
        self.reduced_integrated = reduced_integrated
        self.clamped = True
        self.locking_aware_prolong = locking_aware_prolong
        self.prolong_mode = prolong_mode
        self.lam = lam
        self._P0_cache = {}
        self._P1_cache = {}
        self._kmat_cache = {}

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()

        if self.reduced_integrated:
            shear_pts, shear_wts = zero_order_quadrature()
        else:
            shear_pts, shear_wts = second_order_quadrature()

        kelem = np.zeros((4, 4))
        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / 2.0 / (1 + nu)
        ksGA = ks * G * thick
        J = elem_length / 2.0 # dx/dxi jacobian

        # trv shear energy
        for xi, wt in zip(shear_pts, shear_wts):
            # basis vecs
            psi_val = [lagrange(i, xi) for i in range(2)]
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]

            # w_{,x} - th
            for i in range(2):
                for j in range(2):
                    c_shear = ksGA * wt * J
                    kelem[i,j] += c_shear * dpsi[i] * dpsi[j]
                    kelem[i, 2+j] -= c_shear * dpsi[i] * psi_val[j]
                    kelem[2+i, j] -= c_shear * psi_val[i] * dpsi[j]
                    kelem[2+i, 2+j] += c_shear * psi_val[i] * psi_val[j]
            
        # bending energy
        for xi, wt in zip(pts, weights):
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]
            for i in range(2):
                for j in range(2):
                    kelem[2+i, 2+j] += EI * wt * J * dpsi[i] * dpsi[j]

        # import matplotlib.pyplot as plt
        # plt.imshow(kelem)
        # plt.show()

        # change order from [w1,w2,th1,th2] => [w1, th1, w2, th2]
        new_order = np.array([0, 2, 1, 3])
        kelem = kelem[new_order, :][:, new_order]
        return kelem

    def get_felem(self, mag, elem_length):
        """get element load vector"""
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()
        felem = np.zeros(4)
        for xi, wt in zip(pts, wts):
            psi_val = [lagrange(i, xi) for i in range(2)]
            for i in range(2):
                felem[2 * i] += mag * wt * J * psi_val[i]
        return felem
    
    def _build_P1_scalar(self, nxe_coarse: int, apply_bcs: bool = False) -> sp.csr_matrix:
        """
        1D scalar nodal prolongation from nxe_coarse -> 2*nxe_coarse.
        Nodes: nc = nxe_coarse+1, nf = 2*nxe_coarse+1 = 2*nc-1.

        IMPORTANT: For w/theta systems, do NOT apply BCs here, because kron(P1, I2)
        would incorrectly apply them to theta too. Apply BCs at the P0 level instead.
        """
        key = (nxe_coarse, apply_bcs)
        if key in self._P1_cache:
            return self._P1_cache[key]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1

        rows, cols, vals = [], [], []

        # even fine nodes copy coarse
        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        # odd fine nodes average
        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()

        # Optional scalar BCs (only use for truly scalar problems)
        if apply_bcs:
            P1[[0, nf - 1], :] = 0.0
            P1[:, [0, nc - 1]] = 0.0
            P1.eliminate_zeros()

        self._P1_cache[key] = P1
        return P1

    def _build_P0_wth(self, nxe_coarse: int, ndof_per_node: int = 2) -> sp.csr_matrix:
        """
        Build initial P0 for [w, th] using same scalar P1 on each dof, BUT apply BCs
        at the P0 level so SS only constrains w (not th).

        SS:    w(0)=w(L)=0 enforced in P0 (rows+cols for w only)
        clamp: w and th at ends enforced in P0
        """
        key = (nxe_coarse, ndof_per_node, self.clamped)
        if hasattr(self, "_P0_cache") and key in self._P0_cache:
            return self._P0_cache[key]

        # build scalar prolongation WITHOUT BCs
        P1 = self._build_P1_scalar(nxe_coarse, apply_bcs=False)

        # lift to block prolongation
        P0 = sp.kron(P1, sp.eye(ndof_per_node, format="csr"), format="csr").tocsr()

        # dimensions for indexing end DOFs
        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1

        # DOF indices for w at ends
        wL_f, wR_f = 0, 2 * (nf - 1)
        wL_c, wR_c = 0, 2 * (nc - 1)

        # zero fine w boundary rows
        P0[[wL_f, wR_f], :] = 0.0
        # zero coarse w boundary cols
        P0[:, [wL_c, wR_c]] = 0.0

        if self.clamped:
            # also constrain theta at ends
            thL_f, thR_f = 1, 2 * (nf - 1) + 1
            thL_c, thR_c = 1, 2 * (nc - 1) + 1

            P0[[thL_f, thR_f], :] = 0.0
            P0[:, [thL_c, thR_c]] = 0.0

        P0.eliminate_zeros()

        if not hasattr(self, "_P0_cache"):
            self._P0_cache = {}
        self._P0_cache[key] = P0
        return P0


    def _locking_aware_prolong(self, P0:sp.csr_matrix, nxe_c:int, length:float):
        nx_c = nxe_c + 1
        nxe_f = 2 * nxe_c; nx_f = nxe_f + 1
        dx_c = length / nxe_c; dx_f = length / nxe_f
        a_c = 1.0/dx_c; a_f = 1.0/dx_f
        # convert P0 to dense array (could do sparse but I'm just doing dense for now)
        P0 = P0.toarray()

        # build trv shear matrix operators
        G_c = np.zeros((nxe_c, 2*nx_c))
        G_f = np.zeros((nxe_f, 2*nx_f))
        for ielem in range(nxe_c):
            # (w_{i+1} - w_i)/dx - 0.5 * (th_i + th_{i+1})
            G_c[ielem, 2*ielem+0] += -a_c
            G_c[ielem, 2*ielem+1] += -0.5
            G_c[ielem, 2*ielem+2] += a_c
            G_c[ielem, 2*ielem+3] += -0.5

        for ielem in range(nxe_f):
            # (w_{i+1} - w_i)/dx - 0.5 * (th_i + th_{i+1})
            G_f[ielem, 2*ielem+0] += -a_f
            G_f[ielem, 2*ielem+1] += -0.5
            G_f[ielem, 2*ielem+2] += a_f
            G_f[ielem, 2*ielem+3] += -0.5

        # now build P_gam interpolation matrix from DeRham diagram (see 1_opt_pro.ipynb)
        P_gam = np.zeros((nxe_f, nxe_c))
        # not exactly sure how to best interp this, TBD try diff coeffs
        a = 1.0
        # a = 0.75
        for ielem_f in range(nxe_f):
            ielem_c = ielem_f // 2
            P_gam[ielem_f, ielem_c] += a
            if ielem_f % 2 == 0 and ielem_c - 1 >= 0:
                P_gam[ielem_f, ielem_c-1] += 1 - a
            elif ielem_f % 2 == 1 and ielem_c + 1 < nxe_c-1:
                P_gam[ielem_f, ielem_c+1] += 1 - a

        # # need to ensure DeRham system of eqns respects DeRham system of eqns
        # # rewrite DeRham system from G_f * P = P_gam * G_c = B (with B stored now)
        # # then concatenate two new eqns e_1^T * P = 0 and e_{2(N-1)}^T * P = 0 for SS BCs to LHS and RHS of DeRham system
        # B = P_gam @ G_c # full RHS matrix of DeRham system
        # E = np.zeros((2, 2*nx_f))
        # E[0,0] = 1.0 # e_1^T
        # E[1,2*(nx_f-1)] = 1.0 # w_{N-1} = 0
        # # NOTE : didn't include coarse boundary columns in DeRham constraint (for restriction).. think prolong mostly only matters?
        # # we'll see about that..
        # A = np.concatenate([G_f, E], axis=0)
        # B2 = np.concatenate([B, np.zeros((2, 2*nx_c))], axis=0)
        # # now new system of eqns is A * P = B2 for DeRham + BCs
        # assert not self.clamped # since I didn't implement clamped BC in DeRham system yet (easy to add just add two more eqns to E and B2)

        # # import matplotlib.pyplot as plt
        # # plt.imshow(P_gam)
        # # plt.show()

        # # print(f"{G_f.shape=} {G_c.shape=} {P0.shape=} {P_gam.shape=}")
        
        # # now construct the locking-aware prolong from DeRham constraint
        # # G_f * P ~= P_gam * G_c with P ~= P_0 as well and lam lagrange mult there
        # # lam = self.lam
        # lam = 0.1
        # P_0 = P0
        # # P = np.linalg.inv(G_f.T @ G_f + np.eye(2*nx_f) * lam) @ (G_f.T @ P_gam @ G_c + lam * P_0)
        # # P update with DeRham + fine BCs
        # P = np.linalg.inv(A.T @ A + lam * np.eye(2*nx_f)) @ (A.T @ B2 + lam * P_0)

        # # # re-enforce BCs? No this breaks DeRham system then (need to include as extra eqns in DeRham system to ensure respects BCs too in P update)
        # # P[[0, 2*(nx_f - 1)], :] = 0.0      # zero fine boundary rows

        # # P[:, [0, 2*(nx_c - 1)]] = 0.0      # zero coarse boundary cols
        # return P

        # build E correctly (two separate row constraints)
        B = P_gam @ G_c # full RHS matrix of DeRham system
        # E = np.zeros((2, 2*nx_f))
        # E[0, 0] = 1.0
        # E[1, 2*(nx_f-1)] = 1.0

        # A = np.concatenate([G_f, E], axis=0)
        # B2 = np.concatenate([B, np.zeros((2, 2*nx_c))], axis=0)

        # no need for the BC constraint if only changing non-fixed nodes
        A = G_f.copy()
        B2 = B.copy()

        # lam = 0.1
        # lam = 1e-2
        # lam = 1e-3
        lam = self.lam
        P_0 = P0

        # enforce coarse BC columns exactly by eliminating them
        fixed_cols = [0, 2*(nx_c - 1)]          # SS: w left/right
        # if self.clamped: fixed_cols += [1, 2*(nx_c - 1) + 1]

        all_cols = np.arange(2*nx_c, dtype=int)
        free_cols = np.array([j for j in all_cols if j not in set(fixed_cols)], dtype=int)

        B2_free = B2[:, free_cols]
        P0_free = P_0[:, free_cols]

        M = A.T @ A + lam * np.eye(2*nx_f)
        rhs = A.T @ B2_free + lam * P0_free
        P_free = np.linalg.solve(M, rhs)

        P = np.zeros((2*nx_f, 2*nx_c))
        P[:, free_cols] = P_free
        return P
    
    def _locking_aware_prolong_local(self, P0: sp.csr_matrix, nxe_c: int, length: float,
                                    block_nodes: int = 1,
                                    n_sweeps: int = 1,
                                    omega: float = 1.0):
        """
        Locking-aware prolong using ONLY local 1- or 2-node stencils (no global solve).

        Solves (approximately) for P_free:
            (A^T A + lam I) P_free = A^T B2_free + lam P0_free
        using block Gauss-Seidel sweeps with blocks of:
            block_nodes=1 -> 2x2 blocks (w_i, th_i)
            block_nodes=2 -> 4x4 blocks (w_i, th_i, w_{i+1}, th_{i+1})

        Parameters
        ----------
        block_nodes : {1,2}
            Local stencil size in nodes.
        n_sweeps : int
            Number of forward GS sweeps (typically 3-10 is enough).
        omega : float
            Relaxation (1.0 = GS; <1 under-relax; >1 over-relax cautiously).

        Returns
        -------
        P : (2*nx_f, 2*nx_c) dense ndarray
        """

        # -----------------------------
        # sizes
        # -----------------------------
        nx_c = nxe_c + 1
        nxe_f = 2 * nxe_c
        nx_f = nxe_f + 1

        dx_c = length / nxe_c
        dx_f = length / nxe_f
        a_c = 1.0 / dx_c
        a_f = 1.0 / dx_f

        # convert P0 to dense (fine x coarse)
        P0 = P0.toarray()

        # -----------------------------
        # build G_c, G_f (dense; small-ish)
        # -----------------------------
        G_c = np.zeros((nxe_c, 2 * nx_c))
        for ielem in range(nxe_c):
            G_c[ielem, 2 * ielem + 0] += -a_c
            G_c[ielem, 2 * ielem + 1] += -0.5
            G_c[ielem, 2 * ielem + 2] += +a_c
            G_c[ielem, 2 * ielem + 3] += -0.5

        G_f = np.zeros((nxe_f, 2 * nx_f))
        for ielem in range(nxe_f):
            G_f[ielem, 2 * ielem + 0] += -a_f
            G_f[ielem, 2 * ielem + 1] += -0.5
            G_f[ielem, 2 * ielem + 2] += +a_f
            G_f[ielem, 2 * ielem + 3] += -0.5

        # -----------------------------
        # build P_gam (same as your current choice)
        # -----------------------------
        P_gam = np.zeros((nxe_f, nxe_c))
        a = 1.0
        for ielem_f in range(nxe_f):
            ielem_c = ielem_f // 2
            P_gam[ielem_f, ielem_c] += a
            if ielem_f % 2 == 0 and ielem_c - 1 >= 0:
                P_gam[ielem_f, ielem_c - 1] += 1 - a
            elif ielem_f % 2 == 1 and ielem_c + 1 < nxe_c - 1:
                P_gam[ielem_f, ielem_c + 1] += 1 - a

        # -----------------------------
        # build A, B2 (DeRham + fine BC rows)
        # -----------------------------
        B = P_gam @ G_c  # (nxe_f, 2*nx_c)

        # E = np.zeros((2, 2 * nx_f))
        # E[0, 0] = 1.0
        # E[1, 2 * (nx_f - 1)] = 1.0

        # A = np.concatenate([G_f, E], axis=0)
        # B2 = np.concatenate([B, np.zeros((2, 2 * nx_c))], axis=0)

        # no need for the BC constraint if only changing non-fixed nodes
        A = G_f.copy()
        B2 = B.copy()

        lam = float(self.lam)

        # -----------------------------
        # eliminate coarse BC columns (same as you already do)
        # -----------------------------
        fixed_cols = [0, 2 * (nx_c - 1)]  # SS: w left/right
        all_cols = np.arange(2 * nx_c, dtype=int)
        free_cols = np.array([j for j in all_cols if j not in set(fixed_cols)], dtype=int)

        B2_free = B2[:, free_cols]
        P0_free = P0[:, free_cols]

        # -----------------------------
        # build sparse M and RHS (still cheap; no factorization)
        # -----------------------------
        Asp = sp.csr_matrix(A)
        I = sp.eye(2 * nx_f, format="csr")
        M = (Asp.T @ Asp) + lam * I                # (2*nx_f, 2*nx_f) sparse SPD banded
        RHS = (Asp.T @ B2_free) + lam * P0_free    # (2*nx_f, n_free) dense ndarray

        # initial guess: baseline prolong
        X = P0_free.copy()  # shape: (2*nx_f, n_free)

        # -----------------------------
        # precompute block structure (local stencil)
        # -----------------------------
        if block_nodes not in (1, 2):
            raise ValueError("block_nodes must be 1 or 2")

        dofs_per_node = 2
        block_size = dofs_per_node * block_nodes

        n_dofs = 2 * nx_f
        n_blocks = int(np.ceil(n_dofs / block_size))

        # helper: slice indices for block b
        def blk_slice(b):
            i0 = b * block_size
            i1 = min(n_dofs, (b + 1) * block_size)
            return slice(i0, i1)

        # convert to CSR for fast row slicing
        M = M.tocsr()

        # cache each diagonal block inverse (tiny)
        Dinv = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            Db = M[sl, sl].toarray()
            # tiny regularization safeguard (shouldn't be needed, but helps numerical stability)
            Db = 0.5 * (Db + Db.T)
            Dinv.append(np.linalg.inv(Db))

        # -----------------------------
        # block Gauss–Seidel sweeps
        # -----------------------------
        for _ in range(n_sweeps):
            # forward sweep
            for b in range(n_blocks):
                sl = blk_slice(b)

                # residual for this block: r_b = RHS_b - sum_{q != b} M_bq X_q
                # since M is banded/local, we can just compute with sparse matvec on the slice:
                # r_b = RHS[sl] - M[sl,:] @ X + M[sl,sl] @ X[sl]
                Mb_all = M[sl, :]                 # sparse
                r = RHS[sl, :] - Mb_all @ X       # dense (block_size x n_free)
                # add back diagonal contribution (because we subtracted it above)
                r += (M[sl, sl] @ X[sl, :])

                # local block solve and relaxation
                dX = Dinv[b] @ r
                X[sl, :] = X[sl, :] + omega * (dX - X[sl, :])

        # reinsert into full P
        P = np.zeros((2 * nx_f, 2 * nx_c))
        P[:, free_cols] = X
        # fixed coarse BC cols remain identically zero
        return P

    def _energy_smooth_prolong_local(self, P0: sp.csr_matrix, nxe_c: int,
                                    block_nodes: int = 1,
                                    n_sweeps: int = 1,
                                    omega: float = 1.0,
                                    tau: float = 1.0):
        """
        Energy smoothing of prolongator using the trace objective:

            E(P) = tr(P^T K_f P)

        Gradient: dE/dP = 2 K_f P.

        Local preconditioned gradient descent / Richardson step:
            P <- P - tau * D^{-1} (K_f P)

        where D is the block-diagonal of K_f with 1- or 2-node blocks.

        Parameters
        ----------
        P0 : csr_matrix
            Baseline prolong, shape (n_f, n_c).
        nxe_c : int
            Coarse elements, used to eliminate coarse BC columns (same convention as your locking routine).
        block_nodes : {1,2}
            1 -> 2x2 blocks (node-local), 2 -> 4x4 blocks (two-node local).
        n_sweeps : int
            Number of smoothing sweeps (each sweep applies one local-preconditioned gradient step).
        omega : float
            Relaxation on the block update (like your locking GS): X <- X + omega*(Xnew - X).
        tau : float
            Step size in the gradient step. Often tau ~ 0.5–1.0 works; if it oversmooths, reduce tau.

        Returns
        -------
        P : dense ndarray, shape (n_f, n_c)
        """

        # fine operator
        K = self._kmat_cache[nxe_c]
        n_f = K.shape[0]

        # coarse column handling (same as your locking routine)
        nx_c = nxe_c + 1
        n_c = 2 * nx_c
        fixed_cols = [0, 2 * (nx_c - 1)]  # SS: w left/right
        all_cols = np.arange(n_c, dtype=int)
        free_cols = np.array([j for j in all_cols if j not in set(fixed_cols)], dtype=int)

        # dense working array
        P0 = P0.toarray()
        X = P0[:, free_cols].copy()   # (n_f, n_free)

        # block structure
        if block_nodes not in (1, 2):
            raise ValueError("block_nodes must be 1 or 2")

        dofs_per_node = 2
        block_size = dofs_per_node * block_nodes
        n_blocks = int(np.ceil(n_f / block_size))

        def blk_slice(b):
            i0 = b * block_size
            i1 = min(n_f, (b + 1) * block_size)
            return slice(i0, i1)

        # cache block-diagonal inverses of K (local)
        Dinv = []
        for b in range(n_blocks):
            sl = blk_slice(b)
            Kb = K[sl, sl].toarray()
            Kb = 0.5 * (Kb + Kb.T)
            # small safeguard in case Kb is singular-ish locally (optional)
            # eps = 1e-14 * np.trace(Kb) / max(1, Kb.shape[0])
            # Kb = Kb + eps * np.eye(Kb.shape[0])
            Dinv.append(np.linalg.inv(Kb))

        # print(f"{K.shape=} {X.shape=}")

        # smoothing sweeps
        for _ in range(n_sweeps):
            # compute global gradient piece once: G = K X
            # (sparse-dense multiply; gives dense n_f x n_free)
            GX = K @ X

            # local block-preconditioned update
            for b in range(n_blocks):
                sl = blk_slice(b)

                # preconditioned gradient step on this block:
                # Xnew_b = X_b - tau * Dinv_b * (GX_b)
                Xnew = X[sl, :] - tau * (Dinv[b] @ GX[sl, :])

                # relax like your locking routine
                X[sl, :] = X[sl, :] + omega * (Xnew - X[sl, :])

        # print("ENERGY SMOOTH PROLONG LOCAL")

        # reinsert into full P
        P = np.zeros((n_f, n_c))
        P[:, free_cols] = X
        return P

    
    def prolongate(self, coarse_disp, length: float):
        """
        Prolongation using cached P0 = kron(P1, I_ndof_per_node).
        Assumes coarse_disp ordering [w0, th0, w1, th1, ...] (ndof_per_node=2 by default).
        """
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 2
        nelems_coarse = nnodes_coarse - 1

        P0 = self._build_P0_wth(nelems_coarse, ndof_per_node=2)
        if self.prolong_mode == 'global-locking':
            P = self._locking_aware_prolong(P0, nelems_coarse, length)
        elif self.prolong_mode == 'local-locking':
            P = self._locking_aware_prolong_local(P0, nelems_coarse, length, block_nodes=2, n_sweeps=1, omega=1.0)
        elif self.prolong_mode == 'global-kmat':
            P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)
        else:
            P = P0
        fine_disp = P @ coarse_disp
        return np.asarray(fine_disp).ravel()


    def restrict_defect(self, fine_defect, length: float):
        """
        Restriction using cached P0^T (Galerkin restriction).
        """
        ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // 2
        nelems_fine = nnodes_fine - 1

        nelems_coarse = nelems_fine // 2

        P0 = self._build_P0_wth(nelems_coarse, ndof_per_node=2)
        if self.prolong_mode == 'global-locking':
            P = self._locking_aware_prolong(P0, nelems_coarse, length)
        elif self.prolong_mode == 'local-locking':
            P = self._locking_aware_prolong_local(P0, nelems_coarse, length, block_nodes=2, n_sweeps=1, omega=1.0)
        elif self.prolong_mode == 'global-kmat':
            P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=8, omega=1.0, tau=0.5)
        else:
            P = P0

        coarse_defect = P.T @ fine_defect
        return np.asarray(coarse_defect).ravel()