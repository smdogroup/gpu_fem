import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
import scipy.sparse as sp

class AlgebraicSubGridScaleElement:
    """
    algebraic sub grid scales to stabilize mesh convergence of the Timoshenko beam element
    see ref: 
        A variational multiscale stabilized finite element formulation for Reissner–Mindlin plates and Timoshenko beams
        https://upcommons.upc.edu/server/api/core/bitstreams/73fce476-07ab-4ea8-84b0-3552f852f9e7/content
    """
    def __init__(self, prolong_mode="standard", omega=0.7, n_sweeps:int=2):              
        self.dof_per_node = 2
        self.nodes_per_elem = 2
        self.clamped = True
        
        self.prolong_mode = prolong_mode
        self.omega = omega
        self.n_sweeps = n_sweeps

        self._P0_cache = {}
        self._P1_cache = {}
        self._kmat_cache = {}

        # stabilization constants
        self.c1 = self.c3 = 12.0
        self.c2 = self.c4 = 1.0

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()

        kelem = np.zeros((4, 4))
        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / 2.0 / (1 + nu)
        ksGA = ks * G * thick
        J = elem_length / 2.0 # dx/dxi jacobian

        # compute the stabilization parameters
        tau_w = 1.0 / ( self.c3 * ksGA / elem_length**2 + self.c4 * ksGA**2 / EI )
        tau_th = 1.0 / ( self.c1 * EI / elem_length**2 + self.c2 * ksGA )

        # trv shear energy
        for xi, wt in zip(pts, weights):
            # basis vecs
            psi_val = [lagrange(i, xi) for i in range(2)]
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]

            for i in range(2):
                for j in range(2):
                    c_shear = (ksGA - tau_th * ksGA**2) * wt * J
                    kelem[i,j] += c_shear * dpsi[i] * dpsi[j]
                    kelem[i, 2+j] -= c_shear * dpsi[i] * psi_val[j]
                    kelem[2+i, j] -= c_shear * psi_val[i] * dpsi[j]
                    kelem[2+i, 2+j] += c_shear * psi_val[i] * psi_val[j]
            
        # bending energy
        for xi, wt in zip(pts, weights):
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]
            for i in range(2):
                for j in range(2):
                    kelem[2+i, 2+j] += (EI - tau_w * ksGA**2) * wt * J * dpsi[i] * dpsi[j]

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
    
    # def prolongate(self, coarse_disp, length:float):
    #     # assume coarse disp is for half as many elements

    #     # coarse size
    #     ndof_coarse = coarse_disp.shape[0]
    #     nnodes_coarse = ndof_coarse // 2
    #     nelems_coarse = nnodes_coarse - 1
    #     # coarse_xscale = self.L / nelems_coarse

    #     # fine size
    #     nelems_fine = 2 * nelems_coarse 
    #     # assert(nelems_fine == self.num_elements)
    #     nnodes_fine = nelems_fine + 1
    #     ndof_fine = 2 * nnodes_fine

    #     # allocate final array
    #     fine_disp = np.zeros(ndof_fine)
    #     fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

    #     # loop through coarse elements
    #     for ielem_c in range(nelems_coarse):

    #         # get the coarse element DOF
    #         coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
    #         coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
    #         # interpolate the w DOF first using FEA basis
    #         # start_inode_c = ielem_c
    #         # start_inode_f = 2 * ielem_c
    #         for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
    #             xi = -1.0 + 1.0 * i

    #             w, th = interp_lagrange(xi, coarse_elem_disps)

    #             fine_disp[2 * inode_f] += w
    #             fine_disp[2 * inode_f + 1] += th

    #             fine_weights[2 * inode_f] += 1.0
    #             fine_weights[2 * inode_f + 1] += 1.0

    #     # normalize by fine weights now
    #     fine_disp /= fine_weights

    #     # apply bcs..
    #     fine_disp[0] = 0.0
    #     fine_disp[-2] = 0.0

    #     if self.clamped:
    #         fine_disp[1] = 0.0
    #         fine_disp[-1] = 0.0

    #     return fine_disp

    # def restrict_defect(self, fine_defect, length:float):
    #     # from fine defect to this assembler as coarse defect

    #     # fine size
    #     ndof_fine = fine_defect.shape[0]
    #     nnodes_fine = ndof_fine // 2
    #     nelems_fine = nnodes_fine - 1

    #     # coarse size
    #     nelems_coarse = nelems_fine // 2
    #     # assert(nelems_coarse == self.num_elements)
    #     nnodes_coarse = nelems_coarse + 1
    #     ndof_coarse = 2 * nnodes_coarse

    #     # allocate final array
    #     coarse_defect = np.zeros(ndof_coarse)
    #     fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

    #     # compute first the fine weights (I do this better way on GPU).. and other codes, this is lightweight implementation, don't care here
    #     for ielem_c in range(nelems_coarse):
    #         for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
    #             fine_weights[2 * inode_f] += 1.0
    #             fine_weights[2 * inode_f + 1] += 1.0

    #     # begin by apply bcs to fine defect in (usually not necessary)
    #     fine_defect[0] = 0.0
    #     fine_defect[1] = 0.0
    #     fine_defect[-2] = 0.0
    #     fine_defect[-1] = 0.0

    #     # loop through coarse elements to compute restricted defect
    #     for ielem_c in range(nelems_coarse):
    #         coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            
    #         # interpolate the w DOF first using FEA basis
    #         for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
    #             xi = -1.0 + 1.0 * i
    #             nodal_in = fine_defect[2 * inode_f : (2 * inode_f + 2)] / fine_weights[2 * inode_f : (2 * inode_f + 2)]
    #             coarse_out = interp_lagrange_transpose(xi, nodal_in)
    #             coarse_defect[coarse_elem_dof] += coarse_out

            
    #     # apply bcs.. to coarse defect also
    #     coarse_defect[0] = 0.0
    #     coarse_defect[-2] = 0.0
        
    #     if self.clamped:        
    #         coarse_defect[1] = 0.0
    #         coarse_defect[-1] = 0.0

    #     return coarse_defect

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
        # if self.prolong_mode == 'global-locking':
        #     P = self._locking_aware_prolong(P0, nelems_coarse, length)
        # elif self.prolong_mode == 'local-locking':
        #     P = self._locking_aware_prolong_local(P0, nelems_coarse, length, block_nodes=2, n_sweeps=1, omega=1.0)
        if self.prolong_mode == 'standard':
            P = P0
        elif self.prolong_mode == 'global-kmat':
            P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=2, omega=1.0, tau=self.omega)
        
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
        # if self.prolong_mode == 'global-locking':
        #     P = self._locking_aware_prolong(P0, nelems_coarse, length)
        # elif self.prolong_mode == 'local-locking':
        #     P = self._locking_aware_prolong_local(P0, nelems_coarse, length, block_nodes=2, n_sweeps=1, omega=1.0)
        if self.prolong_mode == 'standard':
            P = P0
        elif self.prolong_mode == 'global-kmat':
            P = self._energy_smooth_prolong_local(P0, nelems_coarse, block_nodes=1, n_sweeps=2, omega=1.0, tau=self.omega)
        

        coarse_defect = P.T @ fine_defect
        return np.asarray(coarse_defect).ravel()