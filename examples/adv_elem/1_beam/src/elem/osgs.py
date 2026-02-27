import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
import scipy.sparse as sp

class OrthogonalSubGridScaleElement:
    """
    orthogonal sub grid scales to stabilize mesh convergence of the Timoshenko beam element
    see ref: 
        A variational multiscale stabilized finite element formulation for Reissner–Mindlin plates and Timoshenko beams
        https://upcommons.upc.edu/server/api/core/bitstreams/73fce476-07ab-4ea8-84b0-3552f852f9e7/content
    """
    def __init__(self):              
        self.dof_per_node = 4
        self.nodes_per_elem = 2
        self.clamped = True

        # stabilization constants
        self.c1 = 1e-4 # needs to be chosen < 1e-3 in paper? other constants the same?
        # self.c1 = 1e-3
        # self.c1 = 1e-2
        # self.c1 = 1.0
        self.c3 = 12.0
        self.c2 = self.c4 = 1.0

        self._P0_cache = {}
        self._P1_cache = {}

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()

        kelem = np.zeros((8, 8))
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
            for i in range(2):
                for j in range(2):
                    kelem[2+i, 2+j] += (EI - tau_w * ksGA**2) * wt * J * dpsi[i] * dpsi[j]

            # KKT constraints of OSGS  (except there is diag part, so not just saddle point)
            for i in range(2):
                for j in range(2):
                    # (th', xi_w) and transpose block
                    kelem[4+i, 2+j] += ksGA * wt * J * psi_val[i] * dpsi[j]
                    # kelem[2+j, 4+i] += tau_w * ksGA * wt * J * psi_val[i] * dpsi[j]
                    kelem[2+i, 4+j] += tau_w * ksGA * wt * J * psi_val[j] * dpsi[i]

                    # (w', xi_th) and transpose block
                    kelem[i, 6+j] += tau_th * ksGA * wt * J * dpsi[i] * psi_val[j]
                    kelem[6+j, i] += ksGA * wt * J * dpsi[i] * psi_val[j]

                    # - (th, xi_th) and transpose block
                    kelem[2+i, 6+j] -= tau_th * ksGA * wt * J * psi_val[i] * psi_val[j]
                    kelem[6+j, 2+i] -= ksGA * wt * J * psi_val[i] * psi_val[j]

                    # (xi_w, xi_w) block
                    kelem[4+i, 4+j] -= wt * J * psi_val[i] * psi_val[j]

                    # (xi_th, xi_th) block
                    kelem[6+i, 6+j] -= wt * J * psi_val[i] * psi_val[j]


        # import matplotlib.pyplot as plt
        # plt.imshow(kelem)
        # plt.show()

        # change order from [w1,w2,th1,th2,xiw1,xiw2,xith1,xith2] => [w1, th1, xiw1, xith1, w2, th2, xiw2, xith2]
        new_order = np.array([0, 2, 4, 6, 1, 3, 5 ,7])
        kelem = kelem[new_order, :][:, new_order]
        return kelem

    def get_felem(self, mag, elem_length):
        """get element load vector"""
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()
        felem = np.zeros(8)
        for xi, wt in zip(pts, wts):
            psi_val = [lagrange(i, xi) for i in range(2)]
            for i in range(2):
                felem[4 * i] += mag * wt * J * psi_val[i]
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

    def _build_P0_wth(self, nxe_coarse: int, ndof_per_node: int = 4) -> sp.csr_matrix:
        """
        Build initial P0 for [w, th, xiw, xith] using same scalar P1 on each dof,
        and apply BCs at the P0 level selectively.

        Ordering per node: [w, th, xiw, xith]
        SS:    enforce w(0)=w(L)=0 in P0 (rows+cols for w only)
        clamp: enforce w and th at ends in P0
        (xiw/xith left unconstrained by default)
        """
        key = (nxe_coarse, ndof_per_node, self.clamped)
        if hasattr(self, "_P0_cache") and key in self._P0_cache:
            return self._P0_cache[key]

        assert ndof_per_node == 4, "This builder is intended for 4 DOF/node: [w, th, xiw, xith]"

        # scalar prolongation without BCs
        P1 = self._build_P1_scalar(nxe_coarse, apply_bcs=False)

        # lift to block prolongation
        P0 = sp.kron(P1, sp.eye(ndof_per_node, format="csr"), format="csr").tocsr()

        # dimensions for indexing end DOFs
        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1

        # helper to map (node, local_dof) -> global dof index
        def gdof(node: int, ldof: int) -> int:
            return ndof_per_node * node + ldof

        # local dof ids in [w, th, xiw, xith]
        W, TH, XIW, XITH = 0, 1, 2, 3

        # --- Always constrain w at ends ---
        wL_f, wR_f = gdof(0, W), gdof(nf - 1, W)
        wL_c, wR_c = gdof(0, W), gdof(nc - 1, W)

        P0[[wL_f, wR_f], :] = 0.0
        P0[:, [wL_c, wR_c]] = 0.0

        # --- Clamped: also constrain theta at ends ---
        if self.clamped:
            thL_f, thR_f = gdof(0, TH), gdof(nf - 1, TH)
            thL_c, thR_c = gdof(0, TH), gdof(nc - 1, TH)

            P0[[thL_f, thR_f], :] = 0.0
            P0[:, [thL_c, thR_c]] = 0.0

        # If you ALSO want to constrain xiw/xith at ends (often you DON'T), uncomment:
        # for ldof in (XIW, XITH):
        #     fL, fR = gdof(0, ldof), gdof(nf - 1, ldof)
        #     cL, cR = gdof(0, ldof), gdof(nc - 1, ldof)
        #     P0[[fL, fR], :] = 0.0
        #     P0[:, [cL, cR]] = 0.0

        P0.eliminate_zeros()

        if not hasattr(self, "_P0_cache"):
            self._P0_cache = {}
        self._P0_cache[key] = P0
        return P0


    def prolongate(self, coarse_disp, length: float):
        """
        Prolongation using cached P0 for 4 DOF/node: [w, th, xiw, xith].
        coarse_disp ordering: [w0, th0, xiw0, xith0, w1, th1, ...]
        """
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 4
        nelems_coarse = nnodes_coarse - 1

        P = self._build_P0_wth(nelems_coarse, ndof_per_node=4)
        fine_disp = P @ coarse_disp
        return np.asarray(fine_disp).ravel()


    def restrict_defect(self, fine_defect, length: float):
        """
        Galerkin restriction using P0^T for 4 DOF/node: [w, th, xiw, xith].
        """
        ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // 4
        nelems_fine = nnodes_fine - 1

        nelems_coarse = nelems_fine // 2

        P = self._build_P0_wth(nelems_coarse, ndof_per_node=4)
        coarse_defect = P.T @ fine_defect
        return np.asarray(coarse_defect).ravel()