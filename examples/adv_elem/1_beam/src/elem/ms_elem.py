import numpy as np
from .basis import third_order_quadrature, zero_order_quadrature, second_order_quadrature
from .basis import get_iga2_basis

class MixedShearIsogeometricElement:
    """
    # mixed finite element formulation with shear strain variable, IGA
    
    # https://www.numdam.org/item/M2AN_1997__31_4_517_0/
    # Preconditioning discrete approximations of the Reissner-Mindlin plate model

    # De Rham (cohomology) IGA plate element
    # based on paper: https://www.sciencedirect.com/science/article/abs/pii/S0045782511003215
    # "An isogeometric method for the Reissner-Mindlin plate bending problem" by Veiga

    # see also the thickness-independent Schwarz multigrid smoothers for it in Benzaken et al.,
    # https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf
    # "Multigrid Methods for Isogeometric Thin Plate Discretizations" 

    # multi-patch and domain decomp methods from this book on mixed variational FEA Methods
    # https://dmvn.mexmat.net/content/books/brezzi-fortin-mixed-and-hybrid-finite-elements-methods.pdf

    # except here I'm just doing a beam case..
    """

    # 2nd order IGA with De Rham mixed variational formulation
    def __init__(self):              
        self.dof_per_node = 3
        self.nodes_per_elem = 3
        self.clamped = True
        self.ORDER = 2 # 2nd order IGA

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float, left_bndry:bool, right_bndry:bool):
        # quadratic element with 2 DOF per node
        pts, weights = third_order_quadrature()
        # pts, weights = second_order_quadrature()

        kelem = np.zeros((9, 9))
        EI = E * thick**3 / 12.0
        # EI2 = EI / thick**2
        ks = 5.0 / 6.0
        G = E / 2.0 / (1 + nu)
        ksGA = ks * G * thick
        J = elem_length # dx/dxi jacobian

        # bending energy = int EI * th_{,x} * dth_{,x} dx
        for _xi, _wt in zip(pts, weights):
            # IGA quadpts are [0,1] not [-1,1] so slight adjustment here
            xi = 0.5 * (_xi + 1)
            wt = 0.5 * _wt # for smaller window of xi
            N, dN = get_iga2_basis(xi, left_bndry, right_bndry)
            dN /= J

            # from ref https://www.numdam.org/item/M2AN_1997__31_4_517_0.pdf
            # eqns 1.4-1.6 give the system of eqns of new mixed variational weak form
            for i in range(3):
                for j in range(3):
                    c_shear = ksGA * wt * J * thick**2 # boosted now...

                    # dw row eqn is int gamma * dw' - g * dw [dx]
                    kelem[i, 6+j] += c_shear * N[i] * dN[j]

                    # dth row eqn is int EI * th_{,x} * dth_{,x} - gamma * dth [dx]
                    kelem[3+i, 3+j] += EI * wt * J * dN[i] * dN[j]
                    kelem[3+i, 6+j] -= c_shear * N[i] * N[j]

                    # dgam equation
                    kelem[6+i, j] -= ksGA * wt * J * N[i] * dN[j] # (dgam, w)
                    kelem[6+i, 3+j] += ksGA * wt * J * N[i] * N[j] # (dgam, th)
                    kelem[6+i, 6+j] -= ksGA * wt * J * thick**2 * N[i] * N[j] # (dgam,gam)
                    

        # import matplotlib.pyplot as plt
        # plt.imshow(np.log(np.abs(kelem) + 1e-14))
        # plt.show()
                    

        # change order from [w1,w2,w3,th1,th2,th3,gam1,gam2,gam3] => [w1, th1, gam1, w2, th2, gam2, w3, th3, gam3]
        new_order = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
        kelem = kelem[new_order, :][:, new_order]
        return kelem

    def get_felem(self, mag, elem_length, left_bndry:bool, right_bndry:bool, xbnd:list):
        """get element load vector"""
        J = elem_length # not half because xi in [0, 1] for IGA
        pts, wts = second_order_quadrature()
        # pts, wts = third_order_quadrature()
        x0, x1 = xbnd[0], xbnd[1]
        felem = np.zeros(9)
        for _xi, _wt in zip(pts, wts):
            xi = 0.5 * (_xi + 1)
            wt = 0.5 * _wt # for smaller window of xi
            xval = x0 * (1.0 - xi) + x1 * xi
            load_mag = mag(xval)
            N, _ = get_iga2_basis(xi, left_bndry, right_bndry)
            for i in range(3):
                felem[3 * i + 0] += load_mag * wt * J * N[i]
        return felem
    
    def _build_restriction_matrix(self, nxe_c):
        # restriction operator or P^T global
        n_coarse = nxe_c + 2
        nxe_f = 2 * nxe_c
        n_fine = nxe_f + 2
        R = np.zeros((n_coarse, n_fine))
        counts = 1e-14 * np.ones((n_coarse, n_fine))

        # see asym_iga/_python_demo/2_iga/4_multigrid_oned.ipynb
        for ielem_c in range(nxe_c):
            # left half elem
            left_felem = 2 * ielem_c
            l_mat = np.array([
                [0.75, 0.25, 0.0],
                [0.25, 0.75, 0.75],
                [0.0, 0.0, 0.25],
            ])
            if ielem_c == 0:
                l_mat[0,:2] = np.array([1.0, 0.5])
                l_mat[1,:2] = np.array([0.0, 0.5])
            if ielem_c == nxe_c - 1:
                l_mat[1, 2] = 0.5
                l_mat[2, 2] = 0.5

            # print(F"{l_mat=} {l_mat.shape=}")

            l_nz_mat = l_mat / (l_mat + 1e-14)
            R[ielem_c:(ielem_c+3), left_felem:(left_felem+3)] += l_mat
            counts[ielem_c:(ielem_c+3), left_felem:(left_felem+3)] += l_nz_mat

            # right half elem
            right_felem = 2 * ielem_c + 1
            r_mat = np.array([
                [0.25, 0.0, 0.0],
                [0.75, 0.75, 0.25],
                [0.0, 0.25, 0.75],
            ])
            if ielem_c == 0:
                r_mat[0,0] = 0.5
                r_mat[1,0] = 0.5
            if ielem_c == nxe_c - 1:
                r_mat[1, 1:] = np.array([0.5, 0.0])
                r_mat[2, 1:] = np.array([0.5, 1.0])
            r_nz_mat = r_mat / (r_mat + 1e-14)
            R[ielem_c:(ielem_c+3), right_felem:(right_felem+3)] += r_mat
            counts[ielem_c:(ielem_c+3), right_felem:(right_felem+3)] += r_nz_mat

        # normalize it by weights added into each spot?
        R /= counts
        # print(f"{R=}")
        return R
    
    def prolongate(self, coarse_disp:np.ndarray, length:float):
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // self.dof_per_node
        nelems_coarse = nnodes_coarse - 2
        nelems_fine = 2 * nelems_coarse
        nnodes_fine = nelems_fine + 2
        ndof_fine = nnodes_fine * self.dof_per_node

        R = self._build_restriction_matrix(nelems_coarse)
        P = R.T

        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # to do row-sum norm on P (since P^T is properly normalized but P is not?)
        dpn = self.dof_per_node
        for idof in range(dpn):
            fine_disp[idof::dpn] += np.dot(P, coarse_disp[idof::dpn])
            fine_weights[idof::dpn] += np.dot(P, np.ones(nnodes_coarse))

        fine_disp /= fine_weights

        # apply bcs again, no need same answer either way
        fine_disp[0] = 0.0
        fine_disp[-3] = 0.0

        if self.clamped:
            fine_disp[1] = 0.0
            fine_disp[-2] = 0.0
            
        return fine_disp
    
    def restrict_defect(self, fine_defect:np.ndarray, length:float):
        ndof_coarse = ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // self.dof_per_node
        nelems_fine = nnodes_fine - 2

        nelems_coarse = nelems_fine // 2
        nnodes_coarse = nelems_coarse + 2
        dpn = self.dof_per_node
        ndof_coarse = dpn * nnodes_coarse

        R = self._build_restriction_matrix(nelems_coarse)

        coarse_defect = np.zeros(ndof_coarse)
        dpn = self.dof_per_node
        for idof in range(dpn):
        # for idof in range(2): # not all DOF
            coarse_defect[idof::dpn] += np.dot(R, fine_defect[idof::dpn])

        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-3] = 0.0

        if self.clamped:
            coarse_defect[1] = 0.0
            coarse_defect[-2] = 0.0

        return coarse_defect
