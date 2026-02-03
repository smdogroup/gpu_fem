import numpy as np
from .basis import third_order_quadrature, zero_order_quadrature, second_order_quadrature
from .basis import get_iga2_basis

# based on this paper AIG, https://www.sciencedirect.com/science/article/pii/S0045782525005997
# "Asymptotically accurate and geometric locking-free finite element implementation of a refined shell theory"

class AsymptoticIsogeometricTimoshenkoElementV2:
    """AIG-scaled Timoshenko: x' = x/h and theta' = h*theta (DOF stores theta')"""
    def __init__(self, reduced_integrated: bool = False):
        self.dof_per_node = 2              # [w, theta_prime]
        self.nodes_per_elem = 3
        self.reduced_integrated = reduced_integrated
        self.clamped = True
        self.ORDER = 2

    def get_kelem(self, E: float, nu: float, thick: float, elem_length: float,
                  left_bndry: bool, right_bndry: bool):

        pts, weights = second_order_quadrature()
        if self.reduced_integrated:
            shear_pts, shear_wts = zero_order_quadrature()
        else:
            shear_pts, shear_wts = third_order_quadrature()

        kelem = np.zeros((6, 6))
        opsigma = 1.0

        # Physical stiffnesses (start as you requested)
        EI = (thick**3) * E / 12.0
        ks = 5.0 / 6.0
        G  = E / (2.0 * (1.0 + nu))
        ksGA = ks * G * thick

        # ---- AIG scaling: integrate over x' = x/thick ----
        # With xi in [0,1], x = elem_length * xi so x' = (elem_length/thick) * xi
        Jp = elem_length / thick   # dx'/dxi

        # After scaling:
        # bending prefactor: EI / thick^3 = E/12
        # shear prefactor:   ksGA / thick  = ks*G
        Cb = (EI / (thick**3)) * opsigma   # = E/12 * opsigma
        Cs = (ksGA / thick)                # = ks*G

        # --------------------
        # Shear: ∫ Cs (w_{,x'} - theta')^2 dx'
        # --------------------
        for xi, wt in zip(shear_pts, shear_wts):
            xi = 0.5 * (xi + 1.0)  # map [-1,1] -> [0,1]
            N, dN = get_iga2_basis(xi, left_bndry, right_bndry)

            # d/dx' = (1/Jp) d/dxi
            dNdxp = dN / Jp

            c = Cs * wt * Jp  # dx' = Jp dxi

            for i in range(3):
                for j in range(3):
                    # (w')^2 block
                    kelem[i, j] += c * dNdxp[i] * dNdxp[j]
                    # - w' * theta' blocks
                    kelem[i, 3 + j] -= c * dNdxp[i] * N[j]
                    kelem[3 + i, j] -= c * N[i] * dNdxp[j]
                    # (theta')^2 block
                    kelem[3 + i, 3 + j] += c * N[i] * N[j]

        # --------------------
        # Bending: ∫ Cb (theta'_{,x'})^2 dx'
        # --------------------
        for xi, wt in zip(pts, weights):
            xi = 0.5 * (xi + 1.0)
            N, dN = get_iga2_basis(xi, left_bndry, right_bndry)
            dNdxp = dN / Jp

            c = Cb * wt * Jp
            for i in range(3):
                for j in range(3):
                    kelem[3 + i, 3 + j] += c * dNdxp[i] * dNdxp[j]

        # reorder [w1,w2,w3,thp1,thp2,thp3] -> [w1,thp1,w2,thp2,w3,thp3]
        new_order = np.array([0, 3, 1, 4, 2, 5])
        kelem = kelem[new_order, :][:, new_order]
        return kelem


    def get_felem(self, mag, elem_length, left_bndry:bool, right_bndry:bool, xbnd:list):
        """get element load vector"""
        J = elem_length # not half because xi in [0, 1] for IGA
        # pts, wts = third_order_quadrature()
        pts, wts = second_order_quadrature()
        x0, x1 = xbnd[0], xbnd[1]
        felem = np.zeros(6)
        for xi, wt in zip(pts, wts):
            xi = 0.5 * (xi + 1)
            xval = x0 * (1.0 - xi) + x1 * xi
            load_mag = mag(xval)
            N, dN = get_iga2_basis(xi, left_bndry, right_bndry)
            for i in range(3):
                felem[2 * i] += load_mag  * wt * J * N[i]
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

        # fine_disp /= fine_weights

        # apply bcs again, no need same answer either way
        fine_disp[0] = 0.0
        fine_disp[-2] = 0.0

        if self.clamped:
            fine_disp[1] = 0.0
            fine_disp[-1] = 0.0
            
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
            coarse_defect[idof::dpn] += np.dot(R, fine_defect[idof::dpn])

        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0
        if self.clamped:
            coarse_defect[1] = 0.0
            coarse_defect[-1] = 0.0

        return coarse_defect