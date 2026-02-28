

import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose
import scipy.sparse as sp

# subclass of this so same multigrid operators
from .tsp_elem import TimoshenkoElement_OptProlong

class HellingerReissnerAnsatzElement(TimoshenkoElement_OptProlong):
    """
    fully integrated timoshenko beam element using Hellinger-Reissner Ansatz (with strain unknown
    removed by static condensation) see https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.5766?saml_referrer
    """

    def __init__(
        self, 
        schur_complement:bool=False,
        hra_method:str="strain-int",
        prolong_mode:str='energy', 
        lam:float=1e-2
    ):              
        self.dof_per_node = 2 if schur_complement else 3
        # self.dof_per_node = 2
        # self.dof_per_node = 3
        self.full_dof_per_node = 3 # for schur complement global assembler
        self.nodes_per_elem = 2
        self.schur_complement = schur_complement # static condensation
        self.clamped = True
        self.hra_method = hra_method

        TimoshenkoElement_OptProlong.__init__(self, prolong_mode=prolong_mode, lam=lam)

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()

        kelem = np.zeros((6, 6)) if not self.hra_method == 'strain-int' else np.zeros((5,5))
        EI = E * thick**3 / 12.0
        ks = 5.0 / 6.0
        G = E / 2.0 / (1 + nu)
        ksGA = ks * G * thick
        J = elem_length / 2.0 # dx/dxi jacobian

        # ksGA *= 8.0
        # ksGA *= 4.0
        ksGA *= 1.0

        # trv shear energy
        for xi, wt in zip(pts, weights):
            # basis vecs
            psi_val = [lagrange(i, xi) for i in range(2)]
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]

            for i in range(2):
                for j in range(2):

                    if self.hra_method == 'disp': 
                        # extra shear disp unknown makes static condensation singular

                        # w, ws block of Kelem
                        c_shear = ksGA * wt * J
                        kelem[i,4 + j] -= c_shear * dpsi[i] * dpsi[j]
                        kelem[4 + i, j] -= c_shear * dpsi[i] * dpsi[j]

                        # theta, ws block of Kelem
                        kelem[2 + i,4 + j] -= c_shear * psi_val[i] * dpsi[j]
                        kelem[4 + i, 2 + j] -= c_shear * dpsi[i] * psi_val[j]

                        # ws, ws block of Kelem
                        kelem[4 + i, 4 + j] += c_shear * dpsi[i] * dpsi[j]

                        # th, th block of Kelem
                        kelem[2 + i, 2 + j] -= EI * wt * J * dpsi[i] * dpsi[j]

                    elif self.hra_method == 'strain':
                        # mixed shear STRAIN unknown gamma
                        c_shear = ksGA * wt * J

                        # --------------------------------
                        # w - gamma coupling  : -∫ ksGA gamma w'
                        # --------------------------------
                        kelem[i, 4 + j]     += -c_shear * dpsi[i] * psi_val[j]
                        kelem[4 + i, j]     += -c_shear * psi_val[i] * dpsi[j]

                        # --------------------------------
                        # theta - gamma coupling : +∫ ksGA gamma theta
                        # --------------------------------
                        kelem[2 + i, 4 + j] +=  -c_shear * psi_val[i] * psi_val[j]
                        kelem[4 + i, 2 + j] +=  -c_shear * psi_val[i] * psi_val[j]

                        # --------------------------------
                        # gamma - gamma block (L2 MASS)
                        # --------------------------------
                        kelem[4 + i, 4 + j] += c_shear * psi_val[i] * psi_val[j]

                        # # --------------------------------
                        # # bending block
                        # # --------------------------------
                        kelem[2 + i, 2 + j] -= EI * wt * J * dpsi[i] * dpsi[j]

                    elif self.hra_method == 'strain-int':
                        # interior edge DOF strain

                        c_shear = ksGA * wt * J

                        # NOTE (implementation trick): for 0.5 and 0.25 factors (instead of 1.0 if not 2x2 sum)
                        #   gamma_e is a SINGLE element-interior DOF (constant on the element), so the weak form
                        #   contributions are single-sum in the nodal index only:
                        #       K[w_i, g] += -∫ ksGA * (dN_i/dx) * 1 dx
                        #       K[g, w_j] += -∫ ksGA * 1 * (dN_j/dx) dx
                        #   But this branch sits inside the existing 2×2 (i,j) loops used for nodal–nodal blocks.
                        #   Since the integrand here does NOT depend on the “other” index, the extra dummy loop would
                        #   double-count each term by a factor of 2. We compensate by scaling with 0.5 (and 0.25 for g–g)
                        #   so the net contribution matches the intended single-sum operator.
                        #   This equivalence relies on gamma_e having a constant shape (=1) on the element.

                        # --------------------------------
                        # w - gamma coupling  : -∫ ksGA gamma w'
                        # --------------------------------

                        kelem[i, 4]     += -c_shear * dpsi[i] * 0.5
                        kelem[4, j]     += -c_shear *  0.5 * dpsi[j]

                        # --------------------------------
                        # theta - gamma coupling : +∫ ksGA gamma theta
                        # --------------------------------
                        kelem[2 + i, 4] +=  -c_shear * psi_val[i] *  0.5
                        kelem[4, 2 + j] +=  -c_shear *  0.5 * psi_val[j]

                        # --------------------------------
                        # gamma - gamma block (L2 MASS)
                        # --------------------------------
                        kelem[4, 4] += c_shear * 0.25

                        # # --------------------------------
                        # # bending block
                        # # --------------------------------
                        kelem[2 + i, 2 + j] -= EI * wt * J * dpsi[i] * dpsi[j]

        kelem *= -1
        # import matplotlib.pyplot as plt
        # plt.imshow(kelem)
        # plt.show()

        # check matrix is transpose of itself
        # sym_part = 0.5 * (kelem + kelem.T)
        # asym_part = 0.5 * (kelem - kelem.T)
        # sym_norm = np.linalg.norm(sym_part)
        # asym_norm = np.linalg.norm(asym_part)
        # print(f"{sym_norm=:.4e} {asym_norm=:.4e}")

        if self.schur_complement:
            return kelem
        else:
            assert self.hra_method != 'strain-int'
            new_order = np.array([0, 2, 4, 1, 3, 5]) 
            kelem = kelem[new_order, :][:, new_order]
            return kelem

        # # now perform static condensation (Schur complement to eliminate ws unknown)
        # if self.schur_complement:
        #     # remove the gamma DOF from the system
        #     Kaa = kelem[:4, :][:,:4]
        #     Kab = kelem[:4,:][:,4:]
        #     Kba = kelem[4:,:][:,:4]
        #     Kbb = kelem[4:,:][:,4:]

        #     # kelem = Kaa - Kab @ np.linalg.inv(Kbb) @ Kba
        #     X = np.linalg.solve(Kbb, Kba)
        #     kelem = Kaa - Kab @ X

        #     print(f"kelem after condensation / schur complement")
        #     import matplotlib.pyplot as plt
        #     plt.imshow(np.log(np.abs(kelem + 1e-14)) * np.sign(kelem) ) 
        #     plt.show()

        #     new_order = np.array([0, 2, 1, 3])
        #     kelem = kelem[new_order, :][:, new_order]
        #     return kelem
        # else:
        #     # new_order = np.array([0, 2, 4, 1, 3, 5])
        #     # kelem = kelem[new_order, :][:, new_order]
        #     return kelem

    def get_felem(self, mag, elem_length):
        """get element load vector"""
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()

        dpn = self.dof_per_node
        # dpn = self.full_dof_per_node

        felem = np.zeros(2*dpn)
        for xi, wt in zip(pts, wts):
            psi_val = [lagrange(i, xi) for i in range(2)]
            for i in range(2):
                felem[dpn * i] += mag * wt * J * psi_val[i]

        # felem *= -1.0
        return felem