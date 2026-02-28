

import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose


class HellingerReissnerAnsatzElement:
    """
    fully integrated timoshenko beam element using Hellinger-Reissner Ansatz (with strain unknown
    removed by static condensation) see https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.5766?saml_referrer
    """

    def __init__(
        self, 
        schur_complement:bool=False,
        hra_method:str="strain"
    ):              
        self.dof_per_node = 2
        # self.dof_per_node = 3
        self.full_dof_per_node = 3 # for schur complement global assembler
        self.nodes_per_elem = 2
        self.schur_complement = schur_complement # static condensation
        self.clamped = True
        self.hra_method = hra_method

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()

        kelem = np.zeros((6, 6))
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


        # now perform static condensation (Schur complement to eliminate ws unknown)
        if self.schur_complement:
            # remove the gamma DOF from the system
            Kaa = kelem[:4, :][:,:4]
            Kab = kelem[:4,:][:,4:]
            Kba = kelem[4:,:][:,:4]
            Kbb = kelem[4:,:][:,4:]

            # kelem = Kaa - Kab @ np.linalg.inv(Kbb) @ Kba
            X = np.linalg.solve(Kbb, Kba)
            kelem = Kaa - Kab @ X

            print(f"kelem after condensation / schur complement")
            import matplotlib.pyplot as plt
            plt.imshow(np.log(np.abs(kelem + 1e-14)) * np.sign(kelem) ) 
            plt.show()

            new_order = np.array([0, 2, 1, 3])
            kelem = kelem[new_order, :][:, new_order]
            return kelem
        else:
            new_order = np.array([0, 2, 4, 1, 3, 5])
            kelem = kelem[new_order, :][:, new_order]
            return kelem

    def get_felem(self, mag, elem_length):
        """get element load vector"""
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()

        # dpn = self.dof_per_node
        dpn = self.full_dof_per_node

        felem = np.zeros(2*dpn)
        for xi, wt in zip(pts, wts):
            psi_val = [lagrange(i, xi) for i in range(2)]
            for i in range(2):
                felem[dpn * i] += mag * wt * J * psi_val[i]

        # felem *= -1.0
        return felem
    
    def prolongate(self, coarse_disp, length:float):
        # assume coarse disp is for half as many elements

        # coarse size
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 2
        nelems_coarse = nnodes_coarse - 1
        # coarse_xscale = self.L / nelems_coarse

        # fine size
        nelems_fine = 2 * nelems_coarse 
        # assert(nelems_fine == self.num_elements)
        nnodes_fine = nelems_fine + 1
        ndof_fine = 2 * nnodes_fine

        # allocate final array
        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # loop through coarse elements
        for ielem_c in range(nelems_coarse):

            # get the coarse element DOF
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
            # interpolate the w DOF first using FEA basis
            # start_inode_c = ielem_c
            # start_inode_f = 2 * ielem_c
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i

                w, th = interp_lagrange(xi, coarse_elem_disps)

                fine_disp[2 * inode_f] += w
                fine_disp[2 * inode_f + 1] += th

                fine_weights[2 * inode_f] += 1.0
                fine_weights[2 * inode_f + 1] += 1.0

        # normalize by fine weights now
        fine_disp /= fine_weights

        # apply bcs..
        fine_disp[0] = 0.0
        fine_disp[-2] = 0.0

        if self.clamped:
            fine_disp[1] = 0.0
            fine_disp[-1] = 0.0

        return fine_disp

    def restrict_defect(self, fine_defect, length:float):
        # from fine defect to this assembler as coarse defect

        # fine size
        ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // 2
        nelems_fine = nnodes_fine - 1

        # coarse size
        nelems_coarse = nelems_fine // 2
        # assert(nelems_coarse == self.num_elements)
        nnodes_coarse = nelems_coarse + 1
        ndof_coarse = 2 * nnodes_coarse

        # allocate final array
        coarse_defect = np.zeros(ndof_coarse)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # compute first the fine weights (I do this better way on GPU).. and other codes, this is lightweight implementation, don't care here
        for ielem_c in range(nelems_coarse):
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                fine_weights[2 * inode_f] += 1.0
                fine_weights[2 * inode_f + 1] += 1.0

        # begin by apply bcs to fine defect in (usually not necessary)
        fine_defect[0] = 0.0
        fine_defect[1] = 0.0
        fine_defect[-2] = 0.0
        fine_defect[-1] = 0.0

        # loop through coarse elements to compute restricted defect
        for ielem_c in range(nelems_coarse):
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                nodal_in = fine_defect[2 * inode_f : (2 * inode_f + 2)] / fine_weights[2 * inode_f : (2 * inode_f + 2)]
                coarse_out = interp_lagrange_transpose(xi, nodal_in)
                coarse_defect[coarse_elem_dof] += coarse_out

            
        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0
        
        if self.clamped:        
            coarse_defect[1] = 0.0
            coarse_defect[-1] = 0.0

        return coarse_defect