import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import interp_lagrange, interp_lagrange_transpose


class TimoshenkoElement:
    """fully integrated timoshenko beam element (experiences locking"""
    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 2
        self.nodes_per_elem = 2
        self.reduced_integrated = reduced_integrated
        self.clamped = True

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