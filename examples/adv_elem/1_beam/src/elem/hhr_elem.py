import numpy as np
from .basis import second_order_quadrature, lagrange, lagrange_grad, zero_order_quadrature
from .basis import hermite_cubic_hess, hermite_cubic
from .basis import interp6_hermite_disp, interp6_hermite_disp_transpose
from .basis import interp6_lagrange_rotation, interp6_lagrange_rotation_transpose


# based on Oesterle paper, "A shear deformable, rotation-free isogeometric shell formulation"
# https://www.sciencedirect.com/science/article/pii/S004578251630202X

class HierarchicRotHermiteElement:
    """fully integrated timoshenko beam element (experiences locking in thick shell, see https://www.sciencedirect.com/science/article/pii/S004578251630202X)"""
    def __init__(self, reduced_integrated:bool=False):              
        self.dof_per_node = 3
        self.nodes_per_elem = 2
        self.reduced_integrated = reduced_integrated
        self.clamped = True

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
    # def get_kelem(J, EI, GA, k_shear=5.0/6.0, use_reduced_integration_for_shear=True, schur_complement=False):
        """
        J : Jacobian = L/2 (so L = 2*J)
        EI : bending stiffness
        GA : shear rigidity (A*G)  (we will multiply by k_shear)
        k_shear : shear correction factor (default 5/6)
        """
        kelem = np.zeros((6, 6))
        pts, weights = second_order_quadrature()
        # do need to reduce integrate the shear part still.. otherwise experiences thick plate locking
        if self.reduced_integrated:
            shear_pts, shear_wts = zero_order_quadrature() 
        else:
            shear_pts, shear_wts = second_order_quadrature() # not reduced integrated
        J = elem_length / 2.0
        EI = E * thick**3 / 12.0
        GA = E * thick / 2.0 / (1 + nu)
        k_shear = 5.0 / 6.0

        # bending energy
        for xi, wt in zip(pts, weights):
            # quadrature factor dx = J * dxi
            qfactor = wt * J
            psi_val = [lagrange(i, xi) for i in range(2)]
            dpsi = [lagrange_grad(i, xi, J) for i in range(2)]
            d2phi = [hermite_cubic_hess(i, xi) / J**2 for i in range(4)]

            # assemble bending-related terms (w-w, w-ths, ths-ths)
            # index mapping prior to reorder (0..5): w1, th1, w2, th2, gs1, gs2
            for i in range(4):            
                i_scale = 1.0 if i % 2 == 0 else J
                for j in range(4):
                    j_scale = 1.0 if j % 2 == 0 else J
                    kelem[i,j] += i_scale * j_scale * EI * qfactor * d2phi[i] * d2phi[j]
                
                # coupling w_i with shear DOFs (th_s)
                for j_s in range(2):
                    idx_s = 4 + j_s
                    kelem[i, idx_s] += i_scale * EI * qfactor * d2phi[i] * dpsi[j_s]   # -2 folded later (off-diagonal)
                    kelem[idx_s, i] += i_scale * EI * qfactor * dpsi[j_s] * d2phi[i]
            # th_s-th_s part from EI*(th_s_x)^2
            for i_s in range(2):
                for j_s in range(2):
                    kelem[4+i_s, 4+j_s] += EI * qfactor * dpsi[i_s] * dpsi[j_s]

        # shear penalty kGA * ∫ th_s^2 dx
        for xi, wt in zip(shear_pts, shear_wts): # fully integrated
            qfactor = wt * J
            psi_val = [lagrange(i, xi) for i in range(2)]
            for i_s in range(2):
                for j_s in range(2):
                    kelem[4+i_s, 4+j_s] += k_shear * GA * qfactor * psi_val[i_s] * psi_val[j_s]

        # reorder to your desired ordering: [w1, th1, gam1, w2, th2, gam2]
        new_order = np.array([0, 1, 4, 2, 3, 5])
        kelem = kelem[new_order, :][:, new_order]
        return kelem

    def get_felem(self, mag, elem_length):
        """get element load vector"""
        J = elem_length / 2.0
        pts, wts = second_order_quadrature()
        felem = np.zeros(6)
        for xi, wt in zip(pts, wts):
            for i in range(4):
                i_scale = 1.0 if i % 2 == 0 else J
                felem[i] += mag * wt * J * i_scale * hermite_cubic(i, xi)
        new_order = np.array([0, 1, 4, 2, 3, 5])
        felem = felem[new_order]
        return felem
    
    def prolongate(self, coarse_disp, length:float):
        # assume coarse disp is for half as many elements

        # coarse size
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 3
        nelems_coarse = nnodes_coarse - 1
        coarse_elem_length = length / nelems_coarse
        coarse_xscale = 0.5 * coarse_elem_length

        # fine size
        nelems_fine = 2 * nelems_coarse 
        # assert(nelems_fine == self.num_elements)
        nnodes_fine = nelems_fine + 1
        ndof_fine = 3 * nnodes_fine

        # allocate final array
        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # loop through coarse elements
        for ielem_c in range(nelems_coarse):

            # get the coarse element DOF
            coarse_elem_dof = np.array([3 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(3)])
            coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
            # interpolate the w DOF first using FEA basis
            # start_inode_c = ielem_c
            # start_inode_f = 2 * ielem_c
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i

                w = interp6_hermite_disp(xi, coarse_elem_disps, coarse_xscale=coarse_xscale)
                th, th_s = interp6_lagrange_rotation(xi, coarse_elem_disps)
                # even though we're interpolating from fine to coarse elements with hermite cubic
                # the in-element dw/dxi interp better if scaled down to fine dw/dxi size (hard to explain but works for hermtie cubic and hugely improves conv back to omega = 1 from omega = 0.25 on two grid case)

                fine_disp[3 * inode_f] += w
                fine_disp[3 * inode_f + 1] += th
                fine_disp[3 * inode_f + 2] += th_s

                for _i in range(3):
                    fine_weights[3 * inode_f + _i] += 1.0

        # normalize by fine weights now
        fine_disp /= fine_weights

        # apply bcs..
        fine_disp[0] = 0.0
        fine_disp[-3] = 0.0

        if self.clamped:
            fine_disp[2] = -fine_disp[1]
            fine_disp[-1] = -fine_disp[-2]

        return fine_disp

    def restrict_defect(self, fine_defect, length:float):
        # from fine defect to this assembler as coarse defect

        # fine size
        ndof_fine = fine_defect.shape[0]
        nnodes_fine = ndof_fine // 3
        nelems_fine = nnodes_fine - 1

        # coarse size
        nelems_coarse = nelems_fine // 2
        # assert(nelems_coarse == self.num_elements)
        nnodes_coarse = nelems_coarse + 1
        ndof_coarse = 3 * nnodes_coarse
        coarse_elem_length = length / nelems_coarse
        coarse_xscale = 0.5 * coarse_elem_length

        # allocate final array
        coarse_defect = np.zeros(ndof_coarse)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # compute first the fine weights (I do this better way on GPU).. and other codes, this is lightweight implementation, don't care here
        for ielem_c in range(nelems_coarse):
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                fine_weights[3 * inode_f] += 1.0
                fine_weights[3 * inode_f + 1] += 1.0
                fine_weights[3 * inode_f + 2] += 1.0

        # loop through coarse elements to compute restricted defect
        for ielem_c in range(nelems_coarse):
            coarse_elem_dof = np.array([3 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(3)])
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                nodal_in = fine_defect[3 * inode_f : (3 * inode_f + 3)] / fine_weights[3 * inode_f : (3 * inode_f + 3)]
                coarse_out = interp6_hermite_disp_transpose(xi, nodal_in[0], coarse_xscale=coarse_xscale)
                coarse_out += interp6_lagrange_rotation_transpose(xi, nodal_in[1], nodal_in[2])
                coarse_defect[coarse_elem_dof] += coarse_out

            
        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-3] = 0.0
        
        if self.clamped:
            coarse_defect[2] = -coarse_defect[1]
            coarse_defect[-1] = -coarse_defect[-2]

        return coarse_defect