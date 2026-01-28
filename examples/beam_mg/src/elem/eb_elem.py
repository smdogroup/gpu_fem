import numpy as np
from .basis import second_order_quadrature, hermite_cubic_hess, hermite_cubic
from .basis import interp_hermite_disp, interp_lagrange_rotation
# from .basis import restrict_hermite_disp, restrict_lagrange_rotation
from .basis import interp_hermite_disp_transpose, interp_lagrange_rotation_transpose

class EulerBernoulliElement:
    def __init__(self):
        self.dof_per_node = 2
        self.nodes_per_elem = 2
        self.clamped = True

    def get_kelem(self, E:float, nu:float, thick:float, elem_length:float):
        # quadratic element with 2 DOF per node
        pts, weights = second_order_quadrature()
        xscale = elem_length/2.0 # to convert from xi to x coord
        kelem = np.zeros((4, 4))
        EI = E * thick**3 / 12.0
        for xi, wt in zip(pts, weights):
            for row in range(4):
                row_scale = xscale if row % 2 == 1 else 1.0
                row_xx = hermite_cubic_hess(row, xi) / xscale**2 * row_scale
                for col in range(4):
                    col_scale = xscale if col % 2 == 1 else 1.0
                    col_xx = hermite_cubic_hess(col, xi) / xscale**2 * col_scale

                    kelem[row, col] += EI * wt * xscale * row_xx * col_xx
        return kelem
    
    def get_felem(self, mag:float, elem_length:float):
        pts, weights = second_order_quadrature()
        xscale = elem_length/2.0 # to convert from xi to x coord
        felem = np.zeros(4)
        for xi, wt in zip(pts, weights):
            for row in range(4):
                row_scale = xscale if row % 2 == 1 else 1.0
                row_val = row_scale * hermite_cubic(row, xi)
                felem[row] += mag * wt * xscale * row_val
        return felem    

    def prolongate(self, coarse_disp:np.ndarray, length:float):
        # coarse size
        ndof_coarse = coarse_disp.shape[0]
        nnodes_coarse = ndof_coarse // 2
        nelems_coarse = nnodes_coarse - 1
        coarse_elem_length = length / nelems_coarse
        coarse_xscale = coarse_elem_length * 0.5

        # fine size
        nelems_fine = 2 * nelems_coarse 
        # assert(nelems_fine == sel)
        nnodes_fine = nelems_fine + 1
        ndof_fine = 2 * nnodes_fine

        # allocate final array
        fine_disp = np.zeros(ndof_fine)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # loop through coarse elements
        for ielem_c in range(nelems_coarse):

            # get the coarse element DOF
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            # print(F"{ielem_c=} {coarse_elem_dof=}")
            coarse_elem_disps = coarse_disp[coarse_elem_dof]
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                w = interp_hermite_disp(xi, coarse_elem_disps, coarse_xscale)
                # even though we're interpolating from fine to coarse elements with hermite cubic
                # the in-element dw/dxi interp better if scaled down to fine dw/dxi size (hard to explain but works for hermtie cubic and hugely improves conv back to omega = 1 from omega = 0.25 on two grid case)
                
                th = interp_lagrange_rotation(xi, coarse_elem_disps)
                # th = interp_hermite_rotation(xi, coarse_elem_disps, coarse_xscale=coarse_xscale)

                # print(f"{ielem_c=} interp to {inode_f=} with {xi=} and {w=}")

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
        coarse_elem_length = length / nelems_coarse
        coarse_xscale = coarse_elem_length * 0.5

        # allocate final array
        coarse_defect = np.zeros(ndof_coarse)
        fine_weights = np.zeros(ndof_fine) # for global partition of unity normalization

        # compute first the fine weights (I do this better way on GPU).. and other codes, this is lightweight implementation, don't care here
        for ielem_c in range(nelems_coarse):
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                fine_weights[2 * inode_f] += 1.0
                fine_weights[2 * inode_f + 1] += 1.0

        # loop through coarse elements to compute restricted defect
        for ielem_c in range(nelems_coarse):
            coarse_elem_dof = np.array([2 * _node + _dof for _node in [ielem_c, ielem_c + 1] for _dof in range(2)])
            
            # interpolate the w DOF first using FEA basis
            for i, inode_f in enumerate(range(2 * ielem_c, 2 * ielem_c + 3)):
                xi = -1.0 + 1.0 * i
                nodal_in = fine_defect[2 * inode_f : (2 * inode_f + 2)] / fine_weights[2 * inode_f : (2 * inode_f + 2)]
                coarse_out = interp_hermite_disp_transpose(xi, nodal_in[0], coarse_xscale)
                # coarse_out = restrict_hermite_disp(xi, nodal_in[0])

                # lagrange interp of rotations actually works better interestingly (element allows small rotation in each element to min energy, so weird theta interp with hermite grad)
                coarse_out += interp_lagrange_rotation_transpose(xi, nodal_in[1])
                # coarse_out += restrict_lagrange_rotation(xi, nodal_in[1], fine_xscale, coarse_xscale)
                # coarse_out[np.array([1,3])] = 0.0 # temp debug

                coarse_defect[coarse_elem_dof] += coarse_out

            
        # apply bcs.. to coarse defect also
        coarse_defect[0] = 0.0
        coarse_defect[-2] = 0.0

        if self.clamped:
            coarse_defect[1] = 0.0
            coarse_defect[-1] = 0.0

        # coarse_defect[1::2] = 0.0 # DEBUG remove theta load

        return coarse_defect