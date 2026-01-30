import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from _sparse_utils import build_csr_from_conn

class IGAPlateAssembler:
    # for IGA bases or elements

    def __init__(
        self,
        ELEMENT,
        nxe:int,
        E:float=70e9,
        nu:float=0.3,
        thick:float=1.0e-2,
        length:float=1.0,
        width:float=1.0,
        load_fcn=lambda x : 1.0,
        clamped:bool=False,
        split_disp_bc:bool=False,
    ):
        
        self.element = ELEMENT
        self.nxe = nxe
        self.E = E
        self.nu = nu
        self.thick = thick
        self.length = length
        self.width = width
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc
        
        # internal data
        self.kmat = None
        self.force = None
        self.u = None
        
        # square structured grid only
        assert self.element.ORDER == 2 # 2nd order IGA only currently
        self.nnx = self.nxe + 2 
        self.nnodes = self.nnx**2
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.elem_length = self.length / (self.nnodes - 1)
        self.elem_width = self.width / (self.nnodes - 1)
        self.num_elements = self.nxe**2
        
        assert not clamped # for now
        self.clamped = clamped
        # SS BCs then
        self.element.clamped = False
        self.bcs = []
        for inode in range(self.nnodes):
            ix = inode % self.nnx; iy = inode // self.nnx
            if ix == 0 or iy == 0 or ix == self.nnx - 1 or iy == self.nnx - 1:
                self.bcs += [inode]

        # ELEMENT CONN
        self.conn = []
        for ielem in range(self.num_elements):
            ixe = ielem % self.nxe; iye = ielem // self.nxe
            loc_conn = []
            for lnode in range(9):
                lx = lnode % 3; ly = lnode // 3
                ix = ixe + lx; iy = iye + ly
                inode = self.nnx * iy + ix
                loc_conn += [inode]
            self.conn += [loc_conn]
        # print(f"{self.conn=}")

        # matrix sparsity constructed directly from elem conn
        self.rowp, self.cols, self.nnzb = build_csr_from_conn(self.conn, self.nnodes)

    def _assemble_system(self):
        # assemble BSR matrix
        self.data = np.zeros((self.nnzb, self.dof_per_node, self.dof_per_node), dtype=np.double)

        dpn = self.dof_per_node
        self.force = np.zeros(self.N)
        self.dx = self.length / (self.nnx - 1)
        
        # compute LHS and RHS no BCs
        for ielem in range(self.num_elements):
            ixe = ielem % self.nxe; iye = ielem // self.nxe 
            loc_conn = self.conn[ielem]
            elem_xpts = np.zeros(27)
            for lnode, gnode in enumerate(elem_xpts):
                ix, iy = gnode % self.nnx, gnode // self.nnx
                elem_xpts[3 * lnode] = ix * self.dx
                elem_xpts[3 * lnode + 1] = iy * self.dx
            
            kelem = self.element.get_kelem(
                self.E, self.nu, self.thick, elem_xpts,
                left_bndry=ixe == 0,
                right_bndry=ixe == self.nnx - 1,
                bot_bndry=iye == 0,
                top_bndry=iye == self.nnx - 1,
            )

            felem = self.element.get_felem(
                mag=self.load_fcn,
                elem_xpts=elem_xpts,
                left_bndry=ixe == 0,
                right_bndry=ixe == self.nnx - 1,
                bot_bndry=iye == 0,
                top_bndry=iye == self.nnx - 1,
            )

            # add kelem into LHS sparse structure
            for lblock_row,block_row in enumerate(loc_conn):
                for colp in range(self.rowp[block_row], self.rowp[block_row+1]):
                    block_col = self.cols[colp]
                    if block_col in loc_conn:
                        # get which val
                        lblock_col = np.argwhere(loc_conn == block_col)                 
                        
                        # add in one nodal block of kelem
                        self.data[colp,:,:] += kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                                                     dpn*lblock_col:dpn*(lblock_col+1)]

            # add felem into RHS
            np.add.at(self.force, loc_conn, felem)

        if self.split_disp_bc:

            for block_row in range(self.nnodes):
                ix, iy = block_row % self.nnx, block_row // self.nnx
                
                # bottom-left corner
                if ix == 0 and iy == 0:
                    # w_b = w_s1 = w_s2 = 0
                    for colp in range(self.rowp[block_row], self.rowp[block_row + 1]):
                        block_col = self.cols[colp]
                        if block_col == block_row:
                            self.data[colp, :, :] = np.eye(3) 
                        else:
                            self.data[colp, :, :] = 0.0
                    
                    self.force[dpn * block_row + 0] = 0.0 # w_b = 0
                    self.force[dpn * block_row + 1] = 0.0 # w_s1 = 0
                    self.force[dpn * block_row + 2] = 0.0 # w_s2 = 0

                elif ix == 0: # left edge
                    # w_b + w_s1 + w_s2 = 0
                    # and w_s2 = 0
                    for colp in range(self.rowp[block_row], self.rowp[block_row + 1]):
                        block_col = self.cols[colp]
                        self.data[colp, 0, :] = 0.0
                        self.data[colp, 2, :] = 0.0
                        if block_col == block_row:
                            self.data[colp, 0, 0] = 1.0
                            self.data[colp, 0, 1] = 1.0
                            self.data[colp, 0, 2] = 1.0
                            self.data[colp, 2, 2] = 1.0
                            
                    self.force[dpn * block_row + 0] = 0.0 # w_b + w_s1 + w_s2 = 0
                    self.force[dpn * block_row + 2] = 0.0 # w_s2 = 0

                elif iy == 0: # bottom edge 
                    # w_b + w_s1 + w_s2 = 0
                    # and w_s1 = 0
                    for colp in range(self.rowp[block_row], self.rowp[block_row + 1]):
                        block_col = self.cols[colp]
                        self.data[colp, 0, :] = 0.0
                        self.data[colp, 1, :] = 0.0
                        if block_col == block_row:
                            self.data[colp, 0, 0] = 1.0
                            self.data[colp, 0, 1] = 1.0
                            self.data[colp, 0, 2] = 1.0
                            self.data[colp, 1, 1] = 1.0

                    self.force[dpn * block_row + 0] = 0.0 # w_b + w_s1 + w_s2 = 0
                    self.force[dpn * block_row + 1] = 0.0 # w_s1 = 0

                else: # all other edges just w_b + w_s1 + w_s2 = 0 constr
                    for colp in range(self.rowp[block_row], self.rowp[block_row + 1]):
                        block_col = self.cols[colp]
                        self.data[colp, 0, :] = 0.0
                        if block_col == block_row:
                            self.data[colp, 0, 0] = 1.0
                            self.data[colp, 0, 1] = 1.0
                            self.data[colp, 0, 2] = 1.0

                    self.force[dpn * block_row + 0] = 0.0 # w_b + w_s1 + w_s2 = 0


        # else: # not split disp BC (regular SS or clamped)

        #     # apply bcs to LHS and RHS
        #     # node 1 - SS BC
        #     for colp in range(self.rowp[0], self.rowp[1]):
        #         block_col = self.cols[colp]
        #         for idof in range(dpn):
        #             row = idof
        #             if not(row in self.bcs): continue
        #             for jdof in range(dpn):
        #                 col = dpn * block_col + jdof
        #                 self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0
        #     # last node - SS BC
        #     for colp in range(self.rowp[self.nnodes-1], self.rowp[self.nnodes]):
        #         block_col = self.cols[colp]
        #         for idof in range(dpn):
        #             row = dpn * (self.nnodes-1) + idof
        #             if not(row in self.bcs): continue
        #             for jdof in range(dpn):
        #                 col = dpn * block_col + jdof
        #                 self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0

        #     for bc in self.bcs:
        #         self.force[bc] = 0.0
        
        self.kmat = sp.bsr_matrix(
            (self.data, self.cols, self.rowp),
            shape=(self.N, self.N)
        )

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        # print(f"{self.u=}")
        return self.u
    
    # def get_node_pts(self) -> list:
    #     """
    #     Quadratic (p=2) 1D IGA: return physical 'node points' using Greville abscissae.
    #     Output length = nxe + 2.

    #     u_i = (U[i+1] + U[i+2]) / 2   for i = 0..n_ctrl-1, with n_ctrl = nxe + 2
    #     x(u) = sum_j N_j(u) * x_ctrl[j]
    #     """
    #     nxe = int(self.nxe)
    #     p = 2
    #     n_ctrl = nxe + p  # = nxe + 2

    #     # Open-uniform knot vector on [0,1], length = n_ctrl + p + 1 = nxe + 5
    #     U = np.array([0.0] * (p + 1) +
    #                 [i / nxe for i in range(1, nxe)] +
    #                 [1.0] * (p + 1), dtype=float)

    #     x_ctrl = np.asarray(self.control_pts, dtype=float)
    #     if x_ctrl.size != n_ctrl:
    #         raise ValueError(
    #             f"Expected {n_ctrl} control points for p=2, nxe={nxe}, "
    #             f"but got {x_ctrl.size}."
    #         )

    #     def find_span(n_ctrl, degree, u, U):
    #         # Cox–de Boor span search; returns span in [degree, n_ctrl-1]
    #         if u >= U[-1] - 1e-14:
    #             return n_ctrl - 1
    #         low = degree
    #         high = len(U) - degree - 2
    #         mid = (low + high) // 2
    #         while True:
    #             if u < U[mid]:
    #                 high = mid - 1
    #             elif u >= U[mid + 1]:
    #                 low = mid + 1
    #             else:
    #                 return mid
    #             mid = (low + high) // 2

        
    #     def basis_functions_and_derivatives(span, u, degree, U, n_deriv=1):
    #         # Compute nonzero basis functions and first derivatives using Cox-de Boor + derivative formula
    #         # Returns arrays N[0:degree] and dN[0:degree]
    #         left = np.zeros(degree+1)
    #         right = np.zeros(degree+1)
    #         ndu = np.zeros((degree+1, degree+1))
    #         ndu[0,0] = 1.0
    #         for j in range(1, degree+1):
    #             left[j] = u - U[span+1-j]
    #             right[j] = U[span+j] - u
    #             saved = 0.0
    #             for r in range(j):
    #                 ndu[j,r] = right[r+1] + left[j-r]
    #                 temp = ndu[r,j-1]/ndu[j,r]
    #                 ndu[r,j] = saved + right[r+1]*temp
    #                 saved = left[j-r]*temp
    #             ndu[j,j] = saved
    #         N = ndu[:,degree].copy()
    #         # derivatives
    #         ders = np.zeros((n_deriv+1, degree+1))
    #         a = np.zeros((2, degree+1))
    #         # compute a triangular table of derivatives
    #         for r in range(degree+1):
    #             s1 = 0; s2 = 1
    #             a[0,0] = 1.0
    #             for k in range(1, n_deriv+1):
    #                 d = 0.0
    #                 rk = r - k
    #                 pk = degree - k
    #                 if r >= k:
    #                     a[s2,0] = a[s1,0]/ndu[pk+1,rk]
    #                     d = a[s2,0]*ndu[rk,pk]
    #                 j1 = 1 if rk >= -1 else -rk
    #                 j2 = k-1 if r-1 <= pk else degree - r
    #                 for j in range(j1, j2+1):
    #                     a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1, rk+j]
    #                     d += a[s2,j]*ndu[rk+j, pk]
    #                 if r <= pk:
    #                     a[s2,k] = -a[s1,k-1]/ndu[pk+1, r]
    #                     d += a[s2,k]*ndu[r, pk]
    #                 ders[k,r] = d
    #                 s1, s2 = s2, s1
    #         # Multiply by correct factors
    #         for k in range(1, n_deriv+1):
    #             for j in range(degree+1):
    #                 ders[k,j] *= degree
    #         return N, ders[1]
    #     # Greville abscissae (parametric "nodes"), length n_ctrl = nxe+2 for p=2
    #     u_nodes = 0.5 * (U[1:1 + n_ctrl] + U[2:2 + n_ctrl])

    #     node_pts = []
    #     for u in u_nodes:
    #         span = find_span(n_ctrl, p, float(u), U)
    #         N, _ = basis_functions_and_derivatives(span, float(u), p, U, n_deriv=1)

    #         i0 = span - p  # active basis indices: i0..i0+p
    #         x = 0.0
    #         for a in range(p + 1):
    #             x += N[a] * x_ctrl[i0 + a]
    #         node_pts.append(float(x))
    #     # print(f"{node_pts=}\n{self.control_pts=}")

    #     return node_pts
    
    # @property
    # def control_pts(self) -> list:
    #     # control points for IGA
    #     return [i*self.elem_length for i in range(self.nnodes)]
    
    # @property
    # def xvec(self) -> list:
    #     return self.control_pts # just for simple plots debugging

    # def plot_disp(self, idof:int=0):
    #     xvec = self.get_node_pts() # not same as control points
    #     # xvec = self.control_pts
    #     # print(f"{self.u=}")
    #     dpn = self.dof_per_node
    #     w = self.u[idof::dpn]
    #     if self.split_disp_bc:
    #         if dpn == 3: # hhd hermite hierarchic disp
    #             w = self.u[0::3] + self.u[2::3] # wb + ws
    #         elif dpn == 2: # higd iga hierarchic disp
    #             w = self.u[0::2] + self.u[1::2] # wb + ws
    #     plt.figure()
    #     plt.plot(xvec, w)
    #     plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
    #     plt.xlabel("x")
    #     plt.ylabel("w(x)" if idof == 0 else "th(x)")
    #     plt.show()     

    # def prolongate(self, coarse_soln):
    #     return self.element.prolongate(coarse_soln, self.L)
    
    # def restrict_defect(self, fine_defect):
    #     return self.element.restrict_defect(fine_defect, self.L)