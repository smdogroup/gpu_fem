# TBD..import numpy as np
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

class IGABeamAssembler:
    # for IGA bases or elements

    def __init__(
        self,
        ELEMENT,
        nxe:int,
        E:float=70e9,
        nu:float=0.3,
        thick:float=1.0e-2,
        L:float=1.0,
        load_fcn=lambda x : 1.0,
        clamped:bool=False,
        split_disp_bc:bool=False,
    ):
        
        self.element = ELEMENT
        self.nxe = nxe
        self.E = E
        self.nu = nu
        self.thick = thick
        self.L = L
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc
        
        # internal data
        self.kmat = None
        self.force = None
        self.u = None
        self.nnodes = nxe + 2 # for 2nd order IGA elements p = 2, nnodes = nxe + p generally
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.elem_length = self.L / (self.nnodes - 1)
        dpn = self.dof_per_node
        
        if clamped:
            self.element.clamped = True
            if dpn == 3:
                self.bcs = [0,dpn*(self.nnodes-1)]
            else:
                # standard fully clamped
                self.bcs = list(range(dpn)) + list(range(dpn*(self.nnodes-1), dpn*self.nnodes))
        else:
            self.element.clamped = False
            self.bcs = [0, dpn*(self.nnodes-1)]
        
        self.conn = [[ielem,ielem+1,ielem+2] for ielem in range(self.nxe)]

        # matrix sparsity
        self.rowp = [0]; self.cols = []; self.nnzb = 0
        for inode in range(self.nnodes):
            if inode == 0:
                current_cols = [0, 1, 2]
            elif inode == 1:
                current_cols = [0,1,2,3]
            elif inode == self.nnodes-2:
                current_cols = [self.nnodes-4, self.nnodes-3, self.nnodes-2, self.nnodes-1]
            elif inode == self.nnodes-1:
                current_cols = [self.nnodes-3, self.nnodes-2, self.nnodes-1]
            else:
                current_cols = [inode-2, inode-1, inode, inode+1, inode+2]
            self.nnzb += len(current_cols)
            self.rowp += [self.nnzb]
            self.cols += current_cols
        # print(f"{self.rowp=}\n{self.nnzb=}\n{self.cols=}")
        self.rowp = np.array(self.rowp); self.cols = np.array(self.cols)
    
    @property
    def dof_conn(self):
        dpn = self.dof_per_node
        return [[dpn*ix + j for j in range(3*dpn)] for ix in range(self.nxe)]


    def _assemble_system(self):
        # assemble BSR matrix
        self.data = np.zeros((self.nnzb, self.dof_per_node, self.dof_per_node), dtype=np.double)
        x_vals = [(ielem+0.5) * self.elem_length for ielem in range(self.nxe)]
        load_vals = [self.load_fcn(x_val) / self.nxe for x_val in x_vals]

        interior_kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=False, right_bndry=False)
        interior_unit_felem = self.element.get_felem(mag=1.0, elem_length=self.elem_length, left_bndry=False, right_bndry=False)
        dpn = self.dof_per_node
        self.force = np.zeros(self.N)
        
        # compute LHS and RHS no BCs
        for ielem in range(self.nxe):
            if ielem == 0:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=True, right_bndry=False)
                unit_felem = self.element.get_felem(mag=1.0, elem_length=self.elem_length, left_bndry=True, right_bndry=False)
            elif ielem == self.nxe - 1:
                kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length, left_bndry=False, right_bndry=True)
                unit_felem = self.element.get_felem(mag=1.0, elem_length=self.elem_length, left_bndry=False, right_bndry=True)
            else: # interior
                kelem = interior_kelem
                unit_felem = interior_unit_felem

            local_conn = np.array(self.dof_conn[ielem])
            # add kelem into LHS sparse structure
            for lblock_row,block_row in enumerate([ielem, ielem+1, ielem+2]):
                for colp in range(self.rowp[block_row], self.rowp[block_row+1]):
                    block_col = self.cols[colp]
                    if block_col in [ielem, ielem+1, ielem+2]:
                        lblock_col = block_col - ielem

                        # my_mat = kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                        #                              dpn*lblock_col:dpn*(lblock_col+1)]
                        # print(f"{my_mat.shape=} {lblock_col=} {kelem.shape=}")                    
                        
                        self.data[colp,:,:] += kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                                                     dpn*lblock_col:dpn*(lblock_col+1)]

            # add felem into RHS
            felem = unit_felem * load_vals[ielem]
            np.add.at(self.force, local_conn, felem)

        if self.split_disp_bc:
            # dpn = 3 with local dofs: [w_b, (dw/dxi)_b, w_s]
            # SS: enforce w_b + w_s = 0 at BOTH ends by overwriting the w_b row (idof=0).
            # Extra gauge-fix: pin w_s(0) = 0 by overwriting the w_s row (idof=2) at the left end.

            # ---- LEFT END (node 0): w_b + w_s = 0 (overwrite w_b row) ----
            # tried changing it to just w_b = 0 on left side since also w_s = 0
            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0  # w_b row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0  # w_b
                    self.data[colp, 0, 1] = 1.0  # w_s

            # ---- LEFT END (node 0): gauge fix w_s(0) = 0 (overwrite w_s row) ----
            # the extra gauge constraint here removes constant mode from integrated shear strains th_s => w_s (cause non-unique)
            inode = 0
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 1  # w_s row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 1, 1] = 1.0  # w_s = 0

            # ---- RIGHT END (node nnodes-1): w_b + w_s = 0 (overwrite w_b row) ----
            inode = self.nnodes - 1
            for colp in range(self.rowp[inode], self.rowp[inode + 1]):
                block_col = self.cols[colp]
                idof = 0  # w_b row
                for jdof in range(dpn):
                    self.data[colp, idof, jdof] = 0.0
                if block_col == inode:
                    self.data[colp, 0, 0] = 1.0  # w_b
                    self.data[colp, 0, 1] = 1.0  # w_s

            # RHS for those constraint rows:
            self.force[dpn * 0 + 0] = 0.0                 # (w_b + w_s)(0) = 0
            self.force[dpn * 0 + 1] = 0.0                 # w_s(0) = 0  (gauge fix)
            self.force[dpn * (self.nnodes - 1) + 0] = 0.0 # (w_b + w_s)(L) = 0


        else: # not split disp BC (regular SS or clamped)

            # apply bcs to LHS and RHS
            # node 1 - clamped BC
            for colp in range(self.rowp[0], self.rowp[1]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = idof
                    if not(row in self.bcs): continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0
            # last node - clamped BC
            for colp in range(self.rowp[self.nnodes-1], self.rowp[self.nnodes]):
                block_col = self.cols[colp]
                for idof in range(dpn):
                    row = dpn * (self.nnodes-1) + idof
                    if not(row in self.bcs): continue
                    for jdof in range(dpn):
                        col = dpn * block_col + jdof
                        self.data[colp, idof, jdof] = 1.0 if (row == col) else 0.0

            for bc in self.bcs:
                self.force[bc] = 0.0
        
        self.kmat = sp.bsr_matrix(
            (self.data, self.cols, self.rowp),
            shape=(self.N, self.N)
        )

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        # print(f"{self.u=}")
        return self.u
    
    @property
    def xvec(self) -> list:
        return [i*self.elem_length for i in range(self.nnodes)]

    def plot_disp(self, idof:int=0):
        xvec = self.xvec
        # print(f"{self.u=}")
        dpn = self.dof_per_node
        w = self.u[idof::dpn]
        if self.split_disp_bc:
            if dpn == 3: # hhd hermite hierarchic disp
                w = self.u[0::3] + self.u[2::3] # wb + ws
            elif dpn == 2: # higd iga hierarchic disp
                w = self.u[0::2] + self.u[1::2] # wb + ws
        plt.figure()
        plt.plot(xvec, w)
        plt.plot(xvec, np.zeros((self.nnodes,)), "k--")
        plt.xlabel("x")
        plt.ylabel("w(x)" if idof == 0 else "th(x)")
        plt.show()     

    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.L)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.L)