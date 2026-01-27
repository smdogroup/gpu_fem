import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class StandardBeamAssembler:
    # for hermite or lagrange bases / elements

    def __init__(
        self,
        ELEMENT,
        nxe:int,
        E:float=70e9,
        nu:float=0.3,
        thick:float=1.0e-2,
        L:float=1.0,
        load_fcn=lambda x : 1.0,
    ):
        
        self.element = ELEMENT
        self.nxe = nxe
        self.E = E
        self.nu = nu
        self.thick = thick
        self.L = L
        self.load_fcn = load_fcn
        
        # internal data
        self.kmat = None
        self.force = None
        self.u = None
        self.nnodes = nxe + 1
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.elem_length = self.L / self.nxe
        dpn = self.dof_per_node
        self.bcs = list(range(dpn)) + list(range(dpn*(self.nnodes-1), dpn*self.nnodes))
        # self.bcs = [0, dpn*(self.nnodes-1)]
        self.conn = [[ielem,ielem+1] for ielem in range(self.nxe)]

        # matrix sparsity
        self.rowp = [0]; self.cols = []; self.nnzb = 0
        for inode in range(self.nnodes):
            if inode == 0:
                current_cols = [0, 1]
            elif inode == self.nnodes-1:
                current_cols = [self.nnodes-2, self.nnodes-1]
            else:
                current_cols = [inode-1, inode, inode+1]
            self.nnzb += len(current_cols)
            self.rowp += [self.nnzb]
            self.cols += current_cols
        self.rowp = np.array(self.rowp); self.cols = np.array(self.cols)
    
    @property
    def dof_conn(self):
        dpn = self.dof_per_node
        # 2 nodes per element, dpn dofs each
        return [[dpn*ix + j for j in range(2*dpn)] for ix in range(self.nxe)]


    def _assemble_system(self):
        # assemble BSR matrix
        self.data = np.zeros((self.nnzb, self.dof_per_node, self.dof_per_node), dtype=np.double)
        x_vals = [(ielem+0.5) * self.elem_length for ielem in range(self.nxe)]
        load_vals = [self.load_fcn(x_val) / self.nxe for x_val in x_vals]

        kelem = self.element.get_kelem(self.E, self.nu, self.thick, self.elem_length)
        unit_felem = self.element.get_felem(mag=1.0, elem_length=self.elem_length)
        dpn = self.dof_per_node
        self.force = np.zeros(self.N)
        
        # compute LHS and RHS no BCs
        for ielem in range(self.nxe):
            local_conn = np.array(self.dof_conn[ielem])
            # add kelem into LHS sparse structure
            for lblock_row,block_row in enumerate([ielem, ielem+1]):
                for colp in range(self.rowp[block_row], self.rowp[block_row+1]):
                    block_col = self.cols[colp]
                    if block_col in [ielem, ielem+1]:
                        lblock_col = block_col - ielem
                        
                        self.data[colp,:,:] += kelem[dpn*lblock_row:dpn*(lblock_row+1), 
                                                     dpn*lblock_col:dpn*(lblock_col+1)]

            # add felem into RHS
            felem = unit_felem * load_vals[ielem]
            np.add.at(self.force, local_conn, felem)

        # apply bcs to LHS and RHS
        # node 1 - clamped BC
        for colp in range(self.rowp[0], self.rowp[1]):
            block_col = self.cols[colp]
            if block_col == 0:
                self.data[colp,:,:] = np.eye(self.dof_per_node)
            else:
                self.data[colp,:,:] = 0.0
        # last node - clamped BC
        for colp in range(self.rowp[self.nnodes-1], self.rowp[self.nnodes]):
            block_col = self.cols[colp]
            if block_col == self.nnodes-1:
                self.data[colp,:,:] = np.eye(self.dof_per_node)
            else:
                self.data[colp,:,:] = 0.0
        for bc in self.bcs:
            self.force[bc] = 0.0
        
        self.kmat = sp.bsr_matrix(
            (self.data, self.cols, self.rowp),
            shape=(self.N, self.N)
        )

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        print(f"{self.u=}")
        return self.u
    
    @property
    def xvec(self) -> list:
        return [i*self.elem_length for i in range(self.nnodes)]

    def plot_disp(self, idof:int=0):
        xvec = self.xvec
        # print(f"{self.u=}")
        dpn = self.dof_per_node
        w = self.u[idof::dpn]
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