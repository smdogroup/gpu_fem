import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class OnedimAddSchwarz:
    def __init__(self, K:sp.bsr_matrix, F:np.ndarray, block_dim:int=2, coupled_size:int=2, omega:float=0.7, iters:int=1):
        assert sp.isspmatrix_bsr(K) or sp.isspmatrix_csr(K)

        self.K = K
        self.F = F
        self.N = K.shape[0]
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim

        self.coupled_size = coupled_size
        self.omega = omega
        self.iters = iters # number of times we apply the solver per iteration

    def solve(self, rhs:np.ndarray):
        bs = self.block_dim
        soln = np.zeros_like(rhs)
        defect = rhs.copy()

        for iter in range(self.iters):

            # loop over each subspace of 1D
            for ind in range(self.nnodes - (self.coupled_size - 1)):
                # extract small dense matrix..
                c_dof = self.coupled_size * self.block_dim
                Kc = np.zeros((c_dof, c_dof))
                # extract these blocks into Kc
                for row in range(ind, ind + self.coupled_size):
                    for jp in range(self.K.indptr[row], self.K.indptr[row+1]):
                        j = self.K.indices[jp]
                        if j in list(range(ind, ind + self.coupled_size)):
                            inode = row - ind; jnode = j - ind
                            Kc[bs * inode : bs * (inode + 1), bs * jnode : bs * (jnode+1)] = self.K.data[jp] * 1.0
                
                Fc = defect[bs * ind : bs * (ind + self.coupled_size)].copy()

                uc = np.linalg.solve(Kc, Fc)
                soln[bs * ind : bs * (ind + self.coupled_size)] += self.omega * uc

            # compute new defect
            defect = rhs - self.K.dot(soln)

        return soln


class TwodimAddSchwarz:
    def __init__(self, K:sp.bsr_matrix, F:np.ndarray, nx:int, ny:int, block_dim:int=2, coupled_size:int=2, omega:float=0.7, iters:int=1):
        assert sp.isspmatrix_bsr(K) or sp.isspmatrix_csr(K)

        self.K = K
        self.F = F
        self.N = K.shape[0]
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim

        self.coupled_size = coupled_size
        self.omega = omega
        self.iters = iters # number of times we apply the solver per iteration

        self.nx = nx
        self.ny = ny
        self._nnx = nx - (coupled_size - 1)
        self._nny = ny - (coupled_size - 1)
        self._num_blocks = self._nnx * self._nny
        self._sblock_size = coupled_size**2 * block_dim
        self._sch_nodes = []
        self._blocks = np.zeros((self._num_blocks, self._sblock_size, self._sblock_size))
        self._compute_schwarz_blocks()

    def _compute_schwarz_blocks(self):
        """pre-compute additive schwarz block inverses"""
        bs = self.block_dim
        for iblock in range(self._num_blocks):
            # starting node of each block
            ix0 = iblock % self._nnx
            iy0 = iblock // self._nnx
            schw_nodes = [self.nx * (iy0+iiy) + ix0+iix for iix in range(self.coupled_size) for iiy in range(self.coupled_size)]
            self._sch_nodes += [schw_nodes]
            
            for i, row_node in enumerate(schw_nodes):
                for j, col_node in enumerate(schw_nodes):
                    # find the corresponding row + col block node in the matrix
                    for jp2 in range(self.K.indptr[row_node], self.K.indptr[row_node+1]):
                        col2 = self.K.indices[jp2]
                        if col_node == col2:
                            break
                    
                    self._blocks[iblock,bs*i:bs*(i+1), bs*j:bs*(j+1)] = self.K.data[jp2] * 1.0
        
            # then invert each schwarz block in place
            self._blocks[iblock] = np.linalg.inv(self._blocks[iblock] * 1.0)

        return

    @classmethod
    def from_assembler(cls, assembler, omega: float = 0.7, iters: int = 1, coupled_size: int = 2):
        return cls(
            assembler.kmat,
            assembler.force if hasattr(assembler, "force") else assembler.F,
            nx=assembler.nnx,
            ny=assembler.nnx,
            block_dim=assembler.dof_per_node,
            coupled_size=coupled_size,
            omega=omega,
            iters=iters,
        )



    def solve(self, rhs:np.ndarray):
        bs = self.block_dim
        soln = np.zeros_like(rhs)
        defect = rhs.copy()

        for iter in range(self.iters):

            for iblock in range(self._num_blocks):
                Kc_inv = self._blocks[iblock]
                Fc = np.concatenate([defect[bs*inode:bs*(inode+1)] for inode in self._sch_nodes[iblock]])
                # print(f"{Kc_inv.shape=} {Fc.shape=}")
                uc = np.dot(Kc_inv, Fc)
                for i,inode in enumerate(np.array(self._sch_nodes[iblock])):
                    soln[bs*inode:bs*(inode+1)] += self.omega * uc[bs*i:bs*(i+1)]

            # compute new defect
            defect = rhs - self.K.dot(soln)

        return soln

    def smooth_defect(self, soln: np.ndarray, defect: np.ndarray):
        bs = self.block_dim

        for _ in range(self.iters):
            dsoln = np.zeros_like(soln)

            for iblock in range(self._num_blocks):
                Kc_inv = self._blocks[iblock]

                # gather local defect
                Fc = np.concatenate([defect[bs*inode:bs*(inode+1)] for inode in self._sch_nodes[iblock]])

                # local solve
                uc = Kc_inv @ Fc

                # scatter-add into dsoln
                for i, inode in enumerate(self._sch_nodes[iblock]):
                    dsoln[bs*inode:bs*(inode+1)] += self.omega * uc[bs*i:bs*(i+1)]

            # update global soln/defect
            soln += dsoln
            defect -= self.K.dot(dsoln)

        return