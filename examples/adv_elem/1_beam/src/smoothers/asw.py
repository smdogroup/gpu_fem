import numpy as np
import scipy.sparse as sp

class OnedimAddSchwarz:
    def __init__(self, K:sp.bsr_matrix, block_dim:int=2, coupled_size:int=2, omega:float=0.7, iters:int=1):
        assert sp.isspmatrix_bsr(K) or sp.isspmatrix_csr(K)

        self.K = K
        self.N = K.shape[0]
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim

        self.coupled_size = coupled_size
        self.omega = omega
        self.iters = iters # number of times we apply the solver per iteration

    @classmethod
    def from_assembler(cls, assembler, omega:float=0.7, iters:float=1, coupled_size:int=2):
        return cls(assembler.kmat, omega=omega, block_dim=assembler.dof_per_node, iters=iters, coupled_size=coupled_size)
    
    def solve(self, rhs:np.ndarray):
        bs = self.block_dim
        soln = np.zeros_like(rhs)
        defect = rhs.copy()

        # print(f"{self.K.toarray()=} {self.block_dim=}")

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
                # print(f"{Kc.shape=} {Fc.shape=} for {ind=} to {ind+self.coupled_size-1=}")

                # print(f"{Kc.shape=} {Fc.shape=}")
                # import matplotlib.pyplot as plt
                # plt.imshow(Kc)
                # plt.show()

                uc = np.linalg.solve(Kc, Fc)
                soln[bs * ind : bs * (ind + self.coupled_size)] += self.omega * uc

            # compute new defect
            defect = rhs - self.K.dot(soln)
        return soln

    def smooth_defect(self, soln:np.ndarray, defect:np.ndarray):
        bs = self.block_dim
        for iter in range(self.iters):
            dsoln = np.zeros_like(soln)

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
                            Kc[bs * inode : bs * (inode + 1), bs * jnode : bs * (jnode+1)] += self.K.data[jp] * 1.0
                
                Fc = defect[bs * ind : bs * (ind + self.coupled_size)].copy()

                uc = np.linalg.solve(Kc, Fc)
                dsoln[bs * ind : bs * (ind + self.coupled_size)] += self.omega * uc

            # compute new defect
            soln += dsoln
            defect -= self.K.dot(dsoln)
        return