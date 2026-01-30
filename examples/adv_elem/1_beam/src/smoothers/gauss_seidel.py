import numpy as np

import scipy.sparse as sp


def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, dof_per_node:int=2):
    """
    Perform Block Gauss-Seidel smoothing for 6 DOF per node.
    A: csr_matrix of size (6*nnodes, 6*nnodes)
    b: RHS vector (6*nnodes,)
    x0: initial guess (6*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    ndof = dof_per_node
    n = A.shape[0] // ndof

    # print(f"{A.shape=} {b.shape=} {x0.shape=}")

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue
    return x

class BlockGaussSeidel:
    def __init__(self, K:sp.bsr_matrix, omega:float=0.7, block_dim:int=2, iters:int=1):
        assert sp.isspmatrix_bsr(K)

        self.K_bsr = K.copy()
        self.K = K.tocsr() # must convert to csr matrix
        self.N = K.shape[0]
        self.omega = omega
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim
        self.iters = iters # number of times we apply the solver per iteration
        # print(f"{self.K.shape=} {self.N=} {self.block_dim=} {self.nnodes=}")

    @classmethod
    def from_assembler(cls, assembler, omega:float=0.7, iters:float=1):
        return cls(assembler.kmat, omega=omega, block_dim=assembler.dof_per_node, iters=iters)

    def solve(self, rhs:np.ndarray):
        defect = rhs.copy()
        x = block_gauss_seidel(self.K, defect.copy(), x0=np.zeros_like(defect), 
                                num_iter=self.iters, dof_per_node=self.block_dim)
        return x

    def smooth_defect(self, x:np.ndarray, defect:np.ndarray):
        # for iter in range(self.iters):
        dx = block_gauss_seidel(self.K, defect.copy(), x0=np.zeros_like(defect), 
                                num_iter=self.iters, dof_per_node=self.block_dim)
        dx *= self.omega
        x += dx
        defect -= self.K_bsr.dot(dx)
        return