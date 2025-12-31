import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def gauss_seidel_csr(A, b, x0, num_iter=1):
    """
    Perform Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]
    for _ in range(num_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]
            x[i] = (b[i] - sum_) / diag
    return x

def block_gauss_seidel(A, b: np.ndarray, x0: np.ndarray, num_iter=1, ndof:int=3):
    """
    Perform Block Gauss-Seidel smoothing for 3 DOF per node.
    A: csr_matrix of size (3*nnodes, 3*nnodes)
    b: RHS vector (3*nnodes,)
    x0: initial guess (3*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    n = A.shape[0] // ndof

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

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

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

class ILU0Preconditioner:
    def __init__(self, A: sp.csr_matrix, dof_per_node=3):
        """
        Build ILU(0) preconditioner with node-wise random reordering.
        """
        A = A.copy()
        assert sp.isspmatrix_csr(A)
        n_dof = A.shape[0]
        n_nodes = n_dof // dof_per_node

        # Random permutation of nodes
        node_perm = np.random.permutation(n_nodes)
        dof_perm = np.concatenate([
            np.arange(i*dof_per_node, (i+1)*dof_per_node)
            for i in node_perm
        ])

        # temp debug
        dof_perm = np.arange(n_dof)

        A_perm = A[dof_perm, :][:, dof_perm]
        # ilu = spla.spilu(A_perm, fill_factor=1.0, drop_tol=0.0)
        ilu = spla.spilu(A_perm, fill_factor=10.0, drop_tol=1e-5)

        self.solve = ilu.solve
        self.perm = dof_perm
        self.perm_inv = np.argsort(dof_perm)

    def apply(self, r):
        """
        Apply M⁻¹ to r: z = M⁻¹ r using ILU(0) solve.
        """
        r_perm = r[self.perm]
        z_perm = self.solve(r_perm)
        return z_perm[self.perm_inv]
    
def pcg(A, b, M: ILU0Preconditioner, x0=None, tol=1e-8, maxiter=200):
    """
    PCG with right preconditioning: solve A M⁻¹ y = b, x = M⁻¹ y
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Right-preconditioned system: A M⁻¹ y = b, where x = M⁻¹ y
    r = b - A @ x
    z = M.apply(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    for i in range(maxiter):
        Ap = A @ p
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        if np.linalg.norm(r) < tol:
            print(f"PCG converged in {i+1} iterations.")
            return x

        z = M.apply(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    print("PCG did not converge within maxiter.")
    return x

