import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix


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

def gauss_seidel_csr_transpose(A, b, x0, num_iter=1):
    """
    Backward (transpose) Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]

    for _ in range(num_iter):
        for i in range(n - 1, -1, -1):
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

def block_gauss_seidel_6dof(A, b: np.ndarray, x0: np.ndarray, num_iter=1):
    x = x0.copy()
    ndof = 6
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            Aii = np.zeros((ndof, ndof))
            rhs = b[i*ndof:(i+1)*ndof].copy()

            # loop over blocks in row i
            for idx in range(A.indptr[i], A.indptr[i+1]):
                j = A.indices[idx]
                Ablock = A.data[idx]
                if Ablock.shape != (ndof, ndof):
                    raise ValueError(f"Expected 6x6 block, got {Ablock.shape}")
                
                col_range = slice(j*ndof, (j+1)*ndof)
                if j == i:
                    Aii[:, :] = Ablock
                else:
                    rhs -= Ablock @ x[col_range]

            # solve local 6x6 system
            try:
                x[i*ndof:(i+1)*ndof] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x

def block_gauss_seidel_6dof_transpose(A, b: np.ndarray, x0: np.ndarray, num_iter=1):
    """
    Transpose (backward) Block Gauss-Seidel smoothing for 6 DOF per node (BSR 6x6 blocks).

    A: csr_matrix or bsr_matrix of size (6*nnodes, 6*nnodes)
    b: RHS vector (6*nnodes,)
    x0: initial guess (6*nnodes,)
    num_iter: number of smoothing iterations

    Returns updated solution vector x
    """
    x = x0.copy()
    ndof = 6
    n = A.shape[0] // ndof

    for it in range(num_iter):
        # backward sweep
        for i in reversed(range(n)):
            Aii = np.zeros((ndof, ndof))
            rhs = b[i*ndof:(i+1)*ndof].copy()

            # loop over blocks in row i
            for idx in range(A.indptr[i], A.indptr[i+1]):
                j = A.indices[idx]
                Ablock = A.data[idx]
                if Ablock.shape != (ndof, ndof):
                    raise ValueError(f"Expected 6x6 block, got {Ablock.shape}")

                col_range = slice(j*ndof, (j+1)*ndof)
                if j == i:
                    Aii[:, :] = Ablock
                else:
                    rhs -= Ablock @ x[col_range]

            # solve local 6x6 system
            try:
                x[i*ndof:(i+1)*ndof] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x
