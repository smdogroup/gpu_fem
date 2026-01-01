# test LU on 3 node (6x6 block) BSR matrix

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from __ilu import GaussJordanBlockPrecond, gaussJordan, get_lu_residual
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------------------------------
# Build a 3x3 nodal BSR matrix with 6x6 blocks (18x18 total)
# ------------------------------------------------------------
def make_dense_3x3_bsr(block_size=6, seed=1):
    rng = np.random.default_rng(seed)

    # Create 3x3 blocks
    blocks = {}
    for i in range(3):
        for j in range(3):
            blk = rng.standard_normal((block_size, block_size))
            if i == j:  # diagonal blocks, make well-conditioned
                blk += block_size * np.eye(block_size)
            blocks[(i, j)] = blk

    # Assemble data, indices, indptr
    data = []
    indices = []
    indptr = [0]

    for i in range(3):
        row_blocks = [blocks[(i,j)] for j in range(3)]
        data.extend(row_blocks)
        indices.extend([0,1,2])
        indptr.append(indptr[-1] + 3)

    data = np.array(data)
    indices = np.array(indices)
    indptr = np.array(indptr)

    A_bsr = sp.bsr_matrix(
        (data, indices, indptr),
        shape=(3*block_size, 3*block_size),
        blocksize=(block_size, block_size)
    )
    return A_bsr


# ------------------------------------------------------------
# Exact LU inverse (dense reference)
# ------------------------------------------------------------
def exact_solve(A_dense, b):
    lu, piv = la.lu_factor(A_dense)
    return la.lu_solve((lu, piv), b)


# ------------------------------------------------------------
# Test driver
# ------------------------------------------------------------
def main():
    block_size = 6
    A_bsr = make_dense_3x3_bsr(block_size)
    A_dense = A_bsr.toarray()

    print("Dense matrix A:")
    print(A_dense)

    # Random RHS
    rng = np.random.default_rng(2)
    b = rng.standard_normal(3 * block_size)

    # Exact solution
    x_exact = exact_solve(A_dense, b)

    # --------------------------------------------------------
    # Apply Gauss-Jordan block preconditioner
    # --------------------------------------------------------
    precond = GaussJordanBlockPrecond(A_bsr.copy())
    x_test = precond.solve(b)

    # --------------------------------------------------------
    # Error check
    # --------------------------------------------------------
    rel_err = np.linalg.norm(x_test - x_exact) / np.linalg.norm(x_exact)

    print("\nRHS b:")
    print(b)

    print("\nExact solution:")
    print(x_exact)

    print("\nYour LU solution:")
    print(x_test)

    print(f"\nRelative error: {rel_err:.3e}")
    print("PASS ✅" if rel_err < 1e-12 else "FAIL ❌")

    # --------------------------------------------------------
    # Compute LU residual
    # --------------------------------------------------------
    R2 = get_lu_residual(A_bsr, precond.A.copy())
    R2_norm = np.linalg.norm(R2.toarray())
    print(f"LU residual norm: {R2_norm:.4e}")

    # --------------------------------------------------------
    # Visualize A and LU product
    # --------------------------------------------------------
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(A_bsr.toarray(), cmap='viridis')
    # ax[0].set_title('A BSR matrix')
    # ax[1].imshow(precond.A.toarray(), cmap='viridis')
    # ax[1].set_title('LU factorized (block)')
    # plt.show()


if __name__ == "__main__":
    main()
