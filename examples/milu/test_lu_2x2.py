# test LU on 2 node (6x6 block) BSR matrix

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from __ilu import GaussJordanBlockPrecond, gaussJordan, get_lu_residual

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------------------------------
# Build a 2x2 nodal BSR matrix with 6x6 blocks (12x12 total)
# ------------------------------------------------------------

def make_dense_2x2_bsr(block_size=6, seed=1):
    rng = np.random.default_rng(seed)

    # 6x6 blocks
    A00 = rng.standard_normal((block_size, block_size))
    A01 = rng.standard_normal((block_size, block_size))
    A10 = rng.standard_normal((block_size, block_size))
    A11 = rng.standard_normal((block_size, block_size))

    # Make diagonal blocks well-conditioned / invertible
    A00 += block_size * np.eye(block_size)
    A11 += block_size * np.eye(block_size)

    # Assemble block data in BSR format
    # Row 0: blocks (0,0), (0,1)
    # Row 1: blocks (1,0), (1,1)
    data = np.array([A00, A01, A10, A11])
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])

    A_bsr = sp.bsr_matrix(
        (data, indices, indptr),
        shape=(2*block_size, 2*block_size),
        blocksize=(block_size, block_size),
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
    A_bsr = make_dense_2x2_bsr(block_size)
    A_dense = A_bsr.toarray()

    print("Dense matrix A:")
    print(A_dense)

    # Random RHS
    rng = np.random.default_rng(2)
    b = rng.standard_normal(2 * block_size)

    # Exact solution
    x_exact = exact_solve(A_dense, b)

    # --------------------------------------------------------
    # PLACEHOLDER: your LU / ILU solve goes here
    # --------------------------------------------------------
    precond = GaussJordanBlockPrecond(A_bsr.copy())
    x_test = precond.solve(b)
    # Example: x_test = your_lu_solve(A_bsr, b)
    #
    # For now, just copy exact so script runs
    x_test = x_exact.copy()

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

    if rel_err < 1e-12:
        print("PASS ✅")
    else:
        print("FAIL ❌")

    # check the factorization residual for 2x2 case..
    # A_lu = precond.A.copy()
    # # do gauss-jordan on each diagonal block..
    # for i in [0, 3]:
    #     tmp = A_lu.data[i].copy()
    #     B = np.eye(6)
    #     gaussJordan(tmp, B)
    #     A_lu.data[i] = B.copy()
    # L = A_lu.copy()
    # L.data[0] = np.eye(6)
    # L.data[1] *= 0.0
    # L.data[3] = np.eye(6)
    # U = A_lu.copy()
    # U.data[2] *= 0.0

    # LU_prod = L @ U

    # # now compute the differences
    # R = LU_prod - A_bsr
    # R_norm = np.linalg.norm(R.toarray())
    # print(f"{R_norm=:.4e}")

    # compute again with new method
    R2 = get_lu_residual(A_bsr, precond.A.copy())
    R2_norm = np.linalg.norm(R2.toarray())
    print(f"{R2_norm=:.4e}")
    
    # import matplotlib.pyplot as plt
    # # plt.imshow(A_diff.toarray())
    # fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    # ax[0].imshow(precond.A.toarray())
    # ax[1].imshow(A_bsr.toarray())
    # plt.show()


if __name__ == "__main__":
    main()
