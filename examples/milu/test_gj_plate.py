# test gauss-jordan 6x6 block solves on plate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# now let's test this out and visualize it
import numpy as np
import sys, scipy as sp
from _plate import make_plate_case
from __ilu import gaussJordan
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # ====================================================
    # 1) make plate case
    # ====================================================

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
    parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
    parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
    parser.add_argument("--nxe", type=int, default=20) # 10
    args = parser.parse_args()

    A0, rhs0, A, rhs, perm, xpts0 = make_plate_case(args)

    N = A0.shape[0]
    nnodes = N // 6

    # ====================================================
    # 2) test gauss-jordan on NZ entry
    # ====================================================  

    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            # j = A.indices[jp]
            block = A.data[jp]
            diag_block = np.diag(block)
            is_one = np.array([abs(diag_block[ii] - 1.0) < 1e-5 for ii in range(6)])
            no_ones = not(np.any(is_one))

            if no_ones:
                break

    # 1) test gaussJordan matrix inverse
    A = np.random.rand(6,6)
    B = np.eye(6)

    tmp = A.copy()
    fail, perm = gaussJordan(tmp, B)
    iperm = np.arange(0, 6)
    for i in range(6):
        j = perm[i]
        iperm[j] = i
    print(F"{A=}\n{B=}")
    print(f"{perm=}")

    Ainv = np.linalg.inv(A)
    R = np.abs(Ainv - B) / np.abs(Ainv)

    fig, ax = plt.subplots(1, 3, figsize=(9, 6))
    ax[0].imshow(B)
    ax[1].imshow(Ainv)
    ax[2].imshow(R)
    # plt.colorbar()
    plt.show()

    rhs = np.random.rand(6)
    x1 = np.linalg.solve(A, rhs)
    x2 = np.dot(B, rhs)
    # x2 = x2[iperm]
    err = x1 - x2
    err_nrm = np.linalg.norm(err)
    print(f"{err_nrm=:.4e}")