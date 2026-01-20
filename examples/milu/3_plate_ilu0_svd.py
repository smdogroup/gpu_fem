# single level ILU(0)-GMRES solve of reissner-mindlin plate
# (SVD(alpha) block solves)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

from _plate import make_plate_case
from _milu import BILU_SVD_Precond
import scipy as sp
from __linalg import right_pgmres
from __src import plot_plate_vec
import matplotlib.pyplot as plt
import numpy as np


    


if __name__ == "__main__":
    
    # ====================================================
    # 1) make plate case
    # ====================================================

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
    parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
    parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
    parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
    parser.add_argument("--alpha", type=float, default=1e-4) # coefficient for singular value thresholding
    parser.add_argument("--nxe", type=int, default=20) # 10
    parser.add_argument("--fill", type=int, default=2) # ILU(k) fill level, 0 is also good to try sometimes
    args = parser.parse_args()

    A0, rhs0, A, rhs, perm, xpts0 = make_plate_case(args, qorder_p=0.5)

    N = A0.shape[0]
    nnodes = N // 6

    # rebuild xpts then permute it
    iperm = np.arange(0, nnodes)
    for i in range(nnodes):
        j = perm[i]
        iperm[j] = i
    # build original order xpts
    xpts = np.zeros((nnodes, 3))
    nx = int(nnodes**0.5)
    dx = 1.0 / (nx - 1.0)
    for i in range(nnodes):
        ix = i % nx; iy = i // nx
        xpts[i, 0] = dx * ix
        xpts[i, 1] = dx * iy
        xpts[i, 2] = 0.0
    perm_xpts = xpts[iperm,:].reshape((3 * nnodes))

    # ====================================================
    # 2) direct solve baseline
    # ====================================================

    # equiv solution with no reorder
    x = sp.sparse.linalg.spsolve(A0.copy(), rhs0.copy())

    # =======================================================
    # 3) multi level ILU(0) and GMRES
    # =======================================================

    # print(f"{type(A)=}")
    precond = BILU_SVD_Precond(A, alpha=args.alpha)
        
    x_perm2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=precond if not(args.noprec) else None)
    x2 = x_perm2.reshape(nnodes, 6)[perm].reshape(-1)

    # ========================================================
    # 4) plot direct vs iterative solutions
    # ========================================================

    if not(args.noplot):
        # for plotting
        nxe = int(nnodes**0.5)-1
        sort_fw = np.arange(0, N)
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        plot_plate_vec(nxe, x.copy(), ax, sort_fw, nodal_dof=2)

        # plot right-precond solution
        ax = fig.add_subplot(122, projection='3d')
        plot_plate_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2)
        plt.show()