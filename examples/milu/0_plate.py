# single level ILU(0)-GMRES solve of reissner-mindlin plate (with optional Q-ordering)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# now let's test this out and visualize it
import numpy as np
import sys, scipy as sp
from _plate import make_plate_case
from __src import plot_plate_vec
from __linalg import right_pgmres
from __ilu import GaussJordanBlockPrecond
import matplotlib.pyplot as plt

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
    parser.add_argument("--nxe", type=int, default=20) # 10
    parser.add_argument("--fill", type=int, default=2) # ILU(k) fill level, 0 is also good to try sometimes
    args = parser.parse_args()

    A0, rhs0, A, rhs, perm, xpts0 = make_plate_case(args)

    N = A0.shape[0]
    nnodes = N // 6

    # try higher-precision floating point (didn't help..)
    # A.data = A.data.astype(np.longdouble)

    # ====================================================
    # 2) direct solve baseline
    # ====================================================

    # equiv solution with no reorder
    x = sp.sparse.linalg.spsolve(A0.copy(), rhs0.copy())

    # =======================================================
    # 3) single level ILU(0) and GMRES
    # =======================================================

    precond = GaussJordanBlockPrecond(A)
        
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