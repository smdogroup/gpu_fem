"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _spai import *
from _mlev_spai import MultilevelSPAI
import sys
sys.path.append("../milu/")
from _plate import make_plate_case
from __src import right_pgmres, plot_plate_vec



# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=10) # 10
parser.add_argument("--fill", type=int, default=2) # ILU(k) fill level, 0 is also good to try sometimes
args = parser.parse_args()

A0, rhs0, A, rhs, perm = make_plate_case(args)

N = A0.shape[0]
nnodes = N // 6

# try higher-precision floating point (didn't help..)
# A.data = A.data.astype(np.longdouble)

# ====================================================
# 2) direct solve baseline
# ====================================================

# equiv solution with no reorder
x = sp.linalg.spsolve(A0.copy(), rhs0.copy())

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

iters = 5
# iters = 10
# iters = 30

# precond = None
# precond = SPAI_Precond(A, iters=iters)
# precond = SPAI_MR_Precond(A, iters=iters)
precond = SPAI_MR_SelfPrecond(A, iters=iters) # best one (general)


# x2 = precond.solve(rhs)
x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=precond)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

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