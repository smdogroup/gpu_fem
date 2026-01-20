"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _spai import *
import sys
sys.path.append("../../milu/")
from _plate import make_plate_case
from __linalg import right_pgmres
from __src import plot_plate_vec



# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--noplot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, no: solve with GMRES")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=10, help="num elems each direction x and y")
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
parser.add_argument("--iters", type=int, default=5, help="num SPAI opt iters")
parser.add_argument("--mode", type=str, default='SelfMR', help="type of SPAI precond: ['SDesc', 'MR', 'SelfMR']")
args = parser.parse_args()

complex_load = True
# complex_load = False

A0, rhs0, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load)

N = A0.shape[0]
nnodes = N // 6

# ====================================================
# 2) direct solve baseline
# ====================================================

# equiv solution with no reorder
x = sp.linalg.spsolve(A0.copy(), rhs0.copy())

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

if args.mode == 'Sdesc': # steepest descent
    precond = SPAI_Precond(A, iters=args.iters)
elif args.mode == 'MR': # minimal residual
    precond = SPAI_MR_Precond(A, iters=args.iters)
elif args.mode == 'SelfMR': # self preconditioned minimal residual (best one usually) 
    precond = SPAI_MR_SelfPrecond(A, iters=args.iters)


if args.justpc:
    x2 = precond.solve(rhs)
else:
    x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=precond if not(args.noprec) else None)


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