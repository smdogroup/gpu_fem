"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _spai import *
from _mlev_spai import MultilevelSPAI
import sys
sys.path.append("../../milu/")
from __linalg import right_pgmres
from __src import plot_plate_vec

# algebraic multilevel method mostly based on these papers
# I did add jacobi smoother for restrict + prolong, whereas they do not (cause they have much simpler problem with Poisson I guess)
# 1) https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub
# 2) https://link.springer.com/chapter/10.1007/978-3-031-25820-6_11
# 3) https://epubs.siam.org/doi/10.1137/S106482759732753X


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
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="whether to debug multilevel process")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=10, help="num elems each direction x and y")
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
parser.add_argument("--iters", type=int, default=5, help="num SPAI opt iters")
parser.add_argument("--nsmooth", type=int, default=5, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--omega", type=float, default=0.7, help="jacobi smoother update coeff")
parser.add_argument("--levels", type=int, default=2, help="num multilevel levels (using schur complements)")
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
x_direct = sp.linalg.spsolve(A0.copy(), rhs0.copy())

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

pc = MultilevelSPAI(A, levels=args.levels, iters=args.iters, n_smooth=args.nsmooth, omega=args.omega)


if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)


# ==================================
# DEBUG (to see multilevel/multigrid-style process)
# ==================================

if args.debug:

    # coded the V-cycle like process here of multilevel preconditioner
    # to identify where high defects are coming from

    # rhs inputs
    rhs_fine = rhs[pc.fine_mask]
    rhs_coarse = rhs[pc.coarse_mask]

    # forward elimination (like smooth + restrict)
    y_fine = pc.B_pc.solve(rhs_fine)
    y_coarse = rhs_coarse - pc.E.dot(y_fine)

    # coarse solve
    x_coarse = pc.S_pc.solve(y_coarse)

    # plot coarse rhs and coarse soln
    print("plot coarse rhs and solution")
    nxe_f = int(nnodes**0.5)-1
    assert(nxe_f % 2 == 0) # even number of fine elements
    nxe_c = nxe_f // 2
    sort_fw_c = np.arange(0, 6 * (nxe_c+1)**2)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plot_plate_vec(nxe_c, y_coarse, ax, sort_fw_c, nodal_dof=2)
    ax = fig.add_subplot(122, projection='3d')
    plot_plate_vec(nxe_c, x_coarse, ax, sort_fw_c, nodal_dof=2)
    plt.show()

    # backward elim (like prolong + smooth)
    fc_defect = pc.F.dot(x_coarse)

    # x_coarse = x_coarse (unchanged)
    x_fine = pc.B_pc.solve(y_fine - pc.F.dot(x_coarse))

    # store output soln
    x = np.zeros_like(rhs) # solution / output
    x[pc.fine_mask] = x_fine * 1.0
    x[pc.coarse_mask] = x_coarse * 1.0

    # then apply smoother to this..
    # say jacobi..



# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

if not args.noplot:
    print("plot fine soln using direct vs iterative solver")

    # for plotting
    nxe = int(nnodes**0.5)-1
    sort_fw = np.arange(0, N)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plot_plate_vec(nxe, x_direct.copy(), ax, sort_fw, nodal_dof=2)

    # plot right-precond solution
    ax = fig.add_subplot(122, projection='3d')
    plot_plate_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2)
    plt.show()