"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../milu/")
from _plate import make_plate_case
from __linalg import right_pgmres
from __src import plot_plate_vec

sys.path.append("_src/")
# from asw import OnedimAddSchwarz # works like line smoother
from asw import TwodimAddSchwarz

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
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
parser.add_argument("--nxe", type=int, default=30, help="num elems each direction x and y")
parser.add_argument("--size", type=int, default=2, help="coupling size of schwarz smoother")
parser.add_argument("--omega", type=float, default=0.25, help="additive coefficient for ASW")
parser.add_argument("--iters", type=int, default=5, help="num schwarz smooths per krylov step (more than one allowed)")
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

# pc = OnedimAddSchwarz(A0.copy(), rhs0.copy(), block_dim=6, 
#                   coupled_size=args.size, omega=args.omega, iters=args.iters)

pc = TwodimAddSchwarz(A0.copy(), rhs0.copy(), block_dim=6, nx=args.nxe+1, ny=args.nxe+1, 
                    coupled_size=args.size, omega=args.omega, iters=args.iters)

if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)


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