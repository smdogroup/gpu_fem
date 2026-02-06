# Reissner-Mindlin plate with MITC4 shell elements, 6x6 DOF per node BSR matrix and using SA-AMG

"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
# plate case imports from milu python cases
sys.path.append("../../milu/")
from _plate import make_plate_case
from __src import plot_plate_vec

# krylov imports
sys.path.append("../1_beam/src/")
from smoothers import right_pgmres2

# AMG imports
sys.path.append("../../amg/_py_demo/_src/")
from bsr_aggregation import AMG_BSRSolver

# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, no: solve with GMRES")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="whether to debug multilevel process")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="whether to turnoff kernel or not")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=32, help="num elems each direction x and y")
parser.add_argument("--fill", type=int, default=0, help="ILU(k) fill level")
parser.add_argument("--iters", type=int, default=1, help="num energy-opt iters (if iter == 1 same as SA-AMG)")
parser.add_argument("--nsmooth", type=int, default=2, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--threshold", type=float, default=0.13, help="aggregation sparsity threshold, higher threshold increases operator complexity and takes fewer GMRES iters, but inc memory and more time per iteration (tradeoff)")
parser.add_argument("--omega", type=float, default=None, help="jacobi smoother update coeff, default is None and uses spectral radius")
args = parser.parse_args()

# complex_load = True
complex_load = False

_, _, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load, apply_bcs=True)
_, _, A_free, _, _, _ = make_plate_case(args, complex_load=complex_load, apply_bcs=False)

N = A.shape[0]
nnodes = N // 6

assert not(args.random) # just regular ordering.. 

# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

from bsr_aggregation import get_rigid_body_modes #, get_coarse_rigid_body_modes
# print(f"{xpts0.shape=}")
B = get_rigid_body_modes(xpts0)

pc = AMG_BSRSolver(
    A_free, A, B, threshold=args.threshold, omega=args.omega,             
    # ncyc=1 if args.thick >= 1e-3 else 2,
    ncyc=1,
    # ncyc=2,
    pre_smooth=args.nsmooth, post_smooth=args.nsmooth, near_kernel=not(args.nokernel)
)

if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres2(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None, rtol=1e-6)

print(f"{pc.total_vcycles=}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

if args.plot:
    print("plot fine soln using direct vs iterative solver")


    # ====================================================
    # 2) direct solve baseline
    # ====================================================

    # equiv solution with no reorder
    x_direct = sp.linalg.spsolve(A.copy(), rhs.copy())


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