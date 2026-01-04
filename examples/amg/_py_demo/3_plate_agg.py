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
from __src import right_pgmres, plot_plate_vec, sort_vis_maps, right_pcg
# AMG imports
sys.path.append("_src/")
from csr_aggregation import plot_plate_aggregation
from bsr_aggregation import greedy_serial_aggregation_bsr, tentative_prolongator_bsr, smooth_prolongator_bsr
from smoothers import block_gauss_seidel_6dof
from bsr_aggregation import AMG_BSRSolver

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
parser.add_argument("--iters", type=int, default=1, help="num energy-opt iters (if iter == 1 same as SA-AMG)")
parser.add_argument("--nsmooth", type=int, default=1, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--omega", type=float, default=0.7, help="jacobi smoother update coeff")
parser.add_argument("--mode", type=str, default='SelfMR', help="type of SPAI precond: ['SDesc', 'MR', 'SelfMR']")
args = parser.parse_args()

complex_load = True
# complex_load = False

A0, rhs0, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load, apply_bcs=True)
A0_free, _, A_free, _, _, _ = make_plate_case(args, complex_load=complex_load, apply_bcs=False)

N = A0.shape[0]
nnodes = N // 6

# ====================================================
# 2) direct solve baseline
# ====================================================

# equiv solution with no reorder
x_direct = sp.linalg.spsolve(A0.copy(), rhs0.copy())

# ====================================================
# DEBUG / DEVEL SA-AMG for plate
# ====================================================


# ==========================================================================================
if args.debug:
    # compute node aggregate sets
    # make sure to use unconstrained matrix for aggregation indicators originally
    # need threshold a bit lower sometimes to get proper coarsening
    aggregate_ind = greedy_serial_aggregation_bsr(A0_free, threshold=0.1)
    num_agg = np.max(aggregate_ind) + 1

    # print(f"{aggregate_ind=}")
    print(f"{num_agg=}")
    nx = args.nxe + 1
    plot_plate_aggregation(aggregate_ind, nx, nx, 1.0, 1.0)

    # omega = 0.0
    omega = 0.3
    # omega = 0.7
    # omega = 1.0

    # # create tentative prolongator then smooth it
    T, xpts_c = tentative_prolongator_bsr(xpts0, aggregate_ind)
    # P = T.copy()
    P = smooth_prolongator_bsr(T, A, omega=omega) # single damped jacobi step, so only one step of fillin
    R = P.T # sym matrix so restriction is transpose prolong

    # galerkin coarse grid construction
    Ac = R @ (A @ P)

    # print(f"{Ac.shape=}")

    # ------------------------------------------------------------
    # DEBUG: demo steps in V-cycle
    # ------------------------------------------------------------

    # ignore pre-smooth for prelim demo (ADD LATER)

    # restrict fine residual
    rhs = rhs0.copy()
    rhs_coarse = R.dot(rhs)
    # print(f"{rhs.shape=} {rhs_coarse.shape=}")

    # coarse grid direct solve
    soln_coarse = spla.spsolve(Ac, rhs_coarse)

    # prolong solution to fine
    pr_fine = P.dot(soln_coarse)
    r_fine = rhs - A.dot(pr_fine) # new residual

    # smooth solution with Gauss-seidel
    dx_fine = block_gauss_seidel_6dof(A, r_fine, x0=np.zeros(6 * nnodes), num_iter=1)
    x_fine = pr_fine + dx_fine
    r_fine2 = rhs - A.dot(x_fine)

    # plot soln comparison
    # for plotting
    nxe = int(nnodes**0.5)-1
    sort_fw = np.arange(0, N)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plot_plate_vec(nxe, x_direct.copy(), ax, sort_fw, nodal_dof=2)

    # plot right-precond solution
    ax = fig.add_subplot(122, projection='3d')
    plot_plate_vec(nxe, x_fine.copy(), ax, sort_fw, nodal_dof=2)
    plt.show()


# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

pc = AMG_BSRSolver(A_free, A, xpts0, threshold=0.1, omega=0.7, pre_smooth=1, post_smooth=1) #, near_kernel=args.kernel)

if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)
    # x2 = right_pcg(A, b=rhs, x0=None, M=pc if not(args.noprec) else None)


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