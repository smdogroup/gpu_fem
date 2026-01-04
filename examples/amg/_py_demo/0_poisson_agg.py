"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../milu/")
# from __src import right_pgmres
from __src import right_pcg
sys.path.append("_src/")
from poisson import poisson_2d_csr, plot_poisson_surface, poisson_apply_bcs
from csr_aggregation import greedy_serial_aggregation_csr, plot_plate_aggregation
from csr_aggregation import tentative_prolongator_csr, smooth_prolongator_csr
from csr_aggregation import AMGSolver
from sa import gauss_seidel_csr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=50, help="num nodes each direction")
parser.add_argument("--kernel", action=argparse.BooleanOptionalAction, default=False, help="enforce near kernel constr on prolong update")
# parser.add_argument("--smoothP", type=int, default=1, help="whether to smooth prolongation or not")
parser.add_argument("--debug", type=int, default=0, help="debug printouts")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, default of no: solve with PCG")
args = parser.parse_args()

# ------------------------------------------------------------
# 0) Problem setup
# ------------------------------------------------------------
nx, ny = args.nxe + 1, args.nxe + 1

Lx, Ly = 1.0, 1.0
hx, hy = Lx/(nx-1), Ly/(ny-1)

A_free = poisson_2d_csr(nx, ny, hx, hy)

# RHS: smooth forcing
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# f = np.sin(np.pi * X) * np.sin(np.pi * Y)
f = np.sin(5 * np.pi * X) * np.sin(np.pi * (Y**2 + X**2))
f = f.ravel()

# Apply homogeneous Dirichlet BCs (simple row overwrite)
A = poisson_apply_bcs(A_free, f, nx, ny)

# ------------------------------------------------------------
# 1) Direct Solve (for true soln)
# ------------------------------------------------------------
import time
start_direct = time.time()
u_truth = spla.spsolve(A.copy(), f)
time_direct = time.time() - start_direct

# ------------------------------------------------------------
# 2) aggregation and tentative prolongator construction
# ------------------------------------------------------------

nnodes = A.shape[0]

# based on papers
"""
1) A GPU accelerated aggregation algebraic multigrid method, https://www.sciencedirect.com/science/article/pii/S0898122114004143
2) EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS, https://epubs.siam.org/doi/epdf/10.1137/110838844
also somewhat tips from this RN AMG and pics (though this is more advanced method), https://epubs.siam.org/doi/epdf/10.1137/16M1082706
"""

# ==========================================================================================
if args.debug:
    # compute node aggregate sets
    # make sure to use unconstrained matrix for aggregation indicators originally
    aggregate_ind = greedy_serial_aggregation_csr(A_free, threshold=0.25)
    num_agg = np.max(aggregate_ind) + 1

    # print(f"{aggregate_ind=}")
    print(f"{num_agg=}")
    plot_plate_aggregation(aggregate_ind, nx, ny, Lx, Ly)

    # create tentative prolongator then smooth it
    T = tentative_prolongator_csr(aggregate_ind)
    P = smooth_prolongator_csr(T, A, omega=0.7) # single damped jacobi step, so only one step of fillin
    R = P.T # sym matrix so restriction is transpose prolong

    # galerkin coarse grid construction
    Ac = R @ (A @ P)

    print(f"{Ac.shape=}")

    # ------------------------------------------------------------
    # DEBUG: demo steps in V-cycle
    # ------------------------------------------------------------

    # ignore pre-smooth for prelim demo (ADD LATER)

    # restrict fine residual
    rhs = f.copy()
    rhs_coarse = R.dot(rhs)
    # print(f"{rhs.shape=} {rhs_coarse.shape=}")

    # coarse grid direct solve
    soln_coarse = spla.spsolve(Ac, rhs_coarse)

    # prolong solution to fine
    pr_fine = P.dot(soln_coarse)
    r_fine = rhs - A.dot(pr_fine) # new residual

    # smooth solution with Gauss-seidel
    dx_fine = gauss_seidel_csr(A, r_fine, x0=np.zeros(nnodes), num_iter=1)
    x_fine = pr_fine + dx_fine
    r_fine2 = rhs - A.dot(x_fine)

    # plot soln comparison
    plot_poisson_surface(u_truth, x_fine, nx, ny, Lx, Ly)

# ==========================================================================================

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

# A_free = A.copy() # if A was not BC version.. # makes it much slower, more memory and more DOF

# precond = None
# threshold = 0.25 is default..
# needed a lower threshold to get the coarsening to work right on one of the levels.. (would coarsen from 454 to 454 nodes... then next time would work better)
precond = AMGSolver(A_free, A, threshold=0.1, omega=0.7, pre_smooth=1, post_smooth=1, near_kernel=args.kernel)

if args.justpc:
    u2 = precond.solve(f)
else:
    # needed to do M and then M^T pre and post smooths to make AMG precond symmetric for PCG..
    # u2 = right_pcg_v0(A, b=f, x0=None, M=precond)
    start_pcg = time.time()
    u2 = right_pcg(A, b=f, x0=None, M=precond)
    time_pcg = time.time() - start_pcg
    print(f"{time_direct=:.4e} {time_pcg=:.4e}")

# u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u_truth, u2, nx, ny, Lx, Ly)