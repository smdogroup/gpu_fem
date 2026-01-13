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
parser.add_argument("--nsmooth", type=int, default=2, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--threshold", type=float, default=0.1, help="aggregation sparsity threshold")
parser.add_argument("--omega", type=float, default=None, help="jacobi smoother update coeff, default is None and uses spectral radius")
args = parser.parse_args()

complex_load = True
# complex_load = False

_, _, A, rhs, perm, xpts0 = make_plate_case(args, complex_load=complex_load, apply_bcs=True)
_, _, A_free, _, _, _ = make_plate_case(args, complex_load=complex_load, apply_bcs=False)

N = A.shape[0]
nnodes = N // 6

assert not(args.random) # just regular ordering.. 

# ====================================================
# 2) direct solve baseline
# ====================================================

# equiv solution with no reorder
x_direct = sp.linalg.spsolve(A.copy(), rhs.copy())

# ====================================================
# DEBUG / DEVEL SA-AMG for plate
# ====================================================
if args.debug:
    # compute node aggregate sets
    # make sure to use unconstrained matrix for aggregation indicators originally
    # need threshold a bit lower sometimes to get proper coarsening
    aggregate_ind = greedy_serial_aggregation_bsr(A_free, threshold=0.1)
    num_agg = np.max(aggregate_ind) + 1

    # print(f"{aggregate_ind=}")
    print(f"{num_agg=}")
    nx = args.nxe + 1
    plot_plate_aggregation(aggregate_ind, nx, nx, 1.0, 1.0)

    # get rigid body modes for DEBUG: (check rigid body modes)
    from bsr_aggregation import get_rigid_body_modes, get_bc_flags_bsr #, get_coarse_rigid_body_modes
    B = get_rigid_body_modes(xpts0)
    print(f"{B.shape=}")
    for imode in range(6):
        bi = B[:,:,imode].copy().reshape((N,))
        bi /= np.linalg.norm(bi)
        rvec = np.random.rand(N)
        rvec /= np.linalg.norm(rvec)
        # compare residual norm of Bi vs random displacement vector (with high freq error)
        #    on free matrix of fine grid
        F_bi = A.dot(bi)
        F_rv = A.dot(rvec)
        b_nrm = np.linalg.norm(F_bi)
        r_nrm = np.linalg.norm(F_rv)
        print(f"rigid body mode {imode} with norm {np.linalg.norm(bi):.4e} => A*vec norm {b_nrm:.4e}")
        print(f"\tvs random unit vec with A*vec norm {r_nrm:.4e}")

    omega = None
    # omega = 0.0
    # omega = 0.3
    # omega = 0.7
    # omega = 1.0

    print(f"{B.shape=}")

    xpts_arr = xpts0.reshape((nnodes, 3))

    # # create tentative prolongator then smooth it
    # T, Bc = tentative_prolongator_bsr(xpts_arr, aggregate_ind)
    bc_flags = get_bc_flags_bsr(A)
    T, Bc = tentative_prolongator_bsr(B, aggregate_ind, bc_flags)
    # P = T.copy()
    P = smooth_prolongator_bsr(T, A, Bc, bc_flags, omega=omega) # single damped jacobi step, so only one step of fillin
    R = P.T # sym matrix so restriction is transpose prolong

    # galerkin coarse grid construction
    Ac = R @ (A @ P)
    Ac_free = R @ (A_free @ P)

    # print(f"{Ac.shape=}")

    # # ------------------------------------------------------------
    # # DEBUG: check properties of tentative prolongator..
    # # ------------------------------------------------------------

    # # https://epubs.siam.org/doi/epdf/10.1137/110838844
    # # Bc, xpts_c = get_coarse_rigid_body_modes(B, xpts0.reshape((nnodes, 3)), aggregate_ind)
    # # print(f"{R.shape=}")
    # TT = T.T
    # print("Check tentative prolongator satisfies appropriate conditions..")
    # for imode in range(6):
    #     b_vec = B[:,:,imode].flatten()
    #     r_vec = Bc[:,:,imode].flatten()
    #     T_resid = b_vec - T.dot(r_vec)
    #     b_resid_nrm = np.linalg.norm(T_resid)
    #     b_rel_err = b_resid_nrm / np.linalg.norm(b_vec)
        
    #     # T^T * T = I (prolong then restrict should return same vector each time)
    #     temp1 = T.dot(r_vec)
    #     temp2 = TT.dot(temp1)
    #     TT_resid = temp2 - r_vec
    #     pr_resid_nrm = np.linalg.norm(TT_resid)
    #     TT_rel_err = pr_resid_nrm / np.linalg.norm(r_vec)
    #     print(f"{imode=}, B=T*B_c : {b_resid_nrm=:.4e} {b_rel_err=:.4e}")
    #     print(f"\tT^T * T = I on vec : {pr_resid_nrm=:.4e} {TT_rel_err=:.4e}")

    # # ------------------------------------------------------------
    # # DEBUG: check properties of smoothed prolongator..
    # # ------------------------------------------------------------
    # print("Check smoothed prolongator satisfies appropriate conditions..")
    # for imode in range(6):
    #     b_vec = B[:,:,imode].flatten()
    #     r_vec = Bc[:,:,imode].flatten()
    #     T_resid = b_vec - P.dot(r_vec)
    #     T_resid[bc_flags] = 0.0 # zero out on bcs
    #     b_resid_nrm = np.linalg.norm(T_resid)
    #     b_rel_err = b_resid_nrm / np.linalg.norm(b_vec)
        
    #     # P^T * P = I (prolong then restrict should return same vector each time)
    #     temp1 = P.dot(r_vec)
    #     temp2 = R.dot(temp1)
    #     TT_resid = temp2 - r_vec
    #     pr_resid_nrm = np.linalg.norm(TT_resid)
    #     TT_rel_err = pr_resid_nrm / np.linalg.norm(r_vec)
    #     print(f"{imode=}, B=P*B_c : {b_resid_nrm=:.4e} {b_rel_err=:.4e}")
    #     print(f"\tP^T * P = I on vec : {pr_resid_nrm=:.4e} {TT_rel_err=:.4e}")

    # ------------------------------------------------------------
    # DEBUG: demo steps in V-cycle
    # ------------------------------------------------------------

    # ignore pre-smooth for prelim demo (ADD LATER)

    # restrict fine residual
    rhs = rhs.copy()
    rhs_coarse = R.dot(rhs)
    # print(f"{rhs.shape=} {rhs_coarse.shape=}")

    # coarse grid direct solve
    soln_coarse = spla.spsolve(Ac, rhs_coarse)

    # prolong solution to fine
    pr_fine = P.dot(soln_coarse)
    r_fine = rhs - A.dot(pr_fine) # new residual

    # # smooth solution with Gauss-seidel
    # dx_fine = block_gauss_seidel_6dof(A, r_fine, x0=np.zeros(6 * nnodes), num_iter=2)
    # x_fine = pr_fine + dx_fine
    # r_fine2 = rhs - A.dot(x_fine)

    x_fine = pr_fine.copy()

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

    print("END OF DEBUG, exiting before solve..")
    exit()


# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

from bsr_aggregation import get_rigid_body_modes #, get_coarse_rigid_body_modes
B = get_rigid_body_modes(xpts0)

pc = AMG_BSRSolver(A_free, A, B, threshold=args.threshold, omega=args.omega, pre_smooth=args.nsmooth, post_smooth=args.nsmooth) #, near_kernel=args.kernel)

# # check V-cycle symmetry
# x = np.random.rand(N)
# y = np.random.rand(N)
# lhs_ = x @ pc.solve(y)
# rhs_ = y @ pc.solve(x)
# print(f"{lhs_=} {rhs_=}")
# print(np.allclose(lhs_, rhs_))


if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None)
    # coarse grid matrix is not exactly numerically symmetric rn.. (due to spsolve function?) so GMRES is doing better..
    # after fix orthog projector may work again (cause less near-zero pivots)?
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