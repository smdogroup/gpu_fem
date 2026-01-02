# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os
import pyamg
from _utils import get_tacs_matrix, delete_rows_and_columns, reduced_indices
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="pyAMG options")
    parser.add_argument('--precond', type=str, default='right', help='[left|right] preconditioning')
    parser.add_argument('--solver', type=str, default='gmres', help='[gmres|cg] linear solver')
    parser.add_argument('--rem_bcs', action='store_true', help='remove bcs')
    return parser.parse_args()

if __name__=="__main__":
    import scipy.sparse.linalg as spla
    import time

    args = parse_args()

    # get TACS matrix and convert to CSR since pyAMG only supports CSR
    # (this may mean it is slower or less accurate than a BSR version)
    # -----------------------------------
    tacs_bsr_mat, rhs, xpts = get_tacs_matrix()
    tacs_csr_mat = tacs_bsr_mat.tocsr()

    # remove bcs from the matrix and rhs if specified
    if args.rem_bcs:
        ndof = rhs.shape[0]
        orig_dof = [_ for _ in range(ndof)]
        red_dof = reduced_indices(tacs_csr_mat)
        red_dof = np.array(red_dof)
        rhs = rhs[red_dof]

        tacs_csr_mat = delete_rows_and_columns(tacs_csr_mat, dof=None)
        

    # solve with pyAMG
    # ----------------
    if not args.rem_bcs:
        ndof = rhs.shape[0]
    # print(F"{ndof=} {(ndof%6)=}")

    print(F"{xpts.shape=} {xpts[:10]=}")

    # shell near-kernel modes, see docs https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html#pyamg.aggregation.smoothed_aggregation_solver
    B = np.zeros((ndof, 6))
    B[0::6, 0] = 1.0 # u translation
    B[1::6, 1] = 1.0 # v translation
    B[2::6, 2] = 1.0 # w translation

    nnodes = ndof // 6
    for i in range(nnodes):
        xi = xpts[3 * i]
        yi = xpts[3 * i + 1]
        zi = xpts[3 * i + 2]

        idof = 6 * i
        # Mode 4: Rotation about x-axis (affects v and w DOF)
        B[idof + 1, 3] = -zi  # v DOF: -z displacement
        B[idof + 2, 3] = yi   # w DOF: +y displacement
        B[idof + 3, 3] = 1.0  # thx DOF: rotation

        # Mode 5: Rotation about y-axis (affects u and w DOF)
        B[idof + 0, 4] = zi   # u DOF: +z displacement
        B[idof + 2, 4] = -xi  # w DOF: -x displacement
        B[idof + 4, 4] = 1.0  # thy DOF: rotation

        # Mode 6: Rotation about z-axis (affects u and v DOF)
        B[idof + 0, 5] = -yi  # u DOF: -y displacement
        B[idof + 1, 5] = xi   # v DOF: +x displacement
        B[idof + 5, 5] = 1.0  # thz DOF: rotation

    # then make it reduced dof now
    if args.rem_bcs:
        B = B[red_dof,:]
        strength = 'algebraic_distance'
    else:
        strength = 'symmetric'
        # strength = 'evolution'

    tacs_csr_mat.data = tacs_csr_mat.data.astype(np.float64)
    rhs = rhs.astype(np.float64)
    B = B.astype(np.float64)

    # B = None # default multigrid (bad solution)
    ml = pyamg.smoothed_aggregation_solver(tacs_csr_mat, B=B, strength=strength)
    M = ml.aspreconditioner(cycle='V')

    def print_residual(rk):
        print(f"|rk|/|r0| = {rk:.2e}")

    # it converges very slowly if you print..
    gmres_callback = print_residual
    # gmres_callback = None

    def cg_print_residual(xk):
        r = rhs - tacs_csr_mat @ xk
        norm = np.linalg.norm(r)
        print(f"residual = {norm:.2e}")

    cg_callback = cg_print_residual
    # cg_callback = None

    # solve_type = 'cg', 'gmres'
    solve_type = args.solver

    # precond = 'left', 'right'
    precond = args.precond

    # for right preconditioning, have to wrap the linear operator as so.. (scipy doesn't natively support right precond..)
    def matvec(v):
        return tacs_csr_mat @ M(v)
    A_right = spla.LinearOperator(shape=tacs_csr_mat.shape, matvec=matvec)
        
    # pyAMG linear solve (does CSR vs BSR matter, Dr. K thinks not)
    start_time = time.time()

    # init norm
    r = rhs # x0 = 0
    init_norm = np.linalg.norm(r)
    print(f"{init_norm=:.3e}")

    if precond == 'left':
        if solve_type == 'gmres':
            x, info = spla.gmres(tacs_csr_mat, rhs, M=M, rtol=1e-4, atol=1e0, maxiter=100, callback=gmres_callback)
        else:
            x, info = spla.cg(tacs_csr_mat, rhs, rtol=1e-8, maxiter=30, M=M, callback=cg_callback)   # solve with CG
    elif precond == 'right':
        # no precond input, embedded in A_right operator (so M=None input different here)
        if solve_type == 'gmres':
            y, info = spla.gmres(A_right, rhs, M=None, rtol=1e-4, atol=1e0, maxiter=100, callback=gmres_callback)
        else:
            y, info = spla.cg(A_right, rhs, rtol=1e-8, maxiter=30, M=None, callback=cg_callback)   # solve with CG
        # un-precond the solution (right precond)
        x = M(y)

    # check final residual
    r = rhs - tacs_csr_mat @ x
    final_norm = np.linalg.norm(r)
    print(f"{final_norm=:.2e}")

    amg_solve_time = time.time() - start_time
    print(f"{amg_solve_time=:.3e}")