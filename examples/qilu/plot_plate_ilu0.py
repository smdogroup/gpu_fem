# adapt qordering and other codes from NASA FUN3D SFE + SLAT
# for qordering ILU(0)-GMRES with 6x6 block pivots..

# now let's test this out and visualize it
import numpy as np
import sys, scipy as sp
from __src import get_tacs_matrix, sort_vis_maps, plot_plate_vec
from __src import random_ordering, reorder_bsr6_nofill, right_pgmres, gen_plate_mesh
from __ilu_nasa import GaussJordanBlockPrecond, q_ordering
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--norandom", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
parser.add_argument("--nxe", type=int, default=30)
args = parser.parse_args()

gen_plate_mesh(nxe=args.nxe, lx=1.0, ly=1.0)

# ====================================================
# 1) create and assemble FEA problem
# ====================================================

thickness = args.thick

# load_fcn = None
load_fcn = lambda _x, _y : 1.0

A00, rhs00, xpts = get_tacs_matrix(bdf_file="plate.bdf", thickness=thickness, load_fcn=load_fcn)

# doesn't quite work because the matrix values are not computed to higher precision first?

# ===================================================
# 2) random reordering (instead of Qordering for now)
# ===================================================

np.random.rand(12345678)

N = A00.shape[0]
nnodes = N // 6
nnzb = A00.data.shape[0]
print(f"{nnodes=}")

# permute to lexigraphic ordering
# since TACS reads in a weird order
# ====================================
free_dof = [_ for _ in range(N)]
sort_fw, sort_bk = sort_vis_maps(args.nxe, xpts, free_dof)
perm = np.zeros(nnodes, dtype=np.int32)
iperm = np.zeros_like(perm)
# print(f"{sort_fw=}")
for i in range(perm.shape[0]):
    j = sort_fw[6 * i] // 6
    perm[i] = j
    iperm[j] = i
# print(f"{perm=}")
A0 = reorder_bsr6_nofill(A00.copy(), perm, iperm)
rhs0 = rhs00.reshape(nnodes, 6)[iperm].reshape(-1) 

if not args.norandom:
    print("doing random..")
    # perm, iperm = random_ordering(nnodes)
    # qorder_p = 0.5
    # qorder_p = 1.0
    qorder_p = 2.0

    perm, iperm = q_ordering(A0, prune_factor=qorder_p)
    A = reorder_bsr6_nofill(A0.copy(), perm, iperm)
    rhs = rhs0.reshape(nnodes, 6)[iperm].reshape(-1)
    # print(f"{A0.shape=} {A.shape=} {rhs.shape=}")
else:
    A = A0.copy()
    rhs = rhs0.copy()
    perm, iperm = np.arange(0, nnodes), np.arange(0, nnodes)

# ====================================================
# 3) direct solve baseline
# ====================================================

# x_perm = sp.sparse.linalg.spsolve(A.copy(), rhs.copy())
# x = x_perm.reshape(nnodes, 6)[perm].reshape(-1)

# equiv solution with no reorder
x = sp.sparse.linalg.spsolve(A0.copy(), rhs0.copy())


# for plotting
nxe = int(nnodes**0.5)-1
sort_fw = np.arange(0, sort_fw.shape[0])
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
plot_plate_vec(nxe, x.copy(), ax, sort_fw, nodal_dof=2)
# plt.show()

# =======================================================
# 4) block ILU(0) pivot factorization
# =======================================================

# precond = ILU_pivot_precond(A)
# from __ilu_nasa import GaussJordanBlockPrecond
precond = GaussJordanBlockPrecond(A)
    
# ========================================================
# 5) plot plate ILU(0) approx solution vs true soln
# ========================================================

x_perm2 = precond.solve(rhs)
x2 = x_perm2.reshape(nnodes, 6)[perm].reshape(-1)

# plot right-precond solution
ax = fig.add_subplot(122, projection='3d')
plot_plate_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2)
plt.show()