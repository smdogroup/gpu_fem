# test LU on plate with 128 nodes

# now let's test this out and visualize it
import numpy as np
import sys, scipy as sp
from __src import get_tacs_matrix, sort_vis_maps, random_ordering
from __src import reorder_bsr6_nofill, gen_plate_mesh
from __ilu import GaussJordanBlockPrecond, q_ordering, get_lu_residual
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--norandom", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--thick", type=float, default=2e-3) # 0.01
parser.add_argument("--nxe", type=int, default=40)
args = parser.parse_args()

gen_plate_mesh(nxe=args.nxe, lx=1.0, ly=1.0)

# ====================================================
# 1) create and assemble FEA problem
# ====================================================

thickness = args.thick

A00, rhs00, xpts = get_tacs_matrix(bdf_file="plate.bdf", thickness=thickness)

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
    perm, iperm = q_ordering(A0, prune_factor=0.5)
    A = reorder_bsr6_nofill(A0.copy(), perm, iperm)
    rhs = rhs0.reshape(nnodes, 6)[iperm].reshape(-1)
    # print(f"{A0.shape=} {A.shape=} {rhs.shape=}")
else:
    A = A0.copy()
    rhs = rhs0.copy()
    perm, iperm = np.arange(0, nnodes), np.arange(0, nnodes)


# =======================================================
# 4) block ILU(0) pivot factorization
# =======================================================

A_64 = A.copy()
precond_64 = GaussJordanBlockPrecond(A)

A_128 = A_64.copy()
A_128.data.astype(np.longdouble)
np.finfo(np.longdouble)
precond_128 = GaussJordanBlockPrecond(A_128)
# exit()


# normalize by max value in A (for relative errors)
A_dense = A_128.toarray()
# A_nrm = np.linalg.norm(A_dense)
A_nrm = np.max(np.abs(A_dense))

# check factor error
R_64 = get_lu_residual(A_128, precond_64.A)
R_64_dense = R_64.toarray()
R_64_dense[A_dense == 0] = 0.0 # tmp to check only error on sparsity
R_nrm_64 = np.linalg.norm(R_64_dense) / A_nrm
R_max_64 = np.max(np.abs(R_64_dense)) / A_nrm
R_64_dense[A_dense == 0] = np.nan
print(f"{R_nrm_64=:.4e} {R_max_64=:.4e} on sparsity")


R_128 = get_lu_residual(A_128, precond_128.A)
R_128_dense = R_128.toarray()
R_128_dense[A_dense == 0] = 0.0 # tmp to check only error on sparsity
R_nrm_128 = np.linalg.norm(R_128_dense) / A_nrm
R_max_128 = np.max(np.abs(R_128_dense)) / A_nrm
R_128_dense[A_dense == 0] = np.nan
print(f"{R_nrm_128=:.4e} {R_max_128=:.4e} on sparsity")

# for higher DOF problems, plot the residual by nodal rows
R_nodal_64_err = np.zeros(nnodes)
R_nodal_128_err = np.zeros(nnodes)
for idof in range(6*nnodes):
    i = idof // 6

    row = R_64_dense[idof,:]
    nz_row = row[np.logical_not(np.isnan(row))]
    nrm = np.linalg.norm(nz_row) / A_nrm
    R_nodal_64_err[i] = np.max([R_nodal_64_err[i], nrm])

    row = R_128_dense[idof,:]
    nz_row = row[np.logical_not(np.isnan(row))]
    nrm = np.linalg.norm(nz_row) / A_nrm
    R_nodal_128_err[i] = np.max([R_nodal_128_err[i], nrm])

plt.plot(np.arange(0, nnodes), R_nodal_64_err, label="64-bit")
plt.plot(np.arange(0, nnodes), R_nodal_128_err, label="128-bit")
plt.legend()
plt.yscale('log')
plt.show()

# # plt.imshow(np.log10(np.abs(A.toarray()) + 1e-14))
# # # plt.imshow(np.log10(np.abs(precond.A.toarray()) + 1e-14))
# # plt.imshow(np.log10(np.abs(R.toarray()) + 1e-14))
# # plt.show()