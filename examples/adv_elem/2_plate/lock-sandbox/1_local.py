import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sp
sys.path.append("../src/")
from std_assembler import StandardPlateAssembler
from elem import MITCPlateElement_OptProlong

"""
goal of this method is to solve the locking-aware system approximately using local operators / jacobi smoothing with mat-mat products
    * in a way that I can implement on the GPU reasonably well.. and fast
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitcp', help="--elem, options: mitcp is another good one")
# parser.add_argument("--nxe", type=int, default=32, help="number of elements")
parser.add_argument("--nxe", type=int, default=8, help="number of elements")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
# parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
# parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
# parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
# parser.add_argument("--nsmooth", type=int, default=4, help="number of smoothing steps")
# parser.add_argument("--omega", type=float, default=1.0, help="omega smoother coeff (sometimes needs to be lower)")
# parser.add_argument("--smoother", type=str, default='supp_asw', help="--smooth : [gs, asw, supp_asw]")
# parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
# parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
# parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()

""" verify each beam element and solver type against truth """

ELEMENT = MITCPlateElement_OptProlong(
    prolong_mode='locking-global', # best..
    # prolong_mode='locking-local', # working reasonably..
    # prolong_mode='standard',
    # lam=1e-12,
    lam=1e-6,
    # # n_lock_sweeps=4,
    # # n_lock_sweeps=8,
    # n_lock_sweeps=10,
)
    

# ================================
# make plate assembler
# ================================

# clamped = True
clamped = False # simply supported
load_fcn = lambda x,y : 1.0e2 # simple load


ASSEMBLER = StandardPlateAssembler
f_assembler = ASSEMBLER(
    ELEMENT=ELEMENT,
    nxe=args.nxe,
    thick=args.thick,
    clamped=clamped,
    split_disp_bc=args.elem in ['hhd', 'higd'],
    load_fcn=load_fcn,
)
f_assembler._assemble_system()
f_rhs = f_assembler.force.copy()
f_kmat = f_assembler.kmat.copy()


c_assembler = ASSEMBLER(
    ELEMENT=ELEMENT,
    nxe=args.nxe // 2,
    thick=args.thick,
    clamped=clamped,
    split_disp_bc=args.elem in ['hhd', 'higd'],
    load_fcn=load_fcn,
)
c_assembler._assemble_system()

# # call restrict defect (part of multigrid process)
# c_defect = c_assembler.restrict_defect(f_rhs)

nxe = args.nxe
nxe_c = nxe // 2

P_standard = ELEMENT._build_P2_uncoupled3(nxe_c)
P_standard = ELEMENT._apply_bcs_to_P(P_standard, nxe_c)
ELEMENT._lock_P_cache = {} # reset

P_global = ELEMENT._locking_aware_prolong_global_mitc_v1(nxe_c, length=1.0)
ELEMENT._lock_P_cache = {} # reset

# if omega too high it makes defects worse.. and soln worse..
# P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=1.5)
# P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=0.7)
# P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=0.5)
P_local = ELEMENT._locking_aware_prolong_local_mitc_v3_jacobi(nxe_c, n_sweeps=10, omega=0.5)
ELEMENT._lock_P_cache = {} # reset

# ==========================================================================
# that jacobi solver doesn't match the global one like at all.. (esp BCs are wrong near boundary)
# write my own jacobi solver based on data from P_global stored in ELEMENT
# ==========================================================================

G_f = ELEMENT.G_f
G_c = ELEMENT.G_c
P_gam = ELEMENT.P_gam
LHS = ELEMENT.M
RHS = ELEMENT.RHS
free_cols_c = ELEMENT.free_cols_c
fixed_cols_c = ELEMENT.fixed_cols_c
solve_rows_f = ELEMENT.solve_rows_f
P0_free = ELEMENT.P_0_free


# truth, direct solve
# this gives correct solution
# X = np.linalg.solve(LHS, RHS) 

# let's now try and solve it with fixed-sparsity jacobi
P = P0_free.copy()

# S_lock = RHS.tocsr()
RHS = np.asarray(RHS)                 # ensure ndarray
S_lock_csr = sp.csr_matrix(LHS @ RHS)
# print(f"{S_lock_csr.shape=}")
# S_lock = S_lock_csr.tobsr(blocksize=(3, 3))
mask = (S_lock_csr != 0).astype(np.int8)
# mask = mask.tocsr()
# print(f"{type(mask)=}")
mask = mask.toarray()
# mask = None
# print(f"{mask.shape=}")

# def sparse_control(Xsp):
#     return Xsp.multiply(mask)

def sparse_control(A):
    """
    Apply an elementwise mask to A.
    - If A is sparse: keep it sparse via .multiply(mask)
    - If A is dense: do dense elementwise multiply via A * mask
    """
    # print(f"{A.shape=} {mask.shape=}")
    if sp.issparse(A):
        return A.multiply(mask)   # mask can be dense or sparse; dense is fine
    else:
        return np.asarray(A) * mask


# build Dinv matrix
block_size = 3
n_unknown = P.shape[0]
n_blk = n_unknown // block_size
# Mb = LHS.tobsr(blocksize=(block_size, block_size)).tocsr().tobsr(blocksize=(block_size, block_size))
Mb = sp.csr_matrix(LHS).tobsr(blocksize=(block_size, block_size))
# diagonal blocks (n_blk, bs, bs)
diag_blocks = np.zeros((n_blk, block_size, block_size), dtype=LHS.dtype)

# one-time gather + invert
for i in range(n_blk):
    start, end = Mb.indptr[i], Mb.indptr[i + 1]
    cols = Mb.indices[start:end]
    data = Mb.data[start:end]   # (nblocks_in_row, bs, bs)

    k = np.searchsorted(cols, i)
    if k >= cols.size or cols[k] != i:
        raise RuntimeError("Missing diagonal block in M (unexpected for SPD normal matrix).")

    Db = data[k]
    Db = 0.5 * (Db + Db.T)
    diag_blocks[i, :, :] = np.linalg.inv(Db)

# Build a *diagonal* BSR using (data, indices, indptr) form
Dinv_indptr  = np.arange(n_blk + 1, dtype=np.int32)   # one block per block-row
Dinv_indices = np.arange(n_blk,     dtype=np.int32)   # diagonal column index
Dinv_data    = diag_blocks                              # (nnz_blocks=n_blk, bs, bs)

Dinv_op = sp.bsr_matrix(
    (Dinv_data, Dinv_indices, Dinv_indptr),
    shape=(n_unknown, n_unknown),
    blocksize=(block_size, block_size),
).tocsr()

# Init X as sparse, projected to mask
X = sparse_control(sp.csr_matrix(P))
# X = sp.csr_matrix(P)

# -----------------------------
# Loop-free per-sweep Jacobi: ONLY SpMM operations
# -----------------------------
# n_sweeps = 20
# n_sweeps = 40
n_sweeps = 10
# n_sweeps = 4
# n_sweeps = 2
omega = 0.5
# with_fillin = True
with_fillin = False

print(type(LHS))
print(type(X))
# print(f"{LHS.__dict__=}")
print(f'{LHS.shape=}')
print(f'{X.shape=}')

# allow fillin here.. (to compare)
if with_fillin:
    for _ in range(n_sweeps):
        MX  = LHS @ X             # SpMM
        RES = RHS - MX             # axpy + mask
        X   = (X + omega * (Dinv_op @ RES))  # SpMM + axpy + mask

else:
    for _ in range(n_sweeps):
        MX  = sparse_control(LHS @ X)                 # SpMM
        RES = sparse_control(RHS - MX)              # axpy + mask
        X   = sparse_control(X + omega * (Dinv_op @ RES))  # SpMM + axpy + mask

P = X

# Assemble full P
P_local_v2 = P_standard.copy().toarray()
P_local_v2[:, fixed_cols_c] = 0.0
P_local_v2[np.ix_(solve_rows_f, free_cols_c)] = P
# P[fixed_rows_f, :] = 0.0


# ===============================================================================
# END OF MY JACOBI SOLVER
# ===============================================================================


# now compare prolongations
c_soln = c_assembler.direct_solve()
nx_f = nxe + 1
dx_f = 1.0 / nxe

names = ['standard', 'global', 'local', 'local2']
P_mats = [P_standard, P_global, P_local, P_local_v2]

fig, ax = plt.subplots(4, 2, figsize=(12, 10), subplot_kw={'projection' : '3d'})

for i in range(4):
    name = names[i]
    P_mat = P_mats[i]

    f_soln = P_mat @ c_soln
    f_def = f_rhs - f_kmat @ f_soln

    w_soln = f_soln[0::3]
    w_def = f_def[0::3]
    W_soln = w_soln.reshape((nx_f, nx_f))
    W_def = w_def.reshape((nx_f, nx_f))
    x = np.arange(nx_f) * dx_f
    y = np.arange(nx_f) * dx_f
    X, Y = np.meshgrid(x, y, indexing="xy")

    surf1 = ax[i, 0].plot_surface(X, Y, W_soln, cmap="viridis", 
                                 linewidth=0, antialiased=True, shade=True)
    
    surf2 = ax[i, 1].plot_surface(X, Y, W_def, cmap="viridis", 
                                 linewidth=0, antialiased=True, shade=True)


plt.savefig("p_sandbox.png", dpi=400)
plt.show()



fig, ax = plt.subplots(4, 1, figsize=(12, 10)) #, subplot_kw={'projection' : '3d'})

for i in range(4):
    # print(f"{i=}")
    P_mat = P_mats[i].copy()
    mat = P_mat[:6,:6]

    if sp.isspmatrix_csr(mat):
        np_mat = mat.toarray()
    else:
        np_mat = mat
    
    # print(f"{mat.__dict__}")
    # np_mat = mat.toarray()
    # np_mat = mat.array()
    ax[i].imshow(np_mat)


plt.savefig("p_mat.png", dpi=400)
plt.show()