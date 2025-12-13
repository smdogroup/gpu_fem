from scipy.sparse.linalg import gmres
import pyamg
import numpy as np
import time
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import matplotlib.pyplot as plt

def calculate_bandwidth(A):
    """Calculate the bandwidth of sparse matrix A."""
    A_coo = A.tocoo()
    return np.max(np.abs(A_coo.row - A_coo.col))

def block_shuffle_permutation(perm, block_size):
    """
    Shuffle the permutation `perm` in blocks of size `block_size`.
    Each block of indices is randomly shuffled in place.
    """
    perm = perm.copy()
    n = len(perm)
    num_blocks = n // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = perm[start:end]
        np.random.shuffle(block)
        perm[start:end] = block
    # Shuffle remaining elements if any
    if n % block_size != 0:
        block = perm[num_blocks * block_size :]
        np.random.shuffle(block)
        perm[num_blocks * block_size :] = block
    return perm

# inputs
# ------

# Generate 2D Poisson (Laplace) matrix using finite differences
# n = 50  # grid size (n x n)
# n = 1000
# n = 200
n = 300
# n = 500
A_csr = pyamg.gallery.poisson((n, n), format='csr')  # returns scipy.sparse.csr_matrix

# RHS: constant forcing
b = np.ones(A_csr.shape[0])

# Suppose A_bsr is your BSR matrix
# -------------------
# A_csr = A_bsr.tocsr()

# plt.spy(A_csr, markersize=0.1)
# plt.show()

# Solve with ILU-preconditioned GMRES
# -----------------------------------
if n <= 300: # regular ILU(0)
    start_ilu_factor = time.time()
    ilu = spla.spilu(A_csr.tocsc())  # ILU(0) is slower
    # ilu = spla.spilu(A_csr, fill_factor=10.0, drop_tol=1e-4)  # ILUT
    M_x = lambda x: ilu.solve(x)
    M = spla.LinearOperator(A_csr.shape, M_x)
    ilu_factor_time = time.time() - start_ilu_factor
    # print(f"{ilu_factor_time=:.3e}")
    start_ilu = time.time()
    x_ilu, info_ilu = spla.gmres(A_csr, b, M=M, rtol=1e-8)
    ilu_time = time.time() - start_ilu
    print(f"{ilu_time=:.3e}")
    # print("ILU(0)-GMRES [no reorder] residual norm:", np.linalg.norm(A_csr @ x_ilu - b))

# reordered ILU(0), qorder is not faster on CPU.. actually slower because can't parallelize as well as GPU?
start_ilu_factor = time.time()

# reordering (q-ordering)
p = 0.5
perm_rcm = reverse_cuthill_mckee(A_csr)
A_rcm = A_csr[perm_rcm, :][:, perm_rcm]
bandwidth_rcm = calculate_bandwidth(A_rcm)
# print(f"Bandwidth after RCM: {bandwidth_rcm}")
block_size = int(1/p * bandwidth_rcm)
# print(f"{block_size=}")
perm_q = block_shuffle_permutation(perm_rcm, block_size)
A_q = A_csr[perm_q, :][:, perm_q]

# plot q-ordered matrix
# plt.spy(A_q, markersize=0.1)
# plt.show()

# print(f"Bandwidth after block shuffle q-ordering: {calculate_bandwidth(A_q)}")

ilu = spla.spilu(A_q.tocsc())  # ILU(0) is slower
# ilu = spla.spilu(A_csr, fill_factor=10.0, drop_tol=1e-4)  # ILUT
M_x = lambda x: ilu.solve(x)
M = spla.LinearOperator(A_csr.shape, M_x)
ilu_factor_time = time.time() - start_ilu_factor
# print(f"{ilu_factor_time=:.3e}")
start_ilu = time.time()
x_ilu, info_ilu = spla.gmres(A_q, b, M=M, rtol=1e-8)
iluq_time = time.time() - start_ilu
print(f"{iluq_time=:.3e}")
# print("ILU(0)-GMRES [no reorder] residual norm:", np.linalg.norm(A_csr @ x_ilu - b))

# AMG solver
# ----------
# Build AMG solver
start_amg_factor = time.time()
ml = pyamg.smoothed_aggregation_solver(A_csr)
M_amg = ml.aspreconditioner()
amg_factor_time = time.time() - start_amg_factor
print(f"{amg_factor_time=:.3e}")
start_amg = time.time()
x_amg, info_amg = spla.gmres(A_csr, b, M=M_amg, rtol=1e-8)
amg_time = time.time() - start_amg
print(f"{amg_time=:.3e}")
# print("AMG-GMRES residual norm:", np.linalg.norm(A_csr @ x_amg - b))

# Solve system Ax = b (direct multigrid solver.. ?)
# x = ml.solve(b, tol=1e-10)

# direct solve for reference
# --------------------------
start_direct = time.time()
x_direct = spla.spsolve(A_csr, b)
direct_time = time.time() - start_direct
print(f"{direct_time=:.3e}")
# print("Direct residual norm:", np.linalg.norm(A_csr @ x_direct - b))