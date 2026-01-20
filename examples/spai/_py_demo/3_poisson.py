"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _poisson import *
# from _sparse_spai import SPAI_CSRSolver
from _sparse_spai_v2 import SPAI_CSRSolver
import sys
sys.path.append("../../milu/")
from __linalg import right_pgmres

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=30, help="num nodes in each direction")
parser.add_argument("--fill", type=int, default=2, help="number of fillin / SPAI iterations (no dropping)")
# parser.add_argument("--iters", type=int, default=2, help="num SPAI opt iters")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="just solve pc instead of GMRES")
args = parser.parse_args()

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
nx, ny = args.nx, args.nx

Lx, Ly = 1.0, 1.0
hx, hy = Lx/(nx-1), Ly/(ny-1)

A = poisson_2d_csr(nx, ny, hx, hy)

# RHS: smooth forcing
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# f = np.sin(np.pi * X) * np.sin(np.pi * Y)
f = np.sin(5 * np.pi * X) * np.sin(np.pi * (Y**2 + X**2))
f = f.ravel()

# ------------------------------------------------------------
# Apply homogeneous Dirichlet BCs (simple row overwrite)
# ------------------------------------------------------------
for j in range(ny):
    for i in range(nx):
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            k = i + j * nx
            # print(f"{i=} {j=} : {k=}")
            # A[k, :] = 0.0
            # A[k, k] = 1.0
            for jp in range(A.indptr[k], A.indptr[k+1]):
                col = A.indices[jp]
                if k == col:
                    A.data[jp] = 1.0
                else:
                    A.data[jp] = 0.0
            f[k] = 0.0

# A_nobc = remove_dirichlet_nondiag_entries(A)
A_nobc = A.copy()

# ------------------------------------------------------------
# Direct Solve
# ------------------------------------------------------------
u = spla.spsolve(A.copy(), f)
U = u.reshape((nx, ny))

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

if args.noprec:
    precond = None
else:
    # mode = "global"
    mode = "global-selfmr"
    precond = SPAI_CSRSolver(A_nobc, iters=args.fill, mode=mode) #, power=args.fill + 1

# # # DEBUG.. compare sparse vs dense SPAI preconditioners
# # THE REASON THE DENSE solver has almost infinite fillin
# # is because we do matrix-matrix products repeatedly and update M as we go (not in parallel)
# # need parallel version also..
# # that leads to huge and almost infinite fillin level for dense preconditioner..
# from _spai import SPAI_MR_SelfPrecond
# precond2 = SPAI_MR_SelfPrecond(A.copy(), iters=args.iters)
# M2 = precond.M.toarray()
# M1 = precond2.M.copy()
# # print(f"{M1=}")
# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].imshow(np.log10(np.abs(M1) + 1e-7)) # dense
# ax[1].imshow(np.log10(np.abs(M2) + 1e-7)) # sparse
# plt.show()


# maybe need some re-training still?
# how to make more parallel?
# cause current preconditioner uses serial training.. each column in order..
# and that messes up previous trainings..

if args.justpc:
    u2 = precond.solve(f)
else:
    u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)

# compute operator complexity
if precond is not None:
    opc = precond.operator_complexity
    print(f"{opc=:.4f}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
# plot_poisson_surface(u2, nx, ny, Lx, Ly)