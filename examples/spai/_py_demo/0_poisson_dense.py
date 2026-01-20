"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _poisson import *
from _spai import *
import sys
sys.path.append("../../milu/")
from __linalg import right_pgmres


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=30, help="num nodes in each direction")
parser.add_argument("--iters", type=int, default=5, help="num SPAI opt iters")
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
            A[k, :] = 0.0
            A[k, k] = 1.0
            f[k] = 0.0

# ------------------------------------------------------------
# Direct Solve
# ------------------------------------------------------------
u = spla.spsolve(A.copy(), f)
U = u.reshape((nx, ny))

# ------------------------------------------------------------
# GMRES Solve
# ------------------------------------------------------------

# precond = None
# precond = SPAI_Precond(A, iters=args.iters)
# precond = SPAI_MR_Precond(A, iters=args.iters)
precond = SPAI_MR_SelfPrecond(A, iters=args.iters)


if args.justpc:
    u2 = precond.solve(f)
else:
    u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
# plot_poisson_surface(u2, nx, ny, Lx, Ly)