"""basic SPAI demo for a Poisson linear system.."""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../spai/_py_demo/")
from _poisson import *

sys.path.append("_src/")
from asw import OnedimAddSchwarz # works like line smoother
from asw import TwodimAddSchwarz

sys.path.append("../../milu/")
from __linalg import right_pgmres

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=80, help="num nodes in each direction") # can easily solve higher DOF like nx=100 with Schwarz smoother
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--size", type=int, default=2, help="coupling size of schwarz smoother")
parser.add_argument("--omega", type=float, default=0.25, help="additive coefficient for ASW")
parser.add_argument("--iters", type=int, default=5, help="num schwarz smooths per krylov step (more than one allowed)")
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
    pc = None
else:
    # pc = OnedimAddSchwarz(A, f, block_dim=1, 
    #                   coupled_size=args.size, omega=args.omega, iters=args.iters)
    
    pc = TwodimAddSchwarz(A, f, block_dim=1, nx=nx, ny=ny, 
                      coupled_size=args.size, omega=args.omega, iters=args.iters)

if args.justpc:
    u2 = pc.solve(f)
else:
    start_time = time.time()
    u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=pc)
    dt = time.time() - start_time
    print(f"GMRES solve in {dt=:.4e} sec")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
# plot_poisson_surface(u2, nx, ny, Lx, Ly)