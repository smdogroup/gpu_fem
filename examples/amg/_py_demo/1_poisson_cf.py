"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
sys.path.append("../../milu/")
# from __src import right_pgmres
from __linalg import right_pcg
sys.path.append("_src/")
from poisson import poisson_2d_csr, plot_poisson_surface, poisson_apply_bcs
from cf_coarsening import RS_csr_coarsening, standard_csr_coarsening, aggressive_A2_csr_coarsening, aggressive_A1_csr_coarsening


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=30, help="num elements each direction")
parser.add_argument("--smoothP", type=int, default=1, help="whether to smooth prolongation or not")
parser.add_argument("--debug", type=int, default=0, help="debug printouts")
parser.add_argument("--crs", type=str, default="standard", help="string: type of coarsening [rs, standard, A1, A2]")
args = parser.parse_args()

# ------------------------------------------------------------
# 0) Problem setup
# ------------------------------------------------------------
nx, ny = args.nxe, args.nxe

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
u = spla.spsolve(A.copy(), f)
U = u.reshape((nx, ny))


# ------------------------------------------------------------
# 2) CF_
# ------------------------------------------------------------

nnodes = A.shape[0]

# C1-style coarsening from Ruge-Stuben
# should do it without BCs applied yet? yeah probably
if args.crs == "rs":
    C_mask, F_mask = RS_csr_coarsening(A_free, threshold=0.25)
elif args.crs == "standard":
    C_mask, F_mask = standard_csr_coarsening(A_free, threshold=0.25)
elif args.crs == "A1":
    C_mask, F_mask = aggressive_A1_csr_coarsening(A_free, threshold=0.25)
elif args.crs == "A2":
    C_mask, F_mask = aggressive_A2_csr_coarsening(A_free, threshold=0.25)


# C_mask, F_mask = C1_csr_coarsening(A, threshold=0.25)

# Acc = A_free[C_mask,:][:,C_mask]
# print(f"{Acc.shape=} {type(Acc)=}")

# plot C and F nodes here on the grid
x = np.linspace(0.0, Lx, nx)
y = np.linspace(0.0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")

Xf = X.ravel()
Yf = Y.ravel()


num_C_nodes = np.sum(C_mask)
num_F_nodes = np.sum(F_mask)
print(f"{num_C_nodes=} {num_F_nodes=}")


fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    Xf[C_mask], Yf[C_mask],
    c="tab:blue", s=40, label="C nodes"
)

ax.scatter(
    Xf[F_mask], Yf[F_mask],
    c="tab:red", s=40, label="F nodes"
)

ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"C/F Splitting ({args.crs})")
ax.legend()

plt.show()

# TODO : Not quite done with this demo yet


# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

# precond = None

# # u2 = precond.solve(f)
# # u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)
# u2 = right_pcg(A, b=f, x0=None, M=precond)


# # ------------------------------------------------------------
# # Plot
# # ------------------------------------------------------------
# plot_poisson_surface(u, u2, nx, ny, Lx, Ly)