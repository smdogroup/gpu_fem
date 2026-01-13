"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _spai import *
import sys
sys.path.append("../../milu/")
from __src import right_pgmres

def poisson_2d_csr(nx, ny, hx=1.0, hy=1.0):
    """
    2D Poisson matrix (-Laplace) with 1 DOF per node.
    Dirichlet BCs are NOT applied here (pure operator).
    
    Grid indexing: k = i + j*nx
    """
    N = nx * ny

    cx = 1.0 / hx**2
    cy = 1.0 / hy**2
    cc = 2.0 * (cx + cy)

    data = []
    indices = []
    indptr = [0]

    for j in range(ny):
        for i in range(nx):
            row_entries = {}

            k = i + j * nx
            row_entries[k] = cc

            if i > 0:        # left
                row_entries[k - 1] = -cx
            if i < nx - 1:   # right
                row_entries[k + 1] = -cx
            if j > 0:        # down
                row_entries[k - nx] = -cy
            if j < ny - 1:   # up
                row_entries[k + nx] = -cy

            # sort column indices (CSR best practice)
            for col in sorted(row_entries):
                indices.append(col)
                data.append(row_entries[col])

            indptr.append(len(indices))

    return sp.csr_matrix((data, indices, indptr), shape=(N, N))


def plot_poisson_surface(u, u2, nx, ny, Lx=1.0, Ly=1.0):
    """
    Plot a 2D scalar field stored as a flattened vector.

    Parameters
    ----------
    u  : (nx*ny,) array
        Solution vector, node ordering k = i + j*nx
    nx, ny : int
        Number of nodes in x and y
    Lx, Ly : float
        Domain size
    """
    U = u.reshape((nx, ny))
    U2 = u2.reshape((nx, ny))

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(X, Y, U, linewidth=0.5, antialiased=True, cmap='jet') #, cmap=plt.cmap.jet)

    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(X, Y, U2, linewidth=0.5, antialiased=True, cmap='jet') #, cmap=plt.cmap.jet)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("Poisson solution")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
# nx, ny = 50, 50
nx, ny = 30, 30

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

iters = 5
# iters = 10
# iters = 30

# precond = None
# precond = SPAI_Precond(A, iters=iters)
# precond = SPAI_MR_Precond(A, iters=iters)
precond = SPAI_MR_SelfPrecond(A, iters=iters)


# u2 = precond.solve(f)
u2 = right_pgmres(A, b=f, x0=None, restart=500, max_iter=500, M=precond)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
# plot_poisson_surface(u2, nx, ny, Lx, Ly)