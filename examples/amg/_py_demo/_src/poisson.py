import numpy as np
import scipy.sparse as spp
import matplotlib.pyplot as plt

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

    return spp.csr_matrix((data, indices, indptr), shape=(N, N))

def poisson_apply_bcs(A, f, nx, ny):
    A2 = A.copy()
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                k = i + j * nx
                A2[k, :] = 0.0
                A2[k, k] = 1.0
                f[k] = 0.0
    return A2

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