import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

def _get_dof(ix, iy, nx):
    return nx * iy + ix

def get_bcs_and_free_dof(nx):
    ndof = nx**2
    bcs = [_get_dof(0, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(nx-1, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(ix, 0, nx) for ix in range(1, nx-1)]
    bcs += [_get_dof(ix, nx-1, nx) for ix in range(1, nx-1)]
    free_dof = np.array([_ for _ in range(ndof) if not(_ in bcs)])
    return bcs, free_dof

def get_poisson_lhs_and_rhs(nxe):
    """
    get the left-hand side of the 2D Poisson equation, 
        -nabla u = -(u_xx + u_yy) = f_{int}(x,y)
        u(x,y) = u_{ext}(x,y) [Dirichlet nonzero bcs]

    with method of manufactured solutions to get analytic eqn and the forces for it..
        u(x,y) = e^(x * y) => f_{int}(x,y) = -2*u = -2 * e^{x*y}
    """
    nx = nxe + 1
    ndof = nx**2 # square grid
    # TODO : I could do sparse csr matrix, but I won't (just doing dense)
    # I only care about the multigrid method itself (then I'll put it into gpu_fem later for shells)
    A = np.zeros((ndof,ndof))
    b = np.zeros(ndof)
    h = 1.0 / (nx-1) # grid spacing (for 1 x 1 unit square plate)

    # only looping over interior points here.. (not boundary)
    for iy in range(1, nx-1):
        for ix in range(1, nx-1):
            row = _get_dof(ix, iy, nx)
            A[row, row] += 4.0 / h**2

            if ix > 1:
                col1 = _get_dof(ix-1, iy, nx)
                A[row, col1] += -1.0 / h**2
            
            if ix < nx-1:
                col2 = _get_dof(ix+1, iy, nx)
                A[row, col2] += -1.0 / h**2
            
            if iy > 1:
                col3 = _get_dof(ix, iy-1, nx)
                A[row, col3] += -1.0 / h**2
            
            if iy < nx - 1:
                col4 = _get_dof(ix, iy+1, nx)
                A[row, col4] += -1.0 / h**2

    # plt.imshow(A)
    # plt.show()

    # now also get RHS, for u(x,y) = e^(x*y), f(x,y) = -2 * exp(x*y) on interior
    # and on the interior nodes (near exterior), add u_bndry(x,y) / h^2
    for iy in range(1, nx-1):
        for ix in range(1, nx-1):
            # x and y coords for disp
            x = ix * h
            y = iy * h 

            ind = _get_dof(ix, iy, nx)
            b[ind] = -2.0 * np.exp(x * y)

            # boundary forces to interior
            if ix == 1:
                b[ind] += np.exp((x - h) * y) / h**2

            if ix == nx-2:
                b[ind] += np.exp((x + h) * y) / h**2

            if iy == 1:
                b[ind] += np.exp(x * (y - h)) / h**2

            if iy == nx-2:
                b[ind] += np.exp(x * (y + h)) / h**2

    # remove the bcs from the matrix and rhs
    _, free_dof = get_bcs_and_free_dof(nx)
    
    A_free = A[free_dof, :][:, free_dof]
    b_free = b[free_dof] 

    return A_free, b_free
               
def plot_poisson_soln(nxe, _soln):
    nx = nxe+1
    ndof = nx**2 # square grid
    h = 1.0 / nxe # grid spacing (for 1 x 1 unit square plate)

    # get bcs again and free dof
    bcs = [_get_dof(0, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(nx-1, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(ix, 0, nx) for ix in range(1, nx-1)]
    bcs += [_get_dof(ix, nx-1, nx) for ix in range(1, nx-1)]
    free_dof = np.array([_ for _ in range(ndof) if not(_ in bcs)])

    # true soln
    true_soln = np.zeros(ndof)
    for iy in range(1, nx-1):
        for ix in range(1, nx-1):
            # x and y coords for disp
            x = ix * h
            y = iy * h 

            ind = _get_dof(ix, iy, nx)
            true_soln[ind] = np.exp(x * y)

    # soln with disps on boundary
    soln = np.zeros(ndof)
    soln[free_dof] = _soln

    # then soln on the boundary
    bcs = np.array(bcs)
    soln[bcs] = true_soln[bcs]

    # soln norm
    soln_diff = soln - true_soln
    soln_err_norm = np.max(np.abs(soln - true_soln))
    print(f"{soln_diff=}")
    print(f"{soln_err_norm=}")

    # now plot the two solutions on side by side grids
    X = np.zeros((nx, nx))
    Y = np.zeros((nx, nx))
    for iy in range(nx):
        for ix in range(nx):
            X[ix, iy] = ix * h
            Y[ix, iy] = iy * h

    SOLN = np.reshape(soln, (nx, nx))
    TRUE_SOLN = np.reshape(true_soln, (nx, nx))

    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

    # max and min same for both
    _max = max([np.max(soln), np.max(true_soln)])
    _min = min([np.min(soln), np.min(true_soln)])

    fig, ax = plt.subplots(1, 2, figsize=(13, 7))
    # Plot the true solution.
    # cf1 = ax[0].contourf(X, Y, TRUE_SOLN)
    c = ax[0].pcolormesh(X, Y, TRUE_SOLN, shading='auto', cmap=None,
                  edgecolors='k', linewidth=0.5, vmin=_min, vmax=_max)
    ax[0].set_title("True soln")
    
    # Plot the predicted solution.
    # cf2 = ax[1].contourf(X, Y, SOLN)
    # SOLN - TRUE_SOLN # small error
    c = ax[1].pcolormesh(X, Y, SOLN, shading='auto', cmap=None,
                  edgecolors='k', linewidth=0.5, vmin=_min, vmax=_max)
    ax[1].set_title("Pred soln")
    
    # Create a divider for the axis on the right (ax[1]) and append a new axis for the colorbar.
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(c, cax=cax)

    plt.show()

def plot_error_comparison(nxe, _error1, _error2, filename=None):
    nx = nxe+1
    ndof = nx**2 # square grid
    h = 1.0 / (nx-1) # grid spacing (for 1 x 1 unit square plate)

    # get bcs again and free dof
    bcs = [_get_dof(0, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(nx-1, iy, nx) for iy in range(nx)]
    bcs += [_get_dof(ix, 0, nx) for ix in range(1, nx-1)]
    bcs += [_get_dof(ix, nx-1, nx) for ix in range(1, nx-1)]
    free_dof = np.array([_ for _ in range(ndof) if not(_ in bcs)])

    # extend errors to full-space
    error1 = np.zeros(ndof)
    error1[free_dof] = _error1

    error2 = np.zeros(ndof)
    error2[free_dof] = _error2

    # now plot the two solutions on side by side grids
    X = np.zeros((nx, nx))
    Y = np.zeros((nx, nx))
    for iy in range(nx):
        for ix in range(nx):
            X[ix, iy] = ix * h
            Y[ix, iy] = iy * h

    ERROR1 = np.reshape(error1, (nx, nx))
    ERROR2 = np.reshape(error2, (nx, nx))

    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

    # max and min same for both
    _max = max([np.max(error1), np.max(error2)])
    _min = min([np.min(error1), np.min(error2)])

    fig, ax = plt.subplots(1, 2, figsize=(13, 7), subplot_kw={'projection': '3d'})
    # Plot the true solution.
    cf1 = ax[0].plot_surface(X, Y, ERROR1, vmin=_min, vmax=_max)
    ax[0].set_title("Error init")
    
    # Plot the predicted solution.
    cf2 = ax[1].plot_surface(X, Y, ERROR2, vmin=_min, vmax=_max)
    ax[1].set_title("Final err")
    
    # Create a divider for the axis on the right (ax[1]) and append a new axis for the colorbar.
    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(cf1) #, cax=cax)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)