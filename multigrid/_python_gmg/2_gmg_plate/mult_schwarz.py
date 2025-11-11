"""demo the multiplicative schwarz soln method for two subdomains"""
# if this works well (and moment bc works well for transmission), 
# then multigrid smoothers should be able to work well too

import numpy as np
import matplotlib.pyplot as plt
from src._dkt_plate_elem import *

"""compute true global plate solution first"""
nxe = 10 # small problem so easier to debug (but big enough so can get good overlap region)
nx = nxe + 1
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick

_K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = _K.shape[0]
_F = np.zeros(ndof)
_F[0::3] = 100.0 # vert load for bending..

K, F = apply_bcs(nxe, _K, _F)
u_truth = np.linalg.solve(K, F)

"""plot the plate disp"""

def plot_disp(soln, idof:int=0):
    x_plt, y_plt = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
    X, Y = np.meshgrid(x_plt, y_plt)
    w = soln[idof::3]
    W = np.reshape(w, (nx, nx))
    fig, ax = plt.subplots(1, 1)
    ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    ax0.plot_surface(X, Y, W)
    plt.show()

# # NOTE : this step was verified well, good disp
plot_disp(u_truth, idof=0)

"""plot the bending curvatures in the domain"""
# NOTE : this step was fully verified now
# elem_curvatures = get_elem_quantities(u_truth, nxe, E, thick, nu, name="curvature")
# for idof in range(3):
#     # fig, ax = plt.subplots(1, 2)
#     # for itri in range(2):
#     #     k_elem_mat = np.reshape(elem_curvatures[itri, idof, :], (nxe, nxe))
#     #     ax[itri].imshow(k_elem_mat)
#     # plt.show()

def plot_curvatures(soln, imom:int=0):
    elem_curvs = get_elem_quantities(soln, nxe, E, thick, nu, name="curvature")
    # moments and curvatures all check out now (there was bug before)
    M_elem_mat = np.reshape(elem_curvs[imom, :], (nxe, nxe))
    plt.imshow(M_elem_mat)
    plt.show()

def plot_moments(soln, imom:int=0):
    elem_moments = get_elem_quantities(soln, nxe, E, thick, nu, name="moment")
    # moments and curvatures all check out now (there was bug before)
    M_elem_mat = np.reshape(elem_moments[imom, :], (nxe, nxe))
    plt.imshow(M_elem_mat)
    plt.show()

def plot_laplacian(soln):
    elem_laplacians = get_elem_quantities(soln, nxe, E, thick, nu, name="laplacian")
    M_elem_mat = np.reshape(elem_laplacians, (nxe, nxe))
    plt.imshow(M_elem_mat)
    plt.show()

# for idof in range(3): 
#     plot_curvatures(u_truth, idof)
# for idof in range(3):
#     plot_moments(u_truth, idof)
# plot_laplacian(u_truth)
# exit()

"""try subdomain solves now.."""

# construct the two subdomain problems (first without subdomain bcs applied yet)
nx_list_0 = [_ for _ in range(0, 8)]
nx_list_1 = [_ for _ in range(3, 11)]
nodes_list_0 = [nx * iy + ix for iy in range(nx) for ix in nx_list_0]
nodes_list_1 = [nx * iy + ix for iy in range(nx) for ix in nx_list_1]
dofs_0 = np.array([3*_inode + idof for _inode in nodes_list_0 for idof in range(3)])
dofs_1 = np.array([3*_inode + idof for _inode in nodes_list_1 for idof in range(3)])
# print(f"{dofs_0=}")

# compute bcs in the full problem (for dirichlet bndry corrections..)
bc_nodes_0 = nx_list_0 + [nx*iy + ix for iy in range(1, nx-1) for ix in [0, 7]] + [_node + nx * (nx-1) for _node in nx_list_0]
bc_nodes_1 = nx_list_1 + [nx*iy + ix for iy in range(1, nx-1) for ix in [3, 10]] + [_node + nx * (nx-1) for _node in nx_list_1]
bc_dofs_0 = np.array([3*_inode for _inode in bc_nodes_0])
bc_dofs_1 = np.array([3*_inode for _inode in bc_nodes_1])

# # check subdomains here..
# sd_arr = np.zeros(nx**2)
# sd_arr[np.array(nodes_list_0)] += 1.0
# sd_arr[np.array(nodes_list_1)] += 2.0
# # sd_arr[np.array(bc_nodes_0)] += 5.0
# # sd_arr[np.array(bc_nodes_1)] += 10.0
# sd_mat = np.reshape(sd_arr, (nx, nx))
# plt.imshow(sd_mat)
# plt.show()

# no transmission bcs yet, this is original un-bc'ed problems here..
_K_0 = K[dofs_0, :][:, dofs_0]
_F_0 = F[dofs_0]
_K_1 = K[dofs_1, :][:, dofs_1]
_F_1 = F[dofs_1]

# setting on whether to apply moment penalty or not (just for DEBUG, need it on in final version)
moment_penalty = True
# moment_penalty = False

disp = np.zeros(ndof)
""" mult schwarz loop """
# N = 1
# N = 10
N = 50

defect = F.copy()

for i in range(N): 

    """ compute modified subproblem 1 (with transmission bcs applied) """
    defect = F - np.dot(K, disp)
    _F_0 = defect[dofs_0]
    K_0, F_0, int_dof_0 = apply_subdomain_bcs(nxe, E, thick, nu, _K_0, _F_0, disp[dofs_0], dofs_0, bc_dofs_0, moment_penalty=moment_penalty)

    # compute full bndry solution
    _u_0 = np.linalg.solve(K_0, F_0)
    disp[int_dof_0] += _u_0[:]

    # if i == N-1:
    if i >= 0:
        plot_disp(disp, 0)
        plot_laplacian(disp)
    
    # exit()
    defect_norm = np.linalg.norm(defect)
    print(f"{defect_norm=:.3e}")

    """ compute modified subproblem 2 (with transmission bcs applied) """
    defect_2 = F - np.dot(K, disp)
    _F_1 = defect_2[dofs_1]
    K_1, F_1, int_dof_1 = apply_subdomain_bcs(nxe, E, thick, nu, _K_1, _F_1, disp[dofs_1], dofs_1, bc_dofs_1, moment_penalty=moment_penalty)

    # compute full bndry solution
    _u_1 = np.linalg.solve(K_1, F_1)
    disp[int_dof_1] += _u_1[:]

    # if i == N-1:
    if i >= 0:
        plot_disp(disp, 0)
        plot_laplacian(disp)

    defect_2 = F - np.dot(K, disp)
    defect2_norm = np.linalg.norm(defect_2)
    print(f"{defect2_norm=:.3e}")
