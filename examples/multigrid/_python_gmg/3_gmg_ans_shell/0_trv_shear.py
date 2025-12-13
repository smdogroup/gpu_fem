import numpy as np
import matplotlib.pyplot as plt
from _src import *
import argparse

"""argparse inputs"""
parser = argparse.ArgumentParser()
parser.add_argument("--SR", type=float, default=100.0, help="slenderness ratio of plate L/h")
parser.add_argument("--plot", type=int, default=0, help="can plot intermediate results")
args = parser.parse_args()

SR = int(args.SR) # assume it is an int..
thick = 1.0 / args.SR


"""now solve the disp"""

# NOTE : input num elements here..
nxe = 8

nx = 2 * nxe + 1
nnodes = nx**2
ndof = 5 * nnodes
x, y = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x, y)

# metal
E, nu = 70e9, 0.3

_K = get_plate_K_global(nxe, E, nu, thick)
_F = get_global_plate_loads(nxe, load_type="sine-sine", magnitude=1.0)

# rescale the rhs by thickness**3 so that same disp across different slenderness roughly
_F *= 1e9 * thick**3

# remove all bcs here from the matrix
K, F = apply_bcs(nxe, _K, _F) # this version just zeros out in place
# K, F = remove_bcs(nxe, _K, _F)

# plt.imshow(K)
# plt.show()

u = np.linalg.solve(K, F)

W = u[2::5].reshape((nx,nx))
THX = u[3::5].reshape((nx,nx))
THY = u[4::5].reshape((nx,nx))

# plot the solved disps now..
disps_str = ['w', 'thx' ,'thy']
fig, ax = plt.subplots(1,3, figsize=(12, 6))
for j, VALS in enumerate([W, THX, THY]):
    cax = fig.add_subplot(1, 3, j+1, projection='3d')
    cax.plot_surface(X, Y, VALS, cmap='jet' if j == 0 else 'viridis')
    cax.set_title(disps_str[j])
# plt.show()
plt.tight_layout()
plt.savefig(f"_out/0_{SR=}_disp.svg")

"""now check the gam13 and gam23 transverse shear strains in one of the elements"""
ixe, iye = 2, 2 # just try not to choose on bndry.. (bndry has smaller trv shear sometimes want to make sure small in interior)
node = 2 * nx * iye + 2 * ixe # starting node of elem
elem_nodes = [node+nx*iy+ix for iy in range(3) for ix in range(3)]
elem_dof = np.array([5 * _node + _dof for _node in elem_nodes for _dof in range(5)])
elem_disps = u[elem_dof]

# include integration points in the transv shear here
rt_35 = np.sqrt(3.0 / 5.0)
_xi = np.array([-1.0, -0.9, -rt_35, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, rt_35, 0.9, 1.0])
_eta = _xi.copy()
n_xi, n_eta = _xi.shape[0], _xi.shape[0]
XI, ETA = np.meshgrid(_xi, _eta)

# same elem xpts for everything in this elem
elem_xpts = np.zeros(27)
h = 1.0 / (nx - 1)
for inode in range(3):
    x = h * inode
    for jnode in range(3):
        y = h * jnode
        z = 0.0
        node = 3 * jnode + inode
        elem_xpts[3*node:(3*node+3)] = np.array([x, y, z])[:]

shell_xi_axis = np.array([1.0, 0.0, 0.0])
GAM_13, GAM_23 = np.zeros_like(XI), np.zeros_like(ETA)
for i in range(n_xi):
    for j in range(n_xi):
        xi, eta = XI[i,j], ETA[i,j]
        _strains = get_quadpt_strains(shell_xi_axis, xi, eta, elem_xpts, elem_disps)
        GAM_13[i,j], GAM_23[i,j] = _strains[6], _strains[7]

# plot the transv shear strains now (first in 3d)
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

gam_strs = ['gam13', 'gam23']
for i, VALS in enumerate([GAM_13, GAM_23]):

    cax = fig.add_subplot(1, 2, i+1, projection='3d')
    LOG_VALS = np.log10(np.abs(VALS))
    cax.plot_surface(XI, ETA, LOG_VALS, cmap='viridis')
    cax.set_title(f"log10 {gam_strs[i]}(xi,eta)", fontweight='bold')
    cax.set_xlabel("XI", fontweight='bold')
    cax.set_ylabel("ETA", fontweight='bold')

plt.tight_layout()
plt.savefig(f"_out/1_{SR=}_gam_strains.svg")
# plt.show()

# then also plot them in contour fill form
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

gam_strs = ['gam13', 'gam23']
for i, VALS in enumerate([GAM_13, GAM_23]):

    # cax = fig.add_subplot(1, 2, i+1) #, projection='3d')
    LOG_VALS = np.log10(np.abs(VALS))
    cf = ax[i].contourf(XI, ETA, LOG_VALS, cmap='viridis')
    ax[i].set_title(f"log10 {gam_strs[i]}(xi,eta)", fontweight='bold')
    ax[i].set_xlabel("XI", fontweight='bold')
    ax[i].set_ylabel("ETA", fontweight='bold')
    fig.colorbar(cf)

plt.tight_layout()
plt.savefig(f"_out/2_{SR=}_gam_strains_cf.svg")