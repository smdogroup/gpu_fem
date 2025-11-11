import numpy as np
import matplotlib.pyplot as plt
from _src import *

""" check lagrange interpolation basis """
# 9 point lagrange basis points
w_vals = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [7, 8, 9],
]).reshape((9,1))
_xi, _eta = np.linspace(-1, 1, 100), np.linspace(-1, 1, 50)
XI, ETA = np.meshgrid(_xi, _eta)
W = np.zeros_like(XI)

for i in range(50): # eta first
    for j in range(100):
        W[i,j] = lagrange_basis_2d(XI[i,j], ETA[i,j], w_vals)

# fig, ax = plt.subplots(1, 1)
# cax = fig.add_subplot(1,1,1, projection='3d')
# cax.plot_surface(XI, ETA, W)
# cax.set_title("w(xi,eta) Lagrange2D")
# plt.show()

"""now check the different types of strains"""

# NOTE : you choose idof here.. INPUT
# choose u,v,w, thx, thy to perturb
# idof = 0 
# idof = 1
idof = 2

elem_vars = np.zeros(45)
# nodal_disps = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [10, 14, 17]
# ])
nodal_disps = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
# nodal_disps = nodal_disps.T
nodal_disps = nodal_disps.flatten()

# currently copy nodal disps into one nodal DOF on each node
elem_vars[idof::5] = nodal_disps[:] # u (good)

# some input settings for strains
shell_xi_axis = np.array([1, 0, 0]).astype(np.double)
xpts = np.zeros(27)
h = 0.1
for inode in range(3):
    x = h * inode
    for jnode in range(3):
        y = h * jnode
        z = 0.0
        node = 3 * jnode + inode
        xpts[3*node:(3*node+3)] = np.array([x, y, z])[:]

# now compute the strains at each quadpt
QUADPT_STRAINS = np.zeros((8, 50, 100))
for i in range(50):
    for j in range(100):
        xi, eta = XI[i,j], ETA[i,j]
        _strains = get_quadpt_strains(shell_xi_axis, xi, eta, xpts, elem_vars)
        QUADPT_STRAINS[:, i, j] = _strains[:]

DISPS = np.zeros_like(XI)

for i in range(50): # eta first
    for j in range(100):
        DISPS[i,j] = lagrange_basis_2d(XI[i,j], ETA[i,j], nodal_disps)

disp_strs = ['u', 'v', 'w', 'thx', 'thy']
disp_name = disp_strs[idof]

# # now plot the quadpt strains
strains_str = ['ex', 'ey', 'exy', 'kx', 'ky', 'kxy', 'gxz', 'gyz']

# fig, ax = plt.subplots(3,3)
# cax = fig.add_subplot(3, 3, 1, projection='3d')
# cax.plot_surface(XI * h, ETA * h, DISPS)
# cax.set_title(f"{disp_name} DISP")

# for j in range(3):
#     for i in range(3):
#         iax = j * 3 + i
#         if iax == 8: continue
#         cax = fig.add_subplot(3, 3, iax+2, projection='3d')
#         cax.plot_surface(XI, ETA, QUADPT_STRAINS[iax])
#         cax.set_title(strains_str[iax])
# plt.show()

# exit()

"""now solve the disp"""

# NOTE : input num elements here..
nxe = 8

nx = 2 * nxe + 1
nnodes = nx**2
ndof = 5 * nnodes
x, y = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x, y)

# metal
E, nu, thick = 70e9, 0.3, 1e-2

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
plt.show()
# plt.tight_layout()
# plt.savefig(f"_out/0_{SR=}_disp.svg")