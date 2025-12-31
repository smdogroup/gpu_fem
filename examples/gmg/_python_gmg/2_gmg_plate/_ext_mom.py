"""
get external moments to work..
"""

from src._dkt_plate_elem import *
from src.linalg import *
import numpy as np
import matplotlib.pyplot as plt

"""solve plate first"""
nxe = 32
# nxe = 64
# nxe = 8
nx = nxe + 1
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick

K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = K.shape[0]
F = np.zeros(ndof)
F[0::3] = 100.0 # vert load for bending..

K, F = apply_bcs(nxe, K, F)
u = np.linalg.solve(K, F)

"""plot the plate disp"""
x_plt, y_plt = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x_plt, y_plt)
w = u[0::3]
W = np.reshape(w, (nx, nx))
fig, ax = plt.subplots(1, 1)
ax0 = fig.add_subplot(1, 1, 1, projection='3d')
ax0.plot_surface(X, Y, W)
plt.show()

"""compute the bending moments along a few example elem boundaries (each side)"""

# choose two adjacent elements
# ixe1, iye1 = 2, 2
ixe1, iye1 = 4, 4
ixe2, iye2 = ixe1+1, iye1
ielem1 = nxe * iye1 + ixe1
ielem2 = nxe * iye2 + ixe2

ix1, iy1 = ixe1, iye1
inode1 = nx * iy1 + ix1
elem_nodes1 = [inode1+1, inode1+nx+1, inode1+nx]
elem_dof1 = np.array([3*_inode + _idof for _inode in elem_nodes1 for _idof in range(3)])
elem_disp1 = u[elem_dof1]
h = 1.0 / nxe
x_elem1, y_elem1 = h * np.array([1.0, 1.0, 0.0]), h * np.array([0.0, 1.0, 1.0])
# xi1, eta1 = 0.5, 0.0
xi1, eta1 = 0.0, 0.0
M_1 = get_quadpt_moments(elem_disp1, E, thick, nu, x_elem1, y_elem1, xi1, eta1)

ix2, iy2 = ixe2, iye2
inode2 = nx * iy2 + ix2
elem_nodes2 = [inode2, inode2+1, inode2+nx]
elem_dof2 = np.array([3*_inode + _idof for _inode in elem_nodes2 for _idof in range(3)])
elem_disp2 = u[elem_dof2]
x_elem2, y_elem2 = h * np.array([0.0, 1.0, 0.0]), h * np.array([0.0, 0.0, 1.0])
# xi2, eta2 = 0.0, 0.5
xi2, eta2 = 0.0, 0.0

M_2 = get_quadpt_moments(elem_disp2, E, thick, nu, x_elem2, y_elem2, xi2, eta2) 

print(f"{elem_nodes1=} {elem_nodes2=}")

print(f"{M_1=}")
print(f"{M_2=}")