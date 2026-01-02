import numpy as np
import matplotlib.pyplot as plt
from _src import *


# nxe = 2
# nxe = 4
# nxe = 8
nxe = 16
# nxe = 32


nx = 2 * nxe + 1
x, y = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x, y)

E, nu, thick = 2e7, 0.3, 1e-2

_K = get_plate_K_global(nxe, E, nu, thick)
_F = get_global_plate_loads(nxe, load_type="sine-sine", magnitude=1.0)
K, F = apply_bcs(nxe, _K, _F)

# get constraints red subspace
Z = order1_gam_subspace(nxe)

print(f"{K.shape=} {F.shape=} {Z.shape=}")

Kr = Z.T @ K @ Z
Fr = np.dot(Z.T, F)

# plt.imshow(Z)
# plt.show()

# plt.imshow(K)
# plt.show()

# plt.imshow(Kr)
# plt.show()

# u = np.linalg.solve(K, F)
ur = np.linalg.solve(Kr, Fr)
u = np.dot(Z, ur)

# alternate solve using least-squares soln and extra constraints?

W = u[2::5].reshape((nx,nx))
THX = u[3::5].reshape((nx,nx))
THY = u[4::5].reshape((nx,nx))

# plot the solved disps now..
disps_str = ['w', 'gamxz' ,'gamyz']
fig, ax = plt.subplots(1,3)
for j, VALS in enumerate([W, THX, THY]):
    cax = fig.add_subplot(1, 3, j+1, projection='3d')
    cax.plot_surface(X, Y, VALS, cmap='jet')
    cax.set_title(disps_str[j])
plt.show()