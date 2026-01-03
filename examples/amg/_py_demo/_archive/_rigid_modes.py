"""
get the rigid body modes for SA-AMG of the plate
"""
import sys
sys.path.append("_src/")

import numpy as np
import matplotlib.pyplot as plt
from tacs_ref import get_tacs_matrix, delete_rows_and_columns, reduced_indices
import scipy as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
import argparse

_tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"_src/plate1.bdf", thickness=10.0)
_tacs_csr_mat = _tacs_bsr_mat.tocsr()

kelem = _tacs_csr_mat.toarray()

print(f"{_xpts=}") 
# in space looks like this,
# [n3, n4]
# [n1, n2]

def nullspace(A, tol=1e-12):
    """Compute an orthonormal basis for the nullspace of A."""
    u, s, vh = np.linalg.svd(A)
    print(f"{s=}")
    rank = (s > tol).sum()
    null_space = vh[rank:].T
    return null_space

B = nullspace(kelem, tol=0.5)

# print(f"{B=}")
# plt.imshow(B)
# plt.show()

# plot each rigid body mode as deformations of 4 points in space
x = np.linspace(0.0, 1.0, 2)
y = np.linspace(0.0, 1.0, 2)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# fig, ax = plt.subplots(3, 2, figsize=(8, 8), subplot_kw={'projection': '3d'})

# for imode in range(6):
#     ub = B[0::6, imode].reshape((2,2))
#     vb = B[1::6, imode].reshape((2,2))
#     wb = B[2::6, imode].reshape((2,2))

#     Xd = X + ub
#     Yd = Y + vb
#     Zd = Z + wb

#     ax[imode // 2, imode % 2].plot_surface(Xd, Yd, Zd, cmap='plasma')

# plt.show()
# plt.close('all')

# see if I can make a suggested B matrix myself then
Bpred = np.zeros((24, 6))

# first three modes just as translation
for imode in range(3):
    Bpred[imode::6, imode] = 1.0

# then three linearized rotation modes
# first rotation in (u,v) or (x,y) plane, [u,v] = [0, th; -th, 0] * [x, y]
# for a wing will be trickier?
_x = _xpts[0::3]
_y = _xpts[1::3]
_z = _xpts[2::3]

th = 1.0

# u then v disp (yeah this one doesn't work cause of drill strain penalty)
# so are there really only five modes?
Bpred[0::6, 3] = th * _y
Bpred[1::6, 3] = -th * _x
# ahh => correction from drill strain = 2 * thz - (du/dy - dv/dx) = 2 * thz - omega
# is to compute constant thz everywhere equal to the rotation magnitude => thz = th prescribed
Bpred[5::6, 3] = -th

# v and w disp
Bpred[1::6, 4] = th * _z
Bpred[2::6, 4] = th * _y
# ah but then need to adjust thx or thy disp grads for trv shear error
Bpred[3::6, 4] = th

# u and w disp
Bpred[0::6, 5] = th * _z
Bpred[2::6, 5] = -th * _x
# and then need to adjust thy for dw/dx trv shear strain
Bpred[4::6, 5] = th

# BOOM => now I have all 6 rigid body modes!
# where you do have to equivalently adjust the thx, thy, and thz for const rotations to get zero strains!

fig, ax = plt.subplots(3, 2, figsize=(8, 8), subplot_kw={'projection': '3d'})

# now plot my modes
for imode in range(6):
    ub = Bpred[0::6, imode].reshape((2,2))
    vb = Bpred[1::6, imode].reshape((2,2))
    wb = Bpred[2::6, imode].reshape((2,2))

    Xd = X + ub
    Yd = Y + vb
    Zd = Z + wb

    ax[imode // 2, imode % 2].plot_surface(Xd, Yd, Zd, cmap='plasma')

plt.show()

# check indeed that K * B = 0 (the nullspace)
rand = np.random.rand(24, 6)
resid0 = kelem @ rand
resid = kelem @ Bpred
_resid_nrm = np.linalg.norm(resid0)
resid_nrm = np.linalg.norm(resid)
rel_resid = resid_nrm / _resid_nrm
print(f"{rel_resid=:.2e}")
plt.imshow(resid)
plt.show()

# how would you then get a strain-free B matrix for a whole plate or a curved surface though?
