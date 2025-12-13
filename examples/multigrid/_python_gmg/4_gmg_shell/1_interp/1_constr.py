"""
NEW and BETTER IDEA:
* compute reduced DOF on each coarse/fine element and interp tying strains, then reconstruct true DOF
    * how to do with geom NL tying strains? should still be doable with cylinders or something? 
    * Need to compute free vars and constr vars of each element like eigvalue solve based on shell normal? TBD for generalizing this..

ORIG IDEAS:
* compute the nullspace of the tying point constraint matrix C z = 0 (coarse-fine operator tying point zero constraints)
* has 15 free vars (after elim coarse nodes), 12 tying constraints on fine mesh 
* originally tried 4 boundary constraints on thx, thy free on bndry (need compatible with adjacent elements)
* consider first tying point = 0 constraints (can generalize to FSDT case if RHS = nz then)
* if really necessary, can do something other than injection on 4 coarse node points

* NOTE : it may be revealing to consider one or more adjacent coarse elements, to see how this influences the equation set?
    * we would basically be adding only 12 free DOF and 12 new constraints.. (this is interesting..
    * if we had 4 coarse elems there would be 16 free fine nodes (48 free dof) and 48 tying point constraints.. hmm this is interesting (so not overconstrained then?)
"""

import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from __src import get_tacs_matrix, plot_plate_vec
import scipy as sp
from scipy.sparse.linalg import spsolve

""" first let's solve the linear system on the coarse mesh """

SR = 1000.0 # fairly slender plate
# SR = 10000.0 # really slender plate..
thickness = 1.0 / SR
nxe = 32 # num elements in coarse mesh
_tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"../_in/plate{nxe}.bdf", thickness=thickness)
_tacs_csr_mat = _tacs_bsr_mat.tocsr()

disp = spsolve(_tacs_csr_mat, _rhs)

# choose element 10,10 with nodes:
ix, iy = 9, 9
nx = nxe + 1
inode = iy * nx + ix
elem_nodes = [inode, inode+1, inode+nx+1, inode+nx]
elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
# print(F'{inode=} {elem_nodes=} {elem_dof=}')
elem_disp = disp[elem_dof]
# print(f"{elem_disp=}")

"""red DOF method - determine red to full DOF equations for a fine element at thin plate limit"""
A = np.array([
    [-1, 1, 0, 0],
    [0, -1, 1, 0],
    [0, 0, -1, 1],
    [1, 0, 0, -1],
], dtype=np.float32)

# this matrix is singular
# Ainv = np.linalg.inv(A)
# print(f"{Ainv=}")

Z = sp.linalg.null_space(A)
print(f"{Z=}")

# this only gives you constraints on slope between w (can add constant, so need to add some mean constr?)
A2 = np.concatenate([A, np.reshape([1, 1, 1, 1], (1,4))], axis=0) # this matrix is full rank..
# A2 = np.concatenate([A, np.reshape([1, 0, 0, 0], (1,4))], axis=0)
# print(f"{A2=}")
A2_LS = A2.T @ A2
A2inv = np.linalg.inv(A2_LS) # or could just do pinv here and it would include A2.T part..
# print(f"{A2inv=}")

# can compute reduced DOF on each coarse or fine element
# # there 
# A1_A2inv = A @ A2inv @ A2.T # least-squares min and then check resid (closer to zero..)
# print(F"{A1_A2inv=}")
# hmm, doesn't seem to actually solve the original system

"""now let's reconstruct the DOF and check our constraints"""
dx = 1.0 / nxe # mesh size
# scale down by mesh size (first four rows)
A2[:4] /= dx 
A2[-1] *= 0.25

def get_tying_shear_strains(w, thx, thy):
    gammas = np.zeros(4)
    # first two are gamma_13 on each edge xi = 0, +-\eta edge
    gammas[0] = (w[1] - w[0]) / dx + 0.5 * (thy[0] + thy[1])
    gammas[2] = (w[2] - w[3]) / dx + 0.5 * (thy[2] + thy[3])
    # second two are gamma_23 on each edge eta = 0, +-\xi edge
    gammas[1] = (w[2] - w[1]) / dx - 0.5 * (thx[1] + thx[2])
    gammas[3] = (w[3] - w[0]) / dx - 0.5 * (thx[0] + thx[3])
    return gammas

def get_rhs(w, thx, thy):
    gammas = np.zeros(4)
    # first two are gamma_13 on each edge xi = 0, +-\eta edge
    gammas[0] = (w[1] - w[0]) / dx + 0.5 * (thy[0] + thy[1])
    gammas[2] = (w[2] - w[3]) / dx + 0.5 * (thy[2] + thy[3])
    # second two are gamma_23 on each edge eta = 0, +-\xi edge
    gammas[1] = (w[2] - w[1]) / dx - 0.5 * (thx[1] + thx[2])
    gammas[3] = (w[3] - w[0]) / dx - 0.5 * (thx[0] + thx[3])
    return gammas

def constr_rhs(w0, thx, thy):
    """could construct from gamma and thx,thy or from w0 (here let's just do with w0)"""
    rhs = np.zeros(5)
    # first two are gamma_13 on each edge xi = 0, +-\eta edge
    rhs[0] = (w0[1] - w0[0]) / dx 
    rhs[2] = (w0[2] - w0[3]) / dx 
    # second two are gamma_23 on each edge eta = 0, +-\xi edge
    rhs[1] = (w0[2] - w0[1]) / dx 
    rhs[3] = (w0[3] - w0[0]) / dx 
    rhs[2] *= -1 # flip sign here for 34 edge
    rhs[3] *= -1 # flip sign here
    rhs[4] = np.mean(w0)
    return rhs

def constr_resid(w, w0, thx, thy):
    # each w, thx, thy are length-4 vectors
    _rhs = constr_rhs(w0, thx, thy)
    print(f"{A2.shape=} {_rhs.shape=}")
    resid = np.dot(A2, w) - _rhs
    return resid
    # return np.linalg.norm(resid)

def solve_from_red_dof(w0, thx, thy):
    # 5x4 linear system (not square, but full rank)
    _rhs = constr_rhs(w0, thx, thy)
    _w,_resids,_rank,_s = np.linalg.lstsq(A2, _rhs, rcond=None) # must do least-squares solve since not square matrix
    print(f"{_resids=} {_rank=}")
    # _w = np.linalg.pinv(A2) @ _rhs
    return _w

# compute initial disps
# disps = np.random.rand(12) # oh but here the tying strains aren't zero though so don't expect match after we re-solve..
w0, thx, thy = elem_disp[0::3], elem_disp[1::3], elem_disp[2::3]

# _gams = get_tying_shear_strains(w0, thx, thy)
# print(F"{_gams=}")
_resid0 = constr_resid(w0, w0, thx, thy)
print(F"{_resid0=}")

# now reconstruct initial disps from these constraints
_w2 = solve_from_red_dof(w0, thx, thy)
print(f"{w0=}")
print(f"{_w2=}")

resid = constr_resid(_w2, w0, thx, thy)
print(F"{resid=}")


# plot the coarse-fine interp:
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

ms = 10
plt.plot([-1, -1, 1, 1], [-1, 1, -1, 1], 'o', color='tab:blue', markersize=ms)
plt.plot([-1, 0, 1, 0, 0], [0, -1, 0, 1, 0], 'o', color='tab:orange', markersize=ms)
plt.show()