"""
compare the disp, rotations and transverse shear between different plate slendernesses..
"""


import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from __src import get_tacs_matrix, plot_plate_vec
import scipy as sp
from scipy.sparse.linalg import spsolve
import argparse

import os
if not os.path.exists("../_out"): os.mkdir("../_out")
folder = "../_out/_trv_shear"
if not os.path.exists(folder): os.mkdir(folder)

"""argparse"""
parser = argparse.ArgumentParser()
parser.add_argument("--SR", type=int, default=100, help="slenderness ratio of plate L/h")
args = parser.parse_args()

""" first let's solve the linear system on the coarse mesh """

SR = int(args.SR)
thickness = 1.0 / float(SR)

# NOTE : after fixing nodal ordering issues, as SR -> 0
# I'm getting shear strains which do scale by 1/SR^2 !! Nice, this makes sense
# I also had to scale loads by thick^3 so that disp field is constant across SR changes (otherwise not one to one comparison, other confounding factor)
#   in other words, without thick^3 adjustment of RHS, inc SR => dec thickness => inc disp (leads to higher shear strains actually, confounding var)

nxe = 32
# nxe = 16 # num elements in coarse mesh
# nxe = 8
# nxe = 4

_tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"../_in/plate{nxe}.bdf", thickness=thickness)
_tacs_csr_mat = _tacs_bsr_mat.tocsr()

# need to resort the nodes based on xpts.. (they get slightly misordered near y=0 edge for some reason)
# NOTE : otherwise the gam shear strains later are wrong (cause node ordering not quite right..)
nnodes = _xpts.shape[0] // 3
dof_xpts_list = [ [[6*inode+idof] for idof in range(6)] + [_xpts[3*inode+idof] for idof in range(3)] for inode in range(nnodes)]
dof_xpts_list2 = sorted(dof_xpts_list, key=lambda x : (x[7], x[6])) # sorts by x and then y
# print(f"{dof_xpts_list2=}")
# exit()
sort_map = np.array([dof_xpts_list2[inode][:6] for inode in range(nnodes)])
sort_map = np.reshape(sort_map, (6*nnodes,))

# scale RHS by thick^3 so that get similar disp despite changed thickness (so similar strains across each SR value)
_rhs *= 1e9 * thickness**3

_disp = spsolve(_tacs_csr_mat, _rhs)

disp = _disp[sort_map]

# some other helper data for later
nx = nxe + 1
H = 1.0 / nxe # coarse mesh step size


"""plot the initial disp, rotations, and tying shear strains (averaged across both tying points)"""

rhs = _rhs[sort_map] # apply nodal sorting to rhs also

ax_titles = ['F[w]', 'F[thx]', 'F[thy]']
fig, ax = plt.subplots(1, 1)
cax = fig.add_subplot(1,1,1, projection='3d')
plot_plate_vec(nxe=nxe, vec=rhs, ax=cax, sort_fw=None, nodal_dof=2, cmap='jet')
cax.set_title("F[w]")
plt.tight_layout()
# plt.show()
plt.savefig(folder + "/0_c_force.svg") # c stands for coarse
plt.close('all')
# NOTE after fixing node ordering issues, the polar loads are sym about y=x now so x and y forces, everything should be sym like this (reflect about y=x)

ax_titles = ['w', 'thx', 'thy']
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
for i in range(3):
    cax = fig.add_subplot(1,3,i+1, projection='3d')
    plot_plate_vec(nxe=nxe, vec=disp, ax=cax, sort_fw=None, nodal_dof=2+i, cmap='jet')
    cax.set_title(str(ax_titles[i]))
plt.tight_layout()
plt.savefig(folder + f"/1_{SR=}_disp.svg") # c stands for coarse
plt.close('all')

"""plot the transv shear strains in an element"""

# choose element 10,10 with nodes:
ix, iy = 4, 4
nx = nxe + 1
inode = iy * nx + ix
elem_nodes = [inode, inode+1, inode+nx+1, inode+nx]
elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
elem_disp = disp[elem_dof]

H = 1.0 / nxe # coarse mesh size
h = H / 2.0
xpts_elem = H * np.array([[_inode % nx, _inode // nx, 0.0] for _inode in elem_nodes])

n = 11
xi, eta = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
XI, ETA = np.meshgrid(xi, eta)

N = [
    0.25 * (1 - XI) * (1 - ETA),
    0.25 * (1 + XI) * (1 - ETA),
    0.25 * (1 + XI) * (1 + ETA),
    0.25 * (1 - XI) * (1 + ETA),
]

def interp(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N[i] * nodal_vals[i]
    return VAL

W = interp(elem_disp[0::3])
THX = interp(elem_disp[1::3])
THY = interp(elem_disp[2::3])


""" define disp grad interp methods """

N_XI = [
    -0.25 * (1 - ETA),
    0.25 * (1 - ETA),
    0.25 * (1 + ETA),
    -0.25 * (1 + ETA),
]

N_ETA = [
    -0.25 * (1 - XI),
    -0.25 * (1 + XI),
    0.25 * (1 + XI),
    0.25 * (1 - XI),
]

def interp_dxi(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N_XI[i] * nodal_vals[i]
    return VAL

def interp_deta(nodal_vals):
    # helper interp method
    VAL = np.zeros_like(XI)
    for i in range(4):
        VAL += N_ETA[i] * nodal_vals[i]
    return VAL

def interp_dx(nodal_vals, dx:float=H):
    return interp_dxi(nodal_vals) / (0.5 * dx) # scale by dx elem size

def interp_dy(nodal_vals, dx:float=H):
    return interp_deta(nodal_vals) / (0.5 * dx) # scale by dx elem size


W_X = interp_dx(elem_disp[0::3])
W_Y = interp_dy(elem_disp[0::3])

GAM_13 = (THY + W_X) # H/4.0 *
GAM_23 = (W_Y - THX) #  H/4.0 *

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

LOG_GAM_13, LOG_GAM_23 = np.log10(np.abs(GAM_13)), np.log10(np.abs(GAM_23))

cf = ax[0].contourf(XI, ETA, LOG_GAM_13)
irt3 = 1.0 / np.sqrt(3)
ax[0].plot([0.0, 0.0], [-1, 1], 'k--')
ax[0].scatter([0.0, 0.0], [-irt3, irt3], color='k')
ax[0].set_title('log10 gam13(xi,eta)')
fig.colorbar(cf)

cf = ax[1].contourf(XI, ETA, LOG_GAM_23)
ax[1].plot([-1, 1], [0.0, 0.0], 'k--')
ax[1].scatter([-irt3, irt3], [0.0, 0.0], color='k')
ax[1].set_title('log10 gam23(xi,eta)')
fig.colorbar(cf)
# plt.show()
plt.savefig(folder + f"/2_{SR=}_gam_elem_strains.svg")


"""plot the transv shear strains globally (at tying points in each elem)"""
# compute the initial tying strains (averaged across each of their tying points)
GAM_13, GAM_23 = np.zeros((nxe,nxe)), np.zeros((nxe,nxe))
for iye in range(nxe):
    for ixe in range(nxe):
        # get the elem disps
        inode = iye * nx + ixe # bottom right node of the element
        elem_nodes = [inode, inode+1, inode+nx+1, inode+nx]
        elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
        # print(F'{inode=} {elem_nodes=} {elem_dof=}')
        elem_disp = disp[elem_dof]
        _w, _thx, _thy = elem_disp[0::3], elem_disp[1::3], elem_disp[2::3]

        # compute gam_13 at each tying point
        gam_13_1 = (_w[1] - _w[0]) / H + 0.5 * (_thy[1] + _thy[0])
        gam_13_2 = (_w[2] - _w[3]) / H + 0.5 * (_thy[2] + _thy[3])
        # if ixe < 2 or iye < 2:
        #     print(f"{ixe=} {iye=} {gam_13_1=:.3e} {gam_13_2=:.3e}")
        _gam_13 = 0.5 * (gam_13_1 + gam_13_2)
        GAM_13[ixe, iye] = _gam_13

        # compute gam_23 at each tying point
        gam_23_1 = (_w[2] - _w[1]) / H - 0.5 * (_thx[2] + _thx[1])
        gam_23_2 = (_w[3] - _w[0]) / H - 0.5 * (_thx[3] + _thx[0])
        _gam_23 = 0.5 * (gam_23_1 + gam_23_2)
        GAM_23[ixe, iye] = _gam_23

        # print(f"{_gam_13=:.3e} {_gam_23=:.3e}")

# element data plotting coords
xe, ye = np.linspace(0.0, 1.0, nxe), np.linspace(0.0, 1.0, nxe)
Xe, Ye = np.meshgrid(xe, ye)

ax_titles = ['gam13', 'gam23']
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i,_VALS in enumerate([GAM_13, GAM_23]):
    ax1 = fig.add_subplot(1,2,i+1, projection='3d')
    # VALS = _VALS
    VALS = np.log10(np.abs(_VALS)).T
    ax1.plot_surface(Xe, Ye, VALS, cmap='RdBu_r')
    ax1.set_title(str(ax_titles[i]))
plt.tight_layout()
plt.savefig(folder + f"/3_{SR=}_gam_glob_strains.svg") # c stands for coarse
# plt.show()
plt.close('all')