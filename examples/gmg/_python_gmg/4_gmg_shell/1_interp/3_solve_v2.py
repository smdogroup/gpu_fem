"""
goal here is to solve and actually enforce all the tying point constraints on the fine mesh (for full FEA mesh)

# trying new method => where change w disp field to make gam_13, gam_23 still small..
"""


import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from __src import get_tacs_matrix, plot_plate_vec
import scipy as sp
from scipy.sparse.linalg import spsolve

import os
if not os.path.exists("../_out"): os.mkdir("../_out")
folder = "../_out/_interp2"
if not os.path.exists(folder): os.mkdir(folder)


""" first let's solve the linear system on the coarse mesh """

# SR = 10.0
# SR = 100.0
SR = 1000.0 # fairly slender plate
# SR = 10000.0 # really slender plate..
thickness = 1.0 / SR

# NOTE : after fixing nodal ordering issues, as SR -> 0
# I'm getting shear strains which do scale by 1/SR^2 !! Nice, this makes sense
# I also had to scale loads by thick^3 so that disp field is constant across SR changes (otherwise not one to one comparison, other confounding factor)
#   in other words, without thick^3 adjustment of RHS, inc SR => dec thickness => inc disp (leads to higher shear strains actually, confounding var)

# nxe = 32
nxe = 16 # num elements in coarse mesh
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
plt.savefig(folder + "/1_c_disp.svg") # c stands for coarse
plt.close('all')

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
plt.savefig(folder + "/2_c_gam_strains.svg") # c stands for coarse
# plt.show()
plt.close('all')

"""now make the fine mesh and do naive interp first (to show that indeed, the basic interp has huge non-physical shear strains)"""
coarse_disp = np.zeros((3*nnodes,)) # only w, thx, thy DOF
nxe_coarse, nx_coarse = nxe, nx
for i,j in enumerate([2, 3, 4]): # copy out only w, thx, thy DOF
    coarse_disp[i::3] = disp[j::6]

nxe_fine = 2 * nxe
nx_fine = nxe_fine + 1
nnodes_fine = nx_fine**2
ndof_fine = 3 * nnodes_fine
fine_disp = np.zeros(ndof_fine)
h = 0.5 / nxe # mesh size of fine disp (instead of H for coarse)

# naiive interp.. c stands for coarse, f for fine
for iye_c in range(nxe_coarse):
    for ixe_c in range(nxe_coarse):
        # get coarse disps
        inode_c = iye_c * nx_coarse + ixe_c
        elem_c_nodes = [inode_c, inode_c+1, inode_c+nx_coarse+1, inode_c+nx_coarse]
        elem_c_dof = np.array([3*_inode +_idof for _inode in elem_c_nodes for _idof in range(3)]) # only w,thx,thy DOF I'm getting
        # elem_c_disp = coarse_disp[elem_c_dof]

        # get all 9 fine nodes..
        ixe_f, iye_f = 2 * ixe_c, 2 * iye_c
        inode_f = iye_f * nx_fine + ixe_f # bottom right fine node on coarse elem
        elem_f_nodes = [inode_f + _ix + nx_fine * _iy for _iy in range(3) for _ix in range(3)]
        # print(f"{elem_f_nodes=}"); exit()
        # the above nodes are in order of x then y in the coarse elem (3x3 grid of fine nodes in coarse elem)
        
        for _idof in range(3): # loop over w, thx, thy to do avg interps
            # coarse DOF for this elem
            elem_c_dof = [3*_inode + _idof for _inode in elem_c_nodes] 
            # note this is x-, x+, x+, x- order (not x-, x+, x-, x+ with inc y) in other words STANDARD order of CQUAD4 nodes
            c1, c2, c3, c4 = elem_c_dof[0], elem_c_dof[1], elem_c_dof[2], elem_c_dof[3]

            # fine DOFs for each node (and this DOF) for convenience
            # fine nodes here are in inc x and inc y for 3x3 set
            elem_f_dof = [3*_inode + _idof for _inode in elem_f_nodes] 
            f1, f2, f3 = elem_f_dof[0], elem_f_dof[1], elem_f_dof[2]
            f4, f5, f6 = elem_f_dof[3], elem_f_dof[4], elem_f_dof[5]
            f7, f8, f9 = elem_f_dof[6], elem_f_dof[7], elem_f_dof[8]

            # fine nodes which match coarse nodes (equiv to basis func interp..)
            fine_disp[f1] = coarse_disp[c1]
            fine_disp[f3] = coarse_disp[c2]
            fine_disp[f7] = coarse_disp[c4]
            fine_disp[f9] = coarse_disp[c3]

            # fine nodes on edge midpoints of coarse nodes
            fine_disp[f2] = 0.5 * (coarse_disp[c1] + coarse_disp[c2])
            fine_disp[f4] = 0.5 * (coarse_disp[c1] + coarse_disp[c4])
            fine_disp[f6] = 0.5 * (coarse_disp[c2] + coarse_disp[c3])
            fine_disp[f8] = 0.5 * (coarse_disp[c3] + coarse_disp[c4])

            # fine nodes in coarse elem centroid (only one)
            fine_disp[f5] = 0.25 * (coarse_disp[c1] + coarse_disp[c2] + coarse_disp[c3] + coarse_disp[c4])

# plot naive interp fine disp
_plot_fine_disp = np.zeros(6*nnodes_fine)
for i, j in enumerate([2, 3, 4]):
    _plot_fine_disp[j::6] = fine_disp[i::3]

ax_titles = ['w', 'thx', 'thy']
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
for i in range(3):
    cax = fig.add_subplot(1,3,i+1, projection='3d')
    plot_plate_vec(nxe=2*nxe, vec=_plot_fine_disp, ax=cax, sort_fw=None, nodal_dof=2+i, cmap='jet')
    cax.set_title(str(ax_titles[i]))
plt.tight_layout()
plt.savefig(folder + "/1_f_disp_v1.svg") # f stands for fine
plt.close('all')

# now compute and plot the fine shear strains (v1 for naiive interp)
GAM_13_F_V1, GAM_23_F_V1 = np.zeros((2*nxe,2*nxe)), np.zeros((2*nxe,2*nxe))
for iye in range(2*nxe):
    for ixe in range(2*nxe):
        # get the elem disps
        inode = iye * nx_fine + ixe # bottom right node of the element
        elem_nodes = [inode, inode+1, inode+nx_fine+1, inode+nx_fine]
        elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
        # print(F'{inode=} {elem_nodes=} {elem_dof=}')
        elem_disp = _plot_fine_disp[elem_dof]
        _w, _thx, _thy = elem_disp[0::3], elem_disp[1::3], elem_disp[2::3]

        # compute gam_13 at each tying point
        # NOTE : uses mesh size h not H here for the fine mesh
        gam_13_1 = (_w[1] - _w[0]) / h + 0.5 * (_thy[1] + _thy[0])
        gam_13_2 = (_w[2] - _w[3]) / h + 0.5 * (_thy[2] + _thy[3])
        # if ixe < 2 or iye < 2:
        #     print(f"{ixe=} {iye=} {gam_13_1=:.3e} {gam_13_2=:.3e}")
        _gam_13 = 0.5 * (gam_13_1 + gam_13_2)
        GAM_13_F_V1[ixe, iye] = _gam_13

        # compute gam_23 at each tying point
        gam_23_1 = (_w[2] - _w[1]) / h - 0.5 * (_thx[2] + _thx[1])
        gam_23_2 = (_w[3] - _w[0]) / h - 0.5 * (_thx[3] + _thx[0])
        _gam_23 = 0.5 * (gam_23_1 + gam_23_2)
        GAM_23_F_V1[ixe, iye] = _gam_23

        # print(f"{_gam_13=:.3e} {_gam_23=:.3e}")

# now plot the gam13 and gam23 transverse shear strains
xfe, yfe = np.linspace(0.0, 1.0, 2*nxe), np.linspace(0.0, 1.0, 2*nxe)
Xfe, Yfe = np.meshgrid(xfe, yfe)

ax_titles = ['gam13', 'gam23']
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i,_VALS in enumerate([GAM_13_F_V1, GAM_23_F_V1]):
    ax1 = fig.add_subplot(1,2,i+1, projection='3d')
    # VALS = _VALS
    VALS = np.log10(np.abs(_VALS)).T
    ax1.plot_surface(Xfe, Yfe, VALS, cmap='RdBu_r')
    ax1.set_title(str(ax_titles[i]))
plt.tight_layout()
plt.savefig(folder + "/2_f_gam_strains_v1.svg") # f stands for fine and v1 for naiive version 1 interp
plt.close('all')


"""now fine disp interp V2 : goal to make more realistic transverse shear strain mags (for high SR plates)"""

fine_disp_v2 = fine_disp.copy()
# we'll be simply correcting the disps to match earlier transverse shear strains (from baseline naiive interp)

# define interp methods..
def _shear_strain(w, th, sign:float=1.0, _dx:float=h):
    """helper method to compute shear strains (2 vals each in w and th) for fine vs coarse elements and gam_13, gam_23"""
    return (w[1] - w[0]) / _dx + sign * 0.5 * (th[0] + th[1]) # dw/dx here, same as 2/dx * dw/dxi, etc.

def oned_shear_interp(w, th, sign:float=1.0):
    # here w, th are length 3 vecs each (with interp middle value not set yet, and we set that)

    # any edge and either shear strain will use this interp method..
    gamma_c = _shear_strain(w=[w[0], w[2]], th=[th[0], th[2]], _dx=H, sign=sign) # coarse shear strain

    w_1 = w[0] + h * (gamma_c - 0.5 * (th[0] + th[1]))
    w_2 = w_1 + h * (gamma_c - 0.5 * (th[1] + th[2]))
    return w_1, w_2

def gam13_interp(w, th):
    return oned_shear_interp(w, th, sign=1.0)

def gam23_interp(w, th):
    return oned_shear_interp(w, th, sign=-1.0)

# loop over each coarse element..
for iye_c in range(nxe_coarse):
    for ixe_c in range(nxe_coarse):
        # get coarse disps
        inode_c = iye_c * nx_coarse + ixe_c
        elem_c_nodes = [inode_c, inode_c+1, inode_c+nx_coarse+1, inode_c+nx_coarse]
        elem_c_dof = np.array([3*_inode +_idof for _inode in elem_c_nodes for _idof in range(3)]) # only w,thx,thy DOF I'm getting
        elem_c_disp = coarse_disp[elem_c_dof]
        c1, c2, c3, c4 = elem_c_disp[0], elem_c_disp[1], elem_c_disp[2], elem_c_disp[3]
        c5, c6, c7, c8 = elem_c_disp[4], elem_c_disp[5], elem_c_disp[6], elem_c_disp[7]
        c9, c10, c11, c12 = elem_c_disp[8], elem_c_disp[9], elem_c_disp[10], elem_c_disp[11]

        # get all 9 fine nodes.. and their 3 DOF (27 DOF total)
        ixe_f, iye_f = 2 * ixe_c, 2 * iye_c
        inode_f = iye_f * nx_fine + ixe_f # bottom right fine node on coarse elem
        elem_f_nodes = [inode_f + _ix + nx_fine * _iy for _iy in range(3) for _ix in range(3)]
        elem_f_dof = np.array([3*_inode + _idof for _inode in elem_f_nodes for _idof in range(3)])
        # print(f"{elem_f_nodes=} {elem_f_dof=}")
        # print(f"1: {fine_disp_v2[elem_f_dof]=}")

        """ update step 1 for w DOF"""

        # get current disps
        elem_f_disp = fine_disp_v2[elem_f_dof]
        f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
        f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
        f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
        f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
        f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

        # gam13 update on y- edge (only up)
        _i, _j = elem_f_dof[4-1], elem_f_dof[7-1]
        # _w0, _th0 = f4, f6
        fine_disp_v2[_i], fine_disp_v2[_j] = gam13_interp(
            w=[f1, f4, f7],
            th=[f3, f6, f9],
        )

        # gam23 update on x- edge (west)
        _i, _j = elem_f_dof[10-1], elem_f_dof[19-1]
        fine_disp_v2[_i], fine_disp_v2[_j] = gam23_interp(
            w=[f1, f10, f19],
            th=[f2, f11, f20],
        )

        """update step 2 for gam13, gam23"""

        # udpate disps again for rhs
        elem_f_disp = fine_disp_v2[elem_f_dof]
        f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
        f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
        f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
        f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
        f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

        # gam13 update on ymid edge (central)
        _i, _j = elem_f_dof[13-1], elem_f_dof[16-1]
        fine_disp_v2[_i], fine_disp_v2[_j] = gam13_interp(
            w=[f10, f13, f16],
            th=[f12, f15, f18],
        )

        # # gam13 update on xmid edge (central)
        # _i, _j = elem_f_dof[13-1], elem_f_dof[22-1]
        # fine_disp_v2[_i], fine_disp_v2[_j] = gam23_interp(
        #     w=[f4, f13, f22],
        #     th=[f5, f14, f23],
        # )

        # """now last step of gam13, gam23 updates"""

        # udpate disps again for rhs
        elem_f_disp = fine_disp_v2[elem_f_dof]
        f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
        f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
        f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
        f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
        f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

        # gam13 update on y+ edge (north)
        _i, _j = elem_f_dof[22-1], elem_f_dof[25-1]
        fine_disp_v2[_i], fine_disp_v2[_j] = gam13_interp(
            w=[f19, f22, f25],
            th=[f21, f24, f27],
        )

        # # gam23 update on x+ edge (east)
        # _i, _j = elem_f_dof[16-1], elem_f_dof[25-1]
        # fine_disp_v2[_i], fine_disp_v2[_j] = gam23_interp(
        #     w=[f7, f16, f25],
        #     th=[f8, f17, f26],
        # )        

        """ DEBUG section """
        # get disps again for convenience..
        elem_f_disp = fine_disp_v2[elem_f_dof]
        f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
        f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
        f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
        f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
        f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

        # now double check all gam13, gam23 constraints on the element.. (need all small like 1e-5, 1e-6 here, or depending on SR)
        gam_13_list = [
            (f4 - f1) / h + 0.5 * (f3 + f6),
            (f7 - f4) / h + 0.5 * (f6 + f9),
            (f13 - f10) / h + 0.5 * (f12 + f15),
            (f16 - f13) / h + 0.5 * (f15 + f18),
            (f22 - f19) / h + 0.5 * (f21 + f24),
            (f25 - f22) / h + 0.5 * (f24 + f27),
        ]
        print(f"{gam_13_list=}")

        gam_23_list = [
            (f10 - f1) / h - 0.5 * (f2 + f11),
            (f19 - f10) / h - 0.5 * (f20+f11),
            (f13 - f4) / h - 0.5 * (f14 + f5),
            (f22 - f13) / h - 0.5 * (f23 + f14),
            (f16 - f7) / h - 0.5 * (f17 + f8),
            (f25 - f16) / h - 0.5 * (f26 + f17),
        ]
        print(f"{gam_23_list=}")

        exit()


""" now double check the fine v2 disp (with gam13 and gam23 corrections) have smaller shear strains? """

_plot_fine_disp_v2 = np.zeros(6*nnodes_fine)
for i, j in enumerate([2, 3, 4]):
    _plot_fine_disp_v2[j::6] = fine_disp_v2[i::3]

# now compute and plot the fine shear strains (v2 for shear strains corrected disp)
GAM_13_F_V2, GAM_23_F_V2 = np.zeros((2*nxe,2*nxe)), np.zeros((2*nxe,2*nxe))
for iye in range(2*nxe):
    for ixe in range(2*nxe):
        # get the elem disps
        inode = iye * nx_fine + ixe # bottom right node of the element
        elem_nodes = [inode, inode+1, inode+nx_fine+1, inode+nx_fine]
        elem_dof = np.array([6*_inode +_idof for _inode in elem_nodes for _idof in [2,3,4]]) # only w,thx,thy DOF I'm getting
        # print(F'{inode=} {elem_nodes=} {elem_dof=}')
        elem_disp = _plot_fine_disp_v2[elem_dof]
        _w, _thx, _thy = elem_disp[0::3], elem_disp[1::3], elem_disp[2::3]

        # compute gam_13 at each tying point
        # NOTE : uses mesh size h not H here for the fine mesh
        gam_13_1 = (_w[1] - _w[0]) / h + 0.5 * (_thy[1] + _thy[0])
        gam_13_2 = (_w[2] - _w[3]) / h + 0.5 * (_thy[2] + _thy[3])
        # if ixe < 2 or iye < 2:
        #     print(f"{ixe=} {iye=} {gam_13_1=:.3e} {gam_13_2=:.3e}")
        _gam_13 = 0.5 * (gam_13_1 + gam_13_2)
        GAM_13_F_V2[ixe, iye] = _gam_13

        # compute gam_23 at each tying point
        gam_23_1 = (_w[2] - _w[1]) / h - 0.5 * (_thx[2] + _thx[1])
        gam_23_2 = (_w[3] - _w[0]) / h - 0.5 * (_thx[3] + _thx[0])
        _gam_23 = 0.5 * (gam_23_1 + gam_23_2)
        GAM_23_F_V2[ixe, iye] = _gam_23

        # print(f"{_gam_13=:.3e} {_gam_23=:.3e}")

# now plot the gam13 and gam23 transverse shear strains
xfe, yfe = np.linspace(0.0, 1.0, 2*nxe), np.linspace(0.0, 1.0, 2*nxe)
Xfe, Yfe = np.meshgrid(xfe, yfe)

ax_titles = ['gam13', 'gam23']
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for i,_VALS in enumerate([GAM_13_F_V2, GAM_23_F_V2]):
    ax1 = fig.add_subplot(1,2,i+1, projection='3d')
    # VALS = _VALS
    VALS = np.log10(np.abs(_VALS)).T
    ax1.plot_surface(Xfe, Yfe, VALS, cmap='RdBu_r')
    ax1.set_title(str(ax_titles[i]))
plt.tight_layout()
plt.savefig(folder + "/2_f_gam_strains_v2.svg")
plt.close('all')