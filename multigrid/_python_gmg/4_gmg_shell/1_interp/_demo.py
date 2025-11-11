"""
develop a coarse-fine or prolongation operator for TACS CQUAD4 FSDT MITC shells that works well near thin-plate limit
* previous basic avg interp of u,v,w and rot dof produces unrealistic transverse shear strains (explain more below)
* need a scheme that coarse-fine produces realistic transverse shear strains

* reason why before averaging interp doesn't produce right mag transverse shear strains:
    * actual transverse shear strains in an MITC element are only small or right magnitudes on the tying points (which helps prevent shear locking)
    * tranverse tying strains scale by 1/slenderness^2 (making them very small) => and their only small on tying points line

* I will now demonstrate on the plate geometry (after linear solve => will only cause the expected tying interp behavior),
    what the tying strain behavior in the element is and why my original coarse-fine interp breaks down => and how to change it
"""

import numpy as np
import matplotlib.pyplot as plt
from __src import get_tacs_matrix, plot_plate_vec
import scipy as sp
from scipy.sparse.linalg import spsolve
import os

if not os.path.exists("_out"): os.mkdir("_out")
folder = "_out/_interp"
if not os.path.exists(folder): os.mkdir(folder)

""" first let's solve the linear system on the coarse mesh """

SR = 1000.0 # fairly slender plate
thickness = 1.0 / SR
nxe = 16 # num elements in coarse mesh
_tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"_in/plate{nxe}.bdf", thickness=thickness)
_tacs_csr_mat = _tacs_bsr_mat.tocsr()

_rhs *= 1e9 * thickness**3

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

# get also the xpt coordinates for this element
H = 1.0 / nxe # coarse mesh size
h = H / 2.0
xpts_elem = H * np.array([[_inode % nx, _inode // nx, 0.0] for _inode in elem_nodes])
# print(f"{xpts_elem=}")

# plot to check right disp
fig, ax = plt.subplots(1, 1)
ax = fig.add_subplot(1,1,1, projection='3d')
plot_plate_vec(nxe=nxe, vec=disp, ax=ax, sort_fw=None, nodal_dof=2, cmap='jet')
plt.savefig(folder + "/1_disp.svg")

""" now let's compute the disp, rot, dispgrad, strain fields on the coarse element"""

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

""" plot dof fields w, thx, thy """

W = interp(elem_disp[0::3])
THX = interp(elem_disp[1::3])
THY = interp(elem_disp[2::3])

def plot_multi(DOFS, dof_strs, filename=None):
    n_plot = len(DOFS)
    assert(n_plot == len(dof_strs))
    
    fig, ax = plt.subplots(1, n_plot)
    for i in range(n_plot):
        _ax = fig.add_subplot(1, n_plot, i+1, projection='3d')
        _ax.plot_surface(XI, ETA, DOFS[i])
        _ax.set_title(dof_strs[i])
    if filename:
        plt.savefig(folder + "/" + filename)
    else:
        plt.show()

plot_multi(
    DOFS=[W, THX, THY],
    dof_strs=['w', 'thx', 'thy'],
    filename="2_coarse_elem_disps.svg"
)

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

""" now let's compute disp grads """

W_X = interp_dx(elem_disp[0::3])
W_Y = interp_dy(elem_disp[0::3])

# W_X is linear in y or eta (constant in x and xi)
# W_Y is linear in x or xi (constant in y and eta)
plot_multi(
    DOFS=[W_X, W_Y],
    dof_strs=['w_x', 'w_y'],
    filename="3_coarse_elem_disp_grads.svg"
)

""" transverse shear bending strains """
GAM_13 = H/4.0 * (THY + W_X)
GAM_23 = H/4.0 * (W_Y - THX)

# get gam_13, gam_23 at tying points.. compared to gam_13 and gam_23 overall mag.. (approx locations..)
# also meshgrid flips so [eta,xi] order of arsg
GAM_13_TY = np.array([GAM_13[2, 5], GAM_13[8, 5]])
GAM_23_TY = np.array([GAM_23[5, 2], GAM_23[5, 8]])

# now normalize by mags
GAM_13_MAG = np.max(np.abs(GAM_13))
GAM_23_MAG = np.max(np.abs(GAM_23))
GAM_13_TY_NRM = GAM_13_TY / GAM_13_MAG
GAM_23_TY_NRM = GAM_23_TY / GAM_23_MAG

print(f"{GAM_13_TY=} {GAM_13_MAG=} {GAM_13_TY_NRM=}")
print(f"{GAM_23_TY=} {GAM_23_MAG=} {GAM_23_TY_NRM=}")

# basically gam_13 is near zero on xi=0, eta line (of gam_13 tying points)
# and gam_23 near zero on eta=0, xi changing line (of gam_23 tying pts)
#    recall xi in [-1,1] and same for eta

plot_multi(
    DOFS=[GAM_13, GAM_23],
    dof_strs=['gam13', 'gam23'],
    filename="4_coarse_shear_strains.svg"
)

# now also plot as contours and show the tying points too.. (with manual plot so we can show more here..)
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
cf = ax[0].contourf(XI, ETA, GAM_13)
irt3 = 1.0 / np.sqrt(3)
ax[0].plot([0.0, 0.0], [-1, 1], 'k--')
ax[0].scatter([0.0, 0.0], [-irt3, irt3], color='k')
ax[0].set_title('gam13')
fig.colorbar(cf)

cf = ax[1].contourf(XI, ETA, GAM_23)
ax[1].plot([-1, 1], [0.0, 0.0], 'k--')
ax[1].scatter([-irt3, irt3], [0.0, 0.0], color='k')
ax[1].set_title('gam23')
fig.colorbar(cf)
# plt.show()
plt.savefig(folder + "/5_coarse_shear_strain_contours.svg")

# note the normalized gam_13 and gam_23 scale by approx 1/SR (so get very small near thin-walled limit)

# NOTE : this means if we do averaging, the shear strains on tying strains of the fine points
#   for example xi=-0.5 line doesn't have small shear strains of near thin-plate limit
#   we need to explicitly enforce this by making fine interped elems have similar mag shear strains (with MITC)

""" develop coarse-fine prolongation scheme that respect thin-plate limit for MITC elements"""

# first let's demo it for thy, w_x along y=0 or an x-edge (1D), enforcing first full transverse shear.. (kirchoff condition)
w1, w2 = elem_disp[0], elem_disp[3]
thy1, thy2 = elem_disp[2], elem_disp[5]

# double check constraint satisfied at midpoint on coarse element (should have solved for that, or very small value)
R_m = 0.5 * (thy1 + thy2) + (w2 - w1) / h
print(f'{R_m=}')

# introduce new thm, wm DOF such that midpoints at fine level have constraint satisfied (tying points)
dx = h
dxh = 0.5 * h # half-elem width (finer)
# system of 2 eqns for wm, thm (unique soln below)
wm = (w2 + w1) * 0.5 + 0.25 * dxh * (thy2 - thy1)
thym = -2.0 * (wm - w1) / dxh - thy1

# now check residuals
R1 = 0.5 * (thy1 + thym) + (wm-w1)/dxh
R2 = 0.5 * (thym + thy2) + (w2 - wm)/dxh
# print(F"{R1=} {R2=}") 

# now plot this result for 1D interp
fig, ax = plt.subplots(1, 2)
ax[0].plot([0.0, 1.0], [w1, w2], label='coarse')
ax[0].plot([0.0, 0.5, 1.0], [w1, wm, w2], '--', label='fine')
ax[0].legend()
ax[0].set_title("w(x)")
ax[1].plot([0.0, 1.0], [thy1, thy2], label='coarse')
ax[1].plot([0.0, 0.5, 1.0], [thy1, thym, thy2], '--', label='fine')
ax[1].legend()
ax[1].set_title("thy(x)")
# plt.show()
plt.savefig(folder + "/6_1d_kirchoff_interp.svg")

# could also adjust it to instead have the shear strains at new tying points (instead of zero shear strain)

# now let's make a method that solves it for the whole element (using edge data for boundaries, so compatible with other elements)
# do it step by step manually
fine_disp = np.zeros(3*9) # 3x3 group of nodes now for the coarse-elem

# 1 - copy over the coarse to fine nodes (when nodes overlap) => covers 4 of 9 nodes
fine_disp[0:3] = elem_disp[0:3]
fine_disp[6:9] = elem_disp[3:6]
fine_disp[18:21] = elem_disp[9:]
fine_disp[24:27] = elem_disp[6:9] # swap node order from elem_disp to grid order (lexigraphic)

def _shear_strain(w, th, sign:float=1.0, dx:float=h):
    """helper method to compute shear strains (2 vals each in w and th) for fine vs coarse elements and gam_13, gam_23"""
    return (w[1] - w[0]) / dx + sign * 0.5 * (th[0] + th[1]) # dw/dx here, same as 2/dx * dw/dxi, etc.

# 1.5 (intermediate step), generalize 1D solver to handle nonzero kirchoff strain (and make same kirchoff strain at each spot)
def oned_shear_interp(w, th, sign:float=1.0):
    # here w, th are length 3 vecs each (with interp middle value not set yet, and we set that)

    # any edge and either shear strain will use this interp method..
    gamma_c = _shear_strain(w=[w[0], w[2]], th=[th[0], th[2]], dx=H, sign=sign) # coarse shear strain
    
    print(F"{gamma_c=:.3e}")

    w[1] = 0.5 * (w[0] + w[2]) + sign * 0.25 * h * (th[2] - th[0]) # from subtracting two eqns
    th[1] = -th[0] + 2.0 * sign * gamma_c - 2.0 * sign * (w[1] - w[0]) / h

    # check constraints if you want, TODO : mesh size scaling right here to match fine level?
    gamma_f1 = _shear_strain(w=[w[0], w[1]], th=[th[0], th[1]], dx=h, sign=sign) # fine shear strain so h not H
    gamma_f2 = _shear_strain(w=[w[1], w[2]], th=[th[1], th[2]], dx=h, sign=sign) # fine shear strain
    R1 = gamma_f1 - gamma_c
    R2 = gamma_f2 - gamma_c
    print(f"{R1=:.3e}, {R2=:.3e}")

    return w, th

# TODO : will this result in reasonable nodal loads? Or will this still lead to some error if we just copy shear strains to each fine midpoint tying strains basically on each edge
# certainly some high freq error, but I think it would be fine => just don't want huge blowup of tying strains due to algebraic constraint

# 2 - interp on each of 4 outer edges to determine 8 of the remaining 15/27 values (will be 7/27 left)
# 2.1 - (xi,-1) edge with gamma_13 interp
w_mask = np.array([0,3,6])
thy_mask = np.array([2,5,8])
fine_disp[w_mask], fine_disp[thy_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thy_mask], sign=1.0) 
# NOTE : sign = + for gamma_13, sign = - for gamma_23

fig, ax = plt.subplots(1, 2)
ax[0].plot([0.0, 1.0], [fine_disp[0], fine_disp[6]], label='coarse')
ax[0].plot([0.0, 0.5, 1.0], fine_disp[w_mask], '--', label='fine')
ax[0].legend()
ax[0].set_title("w(x)")
ax[1].plot([0.0, 1.0], [fine_disp[2], fine_disp[8]], label='coarse')
ax[1].plot([0.0, 0.5, 1.0], fine_disp[thy_mask], '--', label='fine')
ax[1].legend()
ax[1].set_title("thy(x)")
# plt.show()
plt.savefig(folder + "/7_1d_fsdt_interp.svg")

# 2.2 - (xi,1) edge with gamma_13 interp
w_mask = np.array([18, 21, 24])
thy_mask = np.array([20, 23, 26])
fine_disp[w_mask], fine_disp[thy_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thy_mask], sign=1.0) 
print(F"{fine_disp[w_mask]=} {fine_disp[thy_mask]=}")

# 2.3 - (-1,eta) edge with gamma_23 interp
w_mask = np.array([0, 9, 18])
thx_mask = np.array([1, 10, 19])
fine_disp[w_mask], fine_disp[thx_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thx_mask], sign=-1.0) # opp sign for gamma_23

# 2.4 - (1,eta) edge with gamma_23 interp
w_mask = np.array([6, 15, 24])
thx_mask = np.array([7, 16, 25])
fine_disp[w_mask], fine_disp[thx_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thx_mask], sign=-1.0) # opp sign for gamma_23

# 2.5 - apply avgs and gamma_23 interp to get middle w and all three thx DOF left
fine_disp[22] = 0.5 * (fine_disp[19] + fine_disp[25])
fine_disp[4] = 0.5 * (fine_disp[1] + fine_disp[7])
w_mask = np.array([3, 12, 21])
thx_mask = np.array([4, 13, 22])
# TODO : this last gamma_c isn't right.. it's huge..
# NOTE : the gamma_c rhs isn't correct here..
fine_disp[w_mask], fine_disp[thx_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thx_mask], sign=-1.0) # opp sign for gamma_23

# TODO : 2.6 - three thy DOF left, apply gamma_13 constraints and maybe some avg or penalty term on them?
# try doing averages and w,thy center solve (but this may overwrite middle w DOF and mess up the gamma_23 constraints.. see if that's true)
fine_disp[11] = 0.5 * (fine_disp[2] + fine_disp[20])
fine_disp[17] = 0.5 * (fine_disp[8] + fine_disp[26])
w_mask = np.array([9, 12, 15])
thy_mask = np.array([11, 14, 17])
fine_disp[w_mask], fine_disp[thy_mask] = oned_shear_interp(fine_disp[w_mask], fine_disp[thy_mask], sign=1.0) # + sign for gamma_13

# NOTE : overwriting w middle DOF does work here and give same w value, but it may not work in other cases or unstructured meshes
# so TBD on that..

# TODO : I still think that maybe the w DOF placed on each side, is messing up the gamma_c compute to be much larger?
# not done  yet..

"""show the matrices of solved nodal values"""
# final - show the matrices of nodal values
W_ELEM = np.reshape(fine_disp[0::3], (3,3)).copy()
THX_ELEM = np.reshape(fine_disp[1::3], (3,3)).copy()
THY_ELEM = np.reshape(fine_disp[2::3], (3,3)).copy()

# normalize each one of them so just 0 or 1 (1 if solved)
W_ELEM /= (W_ELEM + 1e-12)
THX_ELEM /= (THX_ELEM + 1e-12)
THY_ELEM /= (THY_ELEM + 1e-12)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(W_ELEM, cmap='Greys')
ax[0].set_title("w")
ax[1].imshow(THX_ELEM, cmap='Greys')
ax[1].set_title("thx")
ax[2].imshow(THY_ELEM, cmap='Greys')
ax[2].set_title("thy")
for i in range(3):
    ax[i].invert_yaxis()
# plt.show()
plt.savefig(folder + "/8_solved_fine_disps_mask.svg")

plt.close('all')

"""show the matrices of shear strains"""
coarse_red = np.array([0, 5, 10])
GAM_13_coarse = GAM_13[coarse_red,:][:,coarse_red]
GAM_23_coarse = GAM_23[coarse_red,:][:,coarse_red]

GAM_13_fine, GAM_23_fine = np.zeros((5,5)), np.zeros((5,5))

for ielem in range(4): # each fine elem
    ix, iy = ielem % 2, ielem // 2
    inode = iy * 3 + ix
    elem_nodes = [inode, inode+1, inode+4, inode+3]
    elem_dof = np.array([3*_inode+_idof for _inode in elem_nodes for _idof in range(3)])
    this_elem_disp = fine_disp[elem_dof]

    # 2.0 * for W_X, W_Y since it scales by 1/mesh size and we are now at fine level
    _W_X, _W_Y = interp_dx(this_elem_disp[0::3], dx=h), interp_dy(this_elem_disp[0::3], dx=h)
    _THX, _THY = interp(this_elem_disp[1::3]), interp(this_elem_disp[2::3])

    _GAM_13 = h/4.0 * (_THY + _W_X)
    _GAM_23 = h/4.0 * (_W_Y - _THX)

    if ielem == 0:
        cf = plt.contourf(XI, ETA, _GAM_13)     
        plt.colorbar(cf)
        plt.savefig(folder + f"/8_{ielem=}_gam_13_fine_elem1.svg")

    # get coords including midpoints for GAM_13_fine, GAM_23_fine matrices
    GAM_13_fine[2*iy:(2*iy+3), :][:, 2*ix:(2*ix+3)] = _GAM_13[coarse_red,:][:,coarse_red]
    GAM_23_fine[2*iy:(2*iy+3), :][:, 2*ix:(2*ix+3)] = _GAM_23[coarse_red,:][:,coarse_red]

# TODO : form it as a general sparse linear system, and solve the constraints, maybe with 
# norm interp smoothness otherwise?

# plot 
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
cf = ax[0,0].imshow(GAM_13_coarse, cmap='jet')
fig.colorbar(cf)
ax[0,0].set_title("gam_13 coarse")
cf = ax[0,1].imshow(GAM_23_coarse, cmap='jet')
ax[0,1].set_title("gam_23 coarse")
fig.colorbar(cf)

# print(f"{GAM_13_fine}")

cf = ax[1,0].imshow(GAM_13_fine, cmap='jet')
fig.colorbar(cf)
ax[1,0].set_title("gam_13 fine")
cf = ax[1,1].imshow(GAM_23_fine, cmap='jet')
ax[1,1].set_title("gam_23 fine")
fig.colorbar(cf)
for iax in range(4):
    ax[iax%2, iax//2].invert_yaxis() # so imshow has 0 to N upwards

plt.savefig(folder + "/9_cf_shear_strains_mats.svg")
# plt.show()

# TODO : implement it instead by forming small sparse linear systems => and then I can solve each one in parallel?
# or I can figure out some rule and distribute this inverted linear system jacobian to each element (but may be element-specific..)