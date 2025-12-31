from src.assembler import *
from src.linalg import ILU0Preconditioner, pcg, gauss_seidel_csr, block_gauss_seidel
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

"""test a two-grid version first for geometric multigrid"""

"""begin with testing coarse-fine interp operators"""

# fine settings
# nxe, nxc = 64, 4
# nxe, nxc = 48, 4
nxe, nxc = 32, 4
# nxe, nxc = 8, 2

# n_levels = 4
n_levels = 3

m, n = 3, 1
load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

plates = [
    PlateAssembler(
        material=IsotropicMaterial.aluminum(),
        plate_fem_geom=PlateFemGeom(nxe=_nxe, nye=_nxe, nxh=nxc, nyh=nxc, a=1.0, b=1.0),
        plate_loads=PlateLoads(qmag=2e-2, load_fcn=load_fcn),
        rho_KS=100.0, can_print=False
    )  for _nxe in [nxe, nxe//2]
]

fine_plate = plates[0]
coarse_plate = plates[1]

ndv = fine_plate.ncomp
dvs = np.array([5e-3] * ndv)

for plate in plates:
    plate.solve_forward(dvs)
#     plate.plot_disp(figsize=(7,5))

"""now test the coarse-fine and fine-coarse operators (directly in FEA software)"""

# checks out, they match
# print(f"{fine_plate.bcs=}")
# print(f"{coarse_plate.fine_bcs=}")

# fine_vec = np.random.rand(fine_plate.num_dof)
fine_soln, coarse_soln = fine_plate.u.copy(), coarse_plate.u.copy()

# fine to coarse and back (some scaling occurs in fine to coarse as we accumulate multiple fine nodes to coarse like 9x or something)
# fine_vec = fine_soln
# coarse_vec = coarse_plate.to_coarse(fine_vec)
# fine_vec2 = coarse_plate.to_fine(coarse_vec)
# fine_plate.plot_vector(fine_vec), coarse_plate.plot_vector(coarse_vec), fine_plate.plot_vector(fine_vec2)
# exit()

# coarse to fine does not have any scaling issues (direct interpolation with FEA basis)
# coarse to fine only (my coarse-fine and fine-coarse operators are the transpose of each other [not just injection for fine-coarse])
# coarse_vec = coarse_soln
# fine_vec = coarse_plate.to_fine(coarse_vec)
# coarse_plate.plot_vector(coarse_vec), fine_plate.plot_vector(fine_vec)

"""try smoothing of error"""
fine_kmat = fine_plate.Kmat.copy()
fine_rhs = fine_plate.force.copy()
soln = spsolve(fine_kmat.copy(), fine_rhs)

init_res = np.random.rand(fine_plate.num_dof)
init_res =  fine_soln / np.linalg.norm(fine_soln) + 0.1 * init_res
init_res[fine_plate.bcs] = 0.0
# init_res = fine_soln.copy()

# test the ILU0 smoother works ok on small problem, not really large problem (it does now even with permutations to stabilize ILU(0)
# -----------------------------
# x0 = np.zeros(fine_plate.num_dof)
# # gauss seidel smoother much more successful than ILU(0) right now at error smoothing..
# # it seems ILU(0) is doing global solves, which isn't really smoothing anything.. ?
# # fine_ilu0 = ILU0Preconditioner(fine_kmat)
# # x = pcg(A=fine_kmat, b=init_res, M=fine_ilu0, maxiter=5)
# x = gauss_seidel_csr(A=fine_kmat, b=init_res, x0=x0.copy(), num_iter=5)

# final_res = init_res - fine_kmat.dot(x)
# init_norm, final_norm = np.linalg.norm(init_res), np.linalg.norm(final_res)
# print(f"{init_norm=:.3e}, {final_norm=:.3e}")
# fine_plate.plot_vector(init_res)
# fine_plate.plot_vector(final_res)
# exit()

"""construct multigrid hierarchy"""

# construct the grid hierarchy
mat_list, plate_list = [], []
c_nxe = 2 * nxe
for i in range(n_levels):
    c_nxe = c_nxe // 2
    # print(f"{i=} {c_nxe=}")

    _plate = PlateAssembler(
        material=IsotropicMaterial.aluminum(),
        plate_fem_geom=PlateFemGeom(nxe=c_nxe, nye=c_nxe, nxh=nxc, nyh=nxc, a=1.0, b=1.0),
        plate_loads=PlateLoads(qmag=2e-2, load_fcn=load_fcn),
        rho_KS=100.0, can_print=False
    ) 
    _plate._compute_mat_vec(dvs)
    mat_list += [_plate.Kmat.copy()]
    plate_list += [_plate]

x0 = np.zeros(fine_plate.num_dof)
defect = fine_rhs.copy()
last_res_norm = np.linalg.norm(defect)
init_res_norm = last_res_norm
import os
x = x0.copy()

"""check first a hermite cubic 1d w and dw/dx interp"""
vals = np.zeros(4)
# function here is basically sin(x) from x = 0.0 to 0.5
dx = 0.5
# dx = 0.1
vals[0] = np.sin(np.pi * 0.0)
vals[1] = np.pi * np.cos(0.0) * dx / 2 # so becomes dw/dxi same units as w for same vec hermite cubic DOF
vals[2] = np.sin(np.pi * dx)
vals[3] = np.pi * np.cos(dx) * dx / 2

# first plot the 1d hermite cubic basis
from src._plate_elem import hermite_cubic_polynomials_1d, eval_polynomial
xi_vec = np.linspace(-1, 1, 100)
# for ibasis in range(4):
#     poly = hermite_cubic_polynomials_1d(ibasis)
#     h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
#     plt.plot(xi_vec, h_vec, label=f"phi_{ibasis}")
# plt.legend()
# plt.show()

# check w hermite cubic curve here
# x_in = (xi_vec + 1.0) / 2.0 * dx
# w_out = np.zeros_like(xi_vec)
# fig, ax = plt.subplots(1, 2)
# for ibasis in range(4):
#     poly = hermite_cubic_polynomials_1d(ibasis)
#     h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
#     w_out += vals[ibasis] * h_vec
# w_out2 = np.sin(np.pi * x_in)
# ax[0].plot(x_in, w_out)
# ax[0].plot(x_in, w_out2, '--')
# ax[0].set_title("w(x)")

# check now the dw/dx hermite cubic curve
# wxi_out = np.zeros_like(xi_vec)
# for ibasis in range(4):
#     poly = hermite_cubic_polynomials_1d(ibasis)
#     h_vec = np.array([eval_polynomial(poly, xi) for xi in xi_vec])
#     wxi_out += vals[ibasis] * h_vec
# wx_out = wxi_out * 2 / dx
# wx_out2 = np.pi * np.cos(np.pi * x_in)
# ax[1].plot(x_in, wx_out)
# ax[1].plot(x_in, wx_out, '--')
# ax[1].set_title("dw/dx(x)")
# plt.show()

# exit()

"""TODO : then check a hermite cubic 2d interp of w and dw/dx"""
rhs_2 = plate_list[2].force.copy()
elem_vals = list(rhs_2[3*(9+1):3*12]) + list(rhs_2[3*(18+1):3*21])
print(f"{elem_vals=}")
# plate_list[2].plot_vector(rhs_2)

from src._plate_elem import hermite_cubic_2d, get_gradient

# xi, eta = np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
# XI, ETA = np.meshgrid(xi, eta)
# W, W_XI, W_TRUE, W_XI_TRUE = np.zeros_like(XI), np.zeros_like(XI), np.zeros_like(XI), np.zeros_like(XI)
# max_val = np.max(rhs_2[0::3])
# max_val2 = np.max(rhs_2[1::3])
# for i in range(10):
#     for j in range(10):
#         for ibasis in range(12):
#             W[i,j] += elem_vals[ibasis] * hermite_cubic_2d(ibasis, XI[i,j], ETA[i,j])
#             W_XI[i,j] += elem_vals[ibasis] * get_gradient(ibasis, XI[i,j], ETA[i,j], xscale=1.0, yscale=1.0)[0]

#         y, x = 0.125*(i/9.0 + 1.0), 0.125*(j/9.0 + 1.0)
#         W_TRUE[i,j] = max_val * np.sin(np.pi * 3 * x) * np.sin(np.pi * y)
#         W_XI_TRUE[i,j] = max_val2 * np.cos(np.pi * 3 * x) * np.sin(np.pi * y)

# fig, axs = plt.subplots(1, 2, figsize=(15, 8))  # Create a 2x3 grid of subplots
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.plot_surface(XI, ETA, W_TRUE, color='tab:gray')
# ax1.plot_surface(XI, ETA, W, color='b')
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# ax2.plot_surface(XI, ETA, W_XI_TRUE, color='tab:gray')
# ax2.plot_surface(XI, ETA, W_XI, color='b')
# plt.show()
# exit()

"""test only the coarse fine operator on dw/dx DOF"""
rhs_2 = plate_list[2].force.copy()
rhs_1 = plate_list[2].to_fine(rhs_2)
rhs_1_orig = plate_list[1].force.copy()

idof = 0 # check w DOF matches (it does)
# idof = 1
# idof = 2

# fig, axs = plt.subplots(1, 3, figsize=(15, 8))  # Create a 2x3 grid of subplots
# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# cf = plate_list[2]._plot_field_on_ax(
#     field=rhs_2[idof::3],
#     ax=ax1,
#     log_scale=False,
#     cmap='turbo',
#     surface=True,
#     elem_to_node_convert=False,
# )
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# cf = plate_list[1]._plot_field_on_ax( # check the orig RHS rot DOF scales are good now?
#     field=rhs_1_orig[idof::3],
#     ax=ax2,
#     log_scale=False,
#     cmap='turbo',
#     surface=True,
#     elem_to_node_convert=False,
# )
# ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# cf2 = plate_list[1]._plot_field_on_ax(
#     field=rhs_1[idof::3],
#     ax=ax3,
#     log_scale=False,
#     cmap='turbo',
#     surface=True,
#     elem_to_node_convert=False,
# )
# plt.show()
# exit()

"""now go through one V-cycle step by step, with sanity checks at each place"""
# level 0 - smooth
lhs_0 = mat_list[0]

# dx_0 = block_gauss_seidel(lhs_0, defect, np.zeros(lhs_0.shape[0]), num_iter=3)
# plate_list[0].plot_vectors(soln, dx_0, filename=None)
# defect_0_0 = defect - lhs_0 @ dx_0
# plate_list[0].plot_vectors(defect, defect_0_0, filename=None)

# temp don't smoothen to test coarse-fine
defect_0_0 = defect.copy()

# level 0 to 1 - coarsen 
defect_1_0 = plate_list[1].to_coarse(defect_0_0)

# check right scaling of rot force at each layer
# plate_list[1].plot_vectors(plate_list[1].force.copy(), defect_1_0, filename=None)
# exit()

# level 1 - smooth
# lhs_1 = mat_list[1]
# dx_1 = block_gauss_seidel(lhs_1, defect_1_0, np.zeros(lhs_1.shape[0]), num_iter=3)
# defect_1_1 = defect_1_0 - lhs_1 @ dx_1

# temp don't smooth to test coarse-fine
defect_1_1 = defect_1_0.copy()

# level 1 to 2 - coarsen
defect_2_0 = plate_list[2].to_coarse(defect_1_1)

# w_rhs_nrms = [np.round(np.max(_vec[0::3]),5) for _vec in [defect_0_0, defect_1_0, defect_2_0]]
# print(f"{w_rhs_nrms=}")
# wxi_rhs_nrms = [np.round(np.max(_vec[1::3]),5) for _vec in [defect_0_0, defect_1_0, defect_2_0]]
# print(f"{wxi_rhs_nrms=}")

# NOTE : don't check the coarse-fine of a vec (the rhs don't have good agreement of w and w_xi and w_eta DOF in hermite cubic)
# resulting in weird disp transfer (coarse-fine), this is specific to hermite cubic

# DEBUG / TESTING for how trans/rot DOF affected by force trans/rot force scaling (really a hermite cubic specific issue here..)
# ----------------------------------------
# TODO : trying here to adjust the RHS rot / translational to change the final disp rot/translational ratio in hermite cubic
# can't change trans/rot ratio after the fact because that messes up hermite cubic interp
# r = 0.5
# defect_2_0_temp = defect_2_0.copy()
# for i in range(1, 3):
#     defect_2_0_temp[1::3] *= r
# dx_2_temp = spsolve(mat_list[2].copy(), defect_2_0)
# _dx_1_temp = plate_list[2].to_fine(dx_2_temp)
# dx_2_temp2 = spsolve(mat_list[2].copy(), defect_2_0_temp)
# _dx_1_temp2 = plate_list[2].to_fine(dx_2_temp2)
# plate_list[1].plot_vectors(_dx_1_temp, _dx_1_temp2)

# should really only do coarse-fine on the disps? YES (specific to hermite-cubic where rhs w and w_xi and w_eta)
# you can do coarse-fine on rhs, but we will never actually do that in practice, just need good coarse-fine of disps and fine-coarse of rhs
dx_2 = spsolve(mat_list[2].copy(), defect_2_0)
dx_1 = spsolve(mat_list[1].copy(), defect_1_1)

# try halving rot DOF of dx_2 first (mesh size change only for _xi and _eta DOF in hermite cubic)
# 4x because of accumulation issues in hermite cubic?
dx_2 *= 4.0 
# try pre-scaling the rot DOF by 0.5x see if that works.. NO it doesn't work cause then w DOF still cause similar mag but non-continuous rotations..
# dx_2[1::3] *= 0.5
# dx_2[2::3] *= 0.5

_dx_1 = plate_list[2].to_fine(dx_2)
lhs_1 = mat_list[1].copy()

# compare disps of actual fine mesh and the suggested upate
plate_list[1].plot_vectors(dx_1, _dx_1)

# disp diff
# disp_err = _dx_1 - dx_1
# for i in range(3):
#     disp_err[i::3] /= np.max(dx_1[i::3])

# plate_list[1].plot_vectors(dx_1, disp_err)

# can we just update rot DOF only? Yes that actually kind of worked..
# _dx_1[0::3] = 0.0
# what about just update translational DOF (worked too!)
# _dx_1[1::3] = 0.0
# _dx_1[2::3] = 0.0

df_1 = lhs_1 @ dx_1
_df_1 = lhs_1 @ _dx_1
# check the effect of _dx_1 on the forces
plate_list[1].plot_vectors(df_1, _df_1)

# try regular full disp update
# TODO : this scaling needed to get it to match, but maybe this messes up the hermite cubic continuity?
# _dx_1[1::3] *= 0.5
# _dx_1[2::3] *= 0.5
omega = np.dot(defect_1_1, _dx_1) / np.dot(_dx_1, lhs_1.dot(_dx_1))
defect_1_2 = defect_1_1 - lhs_1 @ (omega * _dx_1)
plate_list[1].plot_vectors(defect_1_1, defect_1_2)

# # first apply just w DOF (with line search on min pot energy)
# _dx_1_tr = _dx_1.copy()
# _dx_1_tr[1::3] = 0.0
# _dx_1_tr[2::3] = 0.0
# omega = np.dot(_dx_1_tr, defect_1_1) / np.dot(_dx_1_tr, lhs_1.dot(_dx_1_tr))
# # omega *= -1
# defect_1_2 = defect_1_1 - lhs_1 @ (omega * _dx_1_tr)
# plate_list[1].plot_vectors(defect_1_1, defect_1_2)
# # then apply the translational DOF (with line search on min pot energy)
# _dx_1_rot = _dx_1.copy()
# _dx_1_rot[0::3] = 0.0
# omega = np.dot(_dx_1_rot, defect_1_2) / np.dot(_dx_1_rot, lhs_1.dot(_dx_1_rot))
# defect_1_3 = defect_1_2 - lhs_1 @ (omega * _dx_1_rot)
# plate_list[1].plot_vectors(defect_1_2, defect_1_3)