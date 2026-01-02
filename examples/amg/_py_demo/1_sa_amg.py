"""
SA-AMG for a plate mesh, https://link.springer.com/article/10.1007/s006070050022
"Energy Optimization of Algebraic Multigrid Bases"
"""
import sys
sys.path.append("_src/")

import numpy as np
import matplotlib.pyplot as plt
from __src import get_tacs_matrix, reduced_indices, plot_vec_compare_all, plot_plate_vec
from __src import gauss_seidel_csr, block_gauss_seidel_6dof, mg_coarse_fine_operators_v2, sort_vis_maps
from sa import get_rigid_body_modes, bgs_bsr_smoother, orthog_nullspace_projector
import scipy as sp
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smoothP", type=int, default=1, help="whether to smooth prolongation or not")
parser.add_argument("--n_gs", type=int, default=1, help="num gs smoothing steps")
parser.add_argument("--SR", type=float, default=100.0, help="slenderness ratio of plate L/h")
parser.add_argument("--n_vcyc", type=int, default=100, help="max num v-cycles")
parser.add_argument("--debug", type=int, default=0, help="debug printouts")
parser.add_argument("--plot", type=int, default=0, help="can plot intermediate results")
args = parser.parse_args()

# get TACS matrix and convert to CSR since pyAMG only supports CSR
# (this may mean it is slower or less accurate than a BSR version)
# -----------------------------------
tacs_csr_mat_list = []
tacs_bsr_mat_list = []
rhs_list = []
sort_fw_map_list = []
sort_bk_map_list = []
bcs_list = []
xpts_list = []

remove_bcs = False
thickness = 1.0 / args.SR
# nxe_list = [32, 16]
nxe_list = [16, 8]

for nxe in nxe_list:
    _tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"_src/plate{nxe}.bdf", thickness=thickness)
    # print(f"{_tacs_bsr_mat.shape=}") 
    # exit()
    _tacs_csr_mat = _tacs_bsr_mat.copy().tocsr().copy()

    tacs_bsr_mat_list += [_tacs_bsr_mat.copy()]

    _nnodes = _xpts.shape[0] // 3
    _N = 6 * _nnodes
    # zero forces on the boundary
    _free_dof = reduced_indices(_tacs_csr_mat)
    bcs = np.array([_ for _ in range(_tacs_csr_mat.shape[0]) if not(_ in _free_dof)])
    _rhs[bcs] *= 0.0
    bcs_list += [bcs]
    _free_dof = [_ for _ in range(_N)]
    
    # get the sort maps (to undo any reordering)
    _sort_fw, _sort_bk = sort_vis_maps(nxe, _xpts, _free_dof)
    sort_fw_map_list += [_sort_fw]
    sort_bk_map_list += [_sort_bk]
    
    # put in hierarchy list
    tacs_csr_mat_list += [_tacs_csr_mat.copy()]
    rhs_list += [_rhs.copy()]
    xpts_list += [_xpts.copy()]

"""plot and solve the original res error"""
mat, rhs, sort_fw, sort_bk = tacs_csr_mat_list[0].copy(), rhs_list[0].copy(), sort_fw_map_list[0], sort_bk_map_list[0]
nfree = sort_bk.shape[0]
nnodes = nfree // 6
nxe, N = int(nnodes**0.5) - 1, rhs.shape[0] # here N is the # DOF after bc removal
soln = spsolve(mat, rhs)

# this shows we get the correct solution in the SS plate case
# useful for debugging the solution as well..
# plot_vec_compare_all(nxe, rhs, soln, sort_fw, filename=None)

"""now we'll compute the coarse-fine operators at each level"""
Icf_list = []
Ifc_list = []
n_levels = len(rhs_list)

P_0 = None

smooth_prolong = args.smoothP

for i in range(n_levels-1):
    print(f"constructing cf operator on {i=} mesh level")
    _I_cf, _I_fc = mg_coarse_fine_operators_v2(nxe_list[i], sort_bk_map_list[i], sort_bk_map_list[i+1], 
                                      bcs_list=[bcs_list[i], bcs_list[i+1]] if not(remove_bcs) else None)

    # now smooth this initial prolongation
    # ==================================================

    if smooth_prolong:

        P_0 = _I_cf.copy()
        # convert to bsr matrix
        print(f"P to bsr matrix\n")
        P = bsr_matrix(P_0, blocksize=(6, 6))
        P_0 = P.copy()
        lhs = tacs_bsr_mat_list[i]
        print(f"{lhs.shape=}")

        lhs_dense = lhs.toarray()
        LplusD_dense = np.tril(lhs_dense)
        U_dense = lhs_dense - LplusD_dense
        NZ = (P != 0).toarray()

        # get the coarse rigid body modes from xpts
        print(f"get rigid body modes\n")
        B = get_rigid_body_modes(xpts_list[i+1], bcs_list[i+1], th=1.0) # as bsr style vector
        print(f"{B.shape=}")

        for ismooth in range(1):
        # for ismooth in range(5):

            # compute K * P (defect matrix)
            KP = -lhs @ P
            # P2 = KP.multiply(P != 0)  # mask to preserve fillin for P
            # maybe keep one level of fillin?
            P2 = KP.copy()
            # print(f"{P2.shape=}")

            # apply GS smoother 
            print("do smoothing step\n")
            # P3 = bgs_bsr_smoother(lhs, P2, 
            #                     #   num_iter=2,
            #                     # num_iter=3,
            #                       num_iter=1,
            #                     omega=1e-2)#PU @ UTU_inv @ U.T

            # the above sparse smoother didn't work, so trying dense instead
            # P_defect = P2 - lhs @ P2
            # P_defect = P_defect.multiply(P != 0)

            # dP = np.linalg.solve(LplusD_dense, P_defect.toarray())
            # dP = bsr_matrix(dP, blocksize=(6,6))

            # omega = 0.5
            # P3 = P2 + dP * omega

            UP2 = U_dense @ P2
            UP2 = UP2 * NZ
            P3 = np.linalg.solve(LplusD_dense, P2 - UP2)
            P3 = bsr_matrix(P3, blocksize=(6,6))
            P = P3

            # apply orthogonal projector (it definitely seemed to help)
            print("run orthogonal projector\n")
            P4 = orthog_nullspace_projector(P3, B, bcs_list[i])
            P = P4


        Icf_list += [P.toarray()]
        Ifc_list += [P.T.toarray()]
    else:
        Icf_list += [_I_cf]
        Ifc_list += [_I_fc]

if smooth_prolong:
    print("done with smoothing prolong\n")

    # check that smooth prolong actually helped
    # =============================================

    rhs_coarse = rhs_list[1]
    soln_coarse = spsolve(tacs_csr_mat_list[1], rhs_coarse)

    prolong_0 = P_0.dot(soln_coarse)
    prolong_1 = P.dot(soln_coarse)

    lhs_0 = tacs_csr_mat_list[0]
    def_0 = lhs_0.dot(prolong_0)
    def_1 = lhs_0.dot(prolong_1)

    # 3D plots for debug
    plot_vec_compare_all(nxe, prolong_0, prolong_1, sort_fw, filename=None)
    plot_vec_compare_all(nxe, def_0, def_1, sort_fw, filename=None)


    # don't go here until we do smooth prolong first
    # exit()

"""now do multigrid V-cycle"""

# num of gauss-seidel smoothing steps (fw and backwards)
gw_pre, gs_cf, gs_post = args.n_gs, args.n_gs, args.n_gs

# prolong smoothing setting
# prolong_smooth = True
prolong_smooth = False

# now let's implement a V cycle demo (based on 1_poisson/vcycle_demo.py) 
x0 = np.zeros(rhs.shape[0])
x = x0.copy()
defect = rhs.copy() # beginning defect
last_res_norm = np.linalg.norm(defect)
init_res_norm = last_res_norm
import os
lhs_0 = tacs_csr_mat_list[0]

def check_defect_err(_u_f, _defect):
    # check error here..
    full_defect = rhs - lhs_0.dot(_u_f)
    full_defect_err = np.linalg.norm(full_defect - _defect)
    print(f"\t{full_defect_err=:.3e}")
    return

defect_nrms = [init_res_norm]

"""
going through the smooth and fine-corase step by step
"""

for v_cycle in range(20):

    # don't do smoothing rn so that we can check coarse-fine operators better first (NOTE : don't do coarse-fine on loads, only disps, fine-coarse can be done on loads)
    defect_0_0 = defect.copy()

    # level 0 - smooth
    lhs_0 = tacs_csr_mat_list[0]
    # if args.LDblock:
    #     print("running v2 BGS")
    #     dx_1 = block_gauss_seidel_6dof_v2(lhs_0, defect_0_0, np.zeros(lhs_0.shape[0]), num_iter=gw_pre)
    # else:
    #     print("running v1 BGS")
    dx_1 = block_gauss_seidel_6dof(lhs_0, defect_0_0, np.zeros(lhs_0.shape[0]), num_iter=gw_pre)
    x_1 = x + dx_1
    defect_0_1 = defect_0_0 - lhs_0.dot(dx_1)
    if args.plot: # plot and print we assume are same (like debug here)
        print(f"{v_cycle=} : 1 - pre smooth")
        plot_vec_compare_all(nxe_list[0], defect_0_0, defect_0_1, sort_fw_map=sort_fw_map_list[0], 
                                filename=None)
        
    if args.debug: check_defect_err(x_1, defect_0_1)

    # level 0 to 1 - coarsen 
    defect_1_0 = np.dot(Ifc_list[0], defect_0_1)

    # solve at level 1 (NOTE : we're just doing 2 grids for rn)
    lhs_1 = tacs_csr_mat_list[1]
    dx_1 = spsolve(lhs_1, defect_1_0)

    # coarse-fine the disps
    _dx_cf_0 = np.dot(Icf_list[0], dx_1)

    # loads of coarse-fine update
    _df_cf_0 = lhs_0.dot(_dx_cf_0)
    _dx_cf_1, _df_cf_1 = _dx_cf_0.copy(), _df_cf_0.copy()

    # re-scale using one DOF min
    s = _dx_cf_1
    omega = np.dot(defect_0_1, s) / np.dot(s, lhs_0.dot(s))
    print(f"{omega=:.2e}")
    # compute equiv loads

    defect_0_2 = defect_0_1 - lhs_0.dot(omega * s)
    x_2 = x_1 + omega * s

    cf_nrm_1, cf_nrm_2 = np.linalg.norm(defect_0_1), np.linalg.norm(defect_0_2)

    if args.plot:
        # temp debug printouts
        p1 = np.dot(defect_0_1, s)
        _df = lhs_0.dot(s)
        p2 = np.dot(_df, s)
        print(f"\t{omega=:.3e}, {p1=:.3e}, {p2=:.3e}")

        plot_vec_compare_all(nxe_list[0], defect_0_1, s, sort_fw_map=sort_fw_map_list[0], 
                                filename=None)

        plot_vec_compare_all(nxe_list[0], defect_0_1, _df, sort_fw_map=sort_fw_map_list[0], 
                                filename=None)

    if args.plot:
        print(f"{v_cycle=} : 3 - coarse-fine update")
        print(f"\t{omega=:.3e} {cf_nrm_1=:.3e} => {cf_nrm_2=:.3e}")

        plot_vec_compare_all(nxe_list[0], defect_0_1, defect_0_2, sort_fw_map=sort_fw_map_list[0], 
                                filename=None)
    if args.debug: check_defect_err(x_2, defect_0_2)

    dx_2 = block_gauss_seidel_6dof(lhs_0, defect_0_2, np.zeros(lhs_0.shape[0]), num_iter=gs_post)
    defect_0_3 = defect_0_2 - lhs_0.dot(dx_2)
    x_3 = x_2 + dx_2
    if args.plot:
        print(f"{v_cycle=} : 4 - post smooth")
        plot_vec_compare_all(nxe_list[0], defect_0_2, defect_0_3, sort_fw_map=sort_fw_map_list[0], 
                                filename=None)
    if args.debug: check_defect_err(x_3, defect_0_3)

    defect, x = defect_0_3.copy(), x_3.copy()

    c_norm = np.linalg.norm(defect)
    defect_nrms += [c_norm]
    print(f"v-cycle step {v_cycle}, {c_norm=:.4e}")
    if c_norm < 1e-6 * init_res_norm:
        print(f"v-cycle converged in {v_cycle} steps")
        break

"""show final solution.."""

final_defect = rhs.copy() - lhs_0.dot(x)
final_res_norm = np.linalg.norm(final_defect)
nrm_ratio = init_res_norm / final_res_norm
print(f"{init_res_norm=:.3e} {final_res_norm=:.3e} or {nrm_ratio=:.3e}, in {v_cycle=} steps")
# TODO : check soln error also?
err_nrm = np.linalg.norm(soln - x)
print(f"{err_nrm=:.3e}")

_n = 2
n_total_gs = args.n_gs * _n * v_cycle
print(f"{n_total_gs=} (# total GS steps)")

if args.plot:
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
    fig, ax = plt.subplots(1, 3, figsize=(13, 7))
    dof_strs = ['w', 'thx', 'thy']
    for i, idof in enumerate([2, 3, 4]):
        dof_str = dof_strs[i]
        c_ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_plate_vec(nxe=nxe, vec=x.copy(), ax=c_ax, sort_fw=sort_fw_map_list[0], nodal_dof=idof, cmap='RdBu_r')
    plt.tight_layout()
    plt.show()

    print(f"{init_res_norm=:.3e} {final_res_norm=:.3e} or {nrm_ratio=:.3e}, in {v_cycle=} steps")
    print(f"{n_total_gs=} (# total GS steps)")

    # v-cycle resids here
    iters = [_ for _ in range(len(defect_nrms))]
    plt.plot(iters, defect_nrms, 'ko-')
    plt.xlabel("V-cycles")
    plt.ylabel("Defect residual")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    print(f"{init_res_norm=:.3e} {final_res_norm=:.3e} or {nrm_ratio=:.3e}, in {v_cycle=} steps")
    print(f"{n_total_gs=} (# total GS steps)")