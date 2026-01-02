"""
geometric multigrid for shells

V2 - TBD.. see multigrid PR on github / gpu_fem
"""

"""
geometric multigrid for shells

(first version - likely has wrong scaling of coarse-fine since it is almost a direct adaptation of Poisson's methods now on thin shells)
* uses either decoupled or block Gauss-seidel (block is only node coupled not element coupled like other papers, and multiplicative smoother)
* in second version will try methods of geometric multigrid from this paper [UNSTRUCTURED MULTIGRID METHOD FOR SHELLS](https://www.columbia.edu/cu/civileng/fish/Publications_files/multigrid1996.pdf)

* with/without remove bcs, I don't get great convergence with this first version geometric multigrid, especially when I go to more thin shells
"""
import numpy as np
import matplotlib.pyplot as plt
from __src import get_tacs_matrix, delete_rows_and_columns, reduced_indices, plot_vec_compare, plot_vec_compare_all, plot_plate_vec
from __src import gauss_seidel_csr, block_gauss_seidel_6dof, mg_coarse_fine_operators_v1, mg_coarse_fine_operators_v2, sort_vis_maps, zero_non_nodal_dof
from __src import mg_coarse_fine_operators_v3, mg_coarse_fine_operators_v4, mg_coarse_fine_transv_shear_smooth
# from __src import block_gauss_seidel_6dof_v2
import scipy as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cf_smooth", type=int, default=0, help="smooth coarse-fine step")
parser.add_argument("--trv_shear", type=int, default=0, help="turn on strain-disp transv shear smooth step (cheap)")
# parser.add_argument("--LDblock", type=int, default=0, help="only use L+D part of each nodal block in BGS smoother")
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
rhs_list = []
sort_fw_map_list = []
sort_bk_map_list = []
bcs_list = []

# normally you remove dirichlet bcs in multigrid
# but you don't have to if you're careful (and it will be tricky to remove bcs for block Gauss-seidel smoothing, TBD)
# remove_bcs = True
remove_bcs = False

# set thickness of the shells to see if this is affecting smoothing..
# thickness = 1.0
# thickness = 0.1
# thickness = 0.02 # still somewhat thick..
# thickness = 0.001
thickness = 1.0 / args.SR

# yes the singularity is causing poor scaling and smoothing qualities
# need to think about block / coupled smoothing steps..

# nxe_list = [32, 16, 8, 4]
# nxe_list = [16, 8, 4]
# nxe_list = [8, 4]
nxe_list = [32, 16]
# nxe_list = [16, 8]
# nxe_list = [4,2]

for nxe in nxe_list:
    _tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"_archive/plate{nxe}.bdf", thickness=thickness)
    _tacs_csr_mat = _tacs_bsr_mat.tocsr()

    _nnodes = _xpts.shape[0] // 3
    _N = 6 * _nnodes

    # if nxe == 4:
    #     plt.imshow(_tacs_csr_mat.todense())
    #     plt.show()

    # remove bcs..
    if remove_bcs:
        _free_dof = reduced_indices(_tacs_csr_mat)
        bcs = [_ for _ in range(_tacs_csr_mat.shape[0]) if not(_ in _free_dof)]
        _tacs_csr_mat = delete_rows_and_columns(_tacs_csr_mat, dof=_free_dof)
        _rhs = _rhs[_free_dof]
    
    else: # not removing bcs
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

"""plot and solve the original res error"""
mat, rhs, sort_fw, sort_bk = tacs_csr_mat_list[0].copy(), rhs_list[0].copy(), sort_fw_map_list[0], sort_bk_map_list[0]
nfree = sort_bk.shape[0]
nnodes = nfree // 6
nxe, N = int(nnodes**0.5) - 1, rhs.shape[0] # here N is the # DOF after bc removal
soln = spsolve(mat, rhs)

# this shows we get the correct solution in the SS plate case
# useful for debugging the solution as well..
# plot_vec_compare_all(nxe, rhs, soln, sort_fw, filename=None)

"""now we try error smoothing on fine grid first"""
# let's work first on the fine grid, see if smoothing even works..
x0, res = np.zeros(N), rhs.copy()
x = x0.copy()
x = gauss_seidel_csr(mat, rhs, x, num_iter=5)
# x = block_gauss_seidel_6dof(mat, rhs, x, num_iter=5)
res2 = res - mat.dot(x)

# check if bcs are correct
# res_zero = np.where(res == 0)
# print(f"{res=}")
# print(F"{bcs=}")

# see whether residual smoothing happens or not?
# plot_vec_compare(nxe, res, res2, sort_fw, filename=None)

"""compute and try coarse-fine operators (check error shape stays the same)"""
# coarse_fine_method = mg_coarse_fine_operators_v1 # poisson stencil
coarse_fine_method = mg_coarse_fine_operators_v2 # 1st order accurate lagrange (simple avging => this is fine.. even if high freq error, smoothing reduces that)
# coarse_fine_method = mg_coarse_fine_operators_v3 # 2nd order accurate lagrange
# coarse_fine_method = mg_coarse_fine_operators_v4 # 4th order accurate lagrange

I_cf, I_fc = coarse_fine_method(nxe, sort_bk_map_list[0], sort_bk_map_list[1], 
                                      bcs_list=[bcs_list[0], bcs_list[1]] if not(remove_bcs) else None)
# plt.imshow(I_fc)
# plt.show()
res_c = np.dot(I_fc, res)
res2_c = np.dot(I_fc, res2)
res_f = np.dot(I_cf, res_c)
res2_f = np.dot(I_cf, res2_c)

# it is working now

# see whether residuals look similar on the coarse mesh
# plot_vec_compare_all(nxe//2, res_c, res2_c, sort_fw_map_list[1], filename=None)

# # # and then back to the fine mesh
# plot_vec_compare_all(nxe, res_f, res2_f, sort_fw, filename=None)

"""now we'll compute the coarse-fine operators at each level"""
Icf_list = []
Ifc_list = []
n_levels = len(rhs_list)

for i in range(n_levels-1):
    _I_cf, _I_fc = coarse_fine_method(nxe_list[i], sort_bk_map_list[i], sort_bk_map_list[i+1], 
                                      bcs_list=[bcs_list[i], bcs_list[i+1]] if not(remove_bcs) else None)
    Icf_list += [_I_cf]
    Ifc_list += [_I_fc]

"""now do multigrid V-cycle"""

# num of gauss-seidel smoothing steps (fw and backwards)
gw_pre, gs_cf, gs_post = args.n_gs, args.n_gs, args.n_gs

# prolong smoothing setting
prolong_smooth = True
# prolong_smooth = False

# now let's implement a V cycle demo (based on 1_poisson/vcycle_demo.py) 
x0 = np.zeros(rhs.shape[0])
x = x0.copy()
defect = rhs.copy() # beginning defect
last_res_norm = np.linalg.norm(defect)
init_res_norm = last_res_norm
import os

lhs_0 = tacs_csr_mat_list[0]

# plt.imshow(lhs_0.toarray())
# plt.show()

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

for v_cycle in range(300):

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

    # coarse-fine transv shear correction
    if args.trv_shear:
        _dx_cf_0_2 = mg_coarse_fine_transv_shear_smooth(nxe, _dx_cf_0, h=1.0/nxe, sort_fw_arr=sort_fw_map_list[0])
        _df_cf_0_2 = lhs_0.dot(_dx_cf_0_2)

        if args.plot:
            print(f"2.0.1 - coarse-fine transv shear correction disp")
            plot_vec_compare_all(nxe_list[0], _dx_cf_0, _dx_cf_0_2, sort_fw_map=sort_fw_map_list[0], 
                                    filename=None)
            
            print(f"2.0.2 - coarse-fine transv shear correction loads")
            plot_vec_compare_all(nxe_list[0], defect_0_1, _df_cf_0_2, sort_fw_map=sort_fw_map_list[0], 
                                    filename=None)
        _dx_cf_0 = _dx_cf_0_2.copy()
        _df_cf_0 = _df_cf_0_2.copy()
        

    # TODO : need smoothing of coarse-fine? (neg loads because we add neg loads to next defect)
    # NOTE : this is prolong smoothing here..
    if args.cf_smooth: # NOTE : this is a good settin I think..
        # if args.LDblock:
        #     _dx_cf_update = block_gauss_seidel_6dof_v2(lhs_0, -1.0 * _df_cf_0, np.zeros(lhs_0.shape[0]), num_iter=gs_cf)
        # else:
        _dx_cf_update = block_gauss_seidel_6dof(lhs_0, -1.0 * _df_cf_0, np.zeros(lhs_0.shape[0]), num_iter=gs_cf)
        _dx_cf_1 = _dx_cf_0 + _dx_cf_update
        _df_cf_1 = lhs_0.dot(_dx_cf_1)

        if args.plot:
            print(f"{v_cycle=} : 2 - prolong smooth")
            if args.cf_smooth:
                plot_vec_compare_all(nxe_list[0], _df_cf_0, _df_cf_1, sort_fw_map=sort_fw_map_list[0], 
                                        filename=None)
            else:
                plot_vec_compare_all(nxe_list[0], _dx_cf_0, _df_cf_0, sort_fw_map=sort_fw_map_list[0], 
                                        filename=None)
    else:
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

    # post-smoothing
    # if args.LDblock:
    #     dx_2 = block_gauss_seidel_6dof_v2(lhs_0, defect_0_2, np.zeros(lhs_0.shape[0]), num_iter=gs_post)
    # else:
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

_n = 3 if args.cf_smooth else 2
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