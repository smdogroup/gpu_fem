"""
geometric multigrid for shells

(first version - likely has wrong scaling of coarse-fine since it is almost a direct adaptation of Poisson's methods now on thin shells)
* uses either decoupled or block Gauss-seidel (block is only node coupled not element coupled like other papers, and multiplicative smoother)
* in second version will try methods of geometric multigrid from this paper [UNSTRUCTURED MULTIGRID METHOD FOR SHELLS](https://www.columbia.edu/cu/civileng/fish/Publications_files/multigrid1996.pdf)

* with/without remove bcs, I don't get great convergence with this first version geometric multigrid, especially when I go to more thin shells
"""
import numpy as np
import matplotlib.pyplot as plt
from __src import get_tacs_matrix, delete_rows_and_columns, reduced_indices, plot_vec_compare, plot_vec_compare_all
from __src import gauss_seidel_csr, block_gauss_seidel_6dof, mg_coarse_fine_operators, sort_vis_maps, zero_non_nodal_dof
import scipy as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.


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
thickness = 0.1
# thickness = 0.02 # still somewhat thick..
# thickness = 0.001

# yes the singularity is causing poor scaling and smoothing qualities
# need to think about block / coupled smoothing steps..

# nxe_list = [32, 16, 8, 4]
nxe_list = [16, 8, 4]
# nxe_list = [4,2]

for nxe in nxe_list:
    _tacs_bsr_mat, _rhs, _xpts = get_tacs_matrix(f"in/plate{nxe}.bdf", thickness=thickness)
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
I_cf, I_fc = mg_coarse_fine_operators(nxe, sort_bk_map_list[0], sort_bk_map_list[1], 
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
    _I_cf, _I_fc = mg_coarse_fine_operators(nxe_list[i], sort_bk_map_list[i], sort_bk_map_list[i+1], 
                                      bcs_list=[bcs_list[i], bcs_list[i+1]] if not(remove_bcs) else None)
    Icf_list += [_I_cf]
    Ifc_list += [_I_fc]

"""now do multigrid V-cycle"""

# for TEMP DEBUG (this is very important input though, only when False does it operate normally..)
# zero_non_w_dof = True
zero_non_w_dof = False

# whether gauss-seidel uses block or non-block
use_block_gs = True
# use_block_gs = False

# num of gauss-seidel smoothing steps (fw and backwards)
# gs_fw, gs_bk = 10, 10 # this is a lot here
gs_fw, gs_bk = 3, 3 # this is a lot here

# now let's implement a V cycle demo (based on 1_poisson/vcycle_demo.py) 
x0 = np.zeros(rhs.shape[0])
x = x0.copy()
defect = rhs.copy() # beginning defect
last_res_norm = np.linalg.norm(defect)
init_res_norm = last_res_norm
import os

last_ovr_defect = defect.copy()

for i_vcycle in range(15):

    folder = f"out/vcyc_{i_vcycle}"
    if not os.path.exists(folder):
        os.mkdir(folder)

    x_update_list = []
    defect_list = []
    c_nxe = nxe
    ct = -1
    for i_level in range(n_levels):
        ct += 1

        # get operators for this level (and to level below)
        lhs = tacs_csr_mat_list[i_level]

        if i_level < n_levels - 1: # then smoothing
            # pre-smooth solution
            _x0 = np.zeros(lhs.shape[0])

            if use_block_gs:
                x_update = block_gauss_seidel_6dof(lhs, defect, _x0, num_iter=gs_fw)
            else:
                x_update = gauss_seidel_csr(lhs, defect, _x0, num_iter=gs_fw)
            
            # restrict the defect (rhs error or current resid) to the coarse grid
            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

            # TEMP DEBUG: zero out all error outside the w DOF
            if zero_non_w_dof:
                _defect = zero_non_nodal_dof(_defect, sort_fw_map_list[i_level], inodal=2)

            # coarsen here
            Ifc = Ifc_list[i_level]
            defect = np.dot(Ifc, _defect)

        else: # if last level just do direct solve (v. small or coarsest level)
            x_update = spsolve(lhs, defect)

            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

        # print(f"coarse to fine defects for {i_level=} / [0-3]")
        plot_vec_compare_all(nxe_list[i_level], old_defect, _defect, sort_fw_map=sort_fw_map_list[i_level], 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_smooth_.svg")
        plt.close('all')

        x_update_list += [x_update.copy()] # save the updates from pre-smoothing for later   
        # defect gets passed to the next level

    # print("----------------------\n")
    # print("coarse back to fine part of V-cycle")
    # print("----------------------\n")

    # now we iterate back up the hierarchy
    _coarse_update = x_update_list[-1]
    for i_level in range(n_levels-2, -1, -1):
        ct += 1
        # get operators for this level (and to level below)
        lhs = tacs_csr_mat_list[i_level]
        Icf = Icf_list[i_level]

        # interpolate the correction
        # print(f"{_coarse_update.shape=} {Icf.shape=}")
        _cf_update = np.dot(Icf, _coarse_update) 

        # NOTE : new line search strategy here.. (to get magnitude of update righ   t)
        defect = defect_list[i_level].copy()
        nrm0 = np.linalg.norm(defect)

        # TODO : try doing separate line search updates for the translation vs rot DOF (see if that helps)
        # first translational DOF
        cf_update = _cf_update.copy()
        for i in range(3,6):
            cf_update[i::6] *= 0.0
        omega = np.dot(cf_update, defect) / np.dot(lhs.dot(cf_update), cf_update)
        cf_update *= omega # comment out to change scaling
        fine_update = x_update_list[i_level] + cf_update
        start_defect = defect - lhs @ cf_update
        nrm1 = np.linalg.norm(start_defect)

        # then do rot DOF
        cf_update = _cf_update.copy()
        for i in range(3):
            cf_update[i::6] *= 0.0
        omega = np.dot(cf_update, start_defect) / np.dot(lhs.dot(cf_update), cf_update)
        cf_update *= omega # comment out to change scaling
        # cf_update *= -1 #
        fine_update = fine_update + cf_update
        start_defect = start_defect - lhs @ cf_update
        nrm2 = np.linalg.norm(start_defect)

        print(f"{nrm0=:.3e} {nrm1=:.3e} {nrm2=:.3e}")
        exit()

        # plot to check the update now
        plot_vec_compare_all(nxe_list[i_level], x_update_list[i_level], fine_update, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_up{i_level}_design.svg")

        # TEMP DEBUG: zero out all error outside the w DOF
        if zero_non_w_dof:
            defect = zero_non_nodal_dof(defect, sort_fw_map_list[i_level], inodal=2)
            start_defect = zero_non_nodal_dof(start_defect, sort_fw_map_list[i_level], inodal=2)

        # do a post-smoothing and update the update at this level
        _x0 = np.zeros(lhs.shape[0])

        if use_block_gs:
            _update = block_gauss_seidel_6dof(lhs, start_defect, _x0, num_iter=gs_bk)
        else:
            _update = gauss_seidel_csr(lhs, start_defect, _x0, num_iter=gs_bk)

        x_update = _update + fine_update

        new_defect = start_defect - lhs @ _update
        c_nxe = nxe // 2**i_level # debug
        plot_vec_compare_all(c_nxe, defect, start_defect, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_up{i_level}_interp.svg")
        ct += 1
        plot_vec_compare_all(c_nxe, start_defect, new_defect, sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/{ct}_up{i_level}_smooth.svg")
    
        plt.close('all')

        # new _coarse_update is then: to next level
        _coarse_update = x_update.copy()

    # check final error after V-cycle
    x += x_update
    defect = rhs - mat.dot(x)
    res_norm = np.linalg.norm(defect)
    print(f"vcyc{i_vcycle}: {last_res_norm=:.3e} => {res_norm=:.3e}")
    plot_vec_compare(c_nxe, last_ovr_defect.copy(), defect.copy(), sort_fw_map=sort_fw_map_list[i_level], 
                         filename=f"{folder}/_ovr.svg")
    
    plot_vec_compare_all(c_nxe, soln, x, sort_fw_map_list[i_level], filename=f"{folder}/_design.svg")

    if res_norm < (init_res_norm * 1e-7 + 1e-8):
        print(f"\tvcycle multigrid converged in {i_vcycle+1} steps")
        break

    last_ovr_defect = defect.copy()
    last_res_norm = res_norm

    plt.close('all')


# now check the solution against the true soln
print("final multigrid results:\n")
mat, rhs = tacs_csr_mat_list[0], rhs_list[0]
final_res = rhs - mat.dot(x)
final_res_norm = np.linalg.norm(final_res)
print(f"{init_res_norm=:.3e} => {final_res_norm=:.3e}")
plot_vec_compare_all(c_nxe, final_res.copy(), x.copy(), sort_fw_map=sort_fw_map_list[0], 
                         filename=f"out/_ovr.svg")