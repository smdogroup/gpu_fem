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
from __src import get_tacs_matrix, delete_rows_and_columns, reduced_indices, plot_vec_compare, plot_vec_compare_all
from __src import gauss_seidel_csr, block_gauss_seidel_6dof, mg_coarse_fine_operators_v1, mg_coarse_fine_operators_v2, sort_vis_maps, zero_non_nodal_dof
from __src import mg_coarse_fine_operators_v3, mg_coarse_fine_operators_v4
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
# thickness = 0.1
thickness = 0.02 # still somewhat thick..
# thickness = 0.001

# yes the singularity is causing poor scaling and smoothing qualities
# need to think about block / coupled smoothing steps..

# nxe_list = [32, 16, 8, 4]
# nxe_list = [16, 8, 4]
# nxe_list = [8, 4]
nxe_list = [32, 16]
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
# coarse_fine_method = mg_coarse_fine_operators_v1 # poisson stencil
coarse_fine_method = mg_coarse_fine_operators_v2 # 1st order accurate lagrange
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

"""
going through the smooth and fine-corase step by step
"""

# level 0 - smooth
# lhs_0 = tacs_csr_mat_list[0]
# dx_1 = block_gauss_seidel_6dof(lhs_0, defect, np.zeros(lhs_0.shape[0]), num_iter=gs_fw)
# defect_0_0 = defect - lhs_0 @ dx_1

# don't do smoothing rn so that we can check coarse-fine operators better first (NOTE : don't do coarse-fine on loads, only disps, fine-coarse can be done on loads)
defect_0_0 = defect.copy()

# level 0 to 1 - coarsen 
defect_1_0 = np.dot(Ifc_list[0], defect_0_0)

# # level 1 - smooth
# lhs_1 = tacs_csr_mat_list[1]
# dx_1 = block_gauss_seidel_6dof(lhs_1, defect_1_0, np.zeros(lhs_1.shape[0]), num_iter=gs_fw)
# defect_1_1 = defect_1_0 - lhs_1 @ dx_1

# skip smoothing for now, to just test coarse-fine first
defect_1_1 = defect_1_0.copy()

# solve at level 1 (NOTE : we're just doing 2 grids for rn)
lhs_1 = tacs_csr_mat_list[1]
dx_1 = spsolve(lhs_1, defect_1_1)

# check the loads and soln at coarser mesh
# plot_vec_compare_all(nxe_list[1], defect_1_1, dx_1, sort_fw_map=sort_fw_map_list[1], 
#                         filename=None)

# coarse-fine the disps
_dx_0 = np.dot(Icf_list[0], dx_1)
lhs_0 = tacs_csr_mat_list[0]

# compare with actual dx_0 solved?
dx_0 = spsolve(lhs_0, defect_0_0)

# temp debug: zero out rot DOF in soln?
# _dx_0[3::6] *= 0.0
# _dx_0[4::6] *= 0.0

# compute equiv loads
_df_0 = lhs_0 @ _dx_0

print(f"{_dx_0.shape=} {_df_0.shape=} {defect_0_0.shape=}")

# compare disps coarse-fine to true disps of exact soln
plot_vec_compare_all(nxe_list[0], dx_0, _dx_0, sort_fw_map=sort_fw_map_list[0], 
                        filename=None)

# compare loads to true loads now also
# plot_vec_compare_all(nxe_list[0], defect_0_0, _df_0, sort_fw_map=sort_fw_map_list[0], 
#                         filename=None)

# show just w part of the loads
plot_vec_compare(nxe_list[0], defect_0_0, _df_0, sort_fw_map_list[0], nodal_dof=2)

# try doing line search update to subtract out
s = _dx_0
omega = np.dot(defect_0_0, s) / np.dot(lhs_0.dot(s), s)
print(f"{omega=}")
defect_0_1 = defect_0_0 - lhs_0.dot(omega * s)
plot_vec_compare_all(nxe_list[0], defect_0_0, defect_0_1, sort_fw_map=sort_fw_map_list[0], 
                        filename=None)

exit()