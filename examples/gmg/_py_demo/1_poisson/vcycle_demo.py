import numpy as np
import matplotlib.pyplot as plt
from _pde_src import *
from _multigrid_src import *

"""setup the poisson linear system"""
nxe = 4
# nxe = 8
# nxe = 32 # choose power of 2 please
# nxe = 64
# nxe = 128
A, b = get_poisson_lhs_and_rhs(nxe=nxe)
N = A.shape[0]

# plt.imshow(A)
# plt.show()

"""compute the solution and compare against method of manufactured disps
 with np dense solve just to check we coded PDE discretization right """
x_discrete = np.linalg.solve(A, b)
# plot_poisson_soln(nx, _soln=x_discrete)

"""check error-smoothing properties of Gauss-seidel"""
N = A.shape[0]
x0 = np.random.rand(N)
# x0 = np.zeros(N)
# TODO : should I be over-relaxing here?
xhat = gauss_seidel(A, b, x0, omega=1.0, n_iter=10)

# NOTE : uncomment to check error smoothing.. should work
e_f1 = A @ x0 - b # e_f means error on fine gfrid
e_f2 = A @ xhat - b
# plot_error_comparison(nx, e_f1, e_f2)

"""make coarse operators and grid hierarchy"""
Icf, Ifc = coarse_fine_operators(nxe_fine=nxe, nxe_factor=2)

# print(f"{Icf=}")
# print(f"{Ifc=}")

# try restricting the error (that's all it can do anyways)
e_c1 = Ifc @ e_f1
e_c2 = Ifc @ e_f2

# yo => the restriction operator works!
# plot_error_comparison(nx, e_f1, e_f2)
# plot_error_comparison(nx // 2, e_c1, e_c2)

# try interpolation operator
# e_f1_2 = Icf @ e_c1
# plot_error_comparison(nxe, e_f1, e_f1_2)
# e_f2_2 = Icf @ e_c2
# plot_error_comparison(nxe, e_f2, e_f2_2)
# exit()

"""double check nxe=4 to nxe=2 mesh interp"""
# M1 = np.zeros((4,4))
# for i in range(4):
#     for j in range(4):
#         M1[i,j] = (i + j) % 2
# M2 = np.zeros((4,4))
# for i in range(4):
#     for j in range(4):
#         M2[i,j] = (i // 2 + j // 2) % 2 + 0.1 * M1[i,j]
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(M1)
# ax[1].imshow(M2)
# plt.show()

"""get a hierarchy of coarser operators and matrices"""
Icf_list = []
Ifc_list = []
A_list = []
# n_levels = np.int32(np.log(nxe) / np.log(2)) - 1 # number of levels in hierarchy
n_levels = 5
c_nxe = nxe
for i in range(n_levels):
    # print(f"{c_nx=}")
    _Icf, _Ifc = coarse_fine_operators(nxe_fine = c_nxe)
    Icf_list += [_Icf]
    Ifc_list += [_Ifc]

    _A, _ = get_poisson_lhs_and_rhs(nxe=c_nxe)
    A_list += [_A]

    c_nxe = c_nxe // 2

    # plot each _A
    # plt.imshow(_A)
    # plt.show()

plt.close('all')


"""TODO : now write a V-cycle and have multiple grids here.. maybe list of grid operators and matrices?"""

# four-grid V-cycle
# fine to coarse part first
x0 = np.zeros(b.shape[0])
x = x0.copy()
defect = b.copy() # beginning defect
last_res_norm = np.linalg.norm(defect)
import os

last_ovr_defect = defect.copy()

for i_vcycle in range(10):

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
        lhs = A_list[i_level]
        Icf, Ifc = Icf_list[i_level], Ifc_list[i_level]

        if i_level < n_levels - 1: # then smoothing
            # pre-smooth solution
            _x0 = np.zeros(lhs.shape[0])
            x_update = gauss_seidel(lhs, defect, _x0, omega=1.0, n_iter=3)

            # restrict the defect (rhs error or current resid) to the coarse grid
            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]
            defect = np.dot(Ifc, _defect)

        else: # if last level just do direct solve (v. small or coarsest level)
            x_update = np.linalg.solve(lhs, defect)

            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

        # print(f"coarse to fine defects for {i_level=} / [0-3]")
        c_nxe = nxe // 2**i_level # debug
        plot_error_comparison(c_nxe, old_defect, _defect, filename=f"{folder}/{ct}_down{i_level}_smooth_.svg")

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
        lhs = A_list[i_level]
        Icf, Ifc = Icf_list[i_level], Ifc_list[i_level]

        # interpolate the correction
        # print(f"{_coarse_update.shape=} {Icf.shape=}")
        cf_update = np.dot(Icf, _coarse_update) 
        # add correction from this level to coarser update
        fine_update = x_update_list[i_level] + cf_update

        defect = defect_list[i_level].copy()

        start_defect = defect - lhs @ cf_update

        # do a post-smoothing and update the update at this level
        _x0 = np.zeros(lhs.shape[0])
        _update = gauss_seidel(lhs, start_defect, _x0, omega=1.0, n_iter=3)
        x_update = _update + fine_update

        new_defect = start_defect - lhs @ _update
        c_nxe = nxe // 2**i_level # debug
        plot_error_comparison(c_nxe, defect, start_defect, filename=f"{folder}/{ct}_up{i_level}_interp.svg")
        ct += 1
        plot_error_comparison(c_nxe, start_defect, new_defect, filename=f"{folder}/{ct}_up{i_level}_smooth.svg")

        # new _coarse_update is then: to next level
        _coarse_update = x_update.copy()

    # check final error after V-cycle
    x += x_update
    defect = b - A @ x
    res_norm = np.linalg.norm(defect)
    print(f"vcyc{i_vcycle}: {last_res_norm=:.3e} => {res_norm=:.3e}")
    plot_error_comparison(c_nxe, last_ovr_defect.copy(), defect.copy(), filename=f"{folder}/_ovr.svg")

    last_ovr_defect = defect.copy()
    last_res_norm = res_norm

    plt.close('all')


# now check the solution against the true soln
print("final multigrid results:\n")
hifi_err = np.linalg.norm(x - x_discrete)
print(f"\t{hifi_err=:.3e}")
plot_poisson_soln(nxe, _soln=x)