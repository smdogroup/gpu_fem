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

"""now try multigrid V-cycle with ILU(0) smoothing"""

# construct the grid hierarchy
mat_list, plate_list = [], []
c_nxe = 2 * nxe
for i in range(n_levels):
    c_nxe = c_nxe // 2
    print(f"{i=} {c_nxe=}")

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

# # check coarsen.. works fine here..
# defect2 = plate_list[1].to_coarse(defect)
# plate_list[1].plot_vector(defect2)
# exit()

last_ovr_defect = defect.copy()
if not os.path.exists("out"): os.mkdir("out")

for i_vcycle in range(20):

    folder = f"out/vcyc_{i_vcycle}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    x_update_list, defect_list, c_nxe, ct = [], [], nxe, -1

    for i_level in range(n_levels):
        ct += 1

        # get operators for this level (and to level below)
        lhs = mat_list[i_level].copy()

        if i_level < n_levels - 1: # then smoothing
            # pre-smooth solution
            _x0 = np.zeros(lhs.shape[0])
            # TBD : could I use ILU(0) here? or some PCG version of ILU(0) right precond?
            # x_update = gauss_seidel_csr(lhs, defect, _x0, num_iter=3)
            x_update = block_gauss_seidel(lhs, defect, _x0, num_iter=3, ndof=3)
            
            # restrict the defect (rhs error or current resid) to the coarse grid
            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

            plate_list[i_level].plot_vectors(defect, _defect, filename=None)
            exit()

            plate_list[i_level].plot_vector(defect, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_1rhs.svg")
            
            plate_list[i_level].plot_vector(_defect, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_2smooth.svg")
            
            diff = _defect - defect
            plate_list[i_level].plot_vector(diff, filename=None)

            # coarsen here (TODO : why is the smoothing then messing up my coarse fine operator?)
            # must have something to do with the rot DOF.. something is still wonky..
            # defect2 = plate_list[i_level+1].to_coarse(defect)
            defect = plate_list[i_level+1].to_coarse(_defect)

            # plate_list[i_level+1].plot_vector(defect2)
            plate_list[i_level+1].plot_vector(defect, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_3coarse.svg")

        else: # if last level just do direct solve (v. small or coarsest level)
            x_update = spsolve(lhs, defect)

            plate_list[i_level].plot_vector(defect, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_1rhs.svg")
            plate_list[i_level].plot_vector(x_update, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_2solve.svg")

            old_defect = defect.copy()
            _defect = defect - lhs @ x_update
            defect_list += [_defect.copy()]

            plate_list[i_level].plot_vector(_defect, 
                        # filename=None)
                        filename=f"{folder}/{ct}_down{i_level}_3err.svg")
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
        lhs = mat_list[i_level]

        # interpolate the correction
        # print(f"{_coarse_update.shape=} {Icf.shape=}")
        cf_update = plate_list[i_level+1].to_fine(_coarse_update)
        defect = defect_list[i_level].copy()

        # compute line search of stiffness matrix scalar (for coarse-fine update) to get right magnitude
        # omega = 1.0 # no re-scale
        # # eqn: omega = |s^T R_0| / (s^T K s)
        s = cf_update
        # line search coeff here correct (but coarse update isn't quite right..)
        omega = np.dot(s, defect) / np.dot(lhs.dot(s), s)
        norm1 = np.linalg.norm(defect)
        start_defect = defect - lhs.dot(omega * s)
        norm2 = np.linalg.norm(start_defect)
        print(f"{omega=} {norm1=} {norm2=}")

        # somehow design updates are very strange with the GS smoothing of coupled w and rot DOF,
        # TODO : I think the update mode is nearly orthogonal to the (3,1) mode shape bc of weird GS effects?
        # TODO : also unclear if BCs are influencing the conv results.. (may want to remove bcs from each level..)

        # _defect_change = -lhs.dot(omega * s)
        # plate_list[i_level+1].plot_vector(_coarse_update, 
        #                 filename=None)
        # plate_list[i_level].plot_vector(cf_update, 
        #                 filename=None)
        # plate_list[i_level].plot_vector(defect, 
        #                 filename=None)
        # plate_list[i_level].plot_vector(_defect_change, 
        #                 filename=None)
        # exit()

        # add correction from this level to coarser update
        fine_update = x_update_list[i_level] + s * omega
        plate_list[i_level].plot_vector(cf_update, 
                        filename=f"{folder}/{ct}_up{i_level}_update.svg")

        # do a post-smoothing and update the update at this level
        _x0 = np.zeros(lhs.shape[0])
        # _update = gauss_seidel_csr(lhs, start_defect, _x0, num_iter=3)
        _update = block_gauss_seidel(lhs, start_defect, _x0, num_iter=3, ndof=3)

        x_update = _update + fine_update

        new_defect = start_defect - lhs @ _update
        c_nxe = nxe // 2**i_level # debug

        plate_list[i_level].plot_vector(start_defect, 
                        filename=f"{folder}/{ct}_up{i_level}_interp.svg")
        plate_list[i_level].plot_vector(new_defect, 
                        filename=f"{folder}/{ct}_up{i_level}_smooth.svg")
        plt.close('all')

        # new _coarse_update is then: to next level
        _coarse_update = x_update.copy()

    # check final error after V-cycle
    x += x_update
    defect = fine_rhs - fine_kmat.dot(x)
    res_norm = np.linalg.norm(defect)
    print(f"vcyc{i_vcycle}: {last_res_norm=:.3e} => {res_norm=:.3e}")
    plate_list[0].plot_vector(defect.copy(), 
                        filename=f"{folder}/_ovr.svg")
    plate_list[0].plot_vector(x, 
                        filename=f"{folder}/_design.svg")

    if res_norm < (init_res_norm * 1e-7 + 1e-8):
        print(f"\tvcycle multigrid converged in {i_vcycle+1} steps")

    last_ovr_defect = defect.copy()
    last_res_norm = res_norm

    plt.close('all')


# now check the solution against the true soln
print("final multigrid results:\n")
final_res = fine_rhs - fine_kmat.dot(x)
final_res_norm = np.linalg.norm(final_res)
print(f"{init_res_norm=:.3e} => {final_res_norm=:.3e}")
plate_list[0].plot_vec(final_res.copy(), 
                        filename=f"out/_ovr.svg")