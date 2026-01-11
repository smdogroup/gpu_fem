"""geometric multigrid demo with ANS shell"""

# here I construct the coarse grid stiffness matrix by galerkin projection Kc = P^T K P
# unstructured multigrid for shells paper says that the galerkin projection is slower than recompute Kc from grid and coarse grid FEA basis
# but I beg to differ they seemed comparable, and I want to investigate myself 

import numpy as np
import matplotlib.pyplot as plt
from _src import *
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

"""argparse inputs"""
parser = argparse.ArgumentParser()
parser.add_argument("--SR", type=float, default=100.0, help="slenderness ratio of plate L/h")
parser.add_argument("--nxe", type=int, default=8, help="nxe # elements in x-dir")
parser.add_argument("--cf_smooth", type=int, default=0, help="coarse-fine smooth boolean")
parser.add_argument("--plot", type=int, default=0, help="can plot intermediate results")
args = parser.parse_args()

SR = int(args.SR) # assume it is an int..
thick = 1.0 / args.SR
E, nu = 70e9, 0.3 # aluminum

"""get the coarse-fine operator"""
nxe_fine = args.nxe
# only don't apply bcs to P for the galerkin method..
P_0, R_0 = gmg_plate_coarse_fine_matrices(nxe_fine, apply_bcs=False)

"""make fine mesh problem"""
nx_fine = 2 * nxe_fine + 1
nnodes_fine = nx_fine**2
ndof_fine = 5 * nnodes_fine
x_fine, y_fine = np.linspace(0.0, 1.0, nx_fine), np.linspace(0.0, 1.0, nx_fine)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

_K = get_plate_K_global(nxe_fine, E, nu, thick)
_F = get_global_plate_loads(nxe_fine, load_type="sine-sine", 
                            magnitude=1e9 * thick**3)
K_fine_0, F_fine_0 = apply_bcs(nxe_fine, _K, _F) # this version just zeros out in place

# elim extra gam DOF to remove spurious energy modes (will prob have to add extra basis funcs like rots but w grads)
# later to make second derivs of w full rank (so can have matching DOF per node)

# get constraints red subspace and apply to elim the extra gam DOF (solves in red subspace)
Z = order1_gam_subspace(args.nxe)
K_fine = Z.T @ K_fine_0 @ Z
F_fine = np.dot(Z.T, F_fine_0)

true_soln_r = np.linalg.solve(K_fine, F_fine)
true_soln = np.dot(Z, true_soln_r)

"""make coarse mesh problem"""

nxe_coarse = args.nxe // 2
nx_coarse = 2 * nxe_coarse + 1
nnodes_coarse = nx_coarse**2
ndof_coarse = 5 * nnodes_coarse
x_coarse, y_coarse = np.linspace(0.0, 1.0, nx_coarse), np.linspace(0.0, 1.0, nx_coarse)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse)

# _K = get_plate_K_global(nxe_coarse, E, nu, thick)
# _F = get_global_plate_loads(nxe_coarse, load_type="sine-sine", magnitude=1.0)
# K_coarse, F_coarse = apply_bcs(nxe_coarse, _K, _F) # this version just zeros out in place

# need to modify subspace P such that it is in Z
# namely, need gam_xz, gam_yz linear in each element?
Z_c = order1_gam_subspace(args.nxe // 2)
P_cr_0 = P_0 @ Z_c

# make P subspace in Z with projector operation
Pr = Z.T @ P_cr_0 # coarse
# Rr = P_0.T @ Z # same thing..
P = Z @ Z.T @ P_cr_0 # coarse to fine

# here use P galerkin coarse grid stiffness matrix
K_coarse = Pr.T @ K_fine @ Pr
# assumes P subspace is contained in Z? (or at least pretty close..)

"""convert coarse and fine matrices to csr"""
K_f_csr, K_c_csr = sp.csr_matrix(K_fine), sp.csr_matrix(K_coarse)

# soln_2 = spsolve(K_f_csr, F_fine)
# soln_diff = true_soln - soln_2 
# soln_diff_nrm = np.linalg.norm(soln_diff)
# print(f"{soln_diff_nrm=:.3e}")

"""quick utils for v-cycle process"""
def check_defect(_c_defect, _c_soln):
    c_defect = _c_defect.copy()
    c_soln = _c_soln.copy()
    defect_check = F_fine - K_f_csr.dot(c_soln)
    diff = c_defect - defect_check
    diff_nrm = np.linalg.norm(diff)
    print(f"{diff_nrm=:.3e}")
    assert(diff_nrm < 1e-6)

def plot_compare(_vec1, name1:str, _vec2, name2:str):
    """compare two different vectors"""

    vec1 = _vec1.copy()
    vec2 = _vec2.copy()
    if args.plot:
        print(f"plot compare: {name1} to {name2}")

        w1, w2 = vec1[2::5], vec2[2::5]
        is_f1, is_f2 = w1.shape[0] == nnodes_fine, w2.shape[0] == nnodes_fine
        nx1, nx2 = nx_fine if is_f1 else nx_coarse, nx_fine if is_f2 else nx_coarse
        W1, W2 = w1.reshape((nx1, nx1)), w2.reshape((nx2, nx2))
        THX1, THX2 = vec1[3::5].reshape((nx1,nx1)), vec2[3::5].reshape((nx2, nx2))
        THY1, THY2 = vec1[4::5].reshape((nx1,nx1)), vec2[4::5].reshape((nx2, nx2))

        X1, Y1 = X_fine if is_f1 else X_coarse, Y_fine if is_f1 else Y_coarse
        X2, Y2 = X_fine if is_f2 else X_coarse, Y_fine if is_f2 else Y_coarse

        # plot the solved disps now..
        disps_str = ['w', 'thx' ,'thy']
        fig, ax = plt.subplots(2,3, figsize=(12, 8))
        for ind, VALS in enumerate([W1, THX1, THY1, W2, THX2, THY2]):
            cax = fig.add_subplot(2, 3, ind+1, projection='3d')
            if ind // 3 == 0:
                cax.plot_surface(X1, Y1, VALS, cmap='jet' if ind % 3 == 0 else 'viridis')
            else: # == 0
                cax.plot_surface(X2, Y2, VALS, cmap='jet' if ind % 3 == 0 else 'viridis')
            
            cax.set_title(name1 if ind // 3 == 0 else name2 + disps_str[ind % 3])
        # plt.show()
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"_out/0_{SR=}_disp.svg")

def plot_trv_shear(c_nxe, disps, ixe, iye, name:str):
    """plot gam13 and gam23 in a single element of a solution or soln update"""

    if args.plot:
        print(f"plot trv shear: {name}")

        c_nx = 2 * c_nxe + 1
        
        node = 2 * c_nx * iye + 2 * ixe # starting node of elem
        elem_nodes = [node+c_nx*iy+ix for iy in range(3) for ix in range(3)]
        elem_dof = np.array([5 * _node + _dof for _node in elem_nodes for _dof in range(5)])
        elem_disps = disps[elem_dof]

        # include integration points in the transv shear here
        rt_35 = np.sqrt(3.0 / 5.0)
        _xi = np.array([-1.0, -0.9, -rt_35, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, rt_35, 0.9, 1.0])
        _eta = _xi.copy()
        n_xi, n_eta = _xi.shape[0], _xi.shape[0]
        XI, ETA = np.meshgrid(_xi, _eta)

        # same elem xpts for everything in this elem
        elem_xpts = np.zeros(27)
        h = 1.0 / (c_nx - 1)
        for inode in range(3):
            x = h * inode
            for jnode in range(3):
                y = h * jnode
                z = 0.0
                node = 3 * jnode + inode
                elem_xpts[3*node:(3*node+3)] = np.array([x, y, z])[:]

        shell_xi_axis = np.array([1.0, 0.0, 0.0])
        GAM_13, GAM_23 = np.zeros_like(XI), np.zeros_like(ETA)
        for i in range(n_xi):
            for j in range(n_xi):
                xi, eta = XI[i,j], ETA[i,j]
                _strains = get_quadpt_strains(shell_xi_axis, xi, eta, elem_xpts, elem_disps)
                GAM_13[i,j], GAM_23[i,j] = _strains[6], _strains[7]

        # plot the transv shear strains now (first in 3d)
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))

        gam_strs = ['gam13', 'gam23']
        for i, VALS in enumerate([GAM_13, GAM_23]):

            cax = fig.add_subplot(1, 2, i+1, projection='3d')
            LOG_VALS = np.log10(np.abs(VALS))
            cax.plot_surface(XI, ETA, LOG_VALS, cmap='viridis')
            cax.set_title(f"{name}: log10 {gam_strs[i]}-elem", fontweight='bold')
            cax.set_xlabel("XI", fontweight='bold')
            cax.set_ylabel("ETA", fontweight='bold')

        plt.tight_layout()
        # plt.savefig(f"_out/1_{SR=}_gam_strains.svg")
        plt.show()

"""do V-cycle process now"""

defect_0 = F_fine.copy()
soln = np.zeros_like(true_soln_r)
init_nrm = np.linalg.norm(defect_0)
defect = defect_0.copy()

n_pre, n_post = 2, 2

for i_vcycle in range(100):

    # copy previous defect and soln
    defect_f_0 = defect.copy()
    soln_0 = soln.copy()

    # 1 - fine pre-smooth
    # ndof_gs = 1
    ndof_gs = 8 # that's how many red DOF per node there are..
    # ndof_gs = 16
    # ndof_gs = 32

    du_f_1 = block_gauss_seidel(K_f_csr, defect_f_0.copy(), x0=np.zeros_like(soln), num_iter=n_pre, ndof=ndof_gs)
    # du_f_1 = damped_jacobi(K_fine, defect_f_0, n_iter=5, x0=None, omega=0.1)
    soln_1 = soln_0 + du_f_1
    defect_f_1 = defect_f_0 - K_f_csr.dot(du_f_1)

    # it's prob lost the nodal ordering now... bruh
    plot_compare(np.dot(Z, defect_f_0), "defect0", np.dot(Z, defect_f_1), "defect1")
    # check_defect(defect_f_1, soln_1)

    # 2 - fine to coarse (or restrict)
    # need PT not R for galerkin method (R is row-sum normalized slightly differently)
    defect_c_0 = np.dot(Pr.T, defect_f_1)

    # check_defect(defect_f_1, soln_1)
    plot_compare(np.dot(Z, defect_f_1), "fine-defect", np.dot(P, defect_c_0), "coarse-defect")

    # 3 - solve on coarse level
    du_c_1 = spsolve(K_c_csr, defect_c_0)

    # 4 - coarse to fine (prolongate)
    du_f_2 = np.dot(Pr, du_c_1)

    # coarse-fine smooth
    if args.cf_smooth:
        df_f_2 = K_f_csr.dot(du_f_2)
        du_cf_smooth = block_gauss_seidel(K_f_csr, -df_f_2.copy(), x0=np.zeros_like(soln), num_iter=n_pre, ndof=ndof_gs)
        # du_cf_smooth = damped_jacobi(K_fine, -df_f_2, n_iter=5, x0=None, omega=0.1)
        du_f_2 += du_cf_smooth

    # check the trv shear of coarse-fine
    # plot_trv_shear(nxe_coarse, du_c_1, ixe=2, iye=2, name="coarse-soln")
    # plot_trv_shear(nxe_fine, du_f_2, ixe=4, iye=4, name="fine-soln")

    # also do the line search rescale here..
    s = du_f_2.copy()
    omega = np.dot(defect_f_1, s) / np.dot(s, K_f_csr.dot(s))

    # TEMP debug
    # omega = 1.0
    print(f"{omega=:.3e}")

    soln_2 = soln_1 + omega * s
    defect_f_2 = defect_f_1 - K_f_csr.dot(omega * s)

    # plot coarse to fine disps
    plot_compare(np.dot(P, du_c_1), "coarse-disps", np.dot(Z, du_f_2), "fine-disps")

    plot_compare(np.dot(Z, defect_f_1), "defect-pre-cf", np.dot(Z, defect_f_2 - defect_f_1), "defect-change-cf")

    # plot defect before and after coarse-fine
    plot_compare(np.dot(Z, defect_f_1), "defect-pre-cf", np.dot(Z, defect_f_2), "defect-post-cf")

    # 5 - fine post-
    du_f_3 = block_gauss_seidel(K_f_csr, defect_f_2.copy(), x0=np.zeros_like(soln), num_iter=n_post, ndof=ndof_gs)
    # du_f_3 = block_gauss_seidel(K_f_csr, defect_f_2.copy(), x0=np.zeros(ndof_fine), num_iter=1, ndof=1)
    # du_f_3 = damped_jacobi(K_fine, defect_f_2, n_iter=5, x0=None, omega=0.1)
    soln_3 = soln_2 + du_f_3    
    defect_f_3 = defect_f_2 - np.dot(K_fine, du_f_3)

    plot_compare(np.dot(Z, defect_f_2), "defect_f_2", np.dot(Z, defect_f_3), "defect_f_3")

    # 6.0 - update final soln of this v-cycle step for going to next cycle or exit
    soln = soln_3.copy()
    defect = defect_f_3.copy()

    # double check defect correct here
    # check_defect(defect, soln)
    
    # 6.1 - check progress and defect norms
    defect_nrm = np.linalg.norm(defect)
    print(f"{i_vcycle=} : {defect_nrm=:.3e}")

    if defect_nrm < 1e-6 * init_nrm:
        break

n_vcycles = i_vcycle + 1
R = defect_nrm / init_nrm
print(F"converged {R} in {n_vcycles=}")
    