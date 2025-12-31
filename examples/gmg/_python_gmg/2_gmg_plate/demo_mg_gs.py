from src._dkt_plate_elem import *
from src.linalg import *
import numpy as np
import matplotlib.pyplot as plt

"""demo multigrid with block gauss-seidel"""

"""compute and solve the fine problem (ref soln)"""

# compute the element stiffness matrix for an aluminum material
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick

# let's now try and assemble the global stiffness matrix (dense form)
# nxe = 64
nxe = 32

K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = K.shape[0]

# get x coords at each node
nx = nxe + 1
h = 1.0 / nxe
xpts, ypts = [], []
for iy in range(nx):
    for ix in range(nx):
        xpts += [h * ix]
        ypts += [h * iy]
xpts, ypts = np.array(xpts), np.array(ypts)

# now apply some loads to the structure
F = np.zeros(ndof)

# load_case = 'sine-sine'
load_case = 'polar'

assert(load_case in ['const', 'sine-sine', 'polar'])

if load_case == 'const':
    F[0::3] = 100.0 # vert load for bending..
elif load_case == 'sine-sine':
    # m, n = 1, 1
    m, n = 4, 3
    F[0::3] = 10.0 * np.sin(m * np.pi * xpts) * np.sin(n * np.pi * ypts)
elif load_case == 'polar':
    def load_fcn(x,y):
        import math
        theta = math.atan2(y, x)
        r = np.sqrt(x**2 + y**2)
        return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta)
    # game of life polar load..
    F[0::3] = np.array([load_fcn(xpts[i], ypts[i]) for i in range(xpts.shape[0])])

K, F = apply_bcs(nxe, K, F)


# true soln
u_truth = np.linalg.solve(K, F)

"""solve on coarser mesh now"""
nxe_c = nxe // 2
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
ndof_c = K_c.shape[0]
# F_c = np.zeros(ndof_c)
# F_c[0::3] = 100.0
# K_c, F_c = apply_bcs(nxe_c, K_c, F_c)
# u_c = np.linalg.solve(K_c, F_c)

# try prolongation now
P = prolongation_operator(nxe_c = nxe_c)

"""define a convenient plot method"""
# plot results of random defect schwarz smoothing
def plot_defect_change(_nx, _defect, __du, _defect_2, idof:int=0, all:bool=False, is_defect:bool=True):
    """plot the defect change"""

    x_plt, y_plt = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _nx)
    X, Y = np.meshgrid(x_plt, y_plt)

    if all:
        idof_list = [0,1,2]
    else:
        idof_list = [idof]

    for _idof in idof_list:

        Load = np.reshape(_defect[_idof::3], (_nx, _nx))
        Disp_hat = np.reshape(__du[_idof::3], (_nx, _nx))
        Load_hat = np.reshape(_defect_2[_idof::3], (_nx, _nx))

        fig, ax = plt.subplots(1, 3)
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(X, Y, Load, cmap='jet')
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(X, Y, Disp_hat, cmap='jet')
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(X, Y, Load_hat, cmap='jet')

        dof_str = ['w', 'thx', 'thy'][_idof]
        if is_defect:
            ax0.set_title(f"Defect {dof_str}")
            ax1.set_title(f"Update {dof_str}")
            ax2.set_title(f"Defect v2")
        else: # it's disps comparison
            ax0.set_title(f"Defect {dof_str}")
            ax1.set_title(f"True disp {dof_str}")
            ax2.set_title(f"Pred disp {dof_str}")
        plt.tight_layout()
        plt.show()

def plot_soln(_nx, _u):
    """plot all 3 DOF of the solution.."""

    x_plt, y_plt = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _nx)
    X, Y = np.meshgrid(x_plt, y_plt)

    fig, ax = plt.subplots(1, 3)
    dof_str = ['w', 'thx', 'thy']
    for _idof in [0,1,2]:
        Disp_hat = np.reshape(_u[_idof::3], (_nx, _nx))
        ax0 = fig.add_subplot(1, 3, _idof+1, projection='3d')
        ax0.plot_surface(X, Y, Disp_hat, cmap='jet')
        ax0.set_title(dof_str[_idof])
    plt.tight_layout()
    plt.show()

"""TODO : now try solving the multigrid with two-levels.. and different smoothers"""

K_f = assemble_stiffness_matrix(nxe, E, thick, nu)
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
K_f, _ = apply_bcs(nxe, K_f, F)
K_c, _ = apply_bcs(nxe_c, K_c, np.zeros(K_c.shape[0]))

K_f_csr = sp.csr_matrix(K_f)

P = prolongation_operator(nxe_c)
R = P.T # restriction operator

defect_0 = F.copy()
u_f_0 = np.zeros(ndof)
defect, u_f = defect_0.copy(), u_f_0.copy()

print(f"{defect_0=}")

# number of pre and post-smoothing steps to do
n_pres, n_cfs, n_posts = 1, 1, 1
# n_pres, n_cfs, n_posts = 3, 3, 3

init_norm = np.linalg.norm(defect)
print(f"{init_norm=:.4e}")

def check_defect_err(_u_f, _defect):
    # check error here..
    full_defect = defect_0 - K_f_csr.dot(_u_f)
    full_defect_err = np.linalg.norm(full_defect - _defect)
    print(f"\t{full_defect_err=:.3e}")
    return

# can_plot = True
can_plot = False

# NOTE : just for showing old results am I turning this off, you need this to get good conv..
# prolong_smooth = True
prolong_smooth = False

defect_nrms = [init_norm]

for v_cycle in range(20):
    """do v-cycle loops here"""

    # step 1 - pre-smoothing at fine level
    _du = block_gauss_seidel(K_f_csr, defect, x0=np.zeros(ndof), num_iter=n_pres)
    defect_2 = defect - K_f_csr.dot(_du)
    u_f_2 = u_f + _du # TODOP plus or minus update here?
    if can_plot:
        print(f"{v_cycle=} : 1 - pre-smooth")
        plot_defect_change(nx, defect, u_f_2, defect_2)

    # if can_plot: check_defect_err(u_f_2, defect_2)

    # step 2 - restrict defect to coarse
    defect_c_1 = np.dot(R, defect_2)
    # plot_defect_change(nx_c, defect_c_1, defect_c_1, defect_c_1)
    coarse_bcs = get_bcs(nxe_c)

    # step 3 - coarse solve
    du_c_1 = np.linalg.solve(K_c, defect_c_1)
    # plot_defect_change(nx_c, defect_c_1, du_c_1, defect_c_1)

    # step 4 - coarse-fine or prolongation
    du_f_1 = np.dot(P, du_c_1)
    _dF_f_1 = K_f_csr.dot(du_f_1)
    # plot_defect_change(nx, defect_2, du_f_1, _dF_f_1)

    # gauss-seidel smoothing on coarse-fine update itself (negative loads here since that is what we will be applying to update)
    if prolong_smooth:
        _du_cf = block_gauss_seidel(K_f_csr, -_dF_f_1, x0=np.zeros(ndof), num_iter=n_cfs)
        du_f_2 = du_f_1 + _du_cf
        _dF_f_2 = K_f_csr.dot(du_f_2)
    else:
        du_f_2 = du_f_1
        _dF_f_2 = _dF_f_1

    if can_plot:
        print(f"{v_cycle=} : 2 - prolong smooth")
        plot_defect_change(nx, _dF_f_1, du_f_2, _dF_f_2)

    # exit()

    # compare disps of the smoothing.. (NOTE : they do have about the same magnitude)
    # plot_defect_change(nx, _dF_f_2, du_f_1, du_f_2)

    # step 5 - apply disp update using line search
    # TODO : do two-parameter updates here (two-dof line search, see paper on unstructured multigrid methods for shells)
    # s = du_f_1
    s = du_f_2
    omega = np.dot(s, defect_2) / np.dot(s, K_f_csr.dot(s))
    # defect_3 = defect_2 - K_f_csr.dot(omega * s)
    defect_3 = defect_2 - omega * _dF_f_2
    u_f_3 = u_f_2 + omega * s
    if can_plot:
        print(f"{v_cycle=} : 3 - prolong update, {omega=}")
        plot_defect_change(nx, defect_2, u_f_3, defect_3)
    
    # if can_plot: check_defect_err(u_f_3, defect_3)

    # step 6 - post-smoothing with Gauss-seidel
    _du_2 = block_gauss_seidel(K_f_csr, defect_3, x0=np.zeros(ndof), num_iter=n_posts)
    defect_4 = defect_3 - K_f_csr.dot(_du_2)
    u_f_4 = u_f_3 + _du_2 # TODOP plus or minus update here?
    if can_plot:
        print(f"{v_cycle=} : 4 - postsmooth")
        plot_defect_change(nx, defect_3, u_f_4, defect_4)

    # if can_plot: check_defect_err(u_f_4, defect_4)

    # step 7 - reset for next step
    defect, u_f = defect_4, u_f_4

    c_norm = np.linalg.norm(defect)
    defect_nrms += [c_norm]
    print(f"v-cycle step {v_cycle}, {c_norm=:.4e}")
    if c_norm < 1e-6:
        print(f"v-cycle converged")
        break

# final plot for the solution
u_f_pred = u_f.copy()
final_res = defect_0 - K_f_csr.dot(u_f_pred)
final_res_norm = np.linalg.norm(final_res)
print(f"{final_res_norm=:.3e}")

# plot_defect_change(nx, defect_0, u_f_pred, final_res, idof=0)
# plot_defect_change(nx, defect_0, u_truth, u_f_pred, is_defect=False, all=True)
plot_soln(nx, u_f_pred)

"""plot the convergence history of the V-cycle"""
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

# v-cycle resids here
iters = [_ for _ in range(len(defect_nrms))]
plt.plot(iters, defect_nrms, 'ko-')
plt.xlabel("V-cycles")
plt.ylabel("Defect residual")
plt.yscale("log")
plt.tight_layout()
plt.show()