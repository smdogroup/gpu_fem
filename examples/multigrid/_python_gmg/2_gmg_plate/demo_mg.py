from src._dkt_plate_elem import *
from src.linalg import *
import numpy as np
import matplotlib.pyplot as plt

# right triangle
x = np.array([0.0, 1.0, 0.0])
y = np.array([0.0, 0.0, 1.0])

# equilateral triangle
# x = np.array([0.0, 1.0, 0.5])
# y = np.array([0.0, 0.0, np.sqrt(3)/2.0])

eta = np.linspace(0.0, 1.0, 10)
xi = np.linspace(0.0, 1.0, 10) # technically xi goes from 0 to 1-eta on triangular boundary

XI, ETA = np.meshgrid(xi, eta)

# nodal DOF
U = np.zeros(9)
U[1::3] = np.array([0.1, 0.3, 0.5])
U[2::3] = np.array([-0.1, 0.2, 0.4])

# check Hx, Hy at some intermed disps
Hx, Hy = dkt_H_shape_funcs(x, y, 0.0, 1.0)
# print(f"{Hx=} {Hy=}")

BETAX = np.array([[get_rotations(U, x, y, xi[i], xi[j])[0] for i in range(10)] for j in range(10)])
BETAX[XI > (1.0 - ETA)] = np.nan # zero out past the triangular boundary (for plot)

# fig, ax = plt.subplots(1, 2)
# ax0 = fig.add_subplot(1, 2, 1, projection='3d')
# ax0.plot_surface(XI, ETA, BETAX)
# plt.show()

"""test the rot interp.. on small mesh"""
_nx, _ny = 4, 5
# _nx, _ny = 8, 9

x, y = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _ny)
X, Y = np.meshgrid(x, y)
print(f'{X.shape=}')
W = np.sin(np.pi * X) * np.sin(np.pi * Y)
W_x = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
W_y = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
TH_x = -W_y
TH_y = W_x
# exit()

# now interp the TH_x, TH_y
_hx = 1.0 / (_nx - 1)
_hy = 1.0 / (_ny - 1)
x_elem = _hx * np.array([0.0, 1.0, 0.0])
y_elem = _hy * np.array([0.0, 0.0, 1.0])
# xi, eta = 0.5, 0.0
# indexed by [iy, ix]
elem_vals = []

# ix_offset, iy_offset = 1, 1
# ix_offset, iy_offset = 0, 1
ix_offset, iy_offset = 1, 0

for ix_iy in [0, 1, 2]:
    _ix = ix_iy % 2
    _iy = ix_iy // 2
    
    ix, iy = _ix + ix_offset, _iy + iy_offset
    elem_vals += [W[iy, ix], TH_x[iy, ix], TH_y[iy, ix]]

xi_vec = np.linspace(0.0, 1.0, 10)
TH_X_hat = np.array([[ -get_rotations(elem_vals, x_elem, y_elem, xi_vec[i], xi_vec[j])[1] for i in range(10)] for j in range(10)])
TH_X_hat[XI > (1.0 - ETA)] = np.nan # zero out past the triangular boundary (for plot)

# fig, ax = plt.subplots(1, 1)
# ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot_surface(X, Y, TH_x, alpha=0.8, zorder=0)
# X_elem = _hx * (ix_offset + XI)
# Y_elem = _hy * (iy_offset + ETA)
# ax0.plot_surface(X_elem, Y_elem, TH_X_hat, zorder=1)
# plt.show()

# exit()

"""test derivatives of Hx, Hy w.r.t. xi and eta using complex step"""
xi, eta = 0.461, 0.321
# xi, eta = 0.321, 0.461
Hx_xi, Hy_xi, Hx_eta, Hy_eta = dkt_H_shape_func_grads(x_elem, y_elem, xi, eta)

_Hx_xi2, _Hy_xi2 = dkt_H_shape_funcs(x_elem, y_elem, xi + 1e-30 * 1j, eta)
Hx_xi2, Hy_xi2 = np.imag(_Hx_xi2) / 1e-30, np.imag(_Hy_xi2) / 1e-30
_Hx_eta2, _Hy_eta2 = dkt_H_shape_funcs(x_elem, y_elem, xi, eta + 1e-30 * 1j)
Hx_eta2, Hy_eta2 = np.imag(_Hx_eta2) / 1e-30, np.imag(_Hy_eta2) / 1e-30

err_mat = np.zeros((2, 18))
err_mat[0,:9] = Hx_xi - Hx_xi2
err_mat[0,9:] = Hy_xi - Hy_xi2
err_mat[1,:9] = Hx_eta - Hx_eta2
err_mat[1,9:] = Hy_eta - Hy_eta2
err2 = np.linalg.norm(err_mat)
# print(F"{err2=:.3e}")
# plt.imshow(err_mat)
# plt.show()
# matches now..

"""compute and solve the fine problem (ref soln)"""

# compute the element stiffness matrix for an aluminum material
E, nu, thick = 2e7, 0.3, 1e-2 # 10 cm thick
# quad_wt = 1.0
# Kelem = get_kelem(E, thick, nu, x, y)

# plt.imshow(Kelem)
# plt.show()

# let's now try and assemble the global stiffness matrix (dense form)
# nxe = 64
nxe = 32
# nxe = 8 # fine grid
# nxe = 4
# nxe = 4

K = assemble_stiffness_matrix(nxe, E, thick, nu)
ndof = K.shape[0]

# now apply some loads to the structure
F = np.zeros(ndof)
F[0::3] = 100.0 # vert load for bending..

K, F = apply_bcs(nxe, K, F)
nx = nxe + 1

u = np.linalg.solve(K, F)

# now plot the solution 
x_plt, y_plt = np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x_plt, y_plt)
w = u[0::3]
# w =  u[2::3]
W = np.reshape(w, (nx, nx))

# fig, ax = plt.subplots(1, 1)
# ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot_surface(X, Y, W)
# plt.show()

"""solve on coarser mesh now"""
nxe_c = nxe // 2
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
ndof_c = K_c.shape[0]
F_c = np.zeros(ndof_c)
F_c[0::3] = 100.0
K_c, F_c = apply_bcs(nxe_c, K_c, F_c)
u_c = np.linalg.solve(K_c, F_c)

# try prolongation now
P = prolongation_operator(nxe_c = nxe_c)
# plt.imshow(P)
# plt.show()

u_f_hat = P @ u_c
nx_c = nxe_c + 1

fine_bcs = get_bcs(nxe)
# u_f_hat[fine_bcs] = 0.0
# print(f'{u_f_hat[fine_bcs]=}')

# now measure also the nodal forces after interpolation
F_f_hat = np.dot(K, u_f_hat)
F_f_hat[fine_bcs] = 0.0 # can just zero this out (we don't smooth this..)

"""define a convenient plot method"""
# plot results of random defect schwarz smoothing
def plot_defect_change(_nx, _defect, __du, _defect_2, idof:int=0, all:bool=False):
    """plot the defect change"""

    x_plt, y_plt = np.linspace(0.0, 1.0, _nx), np.linspace(0.0, 1.0, _nx)
    X, Y = np.meshgrid(x_plt, y_plt)
    Load = np.reshape(_defect[idof::3], (_nx, _nx))
    Disp_hat = np.reshape(__du[idof::3], (_nx, _nx))
    Load_hat = np.reshape(_defect_2[idof::3], (_nx, _nx))

    if all:
        idof_list = [0,1,2]
    else:
        idof_list = [idof]

    for idof in idof_list:
        fig, ax = plt.subplots(1, 3)
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(X, Y, Load)
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(X, Y, Disp_hat)
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(X, Y, Load_hat)

        dof_str = ['w', 'thx', 'thy'][idof]
        ax0.set_title(f"Defect {dof_str}")
        ax1.set_title(f"Update {dof_str}")
        ax2.set_title(f"Defect v2")
        plt.show()


"""view results of coarse solve then prolongation"""

# plot_defect_change(nx, u, u_f_hat, F_f_hat)

"""try subtracting this from a defect and see if the coarse-fine operator is reasonable"""

defect = F.copy()
s = u_f_hat.copy()
# omega = np.dot(defect, s) / np.dot(s, K @ s) # chooses omega = 4.0 (should be 2.0)
# omega = 1.0
omega = 2.0
print(f"{omega=}")

defect_2 = defect - K @ (omega * s)
disp_update = omega * s

# NOTE : careful, right now the huge non-centered jumps in nodal forces at the boundary are dominating the omega quantity
# try setting omega yourself

defect_nrms = [np.linalg.norm(defect), np.linalg.norm(defect_2)]
print(f"{defect_nrms=}")

# avgs # for coarse error estimate
# see if we removed the mean err
defect_means = [np.mean(defect[0::3]), np.mean(defect_2[0::3])]
print(f"{defect_means=}")

# plot_defect_change(u, disp_update, defect_2)

"""try incomplete LU smoothing"""
# K_csr = sp.csr_matrix(K)
# precond_0 = ILU0Preconditioner(K_csr, dof_per_node=3)
# defect = np.random.rand(ndof)
# defect[fine_bcs] = 0.0
# defect[1::3], defect[2::3] = 0.0, 0.0
# du = pcg(K, defect, precond_0, x0=np.zeros(ndof), maxiter=10)
# defect_2 = defect - K @ du
# plot_defect_change(defect, du, defect_2)

# NOTE : ilu smoother gets worse in the middle ofthe plate?
# is it some node ordering issues? I did random ordering but didn't seem to help enough with CSR

"""try additive schwarz smoothing"""
# additive = True
additive = False # multiplicative
# can't choose as many subdomains here cause it's so small a mesh?
as_smoother = SchwarzSmoother(K, nx_sd=4, overlap_frac=0.1, additive=additive)
as_smoother.set_material(E, nu, thick)
for i in range(as_smoother.nx_sd):
    ix_no = as_smoother._get_1d_domain(ix_sd=i)
    print(f"{nxe=} {i=} {ix_no=}")
#nodes_list = as_smoother.sd_nodes_list
#print(f"{nodes_list=}")
# exit()

# plot each subdomain (debug to make sure subdomains properly overlap and everything)
# M_sd = np.zeros((nx, nx), dtype=np.int32)
# for i_sd, sd_list in enumerate(nodes_list):
#     # M_sd = np.zeros((nx, nx), dtype=np.int32) # reset it
#     for node in sd_list:
#         iy = node // nx
#         ix = node % nx
#         M_sd[ix, iy] += (i_sd+1)
#     bndry_list = as_smoother._get_boundary_nodes_on_sd(np.array(sd_list))
#     for node in bndry_list:
#         ix, iy = node % nx, node // nx
#         M_sd[ix, iy] += (i_sd+1)*2 # so triple counts here on bndry
# plt.imshow(M_sd)
# plt.show()

# try defect smoothing now..
# NOTE : Schwarz subdomain solves work much better with less rot defects, when only w=w0 on boundary is enforced
defect = np.random.rand(ndof)
defect[fine_bcs] = 0.0
defect[1::3], defect[2::3] = 0.0, 0.0
omega_as = 0.5 # damping factor for additive schwarz
# omega_as = 1.0

du, defect_2 = as_smoother.multi_smooth(defect, np.zeros(ndof), num_smooth=1, omega=omega_as)
defect_2_check = np.dot(K, du)

# exit()

plot_defect_change(nx, defect, du, defect_2)
plot_defect_change(nx, defect_2, du, defect_2_check)
exit()

"""TODO : now try solving the multigrid with two-levels.. and different smoothers"""

K_f = assemble_stiffness_matrix(nxe, E, thick, nu)
K_c = assemble_stiffness_matrix(nxe_c, E, thick, nu)
K_f, _ = apply_bcs(nxe, K_f, F)
K_c, _ = apply_bcs(nxe_c, K_c, np.zeros(K_c.shape[0]))

P = prolongation_operator(nxe_c)
R = P.T # restriction operator

additive = True
# additive = False
ss_f = SchwarzSmoother(K_f, nx_sd=4, overlap_frac=0.1, additive=additive)
ss_f.set_material(E, nu, thick)

# multiplicative schwarz smoother for smoothing boundary
ssm_f = SchwarzSmoother(K_f, nx_sd=4, overlap_frac=0.1, additive=False)
ssm_f.set_material(E, nu, thick)

defect_0 = F.copy()
u_f_0 = np.zeros(ndof)
defect, u_f = defect_0.copy(), u_f_0.copy()

# number of pre and post-smoothing steps to do
n_pres, n_posts = 1, 1
# n_pres, n_posts = 3, 3
# omega_s = 0.7 # damping of additive schwarz update
omega_s = 0.5
# omega_s = 0.1

omega_s = 1.0 if not(ss_f.additive) else omega_s
print(f'{omega_s=}')

init_norm = np.linalg.norm(defect)
print(f"{init_norm=:.4e}")

for v_cycle in range(10):
    """do v-cycle loops here"""

    # step 1 - pre-smoothing at fine level
    # TODO : try using disp guess here, vs. solving update only
    _du, defect_2 = ss_f.multi_smooth(defect, np.zeros(ndof), num_smooth=n_pres, omega=omega_s)
    u_f_2 = u_f + _du # TODOP plus or minus update here?
    # plot_defect_change(nx, defect, u_f_2, defect_2)

    # step 2 - restrict defect to coarse
    defect_c_1 = np.dot(R, defect_2)
    # plot_defect_change(nx_c, defect_c_1, defect_c_1, defect_c_1)
    coarse_bcs = get_bcs(nxe_c)

    # step 3 - coarse solve
    du_c_1 = np.linalg.solve(K_c, defect_c_1)
    # plot_defect_change(nx_c, defect_c_1, du_c_1, defect_c_1)

    # step 4 - coarse-fine or prolongation
    du_f_1 = np.dot(P, du_c_1)
    _dF_f_1 = np.dot(K, du_f_1)
    # plot_defect_change(nx, defect_2, du_f_1, _dF_f_1)

    # try additive schwarz smoothing on the coarse-fine update itself
    # NOTE : we're using ssm_f the multiplicative smoother here..
    _du_f_2, _dF_f_2_sugg = ssm_f.multi_smooth(_dF_f_1, np.zeros(ndof), num_smooth=10, omega=omega_s)
    du_f_2 = du_f_1 -_du_f_2  
    # du_f_2 = du_f_1 + _du_f_2 # TODO : plus or minus here?
    _dF_f_2 = np.dot(K_f, du_f_2)
    plot_defect_change(nx, _dF_f_1, du_f_2, _dF_f_2)

    # compare disps of the smoothing.. (NOTE : they do have about the same magnitude)
    # plot_defect_change(nx, _dF_f_1, du_f_1, du_f_2)

    # step 5 - apply disp update using line search
    # TODO : do two-parameter updates here (two-dof line search, see paper on unstructured multigrid methods for shells)
    # s = du_f_1
    s = du_f_2
    omega = np.dot(s, defect_2) / np.dot(s, np.dot(K_f, s))
    print(f"{omega=}")
    defect_3 = defect_2 - np.dot(K_f, omega * s)
    u_f_3 = u_f_2 + omega * s
    plot_defect_change(nx, defect_2, u_f_3, defect_3)

    # step 6 - schwarz post smoothing
    # TODO : try here solving the update disp starting with zero vs putting disps in (causes state drift?)
    # u_f_4, defect_4 = ss_f.multi_smooth(defect_3, u_f_3, num_smooth=n_posts, omega=omega_s)
    _du_f_4, defect_4 = ss_f.multi_smooth(defect_3, np.zeros(ndof), num_smooth=n_posts, omega=omega_s)
    u_f_4 = _du_f_4 + u_f_3
    plot_defect_change(nx, defect_3, u_f_4, defect_4)

    # step 7 - reset for next step
    defect, u_f = defect_4, u_f_4

    c_norm = np.linalg.norm(defect)
    print(f"v-cycle step {v_cycle}, {c_norm=:.4e}")
