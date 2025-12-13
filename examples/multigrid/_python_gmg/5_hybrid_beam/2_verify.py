# now let's test this out and visualize it
import numpy as np
from src import EBAssembler, plot_hermite_cubic
from src import TimoshenkoAssembler
from src import HybridAssembler, HRBeamAssembler

# verify the hybrid assembler solution against exact Timoshenko beam theory solution for constant distributed load

exact_disps = []
pred_disps = []
SRs = []
conds = []

for SR in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
# for SR in [10000.0]:

    # setup
    # -------------------------------

    E = 2e7; b = 1.0; L = 1.0; rho = 1
    # qmag = 2e-2
    qmag = 1e4
    ys = 4e5

    # L *= 100.0 # see if this helps numerics?

    b *= 0.5

    # scale by slenderness
    thick = L / SR
    qmag *= thick**3

    # scaling inputs
    # if rho_KS was too high like 500, then the constraint was too nonlinear or close to max and optimization failed
    rho_KS = 50.0 # rho = 50 for 100 elements, used 500 later
    # nxe = num_elements = int(3e2) #100, 300, 1e3
    # nxe = num_elements = int(2e3)
    # nxe = 3
    # nxe = 5
    # nxe = 5

    # nxe = 2
    # nxe = 4
    # nxe = 10
    nxe = 100

    # nxe = num_elements = 100
    # nxe = num_elements = int(1e3)
    # nxe = num_elements = int(4e3)

    # num DVs
    # nxh = 100 if nxe > 5 else nxe 
    nxh = nxe
    hvec = np.array([thick] * nxh)

    # plot_hermite_cubic()

    # 3) hybrid beam analysis
    # ------------------------

    hyb_beam = HybridAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
    hyb_beam.solve_forward(hvec)
    # hyb_beam.plot_disp()

    max_kirchoff_rot = np.max(np.abs(hyb_beam.u[1::3]))
    print(f"{max_kirchoff_rot=:.2e}")

    # 4) hellinger-reissner beam analysis
    # ------------------------

    hr_beam = HRBeamAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
    hr_beam.solve_forward(hvec)
    # hyb_beam.plot_disp()

    # numerical solution for center deflection
    w_vec = hyb_beam.u[0::3]
    nnodes = w_vec.shape[0]
    center = (0 + nnodes - 1) // 2
    pred_disp = w_vec[center]

    # exact solution for center deflection
    I = b * thick**3 / 12.0
    A = b * thick
    Ks = 5.0 / 6.0
    nu = 0.3
    G = E / 2.0 / (1 + nu)
    x = L / 2.0 # center of beam
    exact_disp = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4) + qmag * L**2 / 24.0 / G / A / Ks * (x / L - x**2 / L**2)
    # term1 = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4)
    # term2 = qmag * L**2 / 24.0 / G / Ks / A * (x / L - x**2 / L**2)
    # print(f"{term1=:.2e} {term2=:.2e}")

    # EB beam..
    eb_beam = EBAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
    eb_beam.solve_forward(hvec)
    # eb_beam.plot_disp()


    # timoshenko beam (TACS)
    ts_beam = TimoshenkoAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0, use_nondim=False)
    # ts_beam.red_int = True
    ts_beam.red_int = False
    ts_beam.solve_forward(hvec)
    w_vec2 = ts_beam.u[0::2]
    pred_disp_ts = w_vec2[center]
    zeta = 1.0 + E * I / 0.833 / G / A / L**2

    rel_err = abs((pred_disp - exact_disp) / exact_disp)
    print(f"{SR=:.2e} : {pred_disp=:.2e}, {exact_disp=:.2e} with {rel_err=:.2e}")
    print(f"\t{pred_disp_ts=:.2e} {zeta=:.2e}")

    exact_disps += [exact_disp]
    pred_disps += [pred_disp]
    SRs += [SR]

    # compute the condition number of the stiffness matrix
    # Kmat = hyb_beam.Kmat.toarray()
    # Kmat = ts_beam.Kmat.toarray()
    Kmat = hr_beam.Kmat.toarray()
    cond = np.linalg.cond(Kmat)
    print(f"\t{cond=:.2e}")

    conds += [cond]


# plot slenderness verify
# -------------

import matplotlib.pyplot as plt
import niceplots
plt.style.use(niceplots.get_style())
# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
plt.plot(SRs, exact_disps, label='exact')
plt.plot(SRs, pred_disps, '--', label='hybrid-TS')
plt.legend()
plt.xscale('log'); plt.yscale('log')
plt.xlabel("slenderness")
plt.ylabel("Beam center deflection")
plt.margins(x=0.05, y=0.05)
plt.savefig("hybrid-beam-SR.png", dpi=400)
plt.close('all')

# plot cond #s
# -----------

# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
plt.plot(SRs, conds)
plt.legend()
plt.xscale('log'); plt.yscale('log')
plt.xlabel("slenderness")
plt.ylabel("Kmat Cond #")
plt.margins(x=0.05, y=0.05)
plt.savefig("hybrid-beam-cond.png", dpi=400)
plt.close('all')

# plot disps in slender case
# ---------------------------

SR = 1.0
thick = 1.0 / SR
qmag = 1e4 * thick**3
hvec[:] = thick

hyb_beam = HybridAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : 1.0)
hyb_beam.solve_forward(hvec)
disp = hyb_beam.u
w = disp[0::3]
w_x = disp[1::3] / hyb_beam.dx / 0.5
th_s = disp[2::3]

th = w_x - th_s
xvec = np.linspace(0.0, L, w.shape[0])
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].plot(xvec, w_x, label='dw/dx')
ax[0].plot(xvec, -th_s, label='-th_s')
ax[0].plot(xvec, th, label='theta')
ax[0].legend()
ax[0].set_title("Hybrid TS")

# exact
xi = xvec / L
I = b * thick**3 / 12.0
A = b * thick
w_exact = qmag * L**4 / 24.0 / E / I * (xi - 2 * xi**3 + xi**4) + qmag * L**2 / 24.0 / G / A / Ks * (xi - xi**2)
wx_exact = qmag * L**3 / 24.0 / E / I * (1.0 - 6 * xi**2 + 4.0 * xi**3) + qmag * L / 24.0 / G / A / Ks * (1.0 - 2.0 * xi)
th_exact = qmag * L**3 / 24.0 / E / I * (1 - 6 * xi**2 + 4.0 * xi**3)
ths_exact = wx_exact - th_exact
ax[1].plot(xvec, wx_exact, label='dw/dx')
ax[1].plot(xvec, -ths_exact, label='-th_s')
ax[1].plot(xvec, th_exact, label='theta')
ax[1].legend()
ax[1].set_title("Exact TS")

# plt.show()
plt.savefig("hybrid-beam-defl.png", dpi=400)

# ---------------------
# old debug

# plot the displacements in SR = 1.0 case

# print(f"{qmag=}")
# for inode in range(10):
#     nodal_loads = eb_beam.force[2*inode:(2*inode+2)]
#     print(f"{inode=} : {nodal_loads=}")

# compute the total force on the beam compared to qmag
# qmag_tot = np.sum(eb_beam.force)
# qmag_exact = qmag * L
# print(f"{qmag_tot=:.2e} {qmag_exact=:.2e}")

# numerical solution for center deflection
# w_vec2 = eb_beam.u[0::2]
# pred_disp_eb = w_vec2[center]
# print(f"{pred_disp_eb=:.2e}")

# for very thin plate, equivalent is:
# eb_exact = 5.0 * qmag * L**4 / 384 / E / I
# print(f"{eb_exact=:.2e}") # it's not matching this somehow, need to fix this..