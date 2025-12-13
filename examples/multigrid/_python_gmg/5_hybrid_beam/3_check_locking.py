# TBD, the solution is the same no matter how many elements there are..
# maybe because it's exact polynomial basis for the beam problem here (full hermite cubic)
# but not clear why the solution doesn't change to the Timoshenko exact

# now let's test this out and visualize it
import numpy as np
import matplotlib.pyplot as plt
from src import HybridAssembler, HRBeamAssembler
from src import TimoshenkoAssembler
from src import ChebyshevTSAssembler
from scipy.sparse.linalg import spsolve

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=str, default='hyb', help="--beam, options: hyb, ts, ts-nd")
parser.add_argument("--rint", type=int, default=0, help="reduced integrated 0 or 1 for False,True (default False), only applies to regular TS element")
args = parser.parse_args()

# verify the hybrid assembler solution against exact Timoshenko beam theory solution for constant distributed load

# exact_disps = []
# pred_disps = []
# SRs = []
# conds = []

def exact_TS_disp(E, b, thick, L, qmag, nu=0.3, Ks=5.0 / 6.0):
    # exact solution for center deflection
    I = b * thick**3 / 12.0
    A = b * thick
    G = E / 2.0 / (1 + nu)
    x = L / 2.0 # center of beam
    return qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4) + qmag * L**2 / 24.0 / G / A / Ks * (x / L - x**2 / L**2)

# SR_vec = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
# SR_vec = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
# SR_vec = SR_vec[::-1]
SR_vec = np.array([1.0, 10.0, 100.0, 1000.0])

rates = []

for SR in SR_vec:
# for SR in [10000.0]:

    # setup
    # -------------------------------

    E = 2e7; b = 1.0; L = 1.0; rho = 1; qmag = 1e4; ys = 4e5; rho_KS = 50.0
    # scale by slenderness
    thick = 1.0 / SR
    qmag *= thick**3

    dof_vec = []
    disps = []
    exact_disps = []

    # for nxe in [8, 16, 32, 64, 128, 256, 512, 1024]: #, 2048, 4096]:
    for nxe in [8, 16, 32, 128, 256, 512, 1024]: #, 2048, 4096]:
        
        print(f"{SR=:.2e} {nxe=}")

        nxh = nxe
        hvec = np.array([thick] * nxh)

        if "hyb" in args.beam:

            # get hybrid beam disp
            hyb_beam = HybridAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : np.sin(3 * np.pi * x / L))
            hyb_beam.solve_forward(hvec)
            
            # if nxe == 512 and SR == 1.0:
            #     hyb_beam.plot_disp()
            #     # 7.67e-7 disp at SR = 1000.0
            #     # 2.93e-6 disp at SR = 1.0

            # numerical solution for center deflection
            w_vec = hyb_beam.u[0::3]
            nnodes = w_vec.shape[0]
            center = (0 + nnodes - 1) // 2
            pred_disp = w_vec[center]

            disps += [pred_disp]
            dof_vec += [nxe * 3]

        elif "hr" in args.beam:

            # get hybrid beam disp
            hr_beam = HRBeamAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : np.sin(3 * np.pi * x / L))
            hr_beam.solve_forward(hvec)
            # hr_beam.plot_disp()
            
            # if nxe == 512 and SR == 1.0:
            #     hyb_beam.plot_disp()
            #     # 7.67e-7 disp at SR = 1000.0
            #     # 2.93e-6 disp at SR = 1.0

            # numerical solution for center deflection
            w_vec = hr_beam.u[0::3]
            nnodes = w_vec.shape[0]
            center = (0 + nnodes - 1) // 2
            pred_disp = w_vec[center]

            disps += [pred_disp]
            dof_vec += [nxe * 3]

        elif args.beam == 'ts':

            # get standard timoshenko beam disp
            ts_beam = TimoshenkoAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : np.sin(3 * np.pi * x / L), use_nondim=False)
            
            if args.rint:
                ts_beam.red_int = True
            else:
                ts_beam.red_int = False
            
            ts_beam.solve_forward(hvec)
            
            # if nxe == 512 and SR == 1.0:
            #     ts_beam.plot_disp()
            #     # 7.62e-7 at SR = 1000.0
            #     # 2.96e-6 at SR = 10.0

            # numerical solution for center deflection
            w_vec = ts_beam.u[0::2]
            nnodes = w_vec.shape[0]
            center = (0 + nnodes - 1) // 2
            pred_disp = w_vec[center]

            disps += [pred_disp]
            dof_vec += [nxe * 2]

        elif args.beam == 'ts-nd':

            # get standard timoshenko beam disp
            print(f"timoshenko using nondim")
            ts_beam = TimoshenkoAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : np.sin(3 * np.pi * x / L), use_nondim=True)
            
            if args.rint:
                ts_beam.red_int = True
            else:
                ts_beam.red_int = False
            
            ts_beam.solve_forward(hvec)
            
            # if nxe == 512 and SR == 1.0:
            #     ts_beam.plot_disp()
            #     # 7.62e-7 at SR = 1000.0
            #     # 2.96e-6 at SR = 10.0

            # numerical solution for center deflection
            w_vec = ts_beam.u[0::2]
            nnodes = w_vec.shape[0]
            center = (0 + nnodes - 1) // 2
            pred_disp = w_vec[center]

            disps += [pred_disp]
            dof_vec += [nxe * 2]

        elif 'cfe' in args.beam:

            # order = 1
            order = 2
            # order = 3

            cfe_beam = ChebyshevTSAssembler(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=lambda x : np.sin(3 * np.pi * x / L), order=order)
            # cfe_beam.red_int = False
            cfe_beam._compute_mat_vec(np.array([thick]*nxe))
            cfe_beam.u = spsolve(cfe_beam.Kmat, cfe_beam.force)
            # cfe_beam.plot_disp()

            w_vec = cfe_beam.u[0::2]
            nnodes = w_vec.shape[0]
            center = (0 + nnodes - 1) // 2
            pred_disp = w_vec[center]
            # print(f"{pred_disp=}")

            disps += [pred_disp]
            dof_vec += [nnodes * 2]


        # exact_disp = exact_TS_disp(E, b, thick, L, qmag)
        # exact_disps += [exact_disp]

    # plt.close('all')
    disps = np.array(disps)
    disp_errs = np.abs(disps - disps[-1] + 1e-12) / np.abs(disps[-1])

    plt.plot(dof_vec[:-1], disp_errs[:-1], label=f"{SR=:.0f}")
    # plt.xscale('log'); plt.yscale('log')
    # plt.plot(nxe_vec, exact_disps, label=f'{SR=:.0f}-exact')
    # plt.legend()
    # plt.show()

    # print rate of conv
    rate = np.log(disp_errs[-2] / disp_errs[0]) / np.log(dof_vec[-2] / dof_vec[0])
    print(f"{SR=} {rate=}")

    rates += [rate]

rate = float(rate)
if args.beam == 'hyb':
    plt.text(100, 2e-3, f"{SR=} {rate=:.2f}")
else:
    # plot least slender rate
    SR, rate = SR_vec[0], float(rates[0])
    plt.text(100, 2e-3, f"{SR=} {rate=:.2f}")

    # plot most slender rate
    SR, rate = SR_vec[-1], float(rates[-1])
    plt.text(200, 1e-1, f"{SR=} {rate=:.2f}")


plt.xscale('log'); plt.yscale('log')
plt.xlabel("NDOF")
plt.ylabel("Disp Err")
plt.legend()
# plt.show()
if args.beam == 'hyb':
    prefix = 'hybrid'

elif args.beam == 'ts':
    prefix = 'timoshenko'

elif args.beam == 'hr':
    prefix = 'hell-reisner'

elif args.beam == 'ts-nd':
    prefix = 'timoshenko-nondim'

elif args.beam == 'cfe':
    prefix = 'chebyshev'

if args.rint:
    prefix += '-rint'

plt.savefig(f"{prefix}-beam-locking-check.png", dpi=400)