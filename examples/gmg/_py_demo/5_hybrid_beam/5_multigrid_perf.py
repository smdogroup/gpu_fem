# try multigrid with each beam element
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from src import EBAssembler, HybridAssembler, TimoshenkoAssembler, ChebyshevTSAssembler, HRBeamAssembler
from src import vcycle_solve, block_gauss_seidel_smoother
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=192, help="num max elements in the beam assembler")
parser.add_argument("--nxe_min", type=int, default=48, help="num min elements in the beam assembler")
parser.add_argument("--nsmooth", type=int, default=4, help="num smoothing steps")
args = parser.parse_args()



SR_vec = [1.0, 10.0, 30.0, 50.0, 1e2, 300.0, 500.0, 1e3, 1e4]
beam_types = ['eb', 'ts', 'hyb', 'cfe1', 'cfe2', 'cfe3', 'cfe4', 'hr']
runtimes = {key:[] for key in beam_types}

for SR in SR_vec:

    # same problem settings for all
    # -----------------------------
    E = 2e7; b = 1.0; L = 1.0; rho = 1; qmag = 1e4; ys = 4e5; rho_KS = 50.0
    # scale by slenderness
    
    thick = L / SR
    qmag *= 1e5
    qmag *= thick**3
    hvec = np.array([thick] * args.nxe)
    load_fcn = lambda x : np.sin(3.0 * np.pi * x / L)

    # ----------------------------------
    # create all the grids / assemblers
    # ----------------------------------

    for beam in beam_types:

        print(f"\n--------------\n{SR=:.2e} {beam=}\n--------------\n")

        grids = []

        if beam == 'eb':
            # make euler-bernoulli beam assemblers for multigrid
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                eb_grid = EBAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                eb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [eb_grid]
                nxe = nxe // 2

        elif beam == 'ts':
            # make timoshenko beam assemblers
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                ts_grid = TimoshenkoAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                ts_grid.red_int = True
                ts_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [ts_grid]
                nxe = nxe // 2

        elif beam == 'hyb':
            # make hybrid beam assemblers
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                hyb_grid = HybridAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                hyb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [hyb_grid]
                nxe = nxe // 2

        elif beam == 'hr':
            # make hybrid beam assemblers
            nxe = args.nxe
            while (nxe >= args.nxe_min):
                hr_grid = HRBeamAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
                hr_grid.red_int = False
                hr_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [hr_grid]
                nxe = nxe // 2

        elif 'cfe' in beam:

            if beam == 'cfe1':
                order = 1
            elif beam == 'cfe2':
                order = 2
            elif beam == 'cfe3':
                order = 3
            elif beam == 'cfe4':
                order = 4

            nxe = args.nxe // order
            nxe_min = args.nxe_min // order

            while (nxe >= nxe_min):

                cfe_grid = ChebyshevTSAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn, order=order)
                cfe_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
                grids += [cfe_grid]
                nxe = nxe // 2

        # ----------------------------------
        # solve the multigrid using V-cycle
        # ----------------------------------

        start_time = time.time()
        if SR == 1.0 and beam == 'hr':
            runtimes[beam] += [np.nan] # bugged out cause constraint [-H, G; G^T, 0] formulation with BSR is tough to factor
            # better way to handle that?
            continue

        n_cycles = 50
        try:
            fine_soln, n_iters = vcycle_solve(grids, nvcycles=n_cycles, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                # debug_print=True,
                debug_print=False,
                print_freq=5,
                double_smooth=True,
            )
        except:
            n_iters = n_cycles
        dt = time.time() - start_time

        # runtimes[beam] += [dt]
        runtimes[beam] += [n_iters]


# now plot each runtimes
import niceplots
plt.style.use(niceplots.get_style())
for beam in beam_types:
    plt.plot(SR_vec, runtimes[beam], 'o-', label=beam)
    
plt.xlabel("Slenderness")
plt.xscale('log')
plt.ylabel(f"# V({args.nsmooth},{args.nsmooth})-cycles")
# plt.ylabel("Runtime (sec)")
plt.yscale('log')
plt.legend()
# plt.show()
plt.margins(x=0.05, y=0.05)

# plt.savefig("hybrid-beam-multigrid-times.png", dpi=400)
plt.savefig("beam-multigrid-cycles.png", dpi=400)