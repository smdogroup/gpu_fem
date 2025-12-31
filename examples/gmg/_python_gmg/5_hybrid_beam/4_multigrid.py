# try multigrid with each beam element
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from src import EBAssembler, HybridAssembler, TimoshenkoAssembler, ChebyshevTSAssembler
from src import HybridChebyshevAssembler, HRBeamAssembler
from src import vcycle_solve, block_gauss_seidel_smoother

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=str, default='eb', help="--beam, options: eb, hyb, ts")
parser.add_argument("--nxe", type=int, default=192, help="num max elements in the beam assembler")
parser.add_argument("--nxe_min", type=int, default=48, help="num min elements in the beam assembler")
parser.add_argument("--SR", type=float, default=100.0, help="beam slenderness")
parser.add_argument("--nsmooth", type=int, default=2, help="num smoothing steps")
parser.add_argument("--order", type=int, default=2, help="basis order")
args = parser.parse_args()

grids = []

# same problem settings for all
# -----------------------------
E = 2e7; b = 1.0; L = 1.0; rho = 1; qmag = 1e4; ys = 4e5; rho_KS = 50.0

# scale by slenderness
thick = L / args.SR
qmag *= 1e5
qmag *= thick**3
hvec = np.array([thick] * args.nxe)
load_fcn = lambda x : np.sin(3.0 * np.pi * x / L)

# ----------------------------------
# create all the grids / assemblers
# ----------------------------------

if args.beam == 'eb':
    # make euler-bernoulli beam assemblers for multigrid
    nxe = args.nxe
    while (nxe >= args.nxe_min):
        eb_grid = EBAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        eb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [eb_grid]
        nxe = nxe // 2

elif args.beam == 'ts':
    # make timoshenko beam assemblers
    nxe = args.nxe
    # print(f"{args.SR=:.2e}")
    while (nxe >= args.nxe_min):
        ts_grid = TimoshenkoAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        ts_grid.red_int = True
        # ts_grid.red_int = False
        ts_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [ts_grid]
        nxe = nxe // 2

elif args.beam == 'hyb':
    # make hybrid beam assemblers
    nxe = args.nxe
    # print(f"{args.SR=:.2e}")
    while (nxe >= args.nxe_min):
        hyb_grid = HybridAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        hyb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [hyb_grid]
        nxe = nxe // 2

elif args.beam == 'hr':
    # make hybrid beam assemblers
    nxe = args.nxe
    while (nxe >= args.nxe_min):
        hr_grid = HRBeamAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        hr_grid.red_int = False
        hr_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [hr_grid]
        nxe = nxe // 2

elif args.beam == 'cfe':
    # make hybrid beam assemblers
    nxe = args.nxe // args.order
    print(f"{args.SR=:.2e} with {args.order=}")
    nxe_min = args.nxe_min // args.order

    while (nxe >= nxe_min):
        print(F"making Chebyshev grid {nxe=}")
        cfe_grid = ChebyshevTSAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn, order=args.order)
        cfe_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [cfe_grid]
        nxe = nxe // 2

        # mat = cfe_grid.Kmat
        # print(f"{type(mat)=}")

elif args.beam == 'hcfe':
    # make hybrid beam assemblers
    nxe = args.nxe // args.order
    print(f"{args.SR=:.2e} with {args.order=}")
    nxe_min = args.nxe_min // args.order

    while (nxe >= nxe_min):
        print(F"making Hybrid-Chebyshev grid {nxe=}")
        hyb_cfe_grid = HybridChebyshevAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=True, load_fcn=load_fcn, order=args.order)
        hyb_cfe_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [hyb_cfe_grid]
        nxe = nxe // 2

        # mat = cfe_grid.Kmat
        # print(f"{type(mat)=}")

# ----------------------------------
# solve the multigrid using V-cycle
# ----------------------------------

fine_soln, n_iters = vcycle_solve(grids, nvcycles=100, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
    # debug_print=True,
    debug_print=False,
)