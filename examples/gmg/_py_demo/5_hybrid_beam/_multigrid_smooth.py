# try multigrid with each beam element
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from src import EBAssembler, HybridAssembler, TimoshenkoAssembler
from src import block_gauss_seidel_smoother

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=str, default='eb', help="--beam, options: eb, hyb, ts")
parser.add_argument("--nxe", type=int, default=128, help="num max elements in the beam assembler")
parser.add_argument("--SR", type=float, default=100.0, help="beam slenderness")
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
    while (nxe >= args.nxe):
        eb_grid = EBAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        eb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [eb_grid]
        nxe = nxe // 2

elif args.beam == 'ts':
    # make timoshenko beam assemblers
    nxe = args.nxe
    print(f"{args.SR=:.2e}")
    while (nxe >= args.nxe):
        ts_grid = TimoshenkoAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        ts_grid.red_int = True
        ts_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [ts_grid]
        nxe = nxe // 2

elif args.beam == 'hyb':
    # make hybrid beam assemblers
    nxe = args.nxe
    print(f"{args.SR=:.2e}")
    while (nxe >= args.nxe):
        hyb_grid = HybridAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, load_fcn=load_fcn)
        hyb_grid._compute_mat_vec(np.array([thick for _ in range(nxe)]))
        grids += [hyb_grid]
        nxe = nxe // 2

# ----------------------------------
# run multigrid smoother on the defect
# ----------------------------------

defect = grids[0].force.copy()
mat = grids[0].Kmat.copy()
soln = np.zeros_like(defect)

for i in range(100):
    if i % 5 == 0:
        nrm = np.linalg.norm(defect)
        print(f"{i=} => {nrm=:.2e}")

    soln, defect = block_gauss_seidel_smoother(mat, soln, defect, num_iter=1, dof_per_node=grids[0].dof_per_node)