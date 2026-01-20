import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import sys
sys.path.append("_src/")
from beam_assembler import BeamFem
from asw import OnedimAddSchwarz

sys.path.append("../../milu/")
from __linalg import right_pgmres

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=50, help="num elems in each direction")
parser.add_argument("--size", type=int, default=3, help="coupling size of schwarz smoother")
parser.add_argument("--omega", type=float, default=0.3, help="additive coefficient for ASW")
parser.add_argument("--iters", type=int, default=5, help="num schwarz smooths per krylov step (more than one allowed)")
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="just solve pc instead of GMRES")
args = parser.parse_args()

# =================================
# build Euler-Bernoulli beam assembler
# =================================

# nxe = 10
thick = 1e-1 # no reason to change thickness cause it's EB beam
nxh = 1
beam_fea = BeamFem(args.nxe, nxh=nxh, E=70e9, b=1.0, L=1.0, 
                         rho=2.7e3, qmag=1.0, ys=2e6, rho_KS=100.0, dense=False)

# =================================
# direct solve reference
# =================================

hred = np.ones(nxh) * thick
soln = beam_fea.solve_forward(hred)

mat = beam_fea.Kmat.copy()
rhs = beam_fea.force.copy()

# =================================
# additive schwarz Krylov solve
# =================================

# coupled_size = 1 # jacobi, otherwise it's coupled smoothing

pc = OnedimAddSchwarz(mat, rhs, block_dim=2, 
                      coupled_size=args.size, omega=args.omega, iters=args.iters)

if args.justpc:
    u2 = pc.solve(rhs)
else:
    u2 = right_pgmres(mat, b=rhs, x0=None, restart=500, max_iter=500, M=pc)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
# plot_poisson_surface(u, u2, nx, ny, Lx, Ly)
# plot_poisson_surface(u2, nx, ny, Lx, Ly)

w = rhs[0::2]
x = np.arange(0, w.shape[0])

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].plot(x, soln[0::2], label="direct")
ax[1].plot(x, u2[0::2], label="asw")
plt.show()