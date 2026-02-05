import numpy as np
import matplotlib.pyplot as plt
import sys

from diga_assembler import DeRhamIGABeamAssembler_V0

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=128, help="number of elements")
parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='vmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.9, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()


""" verify each beam element and solver type against truth """

# ================================
# make beam assembler
# ================================

clamped = True
# clamped = False # simply supported

load_fcn = lambda x : 1.0 # simple load
# load_fcn = lambda x : np.sin(4.0 * np.pi * x)

# if args.verify:
#     load_fcn = lambda x : 1.0 # simple load
# else:
#     # need non-simple load in general case (otherwise MG performance may be too benign for hermite elems)
#     # as each level can exactly solve it.. no iterative conv
#     load_fcn = lambda x : np.sin(4.0 * np.pi * x)

# # ASSEMBLER = IGABeamAssembler if is_iga else StandardBeamAssembler
# ASSEMBLER = IGABeamAssemblerV2 if is_iga else StandardBeamAssembler

assembler = DeRhamIGABeamAssembler_V0(
    ELEMENT=None,
    nxe=args.nxe,
    thick=args.thick,
    clamped=clamped,
    load_fcn=load_fcn,
)

# assembler._assemble_system()
u = assembler.direct_solve()

w = u[:assembler.nw]
w_max = np.max(w)
print(f"{w_max=:.4e}")


x = np.linspace(0.0, 1.0, assembler.nw)
plt.plot(x, w)
plt.show()