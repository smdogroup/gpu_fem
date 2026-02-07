import numpy as np
import sys
sys.path.append("src/")
from elem import DeRhamIsogeometricCylinderElement
from drig_assembler import DeRhamIGACylinderAssembler


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='drig', help="--elem, options: tbd")
parser.add_argument("--nxe", type=int, default=32, help="number of elements")
parser.add_argument("--nxemin", type=int, default=8, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='vmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()


# R = 10.0
R = 0.1
# R = 1.0 # radius
# R = 4.0

ELEMENT = DeRhamIsogeometricCylinderElement(r=R, reduced_integrated=False)

ASSEMBLER = DeRhamIGACylinderAssembler
assembler = ASSEMBLER(
    ELEMENT,
    nxe=args.nxe,
    E=70e9, nu=0.3, thick=args.thick,
    length=1.0,
    # hoop_length=np.pi, # half-cylinder
    hoop_length=np.pi*0.5*R, # quarter-cylinder
    radius=R,
    load_fcn = lambda x,s : 1.0,
    clamped=False,
)

assembler.direct_solve()
assembler.plot_disp(
    disp_mag=0.5,
    mode="w",
    # mode="u",
    # mode="v",
    # mode="thx",
    # mode="thy",
    # deform="none",
    deform="w"
)