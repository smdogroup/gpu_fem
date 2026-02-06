# Reissner-Mindlin plate with MITC4 shell elements, 6x6 DOF per node BSR matrix and using SA-AMG

"""basic SPAI demo for a Poisson linear system.."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import sys
# plate case imports from milu python cases
sys.path.append("../../milu/")
from _plate import make_plate_case
from __src import plot_plate_vec

sys.path.append("src/")
from iga_assembler import IGAPlateAssembler
from drig_assembler import DeRhamIGAPlateAssembler
from elem import HierarchicIsogeometricDispElement9, DeRhamIsogeometricPlateElement
from elem import DiscreteKirchoffLoveTrianglePlateElement, ReissnerMindlinPlateElement
from dkt_assembler import DKTPlateAssembler
from std_assembler import StandardPlateAssembler

# krylov imports
sys.path.append("../1_beam/src/")
from smoothers import right_pgmres2
from smoothers import BlockGaussSeidel

# AMG imports
sys.path.append("../../amg/_py_demo/_src/")
from bsr_aggregation import AMG_BSRSolver


# ====================================================
# 1) make plate case
# ====================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='rmr', help="--elem, options: tbd")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--thick", type=float, default=1e-2) # 2e-3
parser.add_argument("--justpc", action=argparse.BooleanOptionalAction, default=False, help="yes: just use pc one vec, no: solve with GMRES")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="whether to debug multilevel process")
parser.add_argument("--nokernel", action=argparse.BooleanOptionalAction, default=False, help="whether to turnoff kernel or not")
# it can even do thin plate quite well! maybe even better than multigrid?
parser.add_argument("--nxe", type=int, default=32, help="num elems each direction x and y")
parser.add_argument("--iters", type=int, default=1, help="num energy-opt iters (if iter == 1 same as SA-AMG)")
parser.add_argument("--nsmooth", type=int, default=2, help="num Jacobi ML smoothing steps (multilevel/multigrid-like)")
parser.add_argument("--threshold", type=float, default=0.13, help="aggregation sparsity threshold, higher threshold increases operator complexity and takes fewer GMRES iters, but inc memory and more time per iteration (tradeoff)")
parser.add_argument("--omega", type=float, default=0.4, help="jacobi smoother update coeff, default is None and uses spectral radius")
args = parser.parse_args()


""" verify each beam element and solver type against truth """

is_iga = False
# if args.elem == 'eb':
#     ELEMENT = EulerBernoulliElement()
if args.elem == 'higd':
    ELEMENT = HierarchicIsogeometricDispElement9(reduced_integrated=False)
    is_iga = True
elif args.elem == 'drig':
    ELEMENT = DeRhamIsogeometricPlateElement()
    is_iga = True
elif args.elem == 'drigr':
    # NOTE : drig element shouldn't need to be reduced integrated, was just trying it
    ELEMENT = DeRhamIsogeometricPlateElement(reduced_integrated=True)
    is_iga = True
elif args.elem == 'dkt':
    ELEMENT = DiscreteKirchoffLoveTrianglePlateElement()
elif args.elem == 'rm':
    ELEMENT = ReissnerMindlinPlateElement(reduced_integrated=False)
    if args.nsmooth < 4:
        print("need much more nsmooth like 4 for Reissner-Mindlin element")
elif args.elem == 'rmr':
    ELEMENT = ReissnerMindlinPlateElement(reduced_integrated=True)
    if args.nsmooth < 4:
        print("need much more nsmooth like 4 for Reissner-Mindlin element")

# ================================
# make plate assembler
# ================================

# clamped = True
clamped = False # simply supported

load_fcn = lambda x,y : 1.0e2 # simple load

# m, n = 2, 1
# m, n = 2, 2

# m, n = 3, 2
# load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

# if args.verify:
#     load_fcn = lambda x,y : 1.0 # simple load
# else:
#     m, n = 2, 3
#     # need non-simple load in general case (otherwise MG performance may be too benign for hermite elems)
#     # as each level can exactly solve it.. no iterative conv
#     load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

# ASSEMBLER = IGAPlateAssembler if is_iga else StandardBeamAssembler
if args.elem in ['drig', 'drigr']:
    ASSEMBLER = DeRhamIGAPlateAssembler
elif args.elem == 'dkt':
    ASSEMBLER = DKTPlateAssembler
elif is_iga:
    ASSEMBLER = IGAPlateAssembler
else:
    ASSEMBLER = StandardPlateAssembler


assembler = ASSEMBLER(
    ELEMENT=ELEMENT,
    nxe=args.nxe,
    thick=args.thick,
    clamped=clamped,
    split_disp_bc=args.elem in ['hhd', 'higd'],
    load_fcn=load_fcn,
)

assert not("drig" in args.elem) # doesn't work AMG for this..

assembler._assemble_system(bcs=False)
A_free = assembler.kmat.copy()
assert sp.isspmatrix_bsr(A_free) # DRIG element won't work with this..
assembler._assemble_system(bcs=True)
A = assembler.kmat.copy()
rhs = assembler.force.copy()
xpts = assembler.get_xpts()


# # ------------------------------------------------------------
# # GMRES Solve
# # ------------------------------------------------------------

from bsr_aggregation import get_rigid_body_modes #, get_coarse_rigid_body_modes
B = get_rigid_body_modes(xpts, vpn=3, 
                         th_flip=args.elem in ['rmr', 'rm'])

pc = AMG_BSRSolver(
    A_free, A, B, threshold=args.threshold, omega=args.omega,             
    ncyc=1 if args.thick >= 1e-3 else 2,
    # ncyc=1,
    # ncyc=2,
    pre_smooth=args.nsmooth, post_smooth=args.nsmooth, near_kernel=not(args.nokernel)
)

if args.justpc:
    x2 = pc.solve(rhs)
else:
    x2 = right_pgmres2(A, b=rhs, x0=None, restart=500, max_iter=500, M=pc if not(args.noprec) else None, rtol=1e-6)

print(f"{pc.total_vcycles=}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

if args.plot:
    print("plot fine soln using direct vs iterative solver")


    # ====================================================
    # 2) direct solve baseline
    # ====================================================

    # equiv solution with no reorder
    x_direct = sp.linalg.spsolve(A.copy(), rhs.copy())


    # for plotting
    nxe = int(nnodes**0.5)-1
    sort_fw = np.arange(0, N)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plot_plate_vec(nxe, x_direct.copy(), ax, sort_fw, nodal_dof=2)

    # plot right-precond solution
    ax = fig.add_subplot(122, projection='3d')
    plot_plate_vec(nxe, x2.copy(), ax, sort_fw, nodal_dof=2)
    plt.show()