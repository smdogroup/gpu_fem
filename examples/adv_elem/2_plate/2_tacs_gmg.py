import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from iga_assembler import IGAPlateAssembler
from drig_assembler import DeRhamIGAPlateAssembler
from elem import HierarchicIsogeometricDispElement9, DeRhamIsogeometricPlateElement
from elem import DiscreteKirchoffLoveTrianglePlateElement, ReissnerMindlinPlateElement
from dkt_assembler import DKTPlateAssembler
from std_assembler import StandardPlateAssembler
from tacs_assembler import TACSAssembler

from asw_derham import TwoDimAddSchwarzDeRhamVertexEdges

sys.path.append("../1_beam/src/")
from multigrid2 import vcycle_solve, VMG
from smoothers import BlockGaussSeidel, right_pgmres2

sys.path.append("../../asw/_py_demo/_src/")
from asw import TwodimAddSchwarz

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitc', help="--elem, options: tbd")
parser.add_argument("--nxe", type=int, default=32, help="number of elements")
parser.add_argument("--nxemin", type=int, default=8, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='vmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=1, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()

# NOTE : prefers slightly higher omega at lower thickness?
# but won't converge if theta much larger than 0.5 at high thickness
# can also use only 1 smoothing step for thick = 1e-3 (totally fine)

# current smoother so cheap, similar cost to jacobi, not bad to just do 3 smoothing steps
# with DeRham cohomology, same smoothing cost to low and high thick! Solves like Kirchoff at low thick
# makes sure omega isn't too large otherwise perf can deteriorate, omega = 0.7 was best to me


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
elif args.elem == 'mitc':
    ELEMENT = None

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
if args.elem == 'drig':
    ASSEMBLER = DeRhamIGAPlateAssembler
elif args.elem == 'dkt':
    ASSEMBLER = DKTPlateAssembler
elif args.elem == 'mitc':
    ASSEMBLER = TACSAssembler
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
    bdf_file=f"out/plate{args.nxe}.bdf"
)

if not('mg' in args.solve):
    assembler._assemble_system()
    # plt.imshow(assembler.kmat.toarray())
    # plt.show()

# ================================
# make multigrid object (optional)
# ================================

if 'mg' in args.solve:
    # make V-cycle solver
    nxe = args.nxe
    nxe_min = args.nxemin
    if args.debug:
        nxe_min = nxe // 2
    grids = []
    smoothers = []
    # double_smooth = True
    double_smooth = False

    nsmooth = args.nsmooth
    while (nxe >= nxe_min):
        print(f"{nxe=}")
        grid = ASSEMBLER(
            ELEMENT=ELEMENT,
            nxe=nxe,
            thick=args.thick,
            clamped=clamped,
            split_disp_bc=args.elem in ['hhd', 'higd'],
            load_fcn=load_fcn,
            bdf_file=f"out/plate{nxe}.bdf"
        )
        grid._assemble_system()
        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=args.omega, iters=nsmooth
            )
        elif args.smoother == 'asw':
            smoother = None
            if args.coupled == 1:
                omega = args.omega / 2.0 # because some 2x smoothing on thx, thy
                patch_type = "vertex_edges"
            elif args.coupled == 2:
                omega = args.omega / 4.0 # because 2x smoothing than coupled == 1 schwarz (so ~4x smoothing on thx, thy)
                patch_type = "wblock_vertex_edges"

            if args.elem == 'drig':
                print("using Additive schwarz DeRham smoother")
                smoother = TwoDimAddSchwarzDeRhamVertexEdges.from_assembler(
                    grid, omega=omega, iters=nsmooth,
                    patch_type = patch_type,
                    # patch_type="vertex_edges", # one w vertex and nearby 4 edges (2 of thx and 2 of thy)
                    # patch_type="wblock_vertex_edges",
                )
            else:
                smoother = TwodimAddSchwarz.from_assembler(
                    grid, omega=omega, iters=nsmooth, coupled_size=2
                )
        smoothers += [smoother]
        nxe = nxe // 2
        if double_smooth:
            nsmooth *= 2


# ============================
# linear solve
# ============================

if args.solve == 'direct':
    assembler.direct_solve()
elif args.solve == 'vmg':

    assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                    #  line_search=not(args.elem == 'aig'))
                                    # line search sometimes hurts high cond # cases (high defects in prolong)
                                    # line_search=not(args.elem in ['aig', 'tsr', 'hhd', 'higd']), 
                                    # line_search=False, # often need it turned off.. for best conv
                                    # line_search = args.elem in ['drig', 'mitc'],
                                    line_search = args.elem in ['drig'],
                                    debug=args.debug,
                                    smoothers=smoothers)

elif args.solve == 'kmg':

    vmg2 = VMG(
        grids, nsmooth=args.nsmooth, 
        ncyc=1, # fewer total v-cycles often..
        # ncyc=2,
        smoothers=smoothers, line_search=args.elem in ['drig', 'drigr']
    )
    pc = vmg2
    assembler._assemble_system()

    assembler.u, nsteps = right_pgmres2(
        A=assembler.kmat, b=assembler.force,
        restart=100, M=pc, #M=vmg,
        rtol=1e-6,
    )

    total_vcyc = vmg2.total_vcycles
    print(f"{total_vcyc=}")


idof = 0
# idof = 1
# idof = 2

if args.plot:
    assembler.plot_disp() #idof=idof)


# ==============================
# VERIFICATION
# ==============================

# if args.verify:

#     # exact solution for center deflection
#     b = 1.0; thick = args.thick
#     L = assembler.L; E = assembler.E; qmag = 1.0; nu = assembler.nu
#     I = b * thick**3 / 12.0
#     A = b * thick
#     Ks = 5.0 / 6.0
#     G = E / 2.0 / (1 + nu)
#     x = L / 2.0 # center of beam
#     # fixed mistake is 1/2 on TS part.. 1/24 for EB is correct
#     # this removes correction of 24 in a bunch of TS elems
#     exact_disp = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4) + qmag * L**2 / 2.0 / G / A / Ks * (x / L - x**2 / L**2)

#     # predicted disp
#     u = assembler.u.copy()
#     if args.elem in ['hhd']:
#         # hierarchic disp has shear + bending split
#         w = u[0::3] + u[2::3]
#     elif args.elem in ['higd']:
#         w = u[0::2] + u[1::2]
#     else:
#         w = u[0::assembler.dof_per_node]
#     pred_disp = np.max(w)

#     REL_ERR = abs((pred_disp - exact_disp) / exact_disp)
#     print(f"{pred_disp=:.4e} {exact_disp=:.4e} {REL_ERR=:.4e}")


#     # ratio to EB solution
#     eb_disp = qmag * L**4 / 24.0 / E / I * (x / L - 2.0 * x**3 / L**3 + x**4 / L**4)
#     ts_over_eb = exact_disp / eb_disp
#     margin = ts_over_eb - 1.0
#     print(f"TS/EB disp = 1 + {margin=:.4e}")