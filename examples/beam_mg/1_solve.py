import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from std_assembler import StandardBeamAssembler
from elem import EulerBernoulliElement, TimoshenkoElement, HierarchicRotHermiteElement, HierarchicDispHermiteElement
# sys.path.append("src/elem")
# from eb_elem import EulerBernoulliElement
from multigrid import VcycleSolver
from smoothers import BlockGaussSeidel, OnedimAddSchwarz, right_pcg2, right_pgmres2
from multigrid2 import vcycle_solve, VMG

# IGA elements
from elem import AsymptoticIsogeometricTimoshenkoElement
from iga_assembler import IGABeamAssembler

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='hhd', help="--beam, options: hyb, ts, ts-nd")
parser.add_argument("--nxe", type=int, default=128, help="number of elements")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='vmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=4, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.9, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
args = parser.parse_args()


""" verify each beam element and solver type against truth """

is_iga = False
if args.elem == 'eb':
    ELEMENT = EulerBernoulliElement()
elif args.elem == 'ts':
    ELEMENT = TimoshenkoElement(reduced_integrated=False)
elif args.elem == 'tsr':
    ELEMENT = TimoshenkoElement(reduced_integrated=True)
elif args.elem == 'hhr':
    ELEMENT = HierarchicRotHermiteElement(reduced_integrated=False)
elif args.elem == 'hhd':
    ELEMENT = HierarchicDispHermiteElement(reduced_integrated=False)
elif args.elem == 'aig':
    ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=False)
    is_iga = True

# ================================
# make beam assembler
# ================================

# clamped = True
clamped = False # simply supported

ASSEMBLER = IGABeamAssembler if is_iga else StandardBeamAssembler

assembler = ASSEMBLER(
    ELEMENT=ELEMENT,
    nxe=args.nxe,
    thick=args.thick,
    clamped=clamped,
    split_disp_bc=args.elem in ['hhd']
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
    # nxe_min = 5
    # nxe_min = 8
    nxe_min = 16
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
            split_disp_bc=args.elem in ['hhd']
        )
        grid._assemble_system()
        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=args.omega, iters=nsmooth
            )
        elif args.smoother == 'asw':
            omega = args.omega / 2.0 # since 2x smoothing
            smoother = OnedimAddSchwarz.from_assembler(
                grid, omega=omega, iters=nsmooth, coupled_size=2
            )
        smoothers += [smoother]
        nxe = nxe // 2
        if double_smooth:
            nsmooth *= 2

    vmg = VcycleSolver(
        grids=grids,
        smoothers=smoothers
    )


# ============================
# linear solve
# ============================

if args.solve == 'direct':
    assembler.direct_solve()
elif args.solve == 'vmg':
    # DEBUG
    if args.debug:
        assembler._assemble_system()
        assembler.u = vmg.solve(assembler.force)

    # proper V-cycle solver
    if not args.debug:
        # smoothers = None # DEBUG
        # print("WARNING using internal GS smoother, not ASW/GS created above")

        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                        #  line_search=not(args.elem == 'aig'))
                                        # line search sometimes hurts high cond # cases (high defects in prolong)
                                        line_search=not(args.elem in ['aig', 'tsr', 'hhd']), 
                                        # line_search=True,
                                        smoothers=smoothers)

elif args.solve == 'kmg':
    vmg.n_cycles = 2
    vmg.print = False

    vmg2 = VMG(grids, args.nsmooth, 2)
    
    # assembler.u, nsteps = right_pcg2(
    #     A=assembler.kmat, b=assembler.force,
    #     max_iter=200, M=vmg
    # )

    # pc = vmg
    pc = vmg2
    # pc = None
    assembler._assemble_system()

    assembler.u, nsteps = right_pgmres2(
        A=assembler.kmat, b=assembler.force,
        restart=100, M=pc, #M=vmg
    )


idof = 0
# idof = 1

if args.plot:
    assembler.plot_disp(idof=idof)