import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from std_assembler import StandardBeamAssembler
from elem import EulerBernoulliElement, TimoshenkoElement, HybridHermiteElement
# sys.path.append("src/elem")
# from eb_elem import EulerBernoulliElement
from multigrid import VcycleSolver
from smoothers import BlockGaussSeidel, OnedimAddSchwarz, right_pcg2, right_pgmres2
from multigrid2 import vcycle_solve, VMG

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='eb', help="--beam, options: hyb, ts, ts-nd")
parser.add_argument("--nxe", type=int, default=128, help="number of elements")
parser.add_argument("--thick", type=float, default=1e-2, help="number of elements")
parser.add_argument("--solve", type=str, default='vmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
args = parser.parse_args()


""" verify each beam element and solver type against truth """

if args.elem == 'eb':
    ELEMENT = EulerBernoulliElement()
elif args.elem == 'ts':
    ELEMENT = TimoshenkoElement(reduced_integrated=False)
elif args.elem == 'tsr':
    ELEMENT = TimoshenkoElement(reduced_integrated=True)
elif args.elem == 'hybh':
    ELEMENT = HybridHermiteElement(reduced_integrated=False)

# ================================
# make beam assembler
# ================================

assembler = StandardBeamAssembler(
    ELEMENT=ELEMENT,
    nxe=args.nxe,
    thick=args.thick
)
if not('mg' in args.solve):
    assembler._assemble_system()

# ================================
# make multigrid object (optional)
# ================================

if 'mg' in args.solve:
    # make V-cycle solver
    nxe = args.nxe
    # nxe_min = 5
    # nxe_min = 8
    nxe_min = 16
    # nxe_min = nxe // 2
    grids = []
    smoothers = []
    while (nxe >= nxe_min):
        print(f"{nxe=}")
        grid = StandardBeamAssembler(
            ELEMENT=ELEMENT,
            nxe=nxe,
            thick=args.thick,
        )
        grid._assemble_system()
        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=0.7, iters=args.nsmooth
            )
        elif args.smoother == 'asw':
            smoother = OnedimAddSchwarz.from_assembler(
                grid, omega=0.3, iters=args.nsmooth, coupled_size=2
            )
        smoothers += [smoother]

        nxe = nxe // 2

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
    # new style
    # assembler._assemble_system()
    # assembler.u = vmg.solve(assembler.force)

    # old V-cycle solver
    assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth)

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

assembler.plot_disp(idof=idof)