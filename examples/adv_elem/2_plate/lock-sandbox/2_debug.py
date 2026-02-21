import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
from iga_assembler import IGAPlateAssembler
from drig_assembler import DeRhamIGAPlateAssembler
from elem import HierarchicIsogeometricDispElement9, DeRhamIsogeometricPlateElement
from elem import DiscreteKirchoffLoveTrianglePlateElement, ReissnerMindlinPlateElement
from dkt_assembler import DKTPlateAssembler
from std_assembler import StandardPlateAssembler
from elem import ReissnerMindlinPlateElement_OptProlong, MITCPlateElement_OptProlong


from smooth import TwodimSupportAddSchwarz

sys.path.append("../../1_beam/src/")
from multigrid2 import vcycle_solve, VMG
from smoothers import BlockGaussSeidel, right_pcg2, right_pgmres2

sys.path.append("../../../asw/_py_demo/_src/")
from asw import TwodimAddSchwarz

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitc_lp', help="--elem, options: mitcp is another good one")
parser.add_argument("--nxe", type=int, default=4, help="number of elements")
parser.add_argument("--nxemin", type=int, default=2, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps for MG smoother")
parser.add_argument("--nprolong", type=int, default=1, help="number of smoothing steps for prolongator")
parser.add_argument("--omega", type=float, default=1.0, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='supp_asw', help="--smooth : [gs, asw, supp_asw]")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
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
elif args.elem == 'rmrp':
    ELEMENT = ReissnerMindlinPlateElement_OptProlong(
        reduced_integrated=True,
        prolong_mode='locking-global',
        lam=1e-6,
    )

elif args.elem == 'mitc':
    # standard prolongation so regular MITC
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='standard',
    )


elif args.elem == 'mitc_gp':
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='locking-global', # best..
        lam=1e-8,
    )

elif args.elem == 'mitc_lp':
    a_tying = 1.0
    # a_tying = 1.0 / np.sqrt(3)

    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='locking-local', # working reasonably..
        omega=0.5, # nope nvm 0.5 was better
        # omega=0.4,
        # lam=1e-8,
        lam=1e-12,
        n_lock_sweeps=args.nprolong,
        debug=True, # this is debug script,
        # debug=False,
        a=a_tying, b=a_tying,
    )

elif args.elem == 'mitc_ep':
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='energy-jacobi',
        omega=0.5, # NOPE nvm, 0.5 is better
        n_lock_sweeps=args.nprolong,
    )

# ================================
# make plate assembler
# ================================

# clamped = True
clamped = False # simply supported

# load_fcn = lambda x,y  : 1.0e2

m, n = 2, 3
load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)


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
        )
        grid._assemble_system()

        if args.elem == 'mitc_ep':
            # add kmat into kmat cache for the element
            ELEMENT._kmat_cache[nxe // 2] = grid.kmat.copy()

        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=args.omega, iters=nsmooth
            )
        elif args.coupled == 3 and args.smoother == 'supp_asw': 
            omega = args.omega / 8.0
            coupled = args.coupled
            smoother = TwodimSupportAddSchwarz.from_assembler(
                grid, omega=omega, iters=nsmooth, #, coupled_size=args.coupled
            )
        elif 'asw' in args.smoother:   
            omega = args.omega / 4.0 if args.coupled == 2 else args.omega / 8.0
            coupled = args.coupled
            SMOOTHER_CLASS = TwodimAddSchwarz # BEST

            # old good LU smoother
            smoother = SMOOTHER_CLASS.from_assembler(
                grid, omega=omega, iters=nsmooth, coupled_size=coupled
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

elif args.solve == 'kmg':

    line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc_ep']

    vmg2 = VMG(
        grids, nsmooth=args.nsmooth, 
        ncyc=1, # fewer total v-cycles often..
        # ncyc=2,
        smoothers=smoothers, 
        # line_search=False,
        line_search = line_search,
        # line_search=True,
    )
    pc = vmg2
    assembler._assemble_system()

    assembler, nsteps = right_pcg2(
        A=assembler.kmat, b=assembler.force,
        M=pc, rtol=1e-6, atol=1e-12,
        max_iter=200,
    )

    total_vcyc = vmg2.total_vcycles
    print(f"{total_vcyc=}")




# ================================
# DEBUG AREA
# ================================


G_f = ELEMENT.G_f
G_c = ELEMENT.G_c
P_gam = ELEMENT.P_gam
LHS = ELEMENT.M
RHS = ELEMENT.RHS
# free_cols_c = ELEMENT.free_cols_c
fixed_cols_c = ELEMENT.fixed_cols_c
solve_rows_f = ELEMENT.solve_rows_f
# P0_free = ELEMENT.P_0_free
# Mb = ELEMENT.Mb
P0 = ELEMENT.P_0
P = ELEMENT.P

import scipy.sparse as sp
Mb = sp.csr_matrix(LHS).tobsr(blocksize=(3,3))
P0 = P0.tobsr(blocksize=(3,3))
print(f"{RHS.shape=}")
RHS = sp.csr_matrix(RHS).tobsr(blocksize=(3,3))
P = sp.csr_matrix(P).tobsr(blocksize=(3,3))

sys.path.append("../src/elem/")
from _utils import debug_print_bsr_matrix

if ELEMENT.debug:

    # # DONE and verified LHS on CUDA C++
    # debug_print_bsr_matrix(Mb, name="locking-kmat")

    # # DONE and verified P0 on CUDA C++
    # debug_print_bsr_matrix(P0, name="init-prolongMat")

    # DONE and verified P on CUDA C++
    # debug_print_bsr_matrix(RHS, name="locking-p-rhs")

    # in progress verifying final locking-smoothed P
    debug_print_bsr_matrix(P, name="final-prolongMat")