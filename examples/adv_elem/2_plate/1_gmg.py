import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from iga_assembler import IGAPlateAssembler
from drig_assembler import DeRhamIGAPlateAssembler
from stab_assembler import StabilizedPlateAssembler
from elem import HierarchicIsogeometricDispElement9, DeRhamIsogeometricPlateElement
from elem import DiscreteKirchoffLoveTrianglePlateElement, ReissnerMindlinPlateElement
from dkt_assembler import DKTPlateAssembler
from std_assembler import StandardPlateAssembler
from elem import ReissnerMindlinPlateElement_OptProlong, MITCPlateElement_OptProlong
from elem import AlgebraicSubGridScaleElement

from smooth import TwoDimAddSchwarzDeRhamVertexEdges
# from smooth import TwodimAddSchwarzColored22_BC, TwodimAddSchwarzColored22
from smooth import TwodimSVDAddSchwarz
from smooth import TwodimSupportAddSchwarz, TwodimElementAddSchwarz

sys.path.append("../1_beam/src/")
from multigrid2 import vcycle_solve, VMG
from smoothers import BlockGaussSeidel, right_pcg2, right_pgmres2

sys.path.append("../../asw/_py_demo/_src/")
from asw import TwodimAddSchwarz

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitc_lp', help="--elem, options: mitcp is another good one")
parser.add_argument("--nxe", type=int, default=32, help="number of elements")
parser.add_argument("--nxemin", type=int, default=8, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=3, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps for MG smoother")
parser.add_argument("--nprolong", type=int, default=10, help="number of smoothing steps for prolongator")
parser.add_argument("--omega", type=float, default=1.0, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='supp_asw', help="--smooth : [gs, asw, supp_asw]")
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


# SETTING #1
# fastest K(2,2)-GMG conv: coupled = 3, nsmooth = 2, supp_asw smoother
# but the 3x3 support ASW smoother probably won't be scalable on GPU (3x3 = 9 node and 81 block dense inverse bad, factor and triang not great either prob)
# maybe if did ILU(0) for these subdomains.. but not sure

# SETTING #2
# good K(4,4)-GMG conv: coupled = 2, nsmooth = 4, regular asw smoother (2x2 elem-ASW is scalable on GPU and already written for wing case)


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
        # prolong_mode='standard',
        # lam=1e-12,
        lam=1e-6,
        # lam=1e-4,
        # lam=1e-2,
        # lam=1.0,
    )
elif args.elem == 'mitc':
    # standard prolongation so regular MITC
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='standard',
    )
elif args.elem == 'mitc_gp':
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='locking-global', # best..
        # lam=1e-12,
        lam=1e-8,
        # lam=1e-6,
        # lam=1e-4,
    )
elif args.elem == 'mitc_lp':
    ELEMENT = MITCPlateElement_OptProlong(
        # prolong_mode='locking-global', # best..
        prolong_mode='locking-local', # working reasonably..
        # prolong_mode='standard',
        # omega=0.4, # needed smaller omega for coarser mesh (in general should use CG-Lanczos to get spectral radius.. helps)
        omega=0.5, # nope nvm 0.5 was better
        # omega=0.6,
        # omega=0.7,
        # lam=1e-12,
        lam=1e-8,
        # lam=1e-6,
        # lam=1e-4,
        # lam=1e-2,
        # lam=1.0,
        # n_lock_sweeps=4,
        # n_lock_sweeps=8,
        # n_lock_sweeps=10, # default,
        n_lock_sweeps=args.nprolong,
        # n_lock_sweeps=20,
        # n_lock_sweeps=30,
        a=1.0, b=1.0, # true MITC shells use edge tying points for first order (but leads to slower and a bit more unstable smooth prolong conv) than a,b = 1/sqrt(3) below
        # a=1.0/np.sqrt(3), b=1.0/np.sqrt(3)
    )
elif args.elem == 'mitc_ep':
    ELEMENT = MITCPlateElement_OptProlong(
        prolong_mode='energy-jacobi',
        # omega=0.4, # needed smaller omega for coarser mesh (in general should use CG-Lanczos to get spectral radius.. helps)
        # omega=0.4,
        omega=0.5, # NOPE nvm, 0.5 is better
        # omega=0.7,
        # n_lock_sweeps=4,
        # n_lock_sweeps=8,
        # n_lock_sweeps=10,
        # n_lock_sweeps=10, # default,
        n_lock_sweeps=args.nprolong,
        # n_lock_sweeps=30,
        a=1.0, b=1.0, # true MITC shells use edge tying points for first order (but leads to slower and a bit more unstable smooth prolong conv) than a,b = 1/sqrt(3) below
        # a=1.0/np.sqrt(3), b=1.0/np.sqrt(3)
    )
elif args.elem == 'asgs':
    # ELEMENT = AlgebraicSubGridScaleElement()
    ELEMENT = AlgebraicSubGridScaleElement(
        edge_stab=True,
        # edge_stab=False
    )

# ================================
# make plate assembler
# ================================

# clamped = True
clamped = False # simply supported

# load_fcn = lambda x,y : 1.0e2 # simple load

# m, n = 2, 1
# m, n = 2, 2

m, n = 2, 3
load_fcn = lambda x,y : np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

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
elif args.elem == 'asgs':
    ASSEMBLER = StabilizedPlateAssembler
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
        elif 'asw' in args.smoother:
            if args.elem in ['drig', 'drigr']:
                omega = args.omega * 0.5 # need extra mult for them (default best values)
                coupled = args.coupled
                if coupled > 2:
                    print(f"{args.elem=} and {args.coupled=} dropped to 2")
                    coupled = 2
            else:
                omega = args.omega
                coupled = args.coupled

            smoother = None
            if coupled == 1:
                omega = omega / 2.0 # because some 2x smoothing on thx, thy
                patch_type = "vertex_edges"
            elif coupled == 2:
                omega = omega / 4.0 # because 2x smoothing than coupled == 1 schwarz (so ~4x smoothing on thx, thy)
                patch_type = "wblock_vertex_edges"
            elif coupled == 3:
                assert not (args.elem in ['drig', 'drigr'])
                omega = omega / 8.0

            if args.elem in ['drig', 'drigr']:
                print("using Additive schwarz DeRham smoother")
                smoother = TwoDimAddSchwarzDeRhamVertexEdges.from_assembler(
                    grid, omega=omega, iters=nsmooth,
                    patch_type = patch_type,
                    # patch_type="vertex_edges", # one w vertex and nearby 4 edges (2 of thx and 2 of thy)
                    # patch_type="wblock_vertex_edges",
                )
            # elif args.coupled == 2:
            #     print("doing colored asw 2x2")
            #     # smoother = TwodimAddSchwarzColored22.from_assembler(
            #     #     grid, omega=omega, iters=nsmooth, coupled_size=args.coupled,
            #     #     nugget_rel=1e-12, # doesn't help the nugget
            #     # )

            #     smoother = TwodimAddSchwarzColored22_BC.from_assembler(
            #         grid, omega=omega, iters=nsmooth,
            #     )

            elif args.coupled == 3 and args.smoother == 'supp_asw':
                smoother = TwodimSupportAddSchwarz.from_assembler(
                    grid, omega=omega, iters=nsmooth, #, coupled_size=args.coupled
                )
                
            else:
                # SMOOTHER_CLASS = TwodimSupportAddSchwarz # can't use this on 2x2 coupled or lower
                SMOOTHER_CLASS = TwodimAddSchwarz # BEST
                # SMOOTHER_CLASS = TwodimElementAddSchwarz # ends up being worse than regular add-Schwarz by a few iterations

                # old good LU smoother
                smoother = SMOOTHER_CLASS.from_assembler(
                    grid, omega=omega, iters=nsmooth, coupled_size=coupled
                )

                # NOT GOOD IDEA SVD smoother
                # smoother = TwodimSVDAddSchwarz.from_assembler(
                #     grid, omega=omega, iters=nsmooth, coupled_size=args.coupled,
                #     # use_svd_threshold=False,
                #     use_svd_threshold=True,
                #     svd_alpha=1e-10
                # )

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

    # mitc_ep does better without line search btw
    # line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp']
    line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc', 'mitc_ep'] # better not with asgs?
    # line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc', 'mitc_ep', 'asgs']
    # if args.elem == 'mitcp':
    #     line_search = ELEMENT.prolong_mode != 'locking-local'

    assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                    #  line_search=not(args.elem == 'aig'))
                                    # line search sometimes hurts high cond # cases (high defects in prolong)
                                    # line_search=not(args.elem in ['aig', 'tsr', 'hhd', 'higd']), 
                                    # line_search=False, # often need it turned off.. for best conv
                                    line_search = line_search,
                                    # line_search=True,
                                    debug=args.debug,
                                    # nvcycles=100,
                                    nvcycles=200,
                                    # nvcycles=1000,
                                    rtol=1e-6,
                                    smoothers=smoothers)

elif args.solve == 'kmg':

    # line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp']
    line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc', 'mitc_ep'] # better not with asgs?
    # line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc', 'mitc_ep', 'asgs']
    # if args.elem == 'mitcp':
    #     line_search = ELEMENT.prolong_mode != 'locking-local'

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

    # pc = smoothers[0] # DEBUG

    if args.elem in ['rm', 'rmr']:
        # these elems have nonsym P and R (can fix) but for now use this
        assembler.u, nsteps = right_pgmres2(
            A=assembler.kmat, b=assembler.force,
            restart=200, 
            M=pc, #M=vmg,
            rtol=1e-6,
        )

    else:

        assembler.u, nsteps = right_pcg2(
            A=assembler.kmat, b=assembler.force,
            M=pc, rtol=1e-6, atol=1e-12,
            max_iter=200,
            print_freq=3
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

w = assembler.u[0::3]
w_max = np.max(np.abs(w))
print(f"{args.nxe=} {w_max=:.4e}")

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