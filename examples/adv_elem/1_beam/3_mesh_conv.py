import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")
from std_assembler import StandardBeamAssembler
from elem import EulerBernoulliElement, TimoshenkoElement, HierarchicRotHermiteElement, HierarchicDispHermiteElement
# sys.path.append("src/elem")
# from eb_elem import EulerBernoulliElement
from multigrid import VcycleSolver
from smoothers import BlockGaussSeidel, OnedimAddSchwarz, right_pcg2, right_pgmres2, OneDimAddSchwarzVertex2Edges
from multigrid2 import vcycle_solve, VMG

# IGA elements
from elem import AsymptoticIsogeometricTimoshenkoElement, HierarchicIsogeometricDispElement
from iga_assembler import IGABeamAssembler
from iga_assembler2 import IGABeamAssemblerV2

# special vertex-edge DeRham style element
from elem import DeRhamIsogeometricElement
from diga_assembler import DeRhamIGABeamAssembler

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.8, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
args = parser.parse_args()

# nxe_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
nxe_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
beam_types = ['eb', 'ts', 'tsr', 'aig', 'hhd', 'higd', 'drig']
# beam_types = ['eb', 'hhr', 'hhd', 'higd']
deflections = {key:[] for key in beam_types}

load_fcn = lambda x : np.sin(4 * np.pi * x)

# m = 4.0
# load_fcn = lambda x: (
#             np.sin(m*np.pi*x)
#             + 0.35*np.sin(3*m*np.pi*x)
#             + 0.2*np.sin(7*m*np.pi*x)
#         ) * np.exp(-4*(x-0.5)**2)

for nxe in nxe_vec:
    thick = args.thick

    for elem in beam_types:
            
        is_iga = False
        if elem == 'eb':
            ELEMENT = EulerBernoulliElement()
        elif elem == 'tsr':
            ELEMENT = TimoshenkoElement(reduced_integrated=True)
        elif elem == 'ts':
            ELEMENT = TimoshenkoElement(reduced_integrated=False)
        elif elem == 'hhr':
            ELEMENT = HierarchicRotHermiteElement(reduced_integrated=False)
        elif elem == 'hhd':
            ELEMENT = HierarchicDispHermiteElement(reduced_integrated=False)
        elif elem == 'higd':
            ELEMENT = HierarchicIsogeometricDispElement(reduced_integrated=False)
            is_iga = True
        elif elem == 'aig':
            # ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=True)
            ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=False)
            is_iga = True
        elif elem == 'drig':
            # derham isogeometric element, special vertex-edge style IGA 2nd order basis that is auto locking-free!
            ELEMENT = DeRhamIsogeometricElement()
            is_iga = True

        # ================================
        # make beam assembler
        # ================================

        # clamped = True
        clamped = False # simply supported
        # ASSEMBLER = IGABeamAssembler if is_iga else StandardBeamAssembler
        ASSEMBLER = IGABeamAssemblerV2 if is_iga else StandardBeamAssembler
        if elem == 'drig':
            ASSEMBLER = DeRhamIGABeamAssembler

        assembler = ASSEMBLER(
            ELEMENT=ELEMENT,
            nxe=nxe,
            thick=thick,
            clamped=clamped,
            split_disp_bc=elem in ['hhd', 'higd'],
            # load_fcn=lambda x : np.sin(m * np.pi * x) # need non-standard load fcn to get mesh conv behavior
            load_fcn=load_fcn
        )

        # ================================
        # make multigrid object (optional)
        # ================================

        # use_mg = True
        # use_mg = False
        use_mg = nxe > args.nxemin and not(elem in ['ts', 'tsr', 'aig'])

        if use_mg:

            # make V-cycle solver
            nxe_min = args.nxemin
            grids = []
            smoothers = []
            # double_smooth = True
            double_smooth = False

            nsmooth = args.nsmooth
            _nxe = nxe
            while (_nxe >= nxe_min):
                # print(f"{nxe=}")
                grid = ASSEMBLER(
                    ELEMENT=ELEMENT,
                    nxe=_nxe,
                    thick=thick,
                    clamped=clamped,
                    split_disp_bc=elem in ['hhd', 'higd'],
                    # load_fcn=lambda x : np.sin(m * np.pi * x) # need non-standard load fcn to get mesh conv behavior
                    load_fcn = load_fcn
                )
                grid._assemble_system()
                grids += [grid]
                if args.smoother == 'gs':
                    smoother = BlockGaussSeidel.from_assembler(
                        grid, omega=args.omega, iters=nsmooth
                    )
                elif args.smoother == 'asw':
                    omega = args.omega / 2.0 # since 2x smoothing
                    if elem == 'drig': # needs custom schwarz smoother with vertex-edge based
                        smoother = OneDimAddSchwarzVertex2Edges.from_assembler(
                            grid, omega=omega, iters=nsmooth,
                            # patch_type="vertex2edges",
                            patch_type="2nodes3edges", # slightly better conv
                        )
                    else:
                        smoother = OnedimAddSchwarz.from_assembler(
                            grid, omega=omega, iters=nsmooth, coupled_size=2
                        )
                smoothers += [smoother]
                _nxe = _nxe // 2
                if double_smooth:
                    nsmooth *= 2

            # run MG analysis
            assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                                #  line_search=not(args.elem == 'aig'))
                                                # line search sometimes hurts high cond # cases (high defects in prolong)
                                                # line_search=not(elem in ['aig', 'tsr', 'hhd', 'higd']), 
                                                line_search=False,
                                                # line_search=True,
                                                smoothers=smoothers, rtol=1e-9)
            
        else: # not use mg
            assembler.direct_solve()

        # predicted disp
        u = assembler.u.copy()
        if elem in ['hhd']:
            # hierarchic disp has shear + bending split
            w = u[0::3] + u[2::3]
        elif elem in ['higd']:
            w = u[0::2] + u[1::2]
        elif elem in ['drig']:
            w = u[:assembler.nw]
        else:
            w = u[0::assembler.dof_per_node]
        pred_disp = np.max(w)

        deflections[elem] += [pred_disp]

# ==================================
# now plot each runtimes
# ==================================

import niceplots
plt.style.use(niceplots.get_style())

fig, ax = plt.subplots()

# colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "k", 'tab:gray']

for ibeam, beam in enumerate(beam_types):
    h_vec = 1.0 / np.array(nxe_vec)
    deflns = np.array(deflections[beam])
    rel_err = np.abs((np.array(deflns[:-1]) - deflns[-1]) / deflns[-1])
    print(f"{deflns=}\n{rel_err=}")

    ax.plot(h_vec[:-1], rel_err, 'o-' if not(beam == 'hhd') else 'o--', 
             color=colors[ibeam], label=beam,
             linewidth=2.5, markersize=7)
    
# plt.xlabel("Slenderness")
plt.xlabel("h (m)")
plt.xscale('log')
plt.ylabel(f"Defln L2 rel err")
plt.yscale('log')
# plt.ylabel("Runtime (sec)")
# plt.yscale('log')
plt.legend()
# plt.show()
plt.margins(x=0.05, y=0.05)
# plt.axis([0.5, 2e3, -1, 30])
# plt.axis([1.0/2e3, 2.0, -1, 50])

# fix xtick labels
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)


# yticks = ax.get_yticks()
# ylabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in yticks]
# ax.set_yticklabels(ylabels)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    fontsize=12   # ↓ shrink legend font only
)
plt.tight_layout()

# plt.savefig("hybrid-beam-multigrid-times.png", dpi=400)
plt.savefig(f"beam-multigrid-conv.png", dpi=400)