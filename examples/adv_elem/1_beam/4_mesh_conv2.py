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
from elem import AlgebraicSubGridScaleElement


# IGA elements
from elem import AsymptoticIsogeometricTimoshenkoElement, HierarchicIsogeometricDispElement
# from iga_assembler import IGABeamAssembler
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

# nxe_vec = [4, 8, 16, 32, 64, 128]
# nxe_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
nxe_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# beam_types = ['eb', 'tsr', 'hhd', 'higd']
# beam_types = ['eb', 'aig', 'ts', 'tsr', 'hhr', 'hhd', 'higd']
beam_types = ['eb', 'ts', 'tsr', 'aig', 'hhr', 'hhd', 'higd', 'drig']
deflections = {key:[] for key in beam_types}

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
        elif args.elem == 'asgs':
            ELEMENT = AlgebraicSubGridScaleElement()

        # ================================
        # make beam assembler
        # ================================

        # m = 1.0
        m = 4.0

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
            load_fcn=lambda x : np.sin(m * np.pi * x) # need non-standard load fcn to get mesh conv behavior
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

            nsmooth = args.nsmooth if not(elem in ['tsr', 'ts', 'aig']) else 4
            _nxe = nxe
            while (_nxe >= nxe_min):
                # print(f"{nxe=}")
                grid = ASSEMBLER(
                    ELEMENT=ELEMENT,
                    nxe=_nxe,
                    thick=thick,
                    clamped=clamped,
                    split_disp_bc=elem in ['hhd', 'higd'],
                    load_fcn=lambda x : np.sin(m * np.pi * x) # need non-standard load fcn to get mesh conv behavior
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
        if pred_disp >= 1.0e0:
           pred_disp = np.nan 
        elif pred_disp <= 1e-12:
            pred_disp = np.nan
            
        if is_iga and not(elem == 'drig'):
            # trying to see if this makes a difference in IGA conv
            pred_disp = assembler.get_max_deflection_greville()

        deflections[elem] += [pred_disp]
# ==================================
# plot max displacement vs nxe
# ==================================
import niceplots
plt.style.use(niceplots.get_style())

fig, ax = plt.subplots()

# colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "k", 'tab:gray']

nxe_arr = np.array(nxe_vec, dtype=int)

for ibeam, beam in enumerate(beam_types):
    deflns = np.array(deflections[beam], dtype=float)

    # "converged" reference: finest mesh value
    if ibeam == 0:
        ax.hlines(deflns[-1], nxe_arr[0], nxe_arr[-1],
                  linestyles='--', linewidth=1.5, color='k', alpha=1.0)

    # main curve: max displacement vs nxe
    ax.plot(
        nxe_arr, deflns,
        'o-' if beam != 'hhd' else 'o--',
        color=colors[ibeam], label=beam,
        linewidth=2.5, markersize=7
    )
   
              

ax.set_xlabel("nxe")
ax.set_xscale("log")
ax.set_ylabel("max displacement (predicted)")
# ax.set_yscale("log")  # optional; comment out if you prefer linear y

ax.legend()
plt.margins(x=0.05, y=0.05)

# nice tick labels for log-x
xticks = ax.get_xticks()
xlabels = []
for x in xticks:
    if x <= 0:
        xlabels.append("")
    else:
        # show powers of 2-ish cleanly if you want, otherwise generic 10^k
        xlabels.append(f"$10^{{{int(np.log10(x))}}}$")
ax.set_xticklabels(xlabels)

# optional: y tick formatting (same style you used)
# yticks = ax.get_yticks()
# ylabels = []
# for y in yticks:
#     if y <= 0:
#         ylabels.append("")
#     else:
#         ylabels.append(f"$10^{{{int(np.log10(y))}}}$")
# ax.set_yticklabels(ylabels)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    fontsize=12   # ↓ shrink legend font only
)
plt.tight_layout()

plt.savefig("beam-maxdisp-vs-nxe.png", dpi=400)
