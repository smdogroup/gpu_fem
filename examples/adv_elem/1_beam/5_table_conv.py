import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")

from std_assembler import StandardBeamAssembler
from elem import (
    EulerBernoulliElement,
    TimoshenkoElement,
    HierarchicRotHermiteElement,
    HierarchicDispHermiteElement,
    AsymptoticIsogeometricTimoshenkoElement,
    HierarchicIsogeometricDispElement,
    DeRhamIsogeometricElement,
    AlgebraicSubGridScaleElement,
    HellingerReissnerAnsatzElement,
)

from smoothers import BlockGaussSeidel, OnedimAddSchwarz, OneDimAddSchwarzVertex2Edges
from multigrid2 import vcycle_solve

from iga_assembler import IGABeamAssembler
from iga_assembler2 import IGABeamAssemblerV2
from diga_assembler import DeRhamIGABeamAssembler

import argparse


# ----------------------------
# mesh convergence-rate helpers
# ----------------------------
def mesh_rate(eps1, eps2, h1, h2):
    """
    Rc = log(eps1/eps2) / log(h2/h1)
    Returns np.nan if invalid.
    """
    eps1 = float(eps1)
    eps2 = float(eps2)
    h1 = float(h1)
    h2 = float(h2)

    if not np.isfinite(eps1) or not np.isfinite(eps2):
        return np.nan
    if eps1 <= 0.0 or eps2 <= 0.0:
        return np.nan
    if h1 <= 0.0 or h2 <= 0.0 or h1 == h2:
        return np.nan

    return np.log(eps1 / eps2) / np.log(h2 / h1)


# def compute_Rc(eps_list, h_list, mode="last2"):
#     """
#     mode:
#       - "last2": use last two (finest two) mesh levels
#       - "coarse_to_fine": use first and last (coarsest vs finest)
#     """
#     if len(eps_list) < 2 or len(h_list) < 2 or len(eps_list) != len(h_list):
#         return np.nan

#     if mode == "coarse_to_fine":
#         i1, i2 = 0, -1
#     else:
#         i1, i2 = -2, -1

#     return mesh_rate(eps_list[i1], eps_list[i2], h_list[i1], h_list[i2])

def compute_Rc(eps_list, h_list, mode="overall"):
    """
    Compute mesh convergence rate Rc.

    modes:
      - "overall"         : least-squares slope of log(eps) vs log(h)
      - "coarse_to_fine" : first vs last
      - "last2"          : last two (finest two)
    """
    eps = np.asarray(eps_list, dtype=float)
    h   = np.asarray(h_list,   dtype=float)

    if len(eps) < 2 or len(h) < 2 or len(eps) != len(h):
        return np.nan

    # remove invalid points
    mask = (
        np.isfinite(eps) &
        np.isfinite(h) &
        (eps > 0.0) &
        (h > 0.0)
    )

    eps = eps[mask]
    h   = h[mask]

    if len(eps) < 2:
        return np.nan

    if mode == "coarse_to_fine":
        return mesh_rate(eps[0], eps[-1], h[0], h[-1])

    if mode == "last2":
        return mesh_rate(eps[-2], eps[-1], h[-2], h[-1])

    # ------------------------
    # OVERALL regression slope
    # ------------------------
    log_eps = np.log(eps)
    log_h   = np.log(h)

    # slope of least-squares line
    Rc, _ = np.polyfit(log_h, log_eps, 1)

    return Rc


# ----------------------------
# args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--thick", type=float, default=1e-3, help="beam thickness")
parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
# heftier and more robust MG iterations cause need to fully converge (otherwise will be misleading)
parser.add_argument("--nsmooth", type=int, default=4, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff")
parser.add_argument("--smoother", type=str, default="asw", help="--smoother : [gs, asw]")
parser.add_argument(
    "--Rc_mode",
    type=str,
    default="overall",
    choices=["last2", "coarse_to_fine", "overall"],
    help="How to compute mesh-rate Rc: last2 (finest two) or coarse_to_fine (first vs last) or 'overall'.",
)
args = parser.parse_args()

# ----------------------------
# problem setup
# ----------------------------
nxe_vec = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# beam_types = ["eb", "ts", "tsr", "aig", "hhr", "hhd", "higd", "drig", "asgs", "hra"]
beam_types = ['hra']
deflections = {key: [] for key in beam_types}

load_fcn = lambda x: np.sin(4 * np.pi * x)


def build_element(elem):
    is_iga = False
    if elem == "eb":
        ELEMENT = EulerBernoulliElement()
    elif elem == "tsr":
        ELEMENT = TimoshenkoElement(reduced_integrated=True)
    elif elem == "ts":
        ELEMENT = TimoshenkoElement(reduced_integrated=False)
    elif elem == "hhr":
        ELEMENT = HierarchicRotHermiteElement(reduced_integrated=False)
    elif elem == "hhd":
        ELEMENT = HierarchicDispHermiteElement(reduced_integrated=False)
    elif elem == "higd":
        ELEMENT = HierarchicIsogeometricDispElement(reduced_integrated=False)
        is_iga = True
    elif elem == "aig":
        ELEMENT = AsymptoticIsogeometricTimoshenkoElement(reduced_integrated=False)
        is_iga = True
    elif elem == "drig":
        ELEMENT = DeRhamIsogeometricElement()
        is_iga = True
    elif elem == 'asgs':
        ELEMENT = AlgebraicSubGridScaleElement()
    elif elem == 'hra':
        ELEMENT = HellingerReissnerAnsatzElement(
            schur_complement=True, # static condensation at element-level
            hra_method='strain-int', # has a discontinuous (lower-order polynomial strain DOF to remove locking that is eliminated from element Schur-complement)
        )
    else:
        raise ValueError(f"Unknown elem type: {elem}")
    return ELEMENT, is_iga


def build_assembler(elem, is_iga):
    # (keeping your original choice: V2 for IGA, Standard for non-IGA)
    ASSEMBLER = IGABeamAssemblerV2 if is_iga else StandardBeamAssembler
    if elem == "drig":
        ASSEMBLER = DeRhamIGABeamAssembler
    return ASSEMBLER


# ----------------------------
# run solves across meshes
# ----------------------------
for nxe in nxe_vec:
    thick = args.thick

    for elem in beam_types:
        ELEMENT, is_iga = build_element(elem)

        clamped = False  # simply supported
        ASSEMBLER = build_assembler(elem, is_iga)

        assembler = ASSEMBLER(
            ELEMENT=ELEMENT,
            nxe=nxe,
            thick=thick,
            clamped=clamped,
            split_disp_bc=elem in ["hhd", "higd"],
            load_fcn=load_fcn,
        )

        # Use MG only on bigger problems and not for the locking-prone TS family (your original logic)
        # use_mg = (nxe > args.nxemin) and not (elem in ["ts", "tsr", "aig"])
        use_mg = False # for time being, DEBUG

        if use_mg:
            nxe_min = args.nxemin
            grids = []
            smoothers = []
            double_smooth = False

            nsmooth = args.nsmooth
            _nxe = nxe
            while _nxe >= nxe_min:
                grid = ASSEMBLER(
                    ELEMENT=ELEMENT,
                    nxe=_nxe,
                    thick=thick,
                    clamped=clamped,
                    split_disp_bc=elem in ["hhd", "higd"],
                    load_fcn=load_fcn,
                )
                grid._assemble_system()
                grids.append(grid)

                if args.smoother == "gs":
                    smoother = BlockGaussSeidel.from_assembler(
                        grid, omega=args.omega, iters=nsmooth
                    )
                elif args.smoother == "asw":
                    omega = args.omega / 2.0  # since 2x smoothing
                    if elem == "drig":
                        smoother = OneDimAddSchwarzVertex2Edges.from_assembler(
                            grid,
                            omega=omega,
                            iters=nsmooth,
                            patch_type="2nodes3edges",
                        )
                    else:
                        smoother = OnedimAddSchwarz.from_assembler(
                            grid, omega=omega, iters=nsmooth, coupled_size=2
                        )
                else:
                    raise ValueError(f"Unknown smoother: {args.smoother}")

                smoothers.append(smoother)

                _nxe //= 2
                if double_smooth:
                    nsmooth *= 2

            assembler.u, ncyc = vcycle_solve(
                grids,
                pre_smooth=args.nsmooth,
                post_smooth=args.nsmooth,
                line_search=False,
                smoothers=smoothers,
                rtol=1e-9,
            )
        else:
            assembler.direct_solve()

        # predicted deflection (your original extraction logic)
        u = assembler.u.copy()
        if elem in ["hhd"]:
            w = u[0::3] + u[2::3]  # shear + bending split
        elif elem in ["higd"]:
            w = u[0::2] + u[1::2]
        elif elem in ["drig"]:
            w = u[: assembler.nw]
        else:
            w = u[0:: assembler.dof_per_node]

        pred_disp = np.max(w)
        deflections[elem].append(pred_disp)
# ----------------------------
# compute rel error vs finest + Rc and print (MINIMAL)
# ----------------------------
h_vec = 1.0 / np.array(nxe_vec)

print("\n================ Mesh convergence rates Rc (per element) ================")
print(f"thick = {args.thick:.6e}")
print(f"Rc_mode = {args.Rc_mode}")
print("--------------------------------------------------------------------------")

rel_err_by_elem = {}
Rc_by_elem = {}

for beam in beam_types:
    deflns = np.array(deflections[beam], dtype=float)

    # rel error vs finest deflection (exclude finest point, like your plot)
    rel_err = np.abs((deflns[:-1] - deflns[-1]) / deflns[-1])
    rel_err_by_elem[beam] = rel_err

    # mesh-rate Rc using rel_err as epsilon
    Rc = compute_Rc(rel_err.tolist(), h_vec[:-1].tolist(), mode=args.Rc_mode)
    Rc_by_elem[beam] = Rc

    print(f"{beam:>4s}: Rc = {Rc: .6f}")


# ----------------------------
# plot rel error vs h
# ----------------------------
import niceplots
plt.style.use(niceplots.get_style())

fig, ax = plt.subplots()

colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "k", "tab:gray"]

for ibeam, beam in enumerate(beam_types):
    ax.plot(
        h_vec[:-1],
        rel_err_by_elem[beam],
        "o-" if beam != "hhd" else "o--",
        color=colors[ibeam],
        label=beam,
        linewidth=2.5,
        markersize=7,
    )

plt.xlabel("h (m)")
plt.xscale("log")
plt.ylabel("Defln L2 rel err")
plt.yscale("log")

plt.margins(x=0.05, y=0.05)

# fix xtick labels (keep your style)
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    fontsize=12,
)
plt.tight_layout()

plt.savefig("beam-multigrid-conv.png", dpi=400)
