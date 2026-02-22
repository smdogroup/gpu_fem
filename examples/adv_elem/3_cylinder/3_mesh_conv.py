import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/")

from elem import DeRhamIsogeometricCylinderElement
from drig_assembler import DeRhamIGACylinderAssembler

import argparse


# ----------------------------
# args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--thick", type=float, default=1e-2, help="shell thickness")
parser.add_argument("--radius", type=float, default=1.0, help="cylinder radius")
parser.add_argument("--length", type=float, default=1.0, help="cylinder length")
parser.add_argument("--nxemin", type=int, default=8, help="min # elems multigrid (if/when MG used)")
parser.add_argument("--solve", type=str, default="direct", choices=["direct", "vmg", "kmg"],
                    help="solver to use (direct for convergence study baseline)")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps (if MG used)")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff (if MG used)")
parser.add_argument("--smoother", type=str, default="asw", choices=["gs", "asw"],
                    help="smoother (if MG used)")
parser.add_argument("--clamped", action=argparse.BooleanOptionalAction, default=True,
                    help="use clamped cylinder BCs if True, else simply supported")
parser.add_argument("--arc", type=str, default="quarter", choices=["quarter", "half", "full"],
                    help="open hoop arc: quarter/half/full cylinder (affects hoop_length)")
parser.add_argument(
    "--Rc_mode",
    type=str,
    default="overall",
    choices=["last2", "coarse_to_fine", "overall"],
    help="How to compute mesh-rate Rc.",
)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="make rel-error plot")
args = parser.parse_args()

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


def compute_Rc(eps_list, h_list, mode="overall"):
    """
    Compute mesh convergence rate Rc.

    modes:
      - "overall"         : least-squares slope of log(eps) vs log(h)
      - "coarse_to_fine"  : first vs last
      - "last2"           : last two (finest two)
    """
    eps = np.asarray(eps_list, dtype=float)
    h   = np.asarray(h_list,   dtype=float)

    if len(eps) < 2 or len(h) < 2 or len(eps) != len(h):
        return np.nan

    mask = np.isfinite(eps) & np.isfinite(h) & (eps > 0.0) & (h > 0.0)
    eps = eps[mask]
    h   = h[mask]

    if len(eps) < 2:
        return np.nan

    if mode == "coarse_to_fine":
        return mesh_rate(eps[0], eps[-1], h[0], h[-1])

    if mode == "last2":
        return mesh_rate(eps[-2], eps[-1], h[-2], h[-1])

    # overall regression slope of log(eps) vs log(h)
    log_eps = np.log(eps)
    log_h   = np.log(h)
    Rc, _ = np.polyfit(log_h, log_eps, 1)
    return Rc


# ----------------------------
# problem setup
# ----------------------------
R = args.radius
L = args.length

# choose hoop_length based on arc option
if args.arc == "quarter":
    hoop_length = 0.5 * np.pi * R
elif args.arc == "half":
    hoop_length = 1.0 * np.pi * R
else:
    hoop_length = 2.0 * np.pi * R

# mesh sequence
nxe_vec = [4, 8, 16, 32, 64]   # you can extend to 128 if you want
# nxe_vec = [8, 16, 32, 64, 128]

# cylinder element/assembler (single element type for now)
elem_types = ["drig_cyl"]
deflections = {key: [] for key in elem_types}

# your cylinder load: f(x,s) -> scalar
load_fcn = lambda x, s: 1.0


def build_element(elem):
    if elem == "drig_cyl":
        axial_factor = 0.0
        # axial_factor = 0.3
        ELEMENT = DeRhamIsogeometricCylinderElement(r=R, reduced_integrated=False, axial_factor=axial_factor)
        is_iga = True
    else:
        raise ValueError(f"Unknown elem type: {elem}")
    return ELEMENT, is_iga


def build_assembler(elem, is_iga):
    # for now only DRIG cylinder assembler
    if elem == "drig_cyl":
        return DeRhamIGACylinderAssembler
    raise ValueError(f"Unknown assembler mapping for elem: {elem}")


# ----------------------------
# run solves across meshes
# ----------------------------
for nxe in nxe_vec:
    for elem in elem_types:
        print(f"{elem=} {nxe=}")

        ELEMENT, is_iga = build_element(elem)
        ASSEMBLER = build_assembler(elem, is_iga)

        assembler = ASSEMBLER(
            ELEMENT,
            nxe=nxe,
            E=70e9,
            nu=0.3,
            thick=args.thick,
            length=L,
            hoop_length=hoop_length,
            radius=R,
            load_fcn=load_fcn,
            clamped=args.clamped,
        )

        # solve
        if args.solve == "direct":
            assembler.direct_solve()
        elif args.solve == "vmg":
            assembler.vmg_solve()
        else:
            assembler.kmg_solve()

        # ---- extract predicted deflection (DRIG cylinder: w is first block) ----
        u = assembler.u.copy()
        w = u[: assembler.nw]

        pred_disp = np.max(w)   # keep consistent with your plate study
        deflections[elem].append(pred_disp)


# ----------------------------
# compute rel error vs finest + Rc and print (MINIMAL)
# ----------------------------
# define a scalar mesh size h; for this study use axial spacing ~ L/nxe
h_vec = L / np.array(nxe_vec, dtype=float)

print("\n================ Mesh convergence rates Rc (per element) ================")
print(f"thick   = {args.thick:.6e}")
print(f"R, L    = {R:.6e}, {L:.6e}")
print(f"arc     = {args.arc}  (hoop_length = {hoop_length:.6e})")
print(f"clamped = {args.clamped}")
print(f"Rc_mode = {args.Rc_mode}")
print("--------------------------------------------------------------------------")

rel_err_by_elem = {}
Rc_by_elem = {}

for etype in elem_types:
    deflns = np.array(deflections[etype], dtype=float)

    # rel error vs finest deflection (exclude finest point)
    rel_err = np.abs((deflns[:-1] - deflns[-1]) / deflns[-1])
    rel_err_by_elem[etype] = rel_err

    Rc = compute_Rc(rel_err.tolist(), h_vec[:-1].tolist(), mode=args.Rc_mode)
    Rc_by_elem[etype] = Rc

    print(f"{etype:>8s}: Rc = {Rc: .6f}")


# ----------------------------
# plot rel error vs h
# ----------------------------
if args.plot:
    try:
        import niceplots
        plt.style.use(niceplots.get_style())
    except Exception:
        pass

    fig, ax = plt.subplots()

    for etype in elem_types:
        ax.plot(
            h_vec[:-1],
            rel_err_by_elem[etype],
            "o-",
            linewidth=2.5,
            markersize=7,
            label=etype,
        )

    plt.xlabel("h (axial spacing)")
    plt.xscale("log")
    plt.ylabel("Defln rel err (vs finest)")
    plt.yscale("log")
    plt.margins(x=0.05, y=0.05)

    # optional xtick formatting like your plate script
    xticks = ax.get_xticks()
    xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
    ax.set_xticklabels(xlabels)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    plt.tight_layout()

    out = f"cyl-mesh-conv_{args.arc}_{'clamped' if args.clamped else 'ss'}.png"
    plt.savefig(out, dpi=400)
    print(f"\nSaved plot: {out}")
