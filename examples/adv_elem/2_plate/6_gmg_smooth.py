"""
Plot # of K-cycles (total V-cycles inside KMG PCG) vs prolongator smoothing steps (nprolong) for:
  - mitc_ep : varies with nprolong (x-axis)
  - mitc_lp : varies with nprolong (x-axis)
  - mitc_gp : constant (dashed)  [nprolong does nothing]
  - mitc    : constant (dashed)  [nprolong does nothing]

This script shells out to your driver (DRIVER) and scrapes "total_vcyc=".

It prints a tabbed one-liner AFTER each run:
    <elem>\t nprolong=<p>\t nsmooth=<s>\t total_vcyc=<k>

NOTE:
- We keep MG smoother steps fixed at --nsmooth (default 2) for ALL runs.
- We sweep --nprolong ONLY for mitc_lp and mitc_ep.
- Plot is linear y-axis with a cap (default 200).
  Values above the cap are clipped; clipped points are marked with 'x' at the cap.
  If MITC or MITC-GP are clipped, we annotate the true value with text.
"""

import re
import sys
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

DRIVER = "1_gmg.py"  # your big script


def run_one(elem: str, nprolong: int, args) -> int:
    cmd = [
        sys.executable, DRIVER,
        "--solve", "kmg",
        "--elem", elem,
        "--nxe", str(args.nxe),
        "--nxemin", str(args.nxemin),
        "--thick", str(args.thick),
        "--coupled", str(args.coupled),
        "--omega", str(args.omega),
        "--smoother", str(args.smoother),
        "--nsmooth", str(args.nsmooth),         # MG smoother steps (fixed)
        "--nprolong", str(int(nprolong)),       # prolongator sweeps (varies only for ep/lp)
    ]
    if args.debug:
        cmd += ["--debug"]

    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")

    m = re.search(r"total_vcyc\s*=\s*(\d+)", out)
    if m is None:
        raise RuntimeError(
            f"Failed to parse total_vcyc for elem={elem}, nprolong={nprolong}.\n"
            f"Command:\n  {' '.join(cmd)}\n\nOutput tail:\n{out[-2000:]}"
        )
    total_vcyc = int(m.group(1))

    print(f"{elem}\t nprolong={int(nprolong)}\t nsmooth={args.nsmooth}\t total_vcyc={total_vcyc}")
    return total_vcyc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nxe", type=int, default=32)
    parser.add_argument("--nxemin", type=int, default=8)
    parser.add_argument("--thick", type=float, default=1e-3)
    parser.add_argument("--coupled", type=int, default=3)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--smoother", type=str, default="supp_asw")
    parser.add_argument("--nsmooth", type=int, default=2, help="MG smoother steps (fixed for all runs)")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--nprolong_list",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,20",
        help="comma-separated nprolong sweep for mitc_ep and mitc_lp",
    )
    parser.add_argument(
        "--nprolong_const",
        type=int,
        default=10,
        help="single nprolong value used when running mitc and mitc_gp (doesn't matter for them)",
    )

    parser.add_argument("--title", type=str, default="")  # intentionally unused in plot
    parser.add_argument(
        "--save",
        type=str,
        default="out/kcycles_vs_nprolong.svg",
        help="output figure filename (default: kcycles_vs_nprolong.svg). Set to '' to skip saving.",
    )
    parser.add_argument(
        "--ycap",
        type=float,
        default=200.0,
        help="cap y-axis at this value (values above are clipped and annotated)",
    )

    args = parser.parse_args()
    ycap = float(args.ycap)

    p_list = np.array([int(s) for s in args.nprolong_list.split(",") if s.strip()], dtype=int)

    print("elem\t nprolong\t nsmooth\t total_vcyc")
    print("-" * 40)

    # varying curves (sweep nprolong)
    curves = {}
    for e in ["mitc_ep", "mitc_lp"]:
        ys = []
        for p in p_list:
            ys.append(run_one(e, int(p), args))
        curves[e] = np.array(ys, dtype=float)

    # constant lines (run once; nprolong irrelevant)
    const = {}
    for e in ["mitc_gp", "mitc"]:
        const[e] = float(run_one(e, int(args.nprolong_const), args))

    # ----------------
    # plot (nicer style)
    # ----------------
    try:
        import niceplots
        plt.style.use(niceplots.get_style())
    except Exception:
        pass

    fs = 20
    ms = 7
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": fs,
    })

    # your preferred palette (4 colors)
    colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
    c_gp, c_ep, c_lp, c_std = colors[0], colors[1], colors[2], colors[3]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # ----- clip for plotting (no log scale) -----
    lp_raw = curves["mitc_lp"]
    ep_raw = curves["mitc_ep"]
    gp_raw = const["mitc_gp"]
    std_raw = const["mitc"]

    lp_plot = np.minimum(lp_raw, ycap)
    ep_plot = np.minimum(ep_raw, ycap)
    gp_plot = min(gp_raw, ycap)
    std_plot = min(std_raw, ycap)

    # Slightly extend ylim so the cap line doesn't hide under the top boundary
    ax.set_ylim(0.0, ycap * 1.03)

    # main curves
    ax.plot(p_list, lp_plot, "o-", linewidth=3, markersize=ms, color=c_lp, label="MITC-LP")
    ax.plot(p_list, ep_plot, "o-", linewidth=3, markersize=ms, color=c_ep, label="MITC-EP")

    # constant dashed lines (draw above everything)
    ax.hlines(gp_plot, p_list.min(), p_list.max(), colors=c_gp, linestyles="--", linewidth=3, label="MITC-GP", zorder=3)
    ax.hlines(std_plot, p_list.min(), p_list.max(), colors=c_std, linestyles="--", linewidth=3, label="MITC", zorder=3)

    # explicit cap reference line (so it's obvious where clipping happens)
    ax.hlines(ycap, p_list.min(), p_list.max(),
              colors="k", linestyles=":", linewidth=2, alpha=0.6, zorder=2)

    # mark clipped points with 'x' at the cap (LP/EP)
    lp_over = lp_raw > ycap
    ep_over = ep_raw > ycap
    if np.any(lp_over):
        ax.plot(p_list[lp_over], np.full(np.sum(lp_over), ycap), "x",
                color=c_lp, markersize=ms+2, mew=2, zorder=4)
    if np.any(ep_over):
        ax.plot(p_list[ep_over], np.full(np.sum(ep_over), ycap), "x",
                color=c_ep, markersize=ms+2, mew=2, zorder=4)

    # If constant lines are clipped, show the true value with text + an x marker
    x_anno = p_list.max()
    if std_raw > ycap:
        ax.plot([x_anno], [ycap], "x", color=c_std, markersize=ms+2, mew=2, zorder=5)
        ax.text(x_anno, ycap * 1.01, f"{int(std_raw)}",
                color=c_std, fontsize=fs-4, fontweight="bold",
                ha="right", va="bottom")
    if gp_raw > ycap:
        ax.plot([x_anno], [ycap], "x", color=c_gp, markersize=ms+2, mew=2, zorder=5)
        ax.text(x_anno, ycap * 1.01, f"{int(gp_raw)}",
                color=c_gp, fontsize=fs-4, fontweight="bold",
                ha="right", va="bottom")

    # axis labels (your preference)
    ax.set_xlabel("# smooth prolongator steps")
    ax.set_ylabel("# K(2,2)-ASW cycles")
    ax.set_yscale("linear")

    # tidy y ticks (include the cap explicitly)
    if ycap <= 120:
        yt = [0, 20, 40, 60, 80, 100, ycap]
    else:
        step = 50
        yt = list(np.arange(0, ycap + 1e-9, step))
        if yt[-1] != ycap:
            yt += [ycap]
    ax.set_yticks(yt)

    # cleaner look: no gridlines, remove top/right spines
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend OUTSIDE the axes (no overlap)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # give space for outside legend
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    plt.margins(x=0.05, y=0.05)

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"saved: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
