#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_grid(ax, nxe, step, color, linewidth=1.0, linestyle="-"):
    for i in range(0, nxe + 1, step):
        ax.plot([i, i], [0, nxe], color=color, lw=linewidth, ls=linestyle)
        ax.plot([0, nxe], [i, i], color=color, lw=linewidth, ls=linestyle)


def interior_nodes(nxe, step):
    # Deliberately excludes the physical-boundary coordinates 0 and nxe.
    coordinates = np.arange(step, nxe, step)
    x, y = np.meshgrid(coordinates, coordinates)
    return x.ravel(), y.ravel()


def format_axis(ax, nxe, title):
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, nxe + 0.4)
    ax.set_ylim(-0.4, nxe + 0.4)
    ax.set_xticks(range(0, nxe + 1, 2))
    ax.set_yticks(range(0, nxe + 1, 2))
    ax.set_xlabel("element index, x")
    ax.set_ylabel("element index, y")
    ax.set_title(title)


def make_plot(nxe):
    if nxe <= 0 or nxe % 4 != 0:
        raise ValueError("nxe must be a positive multiple of 4")

    # Level-1 Vc nodes: interior corners of 2x2-element subdomains.
    level1_x, level1_y = interior_nodes(nxe, step=2)

    # Level-2 Vc nodes: interior corners of 2x2 groups of level-1 subdomains.
    level2_x, level2_y = interior_nodes(nxe, step=4)

    # EXACTLY TWO AXES.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # axes[0]: fine elements, first-level splitting, and level-1 Vc nodes.
    draw_grid(axes[0], nxe, 1, "0.84", linewidth=0.6)
    draw_grid(axes[0], nxe, 2, "#1976d2", linewidth=1.8, linestyle="--")
    axes[0].scatter(
        level1_x, level1_y, s=48, color="#00a6d6", edgecolor="white",
        linewidth=0.7, zorder=3, label="level-1 interior $V_c$ nodes"
    )
    format_axis(axes[0], nxe, f"Level 0: {nxe}x{nxe} fine elements")
    axes[0].legend(loc="lower right")

    # axes[1]: level-1 coarse-node problem with the fine grid faintly shown.
    draw_grid(axes[1], nxe, 1, "#e4e9eb", linewidth=0.45)
    draw_grid(axes[1], nxe, 2, "#90a4ae", linewidth=1.0)
    draw_grid(axes[1], nxe, 4, "#8e24aa", linewidth=1.9, linestyle="--")
    axes[1].scatter(
        level1_x, level1_y, s=42, color="#00a6d6", edgecolor="white",
        linewidth=0.7, zorder=3, label="level-1 fine/coarse nodes"
    )
    axes[1].scatter(
        level2_x, level2_y, s=135, facecolor="none", edgecolor="#ef6c00",
        linewidth=2.3, zorder=4, label="level-2 interior $V_c$ nodes"
    )
    format_axis(axes[1], nxe, "Level 1: coarse-node problem")
    axes[1].legend(loc="lower right")

    fig.suptitle("Three-level structured multilevel BDDC plate")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nxe", type=int, default=12)
    parser.add_argument("--output", default="multilevel_bddc_plate_two_axes.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fig = make_plot(args.nxe)
    output = Path(args.output)
    fig.savefig(output, dpi=220, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
