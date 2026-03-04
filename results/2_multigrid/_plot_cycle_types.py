#!/usr/bin/env python3
"""
Plot GMG V/W/F-cycle visit patterns as 1x3 subplots for 4 mesh levels.

Levels:
0 = finest (top)
3 = coarsest (bottom)

Coarsest level is visited only once.
"""

import matplotlib.pyplot as plt
import niceplots


def v_cycle(level: int):
    if level == 0:
        return [0]
    seq = v_cycle(level - 1)
    return [level] + seq + [level]


def w_cycle(level: int):
    if level == 0:
        return [0]

    seq1 = w_cycle(level - 1)
    seq2 = w_cycle(level - 1)

    # remove duplicate join point between the two recursive calls
    seq = seq1 + seq2[1:]

    return [level] + seq + [level]


def f_cycle(level: int):
    if level == 0:
        return [0]

    seqF = f_cycle(level - 1)
    seqV = v_cycle(level - 1)

    # remove duplicate join point between F and V calls
    seq = seqF + seqV[1:]

    return [level] + seq + [level]


# def plot_cycle(ax, seq, title, n_levels):
#     xs = list(range(len(seq)))
#     ys = seq

#     ax.plot(xs, ys, marker="o", markersize=4, linewidth=2)
#     ax.set_title(title)
#     ax.set_xlabel("Step")

#     ax.set_yticks(range(n_levels))
#     ax.set_yticklabels([str(i) for i in range(n_levels)])

#     # ---- IMPORTANT: flip so 0 is at the top ----
#     ax.set_ylim(n_levels - 0.8, -0.2)

#     ax.grid(True, alpha=0.3)

# def plot_cycle(ax, seq, title, n_levels):

#     restrict_color = "#1f77b4"
#     prolong_color = "#d95f02"

#     # draw colored segments
#     for i in range(len(seq) - 1):

#         x0, x1 = i, i + 1
#         y0, y1 = seq[i], seq[i + 1]

#         if y1 > y0:
#             color = restrict_color   # going to coarser grid
#         elif y1 < y0:
#             color = prolong_color    # going to finer grid
#         else:
#             color = "black"

#         ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.4)

#     # draw node markers
#     ax.plot(range(len(seq)), seq, "o", color="black", markersize=6)

#     # mark coarse solve
#     coarse_level = max(seq)
#     coarse_indices = [i for i, v in enumerate(seq) if v == coarse_level]

#     ax.plot(
#         coarse_indices,
#         [coarse_level] * len(coarse_indices),
#         "s",
#         markersize=7,
#         color="black",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Step")

#     ax.set_yticks(range(n_levels))
#     ax.set_yticklabels([str(i) for i in range(n_levels)])

#     ax.set_ylim(n_levels - 0.8, -0.2)

#     ax.grid(True, alpha=0.3)

from matplotlib.lines import Line2D
def plot_cycle(ax, seq, title, n_levels, legend=False):

    restrict_color = "#1f77b4"   # blue
    prolong_color  = "#d95f02"   # orange
    coarse_color   = "#1b9e77"

    marker_size = 8

    # draw colored segments
    for i in range(len(seq) - 1):

        x0, x1 = i, i + 1
        y0, y1 = seq[i], seq[i + 1]

        if y1 > y0:
            color = restrict_color
        elif y1 < y0:
            color = prolong_color
        else:
            color = "black"

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.6)

    # draw node markers
    ax.plot(range(len(seq)), seq, "o", color="black", markersize=marker_size)

    # mark coarse solve
    coarse_level = max(seq)
    coarse_indices = [i for i, v in enumerate(seq) if v == coarse_level]

    ax.plot(
        coarse_indices,
        [coarse_level] * len(coarse_indices),
        "s",
        markersize=marker_size,
        color=coarse_color,
    )

    ax.set_title(title)

    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([str(i) for i in range(n_levels)])

    ax.set_ylim(n_levels - 0.8, -0.2)

    # remove x-axis ticks/labels (no meaning)
    ax.set_xticks([])
    ax.set_xlabel(None)

    # margins so markers aren't clipped
    ax.margins(x=0.05, y=0.05)

    ax.grid(True, alpha=0.3)

    if legend:
        legend_elements = [
            Line2D([0], [0], color=restrict_color, lw=2.6, label="Restriction"),
            Line2D([0], [0], color=prolong_color, lw=2.6, label="Prolongation"),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markerfacecolor=coarse_color,
                markeredgecolor=coarse_color,
                markersize=marker_size,
                label="Coarse direct solve",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper center",
            frameon=True,
            fontsize=13,
            handlelength=1.6,
            borderpad=0.4,
            labelspacing=0.3,
        )


def main():
    plt.style.use(niceplots.get_style())

    n_levels = 4
    coarsest = n_levels - 1

    seq_V = v_cycle(coarsest)
    seq_W = w_cycle(coarsest)
    seq_F = f_cycle(coarsest)

    # reflect sequence..
    seq_V = [3-_ for _ in seq_V]
    seq_W = [3-_ for _ in seq_W]
    seq_F = [3-_ for _ in seq_F]


    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    plot_cycle(axes[0], seq_V, "V-cycle", n_levels, legend=True)
    plot_cycle(axes[1], seq_W, "W-cycle", n_levels)
    plot_cycle(axes[2], seq_F, "F-cycle", n_levels)

    axes[0].set_ylabel("Mesh level")

    fig.tight_layout()
    fig.savefig("gmg_cycles_4levels.svg", bbox_inches="tight")
    fig.savefig("gmg_cycles_4levels.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()