#!/usr/bin/env python3
"""
Grouped BAR charts for the solver-performance table.

Default behavior: two side-by-side panels for t = 1e-1 and t = 1e-2.
Special case: if METRIC == "time_per_iter_ms", plot a SINGLE panel (no subplots)
and do NOT label thickness (time/iter is thickness-independent here).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

# -----------------------------
# Colors (your palette)
# -----------------------------
try:
    import ml_buckling as mlb
    six_colors2 = mlb.six_colors2
except Exception:
    six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

colors = six_colors2[::-1]
plt.style.use(niceplots.get_style())

# -----------------------------
# 1) Hard-code the table data
# -----------------------------
rows = []

def add(method, ordering, t, iters, total_time_s):
    rows.append(dict(Method=method, Ordering=ordering, t=t, iter=iters, time_s=total_time_s))

# NOTE : iters is now the # of smoothing steps of ILU(k) not the # Krylov iterations
# so need to multiply # krylov iters * nsmooth => total # smoothing steps
# cause RCM can't run multiple smoothing steps..

# Direct-LU
add("Direct-LU", "AMD", 1e-1, np.nan, 0.643)
add("Direct-LU", "AMD", 1e-2, np.nan, 0.643)

# ILU(0)
add("ILU(0)", "Lexigraphic", 1e-1, 2*67, 0.975)
add("ILU(0)", "AMD",         1e-1, 2*106, 1.098)
add("ILU(0)", "RCM",         1e-1, 2*70, 0.790)
add("ILU(0)", "Q-order",     1e-1, 2*121, 0.480)

add("ILU(0)", "Lexigraphic", 1e-2, 2*176, 2.67)
add("ILU(0)", "AMD",         1e-2, 2*260, 3.048)
add("ILU(0)", "RCM",         1e-2, 1403, np.nan)
add("ILU(0)", "Q-order",     1e-2, 2*304, 1.698)

# ILU(1)
add("ILU(1)", "Lexigraphic", 1e-1, 2*55, 0.908)
add("ILU(1)", "AMD",         1e-1, 2*64, 0.720)
add("ILU(1)", "RCM",         1e-1, 73, 0.654)
add("ILU(1)", "Q-order",     1e-1, 2*77, 0.361)

add("ILU(1)", "Lexigraphic", 1e-2, 2*112, 1.87)
add("ILU(1)", "AMD",         1e-2, 2*129, 1.462)
add("ILU(1)", "RCM",         1e-2, 168, 1.605)
add("ILU(1)", "Q-order",     1e-2, 2*159, 0.804)

# ILU(2)
add("ILU(2)", "Lexigraphic", 1e-1, 2*43, 0.837)
add("ILU(2)", "AMD",         1e-1, 2*57, 0.692)
add("ILU(2)", "RCM",         1e-1, 56, 0.663)
add("ILU(2)", "Q-order",     1e-1, 2*56, 0.321)

add("ILU(2)", "Lexigraphic", 1e-2, 2*76, 1.416)
add("ILU(2)", "AMD",         1e-2, 2*105, 1.255)
add("ILU(2)", "RCM",         1e-2, 103, 1.214)
add("ILU(2)", "Q-order",     1e-2, 2*104, 0.596)

df = pd.DataFrame(rows)

# compute time/iter (ms)
df["time_per_iter_ms"] = np.where(df["iter"].notna(), 1000.0 * df["time_s"] / df["iter"], np.nan)

# -----------------------------
# 2) Choose what to plot
# -----------------------------
# Options: "time_s", "iter", "time_per_iter_ms"
METRIC = "time_per_iter_ms"
# METRIC = "time_s"

YLABEL = "Total solve time (s)" if METRIC == "time_s" else ("Iterations" if METRIC == "iter" else "Time / iter (ms)")

# -----------------------------
# 3) Plot grouped bars (COMPRESS Direct-LU group width)
# -----------------------------
method_order = ["Direct-LU", "ILU(0)", "ILU(1)", "ILU(2)"]
ordering_order = ["Lexigraphic", "AMD", "RCM", "Q-order"]

ordering_to_color = {
    "Lexigraphic": colors[0],
    "AMD":         colors[2],
    "RCM":         colors[4],
    "Q-order":     colors[5],
}

def plot_panel(ax, t_value, panel_label=None, show_t_title=True):
    sub = df[df["t"] == t_value].copy()

    slots_per_method = {m: (1 if m == "Direct-LU" else len(ordering_order)) for m in method_order}
    gap_slots = 0.9

    centers = []
    cursor = 0.0
    for m in method_order:
        w = slots_per_method[m]
        centers.append(cursor + 0.5 * w)
        cursor += w + gap_slots
    centers = np.array(centers)

    added_legend = set()
    slot_bar_w = 0.85

    for i, method in enumerate(method_order):
        c = centers[i]
        nslots = slots_per_method[method]

        if method == "Direct-LU":
            sel = sub[(sub["Method"] == method) & (sub["Ordering"] == "AMD")]
            if not sel.empty:
                h = float(sel.iloc[0][METRIC])
                label = "AMD" if "AMD" not in added_legend else None
                ax.bar(
                    c, h,
                    width=slot_bar_w,
                    color=ordering_to_color["AMD"],
                    label=label,
                )
                added_legend.add("AMD")
            continue

        for j, ord_name in enumerate(ordering_order):
            sel = sub[(sub["Method"] == method) & (sub["Ordering"] == ord_name)]
            if sel.empty:
                continue
            h = float(sel.iloc[0][METRIC])
            xpos = c - 0.5 * nslots + (j + 0.5)

            label = ord_name if ord_name not in added_legend else None
            ax.bar(
                xpos, h,
                width=slot_bar_w,
                color=ordering_to_color[ord_name],
                label=label,
            )
            added_legend.add(ord_name)

    ax.set_xticks(centers)
    ax.set_xticklabels(method_order, fontweight="bold")
    ax.set_ylabel(YLABEL, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    if show_t_title:
        t_str = r"$t = 10^{-1}$" if np.isclose(t_value, 1e-1) else r"$t = 10^{-2}$"
        ax.set_title(t_str, pad=10)

    if panel_label is not None:
        ax.text(0.01, 0.98, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontweight="bold", fontsize=16)

# -----------------------------
# 4) Figure layout depends on METRIC
# -----------------------------
if METRIC == "time_per_iter_ms":
    # single panel, no thickness labels
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

    # pick any one thickness to plot from (identical time/iter assumption)
    plot_panel(ax, 1e-1, panel_label=None, show_t_title=False)

    # Legend: centered top inside axes (forced order)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = ["Lexigraphic", "RCM", "AMD", "Q-order"]
    ordered_handles = [by_label[l] for l in legend_order if l in by_label]
    ordered_labels  = [l for l in legend_order if l in by_label]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.98),
        ncol=2,
        frameon=False,
        prop={"weight": "bold", "size": 16},
        columnspacing=1.2,
        handlelength=1.6,
    )

    plt.tight_layout()

    out_png = "ilu_time_iter.png"
    out_svg = "ilu_time_iter.svg"

else:
    # default: two panels for t = 1e-1 and t = 1e-2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)

    plot_panel(axes[0], 1e-1, "(a)", show_t_title=True)
    plot_panel(axes[1], 1e-2, "(b)", show_t_title=True)

    # Legend: centered at top of FIRST subplot (forced order)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = ["Lexigraphic", "RCM", "AMD", "Q-order"]
    ordered_handles = [by_label[l] for l in legend_order if l in by_label]
    ordered_labels  = [l for l in legend_order if l in by_label]

    axes[0].legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        prop={"weight": "bold", "size": 16},
        columnspacing=1.2,
        handlelength=1.6,
    )

    plt.tight_layout()

    out_png = "ilu_orderings_barchart.png"
    out_svg = "ilu_orderings_barchart.svg"  # <-- changed svg name (as requested)

plt.savefig(out_png, dpi=400, bbox_inches="tight")
plt.savefig(out_svg, dpi=400, bbox_inches="tight")
print(f"Wrote: {out_png} and {out_svg}")