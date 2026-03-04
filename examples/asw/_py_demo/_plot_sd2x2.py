#!/usr/bin/env python3
"""
Quarter-cylinder mesh + ASW subdomains (REBUILT CLEANLY for 2x2 case ONLY).

What this does (exactly):
- ONLY the 2x2-node (1x1-element) subdomains (one patch per element).
- Gray axis panes as background, but:
  * no grid
  * no ticks
  * no tick labels
  * no axis labels
  * keep axis lines
- Cylinder looks solid/white so the gray panes don't “shine through”.
  (Done with a fully-opaque white *underlay* cylinder at R - eps, so it can’t cover patches.)
- Patches: optional face fill (no tessellation edges) + strong outlines.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# ---------------------------
# Geometry
# ---------------------------
def cylinder_map(x, th, R=1.0):
    return np.array([x, R * np.cos(th), R * np.sin(th)], dtype=float)


def build_nodes(nex=6, net=6, R=1.0, L=2.0, theta0=0.0, theta1=0.5*np.pi):
    x = np.linspace(0.0, L, nex + 1)
    th = np.linspace(theta0, theta1, net + 1)

    X = np.zeros((nex + 1, net + 1))
    Y = np.zeros((nex + 1, net + 1))
    Z = np.zeros((nex + 1, net + 1))

    for i in range(nex + 1):
        for j in range(net + 1):
            p = cylinder_map(x[i], th[j], R=R)
            X[i, j], Y[i, j], Z[i, j] = p

    return X, Y, Z, x, th


def element_quads(X, Y, Z):
    nex = X.shape[0] - 1
    net = X.shape[1] - 1
    quads = []
    for i in range(nex):
        for j in range(net):
            v00 = np.array([X[i, j],         Y[i, j],         Z[i, j]])
            v10 = np.array([X[i + 1, j],     Y[i + 1, j],     Z[i + 1, j]])
            v11 = np.array([X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]])
            v01 = np.array([X[i, j + 1],     Y[i, j + 1],     Z[i, j + 1]])
            quads.append(np.vstack([v00, v10, v11, v01]))
    return quads


def clamp_bounds(x0, x1, t0, t1, xmin, xmax, tmin, tmax):
    return max(xmin, x0), min(xmax, x1), max(tmin, t0), min(tmax, t1)


def expanded_element_bounds(i, j, x_nodes, th_nodes, pad_frac_elem=0.10):
    x0, x1 = x_nodes[i], x_nodes[i + 1]
    t0, t1 = th_nodes[j], th_nodes[j + 1]
    dx = x_nodes[1] - x_nodes[0]
    dth = th_nodes[1] - th_nodes[0]

    x0e = x0 - pad_frac_elem * dx
    x1e = x1 + pad_frac_elem * dx
    t0e = t0 - pad_frac_elem * dth
    t1e = t1 + pad_frac_elem * dth

    return clamp_bounds(
        x0e, x1e, t0e, t1e,
        xmin=x_nodes[0], xmax=x_nodes[-1],
        tmin=th_nodes[0], tmax=th_nodes[-1],
    )


# ---------------------------
# Patch rendering helpers
# ---------------------------
def patch_outline_on_cylinder(x0, x1, t0, t1, R, n=40):
    xs_bottom = np.linspace(x0, x1, n)
    th_bottom = np.full_like(xs_bottom, t0)

    th_right = np.linspace(t0, t1, n)
    xs_right = np.full_like(th_right, x1)

    xs_top = np.linspace(x1, x0, n)
    th_top = np.full_like(xs_top, t1)

    th_left = np.linspace(t1, t0, n)
    xs_left = np.full_like(th_left, x0)

    pts = []
    for x, th in zip(xs_bottom, th_bottom):
        pts.append(cylinder_map(x, th, R))
    for x, th in zip(xs_right, th_right):
        pts.append(cylinder_map(x, th, R))
    for x, th in zip(xs_top, th_top):
        pts.append(cylinder_map(x, th, R))
    for x, th in zip(xs_left, th_left):
        pts.append(cylinder_map(x, th, R))

    pts.append(pts[0])
    return np.array(pts)


def patch_fill_on_cylinder(x0, x1, t0, t1, R, nu=8, nv=8):
    xs = np.linspace(x0, x1, nu)
    ts = np.linspace(t0, t1, nv)
    P = [[cylinder_map(x, th, R) for th in ts] for x in xs]

    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            v00 = P[i][j]
            v10 = P[i + 1][j]
            v11 = P[i + 1][j + 1]
            v01 = P[i][j + 1]
            faces.append(np.vstack([v00, v10, v11, v01]))
    return faces


def cylinder_underlay_faces(R, L, theta0, theta1, nx=60, nth=60):
    xs = np.linspace(0.0, L, nx)
    ts = np.linspace(theta0, theta1, nth)
    P = [[cylinder_map(x, th, R) for th in ts] for x in xs]

    faces = []
    for i in range(nx - 1):
        for j in range(nth - 1):
            v00 = P[i][j]
            v10 = P[i + 1][j]
            v11 = P[i + 1][j + 1]
            v01 = P[i][j + 1]
            faces.append(np.vstack([v00, v10, v11, v01]))
    return faces


# ---------------------------
# Axes styling (your spec)
# ---------------------------
def style_axes_gray_panes_no_ticks(ax, pane_alpha=0.18):
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["grid"]["linestyle"] = "None"
        except Exception:
            pass

    # no ticks/labels
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # gray panes only
    try:
        fc = (0.82, 0.82, 0.82, pane_alpha)
        ax.xaxis.pane.set_facecolor(fc)
        ax.yaxis.pane.set_facecolor(fc)
        ax.zaxis.pane.set_facecolor(fc)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass

# def style_axes_clean(ax):
#     # no grid
#     ax.grid(False)
#     # Kill the 3D axes box/spines completely (this is the key line)
#     # ax._axis3don = False
#     # for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
#     #     try:
#     #         axis._axinfo["grid"]["linewidth"] = 0.0
#     #         axis._axinfo["grid"]["linestyle"] = "None"
#     #     except Exception:
#             # pass

#     # no ticks/labels
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
#     ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
#     ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

#     # remove pane fills completely
#     # try:
#     #     ax.xaxis.pane.fill = False
#     #     ax.yaxis.pane.fill = False
#     #     ax.zaxis.pane.fill = False
#     #     ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
#     #     ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
#     #     ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
#     # except Exception:
#     #     pass

#     # # keep axis lines (but subtle)
#     # try:
#     #     ax.xaxis.line.set_color((0, 0, 0, 0.55))
#     #     ax.yaxis.line.set_color((0, 0, 0, 0.55))
#     #     ax.zaxis.line.set_color((0, 0, 0, 0.55))
#     #     ax.xaxis.line.set_linewidth(1.0)
#     #     ax.yaxis.line.set_linewidth(1.0)
#     #     ax.zaxis.line.set_linewidth(1.0)
#     # except Exception:
#     #     pass

# def style_axes_clean(ax):
#     # keep panes + axis box
#     ax.grid(False)

#     # --- remove tick MARKS ---
#     ax.tick_params(
#         axis='x', which='both', length=0
#     )
#     ax.tick_params(
#         axis='y', which='both', length=0
#     )
#     ax.tick_params(
#         axis='z', which='both', length=0
#     )

#     # --- remove tick LABELS only ---
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])

#     # remove axis titles
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     ax.set_zlabel("")

from matplotlib.ticker import NullFormatter, NullLocator

def style_axes_clean(ax):
    ax.grid(False)

    # -------------------------
    # Remove ALL ticks entirely
    # -------------------------
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.zaxis.set_major_locator(NullLocator())

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())

    # remove tick labels (safety)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())

    # -------------------------
    # Remove axis titles only
    # -------------------------
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

def draw_node_markers(ax, X, Y, Z, ms=3.2, alpha=0.9):
    ax.scatter(
        X.ravel(), Y.ravel(), Z.ravel(),
        s=ms**2, marker="o",
        c=[(0.1, 0.1, 0.1, alpha)],
        depthshade=False,      # IMPORTANT: avoids weird dimming
        zorder=50              # draw on top
    )

# ---------------------------
# Main
# ---------------------------
def main():
    # ---- mesh ----
    nex, net = 6, 6
    # nex, net = 6, 12
    # nex, net =  8, 8
    # nex, net =  9, 9
    # nex, net = 12, 12
    R, L = 1.0, 2.0
    theta0, theta1 = 0.0, 0.5 * np.pi
    # theta0, theta1 = 0.0, 1.0 * np.pi

    # ---- viz knobs ----
    pad_frac_elem = -0.10     # negative = shrink patches a bit (less visual overlap)
    draw_patch_fill = True
    patch_fill_alpha = 0.23
    outline_lw = 2.8
    outline_alpha = 0.98

    # cylinder/base look
    pane_alpha = 0.16
    under_eps = 0.015         # IMPORTANT: underlay at R - under_eps (cannot cover patches)
    base_edge_alpha = 0.55
    base_edge_lw = 1.5

    # Build mesh
    X, Y, Z, x_nodes, th_nodes = build_nodes(nex, net, R, L, theta0, theta1)
    base_quads = element_quads(X, Y, Z)

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False   # <-- CRITICAL: stop mplot3d re-sorting collections
    # style_axes_gray_panes_no_ticks(ax, pane_alpha=pane_alpha)
    style_axes_clean(ax)

    # 1) SOLID WHITE CYLINDER UNDERLAY (inside surface, so it never occludes patches)
    # under_faces = cylinder_underlay_faces(R - under_eps, L, theta0, theta1, nx=70, nth=70)
    # under_poly = Poly3DCollection(
    #     under_faces,
    #     facecolors=(1.0, 1.0, 1.0, 1.0),
    #     edgecolors="none",
    #     linewidths=0.0,
    # )
    # under_poly.set_zsort("min")
    # ax.add_collection3d(under_poly)

    # 1b) very light gray cylinder "skin" on the TRUE radius (helps it look cylindrical)
    fs = 0.6
    skin_faces = cylinder_underlay_faces(R, L, theta0, theta1, nx=70, nth=70)
    skin_poly = Poly3DCollection(
        skin_faces,
        facecolors=(fs , fs, fs, 0.4),   # subtle gray tint
        edgecolors="none",
        linewidths=0.0,
    )
    skin_poly.set_zsort("min")
    ax.add_collection3d(skin_poly)

    # 2) Base mesh edges on the true cylinder surface (light, so patches dominate)
    base_poly = Poly3DCollection(
        base_quads,
        facecolors=(1.0, 1.0, 1.0, 0.02),              # essentially no fill (underlay provides white)
        edgecolors=(0.15, 0.15, 0.15, base_edge_alpha),
        linewidths=base_edge_lw,
    )
    base_poly.set_zsort("min")
    ax.add_collection3d(base_poly)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 3) Patches: one per element (2x2 nodes)
    for i in range(nex):
        for j in range(net):
            c = colors[(i * net + j) % len(colors)]
            x0, x1, t0, t1 = expanded_element_bounds(i, j, x_nodes, th_nodes, pad_frac_elem=pad_frac_elem)

            if draw_patch_fill and patch_fill_alpha > 0.0:
                faces = patch_fill_on_cylinder(x0, x1, t0, t1, R, nu=9, nv=9)
                poly = Poly3DCollection(
                    faces,
                    facecolors=(plt.matplotlib.colors.to_rgba(c, patch_fill_alpha)),
                    edgecolors="none",   # no tessellation edge lines
                    linewidths=0.0,
                )
                poly.set_zsort("max")
                ax.add_collection3d(poly)

            outline = patch_outline_on_cylinder(x0, x1, t0, t1, R, n=30)
            segs = np.stack([outline[:-1], outline[1:]], axis=1)
            lc = Line3DCollection(segs, colors=[c], linewidths=outline_lw, alpha=outline_alpha)
            ax.add_collection3d(lc)

    # limits / view
    ax.set_xlim(0, L)
    ax.set_ylim(0, R + 0.05)
    ax.set_zlim(0, R + 0.05)
    # ax.set_ylim(-R, R + 0.05)
    # ax.set_zlim(-R, R + 0.05)

    try:
        ax.set_box_aspect((L, R, R))
    except Exception:
        pass

    # 4) Node markers (draw last so they sit on top)
    draw_node_markers(ax, X, Y, Z, ms=3.2, alpha=0.9)

    ax.view_init(elev=20, azim=110 - 55)
    plt.tight_layout()
    # plt.show()
    plt.savefig("asw2x2.png", dpi=400)


if __name__ == "__main__":
    main()