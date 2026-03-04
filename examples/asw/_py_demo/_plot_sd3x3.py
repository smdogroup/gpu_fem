#!/usr/bin/env python3
"""
Quarter-cylinder structured surface mesh + ASW subdomains (3x3 node patches).

This is styled to match the 2x2 baseline script:
- same axis/tick removal via YOUR style_axes_clean() (unchanged)
- ax.computed_zorder = False (reduce mplot3d weird ordering)
- subtle gray cylinder "skin" + base mesh edges
- patches: no interior tessellation edges + strong outlines
- IMPORTANT: better patch tessellation (adaptive nu/nv) so 3x3 fills look smooth.

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

def expanded_bounds(x0, x1, t0, t1, dx, dth, pad_frac_elem, x_nodes, th_nodes):
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
# Patch bounds
# ---------------------------
def element_patch_bounds(i, j, x_nodes, th_nodes):
    """1x1 element patch (2x2 nodes): element (i,j)."""
    return x_nodes[i], x_nodes[i + 1], th_nodes[j], th_nodes[j + 1]

def node_centered_2x2elem_patch_bounds(I, J, nex, net, x_nodes, th_nodes):
    """
    3x3 node group centered at node (I,J) corresponds to a 2x2 element patch around it:
      i in [I-1, I], j in [J-1, J] (clamped)
    """
    i0 = max(0, I - 1)
    i1 = min(nex - 1, I)
    j0 = max(0, J - 1)
    j1 = min(net - 1, J)

    x0 = x_nodes[i0]
    x1 = x_nodes[i1 + 1]
    t0 = th_nodes[j0]
    t1 = th_nodes[j1 + 1]
    return x0, x1, t0, t1

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

def patch_fill_on_cylinder(x0, x1, t0, t1, R, nu=10, nv=10):
    """
    Faces for a param-grid tessellation of the patch.
    edgecolors="none" will hide the interior tessellation lines.
    """
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

def cylinder_underlay_faces(R, L, theta0, theta1, nx=70, nth=70):
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

def adaptive_tessellation_counts(x0, x1, t0, t1, dx, dth,
                                base_per_elem=10,   # resolution per 1x1 element span
                                min_n=8,
                                max_n=34):
    """
    For 3x3 patches (2x2 elements), we want higher nu/nv than 2x2 patches.
    Scale nu,nv by how many element-widths the patch spans in x/theta.
    """
    span_x_elems = max(1.0, (x1 - x0) / dx)
    span_t_elems = max(1.0, (t1 - t0) / dth)

    nu = int(np.clip(np.round(base_per_elem * span_x_elems), min_n, max_n))
    nv = int(np.clip(np.round(base_per_elem * span_t_elems), min_n, max_n))

    # need at least 2 points in each direction
    nu = max(nu, 2)
    nv = max(nv, 2)
    return nu, nv

# ---------------------------
# Axes styling
# (KEEP EXACTLY what you have now — unchanged)
# ---------------------------
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
    # ---- outer mesh ----
    nex, net = 6, 6
    R, L = 1.0, 2.0
    theta0, theta1 = 0.0, 0.5 * np.pi

    # ---- ASW mode ----
    patch_nodes = 3  # 2 or 3 (set to 3 for your 3x3 subdomains)

    # ---- viz knobs (kept close to your v1) ----
    pad_frac_elem_base = -0.30
    pad_frac_elem_jitter = -0.10
    radial_jitter = 0.03

    draw_fill = True
    patch_fill_alpha = 0.06   # 3x3 overlap: keep modest
    outline_lw = 2.2
    outline_alpha = 0.95

    # base cylinder look (match 2x2 baseline)
    base_edge_alpha = 0.55
    base_edge_lw = 1.5
    skin_gray = 0.60
    skin_alpha = 0.40

    # tessellation controls (this is the main upgrade)
    tess_per_elem = 12         # smoother than your old fixed 8x8
    tess_min, tess_max = 8, 36 # safety caps

    # Build nodes/mesh (base mesh always at radius R)
    X, Y, Z, x_nodes, th_nodes = build_nodes(nex, net, R, L, theta0, theta1)
    base_quads = element_quads(X, Y, Z)

    dx = x_nodes[1] - x_nodes[0]
    dth = th_nodes[1] - th_nodes[0]

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False   # critical (same as 2x2 script)
    style_axes_clean(ax)

    # 1) Cylinder "skin" (same idea as 2x2 baseline)
    skin_faces = cylinder_underlay_faces(R, L, theta0, theta1, nx=70, nth=70)
    skin_poly = Poly3DCollection(
        skin_faces,
        facecolors=(skin_gray, skin_gray, skin_gray, skin_alpha),
        edgecolors="none",
        linewidths=0.0,
    )
    skin_poly.set_zsort("min")
    ax.add_collection3d(skin_poly)

    # 2) Base mesh edges on the true cylinder surface
    base_poly = Poly3DCollection(
        base_quads,
        facecolors=(1.0, 1.0, 1.0, 0.02),
        edgecolors=(0.15, 0.15, 0.15, base_edge_alpha),
        linewidths=base_edge_lw,
    )
    base_poly.set_zsort("min")
    ax.add_collection3d(base_poly)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Decide patch centers / count
    if patch_nodes == 2:
        patch_list = [("elem", i, j) for i in range(nex) for j in range(net)]
    elif patch_nodes == 3:
        patch_list = [("node", I, J) for I in range(nex + 1) for J in range(net + 1)]
    else:
        raise ValueError("patch_nodes must be 2 or 3.")

    for k, (mode, a, b) in enumerate(patch_list):
        c = colors[k % len(colors)]

        if mode == "elem":
            x0, x1, t0, t1 = element_patch_bounds(a, b, x_nodes, th_nodes)
            R_eff = R
            pad_eff = pad_frac_elem_base
        else:
            x0, x1, t0, t1 = node_centered_2x2elem_patch_bounds(a, b, nex, net, x_nodes, th_nodes)

            # 4-class (I%2,J%2) labeling for both radius + pad dithers
            cls = (a % 2) + 2 * (b % 2)  # 0..3
            offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)

            R_eff = R + offsets[cls] * radial_jitter
            pad_eff = pad_frac_elem_base + offsets[cls] * pad_frac_elem_jitter

        # Expand/shrink in param space
        x0, x1, t0, t1 = expanded_bounds(x0, x1, t0, t1, dx, dth, pad_eff, x_nodes, th_nodes)

        # Better tessellation (adaptive)
        nu, nv = adaptive_tessellation_counts(
            x0, x1, t0, t1, dx, dth,
            base_per_elem=tess_per_elem,
            min_n=tess_min,
            max_n=tess_max
        )

        # Fill (no interior edges)
        if draw_fill and patch_fill_alpha > 0.0:
            faces = patch_fill_on_cylinder(x0, x1, t0, t1, R_eff, nu=nu, nv=nv)
            poly = Poly3DCollection(
                faces,
                facecolors=(plt.matplotlib.colors.to_rgba(c, patch_fill_alpha)),
                edgecolors="none",
                linewidths=0.0,
            )
            poly.set_zsort("max")
            ax.add_collection3d(poly)

        # Outline
        outline = patch_outline_on_cylinder(x0, x1, t0, t1, R_eff, n=32)
        segs = np.stack([outline[:-1], outline[1:]], axis=1)
        lc = Line3DCollection(segs, colors=[c], linewidths=outline_lw, alpha=outline_alpha)
        ax.add_collection3d(lc)

    # limits / view
    ax.set_xlim(0, L)
    rmax = R + max(0.0, radial_jitter) + 0.05
    ax.set_ylim(0, rmax)
    ax.set_zlim(0, rmax)

    try:
        ax.set_box_aspect((L, R, R))
    except Exception:
        pass

    # 4) Node markers (draw last so they sit on top)
    draw_node_markers(ax, X, Y, Z, ms=3.2, alpha=0.9)

    ax.view_init(elev=20, azim=110 - 55)
    plt.tight_layout()
    plt.savefig("asw3x3.png", dpi=400)
    # plt.show()

if __name__ == "__main__":
    main()