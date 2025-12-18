import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots
from matplotlib.lines import Line2D

# --- Load CSV data ---
csv_file = 'csv/_plate10.csv'
# csv_file = 'csv/_plate.csv'
df = pd.read_csv(csv_file)

# --- Plot styling ---
plt.style.use(niceplots.get_style())
fs = 20
plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,
    'figure.titlesize': fs,
})

size = (8, 6.5)
elem_order = ['MITC4', 'MITC9', 'MITC16', 'CFI4', 'CFI9', 'CFI16', 'HRA4', 'LFI16']

# =====================================================
# Compute mesh convergence error
# =====================================================
df = df.copy()
df['mesh_error'] = np.nan
TOL = 1e-12
for elem, sub in df.groupby('elem_type'):
    sub = sub.sort_values('NDOF')
    ref_disp = sub['lin_disp'].iloc[-1] + 1e-14 * np.sign(sub['lin_disp'].iloc[-1])
    err = np.abs(sub['lin_disp'] - ref_disp) / np.abs(ref_disp)
    err[err < TOL] = np.nan
    df.loc[sub.index, 'mesh_error'] = err

# =====================================================
# Explicit color scheme with distinct but related families
# =====================================================
# elem_colors = {
#     # MITC family: blue → blue-green → blue-purple
#     'MITC4': '#4a90e2',   # medium blue
#     'MITC9': '#3db7b3',   # blue-green
#     'MITC16':'#6a4abf',   # blue-purple

#     # CFI family: red → orange-red → red-purple
#     'CFI4':  '#ff595e',   # bright red
#     'CFI9':  '#ff924c',   # orange-red
#     'CFI16': '#b0003a',   # red-purple

#     # HRA family: green-teal
#     'HRA4':  '#66c2a5',   # teal-green

#     # LFI family: yellow-orange
#     'LFI16': '#f2c14e',   # yellow-orange
# }

# elem_colors = {
#     # MITC family: medium blue → teal-blue → muted purple-blue
#     'MITC4': '#4a90e2',   # medium blue
#     'MITC9': '#3db7b3',   # tealish blue
#     'MITC16':'#6b4abf',   # muted purple-blue

#     # CFI family: light red → coral → dark red
#     'CFI4':  '#ff6b6b',   # light red
#     'CFI9':  '#ff8c42',   # coral/orange-red
#     'CFI16': '#b0003a',   # dark red

#     # HRA family: distinct greenish-teal
#     'HRA4':  '#2a9d8f',   # green-teal

#     # LFI family: golden yellow, distinct
#     'LFI16': '#d4a017',   # golden-orange
# }


# elem_colors = {
#     # MITC family: blue → blue-purple → deep purple
#     'MITC4':  '#4a90e2',  # medium blue
#     'MITC9':  '#6b4abf',  # blue-purple
#     'MITC16': '#3a1f7c',  # deep purple

#     # CFI family: red → orange-red → dark red
#     'CFI4':  '#ff4c4c',   # bright red
#     'CFI9':  '#ff704c',   # orange-red
#     'CFI16': '#b0003a',   # dark red

#     # HRA family: neutral gray
#     'HRA4':  '#808080',   # medium gray

#     # LFI family: teal/cyan, clearly different from red/blue/gray
#     'LFI16': '#1abc9c',   # teal/cyan
# }

# elem_colors = {
#     # MITC family: medium blue → teal-blue → muted purple-blue
#     'MITC4':  '#4a90e2',  # medium blue
#     'MITC9':  '#6b4abf',  # blue-purple
#     'MITC16': '#3a1f7c',  # deep purple
#     # CFI family: light red → coral → dark red
#     'CFI4':  '#ff6b6b',   # light red
#     'CFI9':  '#ff8c42',   # coral/orange-red
#     'CFI16': '#b0003a',   # dark red

#     # HRA family: neutral gray
#     'HRA4':  '#808080',   # medium gray

#     # LFI family: golden yellow, distinct
#     'LFI16': '#1abc9c',   # golden-orange
# }

# elem_colors = {
#     # MITC family: medium blue → slightly darker blue → deep blue
#     'MITC4':  '#4a90e2',  # medium blue
#     'MITC9':  '#2a6fc0',  # darker blue
#     'MITC16': '#154f8b',  # deep blue

#     # CFI family: light red → medium red → dark red
#     'CFI4':  '#ff6b6b',   # light red
#     'CFI9':  '#e63946',   # medium red
#     'CFI16': '#900c3f',   # dark red

#     # HRA family: neutral gray
#     'HRA4':  '#808080',   # medium gray

#     # LFI family: green, visually distinct from other families
#     'LFI16': '#2ca02c',   # medium green
# }

# elem_colors = {
#     # MITC family: deep blue → slightly darker blue → medium blue
#     'MITC4':  '#154f8b',  # deep blue
#     'MITC9':  '#2a6fc0',  # darker blue
#     'MITC16': '#4a90e2',  # medium blue

#     # CFI family: dark red → medium red → light red
#     'CFI4':  '#900c3f',   # dark red
#     'CFI9':  '#e63946',   # medium red
#     'CFI16': '#ff6b6b',   # light red

#     # HRA family: neutral gray
#     'HRA4':  '#808080',   # medium gray

#     # LFI family: green, visually distinct from other families
#     'LFI16': '#2ca02c',   # medium green
# }

# elem_colors = {
#     # MITC family: deep blue → medium blue → lighter blue
#     'MITC4':  '#154f8b',  # deep blue
#     'MITC9':  '#2a6fc0',  # medium blue
#     'MITC16': '#4a90e2',  # lighter blue

#     # CFI family: dark red → medium red → light red
#     'CFI4':  '#900c3f',   # dark red
#     'CFI9':  '#e63946',   # medium red
#     'CFI16': '#ff6b6b',   # light red

#     # HRA family: purple/pinkish, complementary to red & blue
#     'HRA4':  '#9b59b6',   # medium purple

#     # LFI family: orange/gold, complementary to blue & purple
#     'LFI16': '#f39c12',   # medium orange
# }


# elem_colors = {
#     # MITC family: deep → medium → lighter blue
#     'MITC4':  '#0b3d91',  # deep navy blue
#     'MITC9':  '#1f5fa3',  # muted medium blue
#     'MITC16': '#3a7fc1',  # soft lighter blue

#     # CFI family: deep → medium → muted red
#     'CFI4':  '#7b1f2f',   # deep maroon
#     'CFI9':  '#a03448',   # muted crimson
#     'CFI16': '#c75b5f',   # soft red

#     # HRA family: muted purple
#     'HRA4':  '#6b4c8a',   # dusty purple

#     # LFI family: muted brown/gold
#     'LFI16': '#b28c4b',   # warm muted brown
# }

# elem_colors = {
#     # MITC family: deep → medium → lighter blue
#     'MITC4':  '#2e59a3',  # medium-dark blue
#     'MITC9':  '#5b7fcf',  # medium blue
#     'MITC16': '#89a4e0',  # light blue

#     # CFI family: deep → medium → soft red
#     'CFI4':  '#a23d4d',   # muted burgundy
#     'CFI9':  '#cf5b6c',   # soft red
#     'CFI16': '#e89aa3',   # light rose

#     # HRA family: muted purple/lilac
#     'HRA4':  '#9b7fcf',   # soft purple

#     # LFI family: warm muted orange/gold
#     'LFI16': '#d4a46a',   # light burnt orange
# }


elem_colors = {
    # MITC family: deep → medium → lighter blue
    'MITC4':  '#2e59a3',  # medium-dark blue
    'MITC9':  '#5b7fcf',  # medium blue
    'MITC16': '#89a4e0',  # light blue

    # CFI family: deep → medium → soft red
    'CFI4':  '#a23d4d',   # muted burgundy
    'CFI9':  '#cf5b6c',   # soft red
    'CFI16': '#e08b7f',   # muted coral (avoid pink)
    # 'CFI16': '#ff6b6b',   # light red

    # # CFI family: dark red → medium red → light red
    # 'CFI4':  '#900c3f',   # dark red
    # 'CFI9':  '#e63946',   # medium red
    # 'CFI16': '#ff6b6b',   # light red

    # HRA family: muted yellow/orange, visually distinct from MITC4/CFI4
    'HRA4':  '#e6c65d',   # soft mustard yellow

    # LFI family: muted purple, visually distinct from CFI16
    'LFI16': '#8e6cb3',   # dusty purple
}


node_markers = {4: 'o', 9: 's', 16: '^'}  # marker per node count
line_alpha = 0.8

# =====================================================
# Plotting function
# =====================================================
def plot_variable(xvar, yvar, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=size)
    for elem_type in elem_order:
        if elem_type not in df['elem_type'].unique():
            continue
        nodes = int(''.join(filter(str.isdigit, elem_type)))
        color = elem_colors[elem_type]
        marker = node_markers.get(nodes, 'o')
        group = df[df['elem_type'] == elem_type]
        ax.plot(group[xvar], group[yvar],
                marker=marker,
                linestyle='-' if not(elem_type == 'LFI16') else '--',
                color=color,
                linewidth=2.5,
                markersize=8,
                alpha=line_alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', ls='--')
    plt.tight_layout()
    plt.margins(x=0.05, y=0.05)

    # Format ticks as powers of 10
    xticks = ax.get_xticks()
    ax.set_xticklabels([f"$10^{{{int(np.log10(x))}}}$" if x>0 else "" for x in xticks])
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"$10^{{{int(np.log10(y))}}}$" if y>0 else "" for y in yticks])

    plt.savefig(fname, dpi=400)
    plt.close()

# =====================================================
# Create plots
# =====================================================
plot_variable('NDOF', 'lin_runtime(s)', "out/plate_lin_runtime_dof.png", 'Number of DOF', 'Linear runtime (s)')
plot_variable('NDOF', 'mesh_error', "out/plate_lin_error_dof.png", 'Number of DOF', 'L2 displacement error')
plot_variable('lin_runtime(s)', 'mesh_error', "out/plate_lin_error_runtime.png", 'Linear runtime (s)', 'L2 displacement error')

# =====================================================
# Legend figure
# =====================================================
legend_order = ['CFI4', 'MITC4', 'CFI9', 'MITC9', 'CFI16', 'MITC16', 'HRA4','LFI16']

legend_handles = []
for elem_type in legend_order:
    nodes = int(''.join(filter(str.isdigit, elem_type)))
    color = elem_colors[elem_type]
    marker = node_markers.get(nodes, 'o')
    line = Line2D([0], [0],
                  color=color,
                  marker=marker,
                  linestyle='-' if not(elem_type == 'LFI16') else '--',
                  linewidth=2.5,
                  markersize=8)
    legend_handles.append(line)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
ax.legend(handles=legend_handles, labels=legend_order, loc='center', ncol=4, frameon=False)
plt.tight_layout()
plt.savefig("out/plate_legend.png", dpi=400)
plt.close()
