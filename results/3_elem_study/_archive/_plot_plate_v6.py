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

# Using a palette derived from six_colors1
elem_colors = {
    # MITC family: blue shades
    'MITC4':  "#3b90b3",  # medium blue
    'MITC9':  "#1a4c60",  # deep blue
    'MITC16': "#3cc7a1",  # teal-ish blue

    # CFI family: red shades
    'CFI4':  "#d95a72",   # muted red
    'CFI9':  "#eb9a79",   # soft red
    'CFI16': "#f3d27d",   # warm light red/orange

    # HRA family: neutral gray
    'HRA4':  '#808080',   # medium gray

    # LFI family: purple (distinct from CFI16)
    'LFI16': "#8e6cb3",   # dusty purple
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
