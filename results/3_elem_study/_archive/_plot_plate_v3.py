import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots
from matplotlib.lines import Line2D

# --- Load CSV data ---
csv_file = 'csv/_plate10.csv'
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

size = (8,6.5)
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
# Define colors by element prefix, linestyles by nodes, markers by element type
# =====================================================
prefix_colors = { "MITC": "#c85c6d", "CFI": "#65b39a", "HRA": "#4d86a3", "LFI": "#e3c877" }
node_linestyles = {4: '-', 9: '--', 16: ':'}
elem_markers = {'MITC': 'o', 'CFI': 's', 'HRA': 'D', 'LFI': '^'}  # different shapes per element

# =====================================================
# Function to plot a variable vs variable
# =====================================================
def plot_variable(xvar, yvar, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=size)
    for elem_type in elem_order:
        if elem_type not in df['elem_type'].unique(): 
            continue
        prefix = ''.join(filter(str.isalpha, elem_type))
        nodes = int(''.join(filter(str.isdigit, elem_type)))
        color = prefix_colors.get(prefix, "gray")
        ls = node_linestyles.get(nodes, '-')
        marker = elem_markers.get(prefix, 'o')
        group = df[df['elem_type'] == elem_type]
        ax.plot(group[xvar], group[yvar],
                marker=marker,
                linestyle=ls,
                color=color,
                linewidth=2.5,
                markersize=8,
                alpha=0.8)  # transparency for overlapping lines

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
# --- Legend figure only (custom order) ---
# =====================================================
legend_order = ['CFI4', 'MITC4', 'CFI9', 'MITC9', 
                'CFI16', 'MITC16', 'HRA4','LFI16']  # two rows (read down, right)

legend_handles = []
for elem_type in legend_order:
    prefix = ''.join(filter(str.isalpha, elem_type))
    nodes = int(''.join(filter(str.isdigit, elem_type)))
    color = prefix_colors.get(prefix, "gray")
    ls = node_linestyles.get(nodes, '-')
    marker = elem_markers.get(prefix, 'o')

    line = Line2D([0], [0],
                  color=color,
                  marker=marker,
                  linestyle=ls,
                  linewidth=2.5,
                  markersize=8)
    legend_handles.append(line)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
ax.legend(handles=legend_handles, labels=legend_order, loc='center', ncol=4, frameon=False)
plt.tight_layout()
plt.savefig("out/plate_legend.png", dpi=400)
plt.close()
