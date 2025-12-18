import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# --- Load CSV data ---
csv_file = 'csv/_plate.csv'  # change to your CSV filename
# csv_file = 'csv/_plate100.csv'
# csv_file = 'csv/_plate300.csv'
# csv_file = 'csv/_plate1000.csv'

# csv_file = 'out/archive/_plate300_old.csv'

df = pd.read_csv(csv_file)
# print(f"{df=}")

intermed_legend = False
# intermed_legend = True


plt.style.use(niceplots.get_style())
ms = 7 # marker size
fs = 20
text_fs = 14

plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,
    'figure.titlesize': fs,
})

size = (8,6.5) # figsize


elem_order = ['MITC4', 'MITC9', 'MITC16', 'CFI4', 'CFI9', 'CFI16', 'HRA4', 'LFI16']

# DEBUG
# elem_order = ['CFI4', 'MITC4']
# elem_order = ['CFI4', 'MITC4', 'CFI9', 'MITC9', 'CFI16', 'MITC16']


# =====================================================
# compute mesh convergence error (per element)
# =====================================================

df = df.copy()
df['mesh_error'] = np.nan

TOL = 1e-12   # threshold below which error is treated as zero / artifact

for elem, sub in df.groupby('elem_type'):
    # Sort so last row is finest mesh
    sub = sub.sort_values('NDOF')

    ref_disp = sub['lin_disp'].iloc[-1]

    # tiny offset only to avoid divide-by-zero
    ref_disp += 1e-14 * np.sign(ref_disp)

    err = np.abs(sub['lin_disp'] - ref_disp) / np.abs(ref_disp)

    # mask out numerical-artifact-level errors (including finest mesh)
    err[err < TOL] = np.nan

    df.loc[sub.index, 'mesh_error'] = err



# ============================================================
# --- Plot 1: NDOF vs linear runtime, mask by element type ---
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6.5))

six_colors1 = ["#c85c6d", "#d78f71", "#e3c877", "#65b39a", "#4d86a3", "#305161", "tab:gray", "c"]
colors = six_colors1

unique_elems = df['elem_type'].unique()
print(f"{unique_elems=}")

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]

        ax.plot(group['NDOF'],
                 group['lin_runtime(s)'],
                 marker='o', linestyle='--' if str(group['solver'].to_numpy()[0]) == 'direct' else '-',
                 color=colors[i],
                 linewidth=2.5,
                 markersize=8,
                 label=elem_type)
        i += 1

plt.xlabel('Number of DOF')
plt.ylabel('Linear runtime (s)')
# plt.title('Linear runtime vs NDOF by element type')
plt.xscale('log')  # log scale can help visualize wide range of NDOF
plt.yscale('log')  # optional: runtime varies widely
plt.legend()
plt.grid(True, which='major', ls='--')
plt.tight_layout()
# plt.show()
plt.margins(x=0.05, y=0.05)

xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)
yticks = ax.get_yticks()
ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
ax.set_yticklabels(ylabels)

plt.savefig("out/plate_lin_runtime_dof.png", dpi=400)
plt.close('all')


# ============================================================
# --- Plot 2: NDOF vs mesh conv error, mask by element type ---
# ============================================================


# --- Plot: NDOF vs mesh convergence error ---
# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))

# elem_order = ['MITC4', 'MITC9', 'MITC16', 'CFI4', 'CFI9', 'CFI16', 'HRA4', 'LFI16']

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        # if elem_type == "HR4": continue
        group = df[df['elem_type'] == elem_type]
# for elem_type, group in df.groupby('elem_type'):
        ax.plot(group['NDOF'], group['mesh_error'],
        linewidth=2.5,
        markersize=8,
        marker='o', linestyle='--' if str(group['solver'].to_numpy()[0]) == 'direct' else '-',
        color=colors[i],
        label=elem_type)
        i += 1

plt.xlabel('Number of DOF')
plt.ylabel("L2 displacement error")
# plt.ylabel('Mesh convergence error (relative to last CFI16)')
# plt.title('Mesh convergence error vs NDOF by element type')
plt.xscale('log')
plt.yscale('log')
# plt.legend()
plt.grid(True, which='major', ls='--')
plt.tight_layout()
plt.margins(x=0.05, y=0.05)
# plt.show()
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)
yticks = ax.get_yticks()
ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
ax.set_yticklabels(ylabels)
plt.savefig("out/plate_lin_error_dof.png", dpi=400)
plt.close('all')
plt.close('all')

# ============================================================
# --- Plot 3: linear runtime by meshconv error, mask by element type ---
# ============================================================

# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
        plt.plot(group['lin_runtime(s)'],
             group['mesh_error'],
             linewidth=2.5,
            markersize=8,
             marker='o', linestyle='--' if str(group['solver'].to_numpy()[0]) == 'direct' else '-', color=colors[i], label=elem_type)
        i += 1

plt.xlabel('Linear runtime (s)')
plt.ylabel('L2 displacement error error')
# plt.title('Mesh convergence error vs linear runtime')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='major', ls='--')
# plt.legend()
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)
yticks = ax.get_yticks()
ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
ax.set_yticklabels(ylabels)
plt.tight_layout()
# plt.show()
plt.margins(x=0.05, y=0.05)
plt.savefig("out/plate_lin_error_runtime.png", dpi=400)
