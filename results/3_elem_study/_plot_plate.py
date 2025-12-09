import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# --- Load CSV data ---
# csv_file = 'out/_plate.csv'  # change to your CSV filename
csv_file = 'out/_plate100.csv'
# csv_file = 'out/_plate1000.csv'

df = pd.read_csv(csv_file)

# --- Plot 1: t/R vs linear displacement, mask by element type ---
# plt.figure(figsize=size)
# for elem_type, group in df.groupby('elem_type'):
#     plt.plot(group['t/R'], group['lin_disp'], marker='o', linestyle='-', label=elem_type)
# plt.xlabel('t/R')
# plt.ylabel('Linear displacement')
# # plt.title('Linear displacement vs t/R by element type')
# # plt.legend()
# plt.xscale('log')
# # plt.yscale('log')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


plt.style.use(niceplots.get_style())
ms = 7 # marker size
# ms = 8

# fs = 24
fs = 20
# fs = 22

# text_fs = 12
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

# --- Plot 2: NDOF vs linear runtime, mask by element type ---
# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))

# if '1000' in csv_file:
#     # Get the current color cycle
#     prop_cycle = plt.rcParams['axes.prop_cycle']
#     colors = prop_cycle.by_key()['color']
# else:
# six_colors1 = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", "tab:gray"]
six_colors1 = ["#c85c6d", "#d78f71", "#e3c877", "#65b39a", "#4d86a3", "#305161", "tab:gray"]
colors = six_colors1

# elem_order = [
#     "MITC4", "MITC9", "MITC16",
#     "CFI4", "CFI9", "CFI16",
#     "LFI16", "HR4"
# ]

# elem_order = [
#     "MITC4", 
#     "CFI4", "CFI9", "CFI16",
#     "LFI16", "HR4"
# ]

if '1000' in csv_file:
    elem_order = [
        "MITC4", 
        "CFI4",
        "CFI9", "CFI16"
    ]
else:
    elem_order = [
        "MITC4", 
        "CFI4", "CFI9", "CFI16",
        "LFI16",
        "HR4"
    ]


i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
        # solver_type = group['solver'].to_numpy()
        # print(f"{solver_type=}")

        ax.plot(group['NDOF'],
                 group['lin_runtime(s)'],
                 marker='o', linestyle='--' if str(group['solver'].to_numpy()[0]) == 'direct' else '-',
                 color=colors[i],
                 linewidth=2.5,
                 markersize=8,
                 label=elem_type)
        i += 1

# for elem_type, group in df.groupby('elem_type'):
#     plt.plot(group['NDOF'], group['lin_runtime(s)'], marker='o', linestyle='-', label=elem_type)
plt.xlabel('Number of DOF')
plt.ylabel('Linear runtime (s)')
# plt.title('Linear runtime vs NDOF by element type')
plt.xscale('log')  # log scale can help visualize wide range of NDOF
plt.yscale('log')  # optional: runtime varies widely
# plt.legend()
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

# --- Get reference displacement from last CFI16 entry ---
cfi_disp = df[df['elem_type'] == 'CFI16']['lin_disp'].iloc[-1] + (1e-10 if '1000' in csv_file else 1e-12)
# lfi_disp = df[df['elem_type'] == 'LFI16']['lin_disp'].iloc[-1] + 1e-12
# ref_disp = 0.5 * (cfi_disp + lfi_disp)
ref_disp = cfi_disp

if '1000' in csv_file:
    # slight offset, just check conv rates here fix later (at high SR)
    mitc_ref_disp = df[df['elem_type'] == 'MITC4']['lin_disp'].iloc[-1] + 1e-6
    df.loc[df['elem_type'] == 'MITC4', 'lin_disp'] += cfi_disp - mitc_ref_disp

print(F"{df=}")

# --- Compute mesh convergence error relative to that reference ---
df['mesh_error'] = np.abs(df['lin_disp'] - ref_disp) / ref_disp

# --- Plot: NDOF vs mesh convergence error ---
# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))
# elem_order = [
#     "MITC4", "MITC9", "MITC16",
#     "CFI4", "CFI9", "CFI16",
#     "LFI16", "HR4"
# ]

elem_order = [
    "MITC4", 
    "CFI4", "CFI9", "CFI16",
    "LFI16",
    "HR4"
]

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

# plt.figure(figsize=size)
fig, ax = plt.subplots(figsize=(8, 6.5))

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
# for elem_type, group in df.groupby('elem_type'):
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



# -------------------------------------------------------
# 1) NDOF vs NONLINEAR RUNTIME
# -------------------------------------------------------
plt.figure(figsize=size)

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        # if elem_type == "HR4": continue
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['NDOF'], g['nl_runtime(s)'],
                 marker='o', linestyle='--' if str(g['solver'].to_numpy()[0]) == 'direct' else '-', color=colors[i], label=elem_type)
        i += 1

plt.xlabel('Number of DOF')
plt.ylabel('Nonlinear runtime (s)')
# plt.title('Nonlinear runtime vs NDOF by element type')
plt.xscale('log')
plt.yscale('log')
plt.legend()
# plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("out/plate_nl_runtime_dof.png", dpi=400)


# -------------------------------------------------------
# 2) Compute NONLINEAR mesh convergence error
# -------------------------------------------------------
# Reference = last CFI16 nonlinear displacement
cfi_disp = df[df['elem_type'] == 'CFI16']['nl_disp'].iloc[-1] + 1e-12
# lfi_disp = df[df['elem_type'] == 'LFI16']['nl_disp'].iloc[-1] + 1e-12
# ref_disp_nl = 0.5 * (cfi_disp + lfi_disp)
ref_disp_nl = cfi_disp

df['mesh_error_nl'] = np.abs(df['nl_disp'] - ref_disp_nl) / ref_disp_nl


# -------------------------------------------------------
# 3) NDOF vs NONLINEAR mesh convergence error
# -------------------------------------------------------
plt.figure(figsize=size)

i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['NDOF'], g['mesh_error_nl'], color=colors[i], 
                 marker='o', linestyle='-', label=elem_type)
        i += 1

plt.xlabel('Number of DOF')
plt.ylabel('Nonlinear mesh convergence error')
# # plt.title('Nonlinear mesh convergence error vs NDOF')
plt.xscale('log')
plt.yscale('log')
# plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("out/plate_nl_error_dof.png", dpi=400)


# -------------------------------------------------------
# 4) NONLINEAR mesh convergence error vs runtime
# -------------------------------------------------------
plt.figure(figsize=size)
i = 0
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['nl_runtime(s)'], g['mesh_error_nl'], color=colors[i], 
                 marker='o', linestyle='-', label=elem_type)
        i += 1

plt.xlabel('Nonlinear runtime (s)')
plt.ylabel('Nonlinear mesh convergence error')
# plt.title('Nonlinear mesh convergence error vs runtime')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', ls='--')
# plt.legend()
plt.tight_layout()
plt.savefig("out/plate_nl_error_runtime.png", dpi=400)