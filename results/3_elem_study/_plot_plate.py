import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load CSV data ---
csv_file = 'out/_plate.csv'  # change to your CSV filename
df = pd.read_csv(csv_file)

# --- Plot 1: t/R vs linear displacement, mask by element type ---
# plt.figure(figsize=(8,6))
# for elem_type, group in df.groupby('elem_type'):
#     plt.plot(group['t/R'], group['lin_disp'], marker='o', linestyle='-', label=elem_type)
# plt.xlabel('t/R')
# plt.ylabel('Linear displacement')
# # plt.title('Linear displacement vs t/R by element type')
# plt.legend()
# plt.xscale('log')
# # plt.yscale('log')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# --- Plot 2: NDOF vs linear runtime, mask by element type ---
plt.figure(figsize=(8,6))

elem_order = [
    "MITC4", "MITC9", "MITC16",
    "CFI4", "CFI9", "CFI16",
    "LFI16", "HR4"
]
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
        plt.plot(group['NDOF'],
                 group['lin_runtime(s)'],
                 marker='o', linestyle='-',
                 label=elem_type)

# for elem_type, group in df.groupby('elem_type'):
#     plt.plot(group['NDOF'], group['lin_runtime(s)'], marker='o', linestyle='-', label=elem_type)
plt.xlabel('Number of DOF')
plt.ylabel('Linear runtime (s)')
# plt.title('Linear runtime vs NDOF by element type')
plt.xscale('log')  # log scale can help visualize wide range of NDOF
plt.yscale('log')  # optional: runtime varies widely
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
# plt.show()
plt.savefig("out/plate_lin_runtime_dof.png", dpi=400)

# --- Get reference displacement from last CFI16 entry ---
ref_disp = df[df['elem_type'] == 'CFI16']['lin_disp'].iloc[-1] + 1e-12

# --- Compute mesh convergence error relative to that reference ---
df['mesh_error'] = np.abs(df['lin_disp'] - ref_disp) / ref_disp

# --- Plot: NDOF vs mesh convergence error ---
plt.figure(figsize=(8,6))
elem_order = [
    "MITC4", "MITC9", "MITC16",
    "CFI4", "CFI9", "CFI16",
    "LFI16", "HR4"
]
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
# for elem_type, group in df.groupby('elem_type'):
        plt.plot(group['NDOF'], group['mesh_error'], marker='o', linestyle='-', label=elem_type)

plt.xlabel('Number of DOF')
plt.ylabel('Mesh convergence error (relative to last CFI16)')
# plt.title('Mesh convergence error vs NDOF by element type')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
# plt.show()
plt.savefig("out/plate_lin_error_dof.png", dpi=400)

plt.figure(figsize=(8,6))
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        group = df[df['elem_type'] == elem_type]
# for elem_type, group in df.groupby('elem_type'):
        plt.plot(group['lin_runtime(s)'],
             group['mesh_error'],
             marker='o', linestyle='-', label=elem_type)

plt.xlabel('Linear runtime (s)')
plt.ylabel('Mesh convergence error')
# plt.title('Mesh convergence error vs linear runtime')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("out/plate_lin_error_runtime.png", dpi=400)



# -------------------------------------------------------
# 1) NDOF vs NONLINEAR RUNTIME
# -------------------------------------------------------
plt.figure(figsize=(8,6))
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['NDOF'], g['nl_runtime(s)'],
                 marker='o', linestyle='-', label=elem_type)

plt.xlabel('Number of DOF')
plt.ylabel('Nonlinear runtime (s)')
# plt.title('Nonlinear runtime vs NDOF by element type')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("out/plate_nl_runtime_dof.png", dpi=400)


# -------------------------------------------------------
# 2) Compute NONLINEAR mesh convergence error
# -------------------------------------------------------
# Reference = last CFI16 nonlinear displacement
ref_disp_nl = df[df['elem_type'] == 'CFI16']['nl_disp'].iloc[-1] + 1e-8
df['mesh_error_nl'] = np.abs(df['nl_disp'] - ref_disp_nl) / ref_disp_nl


# -------------------------------------------------------
# 3) NDOF vs NONLINEAR mesh convergence error
# -------------------------------------------------------
plt.figure(figsize=(8,6))
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['NDOF'], g['mesh_error_nl'],
                 marker='o', linestyle='-', label=elem_type)

plt.xlabel('Number of DOF')
plt.ylabel('Nonlinear mesh convergence error')
# # plt.title('Nonlinear mesh convergence error vs NDOF')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig("out/plate_nl_error_dof.png", dpi=400)


# -------------------------------------------------------
# 4) NONLINEAR mesh convergence error vs runtime
# -------------------------------------------------------
plt.figure(figsize=(8,6))
for elem_type in elem_order:
    if elem_type in df['elem_type'].unique():
        g = df[df['elem_type'] == elem_type]
        plt.plot(g['nl_runtime(s)'], g['mesh_error_nl'],
                 marker='o', linestyle='-', label=elem_type)

plt.xlabel('Nonlinear runtime (s)')
plt.ylabel('Nonlinear mesh convergence error')
# plt.title('Nonlinear mesh convergence error vs runtime')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.savefig("out/plate_nl_error_runtime.png", dpi=400)