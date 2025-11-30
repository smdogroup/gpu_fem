import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gpu_df = pd.read_csv("csv/plate_A100gpu.csv")
cpu_df = pd.read_csv("csv/plate_nas48cpu.csv")

gpu_arr = gpu_df.to_numpy()
gpu_dof, gpu_jac, gpu_res = gpu_arr[:,1], gpu_arr[:,2], gpu_arr[:,3]

cpu_arr = cpu_df.to_numpy()
cpu_dof, cpu_jac, cpu_res = cpu_arr[:,1], cpu_arr[:,2], cpu_arr[:,3]

# print(f"{gpu_jac=}\n{cpu_jac=}")
dof = gpu_dof.copy()
jac_speedup = cpu_jac / gpu_jac
res_speedup = cpu_res / gpu_res
tot_speedup = (cpu_jac + cpu_res) / (gpu_jac + gpu_res)

plt.rcParams.update({
    # 'font.family': 'Courier New',  # monospace font
    'font.family' : 'monospace', # since Courier new not showing up?
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
colors = four_colors6

fig, ax = plt.subplots(figsize=(9, 7.5))

plt.plot(dof, tot_speedup, 'o-', linewidth=3.0, color=colors[0], label='total')
plt.plot(dof, jac_speedup, 'o-', linewidth=3.0, color=colors[1], label='jacobian')
plt.plot(dof, res_speedup, 'o-', linewidth=3.0, color=colors[2], label='residual')
plt.legend()
plt.xlabel("N, Degree of Freedom")
plt.ylabel("CPU to GPU Speedup")
plt.xscale('log')
plt.yscale('log')

plt.savefig("out/assembly_speedup.png", dpi=400)
plt.savefig("out/assembly_speedup.svg")


# ============================================
# now plot the assembly times of higher order elements
# ============================================


gpu2_df = pd.read_csv("csv/plate_A100gpu_p2.csv")
gpu2_arr = gpu2_df.to_numpy()
gpu2_dof, gpu2_jac, gpu2_res = gpu2_arr[:,1], gpu2_arr[:,2], gpu2_arr[:,3]

gpu3_df = pd.read_csv("csv/plate_A100gpu_p3.csv")
gpu3_arr = gpu3_df.to_numpy()
gpu3_dof, gpu3_jac, gpu3_res = gpu3_arr[:,1], gpu3_arr[:,2], gpu3_arr[:,3]



fig, ax = plt.subplots(figsize=(9, 7.5))

colors = ['#440154', '#31688e', '#35b779']   # purple, blue, green

# ------------------ Jacobian curves (colored) ------------------
p1_jac, = ax.plot(dof,      gpu_jac,  'o-', linewidth=3.0, color=colors[0], label='p=1')
p2_jac, = ax.plot(gpu2_dof, gpu2_jac, 'o-', linewidth=3.0, color=colors[1], label='p=2')
p3_jac, = ax.plot(gpu3_dof, gpu3_jac, 'o-', linewidth=3.0, color=colors[2], label='p=3')

# ------------------ Residual curves (same colors, no labels) ------------------
ax.plot(dof,      gpu_res,  'o--', linewidth=3.0, color=colors[0])
ax.plot(gpu2_dof, gpu2_res, 'o--', linewidth=3.0, color=colors[1])
ax.plot(gpu3_dof, gpu3_res, 'o--', linewidth=3.0, color=colors[2])

# ------------------ Proxy handles for Jacobian & Residual ------------------
jac_proxy, = ax.plot([], [], 'o-',  color='black', linewidth=3.0, label='jacobian')
res_proxy, = ax.plot([], [], 'o--', color='black', linewidth=3.0, label='residual')

# ------------------ Legend 1: only p=1,2,3 ------------------
legend1 = ax.legend(handles=[p3_jac, p2_jac, p1_jac],
                    loc='upper left') #title="Polynomial Order")
ax.add_artist(legend1)

# ------------------ Legend 2: jacobian & residual proxies ------------------
legend2 = ax.legend(handles=[jac_proxy, res_proxy],
                    loc='upper center') #title="Quantities")

# ------------------ Axes settings ------------------
ax.set_xlabel("N, Degree of Freedom")
ax.set_ylabel("Assembly time (s)")
ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig("out/assembly_orders.png", dpi=400)

