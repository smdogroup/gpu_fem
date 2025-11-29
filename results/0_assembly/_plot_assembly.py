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
plt.ylabel("CPU/GPU Speedup")
plt.xscale('log')
plt.yscale('log')

plt.savefig("out/assembly_speedup.png", dpi=400)
plt.savefig("out/assembly_speedup.svg")