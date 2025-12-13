import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load and clean
gpu_df = pd.read_csv('gpu-times.csv')
cpu_df = pd.read_csv('cpu-times.csv')
gpu_df.columns = gpu_df.columns.str.strip()
cpu_df.columns = cpu_df.columns.str.strip()

# Remove rows where nxe == 10 or nxe == 20
gpu_df = gpu_df[~gpu_df['nxe'].isin([10, 20])]
cpu_df = cpu_df[~cpu_df['nxe'].isin([10, 20])]

# Filter CPU for 4 processors and sort
cpu_df = cpu_df[cpu_df['nprocs'] == 8].sort_values('nxe')
gpu_df = gpu_df.sort_values('nxe')

# # Option to match GPU NZ times to CPU NZ times
match_gpu_to_cpu_nz = True  # Set this to True to align GPU NZ to CPU NZ

# If match_gpu_to_cpu_nz is True, copy the CPU NZ times into GPU dataframe
if match_gpu_to_cpu_nz:
    # Directly replace GPU 'nz_time(s)' with CPU 'nz_time(s)'
    gpu_df['nz_time(s)'] = cpu_df.set_index('nxe').loc[gpu_df['nxe'], 'nz_time(s)'].values

# Plot setup
sns.set(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # NZ, Assembly, Solve
# color_palette = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
# colors = [color_palette[0], color_palette[2], color_palette[4]]
# colors = ['#092c5c', '#8fbce6', '#f5d130']
# colors = ['#d4a86c', '#408098', '#402e30']
colors = ['#e96100', '#f8b515', '#3d89ec'][::-1]
# colors = ["#6b4226", "#a89f91", "#d8d5b2"]
sns.set_palette(colors)

def plot_stacked_log_percent(ax, df, title):
    ndof = df['ndof']
    nz = df['nz_time(s)']
    asm = df['assembly_time(s)']
    solve = df['solve_time(s)']
    total = nz + asm + solve

    # print(f"{total=}")

    # Log base-10 lower bound
    log_y0 = -2 # -4
    log_total = np.log10(total)

    # Cumulative percentages
    pct1 = nz / total
    pct2 = (nz + asm) / total
    pct3 = (nz + asm + solve) / total  # should be 1

    y0 = 10 ** log_y0
    y1 = 10 ** (log_y0 + pct1 * (log_total - log_y0))
    y2 = 10 ** (log_y0 + pct2 * (log_total - log_y0))
    y3 = 10 ** (log_y0 + pct3 * (log_total - log_y0))  # should match total

    # print(f"{pct3=} {y3=}")

    # Fill regions
    print(f"{ndof=}")
    
    ax.fill_between(ndof, y2, y3, color=colors[0], label='LU Solve')
    ax.fill_between(ndof, y1, y2, color=colors[1], label='Assembly')
    ax.fill_between(ndof, y0, y1, color=colors[2], label='NZ Pattern')

    ax.set_yscale('log')
    ax.set_title(title, fontsize=18)
    # ax.set_xscale('log')
    ax.set_xlabel('# dof', fontsize=16, fontweight='bold')
    # ax.set_xmargin(0.05)

# CPU plot
plot_stacked_log_percent(axes[0], cpu_df, 'CPU (8 procs)')
axes[0].set_ylabel('Time (s)', fontsize=16, fontweight='bold')
axes[0].legend(loc='upper left')

# GPU plot
plot_stacked_log_percent(axes[1], gpu_df, 'GPU')
axes[1].legend().remove()

# Set the x-axis to be log-scaled
axes[0].set_xscale('log')
axes[1].set_xscale('log')

# Ensure that both axes have the same x-limits (log scale)
xlim_min = min(gpu_df['ndof'].min(), cpu_df['ndof'].min())
xlim_max = max(gpu_df['ndof'].max(), cpu_df['ndof'].max())
axes[0].set_xlim([xlim_min, xlim_max])
axes[1].set_xlim([xlim_min, xlim_max])

# Set custom ticks on the x-axis to ensure 1e4 and 1e5 are shown
axes[0].set_xticks([1e4, 1e5])
axes[1].set_xticks([1e4, 1e5])

# Optionally, you can also add labels for these ticks
axes[0].set_xticklabels([r'$10^4$', r'$10^5$'])
axes[1].set_xticklabels([r'$10^4$', r'$10^5$'])

plt.tight_layout()
plt.savefig("cpu_vs_gpu.svg", dpi=400)
plt.savefig("cpu_vs_gpu.png", dpi=400)
# plt.show()
