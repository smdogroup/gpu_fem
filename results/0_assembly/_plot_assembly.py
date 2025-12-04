import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

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

# plt.rcParams.update({
#     # 'font.family': 'Courier New',  # monospace font
#     'font.family' : 'monospace', # since Courier new not showing up?
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 20,
#     'figure.titlesize': 20
# })

plt.style.use(niceplots.get_style())

# fs = 20
fs = 24

plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,
    'figure.titlesize': fs,
})

#e76f51, #e9c46a, #2a9d8f

# four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
# colors = four_colors6

six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

# from six_colors2
# colors = [
#     "#e76f51", # total
#     "#e9c46a", # jacobian
#     "#2a9d8f" # residual
# ]

# colors = six_colors2[::-1]
# colors = six_colors2
# four_colors = [colors[0], colors[2], colors[4], colors[5]]
# colors = [six_colors2[5], six_colors2[3], six_colors2[1]]
# colors = [
#     "#1b9e77",  # teal/green (cool, medium-dark)
#     "#d95f02",  # orange (warm, bold)
#     "#7570b3",  # purple (cool, medium)
# ]

# colors = [
#     "#f4f1bb", # total
#     "#ed6a5a", # jacobian
#     "#9bc1bc" # residual
# ]

# colors = [
#     "#ef767a",
#     "#456990",
#     "#49beaa"
# ]
# colors = [
#     "#006ba6",
#     "#"
# ]

# colors = plt.
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [
    colors[2], colors[0], colors[1]
]

fig, ax = plt.subplots(figsize=(8, 6.5))

plt.plot(dof, tot_speedup, 'o-', linewidth=3.0, markersize=8, color=colors[0], label='total')
plt.plot(dof, jac_speedup, 'o-', linewidth=3.0, markersize=8, color=colors[1], label='jacobian')
plt.plot(dof, res_speedup, 'o-', linewidth=3.0, markersize=8, color=colors[2], label='residual')


# plt.legend()
# ax.legend(facecolor='white', framealpha=1.0, edgecolor='black')
leg = ax.legend(loc='lower right', frameon=True, bbox_to_anchor=(1.0, 0.025))
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(1.0)

plt.xlabel("N, Degree of Freedom")
plt.ylabel("CPU to GPU Speedup")
plt.xscale('log')
plt.yscale('log')

ax.set_xmargin(0.05)
ax.set_ymargin(0.05)

# ✔ Force clean y-axis behavior
ax.set_ylim(0.9, 100*1.1)
ax.set_yticks([1, 10, 100])
# ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.grid(True, which='major', linestyle='--', alpha=0.5)

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



# fig, ax = plt.subplots(figsize=(9, 7.5))
fig, ax = plt.subplots(figsize=(8, 6.5))

# plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["axes.unicode_minus"] = False
# from matplotlib.ticker import LogFormatter
# ax.yaxis.set_major_formatter(LogFormatter(10))
# ax.xaxis.set_major_formatter(LogFormatter(10))

import matplotlib.colors as mcolors

# def make_shades(hex_color, n=3, low=0.6, high=1.0):
#     """Generate n shades of a base hex color from darker(low) to lighter(high)."""
#     base = mcolors.to_rgb(hex_color)
#     return [
#         tuple((high * i / (n - 1) + low * (1 - i / (n - 1))) * c for c in base)
#         for i in range(n)
#     ]

# trying to fix minus signs in niceplots that don't show up in log scale
# plt.rcParams["axes.unicode_minus"] = False
# from matplotlib.ticker import LogFormatter
# ax.yaxis.set_major_formatter(LogFormatter(base=10))
# ax.xaxis.set_major_formatter(LogFormatter(base=10))

def make_shades(hex_color, n=3, strength=0.65):
    """
    Make n lighter tints of the base color by mixing with white.
    strength=0.65 means 35% white added at the lightest shade.
    """
    base = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1, 1, 1])

    # Shades: from slightly lighter → lightest
    return [
        tuple((1 - t) * base + t * white)
        for t in np.linspace(0, strength, n)
    ]

# colors = ['#440154', '#31688e', '#35b779']   # purple, blue, green

# ------------------ Jacobian curves (colored) ------------------
# make this shades of colors[2] = "#e9c46a"
# shades_jac = make_shades(colors[2], n=3)
# p1_jac, = ax.plot(dof,      gpu_jac,  'o-', linewidth=3.0, markersize=8, color=shades_jac[0], label='p=1')
# p2_jac, = ax.plot(gpu2_dof, gpu2_jac, 'o-', linewidth=3.0, markersize=8, color=shades_jac[1], label='p=2')
# p3_jac, = ax.plot(gpu3_dof, gpu3_jac, 'o-', linewidth=3.0, markersize=8, color=shades_jac[2], label='p=3')

jac_color = colors[1]    # "#e9c46a"
jac_styles = ['-', '--', ':']
# jac_styles = ['-', '--', '-.']
p1_jac, = ax.plot(dof,      gpu_jac,  f'o{jac_styles[0]}', linewidth=3.0, markersize=8,
                  color=jac_color, label='p=1')
p2_jac, = ax.plot(gpu2_dof, gpu2_jac, f'o{jac_styles[1]}', linewidth=3.0, markersize=8,
                  color=jac_color, label='p=2')
p3_jac, = ax.plot(gpu3_dof, gpu3_jac, f'o{jac_styles[2]}', linewidth=3.0, markersize=8,
                  color=jac_color, label='p=3')


# ------------------ Residual curves (same colors, no labels) ------------------
# make this shades of colors[4] = "#2a9d8f"
# shades_res = make_shades(colors[4], n=3)
# ax.plot(dof,      gpu_res,  'o-', linewidth=3.0, markersize=8, color=shades_res[0])
# ax.plot(gpu2_dof, gpu2_res, 'o-', linewidth=3.0, markersize=8, color=shades_res[1])
# ax.plot(gpu3_dof, gpu3_res, 'o-', linewidth=3.0, markersize=8, color=shades_res[2])

res_color = colors[2]    # "#2a9d8f"
res_styles = ['-', '--', ':']
ax.plot(dof,      gpu_res,  f'o{res_styles[0]}', linewidth=3.0, markersize=8, color=res_color)
ax.plot(gpu2_dof, gpu2_res, f'o{res_styles[1]}', linewidth=3.0, markersize=8, color=res_color)
ax.plot(gpu3_dof, gpu3_res, f'o{res_styles[2]}', linewidth=3.0, markersize=8, color=res_color)


# ------------------ Proxy handles for Jacobian & Residual ------------------
p1_proxy, = ax.plot([], [], 'o-',  color='black', linewidth=3.0, label='p=1')
p2_proxy, = ax.plot([], [], 'o--', color='black', linewidth=3.0, label='p=2')
p3_proxy, = ax.plot([], [], 'o:', color='black', linewidth=3.0, label='p=3')

# ------------------ Legend 1: only p=1,2,3 ------------------
# legend1 = ax.legend(handles=[p3_jac, p2_jac, p1_jac],
#                     loc='upper left') #title="Polynomial Order")
# ax.add_artist(legend1)

L1 = ax.legend(handles=[p3_proxy, p2_proxy, p1_proxy],
                    loc='upper left', frameon=True) #, bbox_to_anchor=(0.01, 0.99)) #title="Polynomial Order")
# ax.add_artist(legend1)
L1.get_frame().set_facecolor('white')
L1.get_frame().set_alpha(1.0)
# # ------------------ Legend 2: jacobian & residual proxies ------------------
# legend2 = ax.legend(handles=[jac_proxy, res_proxy],
#                     loc='upper center') #title="Quantities")


# ---------- Text for Jacobian ----------
# Get the last point of p3_jac
# x_jac = gpu3_dof[-1]
# y_jac = gpu3_jac[-1]

# Add text slightly above
ax.text(
    # x_jac, y_jac * 1.05,   # 5% above
    1e5,1e-1,
    "Jacobian",
    ha='center', va='bottom',
    fontsize=24,
    # fontweight='bold',
    fontstyle='italic',   # <-- italic instead of bold
    color=jac_color
)

# ---------- Text for Residual ----------
# Get the first point of p1_res
x_res = dof[0]
y_res = gpu_res[0]

# Add text slightly below
ax.text(
    # x_res, y_res * 0.95,   # 5% below
    3e6,2e-4,
    "Residual",
    ha='center', va='top',
    fontsize=24,
    # fontweight='bold',
    fontstyle='italic',   # <-- italic instead of bold
    color=res_color
)


# ------------------ Axes settings ------------------
ax.set_xlabel("N, Degree of Freedom")
ax.set_ylabel("Assembly time (s)")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xmargin(0.05)
ax.set_ymargin(0.05)
ax.grid(True, which='major', linestyle='--', alpha=0.5)

# fix ytick labels
yticks = ax.get_yticks()
ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
ax.set_yticklabels(ylabels)

plt.savefig("out/assembly_orders.png", dpi=400)

