import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

DOFs = [166e3, 591e3, 2.38e6]

speedups = np.array([
    [6.77, 13.0, 11.9],
    [5.42, 11.7, 9.47],
    [2.23, 5.28, 6.67]
])

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

# four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
# colors = four_colors6

fig, ax = plt.subplots(figsize=(8, 6.5))


# colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60"]
colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]



# =======================

names = ["t=10.cm", "t=3.1cm", "t=1.0cm"]
# names = ["LU", "S-PCG", "GMG", "SA-AMG"]

# xmin, xmax = 0.9 * np.min(toverR), 1.2 * np.max(toverR)
# xmin = np.min([0.9 * 1e3, xmin])
# plt.hlines(y=1.0, xmin=1.05 *xmin, xmax=0.95 * xmax, colors="tab:gray", linewidth=2, linestyle="--")

# permute them
# perm = [0, 2, 1, 3]
perm = [0, 1, 2, 3]

for _i in range(3):
# for _i in range(4):
    i = perm[_i]

    _name = names[i]
    plt.plot(DOFs, speedups[i], "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)

    # plot label
    x_pt = DOFs[-1]
    y_pt = speedups[i][-1]

    # call-out text offset factors (log-aware)
    x_off = 0.85 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.05 # if _i == 0 else 0.95  # 15% up

    # if i == 1:
    #     y_off = 0.85
    if i == 2 or i == 1:
        x_off = 0.88

    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{y_pt:.1f}" + r"$\times$",
        color=colors[i],
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

# plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.45))
plt.legend()

plt.xlabel("# DOF")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("GMG to LU Speedup")

# fix xtick labels
# xticks = ax.get_xticks()
# xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
# ax.set_xticklabels(xlabels)

ax.set_xticks([1e5, 1e6, 2e6])
# ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
ax.set_xticklabels([r"$10^{5}$", r"$10^{6}$", r"$2 \cdot 10^{6}$"
                    ])

# fix ymax
ax.set_ylim(0.8*np.min(speedups), np.max(speedups)*1.2)


# fix ytick labels
# yticks = ax.get_yticks()
# print(f"{yticks=}")
# ax.set_yticks()
# ylabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in yticks]
# print(f"{ylabels=}")
# ax.set_yticklabels(ylabels)

ax.set_yticks([1e0, 2, 5, 10])
# ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
ax.set_yticklabels([r"$10^{0}$", "2", "5", r"$10^{1}$"
                    ])

ax.set_xlim(1e5, 3e6)

plt.tight_layout()
plt.margins(x=0.08, y=0.08)
plt.savefig("out/nl_wing.png", dpi=400)
plt.close('all')


