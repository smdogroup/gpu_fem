import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots


DOFs = [6.53e3, 99.8e3, 1.58e6, 25.2e6]

# t/R = 0.1
thick_e1 = np.array([
    [4.64e-2, 5.66e-1, 1.23e1, 3.61e2], # LU
    [8.26e-3, 7e-2, 2.61e0, 2.19e2],
    [7.5e-3, 3.33e-2, 3.11e-1, 4.92e0]
])

thick_e2 = np.array([
    [4.51e-2, 5.31e-1, 1.07e1, 3.30e2],
    [2.16e-2, 7.83e-2, 2.63e0, 2.18e3],
    [2e-2, 1.50e-1, 9.57e-1, 1.33e1]
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

names = ["LU", "S-PCG", "GMG"]
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
    plt.plot(DOFs, thick_e1[i], "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)

    # plot label
    x_pt = DOFs[-1]
    y_pt = thick_e1[i][-1]

    # call-out text offset factors (log-aware)
    x_off = 0.7 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.1 # if _i == 0 else 0.95  # 15% up

    if i == 1:
        y_off = 0.4
    elif i == 2:
        x_off = 0.85

    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{y_pt:.1f}",
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
plt.ylabel("Solve Time (s)")

# fix xtick labels
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)

# fix ymax
ax.set_ylim(0.8*np.min(thick_e1), np.max(thick_e2)*1.2)


# fix ytick labels
yticks = ax.get_yticks()
print(f"{yticks=}")
ylabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in yticks]
print(f"{ylabels=}")
ax.set_yticklabels(ylabels)

plt.tight_layout()
plt.margins(x=0.04, y=0.08)
plt.savefig("out/2_solve_te-1.png", dpi=400)
plt.close('all')




# ============================================================


fig, ax = plt.subplots(figsize=(8, 6.5))



for _i in range(3):
# for _i in range(4):
    i = perm[_i]

    _name = names[i]
    plt.plot(DOFs, thick_e2[i], "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)

    # plot label
    x_pt = DOFs[-1]
    y_pt = thick_e2[i][-1]

    # call-out text offset factors (log-aware)
    x_off = 0.7 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.1 # if _i == 0 else 0.95  # 15% up

    if i == 0:
        y_off = 0.4
    elif i == 2:
        x_off = 0.85

    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{y_pt:.1f}",
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
plt.ylabel("Solve Time (s)")

# fix xtick labels
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)

# fix ymax
# print(F"{np.max(thick_e2)=}")
ax.set_ylim(0.8*np.min(thick_e1), np.max(thick_e2)*1.2)

# fix ytick labels
yticks = ax.get_yticks()
print(f"{yticks=}")
ylabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in yticks]
print(f"{ylabels=}")
ax.set_yticklabels(ylabels)


plt.tight_layout()
plt.margins(x=0.04, y=0.08)
plt.savefig("out/2_solve_te-2.png", dpi=400)
plt.close('all')



