import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

df = pd.read_csv("out/cases-gmg-thick.csv")
arr = df.to_numpy()

SR = arr[:,1]

toverR = 1.0 / SR

# plt.rcParams.update({
#     # 'font.family': 'Courier New',  # monospace font
#     # 'font.family' : 'monospace', # since Courier new not showing up?
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 18,
#     'figure.titlesize': 20
# })

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
# plt.plot(toverR, 6.0 / LU, "o-", linewidth=3, color="tab:gray", label="Full-LU")


# six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
# six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
# six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
# six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]
# # colors = six_colors3
# colors = six_colors2

# choose four colors
# ======================
# six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]

# based on six colors
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#ffd166"]
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#ef476f"]
# colors = ["#073b4c", "#06d6a0", "#ffd166", "#ef476f"]

# colors = ["#118ab2", "#06d6a0", "#ffd166", "#f78c6b"]

# favorites
# colors = ["#073b4c", "#06d6a0", "#ffd166", "#ef476f"]
# colors = ["#118ab2", "#06d6a0", "#ffd166", "#ef476f"]
# colors = ["#118ab2", "#06d6a0", "#ef476f", "#ffd166"]
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#f78c6b"]
# colors = ["#073b4c", "#118ab2", "#ffd166", "#ef476f"]
# colors = ["#073b4c", "#118ab2", "#ffd166", "#f78c6b"]
# colors = ["#073b4c", "#118ab2", "#ffd166", "#ef476f"][::-1]
# colors = ["#073b4c", "#118ab2", "#ffd166", "#ef476f"][::-1]
# colors = ["#118ab2", "#06d6a0", "#ffd166", "#ef476f"][::-1]
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#f78c6b"][::-1]
# colors = ["#073b4c", "#118ab2", "#ffd166", "#f78c6b"][::-1]
# colors = ["#ef476f", "#06d6a0", "#073b4c", "#ffd166"]
# colors = ["#ef476f", "#073b4c", "#06d6a0", "#ffd166"]
# colors = ["#f78c6b", "#073b4c", "#06d6a0", "#ffd166"]
# colors = ["#f78c6b", "#073b4c", "#06d6a0", "#118ab2"]
# colors = ["#073b4c", "#06d6a0", "#ffd166", "#ef476f"][::-1]
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#ffd166"][::-1]
# colors = ["#073b4c", "#118ab2", "#06d6a0", "#f78c6b"][::-1]

# colors = ["#b4445c", "#d1b55c", "#4ba98c", "#3a6f8f"]
# colors = ["#8b2f3c", "#c9a84e", "#4b8c72", "#2f5673"]
# colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
# colors = ["#c45c6d", "#e2c878", "#68b39a", "#4d86a3"]
# colors = ["#b24d62", "#d7bc6a", "#5aa88f", "#447a96"]


# more muted colors
# colors = ["#bd6a78", "#ddcb8f", "#7bb6a6", "#6d8fa3"]
# colors = ["#c85c6d", "#e3c877", "#65b39a", "#4d86a3"]

# colors = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60"]
colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]




# =======================


names = ['plate', 'cylinder', 'hemisphere', 'aob-wing']

xmin, xmax = 0.9 * np.min(toverR), 1.2 * np.max(toverR)
# xmin = np.min([0.9 * 1e3, xmin])
plt.hlines(y=1.0, xmin=1.05 *xmin, xmax=0.95 * xmax, colors="tab:gray", linewidth=2, linestyle="--")

# permute them
# perm = [0, 2, 1, 3]
perm = [0, 1, 2, 3]

for _i in range(4):
    i = perm[_i]

    _arr = arr[:,i+2]
    _name = names[i]
    # if show_time:
    #     _arr = 6.0 / _arr
    #     _arr[_arr > 2.0] = np.nan
    plt.plot(toverR, _arr, "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)

    # plot label
    x_pt = toverR[0]
    y_pt = _arr[0]

    # call-out text offset factors (log-aware)
    x_off = 0.7 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.1 # if _i == 0 else 0.95  # 15% up

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

plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.45))

plt.xlabel("Normalized Thickness, " + r"$\frac{t}{L_{ref}}$")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Speedup to Direct-LU")

# fix xtick labels
xticks = ax.get_xticks()
xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
ax.set_xticklabels(xlabels)

# fix ymax
ax.set_ylim(np.min(arr[:,3]), 60)

ax.set_yticks([1e0, 1e1, 60])
# ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
ax.set_yticklabels([r"$10^{0}$", r"$10^{1}$", "60"])

plt.tight_layout()
plt.margins(x=0.04, y=0.04)
plt.savefig("out/2_geoms_thick.png", dpi=400)
plt.close('all')


# =============================
# plot by NDOF now
# =============================



df2 = pd.read_csv("out/cases-gmg-dof.csv")
arr2 = df2.to_numpy()

NDOF = arr2[:,1]


# fig, ax = plt.subplots(figsize=(9, 7.5))
fig, ax = plt.subplots(figsize=(8, 6.5))
# plt.plot(toverR, 6.0 / LU, "o-", linewidth=3, color="tab:gray", label="Full-LU")


names = ['plate', 'cylinder', 'hemisphere', 'aob-wing']

xmin, xmax = 2e4, 1e7
# xmin = np.min([0.9 * 1e3, xmin])
plt.hlines(y=1.0, xmin=1.05 *xmin, xmax=0.95 * xmax, colors="tab:gray", linewidth=2, linestyle="--")

for _i in range(4):

    i = perm[_i]

    _arr = arr2[:,i+3]
    _name = names[i]
    # if show_time:
    #     _arr = 6.0 / _arr
    #     _arr[_arr > 2.0] = np.nan
    if i < 3:
        plt.plot(NDOF, _arr, "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)
    else:  # wing
        plt.plot(arr2[:,2], _arr, "o-", linewidth=3, markersize=ms, color=colors[i], label=_name)

    # plot label
    if i < 3:
        x_pt = NDOF[-1]
        y_pt = _arr[-1]
    else: # wing
        x_pt = NDOF[-2]
        y_pt = _arr[-2]

    # call-out text offset factors (log-aware)
    x_off = 0.8 if i < 3 else 1.2 # if _i == 0 else 0.45   # 10% to the left

    if i in [0,2]:
        y_off = 1.05 # if _i == 0 else 0.95  # 15% up
    elif i == 1:
        y_off = 0.77
    elif i == 3:
        y_off = 1.05

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

    # then do the first one
    # plot label
    x_pt = NDOF[0]
    y_pt = _arr[0]

    # call-out text offset factors (log-aware)
    x_off = 0.8 if i < 3 else 1.2 # if _i == 0 else 0.45   # 10% to the left

    if i in [0,1]:
        y_off = 1.05 # if _i == 0 else 0.95  # 15% up
    elif i == 2:
        y_off = 0.77
    elif i == 3:
        y_off = 1.05

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

plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.35))

plt.xlabel("NDOF, N")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Speedup to Direct-LU")

ax.set_yticks([1e0, 1e1, 1e2])
# ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
ax.set_yticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])

ax.set_xticks([2e4, 1e5, 1e6, 1e7])
# ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
ax.set_xticklabels([r"$2 \times 10^{4}$", r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"])

plt.tight_layout()
plt.margins(x=0.02, y=0.07)
plt.savefig("out/2_geoms_dof.png", dpi=400)
plt.close('all')