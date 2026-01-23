import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

for case in ["cylinder-A100"]:
# for case in ["cylinder", "wingbox"]:

    if case == "cylinder":
        filename = "out/cylinder-times.csv"
    elif case == "cylinder-A100":
        filename = "out/cylinder-A100-times_old.csv"
    else: # wingbox
        filename = "out/wingbox-times.csv"

    df = pd.read_csv(filename)
    arr = df.to_numpy()

    print(f"{arr=}")

    SR = arr[:,0]
    jacobi = arr[:,1]
    gsmc = arr[:,2]
    cheby = arr[:,3]
    ilu0 = arr[:,4]
    ilu1 = arr[:,5]
    ilu2 = arr[:,6]
    LU = arr[:,8]

    toverR = 1.0 / SR

    show_time = True
    # show_time = False # then show rat eof resid/sec

    # plt.rcParams.update({
    #     # 'font.family': 'Courier New',  # monospace font
    #     'font.family' : 'monospace', # since Courier new not showing up?
    #     'font.size': 20,
    #     'axes.titlesize': 20,
    #     'axes.labelsize': 20,
    #     'xtick.labelsize': 20,
    #     'ytick.labelsize': 20,
    #     'legend.fontsize': 18,
    #     'figure.titlesize': 20
    # })

    plt.style.use(niceplots.get_style())

    # fs = 24
    # fs = 20
    fs = 18
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

    four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
    colors = four_colors6

    # fig, ax = plt.subplots(figsize=(9, 7.5))
    fig, ax = plt.subplots(figsize=(8.5, 7))

    plt.plot(toverR, 14.0 / LU, "o-", linewidth=3, markersize=8, color="tab:gray", label="Full-LU")
    x_pt = toverR[0]
    y_pt = (14.0 / LU)[0]
    # call-out text offset factors (log-aware)
    x_off = 0.7 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.04 # if _i == 0 else 0.95  # 15% up
    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{y_pt:.3f}",
        color="tab:gray",
        fontsize=text_fs,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    x_pt = toverR[-1]
    y_pt = (14.0 / LU)[-1]
    # call-out text offset factors (log-aware)
    x_off = 1.2 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.04 # if _i == 0 else 0.95  # 15% up
    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{y_pt:.3f}",
        color="tab:gray",
        fontsize=text_fs,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


    # six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    # six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    # six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
    # six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

    # more muted version
    # six_colors1 = ["#c85c6d", "#d78f71", "#e3c877", "#65b39a", "#4d86a3", "#305161"]
    
    six_colors1 = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60", 'k']



    # colors = six_colors3
    # colors = six_colors2
    # colors = six_colors4
    colors = six_colors1

    names = ['Jacobi', 'MCGS', 'CP8', 'ILU0', 'ILU1', 'ILU2', 'ASW']
    reorder = [6, 2, 1, 0, 5, 4, 3]

    for _i in range(6, -1, -1):
        i = reorder[_i]

        # if i >= 3:
        #     _arr = arr[:, 9 - i]
        #     _name = names[8 - i]
        # else:
        _arr = arr[:,i+1]
        _name = names[i]
        if show_time:
            _arr = 6.0 / _arr
            _arr[_arr > 2.0] = np.nan
            _arr0 = arr[:,i+1]
            if _name == "ASW":
                print(f"{_arr0=}\n{_arr=}")
        plt.plot(toverR, _arr, "o-" if i != 1 else "o--", linewidth=3, markersize=8, color=colors[_i], label=_name)

        # call out certain values..
        x_pt = toverR[0]
        y_pt = _arr[0]
        # call-out text offset factors (log-aware)
        x_off = 0.7 # if _i == 0 else 0.45   # 10% to the left
        y_off = 1.04 # if _i == 0 else 0.95  # 15% up

        if _i == 2:
            y_off = 0.85
        elif _i == 4:
            y_off = 0.87

        ax.text(
            x_pt * x_off,
            y_pt * y_off,
            f"{y_pt:.3f}",
            color=colors[_i],
            fontsize=text_fs,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

        # then plot left side
        if _i < 3:
            # call out certain values..
            x_pt = toverR[-1]
            y_pt = _arr[-1]
            # call-out text offset factors (log-aware)
            x_off = 1.2 # if _i == 0 else 0.45   # 10% to the left
            y_off = 0.98 # if _i == 0 else 0.95  # 15% up

            if _i == 1:
                y_off = 0.71
            elif _i == 2:
                y_off = 1.12

            ax.text(
                x_pt * x_off,
                y_pt * y_off,
                f"{y_pt:.3f}",
                color=colors[_i],
                fontsize=text_fs,
                fontweight="bold",
                ha="left",
                va="bottom"
            )

    # plt.legend(loc='center left', bbox_to_anchor=(0, 0.3))
    plt.legend(loc='lower left') #, bbox_to_anchor=(0, 0.3))

    plt.xlabel("Normalized Thickness, t/R")
    plt.xscale('log')
    plt.yscale('log')
    if show_time:
        plt.ylabel("Runtime (sec)")
    else:
        plt.ylabel(r"$\log_{10}(resid) \, / \, sec$")

    ax.set_xmargin(0.025)
    ax.set_ymargin(0.05)
    ax.set_ylim(3e-2*0.9, 1e0*1.1)

    # fix xtick labels
    xticks = ax.get_xticks()
    xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
    ax.set_xticklabels(xlabels)

    # # fix ytick labels
    # yticks = ax.get_yticks()
    # ylabels = [f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "" for y in yticks]
    # ax.set_yticklabels(ylabels)

    # ax.set_yticks([4e-2, 1e-1, 1e0])
    # ax.set_yticklabels([f"$4*10^{{-2}}$"] + ylabels)

    ax.set_yticks([4e-2, 1e-1, 1e0])
    # ax.set_yticklabels([r"$4 \cdot 10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]) #, rotation=45, ha='right')
    ax.set_yticklabels(["4e-2", r"$10^{-1}$", r"$10^{0}$"])

    # ax.set_yticks(list(ax.get_yticks()) + [4e-2])
    # ax.set_yticklabels(list(ax.get_yticks()) + ["4e-2"])

    # ax.grid(True, linestyle='--', alpha=0.75) #which='major',
    # ax.grid(True, which='minor', linestyle='--', alpha=0.25)
    # ax.grid(True, which='major', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if case == "cylinder":
        plt.savefig("out/1_cylinder_smoother.png", dpi=400)
    elif case == "cylinder-A100":
        plt.savefig("out/1_cylinder_A100_smoother.png", dpi=400)
    else: # wingbox
        plt.savefig("out/1_wingbox_smoother.png", dpi=400)
    plt.close('all')

# ==================================
# plot the memory now
# ===================================

fig, ax = plt.subplots(figsize=(8, 6.5))

df2 = pd.read_csv("out/ilu-mem-mb.csv")
arr2 = df2.to_numpy()

DOF = arr2[:,1]


# six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
# six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
# six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
# six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

# more muted version
# six_colors1 = ["#c85c6d", "#d78f71", "#e3c877", "#65b39a", "#4d86a3", "#305161"]

# colors = six_colors3
# colors = six_colors2
# colors = six_colors1[2:]

# experimenting with muting
# colors = ["#bd6a78", "#ddcb8f", "#7bb6a6", "#6d8fa3"]
# colors = ["#c85c6d", "#e3c877", "#65b39a", "#4d86a3"]
# six_colors1 = ["#d95a72", "#eb9a79", "#f3d27d", "#3cc7a1", "#3b90b3", "#1a4c60"]
colors = [six_colors1[0], six_colors1[2], six_colors1[3], six_colors1[-2]]

# colors = six_colors1[2:]

plt.plot(DOF, 20e3 * np.ones(DOF.shape[0]), "--", linewidth=2, markersize=8, color="tab:gray", label="A100-mem")

names = ['ILU0', 'ILU1', 'ILU2', 'Full-LU']
for i in range(3, -1, -1):
    plt.plot(DOF, arr2[:,2+i], "o-", linewidth=3, markersize=8, color=colors[i], label=names[i])

    # call out certain values..
    x_pt = DOF[-2]
    y_pt = arr2[-2,2+i] #[-1]
    # call-out text offset factors (log-aware)
    x_off = 0.5 # if _i == 0 else 0.45   # 10% to the left
    y_off = 1.5 # if _i == 0 else 0.95  # 15% up

    # R = 0.92
    R = 0.85

    if i < 3:
        x_off *= 1.2

        y_off = 0.1

        if i == 0:
            y_off *= R**2
        if i == 1:
            y_off *= R * 0.93

    # if i == 1:
    #     y_off *= 0.95

    val = y_pt
    val /= 1e3

    ax.text(
        x_pt * x_off,
        y_pt * y_off,
        f"{val:.1f} GB" if i < 3 else f"{val:.0f} GB",
        color=colors[i],
        fontsize=text_fs,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

# plt.legend(loc='center left', bbox_to_anchor=(0, 0.4))


plt.legend()

# compute log-log slopes and put it in the plot as labels (for full-LU and ILU0)
# indices = {"ILU0": 2, "Full-LU": 5}  # arr2 columns
# for name, col in indices.items():
#     x = DOF
#     y = arr2[:, col]
#     slope = np.polyfit(np.log(x), np.log(y), 1)[0]

#     # Choose a point to place the label (80% of max DOF)
#     # x0 = x[int(0.8 * len(x))]
#     # y0 = y[int(0.8 * len(y))]
#     if col == 2:
#         x0, y0 = 1e5, 2e1
#     else: # col == 5
#         x0, y0 = 1e4, 1e3

#     ax.text(
#         x0, y0,
#         fr"{name} slope = {slope:.2f}",
#         fontsize=13,
#         color=colors[0] if col == 2 else colors[3],
#         ha="left",
#         va="bottom"
#     )

ax.set_xmargin(0.025)
ax.set_ymargin(0.025)

# ax.grid(True, which='major', linestyle='--', alpha=0.5)

plt.xlabel("Num DOF")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Memory Usage (MB)")
plt.tight_layout()
plt.savefig("out/1_ilu_memory.png", dpi=400)
plt.close('all')