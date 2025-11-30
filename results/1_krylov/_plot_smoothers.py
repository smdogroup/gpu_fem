import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for case in ["cylinder-A100"]:
# for case in ["cylinder", "wingbox"]:

    if case == "cylinder":
        filename = "out/cylinder-times.csv"
    elif case == "cylinder-A100":
        filename = "out/cylinder-A100-times.csv"
    else: # wingbox
        filename = "out/wingbox-times.csv"

    df = pd.read_csv(filename)
    arr = df.to_numpy()

    SR = arr[:,0]
    jacobi = arr[:,1]
    gsmc = arr[:,2]
    cheby = arr[:,3]
    ilu0 = arr[:,4]
    ilu1 = arr[:,5]
    ilu2 = arr[:,6]
    LU = arr[:,7]

    toverR = 1.0 / SR

    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.titlesize': 20
    })

    four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
    colors = four_colors6

    fig, ax = plt.subplots(figsize=(9, 7.5))

    plt.plot(toverR, LU, "o-", linewidth=3, color="tab:gray", label="Full-LU")


    six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
    six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

    # colors = six_colors3
    colors = six_colors2

    names = ['Jacobi', 'GSMC', 'CP-8', 'ILU0', 'ILU1', 'ILU2']
    for i in range(6):
        if i >= 3:
            _arr = arr[:, 9 - i]
            _name = names[8 - i]
        else:
            _arr = arr[:,i+1]
            _name = names[i]
        plt.plot(toverR, _arr, "o-" if i != 1 else "o--", linewidth=3, color=colors[i], label=_name)

    plt.legend(loc='center left', bbox_to_anchor=(0, 0.4))

    plt.xlabel("Normalized Thickness, " + r"$\frac{t}{R}$")
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylabel(r"$\log_{10}(resid) \, / \, sec$")
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

fig, ax = plt.subplots(figsize=(9, 7.5))

df2 = pd.read_csv("out/ilu-mem-mb.csv")
arr2 = df2.to_numpy()

DOF = arr2[:,1]


six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

# colors = six_colors3
colors = six_colors2

plt.plot(DOF, 20e3 * np.ones(DOF.shape[0]), "--", linewidth=2, color="tab:gray", label="A100-mem")

names = ['ILU0', 'ILU1', 'ILU2', 'Full-LU']
for i in range(3, -1, -1):
    plt.plot(DOF, arr2[:,2+i], "o-", linewidth=3, color=colors[i], label=names[i])
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


plt.xlabel("Num DOF")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Memory Usage (MB)")
plt.tight_layout()
plt.savefig("out/1_ilu_memory.png", dpi=400)
plt.close('all')