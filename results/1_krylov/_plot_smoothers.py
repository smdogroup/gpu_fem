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