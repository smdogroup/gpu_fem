import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import niceplots

for case in ["cylinder-A100"]:
# for case in ["cylinder", "wingbox"]:

    if case == "cylinder":
        filename = "out/cylinder-thick.csv"
    elif case == "cylinder-A100":
        filename = "out/cylinder-A100-thick.csv"
    else: # wingbox
        filename = "out/wingbox-thick.csv"

    df = pd.read_csv(filename)
    arr = df.to_numpy()

    SR = arr[:,0]

    CP_nogmg = arr[:,1]
    jacobi = arr[:,2]
    gsmc = arr[:,3]
    cheby = arr[:,4]
    LU = arr[:,5]

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


    # four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
    # colors = four_colors6

    fig, ax = plt.subplots(figsize=(9, 7.5))
    # plt.plot(toverR, 6.0 / LU, "o-", linewidth=3, color="tab:gray", label="Full-LU")


    # six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    # six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    # six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
    # six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

    # # colors = six_colors3
    # colors = six_colors2

    four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
    colors = four_colors6

    # names = ['CP8-noMG', 'Jacobi', 'GSMC', 'CP8', 'Full-LU']
    names = ['CP8', 'CP8-noMG', 'GSMC', 'Jacobi', 'Full-LU']

    perm = np.array([3, 0, 2, 1, 4])

    for _i in range(5):
        i = perm[_i]
        _arr = arr[:,i+1]
        _name = names[_i]
        # if show_time:
        #     _arr = 6.0 / _arr
        #     _arr[_arr > 2.0] = np.nan
        plt.plot(toverR, _arr, "o-" if not(i in [0, 4]) else "o--", linewidth=3, color=colors[_i], label=_name)

    plt.legend(loc='lower right') #, bbox_to_anchor=(0, 0.3))

    plt.xlabel("Normalized Thickness, " + r"$\frac{t}{R}$")
    plt.xscale('log')
    plt.yscale('log')
    if show_time:
        plt.ylabel("Runtime (sec)")
    else:
        plt.ylabel(r"$\log_{10}(resid) \, / \, sec$")
    
    plt.tight_layout()
    if case == "cylinder":
        plt.savefig("out/2_cylinder_gmg.png", dpi=400)
    elif case == "cylinder-A100":
        plt.savefig("out/2_cylinder_A100_gmg.png", dpi=400)
    else: # wingbox
        plt.savefig("out/2_wingbox_gmg.png", dpi=400)
    plt.close('all')


    # ============================
    # then plot by num DOF
    # ============================

    if case == "cylinder":
        filename = "out/cylinder-dof.csv"
    elif case == "cylinder-A100":
        filename = "out/cylinder-A100-dof.csv"
    else: # wingbox
        filename = "out/wingbox-dof.csv"

    df = pd.read_csv(filename)
    arr = df.to_numpy()

    nxe = arr[:,1]
    NDOF = 6 * (nxe+1)**2

    four_colors6 = ["#231f20", "#bb4430", "#7ebdc2", "#f3dfa2"]
    colors = four_colors6

    fig, ax = plt.subplots(figsize=(9, 7.5))
    # plt.plot(toverR, 6.0 / LU, "o-", linewidth=3, color="tab:gray", label="Full-LU")


    six_colors1 = ["#ef476f", "#f78c6b", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    six_colors2 = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    six_colors3 = ["#e03c31", "#ff7f41", "#f7ea48", "#2dc84d", "#147bd1", "#753bbd"]
    six_colors4 = ["#003049", "#6b2c39", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"]

    # colors = six_colors3
    colors = six_colors2

    # names = ['CP8-noMG', 'CP8', 'Full-LU']
    names = ['GMG', 'PC'] #, 'Full-LU']

    perm = np.array([1,0]) #,2])

    xmin, xmax = 0.9 * np.min(NDOF), 1.2 * np.max(NDOF)
    xmin = np.min([0.9 * 1e3, xmin])

    plt.hlines(y=1.0, xmin=1.05 *xmin, xmax=0.95 * xmax, colors="tab:gray", linewidth=2, linestyle="--")
    plt.hlines(y=100.0, xmin=1.05 *xmin, xmax=0.95 * xmax, colors="tab:gray", linewidth=2, linestyle="--")

    for _i in range(2):
        i = perm[_i]
        _arr = arr[:,i+2]
        _name = names[_i]
        # if show_time:
        #     _arr = 6.0 / _arr
        #     _arr[_arr > 2.0] = np.nan
        plt.plot(NDOF, _arr / arr[:,-1], "o-", linewidth=3, color=colors[_i], label=_name)

        
        speedup = _arr / arr[:, -1]     # your plotted metric
        
        
        # last (max DOF) point
        x_pt = NDOF[-1]
        y_pt = speedup[-1]

        # call-out text offset factors (log-aware)
        x_off = 0.5 if _i == 0 else 0.45   # 10% to the left
        y_off = 1.1 if _i == 0 else 0.95  # 15% up

        ax.text(
            x_pt * x_off,
            y_pt * y_off,
            f"{y_pt:.1f}×",
            color=colors[_i],
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

        # then three back also
        # last (max DOF) point
        x_pt = NDOF[-3]
        y_pt = speedup[-3]

        # call-out text offset factors (log-aware)
        x_off = 0.5
        y_off = 1.1 if _i == 0 else 0.85  # 15% up

        ax.text(
            x_pt * x_off,
            y_pt * y_off,
            f"{y_pt:.1f}×",
            color=colors[_i],
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

        # then five back also
        # last (max DOF) point
        x_pt = NDOF[-5]
        y_pt = speedup[-5]

        # call-out text offset factors (log-aware)
        x_off = 0.6 + 0.1 * (_i == 1)
        y_off = 1.1 if _i == 0 else 0.8  # 15% up

        ax.text(
            x_pt * x_off,
            y_pt * y_off,
            f"{y_pt:.1f}×",
            color=colors[_i],
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

    plt.legend(loc="center left") #loc='lower left') #, bbox_to_anchor=(0, 0.3))

    plt.xlabel("NDOF, N")
    plt.xscale('log')
    plt.yscale('log')
    # if show_time:
    #     plt.ylabel("Runtime (sec)")
    # else:
    #     plt.ylabel(r"$\log_{10}(resid) \, / \, sec$")
    plt.ylabel("Speedup to Direct-LU")

    # plt.margins(x=0.05)
    plt.axis([xmin, xmax, 0.85, 125.0])

    plt.tight_layout()
    if case == "cylinder":
        plt.savefig("out/2_cylinder_gmg_dof.png", dpi=400)
    elif case == "cylinder-A100":
        plt.savefig("out/2_cylinder_A100_gmg_dof.png", dpi=400)
    else: # wingbox
        plt.savefig("out/2_wingbox_gmg_dof.png", dpi=400)
    plt.close('all')
