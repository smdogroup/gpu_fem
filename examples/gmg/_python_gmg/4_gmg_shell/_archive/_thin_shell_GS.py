import numpy as np
import matplotlib.pyplot as plt


SR = np.array([1.0, 10.0, 100.0, 1000.0])
gs_Y = np.array([18, 21, 207, 1.13e4])
gs_N = np.array([12, 16, 244, 5162])

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

plt.plot(SR, gs_Y, 'o-', label="cfs_Y")
plt.plot(SR, gs_N, 'o-', label="cfs_N")
plt.legend()
plt.xlabel("Slenderness Ratio")
plt.ylabel("# Block-GS Steps")
plt.xscale('log'), plt.yscale('log')
plt.tight_layout()
plt.show()