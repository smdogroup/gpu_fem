import matplotlib.pyplot as plt

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

# v-cycle resids here
resids = [9.575e+04, 1.254e+03, 4.095e+01, 1.546e+00, 6.678e-02, 3.208e-03, 1.656e-04, 8.973e-06, 5.025e-07, 2.883e-08, 1.689e-09]
iters = [_ for _ in range(len(resids))]
plt.plot(iters, resids, 'ko-')
plt.xlabel("V-cycles")
plt.ylabel("Defect residual")
plt.yscale("log")
plt.tight_layout()
plt.show()