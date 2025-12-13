import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# num nodes
# L1 - 23738, L2 - , L3 - 
meshes = [23738, 54439, 88341]

# gpu_fem GPU runtimes data, on H100 GPU's
gpu_dict = {
    # factorization is on the host anyways
    'factorization' : [0.646, 4.214, 11.6],
    'assembly' : [0.01316, 0.018733, 0.0374],
    'LU_solve' : [0.98271, 1.9803, 3.765]
}

# 1 process
cpu_1_dict = {
    'factorization' : [0.345, 1.0412, 1.686],
    'assembly' : [0.705, 1.4856, 2.517],
    'LU_solve' : [3.2891, 9.7261, 22.13]
}

# on 4 processes
cpu_4_dict = {
    'factorization' : [0.2021, 0.7452, 1.2479],
    'assembly' : [0.2752, 0.5211, 0.996],
    'LU_solve' : [3.0511, 9.1801, 21.29]
}

colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
for i,component in enumerate(['factorization', 'assembly', 'LU_solve']):
    plt.style.use(niceplots.get_style())
    plt.figure()
    plt.title(component.capitalize())
    plt.plot(meshes, cpu_1_dict[component], 'o-', markersize=12, color=colors[0], label="CPU-1proc")
    plt.plot(meshes, cpu_4_dict[component], 'o-', markersize=12, color=colors[2], label="CPU-4proc")
    plt.plot(meshes, gpu_dict[component], 'o-', markersize=12, color=colors[4], label="GPU")

    plt.margins(x=0.05, y=0.05)
    plt.xlabel("Mesh size")
    plt.ylabel("Runtime (sec)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(component + ".png", dpi=400)