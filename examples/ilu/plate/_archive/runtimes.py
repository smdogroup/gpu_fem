import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# 10x10 - fact 4.5e-4, assembly 5.99e-3, LU solve 0.3475
# 100x100 - fact 0.1136, assembly 9.48e-3, LU solve 0.30469
# 300x300 - fact 4.4902, assembly 0.0239, LU solve 2.5743
## 1000x1000 - 

# num nodes
sides = [10, 100, 300]
nnodes = [(x+1)**2 for x in sides]

# gpu_fem GPU runtimes data, on H100 GPU's
gpu_dict = {
    # factorization is on the host anyways
    'factorization' : [4.5e-4, 0.1136, 4.4902],
    'assembly' : [5.99e-3, 9.48e-3, 0.0239],
    'LU_solve' : [0.3475, 0.30469, 2.5743]
}


colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
plt.style.use(niceplots.get_style())
plt.figure()
for i,component in enumerate(['factorization', 'assembly', 'LU_solve']):
    
    plt.plot(nnodes, gpu_dict[component], 'o-', markersize=12, color=colors[2*i], label=component)

plt.margins(x=0.05, y=0.05)
plt.xlabel("Mesh size")
plt.ylabel("Runtime (sec)")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("plate-performance.png", dpi=400)