import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

# Define your custom color palette
custom_colors = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# MELD plots
# ----------

meld_df = pd.read_csv("csv/meld.csv")
arr = meld_df.to_numpy()
print(f"{arr=}")
plt.style.use(niceplots.get_style())
plt.figure('meld', figsize=(8, 5))
plt.plot(arr[:, 0], arr[:, 2], label="transferDisps", color=custom_colors[0])
plt.plot(arr[:, 0], arr[:, 3], label="transferLoads", color=custom_colors[1])
plt.legend()
plt.xlabel("Struct nnodes", fontweight='bold')
plt.ylabel("MELD runtime (s)", fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.savefig("out/meld_runtime.svg", dpi=400)
plt.close('meld')

# Plate runtimes
# --------------

linear_static_df = pd.read_csv("csv/linear_static.csv")
arr = linear_static_df.to_numpy()
print(f"{arr=}")
plt.style.use(niceplots.get_style())
plt.figure('linear_static', figsize=(7, 5))

# Use custom colors by ILU ordering index
orderings = ["none", "AMD", "RCM", "qorder"]
# orderings = ["AMD", "RCM", "qorder"]
ct = -1
for i, ordering in enumerate(orderings):
    solve_list = ["LU", "GMRES"] if ordering == "AMD" else ["GMRES"]
    for solve in solve_list:
        ct += 1
        mask = np.logical_and(arr[:, 2] == (" " + ordering), arr[:, 4] == (" " + solve))
        if solve == "GMRES":
            mask = np.logical_and(mask, arr[:, 3] == "  ILU(5)")

        x = np.array(arr[mask, 1], dtype=np.double)
        dof = x * 6
        y = np.array(arr[mask, -1], dtype=np.double)

        if ordering == "none":
            dof = dof[:3]
            y = y[:3]

        plt.plot(
            dof,
            y,
            label=f"{ordering}-{solve}",
            # linestyle="-" if solve == "LU" else "--",
            color=custom_colors[ct]
        )

# plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
plt.legend()
plt.xlabel("# dof", fontweight='bold')
plt.ylabel("linear static runtime (s)", fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("out/plate-runtimes.svg", dpi=400)
plt.close('linear_static')
