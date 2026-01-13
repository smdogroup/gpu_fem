import numpy as np
import pandas as pd
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

colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

df = pd.read_csv("out/poisson_3060Ti.csv")
arr = df.to_numpy()
# print(f"{arr=}")

N = arr[:,1]
startup, DJ, GS = arr[:,2], arr[:,3], arr[:,4]

plt.figure(figsize=(8.5,6.5))
plt.plot(N, startup, 'o-', color=colors[0], label='startup')
plt.plot(N, DJ, 'o-', colors[1], label='Damp-Jac')
plt.plot(N, GS, 'o-', color=colors[2], label='RB-GS')
plt.legend()
plt.xscale('log'); plt.yscale('log')
plt.xlabel("# DOF")
plt.ylabel("Runtime (sec)")

# plt.show()
plt.tight_layout()
plt.savefig("out/poisson-3060ti.png")
plt.savefig("out/poisson-3060ti.svg")



"""plot the 16M dof solution"""

fig, ax = plt.subplots(1,1)

print("reading csv")

# nxe = 16
# soln = np.loadtxt("out/8x8_dof_soln.csv", delimiter=',')

# nxe = 2048
# soln = np.loadtxt("out/4M_dof_soln.csv", delimiter=',')

nxe = 4096
soln = np.loadtxt("out/16M_dof_soln.csv", delimiter=',')

nx = nxe + 1
N = nx**2
SOLN = np.reshape(soln, (nx, nx))
x, y = np.linspace(0, 1, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x, y)

cax = fig.add_subplot(1,1,1, projection='3d')
cax.plot_surface(X, Y, SOLN, cmap='jet')
plt.show()