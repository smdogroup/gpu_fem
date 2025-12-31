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

# colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

"""plot the solution"""

fig, ax = plt.subplots(1,1)

vec = np.loadtxt("out/rhs.csv", delimiter=',')
# vec = np.loadtxt("out/soln.csv", delimiter=',')

# setting input here..
idof_plot = 0 # w
# idof_plot = 1 # thx 
# idof_plot = 2 # thy


ndof = vec.shape[0]
N = ndof // 3
nx = np.int32(N**0.5)
nxe = nx - 1
print(f"python, plot soln with {nxe=}")

w = vec[idof_plot::3]
W = np.reshape(w, (nx, nx))
x, y = np.linspace(0, 1, nx), np.linspace(0.0, 1.0, nx)
X, Y = np.meshgrid(x, y)

print(f'{w=}')

cax = fig.add_subplot(1,1,1, projection='3d')
cax.plot_surface(X, Y, W, cmap='jet')
plt.show()