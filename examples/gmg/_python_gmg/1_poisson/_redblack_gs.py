import numpy as np
import matplotlib.pyplot as plt
from _pde_src import *
from _multigrid_src import *

"""setup the poisson linear system"""
# nxe = 4
nxe = 8
# nxe = 32 # choose power of 2 please
# nxe = 64
# nxe = 128
A_0, b_0 = get_poisson_lhs_and_rhs(nxe=nxe)
N = A_0.shape[0]

# plt.imshow(A_0)
# plt.show()

"""compute the solution and compare against method of manufactured disps
 with np dense solve just to check we coded PDE discretization right """
x_discrete = np.linalg.solve(A_0, b_0)
plot_poisson_soln(nxe, _soln=x_discrete)

# permutations of red-black ordering
perm = np.zeros(N, dtype=np.int32)
iperm = np.zeros(N, dtype=np.int32)

even_reds = True
# even_reds = False

half = (N - 1) // 2 if even_reds else (N+1) // 2
for ind in range(N):
    half_ind = ind // 2

    if even_reds:
        even = ind % 2 == 0
        perm_ind = half_ind + half * even
    else: # odd reds
        odd = ind % 2 == 1
        perm_ind = half_ind + half * odd

    perm[ind] = perm_ind
    iperm[perm_ind] = ind

# print(f"{perm=}")

# try and solve red black GS here
b = b_0[iperm]
A = A_0[iperm,:][:,iperm]
# plt.imshow(A_0)

# style the matrix in red vs black parts (for plotting)
A_plot = A.copy()
for i in range(N):
    for j in range(N):
        i_red = i < half
        j_red = j < half
        if A_plot[i,j] == 0.0: continue

        if i_red and j_red:
            A_plot[i,j] = 1.0
        elif i_red and not j_red:
            A_plot[i,j] = 0.8
        elif not i_red and j_red:
            A_plot[i,j] = 0.1
        else:
            A_plot[i,j] = 0.01
A_plot[A_plot == 0.0] = np.nan

# plt.imshow(A_plot, cmap='RdGy_r')
# plt.show()

# red-black gauss seidel using defects
defect = b.copy()
x0 = np.zeros_like(defect)
x = x0.copy()
temp = np.zeros_like(x)

D = np.diag(A)
Dinv = 1.0 / D

init_defect_nrm = np.linalg.norm(defect)

print("red black GS")

for i in range(100):

    # red update
    defect_red = defect[:half]
    dx_red = Dinv[:half] * defect_red
    x[:half] += dx_red
    temp *= 0.0
    temp[:half] = dx_red
    defect -= np.dot(A, temp)
    
    # black update
    defect_black = defect[half:]
    dx_black = Dinv[half:] * defect_black
    x[half:] += dx_black
    temp *= 0.0
    temp[half:] = dx_black
    defect -= np.dot(A, temp)

    defect_nrm = np.linalg.norm(defect)
    print(f"{i=} {defect_nrm=:.3e}")

    if defect_nrm < 1e-6 * init_defect_nrm:
        print(f"red-black GS on {nxe=} converged in {i+1=} iters")
        break