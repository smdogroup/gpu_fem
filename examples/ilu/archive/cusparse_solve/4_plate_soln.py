import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os

for folder in ["csv", "img"]:
    if not os.path.exists(folder):
        os.mkdir(folder)

# 2x2 elem or 3x3 nodal plate mesh problem
nxe = 2
nx = nxe + 1
num_elements = nxe**2
num_nodes = nx**2

# only solving for (w,thx,thy) 3 of the DOF since geom linear
# u,v,thz are uncoupled from the transverse problem and I'm only applying
# transverse loads so u,v,thz = 0 so we can eliminate from each equation
# and solve reduced problem here (so I can actually see the matrix easier)
block_dim = 3

ndof = block_dim * num_nodes # = 27
# so 27 x 27 full kmat

Kmat = np.zeros((ndof,ndof))

# get Kelem_dense (this is for half-size panel 1x0.5) from kelem_dense.csv
kelem_dense = np.array([2.04825e+09,3.20201e+08,0,0,0,0,-1.84615e+09,2.50312e+08,0,0,0,-1.55805e+07,1.29149e+09,-2.50312e+08,0,0,0,0,-1.49359e+09,-3.20201e+08,0,0,0,-1.55805e+07,3.20201e+08,1.8198e+08,0,0,0,1.55805e+07,-2.50312e+08,-5.37749e+07,0,0,0,1.55805e+07,2.50312e+08,-1.31455e+08,0,0,0,-7.79024e+06,-3.20201e+08,3.24964e+06,0,0,0,-7.79024e+06,0,0,1.30876e+08,-4.67415e+06,-5.60897e+07,0,0,0,-1.30876e+08,4.67415e+06,-5.60897e+07,0,0,0,6.5438e+07,-4.67415e+06,-4.20673e+07,0,0,0,-6.5438e+07,4.67415e+06,-4.20673e+07,0,0,0,-4.67415e+06,4.67517e+06,-7.01109e+06,0,0,0,4.67415e+06,2.33685e+06,7.01123e+06,0,0,0,4.67415e+06,-2.33696e+06,-7.01123e+06,0,0,0,-4.67415e+06,-4.67506e+06,7.01109e+06,0,0,0,-5.60897e+07,-7.01109e+06,4.20678e+07,0,0,0,5.60897e+07,-7.01123e+06,1.40222e+07,0,0,0,-4.20673e+07,7.01123e+06,3.50562e+07,0,0,0,4.20673e+07,7.01109e+06,7.01084e+06,0,0,1.55805e+07,0,0,0,2.0774e+07,-1.55805e+07,-1.55805e+07,0,0,0,1.0387e+07,0,-7.79024e+06,0,0,0,-5.19349e+06,1.55805e+07,7.79024e+06,0,0,0,-1.0387e+07,-1.84615e+09,-2.50312e+08,0,0,0,-1.55805e+07,2.04825e+09,-3.20201e+08,0,0,0,0,-1.49359e+09,3.20201e+08,0,0,0,-1.55805e+07,1.29149e+09,2.50312e+08,0,0,0,0,2.50312e+08,-5.37749e+07,0,0,0,-1.55805e+07,-3.20201e+08,1.8198e+08,0,0,0,-1.55805e+07,3.20201e+08,3.24964e+06,0,0,0,7.79024e+06,-2.50312e+08,-1.31455e+08,0,0,0,7.79024e+06,0,0,-1.30876e+08,4.67415e+06,5.60897e+07,0,0,0,1.30876e+08,-4.67415e+06,5.60897e+07,0,0,0,-6.5438e+07,4.67415e+06,4.20673e+07,0,0,0,6.5438e+07,-4.67415e+06,4.20673e+07,0,0,0,4.67415e+06,2.33685e+06,-7.01123e+06,0,0,0,-4.67415e+06,4.67517e+06,7.01109e+06,0,0,0,-4.67415e+06,-4.67506e+06,-7.01109e+06,0,0,0,4.67415e+06,-2.33696e+06,7.01123e+06,0,0,0,-5.60897e+07,7.01123e+06,1.40222e+07,0,0,0,5.60897e+07,7.01109e+06,4.20678e+07,0,0,0,-4.20673e+07,-7.01109e+06,7.01084e+06,0,0,0,4.20673e+07,-7.01123e+06,3.50562e+07,0,-1.55805e+07,1.55805e+07,0,0,0,1.0387e+07,0,-1.55805e+07,0,0,0,2.0774e+07,1.55805e+07,-7.79024e+06,0,0,0,-1.0387e+07,0,7.79024e+06,0,0,0,-5.19349e+06,1.29149e+09,2.50312e+08,0,0,0,0,-1.49359e+09,3.20201e+08,0,0,0,1.55805e+07,2.04825e+09,-3.20201e+08,0,0,0,0,-1.84615e+09,-2.50312e+08,0,0,0,1.55805e+07,-2.50312e+08,-1.31455e+08,0,0,0,-7.79024e+06,3.20201e+08,3.24964e+06,0,0,0,-7.79024e+06,-3.20201e+08,1.8198e+08,0,0,0,1.55805e+07,2.50312e+08,-5.37749e+07,0,0,0,1.55805e+07,0,0,6.5438e+07,4.67415e+06,-4.20673e+07,0,0,0,-6.5438e+07,-4.67415e+06,-4.20673e+07,0,0,0,1.30876e+08,4.67415e+06,-5.60897e+07,0,0,0,-1.30876e+08,-4.67415e+06,-5.60897e+07,0,0,0,-4.67415e+06,-2.33696e+06,7.01123e+06,0,0,0,4.67415e+06,-4.67506e+06,-7.01109e+06,0,0,0,4.67415e+06,4.67517e+06,7.01109e+06,0,0,0,-4.67415e+06,2.33685e+06,-7.01123e+06,0,0,0,-4.20673e+07,-7.01123e+06,3.50562e+07,0,0,0,4.20673e+07,-7.01109e+06,7.01084e+06,0,0,0,-5.60897e+07,7.01109e+06,4.20678e+07,0,0,0,5.60897e+07,7.01123e+06,1.40222e+07,0,0,-7.79024e+06,0,0,0,-5.19349e+06,-1.55805e+07,7.79024e+06,0,0,0,-1.0387e+07,0,1.55805e+07,0,0,0,2.0774e+07,1.55805e+07,-1.55805e+07,0,0,0,1.0387e+07,-1.49359e+09,-3.20201e+08,0,0,0,1.55805e+07,1.29149e+09,-2.50312e+08,0,0,0,0,-1.84615e+09,2.50312e+08,0,0,0,1.55805e+07,2.04825e+09,3.20201e+08,0,0,0,0,-3.20201e+08,3.24964e+06,0,0,0,7.79024e+06,2.50312e+08,-1.31455e+08,0,0,0,7.79024e+06,-2.50312e+08,-5.37749e+07,0,0,0,-1.55805e+07,3.20201e+08,1.8198e+08,0,0,0,-1.55805e+07,0,0,-6.5438e+07,-4.67415e+06,4.20673e+07,0,0,0,6.5438e+07,4.67415e+06,4.20673e+07,0,0,0,-1.30876e+08,-4.67415e+06,5.60897e+07,0,0,0,1.30876e+08,4.67415e+06,5.60897e+07,0,0,0,4.67415e+06,-4.67506e+06,7.01109e+06,0,0,0,-4.67415e+06,-2.33696e+06,-7.01123e+06,0,0,0,-4.67415e+06,2.33685e+06,7.01123e+06,0,0,0,4.67415e+06,4.67517e+06,-7.01109e+06,0,0,0,-4.20673e+07,7.01109e+06,7.01084e+06,0,0,0,4.20673e+07,7.01123e+06,3.50562e+07,0,0,0,-5.60897e+07,-7.01123e+06,1.40222e+07,0,0,0,5.60897e+07,-7.01109e+06,4.20678e+07,0,-1.55805e+07,-7.79024e+06,0,0,0,-1.0387e+07,0,7.79024e+06,0,0,0,-5.19349e+06,1.55805e+07,1.55805e+07,0,0,0,1.0387e+07,0,-1.55805e+07,0,0,0,2.0774e+07])
red_kelem_dense = np.zeros((4*3,4*3)) # reduce problem size here
for irow in range(12):
    inode = irow // 3
    idof = irow % 3
    old_row = 6 * inode + idof + 2 # w starts at 2

    for jcol in range(12):
        jnode = jcol // 3
        jdof = jcol % 3
        old_col = 6 * jnode + jdof + 2

        ind = 24 * old_row + old_col
        # print(f"{irow=} {jcol=} {ind=}")
        # think I did this right
        red_kelem_dense[irow,jcol] = kelem_dense[ind]

# print(f"{red_kelem_dense.flatten()=}")

# plt.figure()
# ax = sns.heatmap(kelem_dense.reshape((24,24)), cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
# plt.figure()
# ax = sns.heatmap(red_kelem_dense, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
# plt.show()

# add the elements in by hand
conn = np.array([
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [3, 4, 7, 6],
    [4, 5, 8, 7]
])

for ielem,elem_conn in enumerate(conn):
    local_dof_conn = [[3*inode + idof for idof in range(3)] for inode in elem_conn]
    local_dof_conn = np.array(local_dof_conn).flatten()
    # print(f"{local_dof_conn=}")

    global_ind = np.ix_(local_dof_conn, local_dof_conn)
    Kmat[global_ind] += red_kelem_dense

# now apply BCs to it (all nodes have w = 0 except node = 4 middle one)
bcs = [3*_ for _ in range(num_nodes) if _ != 4]
# add 1,2 thx and thy constraint at node 0 corner otherwise that's unconstrained
bcs += [1,2]

for bc in bcs:
    Kmat[bc,:] = 0.0
    Kmat[:,bc] = 0.0 # col zero not necessary unless nonzero bc (and C++ doesn't do that yet)
    Kmat[bc,bc] = 1.0

plt.figure()
ax = sns.heatmap(Kmat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
# plt.show()
plt.savefig("img/kmat_py.png", dpi=400)

# set our rhs now
rhs = np.zeros((ndof,1)) # 27 x1 vec
# load only applied in w dof at middle node 4
rhs[3*4] = 1.0

soln = np.linalg.solve(Kmat, rhs)
print(f"{soln[:,0]=}")

L = sp.linalg.cholesky(Kmat).T
plt.figure()
ax = sns.heatmap(L, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
# plt.show()
plt.savefig("img/kmat_L_py.png", dpi=400)

# run this script to also plot kmat from c++
import os
os.system("python3 _4_plot_kmat.py")