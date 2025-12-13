import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm
import os
import pandas as pd

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

# Example usage
kelem_vec = read_from_csv("csv/kelem.csv")
kelem_mat = np.zeros((24,24))
# have to restructure data from block format to dense here
for iblock in range(16):
    inode = iblock // 4
    jnode = iblock % 4
    for inz in range(36):
        _irow = inz // 6
        _icol = inz % 6
        irow = 6 * inode + _irow
        icol = 6 * jnode + _icol
        kelem_mat[irow,icol] = kelem_vec[36 * iblock + inz]

kelem_dense = read_from_csv("csv/kelem-dense.csv")
kelem_dense = np.reshape(kelem_dense, newshape=(24,24))

# compare two formats of kelem (they match now)
# plt.imshow(kelem_mat)
# plt.figure()
# plt.imshow(kelem_dense)
# plt.show()

# now read in full bsr mat for 3x3 = 9 elem problem
# also need to read in rowPtr, colPtr too

# kmat_vec = read_from_csv("csv/plate_kmat.csv")
kmat_vec = read_from_csv("csv/plate_kmat_nobcs.csv")
rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)

print(f"{kmat_vec.shape[0]}")
_nnodes = rowPtr.shape[0] - 1
nx = int(_nnodes**0.5)
nxe = nx - 1
nnodes = nx**2
num_elements = nxe**2
num_rows = 6 * nnodes
kmat_mat = np.zeros((nnodes*6,nnodes*6))

if not os.path.exists(f"elems{num_elements}"):
    os.mkdir(f"elems{num_elements}")

for block_row in range(nnodes):
    istart = rowPtr[block_row]
    iend = rowPtr[block_row+1]
    for block_ind in range(istart, iend):
        block_col = colPtr[block_ind]
        val_ind = 36 * block_ind
        for inz in range(36):
            _irow = inz // 6
            _icol = inz % 6
            irow = 6 * block_row + _irow
            icol = 6 * block_col + _icol
            bsr_ind = 36 * block_ind + inz
            # if block_row < 2 and block_col < 2:
                # print(f"{irow=},{icol=} from {bsr_ind=}")
            kmat_mat[irow,icol] = kmat_vec[36 * block_ind + inz]

# have to restructure data from block format to dense here
# for iblock in range(nnodes**2):
#     inode = iblock // nnodes
#     jnode = iblock % nnodes
#     for inz in range(36):
#         _irow = inz // 6
#         _icol = inz % 6
#         irow = 6 * inode + _irow
#         icol = 6 * jnode + _icol
#         kmat_mat[irow,icol] = kmat_vec[36 * iblock + inz]

# add 1e-8 to zero entries so I can see with log scale

kmat_mat[np.abs(kmat_mat) < 1e-8] = 1e-8

# plt.imshow(kmat_mat)
plt.figure()
ax = sns.heatmap(kmat_mat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
plt.savefig(f"elems{num_elements}/kmat_heatmap_c++.png", dpi=400)

# also plot kmat with bcs
kmat_vec_bcs = read_from_csv("csv/plate_kmat.csv")
rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)
kmat_mat_bcs = np.zeros((nnodes*6,nnodes*6))

for block_row in range(nnodes):
    istart = rowPtr[block_row]
    iend = rowPtr[block_row+1]
    for block_ind in range(istart, iend):
        block_col = colPtr[block_ind]
        val_ind = 36 * block_ind
        for inz in range(36):
            _irow = inz // 6
            _icol = inz % 6
            irow = 6 * block_row + _irow
            icol = 6 * block_col + _icol
            bsr_ind = 36 * block_ind + inz
            # if block_row < 2 and block_col < 2:
                # print(f"{irow=},{icol=} from {bsr_ind=}")
            kmat_mat_bcs[irow,icol] = kmat_vec_bcs[36 * block_ind + inz]

kmat_mat_bcs[np.abs(kmat_mat_bcs) < 1e-8] = 1e-8

# plt.imshow(kmat_mat)
plt.figure()
ax = sns.heatmap(kmat_mat_bcs, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
plt.savefig(f"elems{num_elements}/kmat_heatmap_c++_bcs.png", dpi=400)

plt.figure()

assembly_csv_df_dict = {
    mystr:[] for mystr in ['ielem', 'grow', 'gcol', 'erow', 'ecol', 'value']
}

# now compare to assembling it myself
kmat_mat2 = kmat_mat * 0.0
for ielem in range(num_elements):
    ix = ielem // (nx-1)
    iy = ielem % (nx-1)
    inode = nx * ix + iy
    local_node_conn = [inode, inode + 1, inode + nx + 1, inode + nx]
    print(local_node_conn)
    local_dof_conn = [[6*inode + idof for idof in range(6)] for inode in local_node_conn]
    local_dof_conn = np.array(local_dof_conn).flatten()

    for erow,grow in enumerate(local_dof_conn):
        for ecol,gcol in enumerate(local_dof_conn):
            assembly_csv_df_dict['ielem'] += [ielem]
            assembly_csv_df_dict['grow'] += [grow]
            assembly_csv_df_dict['gcol'] += [gcol]
            assembly_csv_df_dict['erow'] += [erow]
            assembly_csv_df_dict['ecol'] += [ecol]
            assembly_csv_df_dict['value'] += [kelem_mat[erow,ecol]]

    # write out to csv what entries I'm adding in

    # now add kelem into this as dense matrix
    global_ind = np.ix_(local_dof_conn, local_dof_conn)
    kmat_mat2[global_ind] += kelem_mat
    
kmat_mat2[kmat_mat2 == 0] = 1e-8

# write assembly csv to csv file
assembly_csv_df = pd.DataFrame(assembly_csv_df_dict)
assembly_csv_df.to_csv(f"elems{num_elements}/python-kmat-assembly.csv", float_format='%.4e', index=False)

ax = sns.heatmap(kmat_mat2, cmap='inferno', norm=LogNorm())
plt.savefig(f"elems{num_elements}/kmat_heatmap_python.png", dpi=400)

# compute rel error between the two kmat without bcs
kmat_rel_err = np.abs((kmat_mat2 - kmat_mat) / (kmat_mat2 + 1e-8))
kmat_rel_err[kmat_rel_err == 0] = 1e-12
plt.figure()
ax = sns.heatmap(kmat_rel_err, cmap='inferno', norm=LogNorm())
plt.savefig(f"elems{num_elements}/kmat_rel_err.png", dpi=400)

# show errored regions in the each matrix
# compute rel error between the two kmat without bcs
kmat_indicator = kmat_rel_err.copy()
kmat_indicator[np.abs(kmat_rel_err) <= 1e-8] = -1.0
kmat_indicator[np.abs(kmat_rel_err) > 1e-8] = 1.0
kmat_cpp_err = kmat_mat * kmat_indicator
# kmat_mat2[kmat_mat2 == 0] = 1e-8

plt.figure()
ax = sns.heatmap(kmat_cpp_err, cmap='inferno', norm=SymLogNorm(linthresh = 0.5))
plt.savefig(f"elems{num_elements}/kmat_cpp_err_regions.png", dpi=400)

# plt.show()

# apply bcs to kmat that I built from python
bcs = []
for ix in range(nx):
    for iy in range(nx):
        inode = nx * ix + iy
        if ix == 0 and iy == 0:
            local_bcs = [0, 1, 2, 3, 4, 5]
        elif ix == 0:
            local_bcs = [0, 2]
        elif iy == 0:
            local_bcs = [1, 2]
        elif ix == nx - 1 or iy == nx - 1:
            local_bcs = [2]
        bcs += [6*inode + idof for idof in local_bcs]
print(bcs)

kmat_mat2_nobc = kmat_mat2.copy()

# for nx = 3
# bcs = [0, 1, 2, 3, 4, 5, 7, 8, 13, 14, 18, 20, 32, 36, 38, 44, 50]
for bc in bcs:
    kmat_mat2[bc,:] = 1e-8
    kmat_mat2[:,bc] = 1e-8
    kmat_mat2[bc,bc] = 1.0
plt.figure()
ax = sns.heatmap(kmat_mat2, cmap='inferno', norm=LogNorm())
plt.savefig(f"elems{num_elements}/kmat_heatmap_python_bcs.png", dpi=400)

# also compute the soln from my own linear system
loads = read_from_csv("csv/plate_loads.csv")
loads = np.reshape(loads, newshape=(loads.shape[0], 1))

# add nugget so kmat_mat2 non-singular in thx, thy, thz dof
theta = 1e-6
kmat_mat2 += theta * np.eye(kmat_mat2.shape[0]) 

soln = np.linalg.solve(kmat_mat2, loads)

num_nodes = soln.shape[0]//6
ned = int(num_nodes**0.5 - 1)

# regen mesh here
nxe = ned; nye = ned; Lx = 2.0; Ly = 1.0
x = np.linspace(0.0, Lx, nxe+1)
y = np.linspace(0.0, Ly, nye+1)
X, Y = np.meshgrid(x, y)

w_disps = soln[2::6] # only show w disps in plot
W = np.reshape(w_disps, newshape=X.shape)


# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, W, cmap='viridis', edgecolor='none')

# Add labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('SOLN (Z-axis)')
ax.set_title('3D Surface Plot')

# Add a color bar to show the mapping of colors to values
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

ax.set_zscale('symlog')
plt.savefig(f"elems{num_elements}/my_python_soln.png", dpi=400)

# Show the plot
# plt.show()

# make a csv file to compare the dense kmat

df_dicts = {
    mystr:[] for mystr in ['row', 'col', 'cpp', 'python', 'relerr']
}

# change small <1e-8 values back to zero
kmat_mat[np.abs(kmat_mat) <= 1e-8] = 0.0
kmat_mat2_nobc[np.abs(kmat_mat2_nobc) <= 1e-8] = 0.0
kmat_rel_err[np.abs(kmat_rel_err) <= 1e-8] = 0.0

for row in range(num_rows):
    for col in range(num_rows):
        df_dicts['row'] += [row]
        df_dicts['col'] += [col]
        df_dicts['cpp'] += [kmat_mat[row, col]]
        df_dicts['python'] += [kmat_mat2_nobc[row, col]]
        df_dicts['relerr'] += [kmat_rel_err[row,col]]

my_df = pd.DataFrame(df_dicts)
my_df.to_csv(f"elems{num_elements}/kmat_compare.csv", float_format='%.4e')