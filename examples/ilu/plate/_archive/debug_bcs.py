import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm
import os
import pandas as pd

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

def get_kelem_mat():

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

    return kelem_mat, kelem_dense

def get_cpp_kmat():
    kmat_vec = read_from_csv("csv/plate_kmat.csv")
    # kmat_vec = read_from_csv("csv/plate_kmat_nobcs.csv")
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)

    print(f"{kmat_vec.shape[0]}")
    _nnodes = rowPtr.shape[0] - 1
    nx = int(_nnodes**0.5)
    nxe = nx - 1
    nnodes = nx**2
    num_elements = nxe**2
    kmat_cpp = np.zeros((nnodes*6,nnodes*6))

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
                kmat_cpp[irow,icol] = kmat_vec[36 * block_ind + inz]

    kmat_cpp[np.abs(kmat_cpp) < 1e-8] = 1e-8

    return kmat_cpp, num_elements, nnodes, nx

def get_python_kmat(kelem_mat, num_elements, num_nodes, nx):
    kmat_py = np.zeros((6*num_nodes, 6*num_nodes))
    for ielem in range(num_elements):
        ix = ielem // (nx-1)
        iy = ielem % (nx-1)
        inode = nx * ix + iy
        # local_node_conn = [inode, inode + 1, inode + nx + 1, inode + nx]
        local_node_conn = [inode, inode + 1, inode + nx, inode + nx + 1]
        # print(local_node_conn)
        local_dof_conn = [[6*inode + idof for idof in range(6)] for inode in local_node_conn]
        local_dof_conn = np.array(local_dof_conn).flatten()

        # for erow,grow in enumerate(local_dof_conn):
        #     for ecol,gcol in enumerate(local_dof_conn):
        #         assembly_csv_df_dict['ielem'] += [ielem]
        #         assembly_csv_df_dict['grow'] += [grow]
        #         assembly_csv_df_dict['gcol'] += [gcol]
        #         assembly_csv_df_dict['erow'] += [erow]
        #         assembly_csv_df_dict['ecol'] += [ecol]
        #         assembly_csv_df_dict['value'] += [kelem_mat[erow,ecol]]

        # write out to csv what entries I'm adding in

        # now add kelem into this as dense matrix
        global_ind = np.ix_(local_dof_conn, local_dof_conn)
        kmat_py[global_ind] += kelem_mat
        
    kmat_py[kmat_py == 0] = 1e-8


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

    # for nx = 3
    # bcs = [0, 1, 2, 3, 4, 5, 7, 8, 13, 14, 18, 20, 32, 36, 38, 44, 50]
    for bc in bcs:
        kmat_py[bc,:] = 1e-8
        # kmat_mat2[:,bc] = 1e-8
        kmat_py[bc,bc] = 1.0

    return kmat_py

    
def plot_mat(mat, filename):
    plt.figure()
    ax = sns.heatmap(mat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
    plt.savefig(filename, dpi=400)
    return

# TODO : also compare in bsr format, etc.
kelem_mat,_ = get_kelem_mat()
kmat_cpp, num_elements, num_nodes, nx = get_cpp_kmat()
kmat_py = get_python_kmat(kelem_mat, num_elements, num_nodes, nx)

folder = f"elems{num_elements}"
plot_mat(kmat_cpp, folder + "/kmat-cpp.png")
plot_mat(kmat_py, folder + "/kmat-py.png")