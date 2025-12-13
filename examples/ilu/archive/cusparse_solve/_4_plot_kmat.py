import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import scipy as sp

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

def get_cpp_kmat():
    kmat_vec = read_from_csv("csv/plate_kmat.csv")
    rowPtr = read_from_csv("csv/plate_rowPtr.csv").astype(np.int32)
    colPtr = read_from_csv("csv/plate_colPtr.csv").astype(np.int32)

    print(f"{kmat_vec.shape[0]}")
    _nnodes = rowPtr.shape[0] - 1
    nx = int(_nnodes**0.5)
    nxe = nx - 1
    nnodes = nx**2
    num_elements = nxe**2
    block_dim = 3
    block_dim2 = block_dim**2
    kmat_cpp = np.zeros((nnodes*block_dim,nnodes*block_dim))

    if not os.path.exists(f"elems{num_elements}"):
        os.mkdir(f"elems{num_elements}")

    for block_row in range(nnodes):
        istart = rowPtr[block_row]
        iend = rowPtr[block_row+1]
        for block_ind in range(istart, iend):
            block_col = colPtr[block_ind]
            val_ind = block_dim2 * block_ind
            for inz in range(block_dim2):
                _irow = inz // block_dim
                _icol = inz % block_dim
                irow = block_dim * block_row + _irow
                icol = block_dim * block_col + _icol
                bsr_ind = block_dim2 * block_ind + inz
                # if block_row < 2 and block_col < 2:
                    # print(f"{irow=},{icol=} from {bsr_ind=}")
                kmat_cpp[irow,icol] = kmat_vec[block_dim2 * block_ind + inz]

    kmat_cpp[np.abs(kmat_cpp) < 1e-8] = 0.0

    return kmat_cpp, num_elements, nnodes, nx

def plot_mat(mat, filename):
    plt.figure()
    ax = sns.heatmap(mat, cmap="inferno", norm=LogNorm()) #, annot=kmat_mat, annot_kws={'fontsize': 6}, fmt='.1e') # , fmt='s')
    plt.savefig(filename, dpi=400)
    return

if __name__=="__main__":
    kmat_cpp,_,_,_ = get_cpp_kmat()
    plot_mat(kmat_cpp, "img/kmat_cpp.png")