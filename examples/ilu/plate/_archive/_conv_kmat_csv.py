import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm
import os
import pandas as pd

def read_from_csv(filename):
    return np.loadtxt(filename, delimiter=",")

num_elements = 16
folder = f"elems{num_elements}/"

cpp_kmat_bsr = np.loadtxt(folder + "cpp-kmat-bsr.csv", delimiter=",", skiprows=1)

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
num_rows = 6 * nnodes
kmat_mat = np.zeros((nnodes*6,nnodes*6))

# goal here is to convert kmat-bsr to format with grow, gcol
new_df_dict = {
    mystr:[] for mystr in ['ielem', 'grow', 'gcol', 'gblock',
                            'elem_block', 'erow','ecol','value']
}


print(cpp_kmat_bsr)

for i in range(cpp_kmat_bsr.shape[0]):
    datum = cpp_kmat_bsr[i,:]

    ielem = datum[0]
    gblock = datum[1]
    elem_block = datum[2]
    erow = datum[3]
    ecol = datum[4]
    value = datum[5]

    irow = erow % 6
    icol = ecol % 6

    grow = -1
    gcol = -1

    # find matching grow, gcol in row and colPtr
    found_val = False
    for block_row in range(nnodes):
        istart = rowPtr[block_row]
        iend = rowPtr[block_row+1]

        for block_ind in range(istart, iend):
            block_col = colPtr[block_ind]

            if block_ind == gblock:
                found_val = True
                grow = 6 * block_row + irow
                gcol = 6 * block_col + icol
                break
        if found_val: break

    new_df_dict['ielem'] += [ielem]
    new_df_dict['grow'] += [grow]
    new_df_dict['gcol'] += [gcol]
    new_df_dict['gblock'] += [gblock]
    new_df_dict['elem_block'] += [elem_block]
    new_df_dict['erow'] += [erow]
    new_df_dict['ecol'] += [ecol]
    new_df_dict['value'] += [value]

for key in ['ielem', 'grow', 'gcol', 'gblock', 'elem_block', 'erow', 'ecol']:
    new_df_dict[key] = np.array(new_df_dict[key]).astype(np.int32)

# now sort by ielem
new_df = pd.DataFrame(new_df_dict)
new_df = new_df.sort_values(by=['ielem', 'grow', 'gcol'], ascending=True)
# new_df = new_df.reset_index()
new_df.to_csv(folder + "cpp-kmat-assembly.csv", float_format='%.4e', index=False)

bsr_df = pd.read_csv(folder + "cpp-kmat-bsr.csv")
bsr_df = bsr_df.sort_values(by=['ielem', 'gblock'], ascending=True)
bsr_df.to_csv(folder + "cpp-kmat-bsr2.csv", float_format='%.4e', index=False)