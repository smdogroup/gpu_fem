import numpy as np
from scipy.sparse import bsr_matrix

def load_bsr_matrix(rowp_file, cols_file, kmat_file, block_dim=6, debug=True):
    # Load row pointer (indptr)
    rowptr = np.loadtxt(rowp_file, delimiter=",", dtype=int)
    
    # Load column indices
    cols = np.loadtxt(cols_file, delimiter=",", dtype=int)
    
    # Load block data (flattened)
    data = np.loadtxt(kmat_file, delimiter=",", dtype=float)
    
    # Reshape into blocks
    nblocks = len(cols)
    data = data.reshape((nblocks, block_dim, block_dim))
    
    # Build BSR matrix
    n_block_rows = len(rowptr) - 1
    N = n_block_rows * block_dim
    shape = (N, N)
    A = bsr_matrix((data, cols, rowptr), shape=shape, blocksize=(block_dim, block_dim))
    
    if debug:
        print("=== Debug Info ===")
        print(f"Rowptr (indptr): {rowptr}")
        print(f"Cols: {cols}")
        print(f"Data shape (blocks): {data.shape}")
        print(f"Matrix shape: {A.shape}")
        print(f"nnz blocks: {A.nnz}")
        # if A.shape[0] * A.shape[1] <= 400:  # small matrix → print dense
        #     print("Dense matrix:\n", A.toarray())
        # else:
        #     print("Dense matrix too large, skipped.")
    
    return A

if __name__ == "__main__":
    rowp_file = "out/csv/rowp.csv"
    cols_file = "out/csv/cols.csv"
    kmat_file = "out/csv/kmat.csv"
    
    A = load_bsr_matrix(rowp_file, cols_file, kmat_file)

    print(F"{A.shape=}")

    import matplotlib.pyplot as plt
    # plt.spy(A.tocsr())
    # plt.show()

    A_np = A.toarray()
    A_log = np.log(1.0 + A_np**2)
    plt.imshow(A_log)
    plt.show()