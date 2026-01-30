import numpy as np

def build_csr_from_conn(conn, nnodes, symmetric=True):
    """
    Build CSR sparsity (rowp, cols, nnzb) from element connectivity.

    Parameters
    ----------
    conn : list[list[int]]
        Element connectivity: each entry is a list of global node indices in that element.
    nnodes : int
        Total number of nodes.
    symmetric : bool
        If True, add both (i,j) and (j,i). For most FE stiffness matrices, True.

    Returns
    -------
    rowp : np.ndarray shape (nnodes+1,)
    cols : np.ndarray shape (nnzb,)
    nnzb : int
    """
    # adjacency sets per row
    adj = [set() for _ in range(nnodes)]

    for elem_nodes in conn:
        # add all node-to-node couplings inside this element
        for a in elem_nodes:
            if a < 0 or a >= nnodes:
                raise ValueError(f"Node index {a} out of bounds 0..{nnodes-1}")
            row_set = adj[a]
            for b in elem_nodes:
                row_set.add(b)
                if symmetric:
                    # redundant given looping over all a anyway, but harmless; kept for clarity
                    pass

    # convert adjacency to CSR (sorted columns per row)
    rowp = np.zeros(nnodes + 1, dtype=np.int64)
    cols_list = []
    nnzb = 0
    for i in range(nnodes):
        row_cols = sorted(adj[i])
        cols_list.extend(row_cols)
        nnzb += len(row_cols)
        rowp[i + 1] = nnzb

    cols = np.array(cols_list, dtype=np.int64)
    return rowp, cols, nnzb
