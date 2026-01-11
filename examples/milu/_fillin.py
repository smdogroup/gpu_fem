import numpy as np
import scipy as sp
# from ._utils import *


def ilu_k_pattern_bsr(A, k_fill):
    """
    Generate ILU(k) block fill pattern for a BSR matrix A.
    Copies data from A into the ILU(k) fill-in positions, zeros for new fill-ins.
    Fully BSR-native, no COO conversion.
    """
    assert isinstance(A, sp.sparse.bsr_matrix)
    bs = A.blocksize[0]
    n_blocks = A.shape[0] // bs

    # Initialize level-of-fill matrix
    level = np.full((n_blocks, n_blocks), np.inf)

    # Fill level 0 for existing blocks
    for i_block in range(n_blocks):
        start = A.indptr[i_block]
        end = A.indptr[i_block + 1]
        cols = A.indices[start:end]
        level[i_block, cols] = 0

    # Symbolic ILU(k)
    for i in range(n_blocks):
        for j in range(i):
            if level[i, j] <= k_fill:
                for k in range(j + 1, n_blocks):
                    if level[j, k] < np.inf:
                        new_level = level[i, j] + level[j, k] + 1
                        if new_level < level[i, k]:
                            level[i, k] = new_level

    # ILU(k) pattern
    pattern_bool = (level <= k_fill)
    rows_idx, cols_idx = np.nonzero(pattern_bool)
    n_blocks_total = len(rows_idx)

    # Build A_fillin blocks array
    A_fillin_blocks = np.zeros((n_blocks_total, bs, bs))
    block_counter = 0

    # Copy original data from A
    for i_block in range(n_blocks):
        start = A.indptr[i_block]
        end = A.indptr[i_block + 1]
        for idx in range(start, end):
            j_block = A.indices[idx]
            if pattern_bool[i_block, j_block]:
                # Find the position in A_fillin_blocks
                pos = np.where((rows_idx == i_block) & (cols_idx == j_block))[0][0]
                A_fillin_blocks[pos] = A.data[idx]

    # Build indptr for BSR
    indptr = np.zeros(n_blocks + 1, dtype=int)
    for r in rows_idx:
        indptr[r + 1] += 1
    indptr = np.cumsum(indptr)

    # Build BSR matrix
    A_fillin = sp.sparse.bsr_matrix((A_fillin_blocks, cols_idx, indptr), shape=A.shape)
    return A_fillin

def get_elim_tree(N, rowp, cols):
    # first we compute the elimination tree using Algorithm 4.2 (for sym matrix..)
    parent, ancestor = np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)
    for i in range(N):
        parent[i], ancestor[i] = 0, 0
        
        # for all j such that a_ij neq 0 (this row basically in CSR), do:
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            if not(j < i): continue
            jroot = j

            while ((ancestor[jroot] != 0) and (ancestor[jroot] != i)):
                l = ancestor[jroot]
                ancestor[jroot] = i # path compression to accel future searches
                jroot = l

            if ancestor[jroot] == 0:
                ancestor[jroot] = i
                parent[jroot] = i

    return parent, ancestor

def get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower:bool=True):
    # now compute updated rowp, cols with fillin..
    fill_L_rowp = np.zeros(N+1, dtype=np.int32)
    # first just go through and get row counts.. then we'll come back and alloc cols
    mark = -1 * np.ones(N, dtype=np.int32)

    for i in range(N):
        c_cols = []
        mark[i] = i # encountered diagonal entry

        for jp in range(rowp[i], rowp[i+1]):
            k = cols[jp]
            # this condition gives you only sparsity of L
            if not(k < i): continue
            j = k
            while (mark[j] != i): # while col j has not encountered row i yet
                mark[j] = i
                c_cols += [j]
                j = parent[j]

        ncols = len(c_cols)
        fill_L_rowp[i+1] = fill_L_rowp[i] + ncols

    L_nnz = fill_L_rowp[-1]
    # print(F"{nnz=}")
    # fill_L_rows = np.zeros(L_nnz, dtype=np.int32)
    fill_L_cols = np.zeros(L_nnz, dtype=np.int32)
    mark = -1 * np.ones(N, dtype=np.int32)

    for i in range(N):
        c_cols = []
        mark[i] = i # col i has encountered row i

        for jp in range(rowp[i], rowp[i+1]):
            k = cols[jp]
            # this condition gives you only sparsity of L
            if not(k < i): continue
            j = k
            while (mark[j] != i): # while col j has not encountered row i yet
                mark[j] = i
                c_cols += [j]
                j = parent[j]

        # print(F"{c_cols=}")
        ncols = len(c_cols)
        if ncols > 0:
            c_cols_sort = np.sort(c_cols)
            start, end = fill_L_rowp[i], fill_L_rowp[i+1]
            fill_L_cols[start : end] = c_cols_sort

    if not strict_lower:
        # add diagonal to L (it gives just strict L originally)
        L_rowp = np.zeros(N+1, dtype=np.int32)
        for i in range(N):
            row_ct = fill_L_rowp[i+1] - fill_L_rowp[i]
            L_rowp[i+1] = L_rowp[i] + (row_ct + 1)
        L_nnz = L_rowp[-1]
        L_rows = np.zeros(L_nnz, dtype=np.int32)
        for i in range(N):
            for jp in range(L_rowp[i], L_rowp[i+1]):
                L_rows[jp] = i
        L_cols = np.zeros(L_nnz, dtype=np.int32)
        for i in range(N):
            strict_start = fill_L_rowp[i]
            start = L_rowp[i]
            for jp in range(fill_L_rowp[i], fill_L_rowp[i+1]):
                offset = jp - strict_start
                L_cols[start + offset] = fill_L_cols[jp]
            L_cols[L_rowp[i+1]-1] = i # diag entry

        fill_L_rowp, fill_L_cols = L_rowp, L_cols

    return fill_L_rowp, fill_L_cols

def get_lower_triang_pattern(A, strict_lower:bool=False):
    # get general lower triang pattern from A
    nnodes = A.indptr.shape[0] - 1 # more general way (num nodes instead of num dof) for CSR or BSR matrix
    rowp, cols = A.indptr, A.indices

    L_row_cts = np.zeros(nnodes, dtype=np.int32)
    for i in range(nnodes):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            if strict_lower and i > j:
                L_row_cts[i] += 1
            elif not(strict_lower) and i >= j:
                L_row_cts[i] += 1

    L_rowp = np.zeros(nnodes+1, dtype=np.int32)
    for i in range(nnodes):
        L_rowp[i+1] = L_rowp[i] + L_row_cts[i]

    nnz = L_rowp[-1]
    L_rows = np.zeros(nnz, dtype=np.int32)
    for i in range(nnodes):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            L_rows[jp] = i

    L_cols = np.zeros(nnz, dtype=np.int32)
    next = np.zeros(nnodes, dtype=np.int32) # keeps track of how much we've filled each row in CSR format
    for i in range(nnodes):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]

            insert = False
            if strict_lower and i > j:
                insert = True
            elif not(strict_lower) and i >= j:
                insert = True

            if insert:
                start = L_rowp[i]
                L_cols[start + next[i]] = j
                next[i] += 1
    return L_rowp, L_cols

def get_upper_triang_pattern(A, strict_upper:bool=False):
    # get general upper triang pattern from A
    nnodes = A.indptr.shape[0] - 1 # more general way (num nodes instead of num dof) for CSR or BSR matrix
    rowp, cols = A.indptr, A.indices

    U_row_cts = np.zeros(nnodes, dtype=np.int32)
    for i in range(nnodes):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            if strict_upper and i < j:
                U_row_cts[i] += 1
            elif not(strict_upper) and i <= j:
                U_row_cts[i] += 1

    U_rowp = np.zeros(nnodes+1, dtype=np.int32)
    for i in range(nnodes):
        U_rowp[i+1] = U_rowp[i] + U_row_cts[i]

    nnz = U_rowp[-1]
    U_rows = np.zeros(nnz, dtype=np.int32)
    for i in range(nnodes):
        for jp in range(U_rowp[i], U_rowp[i+1]):
            U_rows[jp] = i

    U_cols = np.zeros(nnz, dtype=np.int32)
    next = np.zeros(nnodes, dtype=np.int32) # keeps track of how much we've filled each row in CSR format
    for i in range(nnodes):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]

            insert = False
            if strict_upper and i < j:
                insert = True
            elif not(strict_upper) and i <= j:
                insert = True

            if insert:
                start = U_rowp[i]
                U_cols[start + next[i]] = j
                next[i] += 1
    return U_rowp, U_cols

def get_transpose_pattern(N, rowp, cols, get_map:bool=False):
    # general get transpose pattern
    # construct also the strict upper triangular part..
    # so we can slice by columns as well..
    tr_row_cts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            tr_row_cts[j] += 1

    tr_rowp = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        tr_rowp[i+1] = tr_rowp[i] + tr_row_cts[i]

    nnz = tr_rowp[-1]
    tr_rows = np.zeros(nnz, dtype=np.int32)
    for i in range(N):
        for jp in range(tr_rowp[i], tr_rowp[i+1]):
            tr_rows[jp] = i

    tr_cols = np.zeros(nnz, dtype=np.int32)
    next = np.zeros(N, dtype=np.int32) # keeps track of how much we've filled each row in CSR format
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            # flip i,j as row,col => j,i in our head
            start = tr_rowp[j]
            tr_cols[start + next[j]] = i
            next[j] += 1

    # map from nnz in transpose to un-transpose storage
    if get_map:
        tr_map = np.zeros(nnz, dtype=np.int32)
        for i in range(N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]

                # find equivalent indptr in transpose matrix
                for ip in range(tr_rowp[j], tr_rowp[j+1]):
                    i2 = tr_cols[ip]
                    if i == i2:
                        tr_map[ip] = jp
                        break
        print(f"{tr_map=}")
    else:
        tr_map = None

    return tr_rowp, tr_cols, tr_map

def get_rows_from_rowp(N, rowp):
    nnz = rowp[-1]
    rows = np.zeros(nnz, dtype=np.int32)
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            rows[jp] = i
    return rows

def strict_L_to_full_LU_patern(N, L_rowp, L_cols):
    # now take strict lower triang sparsity L and compute full fillin sparsity fill_rowp, fill_cols
    fill_row_cts = np.ones(N, dtype=np.int32) # start with 1 for diags
    for i in range(N):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]

            # add one nz above and below diag
            fill_row_cts[i] += 1
            fill_row_cts[j] += 1

    # build new rowp
    fill_rowp = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        fill_rowp[i+1] = fill_rowp[i] + fill_row_cts[i]
    fill_nnz = fill_rowp[-1]
    fill_cols = np.zeros(fill_nnz, dtype=np.int32)
    fill_rows = np.zeros(fill_nnz, dtype=np.int32)

    # build rows nnz vec (needed only for python CSR)
    for i in range(N):
        for jp in range(fill_rowp[i], fill_rowp[i+1]):
            fill_rows[jp] = i

    # build new cols
    next = fill_rowp.copy() # tracks fill in per row
    # go through each row, first only putting in below diag + diag
    for i in range(N):
        # put below diag in first..
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            fill_cols[next[i]] = j
            next[i] += 1

        # add diagonal
        fill_cols[next[i]] = i
        next[i] += 1
        
    # now go back and add above diag in..
    for i in range(N):
        # add above diag
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            
            # i,j as row,col now flipped for above diag
            fill_cols[next[j]] = i
            next[j] += 1
    
    return fill_rowp, fill_rows, fill_cols

def compute_LU_fill_pattern(N, rowp, cols):
    """procedure for full LU fill pattern from symmetric square matrix"""
    parent, ancestor = get_elim_tree(N, rowp, cols)
    L_rowp, L_cols = get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower=True)
    fill_rowp, fill_rows, fill_cols = strict_L_to_full_LU_patern(N, L_rowp, L_cols)
    return fill_rowp, fill_rows, fill_cols

def get_LU_fill_matrix(A_orig):
    """full process of getting LU fill pattern and copying values for symmetric square matrix"""

    # get fill pattern first from original matrix
    N = A_orig.shape[0]
    rowp, cols = A_orig.indptr, A_orig.indices
    if isinstance(A_orig, sp.sparse.csr_matrix):
        block_dim = 1
    elif isinstance(A_orig, sp.sparse.bsr_matrix):
        block_dim = A_orig.data.shape[-1]
    nnodes = N // block_dim

    fill_rowp, fill_rows, fill_cols = compute_LU_fill_pattern(nnodes, rowp, cols)
    fill_nnzb = fill_rowp[-1]

    if isinstance(A_orig, sp.sparse.csr_matrix):
        fill_vals = np.zeros(fill_nnzb, dtype=np.double)
        A_fill = sp.sparse.csr_matrix((fill_vals, (fill_rows, fill_cols)), shape=(N, N))

        # copy values from nofill sparsity
        for i in range(N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]
                A_fill[i,j] = A_orig[i,j]

    elif isinstance(A_orig, sp.sparse.bsr_matrix):
        fill_vals = np.zeros((fill_nnzb, block_dim, block_dim), dtype=np.double)
        
        # copy values
        inzb = 0
        _next = fill_rowp[:-1].copy() # copy fill rowp, temp util array for inserting block vals
        for ib in range(nnodes):
            for jp in range(rowp[ib], rowp[ib+1]):
                jb = cols[jp]
                for inz in range(block_dim**2):
                    ii, jj = inz % block_dim, inz // block_dim
                    fill_vals[_next[ib], ii, jj] = A_orig.data[jp,ii,jj]
                _next[ib] += 1

        A_fill = sp.sparse.bsr_matrix((fill_vals, fill_cols, fill_rowp), shape=(N, N))
    
    return A_fill
    
def get_fillin_only_pattern(N, rowp, cols, fill_rowp, fill_cols):
    """compute the fillin only pattern (excludes nofill entries)"""

    fillin_row_cts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        nofill_row_ct = rowp[i+1] - rowp[i]
        fill_row_ct = fill_rowp[i+1] - fill_rowp[i]

        fillin_row_cts[i] = fill_row_ct - nofill_row_ct
    
    fillin_rowp = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        fillin_rowp[i+1] = fillin_rowp[i] + fillin_row_cts[i]

    nnz = fillin_rowp[-1]
    fillin_cols = np.zeros(nnz, dtype=np.int32)
    next = fillin_rowp[:-1].copy()
    for i in range(N):
        for jp in range(fill_rowp[i], fill_rowp[i+1]):
            j = fill_cols[jp]
            is_fillin = True
            for jp2 in range(rowp[i], rowp[i+1]):
                j2 = cols[jp2]
                if j == j2: 
                    is_fillin = False
                    break
        
            if is_fillin:
                fillin_cols[next[i]] = j
                next[i] += 1
    return fillin_rowp, fillin_cols