import numpy as np
import scipy.sparse as sp

def sort_bsr_indices(Mb: sp.bsr_matrix, *, sum_duplicates: bool = True, copy: bool = True) -> sp.bsr_matrix:
    """
    Sort BSR block-column indices within each block-row, and permute data accordingly.

    Parameters
    ----------
    Mb : sp.bsr_matrix
        Input matrix (will be converted to BSR if needed).
    sum_duplicates : bool
        If True, call sum_duplicates() after sorting (helpful if you may have duplicate block entries).
    copy : bool
        If True, operate on a copy; otherwise modify in-place when possible.

    Returns
    -------
    sp.bsr_matrix
        BSR matrix with sorted indices within each block-row.
    """
    if not sp.isspmatrix_bsr(Mb):
        Mb = Mb.tobsr()

    M = Mb.copy() if copy else Mb

    indptr = M.indptr
    indices = M.indices
    data = M.data

    n_block_rows = indptr.size - 1

    # Fast path: detect if already sorted per row (optional; cheap-ish)
    # If you don't care, you can remove this check.
    for r in range(n_block_rows):
        s, e = indptr[r], indptr[r + 1]
        if e - s <= 1:
            continue
        row_idx = indices[s:e]
        if np.any(row_idx[:-1] > row_idx[1:]):
            # Need to sort this row
            p = np.argsort(row_idx, kind="mergesort")  # stable
            indices[s:e] = row_idx[p]
            data[s:e] = data[s:e][p]

    if sum_duplicates:
        # If duplicates exist, this will combine them (requires sorted indices to be most effective).
        M.sum_duplicates()

    return M

def debug_print_bsr_matrix(
    Mb: sp.bsr_matrix,
    name: str = "Mb",
    max_blocks: int = None,
    atol: float = 0.0,
    print_structure: bool = True,
    print_blocks: bool = True,
    float_fmt: str = "{: .6e}",
):
    """
    Print BSR debug info similar to your C++ snippet:
      - nnodes_fine (block rows)
      - nnzb (# of blocks)
      - row pointers (indptr), cols (indices)
      - each block's (node_row, node_col) and its entries

    Params
    ------
    Mb : scipy.sparse.bsr_matrix
    name : label in prints
    max_blocks : cap number of blocks printed (None = all)
    atol : skip printing blocks whose abs max entry <= atol
    print_structure : print indptr/indices
    print_blocks : print block contents
    float_fmt : formatting for entries
    """
    if not sp.isspmatrix_bsr(Mb):
        Mb = Mb.tobsr()

    Mb = sort_bsr_indices(Mb, sum_duplicates=True, copy=True)

    br, bc = Mb.blocksize
    if br != bc:
        raise ValueError(f"{name}: expected square blocks, got blocksize={Mb.blocksize}")

    n_block_rows = Mb.shape[0] // br
    n_block_cols = Mb.shape[1] // bc
    nnzb = Mb.indices.size  # number of stored blocks

    print(f"{name}: shape={Mb.shape}, blocksize={Mb.blocksize}")
    print(f"{name}: nnodes_fine (block rows) = {n_block_rows}, block cols = {n_block_cols}")
    print(f"{name}: nnzb (num blocks) = {nnzb}")

    if print_structure:
        print(f"\n{name}.indptr (len={Mb.indptr.size}):")
        print(Mb.indptr)
        print(f"\n{name}.indices (len={Mb.indices.size}):")
        print(Mb.indices)

    if not print_blocks:
        print(f"\ndone with {name} structure (blocks not printed)")
        return

    # Helper to pretty-print one row of a block
    def _print_row(row_vals):
        print("\t" + " ".join(float_fmt.format(x) for x in row_vals))

    printed = 0
    for node_row in range(n_block_rows):
        start = Mb.indptr[node_row]
        end = Mb.indptr[node_row + 1]
        for iblock in range(start, end):
            node_col = int(Mb.indices[iblock])
            block = Mb.data[iblock]  # shape (br, bc)

            if atol > 0.0 and np.max(np.abs(block)) <= atol:
                continue

            print(f"\n\n{name} block node ({node_row},{node_col}):")
            for i in range(br):
                _print_row(block[i, :])

            printed += 1
            if max_blocks is not None and printed >= max_blocks:
                print(f"\n[stopped early after printing {printed} blocks (max_blocks={max_blocks})]")
                print(f"\ndone with {name} matrix vals")
                return

    print(f"\ndone with {name} matrix vals (printed {printed} blocks, skipped tol={atol})")


# ---- Example usage ----
# Mb = some_bsr_matrix
# debug_print_bsr_matrix(Mb, name="h_kmat (G_f^T*G_f + lam*I)", max_blocks=None, atol=0.0)