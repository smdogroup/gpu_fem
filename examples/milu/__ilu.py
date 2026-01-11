import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from __src import plot_plate_vec

def gaussJordan(A, RHS):
    # do gauss-jordan A => Ainv factorization in-place
    # like [A, I] => [A', Ainv] or something?

    # A_err = A.copy()
    # B_err = RHS.copy()

    perm = np.arange(0, 6)

    for i in range(6):
        # find the pivot column
        ipeak = i
        for j in range(i+1, 6):
            if abs(A[j,i]) > abs(A[ipeak,i]):
                ipeak = j

        if ipeak != i:
            # swap i with ipeak in perm
            # maybe because the row-perm doens't affect Linv * Uinv?
            # cause cols not permuted
            _temp = perm[ipeak]
            perm[ipeak] = perm[i]
            perm[i] = _temp

        # check for singular pivot
        if abs(A[ipeak,i]) < 1e-35:
            print(f"GJ (6x6) helper: singular pivot ({ipeak},{i})")
            return True, perm # failed
        
        # permute if pivot is off-diag
        if ipeak != i:
            for k in range(6):
                # swap values in A
                temp = A[ipeak,k]
                A[ipeak,k] = A[i,k]
                A[i,k] = temp

            for ii in range(6):
                # swap values in RHS
                temp = RHS[ipeak,ii]
                RHS[ipeak,ii] = RHS[i,ii]
                RHS[i,ii] = temp

        # print(f"{A=}")

        # zero out appropriate columns
        for j in range(6):
            if j != i:
                factor = -A[j,i] / A[i,i]
                # print(f"{factor=}")clear
                for k in range(6):
                    A[j,k] += A[i,k] * factor
                for ii in range(6):
                    RHS[j,ii] += RHS[i,ii] * factor

    # now get solution by dividing by the diagonal
    for i in range(6):
        recip_Aii = 1.0 / A[i,i]
        for ii in range(6):
            RHS[i,ii] *= recip_Aii
    return False, perm # not failed (aka success)

def _get_diagp(A_bsr):
    # locate diagonal blocks
    assert isinstance(A_bsr, sp.bsr_matrix)
    assert A_bsr.blocksize == (6,6)
    rowp, cols = A_bsr.indptr, A_bsr.indices
    nnodes = A_bsr.shape[0] // 6

    diagp = np.full(nnodes, -1, dtype=int)
    for brow in range(nnodes):
        for jp in range(rowp[brow], rowp[brow+1]):
            if cols[jp] == brow:
                diagp[brow] = jp
                break
        if diagp[brow] < 0:
            raise RuntimeError(f"Missing diagonal block at row {brow=}")
    return diagp


def extract_bsr_triangular(A_bsr, lower:bool=True):
    # print(f"{A_bsr.indptr=}\n{A_bsr.indices=}\n")

    # extract either the L or U bsr matrix
    nnodes = A_bsr.shape[0] // 6
    rows = [[] for i in range(nnodes)]
    cols = [[] for i in range(nnodes)]
    data = [[] for i in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A_bsr.indptr[i], A_bsr.indptr[i+1]):
            j = A_bsr.indices[jp]
            if i >= j and lower:
                rows[i] += [i]
                cols[i] += [j]
                data[i] += [np.eye(6)]
            elif i <= j and not(lower):
                rows[i] += [i]
                cols[i] += [j]
                data[i] += [np.eye(6)]
    
    nrows = nnodes
    ncols = nnodes
    # assemble bsr
    indptr = np.zeros(nrows + 1, dtype=int)
    indices = []
    blocks = []

    # print(f"{rows=}\n{cols=}")
    for i in range(nrows):
        indptr[i+1] = indptr[i] + len(cols[i])
        indices.extend(cols[i])
        blocks.extend(data[i])
    # print(f"{indptr=}\n{indices=}\n")
    bs = 6

    return sp.bsr_matrix(
        (np.array(blocks), np.array(indices), indptr),
        shape=(nrows * bs, ncols * bs),
        blocksize=(bs, bs)
    )

def sort_bsr_indices(A):
    """
    Return a BSR matrix with sorted column indices per block row.
    No duplicate handling.
    """
    assert sp.isspmatrix_bsr(A)

    indptr = A.indptr
    indices = A.indices
    data = A.data

    new_indices = indices.copy()
    new_data = data.copy()

    for i in range(A.indptr.size - 1):
        p0, p1 = indptr[i], indptr[i+1]
        if p1 - p0 <= 1:
            continue

        order = np.argsort(indices[p0:p1], kind="stable")
        new_indices[p0:p1] = indices[p0:p1][order]
        new_data[p0:p1] = data[p0:p1][order]

    return sp.bsr_matrix(
        (new_data, new_indices, indptr.copy()),
        shape=A.shape,
        blocksize=A.blocksize,
    )

def block_ilu6_gj_factor(A_bsr):
    """
    ILU factorization based on Saad chapter 7 for BSR matrix:
        * no pivoting between nodes
        * pivoting performed in each 6x6 block nodal matrix using Gauss-Jordan exact inverse
    """

    # print(f"{A_bsr.shape=}")
    # print(f"factor: {A_bsr.indptr=}\n{A_bsr.indices=}\n")

    # preamble (allocation)
    assert isinstance(A_bsr, sp.bsr_matrix)
    assert A_bsr.blocksize == (6,6)
    A_copy = A_bsr.copy()
    rowp = A_copy.indptr
    cols = A_copy.indices
    data = A_copy.data
    nnodes = A_copy.shape[0] // 6
    # null_val = 0
    null_val = -1 # prob should be -1
    iw = np.full(nnodes, null_val, dtype=int)
    diagp = _get_diagp(A_bsr)
    
        
    # ILU(0) with Gauss-Jordan solves (Saad chapter 7)
    for k in range(nnodes):
        j1 = rowp[k]
        j2 = rowp[k+1] - 1
    
        for j in range(j1, j2+1):
            iw[cols[j]] = j
        
        # also based on ilu_generic_template.h
        # which is easier to read

        j = j1 # lower triangular iteration here
        while (j <= j2):
            jrow = cols[j]
            if (jrow >= k): 
                break
            else:
                # make a temp matrix
                # tmat = np.zeros((6,6), dtype=np.double)
                tmat = data[j] @ data[diagp[jrow]]

                if cols[j] < nnodes:
                    data[j] = tmat.copy()

                # upper triangular iteration
                for jj in range(diagp[jrow] + 1, rowp[jrow+1]):
                    # rowp to cols jj
                    jw = iw[cols[jj]]

                    if jw != null_val:
                        if cols[jw] < nnodes and cols[jj] < nnodes:
                            data[jw] -= tmat @ data[jj]

            j += 1
        # done with matmult loop in crout ILU? is this crout ILU?

        diagp[k] = j
        if jrow != k:
            print(f"zero pivot {k=} {jrow=} {j=} in ILUGJ stopping\n")
            return

        # now do diagonal factor with gauss-jordan inverse 6x6 block
        tmat = data[j].copy()
        data[j] = np.eye(6).astype(A_bsr.data.dtype)

        fail, _ = gaussJordan(tmat, data[j])
        if fail:
            print(f"gaussJordan solve failed on node block {j=} in ILUGJ\n")
            return
        
        # reset iw pointer
        for j in range(j1, j2+1):
            iw[cols[j]] = null_val

    return A_copy

        
def block_ilu6_gj_solve(A_lu, x, y):
    """
    ILU solve based on Saad chapter 7, for BSR matrix 6x6 block sizes
        * 6x6 block solves use exact inverse from Gauss-Jordan (intra-block pivoting)
        * no pivoting between blocks
    """

    # preamble (allocation)
    assert isinstance(A_lu, sp.bsr_matrix)
    assert A_lu.blocksize == (6,6)
    rowp = A_lu.indptr
    cols = A_lu.indices
    data = A_lu.data
    nnodes = A_lu.shape[0] // 6
    diagp = _get_diagp(A_lu)

    # forward block solve
    for i in range(nnodes):
        m1 = rowp[i]
        m2 = diagp[i]-1

        # this part is based on ilu_generic_template.h
        # which is easier to read
        for l1 in range(6):
            x[6 * i + l1] = y[6 * i + l1] * 1.0

        for k in range(m1, m2+1):
            start = 6 * cols[k]
            end = start + 6
            tmp = np.dot(data[k], x[start:end])    
            for l1 in range(6):
                x[6 * i + l1] -= tmp[l1]

    # backwards block solve
    for i in range(nnodes-1, -1, -1):
        m1 = diagp[i] + 1
        m2 = rowp[i+1] - 1

        for k in range(m1, m2+1):
            start = 6 * cols[k]
            end = start + 6
            tmp = np.dot(data[k], x[start:end])    
            for l1 in range(6):
                x[6 * i + l1] -= tmp[l1]

        # and then diagonal part
        start = 6 * i
        end = start + 6
        tmp = np.dot(data[diagp[i]], x[start:end])
        for m in range(6):
            x[6 * i + m] = tmp[m]

    return False # fail



class GaussJordanBlockPrecond:
    def __init__(self, A):
        self.A = block_ilu6_gj_factor(A.copy())

    def solve(self, rhs):
        x = np.zeros_like(rhs)
        block_ilu6_gj_solve(self.A, x, rhs)
        return x
 

def block_row_bandwidth(A_bsr):

    rowp = A_bsr.indptr
    cols = A_bsr.indices
    nnodes = A_bsr.shape[0] // 6

    bw = np.zeros(nnodes, dtype=int)
    for i in range(nnodes):
        j1, j2 = rowp[i], rowp[i+1]
        if j2 > j1:
            bw[i] = np.max(np.abs(cols[j1:j2] - i))
    return bw

def q_ordering(A_bsr, prune_factor=1.0, seed=None):
    nnodes = A_bsr.shape[0] // 6
    rng = np.random.default_rng(seed)

    # block bandwidth per row
    bw = block_row_bandwidth(A_bsr)

    # choose a characteristic bandwidth
    bw_char = int(np.max(bw))   # or np.mean(bw)
    group_size = max(1, int(prune_factor * bw_char))

    # form contiguous groups in ORIGINAL order
    groups = [
        np.arange(i, min(i + group_size, nnodes))
        for i in range(0, nnodes, group_size)
    ]

    # randomly permute the GROUPS
    rng.shuffle(groups)

    # flatten
    perm = np.concatenate(groups)

    # inverse permutation
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(nnodes)

    return perm, iperm

def get_lu_residual(A0, A_lu):
    # convert all diagonals from inv to orig in A_lu
    rowp = A0.indptr
    cols = A0.indices
    nnodes = A0.shape[0] // 6

    # re-invert diagonals
    L = A_lu.copy()
    diagp = _get_diagp(L)
    for i in range(nnodes):
        jp = diagp[i]
        tmp = L.data[jp].copy()
        B = np.eye(6).astype(A0.data.dtype)
        gaussJordan(tmp, B)
        L.data[jp] = B.copy()

    # now split into L and U matrices
    U = L.copy()
    for i in range(nnodes):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            if i > j: # lower-triangular
                U.data[jp] *= 0.0
            elif i == j:
                L.data[jp] = np.eye(6)
            elif i < j: # upper-triangular
                L.data[jp] *= 0.0
    
    # compute factor residual
    R = L @ U - A0
    return R

