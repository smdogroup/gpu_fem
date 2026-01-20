import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from typing import Self

#############################
# Helper routines (CSR version, unchanged)
#############################
def remove_dirichlet_nondiag_entries(A: sp.csr_matrix):
    rows, cols, data = sp.find(A)
    threshold = 1e-10
    mask = np.abs(data) > threshold
    rows, cols, data = rows[mask], cols[mask], data[mask]
    A_clean = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
    return A_clean

def _norm1(A):
    col_sums = []
    for j in range(A.shape[1]):
        s = 0.0
        for i in range(A.shape[0]):
            s += abs(A[i, j])
        col_sums.append(s)
    return max(col_sums)

##########################################
# Original CSR solver classes (unchanged)
##########################################
class SparseVec:
    def __init__(self, inds: list):
        self.inds = inds
        self.N = len(self.inds)
        self.data = np.zeros(self.N)

    def set_value(self, ind: int, val: float):
        # set the entry corresponding to index ind to val.
        for i, ii in enumerate(self.inds):
            if ii == ind:
                self.data[i] = val

    def __add__(self, other: Self) -> Self:
        new_obj = SparseVec(self.inds)
        new_obj.data[:] = self.data + other.data
        return new_obj

    def __sub__(self, other: Self) -> Self:
        new_obj = SparseVec(self.inds)
        new_obj.data[:] = self.data - other.data
        return new_obj

def spmat_vec(mat: sp.csr_matrix, vec1: SparseVec) -> SparseVec:
    vec2 = SparseVec(vec1.inds)
    for i_idx, row in enumerate(vec2.inds):
        temp = 0.0
        for jp in range(mat.indptr[row], mat.indptr[row+1]):
            col = mat.indices[jp]
            if col in vec1.inds:
                j_idx = vec1.inds.index(col)
                temp += vec1.data[j_idx] * mat.data[jp]
        vec2.data[i_idx] = temp
    return vec2

def set_spmat_col(mat: sp.csr_matrix, vec1: SparseVec, column: int):
    for i_idx, row in enumerate(vec1.inds):
        for jp in range(mat.indptr[row], mat.indptr[row+1]):
            if column == mat.indices[jp]:
                mat.data[jp] = vec1.data[i_idx]

class SPAI_CSRSolver:
    def __init__(self, K_csr: sp.csr_matrix, iters: int = 10, power: int = 2):
        self.K = K_csr.copy()
        self.N = K_csr.shape[0]
        self.iters = iters
        self.M = self._get_power_sparsity(power)
        self.compute_precond()

    def _get_power_sparsity(self, power: int):
        rowp0 = self.K.indptr
        cols0 = self.K.indices
        if power == 1:
            nnz = self.K.nnz
            rowp = rowp0.copy()
            cols = cols0.copy()
        rowp_prev = rowp0.copy()
        cols_prev = cols0.copy()
        for _ in range(power - 1):
            rowp = np.zeros(self.N + 1, dtype=np.int32)
            for row in range(self.N):
                loc_cols = []
                for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                    inner = cols_prev[inner_ptr]
                    for jp in range(rowp0[inner], rowp0[inner+1]):
                        col = cols0[jp]
                        loc_cols.append(col)
                loc_cols = np.unique(np.array(loc_cols))
                rowp[row+1] = rowp[row] + loc_cols.shape[0]
            nnz = rowp[-1]
            cols = np.zeros(nnz, dtype=np.int32)
            for row in range(self.N):
                loc_cols = []
                for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                    inner = cols_prev[inner_ptr]
                    for jp in range(rowp0[inner], rowp0[inner+1]):
                        col = cols0[jp]
                        loc_cols.append(col)
                loc_cols = np.unique(np.array(loc_cols))
                start = rowp[row]
                for idx, col in enumerate(loc_cols):
                    cols[start + idx] = col
            rowp_prev = rowp.copy()
            cols_prev = cols.copy()
        norm1 = _norm1(self.K)
        inorm1 = 1.0 / norm1
        data = np.zeros(nnz, dtype=np.double)
        for i in range(self.N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]
                if i == j:
                    data[jp] = inorm1
        M_0 = sp.csr_matrix((data, cols, rowp), shape=(self.N, self.N))
        return M_0

    def _get_sparse_vec(self, column: int):
        inds = []
        for row in range(self.N):
            for jp in range(self.M.indptr[row], self.M.indptr[row+1]):
                if self.M.indices[jp] == column:
                    inds.append(row)
                    break
        inds = sorted(set(inds))
        return SparseVec(inds)

    def compute_precond(self):
        for j in range(self.N):
            ej = self._get_sparse_vec(j)
            ej.set_value(j, 1.0)
            s = spmat_vec(self.M, ej)
            for _ in range(self.iters):
                r = ej - spmat_vec(self.K, s)
                z = spmat_vec(self.M, r)
                q = spmat_vec(self.K, z)
                alpha = np.dot(r.data, q.data) / (1e-12 + np.dot(q.data, q.data))
                z.data[:] *= alpha
                s = s + z
            set_spmat_col(self.M, s, column=j)
        return self.M

    def solve(self, rhs):
        return self.M.dot(rhs)

    @property
    def operator_complexity(self) -> float:
        return self.M.nnz / self.K.nnz

######################################################
# NEW: Block (BSR) versions (with block size 6×6)
# 
# In this revision, BSparseBlockVec now stores “block‐vector” entries as 
# vectors of length block_dim (6) rather than 6×6 matrices.
#
# To compute a block column of the inverse preconditioner, we loop over
# the local column indices 0,...,5. For each local column we compute a BSparseBlockVec,
# update it with a scalar‐MINRES scheme (that is, treating each block as a 6‐vector),
# and then place that vector into the corresponding column of the blocks in M.
######################################################
class BSparseBlockVec:
    def __init__(self, inds: list, block_dim: int = 6):
        # inds: list of block row indices where the block vector is nonzero.
        self.inds = inds
        self.block_dim = block_dim
        # data now has shape (number of nonzero blocks, block_dim)
        self.data = np.zeros((len(self.inds), block_dim), dtype=np.double)
    
    def set_block(self, block_index: int, vec):
        # Set the block corresponding to the given block_index to the given vector (of length block_dim).
        for i, bi in enumerate(self.inds):
            if bi == block_index:
                self.data[i] = vec

    def __add__(self, other: Self) -> Self:
        new_obj = BSparseBlockVec(self.inds, self.block_dim)
        new_obj.data = self.data + other.data
        return new_obj

    def __sub__(self, other: Self) -> Self:
        new_obj = BSparseBlockVec(self.inds, self.block_dim)
        new_obj.data = self.data - other.data
        return new_obj

def block_spmat_matvec(mat: sp.bsr_matrix, vec: BSparseBlockVec) -> BSparseBlockVec:
    """
    Multiply a BSR matrix (with block size block_dim x block_dim) by a BSparseBlockVec.
    Here each nonzero entry in vec is a vector of length block_dim.
    """
    block_dim = vec.block_dim
    res = BSparseBlockVec(vec.inds, block_dim)
    # For each block row in the result (using the same block indices as in vec)
    for i_idx, i in enumerate(vec.inds):
        out_vec = np.zeros(block_dim, dtype=np.double)
        row_start = mat.indptr[i]
        row_end = mat.indptr[i+1]
        for ptr in range(row_start, row_end):
            j = mat.indices[ptr]
            if j in vec.inds:
                j_idx = vec.inds.index(j)
                # mat.data[ptr] is a (block_dim, block_dim) block; multiply by vec block (block_dim,)
                out_vec += mat.data[ptr].dot(vec.data[j_idx])
        res.data[i_idx] = out_vec
    return res

def block_set_spmat_col(mat: sp.bsr_matrix, vec: BSparseBlockVec, column: int, local_col: int):
    """
    For each block row i in vec.inds, update the corresponding block in the block column 'column'
    of the BSR matrix mat by replacing its local column (local_col) with the computed vector.
    That is, if the block M[i, column] exists (according to mat.indices), then update:
         M[i, column][:, local_col] = vec corresponding entry.
    """
    block_dim = vec.block_dim
    for i_idx, i in enumerate(vec.inds):
        row_start = mat.indptr[i]
        row_end = mat.indptr[i+1]
        for ptr in range(row_start, row_end):
            if mat.indices[ptr] == column:
                # Update the local column of the block.
                # (Make a copy to avoid side effects.)
                block = mat.data[ptr].copy()
                block[:, local_col] = vec.data[i_idx]
                mat.data[ptr] = block

class SPAI_BSRSolver:
    """
    Block sparse approximate inverse (SPAI) preconditioner
    using a BSR matrix with block size 6×6. In this version we compute the
    inverse one column (per block) at a time.
    """
    def __init__(self, K_bsr: sp.bsr_matrix, iters: int = 10, power: int = 2):
        self.K = K_bsr.copy()
        bs_row, bs_col = self.K.blocksize
        if bs_row != 6 or bs_col != 6:
            raise ValueError("K_bsr must have block size 6×6.")
        self.bs = bs_row
        self.nblocks = self.K.shape[0] // self.bs
        self.iters = iters
        self.M = self._get_power_sparsity(power)
        self.compute_precond()

    def _get_power_sparsity(self, power: int):
        """
        Compute the block-level sparsity pattern of K^power.
        For each block row, start with the set of block columns that are present, 
        update the pattern (power-1) times, and then form a BSR matrix with that structure.
        Diagonal blocks are initialized to (1/||K||₁)*I.
        """
        n = self.nblocks
        pattern = [set() for _ in range(n)]
        for i in range(n):
            row_start = self.K.indptr[i]
            row_end = self.K.indptr[i+1]
            pattern[i] = set(self.K.indices[row_start:row_end])
        for _ in range(power - 1):
            new_pattern = [set() for _ in range(n)]
            for i in range(n):
                for j in pattern[i]:
                    new_pattern[i].update(pattern[j])
            pattern = new_pattern
        rowptr = np.zeros(n+1, dtype=np.int32)
        indices_list = []
        for i in range(n):
            cols = sorted(pattern[i])
            indices_list.extend(cols)
            rowptr[i+1] = rowptr[i] + len(cols)
        nnzb = rowptr[-1]
        indices = np.array(indices_list, dtype=np.int32)
        norm1 = sp.linalg.norm(self.K, ord=1)
        inorm1 = 1.0 / norm1
        # Allocate blocks: each block is 6×6.
        data = np.zeros((nnzb, self.bs, self.bs), dtype=np.double)
        for i in range(n):
            for ptr in range(rowptr[i], rowptr[i+1]):
                j = indices[ptr]
                if i == j:
                    data[ptr] = inorm1 * np.eye(self.bs)
        M0 = sp.bsr_matrix((data, indices, rowptr), shape=(n*self.bs, n*self.bs))
        return M0

    def _get_block_sparse_vec(self, column: int) -> BSparseBlockVec:
        """
        For a given block column (of the preconditioner M), extract the block pattern (row indices)
        from M. (Each block row i is included if M[i, column] exists.)
        """
        inds = []
        for i in range(self.nblocks):
            for ptr in range(self.M.indptr[i], self.M.indptr[i+1]):
                if self.M.indices[ptr] == column:
                    inds.append(i)
                    break
        inds = sorted(set(inds))
        return BSparseBlockVec(inds, self.bs)

    def compute_precond(self):
        """
        Compute the approximate inverse preconditioner, column by column.
        For each block column j (0 <= j < nblocks) and for each local column index (0 <= v < bs),
        we solve for the (v)-th column of M[:, j] using a MINRES-like iterative update.
        """
        for j in range(self.nblocks):
            # Loop over the local column indices within the block j.
            for local_col in range(self.bs):
                # Create the canonical block vector E for the current local column.
                ej = self._get_block_sparse_vec(j)
                # For the block corresponding to the jth block, set its entry to the unit vector with 1 in the local_col position.
                unit_vec = np.zeros(self.bs, dtype=np.double)
                unit_vec[local_col] = 1.0
                ej.set_block(j, unit_vec)
                # Compute the initial approximation s = M*E.
                s = block_spmat_matvec(self.M, ej)
                for _ in range(self.iters):
                    r = ej - block_spmat_matvec(self.K, s)
                    z = block_spmat_matvec(self.M, r)
                    q = block_spmat_matvec(self.K, z)
                    num = 0.0
                    den = 0.0
                    for k in range(len(q.inds)):
                        num += np.dot(r.data[k], q.data[k])
                        den += np.dot(q.data[k], q.data[k])
                    alpha = num / (den + 1e-12)
                    # Update s: s = s + alpha * z.
                    s.data[:] = s.data + alpha * z.data
                # Now update the jth block column of the preconditioner M.
                # For each block row in the pattern, update the local column of M[i,j].
                block_set_spmat_col(self.M, s, column=j, local_col=local_col)
        return self.M

    def solve(self, rhs):
        return self.M.dot(rhs)

    @property
    def operator_complexity(self) -> float:
        return self.M.nnz / self.K.nnz
