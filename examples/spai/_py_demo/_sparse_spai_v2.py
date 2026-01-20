import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
# from typing import Self

#############################
# Helper routines (CSR version, unchanged)
#############################

def _norm1_bsr(A: sp.bsr_matrix):
    """
    Compute ||A||_1 (maximum absolute column sum) for a BSR matrix.
    """
    br, bc = A.blocksize
    n_block_rows = A.shape[0] // br
    n_block_cols = A.shape[1] // bc

    # global column sums
    col_sums = np.zeros(A.shape[1])

    indptr = A.indptr
    indices = A.indices
    data = A.data  # shape (nnz_blocks, br, bc)

    for ib in range(n_block_rows):
        start = indptr[ib]
        end = indptr[ib + 1]

        for k in range(start, end):
            jb = indices[k]       # block column index
            block = data[k]       # (br, bc)

            # absolute column sums of this block
            # shape (bc,)
            block_col_sums = np.sum(np.abs(block), axis=0)

            # add into global columns
            j0 = jb * bc
            col_sums[j0:j0 + bc] += block_col_sums

    return np.max(col_sums)

##########################################
# Original CSR solver classes (unchanged)
##########################################


class SPAI_CSRSolver:
    def __init__(self, K_csr: sp.csr_matrix, iters: int = 2, power:int=0, mode:str="global"):
        self.K = K_csr.copy()  # Presumed to be in CSR
        self.mode = mode
        assert mode in ["global", "global-selfmr"]
        self.N = K_csr.shape[0]
        self.iters = iters
        # Obtain a fixed pattern by considering K^power
        self.M = self._get_power_sparsity(power=1, scalar=None) # nofill but diag
        self.I = self._get_power_sparsity(power=1, scalar=1.0)

        # Precompute the SPAI preconditioner
        self.compute_precond()

    def _get_power_sparsity(self, power: int, scalar:float=None):
        rowp0 = self.K.indptr
        cols0 = self.K.indices
        # For power=1, the pattern is just that of K.
        if power == 1:
            nnz = self.K.nnz
            rowp = rowp0.copy()
            cols = cols0.copy()
        else:
            rowp_prev = rowp0.copy()
            cols_prev = cols0.copy()
            for _ in range(power - 1):
                rowp = np.zeros(self.N + 1, dtype=np.int32)
                for row in range(self.N):
                    loc_cols = []
                    for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                        inner = cols_prev[inner_ptr]
                        for jp in range(rowp0[inner], rowp0[inner+1]):
                            loc_cols.append(cols0[jp])
                    # Deduplicate and update row pointer.
                    unique_cols = np.unique(np.array(loc_cols))
                    rowp[row+1] = rowp[row] + unique_cols.shape[0]
                nnz = rowp[-1]
                cols = np.zeros(nnz, dtype=np.int32)
                for row in range(self.N):
                    loc_cols = []
                    for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                        inner = cols_prev[inner_ptr]
                        for jp in range(rowp0[inner], rowp0[inner+1]):
                            loc_cols.append(cols0[jp])
                    unique_cols = np.unique(np.array(loc_cols))
                    start = rowp[row]
                    for idx, col in enumerate(unique_cols):
                        cols[start + idx] = col
                rowp_prev = rowp.copy()
                cols_prev = cols.copy()

        # Build an initial preconditioner: a diagonal scaling.
        if scalar is None:
            norm1_val = _norm1_bsr(self.K)
            inorm1 = 1.0 / norm1_val
            scalar = inorm1
            print(f"{norm1_val=:.4e} {inorm1=:.4e}")
        data = np.zeros(nnz, dtype=np.double)
        # For demonstration, we'll set the diagonal entries (where i==j, if present in the pattern)
        for i in range(self.N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]
                if i == j:
                    data[jp] = scalar
        M_0 = sp.csr_matrix((data, cols, rowp), shape=(self.N, self.N))
        return M_0
    
    def _frob_norm(self, A1:sp.csr_matrix, A2:sp.csr_matrix):
        out = 0.0
        for row in range(A1.shape[0]):
            for jp in range(A1.indptr[row], A1.indptr[row+1]):
                j = A1.indices[jp]
                for jp2 in range(A2.indptr[row], A2.indptr[row+1]):
                    j2 = A2.indices[jp2]
                    if j == j2:
                        out += A1.data[jp] * A2.data[jp2]
                        break
        return out

    def compute_precond(self):
        # Instead of doing plain products then "masking," we use fixed-pattern multiplication.
        # We assume here that the allowable pattern for every product is given by self._pattern_mask.
        for iter_num in range(self.iters):

            obj = self.spai_objective()
            print(f"{iter_num=} {obj=:.4e}")

            if self.mode == "global":
                R = self.I - self.K @ self.M
                AR = self.K @ R
                num = self._frob_norm(R, AR) 
                den = np.dot(AR.data, AR.data) # frob norm of AR w.r.t. itself
                alpha = num / den
                self.M += alpha * R

            elif self.mode == "global-selfmr":
                R = self.I - self.K @ self.M
                Z = self.M @ R
                AZ = self.K @ Z
                num = self._frob_norm(R, AZ) 
                den = np.dot(AZ.data, AZ.data) # frob norm of AR w.r.t. itself
                alpha = num / den
                self.M += alpha * Z

        obj = self.spai_objective()
        print(f"final SPAI {obj=:.4e}")
        return self.M

    def spai_objective(self):
        """compute SPAI objective"""
        R = self.I - self.K @ self.M
        return np.dot(R.data, R.data)

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

class SPAI_BSRSolver:
    def __init__(self, K_bsr: sp.bsr_matrix, iters: int = 2, power:int=0, block_dim:int=6, mode:str="global-selfmr"):
        self.K = K_bsr.copy()  # Presumed to be in CSR
        self.N = K_bsr.shape[0]
        self.iters = iters
        self.block_dim = block_dim
        self.nnodes = self.N // block_dim
        self.mode = mode
        # Obtain a fixed pattern by considering K^power
        self.M = self._get_power_sparsity(power=1, scalar=None) # nofill but diag
        self.I = self._get_power_sparsity(power=1, scalar=1.0)

        # Precompute the SPAI preconditioner
        self.compute_precond()

    def _get_power_sparsity(self, power: int, scalar:float=None):
        rowp0 = self.K.indptr
        cols0 = self.K.indices
        # For power=1, the pattern is just that of K.
        if power == 1:
            nnz = self.K.data.shape[0]
            rowp = rowp0.copy()
            cols = cols0.copy()
        else:
            rowp_prev = rowp0.copy()
            cols_prev = cols0.copy()
            for _ in range(power - 1):
                rowp = np.zeros(self.nnodes + 1, dtype=np.int32)
                for row in range(self.nnodes):
                    loc_cols = []
                    for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                        inner = cols_prev[inner_ptr]
                        for jp in range(rowp0[inner], rowp0[inner+1]):
                            loc_cols.append(cols0[jp])
                    # Deduplicate and update row pointer.
                    unique_cols = np.unique(np.array(loc_cols))
                    rowp[row+1] = rowp[row] + unique_cols.shape[0]
                nnz = rowp[-1]
                cols = np.zeros(nnz, dtype=np.int32)
                # print(f"{nnz=}")
                for row in range(self.nnodes):
                    loc_cols = []
                    for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
                        inner = cols_prev[inner_ptr]
                        for jp in range(rowp0[inner], rowp0[inner+1]):
                            loc_cols.append(cols0[jp])
                    unique_cols = np.unique(np.array(loc_cols))
                    start = rowp[row]
                    for idx, col in enumerate(unique_cols):
                        cols[start + idx] = col
                rowp_prev = rowp.copy()
                cols_prev = cols.copy()

        # Build an initial preconditioner: a diagonal scaling.
        if scalar is None:
            norm1_val = _norm1_bsr(self.K)
            inorm1 = 1.0 / norm1_val
            scalar = inorm1
            print(f"{norm1_val=:.4e} {inorm1=:.4e}")
        data = np.zeros((nnz, self.block_dim, self.block_dim), dtype=np.double)
        # For demonstration, we'll set the diagonal entries (where i==j, if present in the pattern)
        # print(f"{rowp.shape=} {nnz=} {cols.shape=} {data.shape=}")
        for i in range(self.nnodes):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]
                if i == j:
                    data[jp] = scalar * np.eye(self.block_dim)
        M_0 = sp.bsr_matrix((data, cols, rowp), shape=(self.N, self.N))
        return M_0
    
    def _frob_norm(self, A1:sp.csr_matrix, A2:sp.csr_matrix):
        out = 0.0
        for row in range(A1.shape[0] // self.block_dim):
            for jp in range(A1.indptr[row], A1.indptr[row+1]):
                j = A1.indices[jp]
                for jp2 in range(A2.indptr[row], A2.indptr[row+1]):
                    j2 = A2.indices[jp2]
                    if j == j2:
                        out += np.sum(A1.data[jp] * A2.data[jp2])
                        break
        return out

    def compute_precond(self):
        # Instead of doing plain products then "masking," we use fixed-pattern multiplication.
        # We assume here that the allowable pattern for every product is given by self._pattern_mask.
        for iter_num in range(self.iters):
            obj = self.objective()
            print(f"{iter_num=} {obj=:.4e}")

            if self.mode == "global":
                R = self.I - self.K @ self.M
                AR = self.K @ R
                num = self._frob_norm(R, AR) 
                den = np.sum(AR.data * AR.data) # frob norm of AR w.r.t. itself
                alpha = num / den
                self.M += alpha * R

            elif self.mode == "global-selfmr":
                R = self.I - self.K @ self.M
                Z = self.M @ R
                AZ = self.K @ Z # = Q
                num = self._frob_norm(R, AZ) 
                den = np.sum(AZ.data * AZ.data) # frob norm of AR w.r.t. itself
                alpha = num / den
                self.M += alpha * Z
        
        obj = self.objective()
        print(f"final SPAI {obj=:.4e}")

        return self.M
    
    def objective(self) -> float:
        R = self.I - self.K @ self.M
        return np.sum(R.data * R.data)

    def solve(self, rhs):
        return self.M.dot(rhs)

    @property
    def operator_complexity(self) -> float:
        return self.M.nnz / self.K.nnz

# class SPAI_CSRSolver:
#     def __init__(self, K_csr: sp.csr_matrix, iters: int = 10, power: int = 2):
#         self.K = K_csr.copy()  # Presumed to be in CSR
#         self.N = K_csr.shape[0]
#         self.iters = iters
#         # Obtain a fixed pattern by considering K^power
#         self.M = self._get_power_sparsity(power)
#         # Save the binary mask corresponding to allowed fill
#         self._pattern_mask = self.M.copy()
#         self._pattern_mask.data[:] = 1.0
        
#         # For the iterative updates we also use a pattern "I" derived from M where we
#         # want I to have ones on every allowed entry (which, in the simplest case, might be on-diagonals).
#         self.I = self._pattern_mask.copy()

#         # Precompute the SPAI preconditioner
#         self.compute_precond()

#     def _get_power_sparsity(self, power: int):
#         rowp0 = self.K.indptr
#         cols0 = self.K.indices
#         # For power=1, the pattern is just that of K.
#         if power == 1:
#             nnz = self.K.nnz
#             rowp = rowp0.copy()
#             cols = cols0.copy()
#         else:
#             rowp_prev = rowp0.copy()
#             cols_prev = cols0.copy()
#             for _ in range(power - 1):
#                 rowp = np.zeros(self.N + 1, dtype=np.int32)
#                 for row in range(self.N):
#                     loc_cols = []
#                     for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
#                         inner = cols_prev[inner_ptr]
#                         for jp in range(rowp0[inner], rowp0[inner+1]):
#                             loc_cols.append(cols0[jp])
#                     # Deduplicate and update row pointer.
#                     unique_cols = np.unique(np.array(loc_cols))
#                     rowp[row+1] = rowp[row] + unique_cols.shape[0]
#                 nnz = rowp[-1]
#                 cols = np.zeros(nnz, dtype=np.int32)
#                 for row in range(self.N):
#                     loc_cols = []
#                     for inner_ptr in range(rowp_prev[row], rowp_prev[row+1]):
#                         inner = cols_prev[inner_ptr]
#                         for jp in range(rowp0[inner], rowp0[inner+1]):
#                             loc_cols.append(cols0[jp])
#                     unique_cols = np.unique(np.array(loc_cols))
#                     start = rowp[row]
#                     for idx, col in enumerate(unique_cols):
#                         cols[start + idx] = col
#                 rowp_prev = rowp.copy()
#                 cols_prev = cols.copy()

#         # Build an initial preconditioner: a diagonal scaling.
#         norm1_val = _norm1(self.K)
#         inorm1 = 1.0 / norm1_val
#         data = np.zeros(nnz, dtype=np.double)
#         # For demonstration, we'll set the diagonal entries (where i==j, if present in the pattern)
#         for i in range(self.N):
#             for jp in range(rowp[i], rowp[i+1]):
#                 j = cols[jp]
#                 if i == j:
#                     data[jp] = inorm1
#         M_0 = sp.csr_matrix((data, cols, rowp), shape=(self.N, self.N))
#         return M_0

#     def fixed_pattern_matmul(self, A: sp.csr_matrix, B: sp.csr_matrix, pattern: sp.csr_matrix) -> sp.csr_matrix:
#         """
#         Compute the matrix product C = A @ B but only accumulate values in the positions
#         indicated by "pattern" (which is assumed to be in CSR format). Returns a CSR
#         matrix having the same sparsity structure as "pattern."
#         """
#         N = pattern.shape[0]
#         out_data = np.zeros(pattern.nnz, dtype=np.double)
#         # It will have exactly the same indices and indptr as 'pattern'
#         out_indices = pattern.indices.copy()
#         out_indptr = pattern.indptr.copy()

#         # Pre-convert B to CSC, which makes accessing its columns efficient.
#         B_csc = B.tocsc()

#         # For each row i, accumulate only into pattern entries.
#         for i in range(N):
#             A_start, A_end = A.indptr[i], A.indptr[i+1]
#             A_cols = A.indices[A_start:A_end]
#             A_vals = A.data[A_start:A_end]
#             # For each allowed column j in the pattern for row i...
#             for out_idx in range(out_indptr[i], out_indptr[i+1]):
#                 j = out_indices[out_idx]
#                 dot = 0.0
#                 # Access column j of B in CSC format.
#                 B_start, B_end = B_csc.indptr[j], B_csc.indptr[j+1]
#                 B_rows = B_csc.indices[B_start:B_end]
#                 B_vals = B_csc.data[B_start:B_end]

#                 # Merge the two sorted index arrays A_cols and B_rows.
#                 a_ptr, b_ptr = 0, 0
#                 while a_ptr < A_cols.shape[0] and b_ptr < B_rows.shape[0]:
#                     a_col = A_cols[a_ptr]
#                     b_row = B_rows[b_ptr]
#                     if a_col == b_row:
#                         dot += A_vals[a_ptr] * B_vals[b_ptr]
#                         a_ptr += 1
#                         b_ptr += 1
#                     elif a_col < b_row:
#                         a_ptr += 1
#                     else:
#                         b_ptr += 1

#                 out_data[out_idx] = dot

#         # Build the resulting CSR matrix.
#         return sp.csr_matrix((out_data, out_indices, out_indptr), shape=pattern.shape)

#     def compute_precond(self):
#         # Instead of doing plain products then "masking," we use fixed-pattern multiplication.
#         # We assume here that the allowable pattern for every product is given by self._pattern_mask.
#         for iter_num in range(self.iters):
#             # R = I - K @ M (in our allowed pattern)
#             Prod_KM = self.fixed_pattern_matmul(self.K, self.M, self._pattern_mask)
#             R_data = self.I.data - Prod_KM.data
#             # Build R using the same structure as I/self._pattern_mask.
#             R = sp.csr_matrix((R_data, self._pattern_mask.indices, self._pattern_mask.indptr),
#                               shape=self.I.shape)

#             # Z = M @ R (using fixed pattern)
#             Z = self.fixed_pattern_matmul(self.M, R, self._pattern_mask)

#             # AZ = K @ Z (using fixed pattern)
#             AZ = self.fixed_pattern_matmul(self.K, Z, self._pattern_mask)

#             # Debug print: ensure Z and AZ have the same number of nonzeros.
#             # (They will because the fixed-pattern routine uses the same "pattern".)
#             # print(f"Iteration {iter_num}: Z.nnz = {Z.nnz}, AZ.nnz = {AZ.nnz}")

#             # Compute the scalar alpha. Since the resulting matrices have identical nnz patterns,
#             # we can safely compute dot products over their .data arrays.
#             num = np.dot(Z.data, AZ.data)
#             den = np.dot(AZ.data, AZ.data)
#             alpha = num / den if den != 0.0 else 0.0

#             # Update M = M + alpha * Z (and force the allowed pattern)
#             M_new = self.M + alpha * Z
#             # Here, we reassemble the product in the fixed pattern by simply taking
#             # the values at positions defined by our mask:
#             self.M = sp.csr_matrix((M_new.data, M_new.indices, M_new.indptr), shape=M_new.shape)
#             # Alternatively, you can use fixed_pattern_matmul with an identity multiplication:
#             # self.M = self.fixed_pattern_matmul(M_new, sp.eye(self.N, format='csr'), self._pattern_mask)

#             # It's a good idea to eliminate explicit zeros.
#             self.M.eliminate_zeros()

#         return self.M

#     def solve(self, rhs):
#         return self.M.dot(rhs)

#     @property
#     def operator_complexity(self) -> float:
#         return self.M.nnz / self.K.nnz
    