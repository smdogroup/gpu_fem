import numpy as np
import scipy.sparse as sp

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
            # perm not computed in NASA's SLAT
            # maybe because the row-perm doens't affect Linv * Uinv?
            # cause cols not permuted
            _temp = perm[ipeak]
            perm[ipeak] = perm[i]
            perm[i] = _temp

        # check for singular pivot
        if abs(A[ipeak,i]) < 1e-35:
            print(f"GJ (6x6) helper: singular pivot ({ipeak},{i})")
            return True # failed
        
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

def block_ilu6_gj_factor(A_bsr):
    """
    ILU factorization based on NASA SLAT:
        * Block ILU(0) Gauss-jordan for 6x6 block BSR matrix
        * Directly based on NASA SLAT ilu6_sse code (DO NOT DISTRIBUTE)
    """

    # print(f"{A_bsr.shape=}")

    # preamble (allocation)
    assert isinstance(A_bsr, sp.bsr_matrix)
    assert A_bsr.blocksize == (6,6)
    A_copy = A_bsr.copy()
    rowp = A_copy.indptr
    cols = A_copy.indices
    data = A_copy.data
    nnodes = A_copy.shape[0] // 6
    iw = np.full(nnodes, 0, dtype=int)
    diagp = _get_diagp(A_bsr)
    
        
    # ILU(0) with Gauss-Jordan solves
    # based on NASA SLAT code
    for k in range(nnodes):
        j1 = rowp[k]
        j2 = rowp[k+1] - 1
    
        for j in range(j1, j2+1):
            iw[cols[j]] = j
        
        # also based on ilu_generic_template.h
        # which is easier to read

        j = j1
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

                for jj in range(diagp[jrow] + 1, rowp[jrow+1]):
                    # rowp to cols jj
                    jw = iw[cols[jj]]

                    if jw != 0:
                        if cols[jw] < nnodes and cols[jj] < nnodes:
                            # they used SSE packed double intrinsics in NASA CPU-parallel code here
                            data[jw] -= tmat @ data[jj]

            j += 1
        # done with matmult loop in crout ILU? is this crout ILU?

        # diagp is already set in my code, but SLAT is setting it..
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
            iw[cols[j]] = 0

    return A_copy

        
def block_ilu6_gj_solve(A_lu, x, y):
    """
    ILU solve based on NASA SLAT:
        * Block ILU(0) Gauss-jordan for 6x6 block BSR matrix
        * Directly based on NASA SLAT ilu6_sse code (DO NOT DISTRIBUTE)
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


def block_ilu6_gj_solve_matmat_sym(B_lu, F):
    """
    Compute Y = B^{-1} F for symmetric B using block ILU(0) Gauss-Jordan,
    exploiting the fact that B is symmetric. Pure BSR logic, preallocated sparsity
    from B @ F. No dicts.

    Parameters
    ----------
    B_lu : scipy.sparse.bsr_matrix
        ILU-factorized B matrix (symmetric, block 6x6)
    F : scipy.sparse.bsr_matrix
        RHS block matrix (same block size)

    Returns
    -------
    Y : scipy.sparse.bsr_matrix
        Sparse block matrix equal to B^{-1} F
    """
    assert sp.isspmatrix_bsr(B_lu)
    assert sp.isspmatrix_bsr(F)
    assert B_lu.blocksize == (6,6)
    assert F.blocksize == (6,6)
    bs = 6

    # preallocate Y with correct sparsity
    # Y = B_lu @ F
    Y = F.copy()
    Y.data[:] = 0.0  # zero out values
    nnodes = B_lu.shape[0] // bs

    # --------------------------------------------------
    # forward solve (L y = F)
    # --------------------------------------------------
    rowp_B = B_lu.indptr
    cols_B = B_lu.indices
    data_B = B_lu.data
    diagp_B = _get_diagp(B_lu)

    for i in range(nnodes):

        # copy F into Y for this row
        for F_jp in range(F.indptr[i], F.indptr[i+1]):
            F_col = F.indices[F_jp]
            # find corresponding position in Y
            for Y_jp in range(Y.indptr[i], Y.indptr[i+1]):
                if Y.indices[Y_jp] == F_col:
                    Y.data[Y_jp] = F.data[F_jp].copy()
                    break
        
        # do Y -= Linv * Y
        for k in range(rowp_B[i], diagp_B[i]):
            j = cols_B[k]
            if i > j:  # lower triangular
                Bij = data_B[k]
                for Y_jp in range(Y.indptr[j], Y.indptr[j+1]):
                    Y_j_col = Y.indices[Y_jp]
                    for Y_ip in range(Y.indptr[i], Y.indptr[i+1]):
                        if Y.indices[Y_ip] == Y_j_col:
                            Y.data[Y_ip] -= Bij @ Y.data[Y_jp]
                            break

    # --------------------------------------------------
    # backward solve (L^T x = y)
    # --------------------------------------------------
    for i in range(nnodes-1, -1, -1):
        for k in range(diagp_B[i]+1, rowp_B[i+1]):
            j = cols_B[k]
            if i < j:  # upper triangular
                Bij = data_B[k]
                for Y_jp in range(Y.indptr[j], Y.indptr[j+1]):
                    Y_j_col = Y.indices[Y_jp]
                    for Y_ip in range(Y.indptr[i], Y.indptr[i+1]):
                        if Y.indices[Y_ip] == Y_j_col:
                            Y.data[Y_ip] -= Bij @ Y.data[Y_jp]
                            break
        # diagonal contribution
        diag_block = data_B[diagp_B[i]]
        for Y_ip in range(Y.indptr[i], Y.indptr[i+1]):
            Y.data[Y_ip] = diag_block @ Y.data[Y_ip]

    return Y


class GaussJordanBlockPrecond:
    def __init__(self, A):
        self.A = block_ilu6_gj_factor(A.copy())

    def solve(self, rhs):
        x = np.zeros_like(rhs)
        block_ilu6_gj_solve(self.A, x, rhs)
        return x
    


class MultilevelILU:
    def __init__(self, A_bsr, levels:int=2):
        """
        compute a multilevel ILU splitting into [B, F; E, C]
        B = A_FF, F = A_FC, E = A_CF, C = A_CC
        """
        assert sp.isspmatrix_bsr(A_bsr)
        self.A = A_bsr.copy()

        self.B = None
        self.F = None
        self.E = None
        self.C = None
        self.S = None

        self.levels = levels
        self.B_pc = None
        self.S_pc = None

        self._compute_splitting()
        self._compute_ilu()

    def _compute_splitting(self):
        A = self.A
        bs = A.blocksize[0]
        nnodes = A.shape[0] // bs

        # --- structured coarse/fine split ---
        nx = int(nnodes**0.5)

        coarse_nodes = []
        fine_nodes = []
        for inode in range(nnodes):
            ix, iy = inode % nx, inode // nx
            if ix % 2 == 0 and iy % 2 == 0:
                coarse_nodes.append(inode)
            else:
                fine_nodes.append(inode)

        coarse_nodes = np.array(coarse_nodes, dtype=int)
        fine_nodes   = np.array(fine_nodes, dtype=int)
        self.coarse_nodes = coarse_nodes
        self.fine_nodes = fine_nodes
        # print(f"{fine_nodes=}\n{coarse_nodes=}\n")

        num_fine_nodes = fine_nodes.shape[0]
        num_coarse_nodes = coarse_nodes.shape[0]
        print(f"{num_fine_nodes=}\n{num_coarse_nodes=}\n")
        self.fine_mask = np.array([6 * inode + _ for inode in fine_nodes for _ in range(6)])
        self.coarse_mask = np.array([6 * inode + _ for inode in coarse_nodes for _ in range(6)])

        # maps: global block index → local index
        f_map = {i: k for k, i in enumerate(fine_nodes)}
        c_map = {i: k for k, i in enumerate(coarse_nodes)}

        nf = len(fine_nodes)
        nc = len(coarse_nodes)

        # allocate block storage
        B_rows, B_cols, B_data = [[] for _ in range(nf)], [[] for _ in range(nf)], [[] for _ in range(nf)]
        F_rows, F_cols, F_data = [[] for _ in range(nf)], [[] for _ in range(nf)], [[] for _ in range(nf)]
        E_rows, E_cols, E_data = [[] for _ in range(nc)], [[] for _ in range(nc)], [[] for _ in range(nc)]
        C_rows, C_cols, C_data = [[] for _ in range(nc)], [[] for _ in range(nc)], [[] for _ in range(nc)]

        # --- loop over original BSR structure ---
        for i in range(nnodes):
            row_start = A.indptr[i]
            row_end   = A.indptr[i+1]

            for k in range(row_start, row_end):
                j = A.indices[k]
                blk = A.data[k]

                if i in f_map:
                    ii = f_map[i]
                    if j in f_map:
                        B_rows[ii].append(ii)
                        B_cols[ii].append(f_map[j])
                        B_data[ii].append(blk)
                    elif j in c_map:
                        F_rows[ii].append(ii)
                        F_cols[ii].append(c_map[j])
                        F_data[ii].append(blk)

                elif i in c_map:
                    ii = c_map[i]
                    if j in f_map:
                        E_rows[ii].append(ii)
                        E_cols[ii].append(f_map[j])
                        E_data[ii].append(blk)
                    elif j in c_map:
                        C_rows[ii].append(ii)
                        C_cols[ii].append(c_map[j])
                        C_data[ii].append(blk)

        # --- assemble BSR matrices ---
        self.B = self._assemble_bsr(B_rows, B_cols, B_data, nf, nf, bs)
        self.F = self._assemble_bsr(F_rows, F_cols, F_data, nf, nc, bs)
        self.E = self._assemble_bsr(E_rows, E_cols, E_data, nc, nf, bs)
        self.C = self._assemble_bsr(C_rows, C_cols, C_data, nc, nc, bs)

        print(f"{self.A.shape=}")
        print(f"{self.B.shape=}\n{self.F.shape=}\n{self.E.shape=}\n{self.C.shape=}\n")

    @staticmethod
    def _assemble_bsr(rows, cols, data, nrows, ncols, bs):
        indptr = np.zeros(nrows + 1, dtype=int)
        indices = []
        blocks = []

        for i in range(nrows):
            indptr[i+1] = indptr[i] + len(cols[i])
            indices.extend(cols[i])
            blocks.extend(data[i])
        # print(f"{indptr=}\n{indices=}\n")

        return sp.bsr_matrix(
            (np.array(blocks), np.array(indices), indptr),
            shape=(nrows * bs, ncols * bs),
            blocksize=(bs, bs)
        )

    def _compute_ilu(self):

        # --- compute B ILU factor ---
        self.B_pc = GaussJordanBlockPrecond(self.B)
        new_levels = self.levels - 1

        # --- compute schur complement ---
        # S = C - E * B^{-1} * F
        # with Y = Binv * F computed with mat-mat solve
        Y = block_ilu6_gj_solve_matmat_sym(self.B_pc.A, self.F)

        # check residual here for B * Y = F system
        R = self.B @ Y - self.F
        R_dense = R.toarray()
        R_nrm = np.linalg.norm(R_dense)
        print(f"{R_nrm=:.4e}")
        # extra fillin in B @ Y makes it not true?
        
        # --- compute residual only on nonzero blocks of Y ---
        # R_nrm = np.sqrt(sum(np.linalg.norm(self.B.data[k] @ Y.data[jp] - self.F.data[jp])**2
        #             for i in range(Y.shape[0]//6)
        #             for jp in range(Y.indptr[i], Y.indptr[i+1])
        #             for k in range(self.B.indptr[i], self.B.indptr[i+1])
        #             if self.B.indices[k] == Y.indices[jp]))
        # print(f"Residual over Y pattern: {R_nrm:.4e}")


        self.S = self.C - self.E @ Y
        print(f"{self.S.shape=}")

        # --- compute S ILU factor ---
        if new_levels <= 1:
            self.S_pc = GaussJordanBlockPrecond(self.S)
        else:
            # recursively make new MILU factor
            self.S_pc = MultilevelILU(self.S, levels=new_levels)

    def solve(self, rhs:np.ndarray):
        """solve the MILU preconditioner for multiple levels"""
        
        # from Eq. 6 of https://arxiv.org/pdf/1901.03249
        # first a fine solve
        x = np.zeros_like(rhs)
        fine_rhs1 = rhs[self.fine_mask]
        print(f"{fine_rhs1.shape=}")
        fine_soln1 = self.B_pc.solve(fine_rhs1)
        x[self.fine_mask] = self.B_pc.solve(fine_soln1)
    
        # first pair of outer product terms from coarse solve
        coarse_rhs1 = self.E.multiply(fine_soln1)
        coarse_soln1 = self.S_pc.solve(coarse_rhs1)
        fine_soln1 = self.B_pc.solve(self.F.multiply(coarse_soln1))
        x[self.fine_mask] += fine_soln1
        x[self.coarse_mask] -= coarse_soln1

        # outer product terms from coarse solve
        coarse_rhs2 = rhs[self.coarse_nodes]
        coarse_soln2 = self.S_pc.solve(coarse_rhs2)
        fine_soln2 = self.B_pc.solve(self.F.multiply(coarse_soln2))
        x[self.fine_mask] -= fine_soln2
        x[self.coarse_mask] += coarse_soln2

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

