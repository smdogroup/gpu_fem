
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from __src import plot_plate_vec
from __ilu import extract_bsr_triangular, sort_bsr_indices, _get_diagp, GaussJordanBlockPrecond, block_ilu6_gj_solve
from _milu import BILU_SVD_Precond

# based on multilevel ILU from https://arxiv.org/pdf/1901.03249 

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

    L = extract_bsr_triangular(B_lu, lower=True)
    U = extract_bsr_triangular(B_lu, lower=False)
    UF = sort_bsr_indices(U @ F)
    Y0 = sort_bsr_indices(L @ UF)

    # do another mat-multiply for extra fillin to mimic inverse?
    Ytmp = sort_bsr_indices(U @ Y0)
    Y1 = sort_bsr_indices(L @ Ytmp)
    
    # # do another mat-multiply for extra fillin
    Ytmp2 = sort_bsr_indices(U @ Y1)
    Y = sort_bsr_indices(L @ Ytmp2)
    # Y = Y1.copy()
    Y.data[:] = 0.0  # zero out values

    bs = 6
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


class DirectSolver:
    def __init__(self, A_bsr):
        # convert to dense matrix (full fillin)
        self.A = A_bsr.toarray()

    def solve(self, rhs):
        # use python dense solver..
        x = np.linalg.solve(self.A, rhs)
        return x

class MultilevelILU:
    # based on multilevel ILU from https://arxiv.org/pdf/1901.03249 
    def __init__(self, A_bsr, levels:int=2, alpha:float=None):
        """
        compute a multilevel ILU splitting into [B, F; E, C]
        B = A_FF, F = A_FC, E = A_CF, C = A_CC
        """
        assert sp.isspmatrix_bsr(A_bsr)
        self.A = A_bsr.copy()

        self.alpha = alpha
        self.use_svd = alpha is not None

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

        self.num_fine_nodes = fine_nodes.shape[0]
        self.num_coarse_nodes = coarse_nodes.shape[0]
        # print(f"{self.num_fine_nodes=}\n{self.num_coarse_nodes=}\n")
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

        # print(f"{self.A.shape=}")
        # print(f"{self.B.shape=}\n{self.F.shape=}\n{self.E.shape=}\n{self.C.shape=}\n")

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
        if self.use_svd:
            self.B_pc = BILU_SVD_Precond(self.B, alpha=self.alpha)
        else:
            self.B_pc = GaussJordanBlockPrecond(self.B)
        new_levels = self.levels - 1

        # --- compute schur complement ---
        # S = C - E * B^{-1} * F
        # with Y = Binv * F computed with mat-mat solve
        Y = block_ilu6_gj_solve_matmat_sym(self.B_pc.A, self.F)

        # test correct linear solve vs vectors..
        debug = False
        # debug = True

        if debug:
            vec_fine = np.zeros(6 * self.num_fine_nodes)
            vec_coarse = np.zeros(6 * self.num_coarse_nodes)
            max_rel_err = 0.0
            for i_f in range(6 * self.num_fine_nodes):
                for i_c in range(6 * self.num_coarse_nodes):
                    
                    vec_fine[:] = 0.0
                    vec_fine[i_f] = 1.0
                    vec_coarse[:] = 0.0
                    vec_coarse[i_c] = 1.0

                    # vec_fine = np.random.rand(6 * self.num_fine_nodes)
                    # vec_coarse = np.random.rand(6 * self.num_coarse_nodes)
                    v1 = self.F.dot(vec_coarse)
                    v2 = np.zeros_like(v1)
                    # print(f"{v1.shape=} {v2.shape=}")
                    block_ilu6_gj_solve(self.B_pc.A, v2, v1)
                    s1 = np.dot(vec_fine, v2)

                    v3 = Y.dot(vec_coarse)
                    s2 = np.dot(vec_fine, v3)

                    if abs(s1) >= 1e-12 or abs(s2) >= 1e-12:
                        rel_err = abs((s1-s2)/s1)
                        max_rel_err = np.max([max_rel_err, rel_err])
                        if rel_err > 1e-5:
                            print(f"node ({i_f//6},{i_c//6}), DOF ({i_f},{i_c}) : {s1=:.5e} {s2=:.5e} {rel_err=:.5e}")

            print(f"{max_rel_err=}")


        # self.S = self.C
        self.S = self.C - self.E @ Y
        # self.S = self.C - sort_bsr_indices(self.E @ Y)
        # print(f"{self.S.shape=}")

        # --- compute S ILU factor ---
        if new_levels <= 1:
            # on coarsest level need full LU solve (so ILU(0) solve not acceptable)
            self.S_pc = DirectSolver(self.S)
            # self.S_pc = GaussJordanBlockPrecond(self.S) # can't do ILU(0) here
        else:
            # recursively make new MILU factor
            self.S_pc = MultilevelILU(self.S, levels=new_levels)
    
    def solve(self, rhs:np.ndarray):
        """solve the MILU preconditioner for multiple levels"""
        
        # from Eq. 6 of https://arxiv.org/pdf/1901.03249
        # first a fine solve
        x = np.zeros_like(rhs)
        fine_rhs1 = rhs[self.fine_mask]
        # print(f"{fine_rhs1.shape=}")
        fine_soln1 = self.B_pc.solve(fine_rhs1)
        x[self.fine_mask] = fine_soln1
    
        # first pair of outer product terms from coarse solve
        coarse_rhs1 = self.E.dot(fine_soln1)
        coarse_soln1 = self.S_pc.solve(coarse_rhs1)
        fine_soln1 = self.B_pc.solve(self.F.dot(coarse_soln1))
        x[self.fine_mask] += fine_soln1
        x[self.coarse_mask] -= coarse_soln1

        # outer product terms from coarse solve
        coarse_rhs2 = rhs[self.coarse_mask]
        coarse_soln2 = self.S_pc.solve(coarse_rhs2)
        fine_soln2 = self.B_pc.solve(self.F.dot(coarse_soln2))
        x[self.fine_mask] -= fine_soln2
        x[self.coarse_mask] += coarse_soln2

        nxe_c = int(self.num_coarse_nodes**0.5) - 1
        nxe_f = int(self.num_fine_nodes**0.5) - 1
        # print(f"{self.num_coarse_nodes=} {self.num_fine_nodes=}")
        # print(f"{nxe_c=} {coarse_rhs2.shape=}")
        num_nodes = self.num_coarse_nodes + self.num_fine_nodes
        nxe = int(num_nodes**0.5) - 1

        # fig, ax = plt.subplots(1, 2, figsize=(8, 6), subplot_kw={'projection': '3d'})
        # plot_plate_vec(nxe, rhs, ax=ax[0], sort_fw=np.arange(0, 6 * num_nodes))
        # plot_plate_vec(nxe, x, ax=ax[1], sort_fw=np.arange(0, 6 * num_nodes))
        # plt.show()

        # fig, ax = plt.subplots(1, 2, figsize=(8, 6), subplot_kw={'projection': '3d'})
        # plot_plate_vec(nxe_c, coarse_rhs2, ax=ax[0], sort_fw=np.arange(0, 6 * self.num_coarse_nodes))
        # # coarse_soln22 = np.linalg.solve(self.S.toarray(), coarse_rhs2)
        # plot_plate_vec(nxe_c, coarse_soln2, ax=ax[1], sort_fw=np.arange(0, 6 * self.num_coarse_nodes))
        # plt.show()

        return x