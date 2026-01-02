# V1 of multilevel SPAI
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
from _spai import *

# algebraic multilevel method mostly based on these papers
# I did add jacobi smoother for restrict + prolong, whereas they do not (cause they have much simpler problem with Poisson I guess)
# 1) https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub
# 2) https://link.springer.com/chapter/10.1007/978-3-031-25820-6_11
# 3) https://epubs.siam.org/doi/10.1137/S106482759732753X


class DirectSolver:
    def __init__(self, A_bsr):
        # convert to dense matrix (full fillin)
        if isinstance(A_bsr, np.ndarray):
            self.A = A_bsr
        else:
            self.A = A_bsr.toarray()

    def solve(self, rhs):
        # use python dense solver..
        x = np.linalg.solve(self.A, rhs)
        return x

class MultilevelSPAI:
    # multilevel splitting from MultilevelILU paper https://arxiv.org/pdf/1901.03249 
    # SPAI preconditioner from https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf
    def __init__(self, A_bsr, levels:int=2, iters:int=5, n_smooth:int=4, omega:float=0.7):
        # convert to dense (TODO later is to do sparse version)
        self.A = A_bsr
        self.A_dense = A_bsr.toarray()
        self.levels = levels   
        self.iters = iters 
        self.n_smooth = n_smooth
        self.omega = omega

        self.B = None
        self.F = None
        self.E = None
        self.C = None
        self.S = None

        self.levels = levels
        self.B_pc = None
        self.S_pc = None

        self._compute_splitting()
        self._compute_spai()

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
    
    def _compute_spai(self):
        # compute SPAI preconditioner of fine nodes and the coarse node too

        # --- compute B ILU factor ---
        self.B_pc = SPAI_MR_SelfPrecond(self.B)
        new_levels = self.levels - 1

        # --- compute schur complement ---
        # S = C - E * B^{-1} * F
        C_dense = self.C.toarray()
        E_dense = self.E.toarray()
        B_dense = self.B.toarray()
        F_dense = self.F.toarray()

        # exact schur complement (dense) FOR DEBUG
        # self.S = C_dense - E_dense @ np.linalg.inv(B_dense) @ F_dense
        
        # approx schur complement
        M_Binv = self.B_pc.M
        Y = M_Binv @ F_dense
        self.S = C_dense - E_dense @ Y

        # --- compute S ILU factor ---
        if new_levels <= 1:
            # on coarsest level need full LU solve (so ILU(0) solve not acceptable)
            self.S_pc = DirectSolver(self.S)
            # self.S_pc = GaussJordanBlockPrecond(self.S) # can't do ILU(0) here
        else:
            # recursively make new MILU factor
            self.S_pc = MultilevelSPAI(self.S, levels=new_levels)
    
    # def solve_old(self, rhs:np.ndarray):
    #     """solve the MILU preconditioner for multiple levels"""
    # not great preconditioner.. cause not true multilevel, the separate Binv
    # results in low compatibility..
    #     # from Eq. 6 of https://arxiv.org/pdf/1901.03249
    #     # first a fine solve
    #     x = np.zeros_like(rhs)
    #     fine_rhs1 = rhs[self.fine_mask]
    #     # print(f"{fine_rhs1.shape=}")
    #     fine_soln1 = self.B_pc.solve(fine_rhs1)
    #     x[self.fine_mask] = fine_soln1
    #     # first pair of outer product terms from coarse solve
    #     coarse_rhs1 = self.E.dot(fine_soln1)
    #     coarse_soln1 = self.S_pc.solve(coarse_rhs1)
    #     fine_soln1 = self.B_pc.solve(self.F.dot(coarse_soln1))
    #     x[self.fine_mask] += fine_soln1
    #     x[self.coarse_mask] -= coarse_soln1
    #     # outer product terms from coarse solve
    #     coarse_rhs2 = rhs[self.coarse_mask]
    #     coarse_soln2 = self.S_pc.solve(coarse_rhs2)
    #     fine_soln2 = self.B_pc.solve(self.F.dot(coarse_soln2))
    #     x[self.fine_mask] -= fine_soln2
    #     x[self.coarse_mask] += coarse_soln2
    #     return x

    def block_jacobi_smooth(self, x, rhs, omega=0.7, iters=2):
        """
        Block Jacobi smoother for BSR matrices.

        Parameters
        ----------
        x : ndarray
            Initial guess (modified in-place).
        rhs : ndarray
            Right-hand side.
        omega : float
            Damping factor.
        iters : int
            Number of smoothing sweeps.
        """
        A = self.A
        bs = A.blocksize[0]
        nblocks = A.shape[0] // bs

        # Precompute inverse diagonal blocks
        Dinv = [None] * nblocks
        for i in range(nblocks):
            row_start = A.indptr[i]
            row_end   = A.indptr[i+1]
            cols = A.indices[row_start:row_end]
            data = A.data[row_start:row_end]

            for j, col in enumerate(cols):
                if col == i:
                    Dinv[i] = np.linalg.inv(data[j])
                    break
            else:
                raise RuntimeError(f"Missing diagonal block at row {i}")

        for _ in range(iters):
            r = rhs - A.dot(x)

            for i in range(nblocks):
                ii = slice(i*bs, (i+1)*bs)
                x[ii] += omega * (Dinv[i] @ r[ii])

        return x


    
    def solve(self, rhs:np.ndarray):
        """new multilevel ILU solve based on paper from page 8,
        "Sparse approximate inverse and multilevel block ILU preconditioning techniques for general sparse matrices", 
        https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub"""

        x0 = np.zeros_like(rhs)
        # pre-smoothing
        x1 = self.block_jacobi_smooth(x0, rhs, omega=self.omega, iters=self.n_smooth)
        rhs1 = rhs - self.A.dot(x1)

        # rhs inputs
        rhs_fine = rhs1[self.fine_mask]
        rhs_coarse = rhs1[self.coarse_mask]

        # forward elimination (like smooth + restrict)
        y_fine = self.B_pc.solve(rhs_fine)
        y_coarse = rhs_coarse - self.E.dot(y_fine)

        # coarse solve
        x_coarse = self.S_pc.solve(y_coarse)

        # backward elim (like prolong + smooth)
        # x_coarse = x_coarse (unchanged)
        x_fine = self.B_pc.solve(y_fine - self.F.dot(x_coarse))

        # store output soln
        x = np.zeros_like(rhs) # solution / output
        x[self.fine_mask] = x_fine * 1.0
        x[self.coarse_mask] = x_coarse * 1.0

        # then do a few jacobi smooth steps here.. on top..
        # like in regular multigrid
        x_smooth = self.block_jacobi_smooth(x, rhs, omega=self.omega, iters=self.n_smooth)

        return x_smooth