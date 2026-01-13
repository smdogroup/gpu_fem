import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from smoothers import block_gauss_seidel_6dof, block_gauss_seidel_6dof_transpose

def strength_matrix_bsr(A:sp.bsr_matrix, threshold:float=0.25):
    """
    Compute strength of connections C_{ij} for sparse BSR matrix (slightly different than above strength and strength^T version)
    produces only one matrix

    comes from this paper on aggregation
    https://epubs.siam.org/doi/epdf/10.1137/110838844
    """

    assert sp.isspmatrix_bsr(A)

    # strength of connection graph
    N = A.shape[0]
    block_dim = A.data.shape[-1]
    nnodes = N // block_dim
    
    # get diagonal norms
    diag_nrms = []
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            if i == j:
                diag_nrms += [np.linalg.norm(A.data[jp])]
    
    STRENGTH = [[] for i in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            norm = np.linalg.norm(A.data[jp])
            lb = threshold * np.sqrt(diag_nrms[i] * diag_nrms[j])
            if norm >= lb:
                STRENGTH[i] += [j]
    # can be converted into CSR style NZ pattern (though has no float data, just NZ pattern C_{ij})
    return STRENGTH


def greedy_serial_aggregation_bsr(A:sp.bsr_matrix, threshold:float=0.25):
    """from paper https://epubs.siam.org/doi/epdf/10.1137/110838844, with parallel versions discussed also
    and GPU version here https://www.sciencedirect.com/science/article/pii/S0898122114004143"""

    # greedy serial aggregation
    assert sp.isspmatrix_bsr(A)
    N = A.shape[0]
    block_dim = A.data.shape[-1]
    nnodes = N // block_dim
    STRENGTH = strength_matrix_bsr(A, threshold)
    # print(f"{STRENGTH=}")

    # almost the same code as greedy_serial_aggregation_csr BTW
    
    aggregate_ind = np.full(nnodes, -1) # keep track of aggregate indices (-1 is unpicked, >= 0 is picked and which aggregate it belongs to)\
    aggregate_groups = []
    _ct = 0

    # first phase creates all the aggregates
    for i in range(nnodes):
        strong_neighbors = np.array(STRENGTH[i])
        # if i and each of its strong neighbors not picked, form aggregate:
        picked_neighbors = [ii for ii in strong_neighbors if aggregate_ind[ii] != -1]
        if len(picked_neighbors) == 0:
            # print(f"aggregate {_ct} from node {i=} and {strong_neighbors=}")
            aggregate_groups += [list(strong_neighbors)]
            aggregate_ind[strong_neighbors] = _ct
            _ct += 1 # increment aggregate counter

    # second phase adds all remaining nodes to nearby aggregates
    for i in range(nnodes):
        if aggregate_ind[i] != -1: continue # only look at unpicked nodes
        strong_neighbors = np.array(STRENGTH[i])
        nb_agg_ind = aggregate_ind[strong_neighbors]
        valid_nb_agg_ind = nb_agg_ind[nb_agg_ind != -1]
        # sweep into an aggregate with smallest size
        nb_agg_sizes = np.array([len(aggregate_groups[ind]) for ind in valid_nb_agg_ind])
        min_size_ind = np.argmin(nb_agg_sizes)
        agg_ind = valid_nb_agg_ind[min_size_ind]

        # now add into that aggregate
        aggregate_groups[agg_ind] += [i]
        aggregate_ind[i] = agg_ind

    assert(not np.any(aggregate_ind == -1))
    return aggregate_ind


def get_rigid_body_modes(xpts, th:float=1.0):
    """get rigid body modes of particular mesh"""

    _x = xpts[0::3]; _y = xpts[1::3]; _z = xpts[2::3]
    nnodes = _x.shape[0]

    Bpred = np.zeros((nnodes, 6, 6))

    # first three modes just as translation
    for imode in range(3):
        Bpred[:, imode, imode] = 1.0

    # u then v disp (yeah this one doesn't work cause of drill strain penalty)
    # so are there really only five modes?
    Bpred[:, 0, 3] = th * _y
    Bpred[:, 1, 3] = -th * _x
    # ahh => correction from drill strain = 2 * thz - (du/dy - dv/dx) = 2 * thz - omega
    # is to compute constant thz everywhere equal to the rotation magnitude => thz = th prescribed
    Bpred[:, 5, 3] = -th

    # v and w disp
    Bpred[:, 1, 4] = -th * _z
    Bpred[:, 2, 4] = th * _y
    # ah but then need to adjust thx or thy disp grads for trv shear error
    Bpred[:, 3, 4] = th

    # u and w disp
    Bpred[:, 0, 5] = th * _z
    Bpred[:, 2, 5] = -th * _x
    # and then need to adjust thy for dw/dx trv shear strain
    Bpred[:, 4, 5] = th
    return Bpred

def get_coarse_rigid_body_modes(B:np.ndarray, xpts:np.ndarray, aggregate_ind:np.ndarray):
    """helper method for constructing tentative prolongator"""
    # nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    # 2) compute coarse rigid body modes R for aggregates (average over nodes)
    R = np.zeros((num_agg, 6, 6))
    xpts_c = np.zeros((num_agg, 3))
    for iagg in range(num_agg):
        agg_nodes = aggregate_ind == iagg
        Bk = B[agg_nodes] #.reshape(-1, 6)  # (num_nodes_in_agg * 6) x 6
        # print(f"{Bk.shape=} {Bk=}")
        R[iagg] = np.mean(Bk, axis=0)
        # print(f"{R[iagg]=}")
        xpts_c[iagg] = np.mean(xpts[agg_nodes], axis=0)
    return R, xpts_c

def tentative_prolongator_bsr(B: np.ndarray, aggregate_ind: np.ndarray, bc_flags:np.ndarray):
    nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    rowp = np.arange(0, nnodes + 1)
    cols = np.zeros(nnodes, dtype=np.int32)
    for iagg in range(num_agg):
        cols[aggregate_ind == iagg] = iagg

    # B = get_rigid_body_modes(xpts.reshape(3 * nnodes))  # (nnodes, 6, 6)
    # R, xpts_c = get_coarse_rigid_body_modes(B, xpts, aggregate_ind)
    # print(f"{R[0]=}")
    R = np.zeros((num_agg, 6, 6))

    print(f"{B.shape=}")

    data = np.zeros((nnodes, 6, 6))

    # QR decomposition of B * Qk = Rk (which also discovers aggregate coarse rigid body modes)
    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        nk = np.sum(agg_mask)

        # Stack rigid body modes for this aggregate
        Bk = B[agg_mask].reshape(nk * 6, 6)

        # Thin QR factorization
        Qk, Rk = np.linalg.qr(Bk, mode="reduced")

        # Qk^T Qk = I   → tentative prolongator
        data[agg_mask] = Qk.reshape(nk, 6, 6)

        # Store Rk if you need the coarse nullspace
        R[iagg] = Rk

    # since tentative prolongator has only one block nonzero per fine node
    # was missing this before, need this with AMG (otherwise will get NZ deflection in fine prolong)
    for inode in range(nnodes):
        for ii in range(6):
            idof = 6 * inode + ii
            if bc_flags[idof]: # apply dirichlet bcs
                data[inode,ii,:] = 0.0

    P0 = sp.bsr_matrix((data, cols, rowp), blocksize=(6, 6))
    return P0, R

def get_bc_flags_bsr(A: sp.bsr_matrix, tol=1e-14) -> np.ndarray:
    """Return True for constrained DOFs in a BSR matrix (diagonal≈I, off-diagonal≈0)"""
    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    flags = np.zeros(A.shape[0], dtype=bool)

    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i+1]
        idx = A.indices[start:end]
        blk = A.data[start:end]

        diag = blk[idx == i]
        off = blk[idx != i]

        flags[i*b:(i+1)*b] = (
            np.all(np.abs(diag - np.eye(b)) < tol) and
            (off.size == 0 or np.all(np.abs(off) < tol))
        )

    return flags

def orthog_nullspace_projector(P: sp.bsr_matrix, Bc: np.ndarray, bcs: np.ndarray):
    """Apply the orthogonal projector to prevent nullspace modes (BSR version)"""

    Pnew = sp.bsr_matrix(
        (P.data.copy(), P.indices.copy(), P.indptr.copy()),
        shape=P.shape,
        blocksize=P.blocksize
    )

    block_dim = P.blocksize[0]
    nblocks_row = P.shape[0] // block_dim

    for brow in range(nblocks_row):
        # construct Fi: mask constrained DOFs for this fine node
        Fi_bcs = np.array([_ for _ in range(block_dim) if bcs[block_dim*brow + _]])
        Fi = np.eye(block_dim)
        if Fi_bcs.size > 0:
            Fi[Fi_bcs, :] = 0.0

        # precompute U and store bcols for this block row
        ncols = P.indptr[brow+1] - P.indptr[brow]
        bcols = P.indices[P.indptr[brow]:P.indptr[brow+1]]

        # this method did not work
        # U = np.hstack([Bc[bcols[j]] @ Fi for j in range(ncols)])  # shape (b, ncols*b)
        # dP_row = np.hstack([P.data[P.indptr[brow]+j] for j in range(ncols)])  # shape (b, ncols*b)
        # # orthogonal projector applied in one shot
        # UTU_inv = np.linalg.pinv(U.T @ U)
        # dP_row_proj = dP_row - dP_row @ U @ UTU_inv @ U.T
        # # unstack back into BSR blocks
        # for j in range(ncols):
        #     Pnew.data[P.indptr[brow]+j] = dP_row_proj[:, j*block_dim:(j+1)*block_dim]

        # this method doesn't quite work either..
        U_list = []
        for j in range(ncols):
            U_list.append(Bc[bcols[j]] @ Fi)
        # compute PU and UTU
        PU = sum(P.data[P.indptr[brow]+j] @ U_list[j] for j in range(ncols))
        UTU = sum(U_list[j].T @ U_list[j] for j in range(ncols))
        UTU_inv = np.linalg.pinv(UTU)

        # apply projector to each P block
        for j in range(ncols):
            Pnew.data[P.indptr[brow]+j] -= PU @ UTU_inv @ U_list[j].T

        # DEBUG : double check linear system..
        # new PU should be zero in this row
        # new_PU = sum(Pnew.data[P.indptr[brow]+j] @ U_list[j] for j in range(ncols))
        # new_PU_nrm = np.linalg.norm(new_PU)
        # print(f'{brow=} : {new_PU_nrm=:.4e}')

    # exit()

    # -----------------------------
    # Verification: check nullspace orthogonality
    # -----------------------------
    # unconstrained_dofs = np.where(bcs == 0)[0]
    # print(f"{unconstrained_dofs=}")
    # print(f"{bcs=}")
    # exit()

    # # form Uc = Bc * Fi
    # Uc = Bc.copy()
    # Uc[bcs // 6] = 0.0
    # for i in range(6):
    #     bc_vec = Uc[:,:,i].reshape(Pnew.shape[-1])
    #     p_vec = Pnew.dot(bc_vec)
    #     # print(f"mode {i=} : {p_vec=}")
    #     p_uc_vec = p_vec[unconstrained_dofs]
    #     # print(f"\t{p_uc_vec=}")
    #     nrm = np.max(np.abs(p_vec))
    #     nrm2 = np.max(np.abs(p_uc_vec))
    #     print(f"{nrm=:.4e} {nrm2=:.4e}")
    # exit()

    return Pnew

def spectral_radius_block_DinvA_bsr(
    A: sp.bsr_matrix,
    maxiter=200,
    tol=1e-8,
):
    """
    Estimate spectral radius of block Jacobi operator D_b^{-1} A.

    Parameters
    ----------
    A : scipy.sparse.bsr_matrix
    maxiter : int
    tol : float

    Returns
    -------
    rho : float
        Estimated spectral radius
    """

    if not sp.isspmatrix_bsr(A):
        raise TypeError("A must be a BSR matrix")

    b = A.blocksize[0]
    nblocks = A.shape[0] // b
    n = A.shape[0]

    # --------------------------------------------------
    # Precompute block diagonal inverses
    # --------------------------------------------------
    Dinv_blocks = np.zeros((nblocks, b, b))
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i + 1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        Dblock = A.data[start + diag_idx[0]]
        Dinv_blocks[i] = np.linalg.inv(Dblock)

    # --------------------------------------------------
    # Matrix-free operator y = D^{-1} A x
    # --------------------------------------------------
    def matvec(x):
        y = A @ x
        y = y.reshape(nblocks, b)

        for i in range(nblocks):
            y[i] = Dinv_blocks[i] @ y[i]

        return y.reshape(n)

    M = spla.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=A.dtype,
    )

    eigval = spla.eigs(
        M,
        k=1,
        which="LM",
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=False,
    )

    return np.abs(eigval[0])


def smooth_prolongator_bsr(T: sp.bsr_matrix, A: sp.bsr_matrix, Bc: np.ndarray,
                           bc_flags: np.ndarray = None, omega: float = None, near_kernel: bool = True):
    """
    Energy-smoothing single-step for tentative prolongator (BSR)
    P = T - omega * Dinv * A * T
    Nullspace components removed from the update Dinv*AT before applying omega.
    """
    if A.blocksize != T.blocksize:
        raise ValueError("A and T must have the same blocksize")
    b = A.blocksize[0]
    nblocks = A.shape[0] // b

    # --- omega ---
    if omega is None:
        rho = spectral_radius_block_DinvA_bsr(A)
        omega = 2.0 / rho * 0.9

    # --- block diagonal inverse ---
    Dinv_blocks = np.zeros((nblocks, b, b))
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i+1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        D_block = A.data[start + diag_idx[0]]
        Dinv_blocks[i] = np.linalg.inv(D_block)

    # --- compute update ---
    AT = A @ T
    dP = AT.copy()
    for i in range(nblocks):
        start, end = dP.indptr[i], dP.indptr[i+1]
        dP.data[start:end] = Dinv_blocks[i] @ dP.data[start:end]

    # --- nullspace removal ---
    if near_kernel:
        dP = orthog_nullspace_projector(dP, Bc, bc_flags)

    # --- zero Dirichlet DOFs in dP ---
    if bc_flags is not None:
        ndof = dP.blocksize[0]
        for i in range(dP.shape[0] // ndof):
            start, end = dP.indptr[i], dP.indptr[i+1]
            for j in range(ndof):
                if bc_flags[i*ndof + j]:
                    dP.data[start:end, j, :] = 0.0

    # --- final Jacobi update ---
    P = T - omega * dP

    return P


class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x

# class DirectCSRSolver:
#     def __init__(self, A_csr):
#         # convert to dense numpy array
#         self.A = A_csr.toarray()
#         # ensure exact symmetry
#         self.A = 0.5 * (self.A + self.A.T)
#         # compute Cholesky factor
#         self.L = np.linalg.cholesky(self.A)

#     def solve(self, rhs):
#         # solve A x = rhs via Cholesky
#         y = np.linalg.solve(self.L, rhs)
#         x = np.linalg.solve(self.L.T, y)
#         return x


class AMG_BSRSolver:
    """general multilevel AMG solver..."""
    def __init__(self, A_free:sp.bsr_matrix, A:sp.bsr_matrix, B:np.ndarray, threshold:float=0.25,
                 omega:float=0.7, pre_smooth=1, post_smooth=1, level:int=0, near_kernel:bool=True):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        B : fine rigid body modes
        threshold: float for coarsening threshold
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)

        self.A_free = A_free
        self.A = A
        self.near_kernel = near_kernel
        bs = self.A_free.data.shape[-1]
        nnodes = self.A_free.shape[0] // bs
        self.B = B

        # compute node aggregate sets
        # make sure to use unconstrained matrix for aggregation indicators originally
        aggregate_ind = greedy_serial_aggregation_bsr(A_free, threshold=threshold)
        num_agg = np.max(aggregate_ind) + 1

        # print(f"{aggregate_ind=}")
        # TODO: do this
        # bc_flags = None
        bc_flags = get_bc_flags_bsr(A)
        # print(f"{bc_flags=}")

        # create tentative prolongator then smooth it
        self.T, self.Bc = tentative_prolongator_bsr(self.B, aggregate_ind, bc_flags)
        self.P = smooth_prolongator_bsr(self.T, A, self.Bc, bc_flags, omega=omega, near_kernel=near_kernel) # single damped jacobi step, so only one step of fillin
        # self.P = self.T # DEBUG
        self.R = self.P.T # sym matrix so restriction is transpose prolong

        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.level = level

        # Galerkin coarse operator
        self.Ac = self.R @ (A @ self.P)
        # on GPU would not do extra allocation for this
        self.Ac_free = self.R @ (A_free @ self.P)

        self.Ac = self.Ac.tobsr()
        self.Ac_free = self.Ac_free.tobsr()
        # print(f"{type(self.Ac)=}")

        # plt.spy(self.Ac_free)
        # plt.show()
        # plt.imshow(self.Ac_free.toarray())
        # plt.show()

        # compute two-grid operator complexity
        self.fine_nnz = self.A.nnz
        self.coarse_nnz = self.Ac.nnz
        self.coarse_solver = None 

        if level == 0:
            print("level 0 is AMG solver..")

        # check fillin of coarse grid solver..
        coarse_nnodes = self.Ac.shape[0]
        max_coarse_nnz = coarse_nnodes**2

        # print(f"{level=} {self.fine_nnz=} {self.coarse_nnz=} {max_coarse_nnz=}")
        if self.fine_nnz == self.coarse_nnz:
            raise RuntimeError(f"ERROR: {self.fine_nnz=} == {self.coarse_nnz=} : lower aggregation threshold from {threshold=} to lower value..\n")

        if self.coarse_nnz >= 0.4 * max_coarse_nnz or coarse_nnodes <= 100:
            # then do direct solver
            print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            print(f"level {level+1} building AMG solver")
            # print(f"{type(self.Ac_free)=} {type(self.Ac)=}")
            self.coarse_solver = AMG_BSRSolver(self.Ac_free, self.Ac, self.Bc, threshold, omega, pre_smooth, post_smooth, level+1, near_kernel=near_kernel)

        if level == 0:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=}")
            print(f"\tnum nodes per level = [{self.num_nodes_list}]")

    @property
    def total_nnz(self) -> int:
        # get total nnz across all levels
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else: # direct solver
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz     

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return self.coarse_solver.num_levels + 1
        else: # direct solver
            return 2
        
    @property
    def num_nodes_list(self) -> str:
        bs = self.A.data.shape[-1]
        nnodes_f = self.A.shape[0] // bs
        nnodes_c = self.Ac.shape[0] // bs
        if isinstance(self.coarse_solver, AMG_BSRSolver):
            return str(nnodes_f) + "," + self.coarse_solver.num_nodes_list
        else: # direct solver
            return f"{nnodes_f},{nnodes_c}"

    # def solve(self, rhs):
    #     """
    #     Fully symmetric AMG V-cycle (SPD) for PCG
    #     """
    #     x = np.zeros_like(rhs)

    #     # Pre-smoothing (forward GS)
    #     if self.pre_smooth > 0:
    #         x = block_gauss_seidel_6dof(self.A, rhs, x0=x,
    #                                     num_iter=self.pre_smooth)

    #     # Coarse-grid correction
    #     r = rhs - self.A @ x
    #     rc = self.R @ r
    #     ec = self.coarse_solver.solve(rc)
    #     x += self.P @ ec

    #     # Post-smoothing (backward GS, applied to x!)
    #     if self.post_smooth > 0:
    #         x = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x,
    #                                             num_iter=self.post_smooth)

    #     return x

    def solve(self, rhs):
        """
        Fully symmetric AMG V-cycle (SPD) for PCG
        """
        x = np.zeros_like(rhs)

        # Pre-smoothing (forward GS)
        if self.pre_smooth > 0:
            x = block_gauss_seidel_6dof(self.A, rhs, x0=x,
                                        num_iter=self.pre_smooth)

        # # Coarse-grid correction
        r = rhs - self.A.dot(x)
        rc = self.R.dot(r)
        ec = self.coarse_solver.solve(rc)
        x += self.P.dot(ec)

        # Post-smoothing (backward GS, applied to x!)
        if self.post_smooth > 0:
            x = block_gauss_seidel_6dof_transpose(self.A, rhs, x0=x,
                                                num_iter=self.post_smooth)

        return x