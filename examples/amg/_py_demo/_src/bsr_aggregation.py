import numpy as np
import scipy.sparse as sp
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

def tentative_prolongator_bsr(xpts: np.ndarray, aggregate_ind: np.ndarray):
    nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    # print(f"{nnodes=} {aggregate_ind.shape=} {xpts.shape=}")

    # CSR-like column structure for BSR
    rowp = np.arange(0, nnodes+1)
    cols = np.zeros(nnodes, dtype=np.int32)
    for iagg in range(num_agg):
        cols[aggregate_ind == iagg] = iagg

    # 1) get fine rigid body modes B
    # B shape = (nnodes, 6)
    B = get_rigid_body_modes(xpts.reshape((3 * nnodes)))

    # 2) compute coarse rigid body modes R for aggregates (average over nodes)
    R = np.zeros((num_agg, 6, 6))
    xpts_c = np.zeros((num_agg, 3))
    for iagg in range(num_agg):
        agg_nodes = aggregate_ind == iagg
        Bk = B[agg_nodes].reshape(-1, 6)  # (num_nodes_in_agg * 6) x 6
        R[iagg] = np.mean(Bk, axis=0)
        xpts_c[iagg] = np.mean(xpts[agg_nodes], axis=0)

    # 3) QR decomposition for each aggregate
    data = np.zeros((nnodes, 6, 6))
    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        nk = np.sum(agg_mask)
        # stack the fine RBMs of this aggregate into (nk*6) x 6
        Bk = B[agg_mask].reshape(nk*6, 6)

        # QR decomposition
        Qk, _ = np.linalg.qr(Bk)  # Qk: (nk*6) x 6
        # assign back to data
        data[agg_mask] = Qk.reshape(nk, 6, 6)

    # 4) construct BSR matrix
    P0 = sp.bsr_matrix((data, cols, rowp), blocksize=(6,6))
    return P0, xpts_c


def orthog_nullspace_projector(P: sp.bsr_matrix, B:np.ndarray, bcs:np.ndarray):
    """apply the orthogonal projector to prevent nullspace modes"""

    # TODO : need to fix this method still

    # node_bcs = np.unique(bcs // 6)
    # all nodes with bcs (exterior nodes will have zero update to that block-node of P, the whole row)
    # so we can just skip that and zero out (then ignore F_i part, cause F_i = I in interior fine nodes)

    Pnew = sp.bsr_matrix((P.data.copy(), P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)
    # don't need to zero it out actually

    ndof = 6
    nblocks_row = P.shape[0] // ndof
    nblocks_col = P.shape[1] // ndof

    # loop over each fine node
    for brow in range(nblocks_row):
        # no actually don't skip F_i = 0 fully only in special case..
        # if brow in node_bcs: continue # skip this block row then (keep it unchanged)

        PU = np.zeros((6, 6))
        UTU = np.zeros((6, 6))

        Fi_bcs = np.array([_ for _ in range(6) if (6*brow+_) in bcs])
        Fi = np.eye(6)
        if Fi_bcs.shape[0] > 0:
            Fi[Fi_bcs,:] = 0.0
            # print(f"{Fi=}")

        # looping through the sparsity to compute sums first
        for jp in range(P.indptr[brow], P.indptr[brow+1]):
            bcol = P.indices[jp] # coarse node index

            U = B[bcol] @ Fi 
            
            PU += P.data[jp, :, :] @ U

            UTU += U.T @ U

        UTU_inv = np.linalg.pinv(UTU)
        # if not(brow in node_bcs):
            # print(f"{UTU_inv=}")

        # now loop back through removing projector from each P block (to apply projector)
        for jp in range(P.indptr[brow], P.indptr[brow+1]):
            bcol = P.indices[jp] # coarse node index

            U = B[bcol] @ Fi
            prev = Pnew.data[jp]
            delta = PU @ UTU_inv @ U.T
            final = prev - delta
            # print(f"{PU=}\n{UTU_inv=}\n{U.T=}\n\n")
            # print(f"{prev=} {-delta=} {final=}")
            Pnew.data[jp] -= PU @ UTU_inv @ U.T

    return Pnew

def smooth_prolongator_bsr(T: sp.bsr_matrix, A: sp.bsr_matrix, bc_flags: np.ndarray=None, omega: float = 0.7, near_kernel: bool = True):
    """
    Single-step block Jacobi smoothing of tentative prolongator for BSR matrices (6x6 blocks)
    
    P = (I - omega * Dinv * A) * T
    """
    if A.blocksize != T.blocksize:
        raise ValueError("A and T must have the same blocksize")
    b = A.blocksize[0]  # e.g., 6
    nblocks = A.shape[0] // b

    # 1) Compute inverse of block diagonal
    Dinv_blocks = np.zeros((nblocks, b, b))
    for i in range(nblocks):
        start, end = A.indptr[i], A.indptr[i+1]
        diag_idx = np.where(A.indices[start:end] == i)[0]
        if diag_idx.size == 0:
            raise ValueError(f"No diagonal block for row {i}")
        D_block = A.data[start + diag_idx[0]]
        Dinv_blocks[i] = np.linalg.inv(D_block)

    # 2) Block multiply A @ T
    AT = A @ T  # BSR multiplication is block-aware

    # 3) Optional: enforce near-nullspace orthogonality (commented out)
    # if near_kernel:
    #     AT = scalar_orthog_projector_bsr(AT, bc_flags)  # implement block version if needed

    # 4) Multiply each block row of AT by block diagonal inverse
    # This is equivalent to block Jacobi step
    new_data = np.zeros_like(T.data)
    for i in range(nblocks):
        start, end = T.indptr[i], T.indptr[i+1]
        new_data[start:end] = Dinv_blocks[i] @ AT.data[start:end]

    # 5) P = T - omega * Dinv * AT
    smoothed_data = T.data - omega * new_data

    P = sp.bsr_matrix((smoothed_data, T.indices, T.indptr), shape=T.shape, blocksize=(b, b))
    return P



class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x



class AMG_BSRSolver:
    """general multilevel AMG solver..."""
    def __init__(self, A_free:sp.bsr_matrix, A:sp.bsr_matrix, xpts, threshold:float=0.25,
                 omega:float=0.7, pre_smooth=1, post_smooth=1, level:int=0, near_kernel:bool=True):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        threshold: float for coarsening threshold
        """
        assert sp.isspmatrix_bsr(A_free)
        assert sp.isspmatrix_bsr(A)

        self.A_free = A_free
        self.A = A
        self.near_kernel = near_kernel
        bs = self.A_free.data.shape[-1]
        nnodes = self.A_free.shape[0] // bs
        self.xpts = xpts.reshape((nnodes, 3))

        # compute node aggregate sets
        # make sure to use unconstrained matrix for aggregation indicators originally
        aggregate_ind = greedy_serial_aggregation_bsr(A_free, threshold=threshold)
        num_agg = np.max(aggregate_ind) + 1

        # print(f"{aggregate_ind=}")
        # TODO: do this
        bc_flags = None
        # bc_flags = get_bc_flags(A)
        # print(f"{bc_flags=}")

        # create tentative prolongator then smooth it
        self.T, self.xpts_c = tentative_prolongator_bsr(self.xpts, aggregate_ind)
        self.P = smooth_prolongator_bsr(self.T, A, bc_flags, omega=omega, near_kernel=near_kernel) # single damped jacobi step, so only one step of fillin
        self.R = self.P.T # sym matrix so restriction is transpose prolong

        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.level = level

        # Galerkin coarse operator
        self.Ac = self.R @ (A @ self.P)
        # on GPU would not do extra allocation for this
        self.Ac_free = self.R @ (A_free @ self.P)

        self.Ac = self.Ac.tocsr()
        self.Ac_free = self.Ac_free.tocsr()
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
        if self.coarse_nnz >= 0.4 * max_coarse_nnz or coarse_nnodes <= 100:
            # then do direct solver
            print(f"level {level+1} building direct solver")
            self.coarse_solver = DirectCSRSolver(self.Ac)
        else:
            print(f"level {level+1} building AMG solver")
            self.coarse_solver = AMG_BSRSolver(self.Ac_free, self.Ac, self.xpts_c, threshold, omega, pre_smooth, post_smooth, level+1, near_kernel=near_kernel)

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

    def solve(self, rhs):
        """
        One AMG V-cycle
        """
        # initial guess
        x = np.zeros_like(rhs)

        # -------- pre-smoothing --------
        if self.pre_smooth > 0:
            x = block_gauss_seidel_6dof(
                self.A, rhs,
                x0=x,
                num_iter=self.pre_smooth
            )

        # fine residual
        r = rhs - self.A @ x

        # restrict
        rc = self.R @ r

        # coarse solve
        ec = self.coarse_solver.solve(rc)

        # prolong correction
        x += self.P @ ec

        # -------- post-smoothing --------
        if self.post_smooth > 0:
            r = rhs - self.A @ x
            dx = block_gauss_seidel_6dof_transpose(
                self.A, r,
                x0=np.zeros_like(rhs),
                num_iter=self.post_smooth
            )
            x += dx

        return x