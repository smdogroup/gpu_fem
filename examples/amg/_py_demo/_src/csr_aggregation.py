import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from _smoothers import gauss_seidel_csr, gauss_seidel_csr_transpose

def strength_matrix_csr(A:sp.csr_matrix, threshold:float=0.25):
    """
    Compute strength of connections C_{ij} for sparse CSR matrix (slightly different than above strength and strength^T version)
    produces only one matrix

    comes from this paper on aggregation
    https://epubs.siam.org/doi/epdf/10.1137/110838844
    """

    assert sp.isspmatrix_csr(A)

    # strength of connection graph
    nnodes = A.shape[0]
    
    # get diagonals
    diags = []
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            if i == j:
                diags += [np.abs(A.data[jp])]
    
    STRENGTH = [[] for i in range(nnodes)]
    for i in range(nnodes):
        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            value = np.abs(A.data[jp])
            lb = threshold * np.sqrt(diags[i] * diags[j])
            if value >= lb:
                STRENGTH[i] += [j]
    # can be converted into CSR style NZ pattern (though has no float data, just NZ pattern C_{ij})
    return STRENGTH


def greedy_serial_aggregation_csr(A:sp.csr_matrix, threshold:float=0.25):
    """from paper https://epubs.siam.org/doi/epdf/10.1137/110838844, with parallel versions discussed also
    and GPU version here https://www.sciencedirect.com/science/article/pii/S0898122114004143"""

    # greedy serial aggregation
    assert sp.isspmatrix_csr(A)
    nnodes = A.shape[0]
    STRENGTH = strength_matrix_csr(A, threshold)
    # print(f"{STRENGTH=}")

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

def plot_plate_aggregation(aggregate_ind, nx, ny, Lx, Ly):
    """plot aggregation groups with color + numbers"""
    num_agg = np.max(aggregate_ind) + 1

    # plot color aggregates
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Xf = X.ravel()
    Yf = Y.ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    # optional: still color by aggregate
    colors = plt.cm.jet(np.linspace(0.0, 1.0, num_agg + 1))

    for iagg in range(num_agg):
        agg_mask = aggregate_ind == iagg
        # print(f"{agg_mask=}")
        # print(f"{Xf[:10]=}\n{Yf[:10]=}")
        ax.scatter(
            Xf[agg_mask], Yf[agg_mask],
            color=colors[iagg], s=120, edgecolors="k"
        )

    # write aggregate index inside each node
    for i in range(len(Xf)):
        ax.text(
            Xf[i], Yf[i],
            str(int(aggregate_ind[i])),
            ha="center", va="center",
            fontsize=8, color="white", weight="bold"
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Greedy serial aggregation (aggregate id per node)")

    plt.show()

def tentative_prolongator_csr(aggregate_ind:np.ndarray):
    # construct tentative prolongator T such that T * Bc = B and T^T T = I
    # however first I'm just following instructions from energy minimization, https://link.springer.com/article/10.1007/s006070050022 paper
    # for scalar PDEs
    nnodes = aggregate_ind.shape[0]
    num_agg = np.max(aggregate_ind) + 1

    # construct CSR sparsity pattern.. first construct rowp
    row_cts = np.zeros(nnodes, dtype=np.int32)
    for iagg in range(num_agg):
        row_cts[aggregate_ind == iagg] += 1
    # print(f"{row_cts=}")
    # then build rowp
    rowp = np.zeros(nnodes + 1, dtype=np.int32)
    for i in range(nnodes):
        rowp[i+1] = rowp[i] + row_cts[i]
    nnz = rowp[-1]
    # print(f"{rowp=}")
    # could do this simpler way since each row only contains one NZ in this way..
    cols = np.zeros(nnz, dtype=np.int32)
    assert nnz == nnodes # assumption to make cols part easier
    for iagg in range(num_agg):
        cols[aggregate_ind == iagg] = iagg
    
    data = np.ones(nnz, dtype=np.double)
    return sp.csr_matrix((data, cols, rowp), shape=(nnodes, num_agg))

def spectral_radius_DinvA_csr(A: sp.spmatrix, maxiter=200, tol=1e-8):
    """
    Estimate spectral radius of D^{-1} A for a CSR matrix A.
    Uses a matrix-free LinearOperator.

    Parameters
    ----------
    A : scipy.sparse matrix (CSR preferred)
    maxiter : int
    tol : float

    Returns
    -------
    rho : float
        Estimated spectral radius
    """
    if not sp.sparse.isspmatrix_csr(A):
        A = A.tocsr()

    n = A.shape[0]

    # Diagonal inverse
    D = A.diagonal()
    if np.any(D == 0.0):
        raise ValueError("Zero diagonal entry in A")

    Dinv = 1.0 / D

    def matvec(x):
        return Dinv * (A @ x)

    M = spla.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=A.dtype
    )

    # Largest magnitude eigenvalue
    eigval = spla.eigs(
        M,
        k=1,
        which="LM",
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=False
    )

    return np.abs(eigval[0])

def get_bc_flags(A: sp.csr_matrix, tol=1e-14) -> np.ndarray:
    """Return True for constrained DOFs in CSR matrix A"""
    return np.array([
        # row i
        np.all(np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] != i]) < tol)  # off-diagonals ~ 0
        and np.abs(A.data[A.indptr[i]:A.indptr[i+1]][A.indices[A.indptr[i]:A.indptr[i+1]] == i][0] - 1.0) < tol  # diagonal ~1
        for i in range(A.shape[0])
    ], dtype=bool)


def scalar_orthog_projector(dP:sp.csr_matrix, bc_flags):
    """scalar orthogonal projector from energy opt paper https://link.springer.com/article/10.1007/s006070050022"""
    # compute and subtract away row-sums from dP the update to prolongation matrix
    for i in range(dP.shape[0]):
        if bc_flags[i]: continue
        row_ips = np.arange(dP.indptr[i], dP.indptr[i+1])
        row_vals = dP.data[row_ips]
        dP.data[row_ips] -= np.mean(row_vals)
    return dP



def smooth_prolongator_csr(T:sp.csr_matrix, A:sp.csr_matrix, bc_flags:np.ndarray, omega:float=0.7, near_kernel:bool=True):
    # single step jacobi smoothing of tentative prolongator
    # now have P = (I - omega * Dinv * A) * T where T is tentative and P is smoothed prolong
    # this introduces some fillin from T to P
    Dinv = 1.0 / A.diagonal()
    AT = A @ T
    if near_kernel:
        AT = scalar_orthog_projector(AT, bc_flags)
        AT = AT.tocsr()
    DAT = AT.multiply(Dinv[:, None])
    newP = T - omega * DAT
    return newP

# def galerkin_coarse_grid_csr(R:sp.csr_matrix, A_fine:sp.csr_matrix, P:sp.csr_matrix):
#     return R @ A_fine @ P    
    

class DirectCSRSolver:
    def __init__(self, A_csr):
        # convert to dense matrix (full fillin)
        self.A = A_csr.tocsc()

    def solve(self, rhs):
        # use python dense solver..
        x = sp.linalg.spsolve(self.A, rhs)
        return x



class AMGSolver:
    """general multilevel AMG solver..."""
    def __init__(self, A_free:sp.csr_matrix, A:sp.csr_matrix, threshold:float=0.25,
                 omega:float=0.7, pre_smooth=1, post_smooth=1, level:int=0, near_kernel:bool=True):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        threshold: float for coarsening threshold
        """
        assert sp.isspmatrix_csr(A_free)
        assert sp.isspmatrix_csr(A)

        self.A_free = A_free
        self.A = A
        self.near_kernel = near_kernel

        # compute node aggregate sets
        # make sure to use unconstrained matrix for aggregation indicators originally
        aggregate_ind = greedy_serial_aggregation_csr(A_free, threshold=threshold)
        num_agg = np.max(aggregate_ind) + 1

        # print(f"{aggregate_ind=}")
        bc_flags = get_bc_flags(A)
        # print(f"{bc_flags=}")

        # create tentative prolongator then smooth it
        self.T = tentative_prolongator_csr(aggregate_ind)
        self.P = smooth_prolongator_csr(self.T, A, bc_flags, omega=omega, near_kernel=near_kernel) # single damped jacobi step, so only one step of fillin
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
            self.coarse_solver = AMGSolver(self.Ac_free, self.Ac, threshold, omega, pre_smooth, post_smooth, level+1, near_kernel=near_kernel)

        if level == 0:
            print(f"Multilevel AMG with {self.num_levels=} and {self.operator_complexity=}")
            print(f"\tnum nodes per level = [{self.num_nodes_list}]")

    @property
    def total_nnz(self) -> int:
        # get total nnz across all levels
        if isinstance(self.coarse_solver, AMGSolver):
            return self.fine_nnz + self.coarse_solver.total_nnz
        else: # direct solver
            return self.fine_nnz + self.coarse_nnz

    @property
    def operator_complexity(self) -> float:
        return self.total_nnz / self.fine_nnz     

    @property
    def num_levels(self) -> int:
        if isinstance(self.coarse_solver, AMGSolver):
            return self.coarse_solver.num_levels + 1
        else: # direct solver
            return 2
        
    @property
    def num_nodes_list(self) -> str:
        if isinstance(self.coarse_solver, AMGSolver):
            return str(self.A.shape[0]) + "," + self.coarse_solver.num_nodes_list
        else: # direct solver
            return f"{self.A.shape[0]},{self.Ac.shape[0]}"

    def solve(self, rhs):
        """
        One AMG V-cycle
        """
        # initial guess
        x = np.zeros_like(rhs)

        # -------- pre-smoothing --------
        if self.pre_smooth > 0:
            x = gauss_seidel_csr(
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
            dx = gauss_seidel_csr_transpose(
                self.A, r,
                x0=np.zeros_like(rhs),
                num_iter=self.post_smooth
            )
            x += dx

        return x