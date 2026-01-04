
def bgs_bsr_smoother(A: bsr_matrix, P: bsr_matrix, num_iter: int = 1, omega: float = 1.0):
    """
    Block Gauss-Seidel forward solve for (L + D) X = P using 6x6 BSR blocks.
    Returns X with the same sparsity pattern as P (i.e. same indices/indptr).
    """
    if not isinstance(A, bsr_matrix) or not isinstance(P, bsr_matrix):
        raise TypeError("A and P must be scipy.sparse.bsr_matrix")
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nblocks_row = A.shape[0] // ndof

    # Build quick-access maps for A and P blocks
    A_rows = {i: [] for i in range(nblocks_row)}
    for i in range(nblocks_row):
        for aidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[aidx]
            A_rows[i].append((j, A.data[aidx]))   # (column-block-index, 6x6 block)

    # P_blocks: mapping (i,k) -> block (copy)
    P_blocks = {}
    for i in range(nblocks_row):
        for pidx in range(P.indptr[i], P.indptr[i+1]):
            k = P.indices[pidx]
            P_blocks[(i, k)] = P.data[pidx].copy()

    # Initialize X_blocks possibly from previous iteration (start zeros)
    X_blocks = {}  # mapping (i,k) -> block (6x6)

    for _it in range(num_iter):
        # forward sweep over block-rows
        for i in range(nblocks_row):
            # find diagonal Aii
            Aii = None
            for (j, Aij) in A_rows[i]:
                if j == i:
                    Aii = Aij
                    break
            if Aii is None:
                raise ValueError(f"No diagonal block found in A row {i}")
            
            print(f"brow {i=} => {Aii=}")

            # set of block-columns to compute in row i (only those present in P)
            cols_i = [k for (ii, k) in P_blocks.keys() if ii == i]
            if not cols_i:
                # nothing to do for this row (P has no blocks here) -> continue
                continue

            print(f"brow {i=} => {cols_i=}")

            for k in cols_i:
                # compute rhs = P_{i,k} - sum_{j<i} A_{i,j} * X_{j,k}
                rhs = P_blocks.get((i, k), np.zeros((ndof, ndof)))
                # accumulate contributions from previously solved rows j < i
                for (j, Aij) in A_rows[i]:
                    if j >= i:
                        continue
                    x_jk = X_blocks.get((j, k))
                    if x_jk is None:
                        # if X_{j,k} was never present (sparsity), treat as zero
                        continue
                    rhs = rhs - (Aij @ x_jk)

                # solve Aii * x = rhs
                x_new = np.linalg.solve(Aii, rhs)

                # relaxation with any previous X value (Gauss-Seidel omega)
                x_old = X_blocks.get((i, k), np.zeros_like(x_new))
                X_blocks[(i, k)] = (1.0 - omega) * x_old + omega * x_new

    # assemble new data array preserving P's sparsity order
    new_data = np.empty_like(P.data)
    for i in range(nblocks_row):
        pstart = P.indptr[i]
        pend = P.indptr[i+1]
        for pidx in range(pstart, pend):
            k = P.indices[pidx]
            block = X_blocks.get((i, k))
            if block is None:
                # If no computed block (shouldn't happen unless P had none), set zeros
                block = np.zeros((ndof, ndof))
            new_data[pidx] = block

    Pnew = bsr_matrix((new_data, P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)
    return Pnew




class AMG2GridSolver:
    def __init__(self, A_free:sp.csr_matrix, A:sp.csr_matrix, threshold:float=0.25,
                 omega:float=0.7, pre_smooth=1, post_smooth=1):
        """
        A : fine-grid operator (CSR) without bcs
        A : fine-grid operator (CSR) with bcs
        threshold: float for coarsening threshold
        """
        self.A_free = A_free
        self.A = A

        # compute node aggregate sets
        # make sure to use unconstrained matrix for aggregation indicators originally
        aggregate_ind = greedy_serial_aggregation_csr(A_free, threshold=threshold)
        num_agg = np.max(aggregate_ind) + 1
        print(f"{num_agg=}")

        # create tentative prolongator then smooth it
        self.T = tentative_prolongator_csr(aggregate_ind)
        self.P = smooth_prolongator_csr(self.T, A, omega=omega) # single damped jacobi step, so only one step of fillin
        self.R = self.P.T # sym matrix so restriction is transpose prolong

        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth

        # Galerkin coarse operator
        self.Ac = self.R @ (A @ self.P)

        # compute two-grid operator complexity
        self.fine_nnz = self.A.nnz
        self.coarse_nnz = self.Ac.nnz
        self.operator_complexity = (self.fine_nnz + self.coarse_nnz) / self.fine_nnz
        print(f"Two-grid AMG: {self.operator_complexity=}")

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
        ec = sp.linalg.spsolve(self.Ac, rc)

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
    

def right_pcg_v0(A, b, x0=None, tol=1e-8, max_iter=1000, M=None):
    """
    Right-preconditioned Conjugate Gradient method

    Solves: A M^{-1} y = b, with x = M^{-1} y

    Parameters
    ----------
    A : callable or sparse matrix
        Linear operator A(x) or sparse matrix
    b : ndarray
        Right-hand side
    x0 : ndarray, optional
        Initial guess for x
    tol : float
        Convergence tolerance on ||r||
    max_iter : int
        Maximum iterations
    M : object, optional
        Preconditioner with method M.solve(x) ≈ M^{-1} x

    Returns
    -------
    x : ndarray
        Approximate solution
    """

    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Preconditioner application
    if M is None:
        Minv = lambda x: x
    else:
        Minv = lambda x: M.solve(x)

    # Initial residual
    r = b - (A @ x)
    norm_r0 = np.linalg.norm(r)

    if norm_r0 < tol:
        return x

    # Right-preconditioned variables
    z = Minv(r)              # z = M^{-1} r
    p = z.copy()

    rz_old = np.dot(r, z)

    print(f"PCG iter 0: ||r|| = {norm_r0:.3e}")

    for k in range(1, max_iter + 1):

        # Apply A M^{-1}
        Ap = A @ p

        alpha = rz_old / np.dot(p, Ap)

        # Update y-space implicitly via x
        x += alpha * p

        # Residual in original space
        r -= alpha * Ap
        norm_r = np.linalg.norm(r)

        if (k % 10 == 0) or norm_r < tol:
            print(f"PCG iter {k}: ||r|| = {norm_r:.3e}")

        if norm_r < tol:
            break

        # Apply preconditioner
        z = Minv(r)
        rz_new = np.dot(r, z)

        beta = rz_new / rz_old
        p = z + beta * p

        rz_old = rz_new

    print(f"PCG finished at iter {k}, ||r|| = {norm_r:.3e}")
    return x




def block_jacobi_smooth_operator(A: bsr_matrix, P: bsr_matrix, omega=1.0):
    """
    Apply block-Jacobi smoother:
        P_new = (I - omega * D^{-1} A) P
    """
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nrows = A.shape[0] // ndof

    # Extract D^{-1}
    Dinv = []
    for i in range(nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[p] == i:
                Dinv.append(np.linalg.inv(A.data[p]))
                break
        else:
            raise ValueError(f"Missing diagonal block at row {i}")

    # Compute A * P
    AP = A @ P

    # Scale rows of AP by D^{-1}
    AP_scaled = AP.copy()
    for i in range(nrows):
        for p in range(AP.indptr[i], AP.indptr[i + 1]):
            AP_scaled.data[p] = Dinv[i] @ AP.data[p]

    # P_new = P - omega * Dinv * A * P
    P_new = P - omega * AP_scaled
    return P_new

def block_jacobi_smooth_operator_inplace(A, P, omega=1.0):
    """
    Block-Jacobi smoothing of P:
        P <- (I - omega D^{-1} A) P
    keeping P sparsity.
    """
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nrows = A.shape[0] // ndof

    # Extract D^{-1}
    Dinv = [None] * nrows
    for i in range(nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[p] == i:
                Dinv[i] = np.linalg.inv(A.data[p])
                break

    Pnew = P.copy()

    for i in range(nrows):
        for pidx in range(P.indptr[i], P.indptr[i + 1]):
            k = P.indices[pidx]

            accum = np.zeros((ndof, ndof))
            for aidx in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[aidx]

                # find P_{j,k}
                for q in range(P.indptr[j], P.indptr[j + 1]):
                    if P.indices[q] == k:
                        accum += A.data[aidx] @ P.data[q]
                        break

            Pnew.data[pidx] -= omega * (Dinv[i] @ accum)

    return Pnew
