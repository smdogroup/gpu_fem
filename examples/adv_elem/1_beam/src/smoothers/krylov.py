import sys
sys.path.append("../../../milu/")
import numpy as np
# from __linalg import right_pgmres, right_pcg 


def right_pcg2(A, b, x0=None, rtol=1e-8, atol=1e-7, max_iter=1000, M=None, print_freq:int=10, norm_hist:list=[]):
    """
    Right-preconditioned Conjugate Gradient method (version from my scitech paper)

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

    if norm_r0 < atol:
        return x
    
    rz_old = None
    rz = None

    norm_hist += [1.0]

    print(f"PCG iter 0: ||r|| = {norm_r0:.3e}")

    for k in range(1, max_iter + 1):

        z = Minv(r)
        rz = np.dot(r, z)

        if k == 1:
            p = z.copy()
        else:
            beta = rz / rz_old
            p = z + beta * p_old

        p_old = p.copy()
        rz_old = rz * 1.0

        w = A @ p
        alpha = rz / np.dot(w, p)
        x += alpha * p
        r -= alpha * w
        norm_r = np.linalg.norm(r)

        rel_nrm = norm_r / norm_r0
        norm_hist += [float(rel_nrm)]

        if (k % print_freq == 0) or norm_r < (atol + rtol * norm_r0):
            print(f"PCG iter {k}: ||r|| = {norm_r:.3e}")

        if norm_r < (atol + rtol * norm_r0):
            break

    print(f"PCG finished at iter {k}, ||r|| = {norm_r:.3e}")
    return x, k+1


def right_pgmres2(A, b, x0=None, restart=50, tol=1e-8, max_iter=1000, M=None, rtol:float=1e-6, norm_hist:list=[]):
    """
    Right-precond Modified Gram-Schmidt GMRES
    
    Parameters:
    A : callable or sparse matrix
        Function A(x) or sparse matrix representing the linear system.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess (default: zero vector).
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    restart : int, optional
        Restart parameter (default: 50).
    M : callable or sparse matrix, optional
        Preconditioner (default: None).
    
    Returns:
    x : ndarray
        Approximate solution.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda y: y
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda y : M.solve(y)

    x = x0.copy()
    r = b - (A @ x)
    beta = np.linalg.norm(r)
    init_defect_nrm = beta
    # print(f"{beta=}")

    norm_hist += [1.0]
    
    if beta < tol:
        return x
    
    jj = 0

    for _ in range(max_iter // restart):
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        cs = np.zeros(restart)
        ss = np.zeros(restart)

        # Start Arnoldi process
        V[:, 0] = r / beta
        g[0] = beta
        # print(f"{g[0]=}")

        if jj == 0:
            print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            jj += 1
            z = M_inv(V[:,j])
            w = A @ z

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            # givens rotations 
            # ----------------

            for i in range(j):
                temp = H[i,j]
                H[i,j] = cs[i] * H[i,j] + ss[i] * H[i+1,j]
                H[i+1,j] = -ss[i] * temp + cs[i] * H[i+1,j]

            cs[j] = H[j,j] / np.sqrt(H[j,j]**2 + H[j+1,j]**2)
            ss[j] = cs[j] * H[j+1,j] / H[j,j]

            g_temp = g[j]
            g[j] *= cs[j]
            g[j+1] = -ss[j] * g_temp

            H[j,j] = cs[j] * H[j,j] + ss[j] * H[j+1,j]
            H[j+1,j] = 0.0

            if (jj % 4 == 0): print(f"GMRES [{jj}] : {g[j+1]=}")

            norm_hist += [abs(g[j+1]) / init_defect_nrm]

            # Check convergence
            if abs(g[j+1]) <= (tol + init_defect_nrm * rtol):
                # print(f"g break at iteration {jj}")
                break

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])

        # Update solution
        dz = V[:, :j+1] @ y
        dx = M_inv(dz)
        x += dx

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < (tol + init_defect_nrm * rtol):
            break
    
    print(f"GMRES final resid {beta=}")
    return x, jj
