import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as spp
from scipy.linalg import eig

def get_laplace_system(N):

    n = np.int32(np.sqrt(N))
    nz = 5 * N - 4 * n

    rowp = np.zeros((N+1,), dtype=np.int32)
    cols = np.zeros((nz,), dtype=np.int32)
    vals = np.zeros((nz,))
    b = np.zeros((N,))

    # now define rowp, cols, vals

    idx = 0
    for i in range(N):
        ix = i % n
        iy = i // n

        rowp[i] = idx
        if iy > 0: # up
            vals[idx] = 1.0
            cols[idx] = i - n
            idx += 1
        else:
            b[i] -= 1.0

        if ix > 0: # left
            vals[idx] = 1.0
            cols[idx] = i - 1
            idx += 1
        else:
            pass

        # center
        vals[idx] = -4.0
        cols[idx] = i
        idx += 1

        if (ix < (n-1)):
            vals[idx] = 1.0
            cols[idx] = i + 1
            idx += 1
        else:
            pass

        if (iy < (n-1)):
            vals[idx] = 1.0
            cols[idx] = i + n
            idx += 1

    rowp[N] = idx

    # print(f"{rowp=}\n{cols=}\n{vals=}")

    rows = np.zeros((cols.shape[0],))
    for i in range(cols.shape[0]):
        if i==0:
            rows[i] = 0
        else:
            inc = cols[i] > cols[i-1]
            if inc:
                rows[i] = rows[i-1]
            else:
                rows[i] = rows[i-1] + 1
    # print(F"{rows=}")


    A = spp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A, b

def givens_rotation(a, b):
    """Compute the Givens rotation matrix parameters (c, s) such that:
        [ c  s ] [ a ]  =  [ r ]
        [-s  c ] [ b ]     [ 0 ]
    """
    if b == 0:
        return 1.0, 0.0
    # this part doesn't make sense to me..
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1 / np.sqrt(1 + tau**2)
        c = s * tau
    else:
        tau = -b / a
        c = 1 / np.sqrt(1 + tau**2)
        s = c * tau
    return c, s

def apply_givens_rotation(H, cs, ss, k):
    """Apply the previously computed Givens rotations to H column k."""
    for i in range(k):
        temp = cs[i] * H[i, k] + ss[i] * H[i+1, k]
        H[i+1, k] = -ss[i] * H[i, k] + cs[i] * H[i+1, k]
        H[i, k] = temp

def gmres(A, b, x0=None, tol=1e-8, max_iter=1000, restart=50, M=None):
    """
    Somewhat numerically stable implementation of Modified Gram-Schmidt GMRES
    see Householder one below. This is also a left-precond GMRES, see right PGMRES below (better)
    
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
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r = M_inv(b - (A @ x))
    beta = np.linalg.norm(r)
    print(f"{beta=}")
    
    if beta < tol:
        return x

    for _ in range(max_iter // restart):
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        cs = np.zeros(restart)
        ss = np.zeros(restart)

        # Start Arnoldi process
        V[:, 0] = r / beta
        g[0] = beta
        # print(f"{V[:,0]=}")
        print(f"{g[0]=}")

        print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            w = A @ V[:, j]
            w = M_inv(w)  # Apply preconditioner if given

            # print(f"{j=} {w=}")

            # check norms:
            if j == 0:
                A_norm = np.linalg.norm(A @ V[:, j])
                MA_norm = np.linalg.norm(M_inv(A @ V[:, j]))
                # print(f"{A_norm=} {MA_norm=}")

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
                # print(f"({i=},{j=}) {H[i,j]=}")

            # print(f"post GS {w=}")

            # Reorthogonalization step (fixes numerical loss of orthogonality)
            # optional => can improve numerical stability
            # for i in range(j + 1):
            #     correction = np.dot(V[:, i], w)
            #     w -= correction * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            # print(f"{V[:,j+1]=}")

            givens_option = 2

            if givens_option == 1:
                # there is a mistake in the givens rotations from this one
                # I found the correction in the way we compute givens_rotation(H1, H2)
                # correction on p. 162 of Saad for Sparse Iterative Methods

                # Apply previous Givens rotations
                apply_givens_rotation(H, cs, ss, j)

                # Compute new Givens rotation
                cs[j], ss[j] = givens_rotation(H[j, j], H[j+1, j])
                H[j, j] = cs[j] * H[j, j] + ss[j] * H[j+1, j]
                H[j+1, j] = 0.0

                # Apply rotation to g
                g[j+1] = -ss[j] * g[j]
                g[j] = cs[j] * g[j]

            else:

                # so far all checks out against my old HW3 iterative methods code
                # alternative code from above block:

                for i in range(j):
                    temp = H[i,j]
                    H[i,j] = cs[i] * H[i,j] + ss[i] * H[i+1,j]
                    H[i+1,j] = -ss[i] * temp + cs[i] * H[i+1,j]

                cs[j] = H[j,j] / np.sqrt(H[j,j]**2 + H[j+1,j]**2)
                ss[j] = cs[j] * H[j+1,j] / H[j,j]

                g_temp = g[j]
                g[j] *= cs[j]
                g[j+1] = -ss[j] * g_temp
                # print(f"{cs[j]=}")
                # print(f"{g[j]=}")
                # print(f"{j=} {g[j]=} {g[j+1]=}")

                H[j,j] = cs[j] * H[j,j] + ss[j] * H[j+1,j]
                H[j+1,j] = 0.0

            if (j % 10 == 0): print(f"GMRES [{j}] : {g[j+1]=}")

            # Check convergence
            if abs(g[j+1]) < tol:
                print(f"g break at iteration {j}")
                break

        # debugging check if hessenberg matrix is zero in correct spots
        # plt.imshow(H[:n+1,:n])
        # plt.show()

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])
        # y = np.linalg.solve(H[:j, :j], g[:j])
        # y = np.linalg.solve(H[:restart, :restart], g[:restart])
        # print(f"{H[-1,:j+1]=}")
        Hred = H[:j+1, :j+1]; gred = g[:j+1]
        # print(f"{Hred=} {gred=} {y=}")
        # import matplotlib.pyplot as plt
        # # print(f"{H[j,:j+1]=}")
        # # plt.imshow(H[:j+1, :j+1])
        # # print(f"{j+1=}")
        # plt.imshow(H)
        # plt.show()


        # Update solution
        x += V[:, :j+1] @ y

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
    
    print(f"final resid {beta=}")

    return x

def gmres_householder(A, b, x0=None, tol=1e-8, max_iter=1000, m=50, M=None):
    """
    Householder GMRES (should be more numerically stable than MGS previous one, Modified Gram-Schmidt).
    Also using right PGMRES here
    
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
    m : int, optional
        number of search directions before restart
    M : callable or sparse matrix, optional
        Preconditioner (default: None).
    
    Returns:
    x : ndarray
        Approximate solution.
    """

    # init phase
    # ----------

    n = int(len(b))
    if x0 is None:
        x0 = np.zeros(n)
    
    # define the preconditioner
    if M is None:
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r = b - A @ x # right PGMRES here
    beta = np.linalg.norm(r)
    print(f"{beta=}")

    if beta < tol:
        return x
    
    for _ in range(max_iter // m):
        # V = np.zeros((n, m+1)) # we don't store V in Householder GMRES
        W = np.zeros((n, m+1))
        H = np.zeros((m+1, m))
        g = np.zeros(m+1)
        cs = np.zeros(m)
        ss = np.zeros(m)

        # Start Arnoldi process
        z = r
        beta = None # computed in first iteration

        print(f"HGMRES : outer iter {_} resid {beta=}")
        print("begin main search loop\n------------\n")

        for j in range(m + 1): # j == 0 is one prelim step unlike Gram-Schmidt version
            # compute new Householder unit vector w_j
            # ---------------------------------------

            vsub = z[j:]
            sigma = np.linalg.norm(vsub)
            alpha = -np.sign(vsub[0]) * sigma
            nsub = vsub.shape[0]
            e1 = np.array([1.0] + [0.0] * (nsub - 1))
            u = vsub - alpha * e1
            wtilde = u / np.linalg.norm(u)
            wj = np.concatenate([np.zeros(j), wtilde], axis=0)
            W[:,j] = wj
            # print(f"w[{j}] = {wj}")

            # print(f"{j=} {W[:,j]=}")

            # compute Hessenberg vec
            # -----------------------
            
            # apply reflector Pj = I - 2 w_j w_j^T to get h_{j-1} here
            hvec = z - 2.0 * np.dot(wj, z) * wj
            # print(f"h[{j-1}] = {hvec}")

            # see page 160 of Saad book on Sparse Linear Systems
            if j == 0:
                beta = hvec[0]
                g[0] = beta
                print(f"{g[0]=} {np.linalg.norm(z)=}")
            else:
                # zero out part of hvec past j+1
                H[:j+1,j-1] = hvec[:j+1]

                # print(f"{H[:j+1,j-1]=}")

            # if j == 3: 
            #     print(f"{H[:j+1,:j]=}")
            #     exit()

            # double check householder unit vec requirements (debug/devel)
            # -----------------------------------------------
            
            # need h_{j-1} zero for all j+1 entries onwards
            constr1 = np.abs(np.sum(wj[:(j-1)])) < 1e-10 or (j == 0)
            constr2 = np.abs(np.sum(hvec[j+1:])) < 1e-10
            # print(f"{constr1=} {constr2=}")
            assert(constr1)
            assert(constr2)

            # perform Givens rotations of Hessenberg to upper triangular
            # ----------------------------------------------------------

            givens = True # for debugging
            if j > 0 and givens:
                # print(f"{H[:j+1,:j]=}")

                # trying same as before.. except j is j-1
                for i in range(j-1):
                    # temp debug
                    hx = H[i,j-1]; hy = H[i+1,j-1]
                    c = cs[i]; s = ss[i]
                    # print(f"{hx=} {hy=} {c=} {s=}")
                    res1 = c * hx + s * hy
                    res2 = -s * hx + c * hy

                    temp = H[i,j-1]
                    H[i,j-1] = cs[i] * H[i,j-1] + ss[i] * H[i+1,j-1]
                    H[i+1,j-1] = -ss[i] * temp + cs[i] * H[i+1,j-1]

                    # print(f"{res1=} {H[i,j-1]=}\n{res2=} {H[i+1,j-1]=}")

                # cs[j-1] = H[j-1,j-1] / np.sqrt(H[j-1,j-1]**2 + H[j,j-1]**2)
                # ss[j-1] = cs[j-1] * H[j,j-1] / H[j-1,j-1]

                r = np.hypot(H[j-1, j-1], H[j, j-1])  # This avoids overflow/underflow
                cs[j-1] = H[j-1, j-1] / r
                ss[j-1] = H[j, j-1] / r

                # print(f"{cs[j-1]=} {ss[j-1]=}")
                # print(f"p2 {H[:j+1,j-1]=}")

                # g_temp = g[j-1]
                # g[j-1] *= cs[j-1]
                # g[j] = -ss[j-1] * g_temp

                temp = g[j-1]
                g[j-1] = cs[j-1] * g[j-1] + ss[j-1] * g[j]
                g[j] = -ss[j-1] * temp + cs[j-1] * g[j]
                # print(f"{cs[j]=}")
                # print(f"{g[j]=}")
                # print(f"{j=} {g[j]=} {g[j+1]=}")

                H[j-1,j-1] = cs[j-1] * H[j-1,j-1] + ss[j-1] * H[j,j-1]
                H[j,j-1] = 0.0

                # print(f"{g[j-1]=} {g[j]=}")
                # print(f"p3 {H[:j+1,j-1]=}")

                # print(f"{H[:j+1,:j]=} after")
                if (j % 10 == 0): print(f'HGMRES [{j}] = {g[j]}')

            # Check convergence
            if abs(g[j]) < tol and j > 0 and givens:
                print(f"g break at iteration {j}")
                break
                
            # if not givens and j > 4:
            #     break


            # orthog v = P1 * ... * Pj * ej
            # ----------------------------------
            ej = np.zeros((n,))
            ej[j] = 1.0
            v = ej
            # print(f"{ej.shape=}")

            # apply the reflectors in reverse
            for i in range(j, -1, -1):
                wi = W[:,i]
                v -= 2.0 * np.dot(wi, v) * wi

            # update old v now
            # print(f"v = {v}")
            # print("------------\n")

            # compute new Arnoldi vec and orthog
            # ----------------------------------

            if j <= m:
                # z := Pj * ... * P1 * A * v
                # apply the reflectors forwards
                tmp = M_inv(v) # right precond here
                z = A @ tmp
                for i in range(j+1):
                    wi = W[:,i]
                    z -= 2.0 * np.dot(wi, z) * wi

        # done with search direction loop, solve Hessenberg triang system
        # print(f"{H[:j,:j]=}")

        # I think I'm solving one dim too high here..
        y = np.linalg.solve(H[:j,:j], g[:j])
        # print(f"{y=}")

        # print(f"Hessenberg ({j} x {j}) system\n")
        # for i in range(j):
        #     print(f"{H[i,:j]}")
        # print(f"\n{g[:j]=}")
        # print(f"\n{y[:j]=}")

        # compute change in solution reusing reflectors
        z = np.zeros(n)
        for i in range(j-1, -1, -1):
            # compute z = Pi * (y[i] * ei + z)
            ei = np.zeros(n)
            ei[i] = 1.0
            tmp = y[i] * ei + z
            # print(f"y[{i}]={y[i]}\n\t{tmp=}")

            wi = W[:,i]
            z = tmp - 2 * np.dot(tmp, wi) * wi

        # print(f"{z=}")
        # right precond GMRES update here?
        z = M_inv(z)
        # print(f"precond {z=}")

        # print(f"pre {x=}")

        # update final solution (maybe restarting again)
        x += z

        # print(f"post {x=}")

        # Compute new residual, don't need to compute in final version here only can put in debug block
        # fine here in python version
        r = b - (A @ x)
        resid_norm = np.linalg.norm(r)
        if resid_norm < tol:
            break

    print(f"final resid {resid_norm=}")

    return x

def gmres_dr(A, b, x0=None, tol=1e-8, m=30, k=10, max_iter=300, M=None):
    """
    deflated GMRES with m subspace size, k deflation eigvecs, max_iter total iterations before exit
    should reduce orthogonalization cost and reduce memory requirements for larger problems
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r0 = M_inv(b - (A @ x))
    beta = np.linalg.norm(r0)
    print(f"{beta=}")
    
    if beta < tol:
        return x
    
    # save in perm storage, cause we don't want to zero out in each loop, need to thick restart
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m)) # upper Hessenberg (nearly triang)
    Htmp = np.zeros((m+1, m))
    # would need an Htemp storage here
    g = np.zeros((m+1,))
    Zk = None

    converged = False

    total_iter = 0
    while (total_iter < max_iter):
        
        # fine to zero out cs, ss, and g each new restart
        g *= 0.0

        # Start Arnoldi process
        if total_iter == 0: # first time through or irestart == 0 basically
            V[:, 0] = r0 / beta
            g[0] = beta
            w = V[:,0]
            start = 0
        else: # restarting with deflation
            # g = V[:,:m].T @ r0 # basically c = V^T r0 part
            # could also do g[k] = <w, r0> most of the time, you choose
            g[k] = np.dot(w, r0)
            start = k

        # starts later if deflated
        for j in range(start, m):
            total_iter += 1
            # right precond
            w = M_inv(w) 
            w = A @ V[:, j]

            # Modified GS, compute new H column
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                # print(f"H[{i},{j}]={H[i,j]}")
                w -= H[i, j] * V[:, i]

            # update w => v_{j+1}
            H[j+1, j] = np.linalg.norm(w)
            w = w / H[j+1, j]
            V[:, j+1] = w

        # solve the least-squares Hessenberg system (though has dense subblock in it now)
        y = np.linalg.solve(H[:m,:], g[:m])
        # least-squares solution was worse, not sure why (and worse conv)
        # see prototype gmres_dr below if you want
        # print(f"{H.shape=} {g.shape=}")
        # y, *_ = np.linalg.lstsq(H, g, rcond=None)

        # for i in range(m):
        #     print(f"H[{i},:] = {H[i,:]}")

        # Update solution
        z = V[:,:m] @ y
        z = M_inv(z) # right precond
        x += z

        # Compute new residual
        r0 = b - (A @ x)
        new_beta = np.linalg.norm(r0)
        print(f"GMRES-DR [{total_iter}]: resid {new_beta}")
        if new_beta < tol:
            break

        # TODO : probably should leave it to resid check to break
        if converged: break

        # cleaned up version
        _, eigvecs = np.linalg.eig(H[:m, :])
        Zk = eigvecs[:,:k]
        Phik = V[:, :m] @ Zk

        # set new Krylov basis
        V[:,:k] = Phik
        V[:,k] = V[:,m]
        V[:,k+1:] = 0.0

        # update Hessenberg matrix
        Htmp = H[:m, :m].copy()
        beta = H[m, m-1]
        H[:k,:k] = Zk.T @ Htmp @ Zk
        H[k,:k] = beta * Zk[-1,:]

    print(f"final resid {new_beta=} in {total_iter=}")

    return x

def gmres_dr_prototype(A, b, x0=None, tol=1e-8, m=30, k=10, max_iter=300, M=None):
    """
    deflated GMRES with m subspace size, k deflation eigvecs, max_iter total iterations before exit
    should reduce orthogonalization cost and reduce memory requirements for larger problems
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r0 = M_inv(b - (A @ x))
    beta = np.linalg.norm(r0)
    print(f"{beta=}")
    
    if beta < tol:
        return x
    
    # save in perm storage, cause we don't want to zero out in each loop, need to thick restart
    V = np.zeros((n, m+1))
    H = np.zeros((m+1, m)) # upper Hessenberg (nearly triang)
    R = np.zeros((m+1,m)) # upper triangular
    g = np.zeros(m+1)
    cs = np.zeros(m)
    ss = np.zeros(m)
    Zk = None

    converged = False

    total_iter = 0
    while (total_iter < max_iter):
        
        # fine to zero out cs, ss, and g each new restart
        g *= 0.0
        cs *= 0.0
        ss *= 0.0

        # Start Arnoldi process
        if total_iter == 0: # first time through or irestart == 0 basically
            V[:, 0] = r0 / beta
            g[0] = beta
            # print(f"{V[:,0]=}")
            print(f"{g[0]=}")
            w = V[:,0]
            start = 0
        else: # restarting with deflation
            # compute new starting RHS
            g = V.T @ r0 # basically c = V^T r0 part
            # print(f"{g=}")
            # can just set g[k] = w^T r0
            # g[k] = np.dot(V[:,k], r0) # equiv to above in this case, I checked
            # then add r0 as new unit vec
            # V[:,k] = r0 / np.linalg.norm(r0)
            start = k

        # starts later if deflated
        for j in range(start, m):
            total_iter += 1
            # right precond
            w = M_inv(w) 
            w = A @ V[:, j]

            # Modified GS, compute new H column
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            # copy to Rotated Hessenberg R
            # R[:j+1,j] = H[:j+1,j]

            # update w => v_{j+1}
            H[j+1, j] = np.linalg.norm(w)
            # R[j+1,j] = H[j+1,j] # copy for rotated Hessenberg
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            # print(f"{V[:,j+1]=}")

            # givens rotations doesn't work after restarts.. (k+1 x k block not triangular)

            # # # perform Givens rotations on R rotated Hessenberg
            # # # ------------------------------------------------

            # # # apply prev Givens rotations to R
            # # for i in range(j):
            # #     temp = R[i,j]
            # #     R[i,j] = cs[i] * R[i,j] + ss[i] * R[i+1,j]
            # #     R[i+1,j] = -ss[i] * temp + cs[i] * R[i+1,j]

            # # # compute new Givens rotations
            # # r = np.hypot(R[j,j], R[j+1,j])
            # # cs[j] = R[j,j] / r
            # # ss[j] = R[j+1,j] / r

            # # # update RHS g with Givens
            # # g_temp = g[j]
            # # g[j] *= cs[j]
            # # g[j+1] = -ss[j] * g_temp

            # # # apply new Givens rotation to R, equiv to this
            # # R[j,j] = r
            # # R[j+1,j] = 0.0

            # print(f"iteration {j} : {g[j+1]=}")

            # # Check convergence
            # if abs(g[j+1]) < tol:
            #     converged = True
            #     print(f"g break at iteration {j}")
            #     break

        # debugging check if hessenberg matrix is zero in correct spots
        # plt.imshow(H[:n+1,:n])
        # plt.show()

        # Solve the upper triangular system R * y = g
        # valid for first solve:
        # y = np.linalg.solve(R[:j+1, :j+1], g[:j+1])
        # givens doesn't work in thick restart
        y = np.linalg.solve(H[:j+1,:j+1], g[:j+1])

        # for Arnoldi system check and Eig problem later
        # one higher since this is exclusive and 0 is starting column
        Hbarj = H[:j+2,:j+1].copy()
        Hj = H[:j+1,:j+1].copy()
        Vj = V[:,:j+1].copy()
        Vj1 = V[:,:j+2].copy()
        beta = Hbarj[-1,-1]
        vlast = Vj1[:,-1:].copy()
        # print(f"{np.linalg.norm(vlast)=}")
        # exit()
        # print(f"{Vj1=} {vlast=}")
        # print(f"{Hbarj=} {beta=}")
        # print(f"{Vj1=}")

        # check Arnoldi equation (with right precond)
        AV = A @ M_inv(Vj)
        VH = Vj1 @ Hbarj
        emT = np.zeros((1,j+1))
        emT[0,-1] = 1.0
        VH2 = Vj @ Hj
        # print(f"{vlast.shape=} {emT.shape=} {AV.shape=}")
        VH2 += beta * vlast @ emT
        # arnoldi_diff = AV - VH
        arnoldi_diff = AV - VH2
        # print(f"{arnoldi_diff=}")
        arnoldi_resid = np.linalg.norm(arnoldi_diff)
        # print(f"{arnoldi_resid=}")

        # Update solution
        z = Vj @ y
        z = M_inv(z) # right precond
        x += z

        # Compute new residual
        r0 = b - (A @ x)
        new_beta = np.linalg.norm(r0)
        print(f"{new_beta=}")
        if new_beta < tol:
            break

        # TODO : probably should leave it to resid check to break
        if converged: break

        # otherwise it has not converged, now we start the deflation process
        # paper says to do this
        # eigvals, eigvecs = eig(Hbarj.T @ Hbarj, Hj.T)
        # but this worked way better and has no complex values?
        eigvals, eigvecs = np.linalg.eig(Hj)
        Zk_nonortho = eigvecs[:,:k]
        Zk, _ = np.linalg.qr(Zk_nonortho)
        Phik = Vj @ Zk
        # check orthonormal eigvalues
        ortho_check = Zk.T @ Zk
        # print(f'{ortho_check=}')

        # update V now to restart with these eigvecs
        V *= 0.0
        V[:,:k] = Phik
        # recompute new H matrix
        # more numerically efficient to copy R into H then saves comp
        H *= 0.0
        H[:k,:k] = Zk.T @ Hj @ Zk

        # k+1th (but 0-based) row = beta * em^T Zk or beta * mth row of Zk (last row)
        H[k,:k] = beta * Zk[-1,:]
        Hk = H[:k,:k]
        Hbark = H[:k+1,:k]
        V[:,k] = vlast[:,0]
        # print(f"{H[:k+1,:k]=}")

        # now check A * Phik vs Phik * Hk (where do we need to add residual term)
        AP = A @ M_inv(V[:,:k])
        AP2 = A @ Vj @ Zk
        AP_diff = AP - AP2
        # print(f"{AP_diff=}")
        PH1_1 = Phik @ Hk
        PH1_2 = Vj @ Hj @ Zk
        PHI1_DIFF = PH1_1 - PH1_2
        # this above equation is only true in Zk^T Vk^T reduced space (not in full space)
        # print(f"{PHI1_DIFF=}")
        # diff = AP - PH
        # print(f"{AP=}")
        # print(f"{PH=}") 
        # print(f"{diff=}")

    print(f"final resid {new_beta=} in {total_iter=}")

    return x