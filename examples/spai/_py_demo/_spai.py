import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

def norm1(A):
    col_sums = []
    for j in range(A.shape[1]):
        s = 0.0
        for i in range(A.shape[0]):
            s += abs(A[i, j])
        col_sums.append(s)
    return max(col_sums)

def norm1_AAT(A):
    # rows of A
    row_norms = []
    for j in range(A.shape[0]):
        col_sum = 0.0
        aj = A[j, :]
        for i in range(A.shape[0]):
            col_sum += abs(np.dot(A[i, :], aj))
        row_norms.append(col_sum)
    return max(row_norms)

def initial_guess(K):
    """initial guess for M preconditioners in SPAI"""

    # not good with zero, and don't just want regular identity usually
    # in https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub paper
    # they suggest using identity scaled by 1/||A A^T||_1 (1-norm)
    # one_norm = norm1_AAT(K)
    # scale = 1.0 / one_norm**0.5 #**0.5
    scale = 1.0 / norm1(K)
    print(f"{scale=}")
    return scale * np.eye(K.shape[0])

def compute_spai_precond(K_csr, iters:int=10):
    """sparse approximate inverse preconditioner solve"""
    # based on https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf

    # compute dense numpy matrix for now.. sparse versions later..
    K = K_csr.toarray()
    # M = np.zeros_like(K)
    M = initial_guess(K)
    N = K.shape[0]

    # gradient descent iterations
    for iter in range(iters):
        # compute residual
        I = np.eye(N)
        R = I - K @ M
        G = R # is also the search direction

        # objective
        obj = np.sum(R * R)
        print(f"SPAI {iter=} {obj=:.5e}")

        # compute line search coeff from eqn 2.2
        AG = K @ G
        top = np.sum(R * AG) # elem-wise prod then sum, like inner product
        bot = np.sum(AG * AG)
        alpha = top / bot

        # update M matrix
        M += alpha * G

    return M

class SPAI_Precond:
    def __init__(self, K_csr, iters:int=10):
        self.K = K_csr.copy()
        self.M = compute_spai_precond(K_csr, iters)

    def solve(self, rhs):
        return np.dot(self.M, rhs)

def compute_spai_dense_mr_precond(K_csr, iters:int=10):
    # need to do sparse one next... using sub-matrices of A (precomputed sparsity maybe..)
    # to reduce compute
    # based on https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf

    # compute dense numpy matrix for now.. sparse versions later..
    K = K_csr.toarray()
    N = K.shape[0]
    # M = np.eye(N)
    M = initial_guess(K)

    for j in range(N): # each column of M
        ej = np.zeros(N)
        ej[j] = 1.0
        mj = np.dot(M, ej)
        # print(f"{mj.shape=} {ej.shape=}")
        for i in range(iters): # solving least-squares system
            # temp = np.dot(K, mj)
            # print(f"{temp.shape=}")
            rj = ej - np.dot(K, mj)
            A_rj = np.dot(K, rj)
            alpha_j = np.dot(rj, A_rj) / (np.dot(A_rj, A_rj) + 1e-12)
            mj += alpha_j * rj
        M[:,j] = mj * 1.0
    return M

class SPAI_MR_Precond:
    def __init__(self, K_csr, iters:int=10):
        self.K = K_csr.copy()
        self.M = compute_spai_dense_mr_precond(K_csr, iters)

    def solve(self, rhs):
        return np.dot(self.M, rhs)
    
def compute_spai_dense_mr_self_precond(K_csr, iters:int=10):
    # need to do sparse one next... using sub-matrices of A (precomputed sparsity maybe..)
    # based on https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf

    # compute dense numpy matrix for now.. sparse versions later..
    K = K_csr.toarray()
    N = K.shape[0]
    # M = np.eye(N)
    # M = 1e-7 * np.eye(N)
    M = initial_guess(K) # from https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub
    # they used 1/one-norm(AA^T) * I_N as initial matrix (but that has wrong units, so I changed it for better results)

    # import matplotlib.pyplot as plt
    # plt.spy(M)
    # plt.show()

    # print(f"{M[1,1]=}")

    for j in range(N): # each column of M
        ej = np.zeros(N)
        ej[j] = 1.0
        s = np.dot(M, ej)
        for i in range(iters): # solving least-squares system
            r = ej - np.dot(K, s)
            z = np.dot(M, r)
            q = np.dot(K, z)
            # if j == 1:
            #     print(f"{j=} : {s[:8]=}\n{r[:8]=}\n{z[:8]=}\n{q[:8]=}")
            alpha = np.dot(r, q) / (1e-12 + np.dot(q, q))
            s += alpha * z
        M[:,j] = s * 1.0

    # import matplotlib.pyplot as plt
    # plt.spy(M)
    # plt.show()

    return M

class SPAI_MR_SelfPrecond:
    def __init__(self, K_csr, iters:int=10):
        self.K = K_csr.copy()
        self.M = compute_spai_dense_mr_self_precond(K_csr, iters)

    def solve(self, rhs):
        return np.dot(self.M, rhs)