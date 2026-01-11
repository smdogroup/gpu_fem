# test gauss-jordan 6x6 block solves


import numpy as np
import sys, scipy as sp
from __ilu import gaussJordan
import matplotlib.pyplot as plt

max_err = 0.0

for test_ind in range(100):

    # 1) test gaussJordan matrix inverse
    A = np.random.rand(6,6)
    B = np.eye(6)

    tmp = A.copy()
    fail, perm = gaussJordan(tmp, B)
    iperm = np.arange(0, 6)
    for i in range(6):
        j = perm[i]
        iperm[j] = i
    print(F"{A=}\n{B=}")
    print(f"{perm=}")

    Ainv = np.linalg.inv(A)

    if test_ind == 0:
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].imshow(B)
        ax[1].imshow(Ainv)
        plt.show()

    rhs = np.random.rand(6)
    x1 = np.linalg.solve(A, rhs)
    x2 = np.dot(B, rhs)
    # x2 = x2[iperm]
    err = x1 - x2
    err_nrm = np.linalg.norm(err)
    print(f"{err_nrm=:.4e}")

    if err_nrm >= 1e-13:
        print(f"{err_nrm=}")
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].imshow(B)
        ax[1].imshow(Ainv)
        plt.show()

    max_err = np.max([max_err, err_nrm])

print(f"MAX_ERR = {max_err:.4e}")
