import numpy as np
import scipy as sp

# nz from kenrel matrix then add diagonal to make pos def again
# A = np.array([
#     [1, 2, 0, 4],
#     [2, 4, 6, 0],
#     [0, 6, 9, 0],
#     [4, 0, 0, 16],
# ]) + np.eye(4) * 2

A = np.array([
    [3, 2, 0, 4],
    [2, 6, 6, 0],
    [0, 6, 11, 0],
    [4, 0, 0, 18],
])

b = np.array([[-1, -2, 3, 4]]).T
# print(f"{b.shape=}")

x = np.linalg.solve(A, b)
print(f"{x=}")

L = sp.linalg.cholesky(A).T
print(f"{L=}")