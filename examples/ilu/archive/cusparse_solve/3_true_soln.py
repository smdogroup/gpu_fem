import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# TODO : make 4x4 block_dim and 4x4 block matrix (so 16x16 total) problem

# nz from kenrel matrix then add diagonal to make pos def again but not with blocks
# nodal level sparsity pattern before fillin
S = np.array([
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

A = np.zeros((16,16))
for grow in range(16):
    block_row = grow // 4
    local_row = grow % 4
    for gcol in range(16):
        block_col = gcol // 4
        local_col = gcol % 4

        # linear kernel + 4 * diagonal
        if S[block_row,block_col]:
            A[grow,gcol] = (grow+1) * (gcol+1) + 64.0 * (grow == gcol)

plt.imshow(A)
# plt.show()

rhs = np.array([[_ for _ in range(16)]]).T

x = np.linalg.solve(A, rhs)
print(f"{x[:,0]=}")

L = sp.linalg.cholesky(A).T
# print(f"{L=}")
plt.figure()
plt.imshow(L)
plt.show()