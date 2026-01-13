import sys
sys.path.append("_src/")
import matplotlib.pyplot as plt

from sa import bgs_bsr_smoother

import numpy as np
from scipy.sparse import bsr_matrix, block_diag

np.random.seed(0)

ndof = 6
# nblocks = 4
nblocks = 3

# Random block lower-triangular A (with diagonal)
blocks = []
for i in range(nblocks):
    row_blocks = []
    for j in range(nblocks):
        if j > i:
            row_blocks.append(np.zeros((ndof, ndof)))
        else:
            row_blocks.append(np.random.randn(ndof, ndof))
    blocks.append(row_blocks)

A_dense = np.block([[blocks[i][j] for j in range(nblocks)] for i in range(nblocks)])
A_bsr = bsr_matrix(A_dense, blocksize=(ndof, ndof))

# Random P
P_dense = np.random.randn(nblocks * ndof, ndof)
P_bsr = bsr_matrix(P_dense, blocksize=(ndof, ndof))

# Run smoother
Pnew_bsr = bgs_bsr_smoother(A_bsr, P_bsr, num_iter=1)

# check residual of the smoother equation 
LplusD = np.tril(A_dense)
target = P_bsr.toarray()
pred = LplusD @ Pnew_bsr.toarray()
res = pred - target
print("‖Res‖_F =", np.linalg.norm(res))

# ok, then just try the smoother with dense form instead?
Pnew_dense = np.linalg.solve(LplusD, P_dense)
pred2 = LplusD @ Pnew_dense
res2 = pred2 - P_dense
print("‖Res2‖_F =", np.linalg.norm(res2))

# check error of the np.solve smoother

# Dense reference
# LplusD = np.tril(A_dense)
# ref = np.linalg.solve(LplusD, P_dense)
# print("‖Error‖_F =", np.linalg.norm(ref - Pnew_bsr.toarray()))

# check the matrices with plots
# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].spy(A_bsr)
# ax[1].spy(P_bsr)
# plt.show()

# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].imshow(target)
# ax[1].imshow(pred)
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(P_dense)
ax[1].imshow(Pnew_dense)
plt.show()