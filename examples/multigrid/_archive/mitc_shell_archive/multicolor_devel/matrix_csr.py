# show the nodal matrix sparsity for the 7 node, 14 DOF (2x2 blocks) BsrMat for multicolor testing

import matplotlib.pyplot as plt
import numpy as np

diag = np.concatenate([np.ones(3) * 3, np.ones(2) * 2, np.ones(2)])
Nodal = np.diag(diag)

# colors are 0 - blue, 1 - red, 2 - green
# with color 0 or blue the first three nodes n0, n1, n2
# color 1 or red is nodes n3, n4
# color 2 or green is nodes n5, n6

odiag = 0.4
c1 = 3.0 - odiag
c2 = 2.0 - odiag
c3 = 1.0 - odiag

# now define intercolor coupling blocks (multicolor matrices only have non-diag block matrices in color i, color j coupling subblocks with i neq j)
# color 0 to 1
Nodal[0,3], Nodal[0, 4] = c1, c1
Nodal[3, 0], Nodal[4, 0] = c2, c2

# color 0 to 2
Nodal[0,5] = c1
Nodal[5, 0] = c3

# color 1 to 2
Nodal[3, 6], Nodal[4, 5] = c2, c2
Nodal[6, 3], Nodal[5, 4] = c3, c3

plt.imshow(Nodal)
plt.show()

# for BSR version for now, each 2x2 block is going to be csr_scale * [2,1],[1,2] should be invertible

A = np.zeros((14, 14))

for ib in range(7):
    for jb in range(7):
        node = Nodal[ib, jb]

        for k in range(4):
            k1, k2 = k % 2, k // 2
            val = 2.0 if k1 == k2 else 1.0
            i, j = 2 * ib + k1, 2 * jb + k2

            A[i, j] = node * val

plt.imshow(A)
plt.show()