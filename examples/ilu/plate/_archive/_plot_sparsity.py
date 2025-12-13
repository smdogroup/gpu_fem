import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rowPtr = np.array([0,4,10,15,23,32,40,46,52,57])
colPtr = np.array([0,1,3,4,0,1,2,3,4,5,1,2,3,4,5,
                   0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,
                   1,2,3,4,5,6,7,8,3,4,5,6,7,8,3,4,5,
                   6,7,8,4,5,6,7,8])
values = np.ones((57,))

block_csr_mat = sp.sparse.csr_matrix((values, colPtr, rowPtr), shape=(9,9))
block_dense_mat = block_csr_mat.toarray()
sns.heatmap(block_dense_mat, annot=True)
plt.show()