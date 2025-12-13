import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Lmat = 0.2 + np.random.rand(4,4)
for i in range(4):
    for j in range(i+1,4):
        Lmat[i,j] = 0.0
# sns.heatmap(Lmat)
# plt.show()

# covariance matrix
Kmat = Lmat @ Lmat.T + 1e-2 * np.eye(4)
# sns.heatmap(Kmat)
# plt.show()
# b = np.random.rand(4,1)
# x = np.linalg.solve(Kmat, b)

# now enforce periodic BC (nodes 0 and 3 pythonic)
# strategy 1 - just replace one row with periodic constr
Kmat2 = Kmat.copy()
Kmat2[3,:] = np.array([1, 0, 0, 1])
# sns.heatmap(Kmat2)
# plt.show()
b2 = b.copy()
b2[0,0] += b2[3,0]
b2[3,0] = 0.0
x2 = np.linalg.solve(Kmat2, b2)

# strategy 2 - also add row 4 to row 1 and col 4 to col 1
Kmat3 = Kmat.copy()
Kmat3[0,:] -= Kmat3[3,:]
Kmat3[:,0] -= Kmat3[:,3]
Kmat3[:,3] = 0.0
Kmat3[3,:] = np.array([1, 0, 0, 1])
sns.heatmap(Kmat3)
plt.show()

x3 = np.linalg.solve(Kmat3, b2)
print(f"{x=}")
print(f"{x2=}")
print(f"{x3=}")