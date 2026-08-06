import numpy as np
import matplotlib.pyplot as plt

def poisson_2d_csr_pattern(nx):
    N = nx * nx
    A = np.zeros((N, N), dtype=int)

    for row in range(N):
        ix = row % nx
        iy = row // nx

        if iy > 0:
            A[row, row - nx] = 1
        if ix > 0:
            A[row, row - 1] = 1

        A[row, row] = 1

        if ix < nx - 1:
            A[row, row + 1] = 1
        if iy < nx - 1:
            A[row, row + nx] = 1

    return A

def row_partition(N, ngpu):
    return [(N * g // ngpu, N * (g + 1) // ngpu) for g in range(ngpu)]

nx = 8
ngpu = 4

A = poisson_2d_csr_pattern(nx)
N = A.shape[0]
parts = row_partition(N, ngpu)

colored = np.zeros_like(A, dtype=float)
for g, (rs, re) in enumerate(parts):
    colored[rs:re, :] = (g + 1) * A[rs:re, :]

plt.figure(figsize=(7, 7))
plt.imshow(colored, interpolation="none", aspect="auto")
plt.title(f"Distributed row ownership: {ngpu} GPUs, global matrix {N}x{N}")
plt.xlabel("Global column index")
plt.ylabel("Global row index")
plt.colorbar(label="GPU owner of row block")
plt.tight_layout()
plt.show()

for g, (rs, re) in enumerate(parts):
    print(f"GPU {g}: owns global rows [{rs}, {re}), local matrix shape = ({re-rs}, {N})")