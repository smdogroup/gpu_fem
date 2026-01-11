import numpy as np
import matplotlib.pyplot as plt
from _pde_src import *
from _multigrid_src import *

"""setup the poisson linear system"""
# nxe = 4
nxe = 8
# nxe = 32 # choose power of 2 please
# nxe = 64
# nxe = 128
A, b = get_poisson_lhs_and_rhs(nxe=nxe)
N = A.shape[0]

plt.imshow(A)
plt.show()

"""compute the solution and compare against method of manufactured disps
 with np dense solve just to check we coded PDE discretization right """
x_discrete = np.linalg.solve(A, b)
plot_poisson_soln(nxe, _soln=x_discrete)

# try solving with damped jacobi here..
x0 = np.zeros(N)
x = x0.copy()
Dinv = 1.0 / np.diag(A)
omega = 2.0 / 3.0
for i in range(100):
    r = b - A @ x
    dx = omega * Dinv * r
    x += dx
    r_norm = np.linalg.norm(r)
    print(f"{i=} : {r_norm=:.3e}")

# what if bcs were in it?
