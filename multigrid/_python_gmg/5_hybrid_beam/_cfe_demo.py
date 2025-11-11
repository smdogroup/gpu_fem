from src._hyb_cfe_elem import *
import matplotlib.pyplot as plt
import numpy as np
import niceplots
from src import ChebyshevTSAssembler

# order = 1
# order = 2
order = 3
# order = 4

gps = get_chebyshev_gps(order)
xi = np.linspace(-1.0, 1.0, 100)

plt.style.use(niceplots.get_style())


Nks = []
for ibasis in range(order + 1):
    Nks += [
        np.array([chebyshev_value(ibasis, _xi, order) for _xi in xi])
    ]
    # Nks += [
    #     np.array([chebyshev_d2dx(ibasis, _xi, 1.0, order) for _xi in xi])
    # ]
    plt.plot(xi, Nks[-1], label=f"N{ibasis}")

plt.legend()
plt.xlabel("Xi")
plt.ylabel("CFE Basis")
plt.show()


# SR = 0.1
# SR = 1.0
# SR = 10.0
# SR = 100.0
SR = 1000.0
# SR = 1e4

# 1) Euler-Bernoulli Beam
# -------------------------

E = 2e7; b = 4e-3; L = 1.0; rho = 1
qmag = 2e-2
ys = 4e5

# scale by slenderness
thick = 1.0 / SR
qmag *= thick**3

# scaling inputs
# if rho_KS was too high like 500, then the constraint was too nonlinear or close to max and optimization failed
rho_KS = 50.0 # rho = 50 for 100 elements, used 500 later
# nxe = 3
nxe = 10
hvec = np.array([thick] * nxe)

# 4) chebyshev beam
# ------------------

# order = 1
order = 2
cfe_beam1 = ChebyshevTSAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=False, order=order)
cfe_beam1._compute_mat_vec(np.array([thick]*nxe))
Kmat1 = cfe_beam1.Kmat.copy()
print(f"{Kmat1.shape=}")
cfe_beam1.solve_forward(hvec)

cfe_beam2 = ChebyshevTSAssembler(nxe, nxe, E, b, L, rho, qmag, ys, rho_KS, dense=True, order=order)
cfe_beam2._compute_mat_vec(np.array([thick]*nxe))
Kmat2 = cfe_beam2.Kmat.copy()

# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].spy(Kmat1)
# ax[0].set_title("Sparse")
# ax[1].spy(Kmat2)
# ax[1].set_title("Dense")
# plt.show()

# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].imshow(Kmat1.toarray())
# ax[0].set_title("Sparse")
# ax[1].imshow(Kmat2.toarray())
# ax[1].set_title("Dense")
# plt.show()

# diff = Kmat1.toarray() - Kmat2.toarray()
# plt.imshow(diff)
# plt.show()

cfe_beam1.plot_disp()
