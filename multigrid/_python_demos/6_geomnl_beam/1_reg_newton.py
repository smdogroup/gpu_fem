# now let's test this out and visualize it
import numpy as np
from src import EBNonlinearAssembler

# SR = 0.1
# SR = 1.0
# SR = 10.0
SR = 100.0
# SR = 1000.0
# SR = 1e4

# 1) Euler-Bernoulli Beam
# -------------------------
E = 70e9; b = 4e-3; L = 1.0; rho = 1; ys = 4e5

# scaling inputs
# nxe = num_elements = int(3e2) #100, 300, 1e3
# nxe = num_elements = int(1e3)
# nxe = num_elements = int(3e2)
# nxe = num_elements = int(100)
# nxe = num_elements = int(64)
# nxe = num_elements = int(30)

# nxe = num_elements = int(100)
nxe = num_elements = int(100)

thick = L / SR

# MAG_0 = 2e3
# MAG_0 = 2e2
MAG_0 = 4e2

MAG_0 *= 0.8 # to match TACS somehow?

PRESSURE = MAG_0 * (thick / 0.01)**3
TOTAL_FORCE = PRESSURE * L
nodal_load = TOTAL_FORCE / (nxe - 1)

# because nodal load integral, divide by dx?
nodal_load /= (1.0 / nxe)

# scale by slenderness
print(f"{nodal_load=:.2e}")

# num DVs
nxh = nxe
hvec = np.array([thick] * nxh)

eb_nl_beam = EBNonlinearAssembler(nxe, nxh, E, b, L, rho, nodal_load, ys, 50.0, dense=False, load_fcn=lambda x : 1.0)

SOLVE_NL = True
# SOLVE_NL = False

# nonlinear solve
if not(SOLVE_NL):
    eb_nl_beam.solve_local(hvec, lam=1.0)

if SOLVE_NL:
    # N = 3
    N = 10
    # N = 30

    lam_max = 1.0
    # lam_max = 4.0
    # # lam_max = 10.0
    # # lam_max = 0.1
    eb_nl_beam.solve_load_newton(hvec, lams=[lam_max * (1.0 / N)*i for i in range(1,N+1)])


w = eb_nl_beam.u[1::3]
w_max = np.max(np.abs(w))
print(f"{w_max=:.4e}")

eb_nl_beam.plot_disp(idof=0) # for u
# eb_nl_beam.plot_disp(idof=1) # for w

