import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import time
import matplotlib.pyplot as plt
import niceplots

ndvs_per_side = 4
# ndvs_per_side = 16
# ndvs_per_side = 32
# ndvs_per_side = 64
# ndvs_per_side = 128

# setup GPU solver
solver = gpusolver.TacsGpuMultigridSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    load_mag=3000.0,
    omega=1.5,
    # nxe=128,
    nxe=320,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
)
ndvs = solver.get_num_dvs()

# single analysis
# x0 = np.array([1e-2] * ndvs)
# solver.set_design_variables(x0)
# solver.solve()
# solver.write_solution("out/cylinder_analysis.vtk")

# plot effect of slenderness in cylinder solves
SR_vec = [1.0, 10.0, 25.0, 50.0, 100.0, 300.0, 500.0]
times = []

for SR in SR_vec:
    # print(f"{SR=}")
    x0 = np.array([1.0 / SR] * ndvs)

    # debug writing DVs to not same values..
    solver.set_design_variables(x0)
    start_time = time.time()
    solver.solve()
    end_time = time.time()

    solve_time = end_time - start_time
    times += [solve_time]
    print(f'{SR=:.2f} {solve_time=:.2e}')

print(f"{times=}")

plt.style.use(niceplots.get_style())
plt.plot(SR_vec, times, 'o-', label='GMG')
plt.plot(SR_vec, [6.25] * len(times), '--', label='direct')
plt.margins(x=0.05, y=0.05)
plt.legend()
plt.xlabel("Slenderness")
plt.xscale('log')
plt.ylabel("Solve Time (sec)")
plt.savefig("out/cylinder_slenderness_times.png", dpi=400)
