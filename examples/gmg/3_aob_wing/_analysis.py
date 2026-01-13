import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import time
import matplotlib.pyplot as plt
import niceplots

# setup GPU solver
solver = gpusolver.TacsGpuMultigridSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    load_mag=5000.0,
    mesh_level=3,
    omega=1.5,
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

# SLENDERNESS plot
# ------------------
# MC-BGS smoother without junction specific coloring
# junct_times = [1.5351057052612305, 1.520226001739502, 1.8862605094909668, 4.46341872215271, 17.451205015182495, 26.869178295135498, 27.269441604614258] # V(6,6)
# nojunct_times = [2.3576276302337646, 2.209479808807373, 2.83876895904541, 7.877009391784668, 7.864048719406128, 7.798820734024048, 7.832904815673828] # V(2,2)
# nojunct_times =[1.6975476741790771, 1.6935889720916748, 2.0301249027252197, 4.94269323348999, 18.528635263442993, 25.38693332672119, 25.420616388320923] # V(6,6)
# cylinder_times = [0.4270033836364746, 0.39861512184143066, 0.48015880584716797, 
#                   1.0606026649475098, 3.857357978820801, 5.291945934295654, 5.377862215042114]
# plt.style.use(niceplots.get_style())
# plt.plot(SR_vec, times, 'o-', label='aob-MCjunct')
# plt.plot(SR_vec, nojunct_times, 'o-', label='aob-MCnojunct')
# plt.plot(SR_vec, cylinder_times, 'o-', label='cylinder')
# plt.plot(SR_vec, [4.7286]*len(times), '--', label='aob-direct')
# plt.legend()
# plt.margins(x=0.05, y=0.05)
# plt.xlabel("Slenderness")
# plt.xscale('log')
# plt.ylabel("Solve Time (sec)")
# plt.yscale('log')
# plt.savefig("out/aob_slender_times.png", dpi=400)

old_junct_times = [1.5351057052612305, 1.520226001739502, 1.8862605094909668, 4.46341872215271, 17.451205015182495, 26.869178295135498, 27.269441604614258] # V(6,6)
cylinder_times = [0.4270033836364746, 0.39861512184143066, 0.48015880584716797, 
                  1.0606026649475098, 3.857357978820801, 5.291945934295654, 5.377862215042114]
plt.style.use(niceplots.get_style())
plt.plot(SR_vec, times, 'o-', label='aob-mcperm-fix')
plt.plot(SR_vec, old_junct_times, 'o-', label='aob-mcperm-old')
plt.plot(SR_vec, cylinder_times, 'o-', label='cylinder')
plt.plot(SR_vec, [4.7286]*len(times), '--', label='aob-direct')
plt.legend()
plt.margins(x=0.05, y=0.05)
plt.xlabel("Slenderness")
plt.xscale('log')
plt.ylabel("Solve Time (sec)")
plt.yscale('log')
plt.savefig("out/aob_color_fix.png", dpi=400)

# num smooth steps plot (MC-junct smoother only, wing geom only)
# # --------------------------------------------------------------
# two_cycle_times = [1.66646409034729, 1.8600232601165771, 2.676783561706543, 8.230266571044922, 8.071585178375244, 8.04187273979187, 8.062998056411743]
# plt.close('all')
# plt.plot(SR_vec, times, 'o-', label='V(4,4)')
# plt.plot(SR_vec, two_cycle_times, 'o-', label='V(2,2)')
# plt.plot(SR_vec, [4.7286]*len(times), '--', label='aob-direct')
# plt.legend()
# plt.margins(x=0.05, y=0.05)
# plt.xlabel("Slenderness")
# plt.xscale('log')
# plt.ylabel("Solve Time (sec)")
# plt.yscale('log')
# plt.savefig("out/aob_cycle_times.png", dpi=400)