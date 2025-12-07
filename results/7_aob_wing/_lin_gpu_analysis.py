import sys
sys.path.append("_src/") # contains gpusolver

import wingmultigrid
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse

# setup GPU solver
solver = wingmultigrid.LinearWingSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    force=684e3, # 30 KPa on lower skin from the structural benchmark
    # force=684e3*3, # boost load from static benchmark (to get more deflection?)
    omega=0.85,
    nsmooth=1,
    ORDER=8,
    # ORDER=4,
    rtol=1e-6, # almost want to have high rtol like this only later in the optimization tbh..
    # rtol=1e-5,
    # rtol=1e-4, # faster optimization, but will it converge though?
    print=True, # always on for these kinds of tests..
)

def load_design(filename):
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip first line (mass/ksfail)
    return np.array([float(line.split()[1]) for line in lines])

# linear optimal
filename = "out/1_lin_unstiff_gpu_opt.txt"
if os.path.exists(filename):
    lin_opt_design = load_design(filename)
    x0 = lin_opt_design.copy()

else:
    # init dvs
    ndvs = solver.get_num_dvs()
    # x0 = np.array([3e-2]*ndvs)
    x0 = np.array([2e-2]*ndvs) # very slender
    # x0 = np.array([1e-2]*ndvs)
    # x0 = np.array([5e-3]*ndvs)
    
# debug writing DVs to not same values..
solver.set_design_variables(x0)
solver.solve()
ksfail = solver.evalFunction("ksfailure")
print(f"{ksfail=:.2e}")
solver.writeSolution("out/wing_mg.vtk")

# also write out exploded view
solver.writeExplodedVTKs("out/wing_intstruct.vtk", "out/wing_uskin.vtk", "out/wing_lskin.vtk")

# DEBUG (when loads weren't right)
# solver.writeLoadsToVTK("out/wing_loads.vtk")