import sys
sys.path.append("_src/") # contains gpusolver


import cylindermultigrid
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse

def load_design(filename):
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip first line (mass/ksfail)
    return np.array([float(line.split()[1]) for line in lines])

# load linear optimal design..
lin_opt_design = load_design("out/lin_gpu_opt.txt")

# ndvs_per_side = 4
# ndvs_per_side = 16
ndvs_per_side = 32
# ndvs_per_side = 64
# ndvs_per_side = 128

# setup GPU solver
solver = cylindermultigrid.NonlinearCylinderSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    pressure=3000.0,
    # omega=0.85,
    omega=0.3,
    # nxe=512,
    nxe=128,
    nsmooth=2,
    # nxe=256,
    # nxe=128,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
    # SR=50.0, # slenderness
)

# init dvs
ndvs = solver.get_num_dvs()
# x0 = np.array([3e-2]*ndvs)
# x0 = np.array([4e-3]*ndvs) # very slender
# x0 = np.array([2e-3] * ndvs)
# x0 = np.array([1e-3]*ndvs) # very slender

x0 = lin_opt_design.copy()

# debug writing DVs to not same values..
solver.set_design_variables(x0)
solver.solve()
ksfail = solver.evalFunction("ksfailure")
print(f"{ksfail=:.2e}")
solver.writeSolution("out/plate_mg_nl.vtk")
