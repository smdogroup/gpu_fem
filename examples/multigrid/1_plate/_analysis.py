import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse

# ndvs_per_side = 4
# ndvs_per_side = 16
ndvs_per_side = 32
# ndvs_per_side = 64
# ndvs_per_side = 128

# setup GPU solver
solver = gpusolver.TacsGpuMultigridSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    load_mag=3000.0,
    # omega=1.2,
    # omega=1.1,
    # omega=1.0,
    # omega=0.9,
    omega=0.85,
    # omega=0.8,
    # omega=0.7,
    # omega=0.6,
    # omega=0.4,
    # omega=0.2,
    nxe=512,
    # nxe=256,
    # nxe=128,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
    # SR=50.0, # slenderness
)

# init dvs
ndvs = solver.get_num_dvs()
# x0 = np.array([3e-2]*ndvs)
x0 = np.array([4e-3]*ndvs) # very slender
# x0 = np.array([2e-3] * ndvs)
# x0 = np.array([1e-3]*ndvs) # very slender

# debug writing DVs to not same values..
solver.set_design_variables(x0)
solver.solve()
ksfail = solver.evalFunction("ksfailure")
print(f"{ksfail=:.2e}")
solver.writeSolution("out/plate_mg.vtk")
