import sys
sys.path.append("_src/") # contains gpusolver

import wingmultigrid
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os, time
import argparse

# ARGPARSE
parser = argparse.ArgumentParser(description="Choose the wing element type for the solver.")
parser.add_argument(
    "--element",
    type=str,
    choices=["CFI4", "MITC4"],
    default="MITC4",
    help="Finite element type to use (default: MITC4)."
)
args = parser.parse_args()
element = args.element


SOLVER_CLASS = None
if element == "CFI4":
    SOLVER_CLASS = wingmultigrid.LinearCFIWingSolver
elif element == "MITC4":
    SOLVER_CLASS = wingmultigrid.LinearMITCWingSolver

# setup GPU solver
solver = SOLVER_CLASS(
    rhoKS=100.0,
    safety_factor=1.5,
    force=684e3, # 30 KPa on lower skin from the structural benchmark
    # force=684e3*3, # boost load from static benchmark (to get more deflection?)
    # omega=0.85,
    omega=0.95,
    nsmooth=1,
    ninnercyc=1,
    # ninnercyc=2,
    # nsmooth=8,
    # ORDER=64,
    ORDER=32 if element == "MITC4" else 8, # AOB prefers higher order smoother (but not too high)
    # ORDER=16,
    # ORDER=10,
    # ORDER=8,
    # ORDER=6,
    # ORDER=4,
    # ORDER=4,
    # ORDER=12,
    # ORDER=24,
    # ORDER=4,
    # n_krylov=50,
    # n_krylov=200,
    n_krylov=400,
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
    x0 = np.array([3e-2]*ndvs) # very slender
    # x0 = np.array([1e-2]*ndvs)
    # x0 = np.array([5e-3]*ndvs)
    
# debug writing DVs to not same values..
solver.set_design_variables(x0)

start_time = time.time()
solver.solve()
dt = time.time() - start_time
print(f"Linear solve in {dt=:.2e} sec")

ksfail = solver.evalFunction("ksfailure")
print(f"{ksfail=:.2e}")
solver.writeSolution(f"out/wing_mg_{element}.vtk")

# also write out exploded view
solver.writeExplodedVTKs("out/wing_intstruct.vtk", "out/wing_uskin.vtk", "out/wing_lskin.vtk")

# DEBUG (when loads weren't right)
# solver.writeLoadsToVTK("out/wing_loads.vtk")