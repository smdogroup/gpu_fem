import sys
sys.path.append("_src/") # contains gpusolver


import platemultigrid
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
# lin_opt_design = load_design("out/nl_gpu_opt.txt")

# ndvs_per_side = 4
# ndvs_per_side = 16
ndvs_per_side = 32
# ndvs_per_side = 64
# ndvs_per_side = 128

# setup GPU solver
solver = platemultigrid.NonlinearPlateSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    # load_mag=5e5,
    # load_mag=1e6,
    # load_mag=2e6,
    # load_mag=4e6, # original loads from linear case.., gonna lead to NL deflection?
    # load_mag=8e6,
    load_mag=1.6e7,
    # load_mag=2e7,
    # load_mag=8e7,
    omega=0.85,
    # omega=0.3, # much faster than 0.85 usually..
    # nxe=512,
    # nxe=256,
    # nxe=128,
    # in_plane_frac=0.05,
    # in_plane_frac=0.1,
    in_plane_frac=0.6,
    # in_plane_frac=0.15,
    # in_plane_frac=0.25,
    # in_plane_frac=0.35,
    # in_plane_frac=0.75,
    Lx=1.0,
    # Lx=2.0, # both dimensions
    # Lx=4.0,
    # Lx=8.0,
    # Lx=16.0,
    nxe=64,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
    # SR=50.0, # slenderness
    ORDER=8,
    # rtol=1e-5,
    # rtol=1e-4, # faster optimization, but will it converge though?
    # print=True, # always on for these kinds of tests..
    print=False,
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
