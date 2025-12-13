import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np

solver = gpusolver.TACSGPUSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    load_mag=30.0,
    nxe=100,
    nx_comp=5,
    ny_comp=5,
)

# set initial design
ndvs = solver.get_num_dvs()
x = np.array([1e-2]*ndvs)

# solve
solver.set_design_variables(x)
solver.solve()
mass = solver.evalFunction("mass")
ksfail = solver.evalFunction("ksfailure")
print(f"{mass=:.4e} {ksfail=:.4e}")

mass_grad = solver.evalFunctionSens("mass")
print(f"{mass_grad[:5]=}")
ksfail_grad = solver.evalFunctionSens("ksfailure")
print(f"{ksfail_grad[:5]=}")

solver.writeSolution("out/plate.vtk")