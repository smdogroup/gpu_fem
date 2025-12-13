import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np

rhoKS = 100.0
SF = 1.5
# SF = 1.0
load_mag = 100.0
solver = gpusolver.TACSGPUSolver(rhoKS, SF, load_mag)
ndvs = solver.get_num_dvs()

# test DVs assigned correctly for applying adjacency constraints
# xlist = [2e-2]*48 # LE spars
# xlist += [1.8e-2] * 45 # TE spars
# xlist += [1.6e-2] * 47 # ribs
# xlist += [1.4e-2, 1.2e-2] * 48 # Uskin,Lskin alternating
# xlist += [1.1e-2] # engine mount
# xlist += [1.05e-2] # root disp
# xlist += [1.0e-2] * 4 # side of body
# print(f"{len(xlist)=}")
# x = np.array(xlist)

x = np.array([5e-2]*ndvs)


solver.set_design_variables(x)
solver.solve()
mass = solver.evalFunction("mass")
ksfail = solver.evalFunction("ksfailure")
print(f"{mass=:.4e} {ksfail=:.4e}")

mass_grad = solver.evalFunctionSens("mass")
print(f"{mass_grad[:5]=}")
ksfail_grad = solver.evalFunctionSens("ksfailure")
print(f"{ksfail_grad[:5]=}")

solver.writeSolution("out/uCRM.vtk")