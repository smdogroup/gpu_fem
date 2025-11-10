import sys
sys.path.append("_src/") # contains nlplategpu
import nlplategpu # the so file created by pybind11

# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys
sys.path.append("_src/") # contains nlplategpu
import nlplategpu # the so file created by pybind11

# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse



# setup GPU solver for nl plate
solver = nlplategpu.NonlinearPlateGPUSolver(
    pressure=8e6,
    # pressure=8e5,
    # omega=1.5,
    # omega=1.0, # omega = 1 actually more robust for the nonlinear case
    omega=0.75,
    # nxe=128,
    nxe=256,
    SR=100.0,
    # use_predictor=True, # sometimes works on plate, sometimes not
    use_predictor=False,
    # kmg_print=True,
    kmg_print=False,
    # nl_debug=True, # if debug is on, it will exit on a failure and return the failed state
    nl_debug=False,
    debug_gmg=True,
)

nvars = solver.get_num_vars()
u0 = np.zeros(nvars)


# coarse-fine check for linear problem
# =======================================

lam = 1.0
# technically don't need set_fine_LU=True since zero disps, but want it here as reminder
solver.setGridDefect(u0, lam, set_fine_LU=True) 

i_defect, ism_defect, cf_soln = np.zeros_like(u0), np.zeros_like(u0), np.zeros_like(u0)
ch_defect, fsm_defect, lu_soln = np.zeros_like(u0), np.zeros_like(u0), np.zeros_like(u0)

solver.getCoarseFineStep(i_defect, ism_defect, cf_soln, ch_defect, fsm_defect, lu_soln)

print("cf1")
solver.writeSolution("out/_plate_idefect_0.vtk", i_defect)
solver.writeSolution("out/_plate_ismdefect_0.vtk", ism_defect)
solver.writeSolution("out/_plate_cfsoln_0.vtk", -cf_soln)
solver.writeSolution("out/_plate_lusoln_0.vtk", -lu_soln)
solver.writeSolution("out/_plate_chdefect_0.vtk", -ch_defect)
solver.writeSolution("out/_plate_fdefect_0.vtk", -fsm_defect)

# run NONLINEAR solve, then NL cf
# =================================

start_solve = time.time()
lam0, lamf, inner_atol = 0.2, 1.0, 1e-6
u_nl = solver.continuationSolve(u0, lam0, lamf, inner_atol)
solve_dt = time.time() - start_solve
print(f"continuation solve in {solve_dt=:.4e} sec")

# running with extra loads past final NL state
lam = 1.2
solver.setGridDefect(u_nl, lam, set_fine_LU=True)
solver.getCoarseFineStep(i_defect, ism_defect, cf_soln, ch_defect, fsm_defect, lu_soln)

solver.writeSolution("out/_plate_idefect_1.vtk", i_defect)
solver.writeSolution("out/_plate_ismdefect_1.vtk", ism_defect)
solver.writeSolution("out/_plate_cfsoln_1.vtk", -cf_soln)
solver.writeSolution("out/_plate_lusoln_1.vtk", -lu_soln)
solver.writeSolution("out/_plate_chdefect_1.vtk", -ch_defect)
solver.writeSolution("out/_plate_fdefect_1.vtk", -fsm_defect)


# then try a kcycle solve ?
# ==========================



# FREE memory on GPU
# ===================

solver.free()