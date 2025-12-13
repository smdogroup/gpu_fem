import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse

# ndvs_per_side = 4
# ndvs_per_side = 8
# ndvs_per_side = 16
ndvs_per_side = 32
# ndvs_per_side = 64
# TODO : high number of DVs currently forces the coarsest grid to be that problem size
# can somewhat slow down the multigrid solve.. would like to fix that by allowing some DV interp to coarser grids

# setup GPU solver
solver = gpusolver.TacsGpuMultigridSolver(
    rhoKS=100.0,
    safety_factor=1.5,
    load_mag=30000.0,
    # nxe=512,
    omega=1.5, # over-relaxation (faster solves usually)
    nxe=128,
    nx_comp=ndvs_per_side, # num dvs in x-direction
    ny_comp=ndvs_per_side, # num dvs/comps in y-direction
    # SR=50.0, # slenderness # doesn't matter cause x0 sets that..
)

# init dvs
ndvs = solver.get_num_dvs()
x0 = np.array([5e-2]*ndvs)

# # debug writing DVs to not same values..
solver.set_design_variables(x0)
solver.solve()
solver.writeSolution("out/cylinder_init.vtk")
# ksfail = solver.evalFunction("ksfailure")
# print(f"{ksfail=:.2e}")
# exit()


def get_functions(xdict):
    # update design
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])
    solver.set_design_variables(xarr)

    # solve and get functions
    solver.solve()
    mass = solver.evalFunction("mass")
    ksfail = solver.evalFunction("ksfailure")
    funcs = {
        'mass' : mass,
        'ksfailure' : ksfail,
    }

    # print(f"{funcs=}")

    comp_names = [f"comp{icomp}" for icomp in range(ndvs)]
    with open("out/design_variables.txt", "w") as f:
        f.write(f"{mass=:.4e} {ksfail=:.4e}\n")
        for name, value in zip(comp_names, xarr):
            f.write(f"{name}\t{value:.16e}\n")

    return funcs, False

def get_function_grad(xdict, funcs):
    # update design, this shouldn't really come with a new forward solve, just adj solve
    # but if optimizer calls grads twice in a row or something, this won't work right (needs companion get_functions call every time)
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])
    solver.set_design_variables(xarr)

    # before not including this would result in wrong state vars
    # now it only does the solve again if design changed with this call
    solver.solve() # temp debug just solve again here

    mass_grad = solver.evalFunctionSens("mass")
    ksfail_grad = solver.evalFunctionSens("ksfailure")
    sens = {
        'mass' : {'vars' : mass_grad},
        'ksfailure' : {'vars' : ksfail_grad},
    }

    # print(f"{sens=}")
    return sens, False

opt_problem = Optimization("tuning-fork", get_functions)
opt_problem.addVarGroup(
    "vars",
    ndvs,
    lower=np.array([2e-2]*ndvs), # min thickness very important for max slenderness
    upper=np.array([1e2]*ndvs),
    value=x0,
    scale=np.array([1e2]*ndvs),
)
opt_problem.addObj('mass', scale=1e-1)
opt_problem.addCon("ksfailure", scale=1.0, upper=1.0)

# adjacency constraints (linear so reduces size of opt problem)
# ---------------------
def adj_vec(i, j):
    vec = np.zeros((1,ndvs), dtype=np.double)
    vec[0,i] = 1.0
    vec[0,j] = -1.0
    return vec

# add adjacency constraints
# for icomp in range(25):
#     # TODO
# opt_problem.addCon(
#         f"SOB-{iconstr}",
#         lower=-1.5e-3,
#         upper=1.5e-3,
#         scale=1e2,
#         linear=True,
#         wrt=['vars'],
#         jac={'vars': adj_vec(icomp, icomp+1)}
#     )

# verify_level = -1
verify_level = 0
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-5,
        "Major optimality tolerance": 1e-3,
        "Verify level": verify_level, #-1,
        "Major iterations limit": int(1e4), #1000, # 1000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        "Major step limit": 5e-2, # these causes l to turn on a lot
        "Nonderivative linesearch": True,
        "Linesearch tolerance": 0.9,
        # "Difference interval": 1e-6,
        # "Function precision": 1e-10, #results in btw 1e-4, 1e-6 step sizes
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0, #1.0E2
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("out/SNOPT_print.out"),
        "Summary file": os.path.join("out/SNOPT_summary.out"),
    }
)

hot_start = False
sol = snoptimizer(
    opt_problem,
    sens=get_function_grad,
    storeHistory="out/myhist.hst",
    hotStart="out/myhist.hst" if hot_start else None,
)

print(f"{sol.xStar=}")

solver.solve()
solver.writeSolution("out/cylinder_opt.vtk")
solver.free()