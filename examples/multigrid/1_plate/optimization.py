import sys
sys.path.append("_src/") # contains gpusolver

import gpusolver
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse
from mpi4py import MPI

# ndvs_per_side = 4
# ndvs_per_side = 16
# ndvs_per_side = 32
# ndvs_per_side = 64
ndvs_per_side = 128

# only adding MPI right now to help pyoptsparse.. it runs better with more PROCS?
comm = MPI.COMM_WORLD

# setup GPU solver
solver = None

root = 0
if comm.rank == root:
    solver = gpusolver.TacsGpuMultigridSolver(
        rhoKS=100.0,
        safety_factor=1.5,
        load_mag=3000.0,
        omega=0.85,
        # nxe=512,
        nxe=256,
        # nxe=128,
        nx_comp=ndvs_per_side, # num dvs in x-direction
        ny_comp=ndvs_per_side, # num dvs/comps in y-direction
        SR=50.0, # slenderness
    )

# init dvs
ndvs = None
if comm.rank == root:
    ndvs = solver.get_num_dvs()
ndvs = comm.bcast(ndvs, root=root)
x0 = np.array([3e-2]*ndvs)

# debug writing DVs to not same values..
# solver.set_design_variables(x0)
# solver.solve()
# ksfail = solver.evalFunction("ksfailure")
# print(f"{ksfail=:.2e}")
# solver.writeSolution("out/plate_init.vtk")
# exit()


def get_functions(xdict):
    # update design
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])

    mass, ksfail = None, None
    if comm.rank == root:
        solver.set_design_variables(xarr)

        # solve and get functions
        solver.solve()
        mass = solver.evalFunction("mass")
        ksfail = solver.evalFunction("ksfailure")

    mass = comm.bcast(mass, root=root)
    ksfail = comm.bcast(ksfail, root=root)

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

    mass_grad, ksfail_grad = None, None

    if comm.rank == root:
        solver.set_design_variables(xarr)

        # before not including this would result in wrong state vars
        # now it only does the solve again if design changed with this call
        solver.solve() # temp debug just solve again here

        mass_grad = solver.evalFunctionSens("mass")
        ksfail_grad = solver.evalFunctionSens("ksfailure")

    mass_grad = comm.bcast(mass_grad, root=root)
    ksfail_grad = comm.bcast(ksfail_grad, root=root)

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
    lower=np.array([1e-2]*ndvs), # TODO : change min of non-struct-masses?
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

if comm.rank == root:
    solver.solve()
    solver.writeSolution("out/plate_opt.vtk")
    solver.free()