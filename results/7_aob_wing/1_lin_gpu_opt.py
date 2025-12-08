import sys
sys.path.append("_src/") # contains gpusolver

import wingmultigrid
import numpy as np
from pyoptsparse import SNOPT, Optimization
import os
import argparse
from mpi4py import MPI

# only adding MPI right now to help pyoptsparse.. it runs better with more PROCS?
comm = MPI.COMM_WORLD
root = 0

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


# setup GPU solver on root
root = 0
if comm.rank == root:
    solver = SOLVER_CLASS(
        rhoKS=100.0,
        safety_factor=1.5,
        # force=684e3, # 30 KPa on lower skin from the structural benchmark
        force=684e3*3, # boost load from static benchmark (to get more deflection?)
        omega=0.85,
        nsmooth=1,
        # ORDER=8,
        ORDER=16, # went with this as fast for thin shell and not too much extra smoothing for thick shell
        # ORDER=32, # fastest for very thin shell case (but only a bit faster than ORDER=16)
        # ORDER=4,
        n_krylov=200,
        # rtol=1e-6, # almost want to have high rtol like this only later in the optimization tbh..
        rtol=1e-5,
        # rtol=1e-4, # faster optimization, but will it converge though?
        # print=True, # always on for these kinds of tests..
        print=False,
    )



# init dvs
ndvs = solver.get_num_dvs()
# x0 = np.array([5e-3]*ndvs)
x0 = np.array([3e-2]*ndvs)


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

    if solver.get_num_lin_solves() % 20 == 0 and comm.rank == root: # so we don't affect runtimes for fast problems
        comp_names = [f"comp{icomp}" for icomp in range(ndvs)]
        with open("out/1_lin_unstiff_gpu_opt.txt", "w") as f:
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

opt_problem = Optimization("aob-lin-wing", get_functions)
opt_problem.addVarGroup(
    "vars",
    ndvs,
    lower=np.array([2.5e-3]*ndvs), # TODO : change min of non-struct-masses?
    upper=np.array([1e2]*ndvs),
    value=x0,
    scale=np.array([1e2]*ndvs),
)
opt_problem.addObj('mass', scale=1e-3)
opt_problem.addCon("ksfailure", scale=1.0, upper=1.0)

# adjacency constraints (linear so reduces size of opt problem)
# ---------------------
# def adj_vec(i, j):
#     vec = np.zeros((1,ndvs), dtype=np.double)
#     vec[0,i] = 1.0
#     vec[0,j] = -1.0
#     return vec
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
        "Major feasibility tolerance": 1e-5, # 1e-6
        "Major optimality tolerance": 1e-3, # 1e-4
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
    solver.writeSolution("out/1_wing_unstiff_opt_lin.vtk")
    solver.free()

num_lin_solves = solver.get_num_lin_solves()
print(f"{num_lin_solves=}")
