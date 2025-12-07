# A demonstration of a simple optimization problem in TACS: minimize the
# mass of the CRM model subject to a global stress aggregate constraint
# enforcing that the maximum stress at any quadrature point is less than
# a specified upper bound.

# Import necessary libraries
import numpy as np
from mpi4py import MPI
from pyoptsparse import SNOPT, Optimization
import os
from tacs import pyTACS, elements, constitutive, functions


class LinearPlateAnalysis:
    """
    Mass minimization with a von Mises stress constraint
    """

    def __init__(self, comm, bdf_name):
        self.comm = comm

        # Set constitutive properties
        rho = 2500.0  # density, kg/m^3
        E = 70e9  # elastic modulus, Pa
        nu = 0.3  # poisson's ratio
        ys = 350e6  # yield stress, Pa
        # min_thickness = 0.005
        min_thickness = 0.01
        max_thickness = 2.0
        thickness = 0.02

        # Callback function used to setup TACS element objects and DVs
        def elemCallBack(
            dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
        ):
            # Setup (isotropic) property and constitutive objects
            prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
            # Set one thickness dv for every component
            con = constitutive.IsoShellConstitutive(
                prop, t=thickness, tNum=dvNum, tlb=min_thickness, tub=max_thickness
            )

            # This is an all-quad mesh, return a quad shell
            transform = None
            elem = elements.Quad4Shell(transform, con)
            return elem

        # Create pyTACS assembler object
        self.fea_assembler = pyTACS(bdf_name, comm=comm)
        # Setup elements
        self.fea_assembler.initialize(elemCallBack)

        xpts = self.fea_assembler.Xpts0.getArray()

        # Create a static problem with a uniform z load applied to all nodes
        self.static_problem = self.fea_assembler.createStaticProblem("uniform_z_load")

        # # Assume xpts is your nodal coordinates array: [x0, y0, z0, x1, y1, z1, ...]
        # X = xpts[0::3]
        # Y = xpts[1::3]
        # Z = xpts[2::3]

        # # Cylinder dimensions
        # Lx = np.max(X) - np.min(X)
        # nx = int(np.sqrt(X.shape[0]))  # approximate nodes along axis
        # dx = Lx / (nx - 1)
        # # dy = dx  # assume uniform spacing in hoop
        # R = 0.5
        # dth = 2.0 * np.pi / nx
        # ds = R * dth

        # # Identify boundary nodes (ends at X=0 or X=L)
        # # ends = np.logical_or(X == 0.0, X == Lx)
        # # interior = 1.0 - ends  # mask for interior nodes

        # # Cylinder coordinates
        # # R = np.sqrt(Z**2 + Y**2)
        # # TH = np.arctan2(Y, Z)
        # TH = np.arctan2(Z, Y)

        # # Base load magnitude
        # load_mag = 5e7 * dx * ds #* interior

        # # Correction factor
        # load_corr = 2.14 / 3.56
        # load_mag *= load_corr

        # # Quadrature-style "rose" load function
        # x_hat = X / Lx
        # th = TH
        # th_hat = th / (2.0 * np.pi)

        # mag = load_mag * (
        #     0.3 * np.cos(5.0 * th + 2.0 * np.pi * x_hat) +
        #     0.7 * np.cos(10.0 * th + np.pi/6.0 + 5.3 * np.pi * x_hat)
        # ) * np.sin(5.0 * np.pi * x_hat + 0.5 * 2.0 * x_hat**2)

        # # Create FEA load vector (6 DOF per node)
        # F = self.fea_assembler.createVec()

        # # Map quadrature load to y and z components
        # # F[1::6] += np.sin(th) * mag  # y-component
        # # F[2::6] += np.cos(th) * mag  # z-component
        # F[1::6] += np.cos(th) * mag  # y-component
        # F[2::6] += np.sin(th) * mag  # z-component

        # self.fea_assembler.applyBCsToVec(F)

        # # indicator = np.ones(F.shape)
        # # self.fea_assembler.applyBCsToVec(indicator)
        # # print(f"{indicator=}")
        
        # # print(f"{load_mag=}")
        # # for i in range(10):
        # #     print(f"{Fadd[i]=}")

        # self.static_problem.addLoadToRHS(F)

        # add loads from BDF instead
        self.static_problem.addLoadFromBDF(1)


        # Add failure function (con)
        self.static_problem.addFunction(
            "ks_failure", functions.KSFailure, ksWeight=100.0, safetyFactor=1.5,
        )
        # add mass (obj)
        self.static_problem.addFunction("mass", functions.StructuralMass)

        # Scale the mass objective so that it is O(10)
        # self.mass_scale = 1e-1
        self.mass_scale = 1.0 # do outside this

        # Scale the thickness variables so that they are measured in
        # mm rather than meters
        # self.thickness_scale = 1000.0
        # self.thickness_scale = 100.0
        self.thickness_scale = 1.0 # do outside this

        # The number of thickness variables in the problem
        self.nvars = self.fea_assembler.getNumDesignVars()
        # print(f"{self.nvars=}")

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        xvals = self.static_problem.getDesignVars()
        x[:] = self.thickness_scale * xvals

        xlb, xub = self.static_problem.getDesignVarRange()
        lb[:] = self.thickness_scale * xlb
        ub[:] = self.thickness_scale * xub

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1)
        func_dict = {}

        # Set the new design variable values
        # self.static_problem.setDesignVars(x[:] / self.thickness_scale)
        # print("set DVs")
        if self.comm.rank == 0:
            self.static_problem.setDesignVars(x[:] / self.thickness_scale)
        else:
            self.static_problem.setDesignVars(np.zeros(0))
        # print("solve")

        # Solve
        self.static_problem.solve()
        # print("done solve")

        # Evaluate the function
        self.static_problem.evalFunctions(func_dict)

        if self.comm.rank == 0:
            print(func_dict)

        # Set the mass as the objective
        fobj = self.mass_scale * func_dict["uniform_z_load_mass"]

        # Set the KS function (the approximate maximum ratio of the
        # von Mises stress to the design stress) so that
        # it is less than or equal to 1.0
        con[0] = (
            func_dict["uniform_z_load_ks_failure"]
        )  # changed to just return ksfailure
        # ~= 1.0 - max (sigma/design) >= 0

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0
        sens_dict = {}

        # Set the new design variable values
        if self.comm.rank == 0:
            self.static_problem.setDesignVars(x[:] / self.thickness_scale)
        else:
            self.static_problem.setDesignVars(np.zeros(0))

        # Evaluate the function sensitivities
        self.static_problem.evalFunctionsSens(sens_dict)

        # objective gradient
        if self.comm.rank == 0:
            g[:] = (
                self.mass_scale
                * sens_dict["uniform_z_load_mass"]["struct"]
                / self.thickness_scale
            )
        # In-place broadcast of g
        self.comm.Bcast(g, root=0)
        # print(f"{self.comm.rank=} {g=}")

        

        # Set the constraint gradient
        if self.comm.rank == 0:
            A[0][:] = (
                sens_dict["uniform_z_load_ks_failure"]["struct"] / self.thickness_scale
            )
            
        # In-place broadcast of A[0]
        self.comm.Bcast(A[0], root=0)

        # # Write out the solution file every 10 iterations
        # if self.iter_count % 10 == 0:
        #     self.static_problem.writeSolution()
        # self.iter_count += 1

        return fail

    def writeSolutionVTK(self):
        # write solution to VTK file
        # self.static_problem.solve()
        self.static_problem.writeSolution(outputDir="out")

    def writeSolutionBDF(self, file_name):
        # Write loads and optimized panel properties to BDF
        self.fea_assembler.writeBDF(file_name, self.static_problem)


# Load structural mesh from BDF file
tacs_comm = MPI.COMM_WORLD
bdf_name = "cylinder.bdf"

plate_opt = LinearPlateAnalysis(tacs_comm, bdf_name)
nvars = plate_opt.nvars

g =  np.zeros(nvars)
A = np.zeros((1,nvars))

# print(f"{x0=}")
x0, lb, ub = np.zeros(nvars), np.zeros(nvars), np.zeros(nvars)
plate_opt.getVarsAndBounds(x0, lb, ub)

x0 = plate_opt.comm.bcast(x0)
lb = plate_opt.comm.bcast(lb)
ub = plate_opt.comm.bcast(ub)

g = plate_opt.comm.bcast(g)
A = plate_opt.comm.bcast(A)

nvars = plate_opt.comm.bcast(nvars)



# DEBUG (linear solve)
# fail, obj, con = plate_opt.evalObjCon(x0)
# print(f"{obj=} {con=}")
# plate_opt.writeSolutionVTK()
# exit()

# OPTIMIZATION
# ======================
def get_functions(xdict):
    # update design
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])

    mass, ksfail = None, None

    fail, mass, ksfail = plate_opt.evalObjCon(xarr)
    funcs = {
        'mass' : mass,
        'ksfailure' : ksfail[0],
    }
    # print(f"{mass=} {ksfail=}")

    # print(f"{funcs=}")

    comp_names = [f"comp{icomp}" for icomp in range(nvars)]
    with open("out/lin_cpu_opt.txt", "w") as f:
        f.write(f"{mass=:.4e} {ksfail[0]=:.4e}\n")
        for name, value in zip(comp_names, xarr):
            f.write(f"{name}\t{value:.16e}\n")

    return funcs, False

def get_function_grad(xdict, funcs):
    # update design, this shouldn't really come with a new forward solve, just adj solve
    # but if optimizer calls grads twice in a row or something, this won't work right (needs companion get_functions call every time)
    xlist = xdict["vars"]
    xarr = np.array([float(_) for _ in xlist])

    mass_grad, ksfail_grad = None, None
    fail = plate_opt.evalObjConGradient(xarr, g, A)

    mass_grad = g[:]
    ksfail_grad = A[0][:]

    sens = {
        'mass' : {'vars' : mass_grad},
        'ksfailure' : {'vars' : ksfail_grad},
    }

    # print(f"{sens=}")
    return sens, False


opt_problem = Optimization("lin-cpu-opt", get_functions)
opt_problem.addVarGroup(
    "vars",
    nvars,
    lower=lb, # TODO : change min of non-struct-masses?
    upper=ub,
    value=x0,
    scale=np.array([1e2]*nvars), # handled internally by TACS
)
opt_problem.addObj('mass', scale=1.0e-1) # handled internally by TACS opt
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
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-4,
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

if tacs_comm.rank == 0:
    plate_opt.writeSolutionVTK()