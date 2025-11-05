# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions
import time

start_time = time.time()

# Load structural mesh from BDF file
bdfFile = "../uCRM/CRM_box_2nd.bdf"
# bdfFile = "uCRM-135_wingbox_fine.bdf"
tacs_comm = MPI.COMM_WORLD
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile(bdfFile)

# Set constitutive properties
rho = 2500.0  # density, kg/m^3
E = 70e9  # elastic modulus, Pa
nu = 0.3  # poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress, Pa
min_thickness = 0.002
max_thickness = 2.0
# thickness = 0.02
thickness = 1.0

# Loop over components, creating stiffness and element object for each
num_components = struct_mesh.getNumComponents()
for i in range(num_components):
    descriptor = struct_mesh.getElementDescript(i)
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    stiff = constitutive.IsoShellConstitutive(
        prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i
    )

    element = None
    transform = None
    if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
        element = elements.Quad4Shell(transform, stiff)
    struct_mesh.setElement(i, element)

# Create tacs assembler object from mesh loader
tacs = struct_mesh.createTACS(6)

# Get the design variable values
x = tacs.createDesignVec()
x_array = x.getArray()
tacs.getDesignVars(x)

# Get the node locations
X = tacs.createNodeVec()
tacs.getNodes(X)
tacs.setNodes(X)

# Create the forces
forces = tacs.createVec()
force_array = forces.getArray()
force_array[2::6] += 100.0  # uniform load in z direction
tacs.applyBCs(forces)

# Set up and solve the analysis problem
res = tacs.createVec()
ans = tacs.createVec()
u = tacs.createVec()
# default is AMD ordering
start_nz = time.time()
ordering = TACS.TACS_AMD_ORDER
# ordering = TACS.RCM_ORDER
# ordering = TACS.ND_ORDER
mat = tacs.createSchurMat(ordering)

nz_time = time.time() - start_nz

ILUk = 0
# ILUk = 3
# ILUk = 7
# ILUk = 1e6
pc = TACS.Pc(mat, lev_fill=ILUk) #, lev_fill=ILUk) # for ILU(3), default is full LU or ILU(1000) if not set
subspace = 100
end_nz = time.time()
# subspace = 300
restarts = 0
rtol = 1e-10
atol = 1e-6
gmres = TACS.KSM(mat, pc, subspace, restarts)
gmres.setMonitor(tacs_comm)
gmres.setTolerances(rtol, atol)


# Assemble the Jacobian and factor
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.zeroVariables()
start_jac = time.time()
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
jac_time = time.time() - start_jac

# assemble residual time
start_res = time.time()
tacs.assembleRes(res)
res_time = time.time() - start_res


# Solve the linear system including LU factorization
tacs_comm.Barrier()
start_lu = time.time()

pc.factor()

fact_time = time.time() - start_lu

start_triang = time.time()

gmres.solve(forces, ans)

triang_time = time.time() - start_triang

start_mult = time.time()
mat.mult(res, ans)
mult_time = time.time() - start_mult

tacs_comm.Barrier()
end_time = time.time()
if tacs_comm.rank == 0:
    print(f"uCRM solved with {tacs_comm.size=} procs\n")
    print(f"\thardware,count,nz_time,res_time,jac_time,fact_time,triang_time,mult_time\n")
    print(f"\tCPU,{tacs_comm.size},{nz_time:.4e},{res_time:.4e},{jac_time:.4e},{fact_time:.4e},{triang_time:.4e},{mult_time:.4e}")
    # print(f"\tSpMV in {mult_time:.4e} sec")
    
tacs.setVariables(ans)

dt = time.time() - start_time
if tacs_comm.rank == 0:
    print(f"total time = {dt:.4e}")


# Output for visualization
flag = (
    TACS.OUTPUT_CONNECTIVITY
    | TACS.OUTPUT_NODES
    | TACS.OUTPUT_DISPLACEMENTS
    | TACS.OUTPUT_STRAINS
    | TACS.OUTPUT_STRESSES
    | TACS.OUTPUT_EXTRAS
    | TACS.OUTPUT_LOADS
)
f5 = TACS.ToFH5(tacs, TACS.BEAM_OR_SHELL_ELEMENT, flag)
f5.writeToFile("out/ucrm-cpu-lin.f5")