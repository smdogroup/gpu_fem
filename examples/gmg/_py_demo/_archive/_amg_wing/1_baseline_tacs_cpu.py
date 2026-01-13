# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive
import time

# Load structural mesh from BDF file
bdfFile = "../uCRM/CRM_box_2nd.bdf"
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
max_thickness = 0.20
thickness = 0.02

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
mat = tacs.createSchurMat()
# ILUk = 3
# ILUk = 8
ILUk = 0
# ILUk = 10
# ILUk = 100000 # full fillin
pc = TACS.Pc(mat, lev_fill=ILUk) # for ILU(3), default is full LU or ILU(1000) if not set
# subspace = 100
subspace = 200
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
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
tacs.applyBCs(mat)
pc.factor()

# Solve the linear system
start_time = time.time()
gmres.solve(forces, ans)
tacs_solve_time = time.time() - start_time
print(f"{tacs_solve_time=:.3e}")
tacs.setVariables(ans)


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
f5.writeToFile("ucrm.f5")