# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os, time
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=100, help="# elements along length (and each direc)")
args = parser.parse_args()

# first call the plate mesh generator
os.system(f"python _gen_plate_mesh.py --nxe {args.nxe}")

# Load structural mesh from BDF file
bdfFile = "plate.bdf"
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
nz_start = time.time()
mat = tacs.createSchurMat(TACS.RCM_ORDER)

ILUk = 0
# ILUk = 3
# ILUk = 8
# ILUk = 1000
pc = TACS.Pc(mat, lev_fill=ILUk) # for ILU(3), default is full LU or ILU(1000) if not set
subspace = 100
# subspace = 200
restarts = 0
rtol = 1e-10
atol = 1e-6
gmres = TACS.KSM(mat, pc, subspace, restarts)
gmres.setMonitor(tacs_comm)
gmres.setTolerances(rtol, atol)
nz_time = time.time() - nz_start

# Assemble the Jacobian and factor
assembly_start = time.time()
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.zeroVariables()
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
assembly_time = time.time() - assembly_start

solve_start = time.time()
pc.factor()



# Solve the linear system
gmres.solve(forces, ans)
tacs.setVariables(ans)
solve_time = time.time() - solve_start


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
f5.writeToFile("plate.f5")

# write out runtimes to csv

tot_time = nz_time + assembly_time + solve_time
dof = (args.nxe+1)**2 * 6
if tacs_comm.rank == 0:
    print(f"{tacs_comm.size}, {args.nxe}, {dof}, {nz_time:.4e}, {assembly_time:.4e}, {solve_time:.4e}, {tot_time:.4e}\n")
