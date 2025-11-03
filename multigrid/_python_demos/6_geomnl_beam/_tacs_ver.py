"""
6 noded beam model 1 meter long in x direction.
We apply a tip load in the z direction and clamp it at the root.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint
from mpi4py import MPI
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import constitutive, elements, functions, pyTACS

comm = MPI.COMM_WORLD

PWD = os.path.dirname(__file__)
OUT = os.path.join(PWD, "out")


# Instantiate FEAAssembler
structOptions = {}

# bdfFile = os.path.join(os.path.dirname(__file__), "beam_model.bdf")
bdfFile = os.path.join(os.path.dirname(__file__), "Beam.bdf")
# Load BDF file
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)

# Material properties
rho = 2700.0  # density kg/m^3
E = 70.0e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
ys = 270.0e6  # yield stress

# dimensions
L = 1.0
b = 4e-3 # m
SR = 100.0
thick = L / SR
A = b * thick

# print(f"{thick=:.2e}")

Iy = b * thick**3 / 12.0
# Iy *= 2.0

Iz = thick * b**3 / 12.0

# Iz = b * thick**3 / 12.0
# Iy = thick * b**3 / 12.0

J = Iy + Iz

Ks = 5.0 / 6.0
# Ks = 1000.0

# Callback function used to setup TACS element objects and DVs
def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.BasicBeamConstitutive(
        prop, A=A, Iy=Iy, Iz=Iz, J=J, ky=Ks, kz=Ks
    )

    refAxis = np.array([0.0, 1.0, 0.0])
    # refAxis = np.array([0.0, 0.0, 1.0])

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = elements.BeamRefAxisTransform(refAxis)
    # elem = elements.Beam2(transform, con)
    elem = elements.Beam2ModRot(transform, con)
    return elem


# Set up elements and TACS assembler
FEAAssembler.initialize(elemCallBack)

# ==============================================================================
# Setup static problem
# ==============================================================================
# Static problem

# PRESSURE = 2.0e3
# PRESSURE = 2.0e2
PRESSURE = 4.0e2


SOLVE_NL = True
# SOLVE_NL = False

if not(SOLVE_NL):
    problems = FEAAssembler.createTACSProbsFromBDF().values()
    
    for problem in problems:
        bdfInfo = FEAAssembler.getBDFInfo()
        nastranNodeNums = list(bdfInfo.node_ids)
        numNodes = len(nastranNodeNums)
        numInteriorNodes = numNodes - 2
        totalForce = PRESSURE * L

        # problem.addLoadToNodes(
        #     nastranNodeNums, [0, 0, totalForce / numInteriorNodes, 0, 0, 0], nastranOrdering=True
        # )

        
        # Solve state
        problem.solve()

        U = problem.u_array.copy()
        w = U[2::6]
        w_max = np.max(np.abs(w))
        print(f"LIN : {w_max=:.4e}")

    # Write solution out
    for problem in problems:
        problem.writeSolution(outputDir=OUT)

if SOLVE_NL:

    probOptions = {
        "printTiming": True,
        "printLevel": 1,
    }
    newtonOptions = {"useEW": True, "MaxLinIters": 10}
    continuationOptions = {
        "CoarseRelTol": 1e-3,
        "InitialStep": 0.2,
        "UsePredictor": True,
        "NumPredictorStates": 7,
    }
    forceProblem = FEAAssembler.createStaticProblem("TipForce", options=probOptions)
    try:
        forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)
        forceProblem.nonlinearSolver.setOptions(continuationOptions)
    except AttributeError:
        pass

    bdfInfo = FEAAssembler.getBDFInfo()
    nastranNodeNums = list(bdfInfo.node_ids)
    numNodes = len(nastranNodeNums)
    numInteriorNodes = numNodes - 2
    totalForce = PRESSURE * L

    forceProblem.addLoadToNodes(
        nastranNodeNums, [0, 0, totalForce / numInteriorNodes, 0, 0, 0], nastranOrdering=True
    )


    forceFactor = np.arange(0.10, 1.01, 0.2)
    ForceVec = np.copy(forceProblem.F_array)


    for scale in forceFactor:
        Fext = (scale - 1.0) * ForceVec
        forceProblem.solve(Fext=Fext)

        fileName = f"beam_nl"
        forceProblem.writeSolution(outputDir=OUT, baseName=fileName)
        disps = forceProblem.u_array
        xDisps = disps[0::6]
        zDisps = disps[2::6]
        yRot = disps[4::6]

        w_max = np.max(np.abs(zDisps))
        print(f"NL : {w_max=:.4e}")