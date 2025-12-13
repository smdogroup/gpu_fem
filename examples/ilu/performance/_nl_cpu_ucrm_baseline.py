# A demonstration of basic functions of the Python interface for TACS,
# this example goes through the process of setting up the using the
# tacs assembler directly as opposed to using the pyTACS user interface (see analysis.py):
# loading a mesh, creating elements, evaluating functions, solution, and output
# Import necessary libraries
import numpy as np
import os
from mpi4py import MPI
from tacs import pyTACS, elements, constitutive

# Load structural mesh from BDF file
COMM = MPI.COMM_WORLD
PWD = os.path.join(os.path.dirname(__file__), "out")

# BDF_FILE = "../uCRM/CRM_box_2nd.bdf"
BDF_FILE = "uCRM-135_wingbox_fine.bdf"


structOptions = {
    "printtiming": True,
}
FEAAssembler = pyTACS(BDF_FILE, options=structOptions, comm=COMM)

# material properties, shell thickness, etc.
# ------------------------------------------
RHO = 2500 # density kg/m^3
E = 70e9 # elastic modulus, Pa
NU = 0.3 # poisson's ratio
YIELD_STRESS = 350e6 # yield stress, Pa
THICKNESS = 0.02 # meters

def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    matProps = constitutive.MaterialProperties(rho=RHO, E=E, nu=NU, ys=YIELD_STRESS)
    con = constitutive.IsoShellConstitutive(
        matProps, t=THICKNESS, tNum=dvNum, tlb=1e-2 * THICKNESS, tub=1e2 * THICKNESS
    )
    transform = None
    element = elements.Quad4NonlinearShell(transform, con)
    tScale = [10.0]
    return element, tScale

FEAAssembler.initialize(elemCallBack)

probOptions = {
    "printTiming": True,
    "printLevel": 1,
}
newtonOptions = {"useEW": True, "MaxLinIters": 10}
continuationOptions = {
    "CoarseRelTol": 1e-3,
    "InitialStep": 0.1,
    "UsePredictor": False, # True orig
    "NumPredictorStates": 7,
}
forceProblem = FEAAssembler.createStaticProblem("NLVertLoads", options=probOptions)
try:
    forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)
    forceProblem.nonlinearSolver.setOptions(continuationOptions)
except AttributeError:
    pass

# ==============================================================================
# Add linear vertical loads
# ==============================================================================

bdfInfo = FEAAssembler.getBDFInfo()
nodeIDs = list(bdfInfo.node_ids)

load_mag = 3.0
max_load_factor = 23.0

forceProblem.addLoadToNodes(
    nodeIDs, [0, 0, load_mag * max_load_factor, 0, 0, 0], nastranOrdering=True
)

# ==============================================================================
# Run analysis with load scales in 10% increments from 10% to 100%
# ==============================================================================

forceFactor = np.arange(0.10, 1.01, 0.10)
ForceVec = np.copy(forceProblem.F_array)

print(f"{forceFactor=}")

for scale in forceFactor:
    Fext = (scale - 1.0) * ForceVec
    forceProblem.solve(Fext=Fext)
    forceProblem.writeSolution(outputDir=PWD) 