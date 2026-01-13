import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive
import scipy

def get_tacs_matrix():

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

    # Set up and solve the analysis problem
    res = tacs.createVec()
    mat = tacs.createSchurMat()

    xpts_vec = tacs.createNodeVec()
    tacs.getNodes(xpts_vec)
    xpts_arr = xpts_vec.getArray()

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray()
    force_array[2::6] += 100.0  # uniform load in z direction
    tacs.applyBCs(forces)

    # Assemble the Jacobian and factor
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)

    data = mat.getMat()
    # print(f"{data=}")
    A_bsr = data[0]
    # print(f"{A_bsr=} {type(A_bsr)=}")

    # TODO : just get serial A part in a minute
    return A_bsr, force_array, xpts_arr

def reduced_indices(A):
    # takes in A a csr matrix
    A.eliminate_zeros()

    dof = []
    for k in range(A.shape[0]):
        if (
            A.indptr[k + 1] - A.indptr[k] == 1
            and A.indices[A.indptr[k]] == k
            and np.isclose(A.data[A.indptr[k]], 1.0)
        ):
            # This is a constrained DOF
            pass
        else:
            # Store the free DOF index
            dof.append(k)
    return dof

def delete_rows_and_columns(A, dof=None):
    if dof is None:
        dof = reduced_indices(A)

    iptr = [0]
    cols = []
    data = []

    indices = -np.ones(A.shape[0])
    indices[dof] = np.arange(len(dof))

    for i in dof:
        for jp in range(A.indptr[i], A.indptr[i + 1]):
            j = A.indices[jp]

            if indices[j] >= 0:
                cols.append(indices[j])
                data.append(A.data[jp])

        iptr.append(len(cols))

    return scipy.sparse.csr_matrix((data, cols, iptr), shape=(len(dof), len(dof)))