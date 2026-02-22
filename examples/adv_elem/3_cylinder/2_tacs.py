"""
Run an unstiffened metal cylinder in TACS
"""

import numpy as np
from mpi4py import MPI
from tacs import functions, constitutive, elements, pyTACS, TACS
import matplotlib.pyplot as plt
import scipy.sparse as sp

def get_tacs_matrix(bdf_file, E:float, nu:float, load_fcn, thickness:float=0.02):

    # Load structural mesh from BDF file
    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdf_file)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    # E = 70e9  # elastic modulus, Pa
    # nu = 0.3  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.002
    max_thickness = 10.0 # 0.2

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
    mat = tacs.createSchurMat(TACS.NATURAL_ORDER)

    xpts_vec = tacs.createNodeVec()
    tacs.getNodes(xpts_vec)
    xpts_arr = xpts_vec.getArray()
    nnodes = xpts_arr.shape[0] // 3

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray()

    x = xpts_arr[0::3]
    y = xpts_arr[1::3]
    z = xpts_arr[2::3]
    r = np.sqrt(y**2 + z**2)
    C = y / r
    S = z / r

    nxe = int(np.sqrt(nnodes)) - 1
    int_nnodes = (nxe-1)**2
    force_mag_vec = np.array([load_fcn(x[i], y[i], z[i]) / int_nnodes for i in range(nnodes)])

    # assume cylinder on x axis so (y,z) plane in circle)
    force_array[1::6] += force_mag_vec * C
    force_array[2::6] += force_mag_vec * S
    tacs.applyBCs(forces)

    # Assemble the Jacobian and factor
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)
    # tacs.applyBCs(res, mat)

    data = mat.getMat()
    # print(f"{data=}")
    A_bsr = data[0]
    # print(f"{A_bsr=} {type(A_bsr)=}")

    # TODO : just get serial A part in a minute
    return A_bsr, force_array, xpts_arr

def runTacsToVTK(bdf_file, E:float, nu:float, load_fcn, xpts, thickness):

    import os
    comm = MPI.COMM_WORLD

    # Instantiate FEAAssembler
    structOptions = {
        "printtiming": True,
    }
    FEAAssembler = pyTACS(bdf_file, options=structOptions, comm=comm)

    # Material properties
    rho = 2780.0  # density kg/m^3
    ys = 324.0e6  # yield stress

    # Shell thickness
    # t = 0.01  # m
    t = thickness
    tMin = 0.002  # m
    tMax = 0.05  # m


    # Callback function used to setup TACS element objects and DVs
    def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        # Set one thickness dv for every component
        con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)
        refAxis = np.array([1, 0, 0])

        # For each element type in this component,
        # pass back the appropriate tacs element object
        elemList = []
        transform = elements.ShellRefAxisTransform(refAxis)
        for elemDescript in elemDescripts:
            if elemDescript in ["CQUAD4", "CQUADR"]:
                elem = elements.Quad4Shell(transform, con)
            elif elemDescript in ["CTRIA3", "CTRIAR"]:
                elem = elements.Tri3Shell(transform, con)
            else:
                print("Uh oh, '%s' not recognized" % (elemDescript))
            elemList.append(elem)

        # Add scale for thickness dv
        scale = [100.0]
        return elemList, scale


    # Set up elements and TACS assembler
    FEAAssembler.initialize(elemCallBack)

    # ==============================================================================
    # Setup static problem
    # ==============================================================================
    # Static problem
    problem = FEAAssembler.createStaticProblem("cylinder")

    # add loads here
    # assume cylinder on x axis so (y,z) plane in circle)
    x = xpts[0::3]
    y = xpts[1::3]
    z = xpts[2::3]
    r = np.sqrt(y**2 + z**2)
    C = y / r
    S = z / r
    nnodes = xpts.shape[0] // 3
    nxe = int(np.sqrt(nnodes)) - 1
    int_nnodes = (nxe-1)**2
    force_mag_vec = np.array([load_fcn(x[i], y[i], z[i]) / int_nnodes for i in range(nnodes)])
    force_array = np.zeros(6 * nnodes)
    force_array[1::6] += force_mag_vec * C
    force_array[2::6] += force_mag_vec * S
    nodal_forces = force_array.reshape((nnodes, 6))
    nodeIds = np.arange(0, nnodes) + 1
    problem.addLoadToNodes(nodeIds, nodal_forces, nastranOrdering=True)

    # ==============================================================================
    # Solve static problem
    # ==============================================================================

    # Solve structural problem
    problem.solve()
    problem.writeSolution(outputDir=os.path.dirname(__file__))

    # get and return solution
    soln = problem.u.getArray()
    return soln


def plot_cylinder_vec(nxe, vec, ax, sort_fw, cmap='viridis'):
    """Plot cylinder displacement or field (assumes 1 DOF per node)"""
    ntheta = nxe + 1
    nz = nxe + 1
    N = ntheta * nz

    plot_vec = np.zeros((6*N,))
    plot_vec[sort_fw] = vec[:]

    u = plot_vec[0::6]
    v = plot_vec[1::6]
    w = plot_vec[2::6]

    wn = np.sqrt(v**2 + w**2)
    plot_vec = wn.copy()

    # Build node coordinates
    theta = np.linspace(0, 0.5*np.pi, ntheta) #[:-1]
    z = np.linspace(0, 1.0, nz)
    X = np.zeros((nz, ntheta))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for iz, zi in enumerate(z):
        for it, ti in enumerate(theta):
            idx = iz*ntheta + it
            X[iz,it] = np.cos(ti)
            Y[iz,it] = np.sin(ti)
            Z[iz,it] = zi
    VALS = np.reshape(plot_vec, (nz, ntheta))
    ax.plot_surface(X, Y, Z + VALS*0.0, facecolors=plt.cm.get_cmap(cmap)(VALS/np.max(np.abs(VALS)+1e-12)), rstride=1, cstride=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cylinder')


def sort_vis_maps(nxe, xpts, free_dof):
    """
    Create forward/backward dof maps for visualization or reduced system reordering.
    For cylinder arc..
    
    Parameters
    ----------
    nxe : int
        Number of elements along x (circumference for cylinder, x for plate)
    xpts : array_like
        Flattened nodal coordinates [x0, y0, z0, x1, y1, z1, ...]
    free_dof : array_like
        Indices of free DOFs (flattened, 6 DOF per node)
    
    Returns
    -------
    sort_free_fw_map : np.ndarray
        Mapping from reduced free DOFs to reordered DOFs
    sort_free_bk_map : np.ndarray
        Reverse map from reordered DOFs to reduced free DOFs (-1 if constrained)
    """

    # Determine number of nodes
    npts = len(xpts) // 3
    nx_plate = nxe + 1
    nx_cyl = nxe + 1
    # Cylinder
    shape = "cylinder"
    ntheta = nx_cyl
    nz = nxe + 1
    x, y, z = xpts[0::3], xpts[1::3], xpts[2::3]
    # Compute theta from x,y
    theta = np.arctan2(y, x)
    # sort primarily along z (axial), then theta
    node_sort_map = np.lexsort((theta, z))

    # Inverse mapping: sorted node index -> original node index
    inv_node_sort_map = np.empty_like(node_sort_map)
    inv_node_sort_map[node_sort_map] = np.arange(npts)

    # Now map DOFs
    nred = len(free_dof)
    nfree = 6 * npts
    sort_free_fw_map = np.zeros(nred, dtype=np.int32)
    sort_free_bk_map = -np.ones(nfree, dtype=np.int32)

    for ired in range(nred):
        ifree = free_dof[ired]
        inode = ifree // 6
        idof = ifree % 6

        inode_sort = inv_node_sort_map[inode]
        ifree_sort = 6 * inode_sort + idof

        sort_free_fw_map[ired] = ifree_sort
        sort_free_bk_map[ifree_sort] = ired

    # Debug info
    # print(f"Detected geometry: {shape}, npts={npts}")

    return sort_free_fw_map, sort_free_bk_map


def write_80(fout, line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)

import numpy

def gen_cylinder_mesh(nel_circ:int=12, nel_axial:int=10, radius:float=1.0, length:float=2.0,
                      name="cylinder", apply_bcs:bool=True):
    """
    Generate a cylinder mesh (quadrilateral shell) like gen_plate_mesh.
    Last hoop wraps around to the first.
    Ends fully clamped (all 6 DOFs = 0).
    """

    # Nodes
    nnode_circ = nel_circ + 1
    nnode_axial = nel_axial + 1
    theta = np.linspace(0, 2*np.pi, nnode_circ+1)[:-1]  # wrap last
    z = np.linspace(0, length, nnode_axial)

    nodes = []
    nid = 1
    nid_map = {}
    for iz, zi in enumerate(z):
        for it, ti in enumerate(theta):
            x = radius * np.cos(ti)
            y = radius * np.sin(ti)
            nodes.append([x, y, zi])
            nid_map[(iz,it)] = nid
            nid += 1
    nodes = np.array(nodes)
    nnodes = nodes.shape[0]

    # Connectivity
    elements = []
    ie = 1
    for iz in range(nel_axial):
        for it in range(nel_circ):
            n1 = nid_map[(iz, it)]
            n2 = nid_map[(iz, (it+1)%nel_circ)]
            n3 = nid_map[(iz+1, (it+1)%nel_circ)]
            n4 = nid_map[(iz+1, it)]
            elements.append([ie, n1, n2, n3, n4])
            ie += 1

    # Components (1 per axial layer)
    ncomp = nel_axial
    conn = {i+1: [] for i in range(ncomp)}
    for idx, elem in enumerate(elements):
        compID = idx // nel_circ + 1
        conn[compID].append(elem)

    # BC nodes
    bcnodes = []
    if apply_bcs:
        for it in range(nnode_circ):
            bcnodes.append(nid_map[(0,it)])           # bottom
            bcnodes.append(nid_map[(nel_axial-1,it)]) # top

    # Write BDF
    output_file = name + ".bdf" if apply_bcs else name + "_nobc.bdf"
    with open(output_file, "w") as fout:
        write_80(fout, "SOL 103")
        write_80(fout, "CEND")
        write_80(fout, "BEGIN BULK")

        # Component names
        compNames = {}
        for compID in range(1, ncomp+1):
            compNames[compID] = "CYLINDER.{:03d}".format(compID)

        # Helper for bulk line writing
        def write_bulk_line(key, items, format="small"):
            if format == "small":
                width = 8
                writekey = key
            elif format == "large":
                width = 16
                writekey = key + "*"
            line = "{:8s}".format(writekey)
            for item in items:
                if isinstance(item, (int, np.integer)):
                    line += "{:{width}d}".format(item, width=width)[:width]
                elif isinstance(item, (float, np.floating)):
                    line += "{: {width}f}".format(item, width=width)[:width]
                elif isinstance(item, str):
                    line += "{:{width}s}".format(item, width=width)[:width]
                else:
                    print(type(item), item)
                if len(line) == 72:
                    write_80(fout, line)
                    line = " " * 8
            if len(line) > 8:
                write_80(fout, line)

        # Write nodes
        for i in range(nnodes):
            write_bulk_line("GRID", [i+1, 0, nodes[i,0], nodes[i,1], nodes[i,2], 0,0,0])

        # Write elements
        for compID in conn:
            famPrefix = "$       Shell element data for family    "
            famString = "{}{:39s}".format(famPrefix, compNames[compID])
            write_80(fout, famString)
            for elem in conn[compID]:
                # Insert component ID as in plate
                elem_with_comp = [elem[0], compID] + elem[1:]
                write_bulk_line("CQUAD4", elem_with_comp)

        # Write BCs (clamped boundary conditions)
        if apply_bcs:
            for node in bcnodes:
                write_bulk_line("SPC", [1, node, "123456", 0.0])

        write_80(fout, "ENDDATA")

    print(f"Cylinder mesh written to {output_file} with {nnodes} nodes and {len(elements)} elements.")

def gen_cylinder_mesh_quarter(nel_circ: int = 12, nel_axial: int = 10,
                              radius: float = 1.0, length: float = 2.0,
                              name: str = "cylinder_quarter", apply_bcs: bool = True):
    """
    Generate an *open* quarter-cylinder shell mesh (quadrilateral CQUAD4):
        theta in [0, pi/2]
    Cylinder axis is along +x.
    Each circular cross-section lies in the (y,z) plane.

    No wrap-around / periodic connection in the hoop direction.
    Axial ends are fully clamped (all 6 DOFs = 0) if apply_bcs=True.
    """

    # Nodes
    # Open seam (no wrap): nel_circ elements need nel_circ+1 theta nodes
    nnode_circ = nel_circ + 1
    nnode_axial = nel_axial + 1

    theta = np.linspace(0.0, 0.5 * np.pi, nnode_circ)   # includes both ends
    x_ax = np.linspace(0.0, length, nnode_axial)        # axis direction is x

    nodes = []
    nid = 1
    nid_map = {}
    for ix, xi in enumerate(x_ax):
        for it, ti in enumerate(theta):
            # cross-section in (y,z)
            y = radius * np.cos(ti)
            z = radius * np.sin(ti)
            nodes.append([xi, y, z])
            nid_map[(ix, it)] = nid
            nid += 1

    nodes = np.array(nodes, dtype=float)
    nnodes = nodes.shape[0]

    # Connectivity (no modulo wrap)
    elements = []
    ie = 1
    for ix in range(nel_axial):
        for it in range(nel_circ):
            n1 = nid_map[(ix, it)]
            n2 = nid_map[(ix, it + 1)]
            n3 = nid_map[(ix + 1, it + 1)]
            n4 = nid_map[(ix + 1, it)]
            elements.append([ie, n1, n2, n3, n4])
            ie += 1

    # Components (1 per axial layer)
    ncomp = nel_axial
    conn = {i + 1: [] for i in range(ncomp)}
    for idx, elem in enumerate(elements):
        compID = idx // nel_circ + 1
        conn[compID].append(elem)

    # BC nodes (clamp x=0 and x=length rings)
    bcnodes = []
    if apply_bcs:
        for it in range(nnode_circ):
            bcnodes.append(nid_map[(0, it)])          # x = 0
            bcnodes.append(nid_map[(nel_axial, it)])  # x = length
        # then also clamp the other edges
        for ix in range(nnode_axial):
            bcnodes.append(nid_map[(ix, 0)])          # th = 0
            bcnodes.append(nid_map[(ix, nel_circ)])  # th = th_max

    # Write BDF
    output_file = name + ".bdf" if apply_bcs else name + "_nobc.bdf"
    with open(output_file, "w") as fout:
        write_80(fout, "SOL 103")
        write_80(fout, "CEND")
        write_80(fout, "BEGIN BULK")

        compNames = {compID: f"CYLINDER_QTR.{compID:03d}" for compID in range(1, ncomp + 1)}

        def write_bulk_line(key, items, format="small"):
            if format == "small":
                width = 8
                writekey = key
            elif format == "large":
                width = 16
                writekey = key + "*"
            line = "{:8s}".format(writekey)
            for item in items:
                if isinstance(item, (int, np.integer)):
                    line += "{:{width}d}".format(item, width=width)[:width]
                elif isinstance(item, (float, np.floating)):
                    line += "{: {width}f}".format(item, width=width)[:width]
                elif isinstance(item, str):
                    line += "{:{width}s}".format(item, width=width)[:width]
                else:
                    print(type(item), item)
                if len(line) == 72:
                    write_80(fout, line)
                    line = " " * 8
            if len(line) > 8:
                write_80(fout, line)

        # Nodes
        for i in range(nnodes):
            write_bulk_line("GRID", [i + 1, 0, nodes[i, 0], nodes[i, 1], nodes[i, 2], 0, 0, 0])

        # Elements (with component ID in field 2)
        for compID in conn:
            famPrefix = "$       Shell element data for family    "
            famString = "{}{:39s}".format(famPrefix, compNames[compID])
            write_80(fout, famString)
            for elem in conn[compID]:
                elem_with_comp = [elem[0], compID] + elem[1:]
                write_bulk_line("CQUAD4", elem_with_comp)

        # Clamped BCs
        if apply_bcs:
            for node in bcnodes:
                write_bulk_line("SPC", [1, node, "123456", 0.0])

        write_80(fout, "ENDDATA")

    print(f"Quarter-cylinder mesh written to {output_file} with {nnodes} nodes and {len(elements)} elements.")




if __name__ == "__main__":
    # make TACS objects and run cylinder case

    # nxe = 128
    # nxe = 64
    # nxe = 32
    # nxe = 16
    # nxe = 8
    nxe = 4

    radius = 1.0
    length = 1.0
    # thickness = 1.0e-2
    thickness = 1.0e-3
    bdf_file = "cylinder.bdf"
    # load_fcn = lambda x, y, z : 1.0
    load_fcn = lambda x, y, z : 1.0 * np.sin(np.pi * x) * np.sin(np.pi * z) * np.sin(np.pi * y)
    # have to account for cylinder arc area
    # nodal_load_fcn = lambda x, y, z : load_fcn(x,y,z) * 2.0 * np.pi

    # gen_cylinder_mesh(nel_circ=nxe, nel_axial=nxe, radius=radius, length=length, name=bdf_file.split(".")[0], apply_bcs=True)
    gen_cylinder_mesh_quarter(nel_circ=nxe, nel_axial=nxe, radius=radius, length=length, name=bdf_file.split(".")[0], apply_bcs=True)

    K_bsr, force, xpts = get_tacs_matrix(bdf_file, E=70e9, nu=0.3, load_fcn=load_fcn, thickness=thickness)

    # also run TACS with standard API so we can view stresses in VTK files
    disp = runTacsToVTK(bdf_file, E=70e9, nu=0.3, load_fcn=load_fcn, xpts=xpts, thickness=thickness)

    norm_disp = disp[1::6] * xpts[1::3] + disp[2::6] * xpts[2::3]
    norm_nrm = np.max(np.abs(norm_disp))
    print(f"{norm_nrm=:.4e}")

    # soln = sp.linalg.spsolve(K_bsr, force)
    # # sort_fw_map, sort_bk_map = sort_vis_maps(nxe, xpts, 
    # fig = plt.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # nnodes = (nxe + 1)**2
    # N = 6 * nnodes
    # plot_cylinder_vec(nxe, soln, ax=ax, sort_fw=np.arange(0, N), cmap="viridis")
    # plt.show()
