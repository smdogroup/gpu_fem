import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive
import scipy
import matplotlib.pyplot as plt

def get_tacs_matrix(bdf_file, thickness:float=0.02):

    # Load structural mesh from BDF file
    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdf_file)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    E = 70e9  # elastic modulus, Pa
    nu = 0.3  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.002
    max_thickness = 10.0 # 0.2
    # thickness = 0.02

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
    # force_array[2::6] += 100.0  # uniform load in z direction
    x = xpts_arr[0::3]
    y = xpts_arr[1::3]
    r = np.sqrt(x**2 + y**2)
    # force_array[2::6] += 100.0 # simple loading first
    
    def load_fcn(_x,_y):
        import math
        theta = math.atan2(_y, _x)
        r = np.sqrt(_x**2 + _y**2)
        return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta)
    # game of life polar load..

    force_array[2::6] = np.array([load_fcn(x[i], y[i]) for i in range(nnodes)])
    # force_array[2::6] += np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)

    # force_array[2::6] += 100.0 * np.sin(3 * np.pi * x)
    # force_array[2::6] += 100.0 * np.sin(3 * np.pi * r)
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

    # inverse map from full dof => red dof
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

def plot_init():
    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

def plot_plate_vec(nxe, vec, ax, sort_fw, nodal_dof:int=2, cmap='viridis'):
    """assume vec is one DOF per node only here (and includes bcs)"""
    nx = nxe + 1
    N = nx**2

    plot_vec = np.zeros((6*N,))
    plot_vec[sort_fw] = vec[:]
    plot_vec = plot_vec[nodal_dof::6]

    # plot_vec[_free_dof] = vec[:]
    # # reordering of solution for plotting..
    # x, y = _xpts[0::3], _xpts[1::3]
    # sort_idx = np.lexsort((y, x))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
    # # print(f"{sort_idx=}")
    # plot_vec = plot_vec[nodal_dof::6][sort_idx]

    x = np.linspace(0.0, 1.0, nx)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    VALS = np.reshape(plot_vec, (nx, nx))
    ax.plot_surface(X, Y, VALS, cmap=cmap)

def plot_vec_compare(nxe, old_defect, new_defect, sort_fw_map, filename=None, nodal_dof:int=2):
    from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
    plot_init()
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), subplot_kw={'projection': '3d'})
    plot_plate_vec(nxe=nxe, vec=old_defect, ax=ax[0], sort_fw=sort_fw_map, nodal_dof=nodal_dof)
    plot_plate_vec(nxe=nxe, vec=new_defect, ax=ax[1], sort_fw=sort_fw_map, nodal_dof=nodal_dof)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_vec_compare_all(nxe, old_defect, new_defect, sort_fw_map, filename=None):
    from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
    plot_init()
    fig, ax = plt.subplots(2, 3, figsize=(20, 8), subplot_kw={'projection': '3d'})

    # only w, thy, thz are coupled for shell + plate (plotted all six before)
    for idof in [2,3,4]:
        plot_plate_vec(nxe=nxe, vec=old_defect, ax=ax[0,idof-2], sort_fw=sort_fw_map, nodal_dof=idof)
        plot_plate_vec(nxe=nxe, vec=new_defect, ax=ax[1,idof-2], sort_fw=sort_fw_map, nodal_dof=idof)
    ax[0,0].set_title("w")
    ax[0,1].set_title("thx")
    ax[0,2].set_title("thy")

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def zero_non_nodal_dof(vec, sort_fw, inodal:int=2):
    """zero errors in the other nodal dof"""
    for ired in range(vec.shape[0]):
        ifree = sort_fw[ired]
        if ifree % 6 != inodal:
            vec[ired] = 0.0
    return vec


def gen_plate_bdf(
    lx=1.0,
    ly=1.0,
    nxe=10,
    apply_bcs=True,
    name="plate",
):
    """
    Write a rectangular plate BDF using CQUAD4 elements.

    Parameters
    ----------
    lx, ly : float
        Plate dimensions
    nxe : int
        Number of elements per side
    apply_bcs : bool
        Whether to apply boundary conditions
    name : str
        Output filename prefix (writes name.bdf)
    """

    def write_80(fout, line):
        fout.write("{:80s}\n".format(line.strip("\n")))

    def write_bulk_line(fout, key, items, format="small"):
        if format == "small":
            width = 8
            writekey = key
        else:
            width = 16
            writekey = key + "*"

        line = "{:8s}".format(writekey)
        for item in items:
            if isinstance(item, (int, np.integer)):
                line += f"{item:{width}d}"[:width]
            elif isinstance(item, (float, np.floating)):
                line += f"{item:{width}f}"[:width]
            elif isinstance(item, str):
                line += f"{item:{width}s}"[:width]
            else:
                raise TypeError(type(item), item)

            if len(line) == 72:
                write_80(fout, line)
                line = " " * 8

        if len(line) > 8:
            write_80(fout, line)

    # -------------------------
    # Mesh generation
    # -------------------------
    nex = ney = nxe
    nx = nex + 1
    ny = ney + 1

    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    nodes = np.stack((X, Y, Z), axis=2)
    nmat = nodes.reshape((-1, 3))

    # Node numbering
    nid = np.zeros((nx, ny), dtype=int)
    bcnodes = []
    count = 1
    for iy in range(ny):
        for ix in range(nx):
            nid[ix, iy] = count
            if ix in (0, nx - 1) or iy in (0, ny - 1):
                bcnodes.append(count)
            count += 1

    # Connectivity
    conn = []
    eid = 1
    for iy in range(ney):
        for ix in range(nex):
            conn.append([
                eid,
                1,
                nid[ix, iy],
                nid[ix + 1, iy],
                nid[ix + 1, iy + 1],
                nid[ix, iy + 1],
            ])
            eid += 1

    # -------------------------
    # Write BDF
    # -------------------------
    filename = f"{name}.bdf"
    with open(filename, "w") as fout:
        write_80(fout, "SOL 103")
        write_80(fout, "CEND")
        write_80(fout, "BEGIN BULK")

        # Nodes
        for i in range(nx * ny):
            write_bulk_line(
                fout,
                "GRID",
                [i + 1, 0, nmat[i, 0], nmat[i, 1], nmat[i, 2], 0, 0, 0],
            )

        # Elements
        write_80(fout, "$       Shell element data")
        for e in conn:
            write_bulk_line(fout, "CQUAD4", e)

        # Boundary conditions
        if apply_bcs:
            for node in bcnodes:
                ix = (node - 1) % nx
                iy = (node - 1) // nx

                x_bndry = ix == 0 or ix == nx - 1
                y_bndry = iy == 0 or iy == ny - 1

                if node == 1:
                    bc = "123456"
                elif x_bndry and not y_bndry:
                    bc = "34"
                elif y_bndry and not x_bndry:
                    bc = "35"
                elif x_bndry and y_bndry:
                    bc = "345"
                else:
                    bc = "3"

                write_bulk_line(fout, "SPC", [1, node, bc, 0.0])

        write_80(fout, "ENDDATA")

    return filename