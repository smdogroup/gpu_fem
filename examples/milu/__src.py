import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import os
from mpi4py import MPI
from tacs import TACS, elements, constitutive
import scipy as sp
import matplotlib.pyplot as plt

def get_tacs_matrix(bdf_file, thickness:float=0.02, load_fcn=None):

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
    
    if load_fcn is None:
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
    return A_bsr, force_array, xpts_arr

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


def sort_vis_maps(nxe, xpts, free_dof):
    """create dof maps from reordered red to unsorted full and reverse"""
    # first is unsort_red to sort_free
    nx = nxe + 1
    N = nx**2
    nred = len(free_dof)
    nfree = 6 * N
    x, y = xpts[0::3], xpts[1::3]
    # node_sort_map = np.lexsort((y, x))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
    node_sort_map = np.lexsort((x, y))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
    # temp if don't want to resort basically
    # node_sort_map = np.arange(node_sort_map.shape[0])
    inv_node_sort_map = np.empty_like(node_sort_map)
    inv_node_sort_map[node_sort_map] = np.arange(N)

    sort_free_fw_map = np.zeros(nred, dtype=np.int32)
    sort_free_bk_map = -np.ones(nfree, dtype=np.int32) # -1 if not free var
    for ired in range(nred):
        ifree = free_dof[ired]
        inode = ifree // 6
        idof = ifree % 6

        inode_sort = inv_node_sort_map[inode]
        ifree_sort = 6 * inode_sort + idof

        sort_free_fw_map[ired] = ifree_sort
        sort_free_bk_map[ifree_sort] = ired

    return sort_free_fw_map, sort_free_bk_map



def right_pgmres(A, b, x0=None, restart=50, tol=1e-8, max_iter=1000, M=None):
    """
    Right-precond Modified Gram-Schmidt GMRES
    
    Parameters:
    A : callable or sparse matrix
        Function A(x) or sparse matrix representing the linear system.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess (default: zero vector).
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    restart : int, optional
        Restart parameter (default: 50).
    M : callable or sparse matrix, optional
        Preconditioner (default: None).
    
    Returns:
    x : ndarray
        Approximate solution.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r = b - (A @ x)
    beta = np.linalg.norm(r)
    # print(f"{beta=}")
    
    if beta < tol:
        return x

    for _ in range(max_iter // restart):
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        cs = np.zeros(restart)
        ss = np.zeros(restart)

        # Start Arnoldi process
        V[:, 0] = r / beta
        g[0] = beta
        # print(f"{g[0]=}")

        print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            z = M_inv(V[:,j])
            w = A @ z

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            # givens rotations 
            # ----------------

            for i in range(j):
                temp = H[i,j]
                H[i,j] = cs[i] * H[i,j] + ss[i] * H[i+1,j]
                H[i+1,j] = -ss[i] * temp + cs[i] * H[i+1,j]

            cs[j] = H[j,j] / np.sqrt(H[j,j]**2 + H[j+1,j]**2)
            ss[j] = cs[j] * H[j+1,j] / H[j,j]

            g_temp = g[j]
            g[j] *= cs[j]
            g[j+1] = -ss[j] * g_temp

            H[j,j] = cs[j] * H[j,j] + ss[j] * H[j+1,j]
            H[j+1,j] = 0.0

            if (j % 10 == 0): print(f"GMRES [{j}] : {g[j+1]=}")

            # Check convergence
            if abs(g[j+1]) < tol:
                print(f"g break at iteration {j}")
                break

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])

        # Update solution
        dz = V[:, :j+1] @ y
        dx = M_inv(dz)
        x += dx

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
    
    print(f"GMRES final resid {beta=}")
    return x


def random_ordering(N):
    # give new nofill pattern for random ordering

    # compute permutations
    perm = np.random.permutation(N)
    ind = np.arange(0, N)
    iperm = np.zeros(N, dtype=np.int32)
    iperm[perm] = ind
    return perm, iperm

def reorder_bsr6_nofill(A_bsr, perm, iperm):
    bs = 6
    nb = A_bsr.shape[0] // bs

    indptr = A_bsr.indptr
    indices = A_bsr.indices
    data = A_bsr.data

    new_indptr = np.zeros(nb + 1, dtype=indptr.dtype)
    new_indices = np.empty_like(indices)
    new_data = np.empty_like(data)

    nnzb = 0
    for new_i in range(nb):
        old_i = iperm[new_i]

        start, end = indptr[old_i], indptr[old_i + 1]
        cols = indices[start:end]

        new_cols = perm[cols]

        # 🔑 sort block columns
        order = np.argsort(new_cols)
        new_cols = new_cols[order]
        new_blocks = data[start:end][order]

        k = end - start
        new_indptr[new_i + 1] = new_indptr[new_i] + k

        new_indices[nnzb:nnzb + k] = new_cols
        new_data[nnzb:nnzb + k] = new_blocks

        nnzb += k

    # print(f"{new_indices=}\n{new_indptr=}\n")

    A_new = sp.sparse.bsr_matrix(
        (new_data, new_indices, new_indptr),
        shape=A_bsr.shape,
        blocksize=(bs, bs)
    )

    return A_new


def write_80(fout, line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)

import numpy

def gen_plate_mesh(nxe:int=10, lx:float=1.0, ly:float=1.0, name="plate"):
    # Overall plate dimensions lx, ly
    # Number of components in each direction
    ncx = 1
    ncy = 1
    ncomp = ncx * ncy

    # Number of elements along each edge of a single panel
    nex = nxe
    ney = nex

    # Nodes
    nx = ncx * nex + 1
    ny = ncy * ney + 1
    xtmp = numpy.linspace(0, lx, nx)
    ytmp = numpy.linspace(0, ly, ny)
    X, Y = numpy.meshgrid(xtmp, ytmp)
    Z = numpy.zeros_like(X)
    nodes = numpy.stack((X, Y, Z), axis=2)
    nmat = nodes.reshape((nx * ny, 3))

    # Node numbering
    nid = numpy.zeros((nx, ny), dtype="intc")
    bcnodes = []
    count = 1
    for i in range(ny):
        for j in range(nx):
            nid[j, i] = count
            if j == 0 or j == nx - 1 or i == 0 or i == (ny-1):
                bcnodes.append(count)
            count += 1

    # Connectivity
    nex = nx - 1
    ney = ny - 1
    ne = nex * ney
    conn = {i + 1: [] for i in range(ncomp)}
    ie = 1
    for i in range(ney):
        for j in range(nex):
            compID = i // ney * ncx + j // nex + 1
            conn[compID].append(
                [ie, nid[j, i], nid[j + 1, i], nid[j + 1, i + 1], nid[j, i + 1]]
            )
            ie += 1


    # Write BDF
    output_file = name + ".bdf"

    with open(output_file, "w") as fout:
        write_80(fout, "SOL 103")
        write_80(fout, "CEND")
        write_80(fout, "BEGIN BULK")

        # Make component names
        compNames = {}
        compID = 1
        for i in range(ncy):
            for j in range(ncx):
                compNames[compID] = "PLATE.{:03d}/SEG.{:02d}".format(i, j)
                compID += 1

        def write_bulk_line(key, items, format="small"):
            if format == "small":
                width = 8
                writekey = key
            elif format == "large":
                width = 16
                writekey = key + "*"
            line = "{:8s}".format(writekey)
            for item in items:
                if type(item) in [int, numpy.int64, numpy.int32]:
                    line += "{:{width}d}".format(item, width=width)[:width]
                elif type(item) in [float, numpy.float64]:
                    line += "{: {width}f}".format(item, width=width)[:width]
                elif type(item) is str:
                    line += "{:{width}s}".format(item, width=width)[:width]
                else:
                    print(type(item), item)
                if len(line) == 72:
                    write_80(fout, line)
                    line = " " * 8
            if len(line) > 8:
                write_80(fout, line)

        # Write nodes
        for i in range(nx * ny):
            write_bulk_line("GRID", [i + 1, 0, nmat[i, 0], nmat[i, 1], nmat[i, 2], 0, 0, 0])

        # Write elements
        compID = 1
        for key in conn:
            famPrefix = "$       Shell element data for family    "
            famString = "{}{:39s}".format(famPrefix, compNames[compID])
            write_80(fout, famString)
            compID += 1
            for element in conn[key]:
                element.insert(1, key)
                write_bulk_line("CQUAD4", element)

        # Write boundary conditions
        for node in bcnodes:
            write_bulk_line("SPC", [1, node, "123456", 0.0])

        write_80(fout, "ENDDATA")
