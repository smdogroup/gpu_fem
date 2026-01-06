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
    """Plot plate displacement or field (assumes 1 DOF per node)"""
    nx = nxe + 1
    N = nx**2

    plot_vec = np.zeros((6*N,))
    plot_vec[sort_fw] = vec[:]
    plot_vec = plot_vec[nodal_dof::6]

    x = np.linspace(0.0, 1.0, nx)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    VALS = np.reshape(plot_vec, (nx, nx))
    ax.plot_surface(X, Y, VALS, cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plate')

def plot_cylinder_vec(nxe, vec, ax, sort_fw, nodal_dof:int=2, cmap='viridis'):
    """Plot cylinder displacement or field (assumes 1 DOF per node)"""
    ntheta = nxe
    nz = nxe + 1
    N = ntheta * nz

    plot_vec = np.zeros((6*N,))
    plot_vec[sort_fw] = vec[:]
    plot_vec = plot_vec[nodal_dof::6]

    # Build node coordinates
    theta = np.linspace(0, 2*np.pi, ntheta+1)[:-1]
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

def plot_vec_generic(nxe, vec, ax, sort_fw, nodal_dof:int=2, cmap='viridis'):
    """
    Outer plotting method that detects plate vs cylinder from number of nodes.
    
    Plates: (nxe+1)^2 nodes
    Cylinders: nxe*(nxe+1) nodes
    """
    npts = len(vec) // 6
    print(f"{npts=} {nxe=}")
    if npts == (nxe+1)**2:
        plot_plate_vec(nxe, vec, ax, sort_fw, nodal_dof, cmap)
    elif npts == nxe*(nxe+1):
        plot_cylinder_vec(nxe, vec, ax, sort_fw, nodal_dof, cmap)
    else:
        raise ValueError(f"Cannot detect geometry from {npts} nodes for nxe={nxe}.")

def plot_vec_compare(nxe, old_defect, new_defect, sort_fw_map, nodal_dof:int=2, filename=None):
    """Compare two vectors side-by-side, auto-detecting plate/cylinder"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), subplot_kw={'projection': '3d'})
    plot_vec_generic(nxe, old_defect, ax[0], sort_fw_map, nodal_dof)
    plot_vec_generic(nxe, new_defect, ax[1], sort_fw_map, nodal_dof)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


# def sort_vis_maps(nxe, xpts, free_dof):
#     """create dof maps from reordered red to unsorted full and reverse"""
#     # first is unsort_red to sort_free
#     nx = nxe + 1
#     N = nx**2
#     nred = len(free_dof)
#     nfree = 6 * N
#     x, y = xpts[0::3], xpts[1::3]
#     # node_sort_map = np.lexsort((y, x))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
#     node_sort_map = np.lexsort((x, y))  # lexsort uses last index first, so (y, x) gives x primary, y secondary
#     # temp if don't want to resort basically
#     # node_sort_map = np.arange(node_sort_map.shape[0])
#     inv_node_sort_map = np.empty_like(node_sort_map)
#     inv_node_sort_map[node_sort_map] = np.arange(N)

#     sort_free_fw_map = np.zeros(nred, dtype=np.int32)
#     sort_free_bk_map = -np.ones(nfree, dtype=np.int32) # -1 if not free var
#     for ired in range(nred):
#         ifree = free_dof[ired]
#         inode = ifree // 6
#         idof = ifree % 6

#         inode_sort = inv_node_sort_map[inode]
#         ifree_sort = 6 * inode_sort + idof

#         sort_free_fw_map[ired] = ifree_sort
#         sort_free_bk_map[ifree_sort] = ired

#     return sort_free_fw_map, sort_free_bk_map

import numpy as np

def sort_vis_maps(nxe, xpts, free_dof):
    """
    Create forward/backward dof maps for visualization or reduced system reordering.
    Automatically detects plate vs cylinder based on number of nodes:
      - plate: (nxe+1)^2 nodes
      - cylinder: nxe*(nxe+1) nodes (circumferential x axial)
    
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
    nx_cyl = nxe

    if npts == nx_plate**2:
        # Plate
        shape = "plate"
        nx = nx_plate
        x, y = xpts[0::3], xpts[1::3]
        # sort primarily along x, then y
        node_sort_map = np.lexsort((y, x))
    elif npts == nx_cyl * (nxe + 1):
        # Cylinder
        shape = "cylinder"
        ntheta = nx_cyl
        nz = nxe + 1
        x, y, z = xpts[0::3], xpts[1::3], xpts[2::3]
        # Compute theta from x,y
        theta = np.arctan2(y, x)
        # sort primarily along z (axial), then theta
        node_sort_map = np.lexsort((theta, z))
    else:
        raise ValueError(f"Cannot detect geometry from number of nodes: {npts}")

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

def right_pcg(A, b, x0=None, rtol=1e-8, atol=1e-7, max_iter=1000, M=None):
    """
    Right-preconditioned Conjugate Gradient method (version from my scitech paper)

    Solves: A M^{-1} y = b, with x = M^{-1} y

    Parameters
    ----------
    A : callable or sparse matrix
        Linear operator A(x) or sparse matrix
    b : ndarray
        Right-hand side
    x0 : ndarray, optional
        Initial guess for x
    tol : float
        Convergence tolerance on ||r||
    max_iter : int
        Maximum iterations
    M : object, optional
        Preconditioner with method M.solve(x) ≈ M^{-1} x

    Returns
    -------
    x : ndarray
        Approximate solution
    """

    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Preconditioner application
    if M is None:
        Minv = lambda x: x
    else:
        Minv = lambda x: M.solve(x)

    # Initial residual
    r = b - (A @ x)
    norm_r0 = np.linalg.norm(r)

    if norm_r0 < atol:
        return x
    
    rz_old = None
    rz = None

    print(f"PCG iter 0: ||r|| = {norm_r0:.3e}")

    for k in range(1, max_iter + 1):

        z = Minv(r)
        rz = np.dot(r, z)

        if k == 1:
            p = z.copy()
        else:
            beta = rz / rz_old
            p = z + beta * p_old

        p_old = p.copy()
        rz_old = rz * 1.0

        w = A @ p
        alpha = rz / np.dot(w, p)
        x += alpha * p
        r -= alpha * w
        norm_r = np.linalg.norm(r)

        if (k % 10 == 0) or norm_r < (atol + rtol * norm_r):
            print(f"PCG iter {k}: ||r|| = {norm_r:.3e}")

        if norm_r < (atol + rtol * norm_r):
            break

    print(f"PCG finished at iter {k}, ||r|| = {norm_r:.3e}")
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

def gen_plate_mesh(nxe:int=10, lx:float=1.0, ly:float=1.0, name="plate", apply_bcs:bool=True):
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
                if apply_bcs:
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
    if apply_bcs:
        output_file = name + ".bdf"
    else:
        output_file = name + "_nobc.bdf"

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

def gen_cylinder_mesh(nel_circ:int=12, nel_axial:int=10, radius:float=1.0, length:float=2.0,
                      name="cylinder", apply_bcs:bool=True):
    """
    Generate a cylinder mesh (quadrilateral shell) like gen_plate_mesh.
    Last hoop wraps around to the first.
    Ends fully clamped (all 6 DOFs = 0).
    """

    # Nodes
    nnode_circ = nel_circ
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

        # Write BCs
        if apply_bcs:
            for node in bcnodes:
                write_bulk_line("SPC", [1, node, "123456", 0.0])

        write_80(fout, "ENDDATA")

    print(f"Cylinder mesh written to {output_file} with {nnodes} nodes and {len(elements)} elements.")

