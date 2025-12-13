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

def gauss_seidel_csr(A, b, x0, num_iter=1):
    """
    Perform Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]
    for _ in range(num_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]
            x[i] = (b[i] - sum_) / diag
    return x

def block_gauss_seidel_6dof(A, b: np.ndarray, x0: np.ndarray, num_iter=1):
    """
    Perform Block Gauss-Seidel smoothing for 6 DOF per node.
    A: csr_matrix of size (6*nnodes, 6*nnodes)
    b: RHS vector (6*nnodes,)
    x0: initial guess (6*nnodes,)
    num_iter: number of smoothing iterations
    Returns updated solution vector x
    """
    x = x0.copy()
    ndof = 6
    n = A.shape[0] // ndof

    for it in range(num_iter):
        for i in range(n):
            row_block_start = i * ndof
            row_block_end = (i + 1) * ndof

            # Initialize block and RHS
            Aii = np.zeros((ndof, ndof))
            rhs = b[row_block_start:row_block_end].copy()

            for row_local, row in enumerate(range(row_block_start, row_block_end)):
                for idx in range(A.indptr[row], A.indptr[row + 1]):
                    col = A.indices[idx]
                    val = A.data[idx]

                    j = col // ndof
                    dof_j = col % ndof

                    col_block_start = j * ndof
                    col_block_end = (j + 1) * ndof

                    if j == i:
                        Aii[row_local, dof_j] = val  # Fill local diag block
                    else:
                        rhs[row_local] -= val * x[col]

            # Check for singular or ill-conditioned diagonal block
            try:
                # import matplotlib.pyplot as plt
                # plt.imshow(np.log(1 + Aii**2))
                # plt.show()
                # print(f"{Aii=}")
                x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
            except np.linalg.LinAlgError:
                print(f"Warning: singular block at node {i}, skipping update.")
                continue

    return x

# def block_gauss_seidel_6dof_v2(A, b: np.ndarray, x0: np.ndarray, num_iter=1):
#     """
#     Perform Block Gauss-Seidel smoothing for 6 DOF per node.
#     A: csr_matrix of size (6*nnodes, 6*nnodes)
#     b: RHS vector (6*nnodes,)
#     x0: initial guess (6*nnodes,)
#     num_iter: number of smoothing iterations
#     Returns updated solution vector x
#     """
#     x = x0.copy()
#     ndof = 6
#     n = A.shape[0] // ndof

#     for it in range(num_iter):
#         for i in range(n):
#             row_block_start = i * ndof
#             row_block_end = (i + 1) * ndof

#             # Initialize block and RHS
#             Aii = np.zeros((ndof, ndof))
#             rhs = b[row_block_start:row_block_end].copy()

#             for row_local, row in enumerate(range(row_block_start, row_block_end)):
#                 for idx in range(A.indptr[row], A.indptr[row + 1]):
#                     col = A.indices[idx]
#                     val = A.data[idx]

#                     j = col // ndof
#                     dof_j = col % ndof

#                     # col_block_start = j * ndof
#                     # col_block_end = (j + 1) * ndof

#                     if j == i and dof_j <= row_local:
#                         # print(f"{Aii=}")
#                         Aii[row_local, dof_j] = val  # Fill local diag block only in L + D parts
#                     elif i > j: # below-diag get to keep the full coupling cause that's just updates.. and still in L part
#                         rhs[row_local] -= val * x[col]

#             # Check for singular or ill-conditioned diagonal block
#             try:
#                 # zero strict upper-triang part..
#                 # print(f"{Aii=}")
#                 x[row_block_start:row_block_end] = np.linalg.solve(Aii, rhs)
#             except np.linalg.LinAlgError:
#                 print(f"Warning: singular block at node {i}, skipping update.")
#                 continue

#     return x

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

def mg_coarse_fine_operators_v1(nxe_fine, sort_bk_fine, sort_bk_coarse, bcs_list=None):
    """include a bit of reordering in the Ifc and Icf operators, also we first include then remove the bcs.."""
    # this v1 uses same stencil as Poisson's problem

    nxe_coarse = nxe_fine // 2
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    Nc = nxc**2
    Nf = nxf**2
    # assumes it does have bcs here..
    Nc_dof = Nc * 6
    Nf_dof = Nf * 6

    # first include all nodes, then we'll remove the 
    I_fc = np.zeros((Nc_dof, Nf_dof)) # coarse to fine

    for i_c in range(Nc_dof):
        # coarse point
        idof = i_c % 6
        inode_c = i_c // 6
        iyc = inode_c // nxc
        ixc = inode_c % nxc

        # corresponding fine point
        ixf = 2 * ixc
        iyf = 2 * iyc
        inode_f = nxf * iyf + ixf
        i_f = 6 * inode_f + idof

        # print(f"{i_c=}, [n,d]={inode_c=},{idof=}] c:[{ixc},{iyc}], {inode_f=} {i_f=}/{Nf=}")

        I_fc[i_c, i_f] = 4.0 / 16.0 # middle, middle

        if iyf > 0:
            I_fc[i_c, i_f-6*nxf] = 2.0 / 16.0 # middle, bottom

        if iyf < nxf-1:
            I_fc[i_c, i_f+6*nxf] = 2.0 / 16.0 # middle, top

        if ixf > 0:
            I_fc[i_c, i_f-6] = 2.0 / 16.0 # left, middle

            if iyf > 0:
                I_fc[i_c, i_f-6-6*nxf] = 1.0 / 16.0 # left, bottom

            if iyf < nxf-1:
                I_fc[i_c, i_f-6+6*nxf] = 1.0 / 16.0 # left, top
        
        if ixf < nxf-1:
            I_fc[i_c, i_f+6] = 2.0 / 16.0 # right, middle

            if iyf > 0:
                I_fc[i_c, i_f+6-6*nxf] = 1.0 / 16.0 # right, bottom

            if iyf < nxf-1:
                I_fc[i_c, i_f+6+6*nxf] = 1.0 / 16.0 # right, top

    # remove bcs and apply sorting (since the I_fc currently is perfectly sorted)
    I_fc_0 = I_fc.copy() # fine to coarse mapping

    nc_keep = np.sum(sort_bk_coarse != -1)
    nf_keep = np.sum(sort_bk_fine != -1)
    I_fc = np.zeros((nc_keep, nf_keep))

    # print(f"{nf_keep=} {Nf_dof=}")

    for i_c in range(Nc_dof):
        i2_c = sort_bk_coarse[i_c]
        if i2_c == -1: continue

        for i_f in range(Nf_dof):
            i2_f = sort_bk_fine[i_f]
            if i2_f == -1: continue

            # now copy from old mapping operator
            I_fc[i2_c, i2_f] = I_fc_0[i_c, i_f]

    # plt.imshow(I_fc)
    # plt.show()

    # remove bcs for case where we kept them
    if bcs_list is not None:
        bcs_fine = np.array(bcs_list[0])
        bcs_coarse = np.array(bcs_list[1])
        I_fc[bcs_coarse,:] *= 0.0
        I_fc[:,bcs_fine] *= 0.0
    
    # coarse to fine is transpose operator
    I_cf = I_fc.T * 4

    return I_cf, I_fc

def mg_coarse_fine_operators_v2(nxe_fine, sort_bk_fine, sort_bk_coarse, bcs_list=None):
    """include a bit of reordering in the Ifc and Icf operators, also we first include then remove the bcs.."""
    # uses the FEA Lagrange basis of quad elements (needs to match FEA basis to get disps coarse to fine with consistent loads)
    # 1st order interp is not sufficient to give accurate 2nd derivs or forces for bending (ahh, maybe axial would be fine, but not bending..)

    nxe_coarse = nxe_fine // 2
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    Nc = nxc**2
    Nf = nxf**2
    # assumes it does have bcs here..
    Nc_dof = Nc * 6
    Nf_dof = Nf * 6

    # first include all nodes, then we'll remove the 
    I_cf = np.zeros((Nf_dof, Nc_dof)) # coarse to fine

    for i_c in range(Nc_dof):
        # coarse point
        idof = i_c % 6
        inode_c = i_c // 6
        iyc = inode_c // nxc
        ixc = inode_c % nxc

        # corresponding fine point
        ixf = 2 * ixc
        iyf = 2 * iyc
        inode_f = nxf * iyf + ixf
        i_f = 6 * inode_f + idof

        # lagrange interpolated basis

        # fine and coarse node match
        I_cf[i_f, i_c] = 1.0

        if ixc < nxc - 1:
            # fine node half-way to right
            I_cf[i_f + 6, i_c] = 0.5
            I_cf[i_f + 6, i_c + 6] = 0.5

        if iyc < nxc - 1:
            # fine node half-way up
            I_cf[i_f + 6 * nxf, i_c] = 0.5
            I_cf[i_f + 6 * nxf, i_c + 6 * nxc] = 0.5  

        if ixc < nxc - 1 and iyc < nxc - 1:
            # fine node half to right and half up
            _if = i_f + 6 * nxf + 6
            I_cf[_if, i_c] = 0.25
            I_cf[_if, i_c + 6] = 0.25
            I_cf[_if, i_c + 6 * nxc] = 0.25
            I_cf[_if, i_c + 6 * nxc + 6] = 0.25  

    # remove bcs and apply sorting (since the I_fc currently is perfectly sorted)
    I_cf_0 = I_cf.copy() # fine to coarse mapping

    nc_keep = np.sum(sort_bk_coarse != -1)
    nf_keep = np.sum(sort_bk_fine != -1)
    I_cf = np.zeros((nf_keep, nc_keep))

    # print(f"{nf_keep=} {Nf_dof=}")

    for i_c in range(Nc_dof):
        i2_c = sort_bk_coarse[i_c]
        if i2_c == -1: continue

        for i_f in range(Nf_dof):
            i2_f = sort_bk_fine[i_f]
            if i2_f == -1: continue

            # now copy from old mapping operator
            I_cf[i2_f, i2_c] = I_cf_0[i_f, i_c]

    # plt.imshow(I_fc)
    # plt.show()

    # remove bcs for case where we kept them
    if bcs_list is not None:
        bcs_fine = np.array(bcs_list[0])
        bcs_coarse = np.array(bcs_list[1])
        I_cf[bcs_fine,:] *= 0.0
        I_cf[:,bcs_coarse] *= 0.0
    
    # coarse to fine is transpose operator
    I_fc = I_cf.T

    return I_cf, I_fc

def mg_coarse_fine_transv_shear_smooth(nxe_fine, fine_disp_0, h, sort_fw_arr):
    nxe_coarse = nxe_fine // 2
    # nx_coarse = nxe_coarse + 1
    nx_fine = nxe_fine + 1
    H = 2.0 * h # coarse mesh size
    fine_disp = np.zeros_like(fine_disp_0)
    fine_disp[sort_fw_arr] = fine_disp_0[:]

    # NOTE : please turn this off later => or figure out why 4x nodal diff
    # debug = True # debug to change and try and get rid of 4x diff
    debug = True
    # gam_fact = 0.5**0.5
    # gam_fact = 0.5
    gam_fact = 1.0
    # gam_fact = 2.0
    # gam_fact = 4.0


    def _shear_strain(w, th, sign:float=1.0, _dx:float=h):
        """helper method to compute shear strains (2 vals each in w and th) for fine vs coarse elements and gam_13, gam_23"""
        return (w[1] - w[0]) / _dx + sign * 0.5 * (th[0] + th[1]) # dw/dx here, same as 2/dx * dw/dxi, etc.

    def _oned_shear_interp(w, th, sign:float=1.0):
        # here w, th are length 3 vecs each (with interp middle value not set yet, and we set that)

        # any edge and either shear strain will use this interp method..
        gamma_c = _shear_strain(w=[w[0], w[2]], th=[th[0], th[2]], _dx=H, sign=sign) # coarse shear strain

        if debug:
            gamma_c *= gam_fact

        w_new = 0.5 * (w[0] + w[2]) + sign * 0.25 * h * (th[2] - th[0]) # from subtracting two eqns
        th_new = -th[0] + 2.0 * sign * gamma_c - 2.0 * sign * (w_new - w[0]) / h

        # print(f"{th_new=} {th[1]=}")
        return w_new, th_new

    def _gam13_interp(w, th):
        return _oned_shear_interp(w, th, sign=1.0)

    def _gam23_interp(w, th):
        return _oned_shear_interp(w, th, sign=-1.0)

    for iye_c in range(nxe_coarse):
        for ixe_c in range(nxe_coarse):

            # get all 9 fine nodes.. and their 3 DOF (27 DOF total)
            ixe_f, iye_f = 2 * ixe_c, 2 * iye_c
            inode_f = iye_f * nx_fine + ixe_f # bottom right fine node on coarse elem
            elem_f_nodes = [inode_f + _ix + nx_fine * _iy for _iy in range(3) for _ix in range(3)]
            elem_f_dof = np.array([6*_inode + _idof for _inode in elem_f_nodes for _idof in [2,3,4]])
            # print(f"{elem_f_nodes=} {elem_f_dof=}")
            # print(f"1: {fine_disp_v2[elem_f_dof]=}")
            elem_f_disp = fine_disp[elem_f_dof]
            f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
            f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
            f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
            f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
            f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

            """ solve boundary shear strain updates first => updates 8 DOF """
            
            # gam13 update on y- edge (south)
            _i, _j = elem_f_dof[4-1], elem_f_dof[6-1]

            # _w0, _th0 = fine_disp[_i], fine_disp[_j] # get for DEBUG

            fine_disp[_i], fine_disp[_j] = _gam13_interp(
                w=[f1, f4, f7],
                th=[f3, f6, f9],
            )

            # _w, _th = fine_disp[_i], fine_disp[_j] # get for DEBUG
            # if ixe_c == nxe_coarse-1 and iye_c == 0: # DEBUG since this didn't interp correctly
            #     w_list = [f1, f4, f7]
            #     th_list = [f3, f6, f9]
            #     print(f"{w_list=} {th_list=}")
            #     print(F"{_w0=:.3e} => {_w=:.3e}, {_th0=:.3e} => {_th=:.3e}")

            #     exit()

            # gam23 update on x- edge (west)
            _i, _j = elem_f_dof[10-1], elem_f_dof[11-1]
            fine_disp[_i], fine_disp[_j] = _gam23_interp(
                w=[f1, f10, f19],
                th=[f2, f11, f20],
            )

            # gam13 update on y+ edge (north)
            _i, _j = elem_f_dof[22-1], elem_f_dof[24-1]
            fine_disp[_i], fine_disp[_j] = _gam13_interp(
                w=[f19, f22, f25],
                th=[f21, f24, f27],
            )

            # gam23 update on x+ edge (east)
            _i, _j = elem_f_dof[16-1], elem_f_dof[17-1]
            fine_disp[_i], fine_disp[_j] = _gam23_interp(
                w=[f7, f16, f25],
                th=[f8, f17, f26],
            )

            # solve linear system for last rem DOF u13,u14,u15,u18 (to enforce gam13,gam23 zero on interior tying point lines)
            hi = 1.0 / h
            _A = np.array([
                [hi, 0, 0.5, 0],
                [-hi, 0, 0.5, 0.5],
                [hi, -0.5, 0, 0],
                [-hi, -0.5, 0, 0],
            ])

            # need to use coarse node data (which doesn't change here to get gamij RHS, fine disps in middle change and may mess up coarse gam)
            _gam_13_c_top = (f25 - f19) / H + 0.5 * (f21 + f27)
            _gam_13_c_bot = (f7 - f1) / H + 0.5 * (f9 + f3)
            _gam_13_mid = 0.5 * (_gam_13_c_top + _gam_13_c_bot)

            _gam_23_c_right = (f25 - f7) / H - 0.5 * (f26 + f8)
            _gam_23_c_left = (f19 - f1) / H - 0.5 * (f20 + f2)
            _gam_23_mid = 0.5 * (_gam_23_c_right + _gam_23_c_left)

            if debug:
                _gam_13_mid *= gam_fact
                _gam_23_mid *= gam_fact

            # print(f"{_gam_13_c_top=} {_gam_13_c_bot=}")

            # udpate disps again for rhs
            elem_f_disp = fine_disp[elem_f_dof]
            f1, f2, f3, f4, f5, f6 = elem_f_disp[0], elem_f_disp[1], elem_f_disp[2], elem_f_disp[3], elem_f_disp[4], elem_f_disp[5]
            f7, f8, f9, f10, f11, f12 = elem_f_disp[6], elem_f_disp[7], elem_f_disp[8], elem_f_disp[9], elem_f_disp[10], elem_f_disp[11]
            f13, f14, f15, f16, f17, f18 = elem_f_disp[12], elem_f_disp[13], elem_f_disp[14], elem_f_disp[15], elem_f_disp[16], elem_f_disp[17]
            f19, f20, f21, f22, f23, f24 = elem_f_disp[18], elem_f_disp[19], elem_f_disp[20], elem_f_disp[21], elem_f_disp[22], elem_f_disp[23]
            f25, f26, f27 = elem_f_disp[24], elem_f_disp[25], elem_f_disp[26]

            # print(F"{_gam_13_mid=} {_gam_23_mid=}")

            _b = np.array([
                _gam_13_mid + f10*hi - f12*0.5,
                _gam_13_mid - f16*hi,
                _gam_23_mid + f4*hi + 0.5*f5,
                _gam_23_mid - f22*hi + 0.5*f23,
            ])

            _x = np.linalg.solve(_A, _b)
            # sets f13, f14, f15, f18 now..
            for _i, _j in enumerate([12, 13, 14, 17]):
                _j2 = elem_f_dof[_j]
                _orig = fine_disp[_j2]
                fine_disp[_j2] = _x[_i]

                # if _j in [13, 14, 17]:
                #     print(f"{_orig=:.3e} => {fine_disp[_j2]=:.3e}")

    # then inv sort to output
    fine_disp_out = fine_disp[sort_fw_arr].copy()

    return fine_disp_out

def mg_coarse_fine_operators_v3(nxe_fine, sort_bk_fine, sort_bk_coarse, bcs_list=None):
    """
    Uses quadratic Lagrange interpolation (2nd order) from coarse to fine for quad shell elements with 6 DOFs per node.
    Improves accuracy of interpolated fine displacements and resulting fine nodal forces.
    """
    nxe_coarse = nxe_fine // 2
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    Nc = nxc**2
    Nf = nxf**2
    Nc_dof = Nc * 6
    Nf_dof = Nf * 6

    I_cf_full = np.zeros((Nf_dof, Nc_dof))

    def quadratic_weights(xi):
        """Lagrange weights for xi in [0, 1]"""
        return np.array([0.5 * xi * (xi - 1), (1 - xi**2), 0.5 * xi * (xi + 1)])

    for iyf in range(nxf):
        for ixf in range(nxf):
            inode_f = iyf * nxf + ixf
            for idof in range(6):
                i_f = 6 * inode_f + idof

                # physical coordinates in coarse mesh units
                x_coarse = ixf / 2.0
                y_coarse = iyf / 2.0

                # center index of the stencil in coarse grid
                ixc = int(np.floor(x_coarse))
                iyc = int(np.floor(y_coarse))

                xi = x_coarse - ixc
                eta = y_coarse - iyc

                wx = quadratic_weights(xi)
                wy = quadratic_weights(eta)

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        jxc = ixc + dx
                        jyc = iyc + dy
                        if 0 <= jxc < nxc and 0 <= jyc < nxc:
                            inode_c = jyc * nxc + jxc
                            i_c = 6 * inode_c + idof

                            wx_idx = dx + 1
                            wy_idx = dy + 1
                            weight = wx[wx_idx] * wy[wy_idx]

                            I_cf_full[i_f, i_c] += weight

    # Apply sorting and BCs
    nc_keep = np.sum(sort_bk_coarse != -1)
    nf_keep = np.sum(sort_bk_fine != -1)
    I_cf = np.zeros((nf_keep, nc_keep))

    for i_c in range(Nc_dof):
        i2_c = sort_bk_coarse[i_c]
        if i2_c == -1:
            continue
        for i_f in range(Nf_dof):
            i2_f = sort_bk_fine[i_f]
            if i2_f == -1:
                continue
            I_cf[i2_f, i2_c] = I_cf_full[i_f, i_c]

    if bcs_list is not None:
        bcs_fine = np.array(bcs_list[0])
        bcs_coarse = np.array(bcs_list[1])
        I_cf[bcs_fine, :] *= 0.0
        I_cf[:, bcs_coarse] *= 0.0

    I_fc = I_cf.T
    return I_cf, I_fc

def mg_coarse_fine_operators_v4(nxe_fine, sort_bk_fine, sort_bk_coarse, bcs_list=None):
    """
    Uses quartic Lagrange interpolation (4th order) from coarse to fine for quad shell elements with 6 DOFs per node.
    Accurate for bending-dominated or thin shell cases.
    """
    nxe_coarse = nxe_fine // 2
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    Nc = nxc**2
    Nf = nxf**2
    Nc_dof = Nc * 6
    Nf_dof = Nf * 6

    I_cf_full = np.zeros((Nf_dof, Nc_dof))

    def lagrange_weights(x_nodes, x_eval):
        n = len(x_nodes)
        w = np.ones(n)
        for j in range(n):
            for k in range(n):
                if j != k:
                    w[j] *= (x_eval - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
        return w

    for iyf in range(nxf):
        for ixf in range(nxf):
            inode_f = iyf * nxf + ixf
            for idof in range(6):
                i_f = 6 * inode_f + idof

                # fine node in coarse grid coordinates
                x_fine = ixf / 2.0
                y_fine = iyf / 2.0

                ixc = int(np.floor(x_fine)) - 2  # 5-point stencil centered
                iyc = int(np.floor(y_fine)) - 2

                x_nodes = [ixc + i for i in range(5)]
                y_nodes = [iyc + j for j in range(5)]

                wx = lagrange_weights(x_nodes, x_fine)
                wy = lagrange_weights(y_nodes, y_fine)

                for dy in range(5):
                    for dx in range(5):
                        jxc = x_nodes[dx]
                        jyc = y_nodes[dy]
                        if 0 <= jxc < nxc and 0 <= jyc < nxc:
                            inode_c = jyc * nxc + jxc
                            i_c = 6 * inode_c + idof
                            weight = wx[dx] * wy[dy]
                            I_cf_full[i_f, i_c] += weight

    # Apply sorting and BCs
    nc_keep = np.sum(sort_bk_coarse != -1)
    nf_keep = np.sum(sort_bk_fine != -1)
    I_cf = np.zeros((nf_keep, nc_keep))

    for i_c in range(Nc_dof):
        i2_c = sort_bk_coarse[i_c]
        if i2_c == -1:
            continue
        for i_f in range(Nf_dof):
            i2_f = sort_bk_fine[i_f]
            if i2_f == -1:
                continue
            I_cf[i2_f, i2_c] = I_cf_full[i_f, i_c]

    if bcs_list is not None:
        bcs_fine = np.array(bcs_list[0])
        bcs_coarse = np.array(bcs_list[1])
        I_cf[bcs_fine, :] *= 0.0
        I_cf[:, bcs_coarse] *= 0.0

    I_fc = I_cf.T
    return I_cf, I_fc
