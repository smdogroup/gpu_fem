import numpy as np
from scipy.sparse import csr_matrix, bsr_matrix

def get_rigid_body_modes(xpts, bcs, th:float=1.0):
    """get rigid body modes of particular mesh"""

    _x = xpts[0::3]; _y = xpts[1::3]; _z = xpts[2::3]
    nnodes = _x.shape[0]

    node_bcs = bcs // 6
    # print(f"{nnodes=} {node_bcs=}")

    Bpred = np.zeros((nnodes, 6, 6))

    # first three modes just as translation
    for imode in range(3):
        Bpred[:, imode, imode] = 1.0

    # u then v disp (yeah this one doesn't work cause of drill strain penalty)
    # so are there really only five modes?
    Bpred[:, 0, 3] = th * _y
    Bpred[:, 1, 3] = -th * _x
    # ahh => correction from drill strain = 2 * thz - (du/dy - dv/dx) = 2 * thz - omega
    # is to compute constant thz everywhere equal to the rotation magnitude => thz = th prescribed
    Bpred[:, 5, 3] = -th

    # v and w disp
    Bpred[:, 1, 4] = -th * _z
    Bpred[:, 2, 4] = th * _y
    # ah but then need to adjust thx or thy disp grads for trv shear error
    Bpred[:, 3, 4] = th

    # u and w disp
    Bpred[:, 0, 5] = th * _z
    Bpred[:, 2, 5] = -th * _x
    # and then need to adjust thy for dw/dx trv shear strain
    Bpred[:, 4, 5] = th

    # now zero out all bc nodes (full 6x6 block for each of those nodes)
    # for bc_node in node_bcs:
    #     Bpred[bc_node, :, :] = 0.0
    # actually no it's not like setting F = 0 the other way (it's an outer product not inner)

    return Bpred

def block_jacobi_smooth_operator(A: bsr_matrix, P: bsr_matrix, omega=1.0):
    """
    Apply block-Jacobi smoother:
        P_new = (I - omega * D^{-1} A) P
    """
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nrows = A.shape[0] // ndof

    # Extract D^{-1}
    Dinv = []
    for i in range(nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[p] == i:
                Dinv.append(np.linalg.inv(A.data[p]))
                break
        else:
            raise ValueError(f"Missing diagonal block at row {i}")

    # Compute A * P
    AP = A @ P

    # Scale rows of AP by D^{-1}
    AP_scaled = AP.copy()
    for i in range(nrows):
        for p in range(AP.indptr[i], AP.indptr[i + 1]):
            AP_scaled.data[p] = Dinv[i] @ AP.data[p]

    # P_new = P - omega * Dinv * A * P
    P_new = P - omega * AP_scaled
    return P_new

def block_jacobi_smooth_operator_inplace(A, P, omega=1.0):
    """
    Block-Jacobi smoothing of P:
        P <- (I - omega D^{-1} A) P
    keeping P sparsity.
    """
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nrows = A.shape[0] // ndof

    # Extract D^{-1}
    Dinv = [None] * nrows
    for i in range(nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[p] == i:
                Dinv[i] = np.linalg.inv(A.data[p])
                break

    Pnew = P.copy()

    for i in range(nrows):
        for pidx in range(P.indptr[i], P.indptr[i + 1]):
            k = P.indices[pidx]

            accum = np.zeros((ndof, ndof))
            for aidx in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[aidx]

                # find P_{j,k}
                for q in range(P.indptr[j], P.indptr[j + 1]):
                    if P.indices[q] == k:
                        accum += A.data[aidx] @ P.data[q]
                        break

            Pnew.data[pidx] -= omega * (Dinv[i] @ accum)

    return Pnew

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

def gauss_seidel_csr_transpose(A, b, x0, num_iter=1):
    """
    Backward (transpose) Gauss-Seidel smoothing for Ax = b
    A: csr_matrix (assumed square)
    b: RHS vector
    x0: initial guess
    num_iter: number of smoothing iterations
    Returns: updated solution x
    """
    x = x0.copy()
    n = A.shape[0]

    for _ in range(num_iter):
        for i in range(n - 1, -1, -1):
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


def orthog_nullspace_projector(P: csr_matrix, B:np.ndarray, bcs:np.ndarray):
    """apply the orthogonal projector to prevent nullspace modes"""

    node_bcs = np.unique(bcs // 6)

    # all nodes with bcs (exterior nodes will have zero update to that block-node of P, the whole row)
    # so we can just skip that and zero out (then ignore F_i part, cause F_i = I in interior fine nodes)

    Pnew = bsr_matrix((P.data.copy(), P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)
    # don't need to zero it out actually

    ndof = 6
    nblocks_row = P.shape[0] // ndof
    nblocks_col = P.shape[1] // ndof

    # loop over each fine node
    for brow in range(nblocks_row):
        # no actually don't skip F_i = 0 fully only in special case..
        # if brow in node_bcs: continue # skip this block row then (keep it unchanged)

        PU = np.zeros((6, 6))
        UTU = np.zeros((6, 6))

        Fi_bcs = np.array([_ for _ in range(6) if (6*brow+_) in bcs])
        Fi = np.eye(6)
        if Fi_bcs.shape[0] > 0:
            Fi[Fi_bcs,:] = 0.0
            # print(f"{Fi=}")

        # looping through the sparsity to compute sums first
        for jp in range(P.indptr[brow], P.indptr[brow+1]):
            bcol = P.indices[jp] # coarse node index

            U = B[bcol] @ Fi 
            
            PU += P.data[jp, :, :] @ U

            UTU += U.T @ U

        UTU_inv = np.linalg.pinv(UTU)
        # if not(brow in node_bcs):
            # print(f"{UTU_inv=}")

        # now loop back through removing projector from each P block (to apply projector)
        for jp in range(P.indptr[brow], P.indptr[brow+1]):
            bcol = P.indices[jp] # coarse node index

            U = B[bcol] @ Fi
            prev = Pnew.data[jp]
            delta = PU @ UTU_inv @ U.T
            final = prev - delta
            # print(f"{PU=}\n{UTU_inv=}\n{U.T=}\n\n")
            # print(f"{prev=} {-delta=} {final=}")
            Pnew.data[jp] -= PU @ UTU_inv @ U.T

    return Pnew

