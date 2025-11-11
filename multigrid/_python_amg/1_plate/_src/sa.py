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
    Bpred[:, 1, 4] = th * _z
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

def bgs_bsr_smoother_v1(A: bsr_matrix, P: bsr_matrix, num_iter: int = 1, omega:float=1.0):
    """
    Apply block Gauss-Seidel (L + D)^(-1) to a BSR matrix P.
    Both A and P are assumed to be 6x6 block-structured.
    
    Parameters
    ----------
    A : bsr_matrix, shape (N, N)
        Block sparse matrix with blocksize (6, 6)
    P : bsr_matrix, shape (N, M)
        Block sparse matrix with blocksize (6, 6)
    num_iter : int
        Number of forward sweeps (usually 1)
        
    Returns
    -------
    Pnew : bsr_matrix
        Result of (L + D)^(-1) * P, same sparsity as P
    """
    if not isinstance(A, bsr_matrix) or not isinstance(P, bsr_matrix):
        raise TypeError("A and P must be bsr_matrix")
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nblocks_row = A.shape[0] // ndof
    nblocks_col = P.shape[1] // ndof

    # Copy P to preserve input
    Pnew = bsr_matrix((P.data.copy(), P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)

    for it in range(num_iter):
        for i in range(nblocks_row):
            # Extract diagonal block of A
            diag_block_idx = None
            for k in range(A.indptr[i], A.indptr[i+1]):
                if A.indices[k] == i:
                    diag_block_idx = k
                    break
            Aii = A.data[diag_block_idx]

            # Loop over column blocks in P
            for jblk in range(nblocks_col):
                # Initialize RHS block
                rhs_block = np.zeros((ndof, ndof))

                # Extract RHS from Pnew (only existing block)
                block_idx = None
                for k in range(Pnew.indptr[i], Pnew.indptr[i+1]):
                    if Pnew.indices[k] == jblk:
                        block_idx = k
                        break
                if block_idx is not None:
                    rhs_block[:, :] = Pnew.data[block_idx]

                # Subtract contributions from previously computed lower blocks
                for k in range(A.indptr[i], A.indptr[i+1]):
                    j = A.indices[k]
                    if j < i:
                        Aij = A.data[k]
                        # Extract P block at row j, column jblk
                        row_ptr_start, row_ptr_end = Pnew.indptr[j], Pnew.indptr[j+1]
                        P_block_idx = None
                        for kk in range(row_ptr_start, row_ptr_end):
                            if Pnew.indices[kk] == jblk:
                                P_block_idx = kk
                                break
                        if P_block_idx is not None:
                            P_block = Pnew.data[P_block_idx]
                        else:
                            P_block = np.zeros((ndof, ndof))

                        rhs_block -= omega * Aij @ P_block

                # Solve small 6x6 system
                X_block = np.linalg.solve(Aii, rhs_block)

                # Write back to Pnew
                if block_idx is not None:
                    Pnew.data[block_idx][:, :] = X_block

    return Pnew

def bgs_bsr_smoother_v2(A: bsr_matrix, P: bsr_matrix, num_iter: int = 1, omega:float=1.0):
    """
    Apply block Gauss-Seidel (L + D)^(-1) to a BSR matrix P.
    Both A and P are assumed to be 6x6 block-structured.
    
    Parameters
    ----------
    A : bsr_matrix, shape (N, N)
        Block sparse matrix with blocksize (6, 6)
    P : bsr_matrix, shape (N, M)
        Block sparse matrix with blocksize (6, 6)
    num_iter : int
        Number of forward sweeps (usually 1)
        
    Returns
    -------
    Pnew : bsr_matrix
        Result of (L + D)^(-1) * P, same sparsity as P
    """
    if not isinstance(A, bsr_matrix) or not isinstance(P, bsr_matrix):
        raise TypeError("A and P must be bsr_matrix")
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nblocks_row = A.shape[0] // ndof
    nblocks_col = P.shape[1] // ndof

    # Copy P to preserve input
    Pnew = bsr_matrix((P.data.copy(), P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)

    for it in range(num_iter):
        for i in range(nblocks_row):
            # Extract diagonal block of A
            diag_block_idx = None
            for k in range(A.indptr[i], A.indptr[i+1]):
                if A.indices[k] == i:
                    diag_block_idx = k
                    break
            Aii = A.data[diag_block_idx]

            # get list of columns k in P with current i row (for Dinv part)
            kp_list = []
            k_list = []
            for _kp in range(P.indptr[i], P.indptr[i+1]):
                _k = P.indices[_kp]
                kp_list += [_kp]
                k_list += [_k] 

            # loop over the columns of A (subtracting stuff)
            for jp in range(A.indptr[i], A.indptr[i+1]):
                j = A.indices[jp]
                Aij = A.data[jp]

                # loop over blocks in P
                for kp in range(P.indptr[j], P.indptr[j+1]):
                    k = P.indices[kp]
                    if not(k in k_list): continue

                    # otherwise get _kp the diagonal index
                    for _i, _kp in enumerate(kp_list):
                        _k = k_list[_i]
                        

                    # subtract into the diagonal P_ik (diagonal entry here)



            for j in range(i-1):



                # get Aij
                Aij = A


            # Loop over column blocks in P
            for jblk in range(nblocks_col):
                # Initialize RHS block
                rhs_block = np.zeros((ndof, ndof))

                # Extract RHS from Pnew (only existing block)
                block_idx = None
                for k in range(Pnew.indptr[i], Pnew.indptr[i+1]):
                    if Pnew.indices[k] == jblk:
                        block_idx = k
                        break
                if block_idx is not None:
                    rhs_block[:, :] = Pnew.data[block_idx]

                # Subtract contributions from previously computed lower blocks
                for k in range(A.indptr[i], A.indptr[i+1]):
                    j = A.indices[k]
                    if j < i:
                        Aij = A.data[k]
                        # Extract P block at row j, column jblk
                        row_ptr_start, row_ptr_end = Pnew.indptr[j], Pnew.indptr[j+1]
                        P_block_idx = None
                        for kk in range(row_ptr_start, row_ptr_end):
                            if Pnew.indices[kk] == jblk:
                                P_block_idx = kk
                                break
                        if P_block_idx is not None:
                            P_block = Pnew.data[P_block_idx]
                        else:
                            P_block = np.zeros((ndof, ndof))

                        rhs_block -= omega * Aij @ P_block

                # Solve small 6x6 system
                X_block = np.linalg.solve(Aii, rhs_block)

                # Write back to Pnew
                if block_idx is not None:
                    Pnew.data[block_idx][:, :] = X_block

    return Pnew


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
        if not(brow in node_bcs):
            print(f"{UTU_inv=}")

        # now loop back through removing projector from each P block (to apply projector)
        for jp in range(P.indptr[brow], P.indptr[brow+1]):
            bcol = P.indices[jp] # coarse node index

            U = B[bcol] @ Fi
            prev = Pnew.data[jp]
            delta = PU @ UTU_inv @ U.T
            final = prev - delta
            print(f"{PU=}\n{UTU_inv=}\n{U.T=}\n\n")
            # print(f"{prev=} {-delta=} {final=}")
            Pnew.data[jp] -= PU @ UTU_inv @ U.T

    return Pnew

