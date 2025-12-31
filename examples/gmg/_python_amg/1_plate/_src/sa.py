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


# def bgs_bsr_smoother(A: bsr_matrix, P: bsr_matrix, num_iter: int = 1, omega: float = 1.0):
#     """
#     Block Gauss–Seidel smoother: (L + D)^(-1) * P for BSR matrices with 6x6 blocks.
#     """
#     if not isinstance(A, bsr_matrix) or not isinstance(P, bsr_matrix):
#         raise TypeError("A and P must be bsr_matrix")
#     if A.blocksize != (6, 6) or P.blocksize != (6, 6):
#         raise ValueError("A and P must have blocksize (6,6)")

#     ndof = 6
#     nblocks_row = A.shape[0] // ndof

#     Pnew = P.copy()

#     for _ in range(num_iter):
#         for i in range(nblocks_row):
#             # Find diagonal block
#             diag_idx = None
#             for kk in range(A.indptr[i], A.indptr[i+1]):
#                 if A.indices[kk] == i:
#                     diag_idx = kk
#                     break
#             if diag_idx is None:
#                 raise ValueError(f"No diagonal block in row {i}")
#             Aii = A.data[diag_idx]

#             # Compute update for row i
#             row_update = np.zeros_like(Pnew.data[P.indptr[i]:P.indptr[i+1]])
#             for kk in range(A.indptr[i], A.indptr[i+1]):
#                 j = A.indices[kk]
#                 if j >= i:  # only lower triangular part (L)
#                     continue
#                 Aij = A.data[kk]
#                 for kp in range(P.indptr[j], P.indptr[j+1]):
#                     k = P.indices[kp]
#                     # find matching block in row i of P
#                     for kpi in range(P.indptr[i], P.indptr[i+1]):
#                         if P.indices[kpi] == k:
#                             row_update[kpi - P.indptr[i]] += Aij @ Pnew.data[kp]
#                             break

#             # Apply diagonal inverse
#             for local_idx, kp in enumerate(range(P.indptr[i], P.indptr[i+1])):
#                 Pnew.data[kp] = (1 - omega) * Pnew.data[kp] + omega * np.linalg.solve(Aii, Pnew.data[kp] - row_update[local_idx])

#     return Pnew

import numpy as np
from scipy.sparse import bsr_matrix

def bgs_bsr_smoother(A: bsr_matrix, P: bsr_matrix, num_iter: int = 1, omega: float = 1.0):
    """
    Block Gauss-Seidel forward solve for (L + D) X = P using 6x6 BSR blocks.
    Returns X with the same sparsity pattern as P (i.e. same indices/indptr).
    """
    if not isinstance(A, bsr_matrix) or not isinstance(P, bsr_matrix):
        raise TypeError("A and P must be scipy.sparse.bsr_matrix")
    if A.blocksize != (6, 6) or P.blocksize != (6, 6):
        raise ValueError("A and P must have blocksize (6,6)")

    ndof = 6
    nblocks_row = A.shape[0] // ndof

    # Build quick-access maps for A and P blocks
    A_rows = {i: [] for i in range(nblocks_row)}
    for i in range(nblocks_row):
        for aidx in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[aidx]
            A_rows[i].append((j, A.data[aidx]))   # (column-block-index, 6x6 block)

    # P_blocks: mapping (i,k) -> block (copy)
    P_blocks = {}
    for i in range(nblocks_row):
        for pidx in range(P.indptr[i], P.indptr[i+1]):
            k = P.indices[pidx]
            P_blocks[(i, k)] = P.data[pidx].copy()

    # Initialize X_blocks possibly from previous iteration (start zeros)
    X_blocks = {}  # mapping (i,k) -> block (6x6)

    for _it in range(num_iter):
        # forward sweep over block-rows
        for i in range(nblocks_row):
            # find diagonal Aii
            Aii = None
            for (j, Aij) in A_rows[i]:
                if j == i:
                    Aii = Aij
                    break
            if Aii is None:
                raise ValueError(f"No diagonal block found in A row {i}")
            
            print(f"brow {i=} => {Aii=}")

            # set of block-columns to compute in row i (only those present in P)
            cols_i = [k for (ii, k) in P_blocks.keys() if ii == i]
            if not cols_i:
                # nothing to do for this row (P has no blocks here) -> continue
                continue

            print(f"brow {i=} => {cols_i=}")

            for k in cols_i:
                # compute rhs = P_{i,k} - sum_{j<i} A_{i,j} * X_{j,k}
                rhs = P_blocks.get((i, k), np.zeros((ndof, ndof)))
                # accumulate contributions from previously solved rows j < i
                for (j, Aij) in A_rows[i]:
                    if j >= i:
                        continue
                    x_jk = X_blocks.get((j, k))
                    if x_jk is None:
                        # if X_{j,k} was never present (sparsity), treat as zero
                        continue
                    rhs = rhs - (Aij @ x_jk)

                # solve Aii * x = rhs
                x_new = np.linalg.solve(Aii, rhs)

                # relaxation with any previous X value (Gauss-Seidel omega)
                x_old = X_blocks.get((i, k), np.zeros_like(x_new))
                X_blocks[(i, k)] = (1.0 - omega) * x_old + omega * x_new

    # assemble new data array preserving P's sparsity order
    new_data = np.empty_like(P.data)
    for i in range(nblocks_row):
        pstart = P.indptr[i]
        pend = P.indptr[i+1]
        for pidx in range(pstart, pend):
            k = P.indices[pidx]
            block = X_blocks.get((i, k))
            if block is None:
                # If no computed block (shouldn't happen unless P had none), set zeros
                block = np.zeros((ndof, ndof))
            new_data[pidx] = block

    Pnew = bsr_matrix((new_data, P.indices.copy(), P.indptr.copy()),
                      shape=P.shape, blocksize=P.blocksize)
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

