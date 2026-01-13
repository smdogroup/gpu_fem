from __ilu import gaussJordan, _get_diagp, block_ilu6_gj_solve
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

def get_rigid_body_modes(xpts, node_bcs, th:float=1.0):
    """get rigid body modes of particular mesh"""

    _x = xpts[0::3]; _y = xpts[1::3]; _z = xpts[2::3]
    nnodes = _x.shape[0]

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
    # so that no penalty done for them..
    # for bc_node in node_bcs:
    #     Bpred[bc_node, :, :] = 0.0
    # actually no it's not like setting F = 0 the other way (it's an outer product not inner)

    return Bpred

def get_node_bcs(A_bsr):
    """get node bcs for rigid body modes"""
    diagp = _get_diagp(A_bsr)
    nnodes = A_bsr.shape[0] // 6
    node_bcs = []
    for i in range(nnodes):
        jp = diagp[i]
        diag_block = A_bsr.data[jp]
        diag_vals = np.diag(diag_block)
        is_ones_vec = np.array([abs(diag_vals[i] - 1.0) < 1e-4 for i in range(diag_vals.shape[0])])
        is_bc = np.any(is_ones_vec)
        if is_bc:
            node_bcs += [i]
    return np.array(node_bcs)


def block_milu6_factor(A_bsr, perm_xpts, scale:float=1e0):
    """
    Modified ILU factorization based on Saad chapter 7
        * Block ILU(0) Gauss-jordan for 6x6 block BSR matrix
    """

    # print(f"{A_bsr.shape=}")
    # print(f"factor: {A_bsr.indptr=}\n{A_bsr.indices=}\n")

    # preamble (allocation)
    assert isinstance(A_bsr, sp.sparse.bsr_matrix)
    assert A_bsr.blocksize == (6,6)
    A_copy = A_bsr.copy()
    rowp = A_copy.indptr
    cols = A_copy.indices
    data = A_copy.data
    nnodes = A_copy.shape[0] // 6
    # null_val = 0
    null_val = -1 # prob should be -1
    iw = np.full(nnodes, null_val, dtype=int)
    diagp = _get_diagp(A_bsr)

    # rigid body mode constraints
    node_bcs = get_node_bcs(A_bsr)
    B = get_rigid_body_modes(perm_xpts, node_bcs, th=1.0)

    # DEBUG check this is actually teh rigid body modes..
    # need nobc to check this properly (should just be much smaller rhs than with B)
    # B_nobc = get_rigid_body_modes(perm_xpts, np.array([]), th=1.0)
    # to truly check and ensure zero rigid body modes.. would need to have A matrix without BCs enforced (or subset of A matrix)
    # for i in range(6):
    #     bvec1 = B[:,:,i].reshape((6*nnodes))
    #     bvec2 = B_nobc[:,:,i].reshape((6 * nnodes))
    #     force1 = A_bsr.dot(bvec1)
    #     force2 = A_bsr.dot(bvec2)
    #     tot1 = np.sum(force1)
    #     tot2 = np.sum(force2)
    #     print(f"rigid body mode {i=} with {tot1=:.4e} vs {tot2=:.4e} force with and without BC adjustment\n")
    # exit()

    # print(f"{B.shape=}")
    # exit()
        
    # ILU(0) with Gauss-Jordan solves (based on Saad chapter 7)
    for k in range(nnodes):
        j1 = rowp[k]
        j2 = rowp[k+1] - 1
    
        for j in range(j1, j2+1):
            iw[cols[j]] = j
        
        # also based on ilu_generic_template.h
        # which is easier to read

        j = j1 # lower triangular iteration here
        while (j <= j2):
            jrow = cols[j]
            if (jrow >= k): 
                break
            else:
                # make a temp matrix
                # tmat = np.zeros((6,6), dtype=np.double)
                tmat = data[j] @ data[diagp[jrow]]

                if cols[j] < nnodes:
                    data[j] = tmat.copy()

                # upper triangular iteration
                for jj in range(diagp[jrow] + 1, rowp[jrow+1]):
                    # rowp to cols jj
                    jw = iw[cols[jj]]

                    if jw != null_val:
                        if cols[jw] < nnodes and cols[jj] < nnodes:
                            data[jw] -= tmat @ data[jj]

            j += 1
        # done with matmult loop in crout ILU? is this crout ILU?

        diagp[k] = j
        if jrow != k:
            print(f"zero pivot {k=} {jrow=} {j=} in ILUGJ stopping\n")
            return
        
        diag_block = data[j].copy()

        # -------------------------------------------------
        # Compute RBM residual R_B
        # -------------------------------------------------
        A_B = np.zeros((6,6))
        LU_B = np.zeros((6,6))

        for mp in range(A_bsr.indptr[k], A_bsr.indptr[k+1]):
            m = A_bsr.indices[mp]
            A_B += A_bsr.data[mp] @ B[m]

            if m < k:  # lower triangular
                for qp in range(j+1, rowp[m+1]):
                    q = cols[qp]
                    LU_B += A_copy.data[mp] @ A_copy.data[qp] @ B[q]

        # diagonal contribution (L = I on diagonal)
        LU_B += diag_block @ B[k]

        # think sign might be wrong here?.. and thus it is boosting diagonals.. TBD
        R_B = A_B - LU_B
        # R_B = LU_B - A_B

        # scale = 0.5
        # scale = 0.1
        # scale = 0.0

        # either one is mostly fine.. row-distribute one not working as well as diag modification..
        # just like MILU scalar DOF per node uses 
        # distribute_upper = True
        distribute_upper = False

        # -------------------------------------------------
        # RBM correction
        # -------------------------------------------------
        Bk = B[k]
        if k not in node_bcs:
        # if True:
            G = Bk @ Bk.T
            if np.linalg.cond(G) < 1e8:

                # right RBM projector
                P = Bk.T @ np.linalg.inv(G)
                dD = scale * (R_B @ P)     # <-- this is Δ

                # -----------------------------------------
                # DISTRIBUTED CORRECTION
                # -----------------------------------------
                if distribute_upper:
                    # pass
                    # # this part is NOT CORRECT yet..

                    # count diag + strict upper
                    n_upper = rowp[k+1] - (j + 1)
                    N = 1 + n_upper

                    dD_scaled = dD / N

                    # diagonal update
                    diag_block -= dD_scaled

                    # free to do this because rest of matrix hasn't been factored yet..

                    # strict upper updates
                    for qp in range(j+1, rowp[k+1]):
                        # U_kq -= dD/N
                        A_copy.data[qp] -= dD_scaled

                else:
                    # classic diagonal-only MILU
                    diag_block -= dD
        # print(f"{diag_block.shape=}")

        # now do diagonal factor with gauss-jordan inverse 6x6 block
        # so forward + backward solves are faster to do..
        data[j] = np.eye(6).astype(A_bsr.data.dtype)
        fail, _ = gaussJordan(diag_block, data[j])
        if fail:
            print(f"gaussJordan solve failed on node block {j=} in ILUGJ\n")
            return
        
        # reset iw pointer
        for j in range(j1, j2+1):
            iw[cols[j]] = null_val

    return A_copy

class MILU_GJ_BlockPrecond:
    def __init__(self, A, perm_xpts, scale:float=1.0):
        self.A = block_milu6_factor(A.copy(), perm_xpts, scale)

    def solve(self, rhs):
        x = np.zeros_like(rhs)
        block_ilu6_gj_solve(self.A, x, rhs)
        return x
    
# ==========================================
# Block ILU(k) SVD(alpha) for near singular systems
# ==========================================



def svd_pert_inverse(A, RHS, alpha=1e-12):
    """
    Compute pseudo-inverse of 6x6 matrix A with singular values thresholded by alpha*sigma1.
    Result stored in RHS.
    """
    U, s, VT = np.linalg.svd(A)
    sigma1 = s[0]

    # Threshold singular values
    s_thresh = np.maximum(s, alpha * sigma1)
    # s_thresh = np.minimum(s, alpha * sigma1)

    # Invert singular values
    s_inv = 1.0 / s_thresh

    # Reconstruct pseudo-inverse
    np.dot(VT.T * s_inv, U.T, out=RHS)


def block_ilu6_svdpert_factor(A_bsr, alpha:float=0.1):
    """block-ILU with SVD singular value perturbations, see page 23 of paper https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf
    supposed to handle and precondition near singular systems better"""

    # same as block_ilu6_gj_factor from other script, uses Saad Chapter 7 ILU(0)-factor

    # preamble (allocation)
    assert isinstance(A_bsr, sp.sparse.bsr_matrix)
    assert A_bsr.blocksize == (6,6)
    A_copy = A_bsr.copy()
    rowp = A_copy.indptr
    cols = A_copy.indices
    data = A_copy.data
    nnodes = A_copy.shape[0] // 6
    # null_val = 0
    null_val = -1 # prob should be -1
    iw = np.full(nnodes, null_val, dtype=int)
    diagp = _get_diagp(A_bsr)
    
        
    # ILU(0) with Gauss-Jordan solves based on saad chapter 7
    for k in range(nnodes):
        j1 = rowp[k]
        j2 = rowp[k+1] - 1
    
        for j in range(j1, j2+1):
            iw[cols[j]] = j
        
        # also based on ilu_generic_template.h
        # which is easier to read

        j = j1 # lower triangular iteration here
        while (j <= j2):
            jrow = cols[j]
            if (jrow >= k): 
                break
            else:
                # make a temp matrix
                # tmat = np.zeros((6,6), dtype=np.double)
                tmat = data[j] @ data[diagp[jrow]]

                if cols[j] < nnodes:
                    data[j] = tmat.copy()

                # upper triangular iteration
                for jj in range(diagp[jrow] + 1, rowp[jrow+1]):
                    # rowp to cols jj
                    jw = iw[cols[jj]]

                    if jw != null_val:
                        if cols[jw] < nnodes and cols[jj] < nnodes:
                            data[jw] -= tmat @ data[jj]

            j += 1
        # done with matmult loop in crout ILU? is this crout ILU?

        diagp[k] = j
        if jrow != k:
            print(f"zero pivot {k=} {jrow=} {j=} in ILUGJ stopping\n")
            return

        # now do diagonal factor with gauss-jordan inverse 6x6 block
        tmat = data[j].copy()
        data[j] *= 0.0

        svd_pert_inverse(tmat, data[j], alpha)
        
        # reset iw pointer
        for j in range(j1, j2+1):
            iw[cols[j]] = null_val

    return A_copy

class BILU_SVD_Precond:
    # block ILU(0)-SVD preconditioner (if more fillin in original pattern, equiv to ILU(k))
    def __init__(self, A, alpha:float=0.1):
        self.A = block_ilu6_svdpert_factor(A.copy(), alpha)

    def solve(self, rhs):
        x = np.zeros_like(rhs)
        block_ilu6_gj_solve(self.A, x, rhs)
        return x
    