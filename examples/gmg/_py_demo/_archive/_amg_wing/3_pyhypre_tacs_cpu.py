import numpy as np
import os
import pyamg
from _utils import get_tacs_matrix
from scipy.sparse.linalg import aslinearoperator
from petsc4py import PETSc
import sys

# running with this supposed to set block size to 6
# but don't think that works (it doesn't let you set block size from python, only C++)
# python 3_pyhypre_tacs_cpu.py -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_block_size 6

if __name__=="__main__":
    import scipy.sparse.linalg as spla
    import time

    # get TACS matrix and convert to CSR since pyAMG only supports CSR
    # (this may mean it is slower or less accurate than a BSR version)
    # -----------------------------------
    tacs_bsr_mat, rhs, xpts = get_tacs_matrix()
    tacs_csr_mat = tacs_bsr_mat.tocsr()
    
    blocksize = 6
    A_csr = tacs_csr_mat
    b = rhs

    # solve with pyHypre
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    n = A_csr.shape[0]

    # Set global options before creating the PC
    # this doesn't work in python (the BSR arg), so it's just solving it in TACS CPU
    opts = PETSc.Options()
    opts['pc_type'] = 'hypre'
    opts['pc_hypre_type'] = 'boomeramg'
    # opts['pc_hypre_boomeramg_block_size'] = 6  # This sets the block size

    # start making matrix and precond
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape,
                                     csr=(indptr, indices, data),
                                     comm=PETSc.COMM_WORLD)
    A_petsc.setUp()
    A_petsc.assemble()

    b_petsc = PETSc.Vec().createWithArray(b)
    x_petsc = PETSc.Vec().createSeq(n)

    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setType('cg')  # or 'gmres' if needed

    # pc = ksp.getPC()
    # pc.setType('hypre')
    # pc.setHYPREType('boomeramg')
    # pc.setHYPREBoomerAMGBlockSize(blocksize)
    # pc.setHYPREBoomerAMGRelaxType("block")

    ksp.setTolerances(rtol=1e-8)
    ksp.setFromOptions()

    start_solve = time.time()
    ksp.solve(b_petsc, x_petsc)
    solve_time = time.time() - start_solve

    print(f"pyHypre+petSc BSRMat solve in {solve_time} sec")

    x_np = x_petsc.getArray()
    residual = np.linalg.norm(A_csr @ x_np - b)
    print(f"{residual=:.3e}")