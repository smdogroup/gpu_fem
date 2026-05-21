# Comparison of multi-GPU direct solvers

1. CuDSS direct solver
    - does it require copying matrices to root GPU?
    - requires CSR
    - docs for CuDSS: https://docs.nvidia.com/cuda/cudss/general.html
2. CuSparse BSR direct solver
    - no triangular solve applied to matrix routines
    - so need to use (LU)^-1 * v vector operations to assemble interface Schur complement? Is that extremely expensive for large problems?
3. CuSparse CSR direct solver
    - can we use triang solve to matrix operations to compute S_{Gam,Gam}^i for each partition? or is this tricky because we have to do it in two matrix steps and store intermediate matrix?


Other multi-GPU direct solvers?
4. CuSolver?
5. Trilinos?
6. Hypre?
7. SuperLU_dist?


Description of scripts for direct solvers:
1_1_cudss_mg: CuDSS multi-GPU and single-CPU with inter-GPU copy of matrix (inter-GPU copy of matrix and vectors actually not that bad) [uses CSR matrices]
1_2_cudss_mgmn: CuDSS with multi-GPU and multi-CPU (remains fully distributed with assigned rows), no inter-GPU copy of matrix [uses CSR matrices]
2_cusparse_bsr.cu: CuSparse LU solve with BSR matrices and full LU pattern

## Lessons learned towards BDDC subdomain-parallel solvers:
* inter-GPU copying of the matrix for CuDSSMG (multi-GPU single-node) is not that expensive (esp. since intended only for the coarse S_VV problem)
    * still could limit memory cost of S_VV since has to be copied back to root first (so must store on root too)
    * the reason you don't see ideal speedups on multi-GPU is actually that the decreased problem sizes underutilize the GPU (not inter-GPU communication), that was my misunderstanding before.
* CuDSSMGMN (multi-GPU and multi-node) is much slower than CuDSSMG since inter-CPU copying is slow? Why multi-CPU worse?
    * would allow fully distributed storage.. maybe could improve so no CPU copying (not sure if it does that)
* subdomain-parallel cudss solves shouldn't use CuDSSMG (cause has schur complement)
    * smaller problems on each GPU still don't give ideal speedup (cause underutilize GPU), even though no inter-GPU communication.



## OLD Comments (deprecated)


Big ideas:
- how does BSR vs CSR methods compare
- we need a method that performs well if the matrices start out in a distributed storage format
- do inter-GPU matrix copies become very expensive for large problems? So they need to be assembled on each GPU and stay there mostly?
- interface Schur complement is highly dense but N = 6*n => 6*sqrt(n) DOF so much smaller matrix that it's ok, all partition K_{II}^{-1} become dense
    so S_{Gam,Gam}^i for each i partition is fully dense but only among *it's* interface nodes (not interface nodes it doesn't touch).

