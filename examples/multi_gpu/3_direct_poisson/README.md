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

Big ideas:
- how does BSR vs CSR methods compare
- we need a method that performs well if the matrices start out in a distributed storage format
- do inter-GPU matrix copies become very expensive for large problems? So they need to be assembled on each GPU and stay there mostly?
- interface Schur complement is highly dense but N = 6*n => 6*sqrt(n) DOF so much smaller matrix that it's ok, all partition K_{II}^{-1} become dense
    so S_{Gam,Gam}^i for each i partition is fully dense but only among *it's* interface nodes (not interface nodes it doesn't touch).


## Runtime results
