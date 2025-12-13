
## 3/3/2025 meeting notes (TBD)
Implement AD for more robust MELD implementation
Port the TACS uCRM system onto the GPU and see if it solves
Max and minimum entries in the system besides the zeros (compare against absolute)
Q-ordering slower per iteration
Try the 100x100 case to see the convergence profile
Converges of GMRES with restarts
NAS charge number openMDAO group for free use of H100
RBF fun2fem once you have interpolation matrix should be straight forward to parallelize
19th of May to meet Kevin at NASA
prioritize getting geom NL solve, first with direct LU, then with iterative
then work on MELD again, try AD of SVD 3x3 to get nonlinear MELD to work


## 2/3/2025 meeting notes
Reordering -> Nested dissection –> METIS will be much more efficient
76% compute throughput because you are memory-bound
Speed of light not as important
Prioritize nonlinear solve over MDAO stuff before May
Linear solver is the bottle neck
Q-ordering on ILU -> GMRES -> Read the paper by Kevin [Node Numbering for Stabilizing Preconditioners Based on Incomplete LU Decomposition](https://arc.aiaa.org/doi/epdf/10.2514/6.2020-3022)
RCM Ordering, Cuthill Mckee ordering in TACS utilities cpp
Q ordering + ILU -> speed up? -> May help with nonlinear solves
Direct factorization acts as a fallback or basline
Look in TACS for METIX nested dissections TACS Assembler
Prioritize nonlinear structures solve over the optimization (adjoint, dRdx kernels, etc.) [Kevin]
Prioritize speeding up solver over the assembly
