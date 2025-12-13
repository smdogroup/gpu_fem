# Q ordering a la K. Jacobson
- [ ] [Anderson and K. Jacobson, Node Numbering for Stabilizing Preconditioners Based on Incomplete LU Decomposition](https://arc.aiaa.org/doi/abs/10.2514/6.2020-3022)
    * need to read this and get this to work
    * Q ordering matrix-reordering strategy that reduces the occurence of preconditioner instabilities from incomplete LU => allowing a Krylov subspace method to solve hte linear systems (previously unsolvable)
    * Q ordering greatly increases the robustness and reliability of simulations
    * concerns large linear systems for fluids and structures applications, with iterative Krylov-subspace methods used
    * baseline example : turbulent FUN3D with stabilized finite element solver, with Newton method for nonlinear solve, GMRES for linear solve with ILU(k) preconditioner as in Saad
    * elimination phase of ILU[k] is unstable, resulting in poor iterative method convergence for ILU(0) to ILU(3), but converges for ILU(4) and up
    * looks at magnification of error in the back-solve / triangular solve of the ILU preconditioner


## FUN3D toolkit
- [ ] [Wood, Sparse Linear Algebra Toolkit for Computational Aerodynamics](https://arc.aiaa.org/doi/epdf/10.2514/6.2020-0317)
    * this looks like a great resource here
- [ ] [K. Anderons, Stabilized Finite Elements in FUN3D](https://arc.aiaa.org/doi/10.2514/1.C034482)

## Sparse Linear Systems book
* great description of ILU(k) preconditioner here
- [ ] [Saad book, Iterative Methods for Sparse Linear Systems](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf)

## Iterative Krylov Linear Solvers
- [ ] [Saad, GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems](https://epubs.siam.org/doi/abs/10.1137/0907058)
- [ ] [Axelsson, Conjugate gradient type methods for unsymmetric and inconsistent systems of linear equations](https://www.sciencedirect.com/science/article/pii/0024379580902268)
- [ ] [Eisenstat, Variational Iterative Methods for Nonsymmetric Systems of Linear Equations](https://www.jstor.org/stable/2157222?seq=1)
- [ ] [Saad, Practical Use of Some Krylov Subspace Methods for Solving Indefinite and Nonsymmetric Linear Systems](https://epubs.siam.org/doi/10.1137/0905015)
- [ ] [Elman, Iterative methods for large, sparse, nonsymmetric systems of linear equations](https://dl.acm.org/doi/abs/10.5555/910599)
- [ ] [Vorst, Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems](https://epubs.siam.org/doi/10.1137/0913035)


### Pivoting for Stability
- [ ] [Poole, Gaussian elimination: when is scaling beneficial?](https://www.sciencedirect.com/science/article/pii/002437959290382K?via%3Dihub)
- [ ] [Poole, A geometric analysis of Gaussian elimination. I](https://www.sciencedirect.com/science/article/pii/002437959190337V?via%3Dihub)
- [ ] [Poole, A geometric analysis of Gaussian elimination. II](https://www.sciencedirect.com/science/article/pii/002437959290432A?via%3Dihub)

### Reliability of ILU Preconditioners
- [ ] [Chow, Experimental study of ILU preconditioners for indefinite matrices](https://www.sciencedirect.com/science/article/pii/S0377042797001714?via%3Dihub)

### Reordering Strategies for ILU
- [ ] [Smyth, Algorithms for the reduction of matrix bandwidth and profile](https://www.sciencedirect.com/science/article/pii/0377042785900482?via%3Dihub)
- [ ] [Bernardes, A Systematic Review of Heuristics for Profile Reduction of Symmetric Matrices](https://www.sciencedirect.com/science/article/pii/S187705091501039X)
- [ ] [Chagas, Metaheuristic-based Heuristics for Symmetric-matrix Bandwidth Reduction: A Systematic Review](https://www.sciencedirect.com/science/article/pii/S1877050915010376)
- [ ] [Sloan, Automatic element reordering for finite element analysis with frontal solution schemes](https://onlinelibrary.wiley.com/doi/10.1002/nme.1620190805)
- [ ] [D'Azevedo, Ordering Methods for Preconditioned Conjugate Gradient Methods Applied to Unstructured Grid Problems](https://epubs.siam.org/doi/10.1137/0613057)
- [ ] [Feng, An Improvement of the Gibbs-Poole-Stockmeyer Algorithm](https://journals.sagepub.com/doi/10.1260/1748-3018.4.3.325)
- [ ] [Dutto, The effect of ordering on preconditioned GMRES algorithm, for solving the compressible Navier-Stokes equations](https://onlinelibrary.wiley.com/doi/10.1002/nme.1620360307)
- [ ] [Wang, An Improved Algorithm for Matrix Bandwidth and Profile Reduction in Finite Element Analysis](https://www.jpier.org/PIERL/pier.php?paper=09042305)
- [ ] [Kumfert, Two improved algorithms for envelope and wavefront reduction](https://link.springer.com/article/10.1007/BF02510240)
- [ ] [Gibbs, An Algorithm for Reducing the Bandwidth and Profile of a Sparse Matrix](https://www.jstor.org/stable/2156090?seq=1)
- [ ] [Reid, Ordering symmetric sparse matrices for small profile and wavefront](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0207(19990830)45:12%3C1737::AID-NME652%3E3.0.CO;2-T)
- [ ] [Reid, Reducing the Total Bandwidth of a Sparse Unsymmetric Matrix](https://epubs.siam.org/doi/10.1137/050629938)
- [ ] [Marti, Reducing the bandwidth of a sparse matrix with tabu search](https://www.sciencedirect.com/science/article/pii/S0377221700003258?via%3Dihub)
- [ ] [Mladenovic, Variable neighbourhood search for bandwidth reduction](https://www.sciencedirect.com/science/article/pii/S0377221708010540?via%3Dihub)

#### Notable : Cuthill-McKee Reordering
- [ ] [Wai-Hung Liu, Comparative Analysis of the Cuthill-McKee and the Reverse Cuthill-McKee Ordering Algorithms for Sparse Matrices](https://www.jstor.org/stable/2156087?seq=1)
- [ ] [Cuthill and McKee, Reducing the bandwidth of sparse symmetric matrices](https://dl.acm.org/doi/10.1145/800195.805928)

### Comparisons of Ordering Strategies for LU solve
- [ ] [Duff, The effect of ordering on preconditioned conjugate gradients](https://link.springer.com/article/10.1007/BF01932738)
- [ ] [Dutto, The effect of ordering on preconditioned GMRES algorithm, for solving the compressible Navier-Stokes equations](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620360307)
- [ ] [Benzi, Preconditioning Techniques for Large Linear Systems: A Survey](https://www.sciencedirect.com/science/article/pii/S0021999102971767)
- [ ] [Benzi, Orderings for Incomplete Factorization Preconditioning of Nonsymmetric Problems](https://epubs.siam.org/doi/10.1137/S1064827597326845)
- [ ] [Chernesky, On preconditioned Krylov subspace methods for discrete convection–diffusion problems](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1098-2426(199707)13:4%3C321::AID-NUM2%3E3.0.CO;2-N)
- [ ] [D'Azevedo, Ordering Methods for Preconditioned Conjugate Gradient Methods Applied to Unstructured Grid Problems](https://epubs.siam.org/doi/10.1137/0613057)