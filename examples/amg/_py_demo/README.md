============================================================================================================================================================
============================================================================================================================================================
# Algebraic multigrid methods
============================================================================================================================================================
============================================================================================================================================================


Main kinds of AMG:
1. Root-node AMG (more recent) [RN-AMG]
2. Coarse-fine AMG [CF-AMG]
3. Element-AMG [AMGe] from finite elements
4. Smoothed aggregation AMG (oldest) [SA-AMG]
5. Classical or Ruge-Stuben AMG (oldest) [C-AMG]


============================================================================================================================================================
============================================================================================================================================================
## Current Tasks
============================================================================================================================================================
============================================================================================================================================================

Implementing SA-AMG:
- [x] finish 2d poisson demo
- [x] follow GPU-friendly aggregation AMG demo
    * main paper, ref 4 from SA-AMG (GPU-accelerated AMG)
    * coarsenings from ref. 10, 11 of SA-AMG also [serial + parallel versions]
- [x] demo with Reissner-mindlin plate
- [ ] implement orthogonal projector from ref. 1 of SA-AMG (energy-opt multigrid)
- [ ] demo cylinder SA-AMG in python
- [ ] demo wing SA-AMG in python (~20k node mesh AOB)
- [ ] demo mat-free SA-AMG, ref. 
- [ ] if successful, port over to GPUs! just serial aggregation first (for small problems)
- [ ] try to do nonlinear structures with 

Optional:
- [ ] implement other types of coarsening CLJP, PMIS, HMIS (see coarsening.py)


High-level remaining tasks (to implement first in python):
- [ ] SA-AMG
- [ ] CF-AMG
- [ ] RN-AMG
- [ ] AMGe
- [ ] advanced 



============================================================================================================================================================
============================================================================================================================================================
# Literature
============================================================================================================================================================
============================================================================================================================================================



## Root-node AMG

1. [ ] root-node AMG, https://epubs.siam.org/doi/10.1137/16M1082706


## Coarse-fine AMG (CF-AMG)

* similar to AMGe also..

1. [x] more recent energy min AMG with CF-splitting, https://arxiv.org/pdf/1902.05157
    * read it => nice theory, not better than root-node AMG (next)
    * shows theoretically the importance of constraint vectors (or near-kernel modes) in multigrid convergence
    * also shows hilbert space theory and how to more efficiently precondition fixed-sparsity prolongation optimization
2. [x] see also examples/spai/ folder with SPAI multilevel LU (which also uses CF-splitting)
    * source 1: "Sparse approximate inverse and multilevel block ILU preconditioning techniques for general sparse matrices", https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub
    * source 2: "Enhanced multi-level block ILU preconditioning strategies for general sparse linear systems", https://link.springer.com/chapter/10.1007/978-3-031-25820-6_11


## smoothed aggregation SA-AMG
Literature for smoothed aggregation AMG

1.  [ ] energy minimization, https://link.springer.com/article/10.1007/s006070050022
2.  [ ] smoothed aggregation Vanek, https://link.springer.com/article/10.1007/BF02238511
3.  [x] coarsening using this paper, https://www.sciencedirect.com/science/article/pii/S0096300320307487
4.  [ ] GPU accelerated aggregation AMG, https://www.sciencedirect.com/science/article/pii/S0898122114004143
5.  [ ] mat-free SA-AMG, https://www.osti.gov/servlets/purl/2004001
6.  [ ] smoothed aggreg for elasticity, https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.688
7.  [ ] general interp strategy for SA-AMG, https://epubs.siam.org/doi/10.1137/100803031
8.  [ ] matrix-free approach for SA-AMG, https://www.osti.gov/servlets/purl/2004001
9.  [ ] SA-AMG for Markov chains, https://epubs.siam.org/doi/10.1137/080719157
10. [x] maximal independent set coarsenings for AMG, https://epubs.siam.org/doi/10.1137/110838844
11. [x] Vanek convergence of SA-AMG, https://link.springer.com/article/10.1007/s211-001-8015-y 
12. [x] p. 123 conv of SA-AMG on HPC, https://link.springer.com/content/pdf/10.1007/b104300.pdf
13. [x] SA-AMG Ucolorado, https://amath.colorado.edu/~stevem/appm6640/sa.pdf
14. [ ] modified SA-AMG on HPC, https://www.osti.gov/servlets/purl/1725736
15. [ ] classical vs SA-AMG, https://www.cfd-online.com/Forums/main/258171-difference-between-classical-amg-smoothed-aggregation-amg.html
16. [ ] Puji AMG parallel, https://link.springer.com/chapter/10.1007/0-387-24049-7_6

## classical-AMG 
* earlier kind, think it's the same as Ruge-Stuben. TBD (see paper 14 from SA-AMG).

## AMGe (element-AMG)
* TODO

## advanced / other

1. [ ] machine learning to accelerate AMG, https://www.sciencedirect.com/science/article/pii/S0898122124002256
2. [ ] space decomp + subspace correction, https://epubs.siam.org/doi/10.1137/1034116


## scalabilty
1. [ ] scaling hypre's multigrid to 100,000 cores, https://link.springer.com/chapter/10.1007/978-1-4471-2437-5_13
2. [ ] GPU accelerated aggregation AMG, https://www.sciencedirect.com/science/article/pii/S0898122114004143

* look at ML manifold-constrained HPC, https://arxiv.org/pdf/2512.24880


============================================================================================================================================================
============================================================================================================================================================
## Finished Tasks
============================================================================================================================================================
============================================================================================================================================================
