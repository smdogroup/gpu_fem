# Algebraic multigrid methods

TODO : demo the following AMG methods first in python
There are three main kinds of AMG:
1. Root-node AMG (more recent) [RN-AMG]
2. Coarse-fine AMG [CF-AMG]
3. Element-AMG [AMGe] from finite elements
4. Smoothed aggregation AMG (oldest) [SA-AMG]
Older variants of AMG are classical AMG and more (but are less effective versions of SA-AMG basically)


## Current Tasks

Implementing SA-AMG:
- [ ] finish 2d poisson demo
- [ ] follow GPU-friendly aggregation AMG demo
    * main paper, https://www.sciencedirect.com/science/article/pii/S0898122114004143
    * aggregation from Maximal independent set, https://epubs.siam.org/doi/10.1137/110838844
    * they did parallel version in their paper of MIS (aggregation steps)
    * see also this paper, https://link.springer.com/article/10.1007/s211-001-8015-y
- [ ] implement other types of coarsening CLJP, PMIS, HMIS (see coarsening.py)
- [ ] demo with Reissner-mindlin plate

Implement the following in python
- [ ] CF-AMG
- [ ] SA-AMG
- [ ] RN-AMG
- [ ] AMGe
- [ ] advanced 

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

coarsening using this paper [ ] https://www.sciencedirect.com/science/article/pii/S0096300320307487

1. [ ] energy minimization, https://link.springer.com/article/10.1007/s006070050022
2. [ ] smoothed aggregation Vanek, https://link.springer.com/article/10.1007/BF02238511
3. [ ] smoothed aggreg for elasticity, https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.688
4. [ ] general interp strategy for SA-AMG, https://epubs.siam.org/doi/10.1137/100803031



## AMGe (element-AMG)



## advanced / other

7. [ ] GPU accelerated aggregation AMG, https://www.sciencedirect.com/science/article/pii/S0898122114004143
8. [ ] machine learning to accelerate AMG, https://www.sciencedirect.com/science/article/pii/S0898122124002256
9. [ ] space decomp + subspace correction, https://epubs.siam.org/doi/10.1137/1034116



## scalabilty

- [ ] scaling hypre's multigrid to 100,000 cores, https://link.springer.com/chapter/10.1007/978-1-4471-2437-5_13