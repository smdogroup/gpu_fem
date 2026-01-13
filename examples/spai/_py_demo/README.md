# Sparse Approximate Inverse (SPAI) Preconditioners

SPAI preconditioners don't suffer from zero pivots and may be much better for indefinite or high condition number systems

- [x] implement FSAI based on this paper from Saad, https://faculty.cc.gatech.edu/~echow/pubs/newapinv.pdf
- [ ] implement multilevel SPAI solvers
    - [x] "Sparse approximate inverse and multilevel block ILU preconditioning techniques for general sparse matrices", https://www.sciencedirect.com/science/article/pii/S0168927499000471?via%3Dihub
    - [ ] "Enhanced multi-level block ILU preconditioning strategies for general sparse linear systems", https://link.springer.com/chapter/10.1007/978-3-031-25820-6_11
        - [ ] implement block-level permutations, Schur complement, and multilevel solves...
        - [ ] explore its relationship to AMG..
    - [ ] "BILUM: Block Versions of Multielimination and Multilevel ILU Preconditioner for General Sparse Linear Systems", https://epubs.siam.org/doi/10.1137/S106482759732753X
    - [ ] "A survey of multilevel preconditioned iterative methods", https://link.springer.com/article/10.1007/BF01932745

    - [ ] from paper https://epubs.siam.org/doi/abs/10.1137/S0895479899364441
    - [ ] from paper https://onlinelibrary.wiley.com/doi/full/10.1002/nla.2183
    - [ ] from paper https://www.sciencedirect.com/science/article/pii/S037704279900388X
    - [ ] from paper https://dl.acm.org/doi/10.1145/1048935.1050152
    - [ ] from paper https://arxiv.org/pdf/2107.06973

* implement frobenius norm minimization on plate problem (see bachelor's thesis https://fse.studenttheses.ub.rug.nl/11132/1/Koen_van_Geffen_2013_TWB.pdf)
* try deflation method on SPAI preconditioner (also from https://fse.studenttheses.ub.rug.nl/11132/1/Koen_van_Geffen_2013_TWB.pdf) to help elim near-singular or problematic modes