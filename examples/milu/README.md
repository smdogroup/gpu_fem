# Python examples of single and multilevel ILU for plate

## TODO

- [ ] look at plot precond scripts
    * random clearly stabilizes solve some..
    * and lower thickness breaks down the preconditioner.. but why?
    * I couldn't seem to find errors in the LU factorization.. why is the stencil breaking down?
    * is it because of the mixed integration singularities in the matrix? need better element type? nonphysical energy modes?

### maybe tasks

- [ ] add extra fillin for fine preconditioners Binv so that coarse solves are more accurate
    * otherwise lower thickness schur complement is bad and coarse updates are bad..
- [ ] write my own pivoted ILU factorization on GPU with level sets (to replace CuSparse no pivot ILU factor), if possible. will have to see if it is possible.


## NOTES

* implemented qordering and gauss-jordan ILU like in NASA SLAT and Qordering paper (see below)
    * still did not give enough performance for low thickness problems for ILU(0)-GMRES
    * this is because the smoother is not good enough predictor of global solution in many cases I don't think
    * convergence only 1e-2 per 100 iterations also for qordering on uCRM wing for ILU(0) and Kyle Anderson's paper, https://fun3d.larc.nasa.gov/papers/anderson_aiaa_2020_3022.pdf
* thus I decided to try multilevel ILU which could give more level-independent scaling (similar perf as we h-refine)
* relevant papers: 
    * https://arxiv.org/pdf/1901.03249 for multilevel ILU
    * qordering by NASA: https://fun3d.larc.nasa.gov/papers/anderson_aiaa_2020_3022.pdf
    * ILU factors: https://www.sciencedirect.com/science/article/pii/S0045782519305213
    * sparse LU book: https://www.academia.edu/95989671/Crout_Versions_of_ILU_for_General_Sparse_Matrices
    * multilevel ILU lit review
    * multilevel ILU for saddle point, not that great: https://arxiv.org/pdf/1911.10139
    * multilevel ILU for steady incompressible flows not that great: https://onlinelibrary.wiley.com/doi/epdf/10.1002/fld.4913
    * sparse linear systems book : https://link.springer.com/book/10.1007/978-3-031-25820-6
    * nice paper multilevel ILU implementation: https://arxiv.org/pdf/1901.03249 

* for multigrid and thickness-independent performance (should also put this in GMG or AMG examples):
    * see paper by: https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf


