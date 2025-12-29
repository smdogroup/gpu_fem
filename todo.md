# Current Devel tasks

## Multilevel ILU

- [ ] implement multilevel ILU based on https://arxiv.org/pdf/1901.03249 for linear time complexity
- [ ] then do multipatch version of multilevel ILU for wing case? see someone's master's thesis on 
- [ ] implement schwarz smoother and isogeometric cohomology stuff here, https://grandmaster.colorado.edu/copper/2016/StudentCompetition/Benzaken_Isogeometric_Multigrid.pdf


## writeup
- [ ] show some math on why ILU(0) for very low thickness shells will not be great preconditioner for plate case?
    * could show that no matter the nofill stencil.. it will still have bad condition number compared to 1 the full fillin inverse stencil..
    * maybe using Local fourier series analysis
- [ ] then show how multilevel ILU(0) can more cheaply get something like full inverse stencil by using coarser and multiple levels with less memory



## done tasks

Qorder ILU(0) with NASA
- [x] read through qordering ILU(0) libraries in NASA FUN3D (SFE section is where Kyle's code is)
   * do they do pivoting in each block and how do they do pivoting (YES)
   * do they use cusparse on GPU? (NO)
- [x] test + verify ILU(0) with unittests subfolder
    - [x] verify GaussJordan (including pivoting?) on 100 matrices
    - [x] verify incomplete LU factor R = LU - A error on 10 different matrices
    - [x] check how R = LU - A factor residual changes with reordering, etc.
    - [x] see where factor error builds up and if any bugs / how to fix..
    * conclusion: when plotting the predicted M^-1 * y => x even for a plate
    *  the solution for ILU(0) can be way off and is just a basic smoother
    *  i.e. doesn't solve coarse information very well (is only local and fine information stencil)
    *  also when rereading Kyle Anderson's ILU(0) results with TACS seems like on wing case,
    *  he only gets like 10^{-2} reduction on uCRM wing with the best case
    *  thus: I'm just going to try and do multilevel ILU(0) => you need to do some coarse info solve
    *  in order to get better preconditioner and this makes sense when you look at the physics for more complex and general loadings in elasticity
    * also the random reordering does somewhat improve stability + accuracy of the fine ILU stencil
    *  but when you look at the math of what Uinv * Linv does => it's just a local fine smoother and that isn't gonna be that great..
    *  need to do some more hierarchical stuff like multilevel ILU
    *  need higher fill-in in order to get more accurate inverse stencil for true solution and stronger preconditioner.. but that is computationally and memory expensive
* CONCLUSION: I don't think ILU(0) even with qordering and gauss-jordan pivoting will be strong enough preconditioner for large DOF
- [x] other papers
    - [x] kyle anderson Q-ordering, https://fun3d.larc.nasa.gov/papers/anderson_aiaa_2020_3022.pdf
    - [x] https://www.sciencedirect.com/science/article/pii/S0045782519305213
    - [x] https://www.academia.edu/95989671/Crout_Versions_of_ILU_for_General_Sparse_Matrices

* modified ILU (MILU) based on maintaining row-sums was fine, but in several papers didn't improve time complexity
* only multilevel ILU can achieve near linear time complexity (due to hierarchy and coarse updates)


Multilevel ILU 
- [x] multilevel ILU lit review
    - [x] multilevel ILU for saddle point, not that great: https://arxiv.org/pdf/1911.10139
    - [x] multilevel ILU for steady incompressible flows not that great: https://onlinelibrary.wiley.com/doi/epdf/10.1002/fld.4913
    - [x] sparse linear systems book : https://link.springer.com/book/10.1007/978-3-031-25820-6
    - [x] nice paper multilevel ILU implementation: https://arxiv.org/pdf/1901.03249 

### maybe tasks

- [ ] write my own pivoted ILU factorization on GPU with level sets (to replace CuSparse no pivot ILU factor), if possible. will have to see if it is possible.
