# Python examples of single level, multilevel, and modified ILU for plate

* STATUS: not further implementing ILU(k) preconditioners, as often breakdown due to zero pivots
   and the ignored (nofill) entries lead to high errors for near thin-shell singular matrices and poor approx solutions

### TASKS
- [ ] deflated LU decomposition, https://epubs.siam.org/doi/epdf/10.1137/0721050
- [ ] deflated GMRES (see if helps with BILU(0)-SVD stalling)



### COMMENTS
* for higher thicknesses (less singular system) - the ILU(0) preconditioners are very effective
    * try plot_plate_ilu0.py or the milu version with higher thickness
    * random node ordering can help somewhat stabilize the factorization and backpropagation for thin-shell case
       but it is not quite enough and the plot_plate_ilu0.py scripts show the nofill solutions are not anywhere near energy-smooth like multigrid problems
    * the main problem here is these near-singular matrices can incur near-zero pivots leading to unstable preconditioners
* I also tried multilevel ILU(0) preconditioners, but still has issues with finest level ILU factor which messes up coarser grids also..
    * trying sparse approximate inverse next instead
* in Kyle Anderson's paper https://fun3d.larc.nasa.gov/papers/anderson_aiaa_2020_3022.pdf,
    he showed that Q-ordering can significantly improve the performance of the preconditioners, but the ignored fillin entries for me still made it not perform well for thinner shells.
    * he said he implemented methods from Saad chapter 7, https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf. I did too and followed the same algorithm.
    * there are ways to somewhat stabilize or modify ILU(0) factorization to improve it.. such as energy minimization, modified ILU
    * however modified ILU constraint is often good for single DOF per node problems, not multi. Unclear what constraints (maybe just rigid body) to enforce? Maybe could do something with that..

## OTHER NOTES

* implemented qordering and gauss-jordan ILU
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

* modified ILU(0) for single DOF per node in Chapter 7 page 326 of Saad, https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf


### maybe tasks

- [ ] add extra fillin for fine preconditioners Binv so that coarse solves are more accurate
    * otherwise lower thickness schur complement is bad and coarse updates are bad..
- [ ] write my own pivoted ILU factorization on GPU with level sets (to replace CuSparse no pivot ILU factor), if possible. will have to see if it is possible.
- [ ] look at plot precond scripts
    * random clearly stabilizes solve some..
    * and lower thickness breaks down the preconditioner.. but why?
    * I couldn't seem to find errors in the LU factorization.. why is the stencil breaking down?
    * is it because of the mixed integration singularities in the matrix? need better element type? nonphysical energy modes?
- [ ] how to stabilize the local ILU(0) or ILU(k) factorization?
    * more fillin, change matrix some, use ML, idk? 
    * there has to be some way to recover more stable factorization like what Kyle Anderson did..
- [ ] almost want to remove the nonphysical energy modes from the matrix in order to do the smoothing..
    * so it solves a smoother version? would need asymptotic IGA elements for that no?

### done tasks


- [x] modified ILU with rigid body constraints like we do in AMG..
    * see Saad chapter 7 modified ILU constraints (https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf) on how to change each row factor..
    * see SA-AMG energy opt paper on orthogonal projectors.. https://link.springer.com/article/10.1007/s006070050022
    * this could restore energy smoothness somewhat? but unclear..