# Current Devel tasks

## Thickness-Ind Multigrid

- [ ] further investigate cylinder case
   - [ ] plot locking strains coarse to fine on small mesh, see if the math makes sense or not => can a prolongation matrix exist that maps zero to zero locking? Or we need C1-cont / new element?
   - [ ] look at strain manifold and math, deeper understanding. Issue that C0-cont, do tying strains match on edges?
   * see lit review section on Kirchoff IGA, mem locking and advanced element discretization methods

- [ ] further lit review on curved Reissner-Mindlin shells
   - [ ] read book on shell theory
   - [ ] compile various shell theory solution methods (for curved surf)
   - [ ] read up on and implement Nedelec H1, H-div, H-curl and L2 elements (Seiyon), see if they can do them on curved surfaces. May learn valuable info
   - [ ] read up on Kirchoff-IGA shells and other manifold methods

- [ ] see if new discretization / better smooth prolong can speedup wing + cylinder cases (more energy-smooth prolong and less V(k,k) smooth steps needed)

## Put TACS GPU into main TACS

* before putting into TACS
   - [ ] fix mem leaks and/or deallocate host vecs, etc.
   - [ ] add quad-GMG CF class (uses one BDF and constructs coarser meshes by itself)
- [ ] simple shell example with GPU into main TACS
   - [ ] make interface class (separate folder)
   - [ ] move data from main TACS assembler into Element assemblers
   - [ ] maybe make overall GPU assembler to help 
   - [ ] just post simple shell example first (MITC-EP with K-GMG-ASW)

## Wing + Cylinder High Scalability

- [ ] performance tuning of K-GMG-ASW solver
   - [ ] include all solve components + do percentage + bottleneck checks
   - [ ] better / faster than GS at thin shell / no?
   - [x] check operator complexity of P0 and K*P0 (maybe P0 enough fillin for wing cause of my nearest nodes and elems thing)
      * no fillin didn't help..
   - [ ] check if + how to fix speedup from RTX 3060 to Milan A100 GPU

- [ ] implement multi-GPU for K-GMG-ASW solver
   - [ ] start with multi-GPU standalone demo of point-smoothing Jacobi-GMRES for 2D Poisson -> how to split mat-vec prods, etc. distribute data
   - [ ] read papers from TACS GPU journal on "towards exascale" or AMG highly parallel for GPU, may help a lot (maybe new papers)
   - [ ] how to do + do multi-GPU for K-GMG-ASW / best solver, domain splitting some? need metis or anything?
   - [ ] show weak vs strong scaling of multi-GPUs, maybe compare with typical ILU(k) multi-GPU so I can show ASW or my GMG solver way more scalable

- [ ] do linear + nonlinear optimizations for plate, cylinder + wing with K-GMG
   - [ ] implement prolong + ASW assembly, factor and other setup steps for nonlinear case, optimization
   - [ ] add optimization interfaces for the new solver (easy), setup
   - [ ] setup + debug stiffened shells + buckling, include that as case
   - [ ] make plots + tables for optimizations with K-GMG-ASW and MITC-EP
      * thick vs thin shell, linear vs NL, MITC vs MITC-EP, speedup to direct-LU
      1. unstiffened plate
      2. unstiffened cylinder
      3. unstiffened AOB wing
      4. stiffened AOB wing
      5. maybe HSCT wing


## Other (optional/maybe)

- [ ] look at my AMG method comparisons again
   - [ ] possible to do support-ASW with AMG (prob would be necessary)
   - [ ] different coarsening methods + CF, RN, SA
   - [ ] good in thin plate, cylinder / wing or no?
   - [ ] machine learning coarsening methods?


## Finished Tasks

- [x] add 3x3 node-support based smoother to GPU using Cublas and node support sparsity from kmat (so general for cylinder / wingbox)
   - [x] see if better performance on wing case (better smoother) => not by much and way more memory
   - [x] see if better perf and more stable conv for cylinder => nope
   * there is horrible V and W-cycle convergence in the wing case at SR = 1e3 (only with K-cycle it is somewhat reasonable) => more investigation? Try 3x3 node smoother
   - [x] see if better conv with the new smoother?
   * a bit surprised, not very helpful on wing case (guess because prolong is still very bad?), and this smoother is more expensive to apply, 4x as much storage too (near direct-LU storage)

- [x] why does extra energy-smoothing steps (MITC-EP) for plate, cylinder / wingbox not converge right now? need lower omega?
   - [x] what is happening with it? check fine and coarse BCs (check if those messed up) => it's line search bounds, see below
   - [x] comparing examples/adv_elems/4_wing/2_mitc_ep2.cu and examples/gmg/3_aob_wing (unstruct smooth(0) and unstruct prolong no bsr, the result is much different.. so baseline prolongator doesn't match CSR version? isn't good enough?). Fixing this may help stabilize the convergence of energy-smoothing.. 
   - [x] performance tune PT matrix and/or double storage and copy P into PT.. for faster performance (not bottleneck)
   * soln is line search params (made big difference in wing case 2x speedup with removing line search bounds at high SR)
   * NOTE : csr prolong and no energy smooth with ASW is faster for wing case.. cause energy smoothing doesn't help (strain subspaces in prolong not really compatible, see work on elements and MITC4). Plate matters a lot..