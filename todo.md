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


## Wing + Cylinder High Scalability

- [ ] add 3x3 node-support based smoother to GPU using Cublas and node support sparsity from kmat (so general for cylinder / wingbox)
   - [ ] see if better performance on wing case (better smoother)
   - [ ] see if better perf and more stable conv for cylinder

- [ ] performance tuning of K-GMG-ASW solver
   - [ ] include all solve components + do percentage + bottleneck checks
   - [ ] better / faster than GS at thin shell / no?
   - [ ] check operator complexity of P0 and K*P0 (maybe P0 enough fillin for wing cause of my nearest nodes and elems thing)
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