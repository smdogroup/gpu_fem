# Devel tasks

## Current Task
- [ ] try ASGS on plate and/or shell case some more
   - [ ] add edge stabilization terms to plate (+ maybe beam)
   - [ ] read + annotate ASGS plate element should mesh converge, make sure I've implemented it correctly
   
- [ ] try HRA element on plate + shell now using discont strains + static condensation, [Two-field formulations for isogeometric Reissner–Mindlin plates and shells with global and local condensation](https://link.springer.com/article/10.1007/s00466-021-02080-8)
- [ ] try bubble element or other advanced geometric methods (so consistent), see subdivision surfaces below (fully consistent without IGA or with IGA?)
- [ ] try non-uniform integration or other projection methods => would prefer to remove singularity without penalizing sub-grid modes basically
   * maybe machine learning could stabilize an ASGS or OSGS sub-grid method?

* then add all new papers to the thesis? 

## Reading list
- [ ] [A Consistent Finite Element Formulation of the Geometrically Non‑linear Reissner‑Mindlin Shell Mode](https://link.springer.com/article/10.1007/s11831-021-09702-7)
- [ ] [An efficient and robust rotational formulation for isogeometric Reissner–Mindlin shell elements](https://www.sciencedirect.com/science/article/pii/S0045782516300111)
- [ ] [Improved numerical integration for locking treatment in isogeometric structural elements, Part I: Beams](https://www.sciencedirect.com/science/article/pii/S0045782514002096)
   * and next parts for plate / shells
- [ ] DSG discrete shear gap, [Numerical efficiency, locking and unlocking of NURBS finite elements](https://www.sciencedirect.com/science/article/pii/S0045782509001108)
- [ ] DSG discrete shear gap 2, [A hierarchic family of isogeometric shell finite elements](https://www.sciencedirect.com/science/article/pii/S0045782512003337)

B and F projection methods
- [ ] [B and F projection methods for nearly incompressible linear and non-linear elasticity and plasticity using higher-order NURBS elements](https://www.sciencedirect.com/science/article/pii/S0045782508000248)
- [ ] [Locking free isogeometric formulations of curved thick beams](https://www.sciencedirect.com/science/article/pii/S004578251200196X)
- [ ] [An efficient blended mixed B-spline formulation for removing membrane locking in plane curved Kirchhoff rod](https://www.sciencedirect.com/science/article/pii/S0045782517305467)
- [ ] [Efficient isogeometric NURBS-based solid-shell elements: Mixed formulation and ¯B-method](https://www.sciencedirect.com/science/article/pii/S0045782513002053)

## Thickness-Ind Multigrid

* NOTE : line search needs to be very small (1e-3 updates for wing to get best performance) => smoother doing heavy work, multigrid barely anything

- [ ] try this new subdivison surface shell element [SUBDIVISION SURFACES: A NEW PARADIGM FOR THIN-SHELL FINITE-ELEMENT ANALYSIS](https://multires.caltech.edu/pubs/thinshell.pdf)

New plate elements (that may help shell)
- [ ] implementing ASGS (algebraic sub-grid scale) for plate elements
   * see if can get right mesh convergence again
   - [ ] does OSGS work better?
- [ ] maybe try ASGS with machine learning (and nondim params) to discover good stability constants (like people do for turbulence)?
   * cite some of those papers (could be new method)
- [ ] try bubble and other elements in theory/plate.md

NEW SHELL ELEMENTS

* the DeRham-shear and DeRham for some mem with MITC for exy shear method actually worked (only one membrane constraint is ignored! big perf improvement)
- [ ] check other load cases see if exy being red integrated is issue for multigrid conv.. prob only slightly would reduce it
- [ ] fix load scaling issue + measure mesh convergence rate.. may need to change MITC to 2x2 instead of 1x1 for exy? Not sure yet..
- [ ] try this on GPU + closed cylinder?
- [ ] try it on a doubly-curved shell like a spherical panel now!
- [ ] then figure out how to do it on a general curved surface (with varying curvature?) => harder.. continuous vs discrete director?

- [ ] do a doubly curved sphere case (separate folder from general shell, with varying directors)

- [ ] implement DeRham-shear and MITC-membrane with energy-smoothing for remaining prolongation issues
- [ ] try also DRIG-shear and membrane fully MITC for comparison
- [x] implement DeRham-shear and DeRham for some mem constraints, MITC- for remaining, then energy-smoothing for remaining prolongation issues
- [ ] then maybe try u ~ 1x4 IGA, v ~ 2x3 IGA, thx ~ 1x2, thy ~ 2x1, w ~ 2x2 IGA so fully consistent interpolation spaces for cylinder.. (just to try it) 
   - [ ] need 4th order IGA
- [ ] implement DeRham-shear and Regge-membrane (see Regge mem locking paper) with energy-smoothing, fewer mem-lock constraints?
- [ ] implement new Nedelec shell elements (see thesis in theory)


- [ ] add axial fraction and try line search again of baseline standard prolongator for beam, cylinder (see if still get good speedup to that, fair comparison)
   - [ ] also try NL beam, plate see if my EP and LP methods still help
- [ ] try subgrid method for beams + plates see 3_cylinder/theory
- [ ] try AMGe again get refs for that (comparison to AMG methods)

- [ ] further investigate cylinder case
   - [ ] plot locking strains coarse to fine on small mesh, see if the math makes sense or not => can a prolongation matrix exist that maps zero to zero locking? Or we need C1-cont / new element?
   - [ ] look at strain manifold and math, deeper understanding. Issue that C0-cont, do tying strains match on edges?
   - [ ] check the rigid body modes in P and P^T are respected... tricky on cylinder case? Is that what is degrading perf that we need orthog projector (and plate case it is more benign and doesn't depart from it)?
      * maybe we can't use orthog projector easily for GMG because we would need to orthog project fine and coarse for GMG (while AMG just needs fine projector)?
   * see lit review section on Kirchoff IGA, mem locking and advanced element discretization methods
   * back to cylinder python examples.. in adv_elem folder

- [ ] further lit review on curved Reissner-Mindlin shells
   - [ ] read book on shell theory
   - [ ] read book on manifolds and differential forms
   - [ ] read book on IGA
   - [ ] find papers on C1-continuous Reissner-Mindlin vs Kirchoff shells
   - [ ] compile various shell theory solution methods (for curved surf)
   - [ ] read up on and implement Nedelec H1, H-div, H-curl and L2 elements (Seiyon), see if they can do them on curved surfaces. May learn valuable info
   - [ ] read up on Kirchoff-IGA shells and other manifold methods
- [ ] [Do locking-free finite element schemes lock for holey Reissner-Mindlin plates with mixed boundary conditions?](https://arxiv.org/pdf/2506.21999)

- [ ] see if new discretization / better smooth prolong can speedup wing + cylinder cases (more energy-smooth prolong and less V(k,k) smooth steps needed)
- [ ] how to discretize cylinder with 2nd order IGA (see IGA book)
- [ ] maybe do T-splines

## AMG methods
- [ ] try energy min GMG/AMG like these papers:
   - [ ] [PARALLEL ENERGY-MINIMIZATION PROLONGATION FOR ALGEBRAIC MULTIGRID](https://lukeo.cs.illinois.edu/files/2023_JaFrScOl_paramg.pdf)
   - [ ] [AN ENERGY-MINIMIZING INTERPOLATION FOR ROBUST MULTIGRID METHODS](https://cs.uwaterloo.ca/~jwlwan/papers/WanChanSmith00.pdf)
   - [ ] [A GENERAL INTERPOLATION STRATEGY FOR ALGEBRAIC MULTIGRID USING ENERGY-MINIMIZATION](https://www.unm.edu/~jbschroder/docs/OlSc2011.pdf)
- [ ] look at AMGe again?

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

## Writing 
- [ ] add all my new lit review in the examples/adv_elems/_theory/*.md markdown to the thesis and/or journal paper
- [ ] also add all lit review and results from my Scitech paper to my thesis

## Other (optional/maybe)

- [ ] look at my AMG method comparisons again
   - [ ] possible to do support-ASW with AMG (prob would be necessary)
   - [ ] different coarsening methods + CF, RN, SA
   - [ ] good in thin plate, cylinder / wing or no?
   - [ ] machine learning coarsening methods?

=============================================
=============================================
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
