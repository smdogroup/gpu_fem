* don't remake the multilevel BDDC solver in multi-GPU code.. that would probably take too long

## Fix
<!-- - [ ] zeros in first coarseBDDC rhs solve
    - [ ] leads to nans in later solves
    - [ ] problem is we are not setting internal IEV and lam_rhs for coarseBDDC correctly
            => soln is when fineBDDC does IEV => Vc, we also do IEV => IEV form of Vc for MLIEV
            and transfer that into the inner solver..
    - [ ] TODO: method addVecIEVtoV_MLIEV, which will then set IEV form of Vc vec into the IEV_rhs of the coarseBDDC which will help coarseBDDC=>lam_rhs not be zero!
    - [ ] may need similar transfers for out of soln? (prob not though) -->

Missing pieces of coarseBDDC rhs solve:
    * bddc_assembler in _solve we do solveCoarse with f_V, but need to somehow set in the internal coarse solver an IEV equivalent of f_V rhs..
    * then we do K_c * u_c = f_c krylov solve at the coarse level. and need IEV form of that system so we actually just need to set IEV version of f_V (from coarse subdomains) into the coarseBDDC problem. It's ok that the interface solve and interface pc from Krylov use diff vecs each time. Goal is to solve the K_V * u_V = f_V system basically and so just need IEV version of that in BDDC
    * FetidpSolver=>addVecIEVtoVc is used to setup all coarse rhs problems with standard f_V rhs, so inside that method setup the IEV version of Vc rhs and pass it into the coarseBDDC res_IEV state..
    Actual tasks for this now:
    <!-- - [ ] create IEV splitting of coarse nodes (in setup_coarse_structured_subdomains)
        * we don't have it currently I don't think.. or can communicate back?
        * added saved_Vc_nodes
        - [ ] use the coarseBDDC to do that.. (new interaction method) instead of creating it myself.. -->
    - [x] actually I think I have all the info I already need for IEV(fine) => IEV(coarse) on first BDDC vec maps from the method setup_MLIEV_coarse_matrix_sparsity()
        * now maybe I make a similar/helper method for the vector version..
        * yes cause it gets the kmat_IEV form of coarseBDDC fine nodes (which is equiv to IEV-coarse nodes of fineBDDC. get it?) see script 1_3lev_plate.cu around method setup_MLIEV_coarse_matrix_sparsity
    - [x] setup state vars + maps in FetidpSolver for doing the IEV form of Vc or coarse nodes
        * create these things in create_coarse_structured_subdomains call..
        - [x] d_MLIEVtoV_imap
        - [x] f_ML_coarseIEV vector
        - [x] coarse_IEV_nnodes
        - [x] d_ML_vertex_nsd
        - [x] add CoarseSolver type to FetidpSolver, BDDCSolver, and CoarseBDDCSolver
    - [x] setup helper IEVtoVc_MLIEV add method.. and call it optionally in addVecIEVtoVc
    - [x] coarse DomDecKrylovWrapper manages the full coarseSolver.. so call add_res_IEV with the IEV rhs from within addVecIEVtoVc (with MLIEV check)

    * now it finally has non-nan and nonzero values all the way through but doesn't solve correctly yet..
    - [x] check coarseSolver can adequately solve the coarse system using BDDC (check final residual in coarse gam PCG)
        * it is currently failing.. even on high thickness case..
    - [x] check IEV partitioned matrix adds up / reduces into same S_VV matrix (equivalent splittings) [IT MATCHED] using new method debug_multilevel_SVV_matrix
    - [x] check f_V and f_V iev version are equivalent (including sign..) => by adding IEV fersion of f_V into reg f_V pattern and comparing
        * ahh somehow this part doesn't match rn.. there are some small bugs somehow.. ahh there was a bug in the size of the call in addVecIEVtoVc multilevel part => scaled section (kernel launch params didn't match vec)
        * when fixed this.. went down from 15 => 8 iterations on thick=1e0.. yeah! was real bug here..
    - [x] try changing ML_vertex_nsd values to just be 4 everywhere (basically copy the values out of vertex_nsd instead of the glob nsd..) so should be like 4 instead of 2?
        * correct answer is to copy values from node_nsd (making it 4 on all these nodes currently)
    - [x] somehow the solveCoarse norms in bddc_assembler=>_solve() method are still stuck at 1e-1 rel norms every time?
        * could it be issue in res_IEV and fext_IEV usage on coarse PCG solver? What is the problem? Investigate this better, initial problem setup should at least be close to right..
        * ah it was checking solveCoarse residual on fine BDDC and coarse BDDC (and fine BDDC was working right now.. coarseBDDC residual check not cause of node reordering even though it is right..) so it works now!
        * debug check is in BddcSolver=>_solve=>near this->solveCoarse with debug2 = true && is_fine_bddc

    - [x] now make it work for the cylinder problem also..

## Optional
    - [ ] try mat-free GMRES on output instead of mat-free PCG? maybe is losing symmetry in PC?
    - [ ] change mat-free PCG/GMRES to now use progress bar slider..
    - [ ] then try doing different subdomain sizes (change /2 and subdomain) and diff nxe sizes
        - [ ] also try to fix the occasional seg fault..

    <!-- - [ ] check that we can get some progress with lower thicknesses then? even if it takes more iterations? obv we can solve thick shell case t=1e0 now.. -->


<!-- ## Maybe
- [ ] try changing to GMRES solver on outer part, cause maybe the multiple levels and the multilevel preconditioner with PCG is losing symmetry
    - [ ] try for BDDC-AMG as well.. -->
