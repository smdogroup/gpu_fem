## Methods to create for new BDDC Assembler multi-GPU

* don't worry so much about expense of ghost inter-GPU communication (it's not as expensive as you think). Main loss of multi-GPU ideal speedup
comes from underutilization of GPU when you solve smaller partitioned matrices than the single GPU.

From single-GPU fetiDP
- [x] update_after_assembly
- [x] base solver helper methods: factor, set_print, set_rel_tol, set_abs_tol, get_num_iterations, set_cycle_type, free, getLambdaSize
- [x] helper IEV methods: get_num_IEV_nodes, get_IEV_conn, get_IEV_xpts, get_IEV_vars
- [x] states must be set from setup_structured_subdomains
    * use new methods to take a subdomain splitting object (so modular) with first single GPU to multi GPU splitting
    import_splitting() (single GPU) to create_multigpu_splitting() (multi-gpu form) from a splitting object
    - [x] elem_sd_ind, num_subdomains
    - [x] node_nsd, node_class_ind
    - [x] I_nnodes, IE_nnodes, IEV_nnodes
    - [x] Vc_nnodes, V_nnodes
    - [x] lam_nnodes
    - [x] IEV_sd_ptr, IEV_sd_ind, IEV_nodes
    - [x] IEV_elem_conn
    - [x] d_IEV_elem_conn
    build_IE_I_V_maps()
    - [x] IE_nodes, I_nodes
    - [x] IEVtoIE_map + IEVtoIE_imap
    - [x] IEVtoI_map + IEVtoI_imap
    - [x] IE_interior, IE_general_edge
    - [x] d_IE_interior, d_IE_general_edge, d_IE_nodes
    - [x] d_IEVtoIE_imap, d_IEVtoI_imap
    build_Vc_and_gam_maps()
    - [x] Vc_nodes, Vc_inodes
    - [x] d_Vc_nodes
    - [x] gam_nodes
    - [x] VctoV_imap, IEVtoV_imap
    - [x] d_VctoV_imap, d_IEVtoV_imap
    build_IEV_sparsity()
    - [x] IEV partitioner from elem_conn
    - [x] IEV multi-GPU matrix
    - [x] IEV_rowp, cols, nnzb, and vals from multi-GPU matrix
    build_IE_and_I_sparsity()
    * can't use multi-GPU object directly here since it expects to build from a simple element connectivity, but instead IE and I don't have those (missing some V nodes)
    * so we'll just handle these like before?
    - [x] IE_rowp, cols, nnzb, vals
    - [x] I_rowp, cols, nnzb, vals
    * maybe make kmat_IE and kmat_I from local rowp, cols alternate GPUbsrmat constructor? Not elem_conn? Or not.. also this matrix needs to be copied to create it also..
- [ ] make submethods from setup_matrix_sparsity and setup_coarse_matrix_sparsity
    * since no reordering / LU pattern stuff needs to be done anymore => don't need multiple setup steps
    from setup_matrix_sparsity()
        new method: create_kmat_copy_maps()
            - [x] kmat_IEnofill_map, kmat_IEtoIEV_map
            - [x] d_kmat_IEnofill_map, d_kmat_IEtoIEV_map
            - [x] kmat_Inofill_map, kmat_ItoIEV_map
            - [x] d_kmat_Inofill_map, d_kmat_ItoIEV_map
        new method: build_Svv_sparsity()
            - [x] temp objs: Svv_adj, Svv_rowcts
            - [x] local: Vc_node_imap, Svv_rowp, Svv_nnzb, Svv_cols, Svv_rows, d_Svv_vals
            - [ ] prob also need global pattern of Svv later for CuDSSMG
    from setup_coarse_matrix_sparsity()
        - [ ] user sets: MAX_NUM_VERTEX_PER_SUBDOMAIN (on constructor maybe or from splitting)
        new method: build_Svv_maps()
            - [x] Svv_copy_nnzb, Svv_IEV_copyBlocks, Svv_Vc_copyBlocks
            - [x] d_Svv_IEV_copyBlocks, d_Svv_Vc_copyBlocks
            - [x] IEVset_nnzb, IEVtoSVV_nnzb, d_IEVset_blocks, d_IEVout_blocks, d_IEVtoSVV_blocks
        new method: build_iev_bcs()
            * moved this task below
* don't need setup_MLIEV_coarse_matrix_sparsity (that's for multilevel BDDC with more than 2 levels)
- [x] assemble subdomains
    - [x] add_IEV_jacobian()
    * make new apply_iev_bcs method below
- [x] assemble_coarse_problem
    - [x] did we create Svv values yet?
- [x] build_iev_bcs()
- [x] apply_iev_bcs() new method to apply to each local kmat_IEV matrix
- [x] add_subdomain_fext
<!-- - [ ] set_inner_solvers -->
- [x] get_lam_rhs
- [x] mat_vec
- [x] solve
- [x] get_global_soln
- [x] copyKmat_IEVtoIE
- [x] copyKmatIEVtoI
- [x] copyKmatIEVtoSvv
- [x] computeSvvInverseTerm
- [x] setVec_IEVtoV_vals
- [x] addMat_IEVtoV_vals
- [x] addVec_globaltoIEV
- [ ] new states for previous method:
    - [ ] d_IE_nsd, d_V_nsd
- [x] addVecIEVtoIE
- [x] addVecIEVtoI
- [x] addVecIEtoI
- [x] addVecIEVtoVc
- [x] addVecLamtoIE
- [x] addVecIEtoIEV
- [x] addVecItoIEV
- [x] addVecItoIE
- [x] addVecVctoIEV
- [x] addVecIEtoLam
- [x] zeroInteriorIE
- [x] addGlobalSoln
- [ ] carefully figure out, double check, and think about for each addVec / other methods where ghost node reductions should be done / not
    * this is not as trivial as you think.. need to draw it probably (and not do redundant ones)..
    - [ ] also where to put expand from local also..
    - [ ] add ctx->sync() in other places also?
<!-- - [ ] sparseMatVec -->
<!-- * sparseTransposeMatVec unused -->
- [ ] solveSubdomainIE
- [ ] solveSubdomainI
- [ ] solveCoarse
- [ ] _compute_jump_operators
- [ ] find_block_index
utils
- [ ] allocate_workspace (is this same as allocate_vecs now?)
- [ ] clear_host_data
- [ ] clear_structured_host_data

For nonlinear + adjoint
* ignore set_global_rhs
- [ ] set_IEV_linear_rhs
- [ ] set_IEV_adjoint_rhs
- [ ] set_IEV_residual


From single-GPU BDDC
- [ ] new versions of: getLambdaSize(), set_IEV_residual, get_lam_rhs, update_after_assembly, mat_vec, solve, get_global_soln
- [ ] zeroIEinIEV
- [ ] zeroIinIEV
- [ ] addVecIEVtoGam
- [ ] addVecGamtoIEV
- [ ] addVecGamtoIE
- [ ] addVecIEtoGam


For multi-GPU CuDSS
* no need for perm/iperm maps since CuDSS does reordering internally
* no need for AMD reordering / compute LU pattern from our code (just do nofill pattern) cause again CuDSS does that stuff now.
- [ ] new class for CuDSS Subdomain parallel multi-GPU direct solves
    - [ ] threads, etc.
- [ ] new class for CuDSSMG solves (multi-GPU schur complement) for S_VV
    - [ ] BSRtoCSR copy maps on each GPU
    - [ ] inter-GPU copy of CSR and add on root (cause some repeated nodes for S_VV)
    - [ ] inter-GPU copy of the rhs and soln also? (ghost + inter-GPU)
    - [ ] full sparsity pattern of S_VV on root GPU
- [ ] something to make the kmat_I, kmat_IE CuDSS subdomain parallel solvers
- [ ] something to make the S_VV CuDSSMG solver
- [ ] factorIEsubdomains
- [ ] factorIsubdomains
- [ ] factorCoarseVertex


Wrapper classes:
- [ ] domdec_pcg_wrapper.h
- [ ] linear BDDCInterface (for optimization)
- [ ] nonlinear BDDCInterface (for optimization)