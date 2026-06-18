#pragma once

#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_bddc.cuh"
#include "assembler/gpu_assembler.h"
#include "cuda_utils.h"
#include "direct/cudss_mg_v2.h"
#include "direct/cudss_subdomain.h"
#include "domdec/subdomains/_iev.h"
#include "element/shell/_shell.cuh"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "partition/subdomain_partitioner.h"
#include "utils/multigpu_context.h"

template <typename T, class Assembler_, class Partition, class IEVSplit>
class MultiGPUBDDC_LUSolver {
   public:
    using Assembler = Assembler_;
    using Director = typename Assembler::Director;
    using Basis = typename Assembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename Assembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    using Vec = GPUvec<T, Partition>;
    using SDPartition = SubdomainGPUPartitioner<Partition>;
    using SDVec = GPUvec<T, SDPartition>;
    using Mat = GPUbsrmat<T, Partition>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    MultiGPUBDDC_LUSolver(MultiGPUContext *ctx_, Partition *part_, Assembler *assembler_, Mat *mat_,
                          IEVSplit *split_, bool debug_ = false)
        : ctx(ctx_),
          part(part_),
          assembler(assembler_),
          mat(mat_),
          cublasHandles(ctx_->cublasHandles),
          cusparseHandles(ctx_->cusparseHandles),
          streams(ctx_->streams),
          split(split_),
          debug(debug_) {
        // if (debug) printf("[BDDC] checkpt1\n");
        ngpus = ctx->ngpus;
        num_elements = assembler->get_num_elements();
        num_nodes = assembler->get_num_nodes();
        N = num_nodes * vars_per_node;

        // if (debug) printf("[BDDC] checkpt2\n");
        block_dim = mat->getBlockDim();
        block_dim2 = block_dim * block_dim;

        // if (debug) printf("[BDDC] checkpt3\n");
        d_xpts = assembler->getDeviceXpts();
        d_vars = assembler->getDeviceVars();
        d_loc_elem_components = assembler->getDeviceElemComponents();
        d_loc_comp_data = assembler->getDeviceCompData();
        // if (debug) printf("[BDDC] checkpt4\n");
        assembler->getLocalDeviceBCs(n_owned_bcs, n_local_bcs, d_owned_bcs, d_local_bcs);
        MAX_NUM_VERTEX_PER_SUBDOMAIN = split_->MAX_NUM_VERTEX_PER_SUBDOMAIN;

        // setup on construction (sparsity patterns, maps, etc.)
        // import_splitting();
        // build_IE_I_V_maps();
        // build_IEV_sparsity();
        // build_IE_and_I_sparsity();
        // create_kmat_copy_maps();
        // build_Svv_sparsity();
        // build_Svv_maps();
        // build_iev_bcs();
        // compute_reduced_partitions();
        // allocate_vectors();
        // create_cudss_solvers();

        // if (debug) printf("[BDDC] import_splitting()...\n");
        import_splitting();
        // if (debug) printf("[BDDC] done import_splitting()\n");

        // if (debug) printf("[BDDC] build_IE_I_V_maps()...\n");
        build_IE_I_V_maps();
        // if (debug) printf("[BDDC] done build_IE_I_V_maps()\n");

        // if (debug) printf("[BDDC] build_IEV_sparsity()...\n");
        build_IEV_sparsity();
        // if (debug) printf("[BDDC] done build_IEV_sparsity()\n");

        // if (debug) printf("[BDDC] build_IE_and_I_sparsity()...\n");
        build_IE_and_I_sparsity();
        // if (debug) printf("[BDDC] done build_IE_and_I_sparsity()\n");

        // if (debug) printf("[BDDC] create_kmat_copy_maps()...\n");
        create_kmat_copy_maps();
        // if (debug) printf("[BDDC] done create_kmat_copy_maps()\n");

        // if (debug) printf("[BDDC] build_Svv_sparsity()...\n");
        build_Svv_sparsity();
        // if (debug) printf("[BDDC] done build_Svv_sparsity()\n");

        // if (debug) printf("[BDDC] build_Svv_maps()...\n");
        build_Svv_maps();
        // if (debug) printf("[BDDC] done build_Svv_maps()\n");

        // if (debug) printf("[BDDC] build_iev_bcs()...\n");
        build_iev_bcs();
        // if (debug) printf("[BDDC] done build_iev_bcs()\n");

        // if (debug) printf("[BDDC] compute_jump_operators()...\n");
        compute_jump_operators();
        // if (debug) printf("[BDDC] done compute_jump_operators()\n");

        // if (debug) printf("[BDDC] compute_reduced_partitions()...\n");
        compute_reduced_partitions();
        // if (debug) printf("[BDDC] done compute_reduced_partitions()\n");

        // if (debug) printf("[BDDC] allocate_vectors()...\n");
        allocate_vectors();
        // if (debug) printf("[BDDC] done allocate_vectors()\n");

        // if (debug) printf("[BDDC] create_cudss_solvers()...\n");
        create_cudss_solvers();
        // if (debug) printf("[BDDC] done create_cudss_solvers()\n");
    }

    void free() {
        // TBD
    }
    SDVec *createGamVec() { return new SDVec(ctx, part_gam, block_dim); }
    void set_print(bool print) {}
    void set_rel_tol(T rtol) {}
    void set_abs_tol(T atol) {}
    int get_num_iterations() { return 1; }
    void set_cycle_type(std::string cycle_) {}
    int get_num_IEV_nodes(int gpu) { return this->IEV_nnodes[gpu]; }
    int *get_IEV_conn(int gpu) { return d_IEV_elem_conn[gpu]; }
    T *get_IEV_xpts(int gpu) { return this->d_IEV_xpts->getLocalPtr(gpu); }
    T *get_IEV_vars(int gpu) { return this->d_IEV_vars->getLocalPtr(gpu); }
    SDPartition *get_part_gam() { return part_gam; }

    void update_after_assembly(Vec *vars) {
        // printf("[BDDC-update_after_assembly] copy vars\n");
        vars->copyTo(d_vars);
        // printf("[BDDC-update_after_assembly] assemble_subdomains\n");
        assemble_subdomains();
        // printf("[BDDC-update_after_assembly] subdomain_I_solver->factor()\n");
        subdomain_I_solver->factor();
        // printf("[BDDC-update_after_assembly] subdomain_IE_solver->factor()\n");
        subdomain_IE_solver->factor();
        // printf("[BDDC-update_after_assembly] assemble_coarse_problem\n");
        assemble_coarse_problem();
        // printf("[BDDC-update_after_assembly] Svv_solver->factor()\n");
        Svv_solver->factor();
        // printf("[BDDC-update_after_assembly] done with method\n");
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, Vec *vars) {
        // compute res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV

        // prelim debug
        // compare d_xpts, d_IE_nsd, d_vertex_nsd
        // T *h_xpts = DeviceVec<T>(3 * num_nodes, d_xpts->getLocalPtr(0)).createHostVec().getPtr();
        // printf("h_xpts: ");
        // printVec<T>(3 * num_nodes, h_xpts);
        // int *h_IE_nsd =
        //     DeviceVec<int>(this->IE_nnodes[0], this->d_IE_nsd[0]).createHostVec().getPtr();
        // printf("h_IE_nsd: ");
        // printVec<int>(this->IE_nnodes[0], h_IE_nsd);
        // int *h_vertex_nsd =
        //     DeviceVec<int>(this->Vc_nnodes[0], this->d_vertex_nsd[0]).createHostVec().getPtr();
        // printf("h_vertex_nsd: ");
        // printVec<int>(this->Vc_nnodes[0], h_vertex_nsd);

        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt1\n");
        addVec_globalToIEV(1.0, d_xpts, 0.0, d_IEV_xpts, 3);
        addVec_globalToIEV(1.0, d_vars, 0.0, d_IEV_vars, block_dim);

        // T *h_IEV_xpts =
        //     DeviceVec<T>(3 * IEV_nnodes[0], d_IEV_xpts->getLocalPtr(0)).createHostVec().getPtr();
        // printf("h_IEV_xpts on GPU[0]: ");
        // printVec<T>(3 * IEV_nnodes[0], h_IEV_xpts);

        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt1.2\n");
        fint_IEV->zeroAll();
        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt1.3\n");
        // we already computed it in local form, so no need to expandToLocal?
        // d_IEV_xpts->expandToLocal();
        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt1.4\n");
        // d_IEV_vars->expandToLocal();

        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt2\n");
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int *loc_elem_comps = d_loc_elem_components[g];
            int *loc_elem_conn_ptr = kmat_IEV->getLocalElemConn(g);

            T *loc_xpts_ptr = d_IEV_xpts->getLocalPtr(g);
            T *loc_vars_ptr = d_IEV_vars->getLocalPtr(g);
            Data *loc_comp_data_ptr = d_loc_comp_data[g];
            T *loc_fint_IEV = fint_IEV->getLocalPtr(g);

            dim3 block(num_quad_pts, elems_per_block);
            dim3 grid(this->num_elements);
            k_add_multigpu_residual_fast<T, elems_per_block, Assembler>
                <<<grid, block, 0, streams[g]>>>(IEV_nnodes[g], local_nelems[g], loc_elem_comps,
                                                 loc_elem_conn_ptr, loc_elem_conn_ptr, loc_xpts_ptr,
                                                 loc_vars_ptr, loc_comp_data_ptr, loc_fint_IEV);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        fint_IEV->reduceFromLocal();

        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt3\n");

        this->fint_IEV->apply_bcs(n_IEV_owned_bcs, d_IEV_owned_bcs, n_IEV_local_bcs,
                                  d_IEV_local_bcs);
        this->fint_IEV->scale(-lambdaI);
        this->fint_IEV->copyTo(this->res_IEV);
        res_IEV->axpy(lambdaE, fext_IEV);

        // if (debug) printf("[BDDC-set_IEV_residual]: checkpt4\n");
    }

    void get_lam_rhs(SDVec *gam_rhs) {
        // gets rhs of interface BDDC system
        gam_rhs->zeroAll();

        // harmonic extension from interface (Gam) to interior (I) nodes
        // if (debug) printf("[BDDC-get_lam_rhs]: vec copy\n");
        res_IEV->copyTo(f_IEV);
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: res_IEV\n");
        //     printDeviceNodeVec(IEV_nnodes[0], res_IEV->getPtr(0));
        // }
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: f_IEV\n");
        //     printDeviceNodeVec(IEV_nnodes[0], f_IEV->getPtr(0));
        // }

        // if (debug) printf("[BDDC-get_lam_rhs]: addVecIEVtoI\n");
        addVecIEVtoI(1.0, f_IEV, 0.0, f_I);
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: f_I\n");
        //     printDeviceNodeVec(I_nnodes[0], f_I->getPtr(0));
        // }

        // solve interior subdomain-parallel matrix
        // if (debug) printf("[BDDC-get_lam_rhs]: solveSubdomainI\n");
        solveSubdomainI(f_I, u_I);
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: u_I\n");
        //     printDeviceNodeVec(I_nnodes[0], u_I->getPtr(0));
        // }

        // harmonic extension from interior (I) to interface (Gam)
        // if (debug) printf("[BDDC-get_lam_rhs]: addVecItoIEV\n");
        addVecItoIEV(1.0, u_I, 0.0, u_IEV);
        // if (debug) printf("[BDDC-get_lam_rhs]: kmat_IEV->mult\n");
        kmat_IEV->mult(-1.0, u_IEV, 1.0, f_IEV);
        // if (debug) printf("[BDDC-get_lam_rhs]: addVecIEVtoGam\n");
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: f_IEV2\n");
        //     printDeviceNodeVec(IEV_nnodes[0], f_IEV->getPtr(0));
        // }

        addVecIEVtoGam(1.0, f_IEV, 0.0, gam_rhs);
        // if (debug) printf("[BDDC-get_lam_rhs]: done with method\n");
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: final_gam_rhs\n");
        //     printDeviceNodeVec(ngam[0], gam_rhs->getLocalPtr(0));
        // }
        // gam_rhs->reduceFromLocal();
        // gam_rhs->expandToLocal();
        // if (debug) printf("[BDDC-get_lam_rhs]: done with method\n");
        // if (debug) {
        //     printf("[BDDC-get_lam_rhs]: final_gam_rhs v2\n");
        //     printDeviceNodeVec(ngam[0], gam_rhs->getLocalPtr(0));
        // }
    }

    void mat_vec(SDVec *gam_in, SDVec *gam_out) {
        // gets K_{Gam,Gam}*x_{Gam} internal residual of interface BDDC system
        gam_out->zeroAll();
        // if (debug) {
        //     printf("[BDDC-mat_vec]: gam_in\n");
        //     printDeviceNodeVec(ngam[0], gam_in->getPtr(0));
        // }

        // harmonic extension from interface (Gam) to interior (I) nodes
        addVecGamtoIEV(1.0, gam_in, 0.0, u_IEV);
        // if (debug) {
        //     printf("[BDDC-mat_vec]: u_IEV\n");
        //     printDeviceNodeVec(IEV_nnodes[0], u_IEV->getPtr(0));
        // }
        kmat_IEV->mult(1.0, u_IEV, 0.0, f_IEV);
        addVecIEVtoI(1.0, f_IEV, 0.0, f_I);
        // if (debug) {
        //     printf("[BDDC-mat_vec]: f_I\n");
        //     printDeviceNodeVec(I_nnodes[0], f_I->getPtr(0));
        // }

        // solve interior (I) subdomain-parallel matrix
        solveSubdomainI(f_I, u_I);
        // if (debug) {
        //     printf("[BDDC-mat_vec]: u_I\n");
        //     printDeviceNodeVec(I_nnodes[0], u_I->getPtr(0));
        // }

        // harmonic extension from interior (I) to interface (Gam)
        addVecItoIEV(1.0, u_I, 0.0, u_IEV);
        kmat_IEV->mult(-1.0, u_IEV, 1.0, f_IEV);
        addVecIEVtoGam(1.0, f_IEV, 0.0, gam_out);
        // if (debug) {
        //     printf("[BDDC-mat_vec]: gam_out\n");
        //     printDeviceNodeVec(ngam[0], gam_out->getPtr(0));
        // }
    }

    bool solve(SDVec *gam_rhs, SDVec *gam, bool check_conv = false) {
        // gets preconditioner solve for interface BDDC system M_{gam}^{-1} * y_{Gam}

        // if (debug) {
        //     printf("[BDDC-solve]: gam_rhs\n");
        //     printDeviceNodeVec(ngam[0], gam_rhs->getPtr(0));
        // }

        // get coarse vertex (V) loads for later
        constexpr bool SCALED = true;
        addVecGamtoIEV<SCALED>(1.0, gam_rhs, 0.0, f_IEV);
        addVecIEVtoVc<SCALED>(1.0, f_IEV, 0.0, f_V);

        // if (debug) {
        //     printf("[BDDC-solve]: f_IEV\n");
        //     printDeviceNodeVec(IEV_nnodes[0], f_IEV->getPtr(0));
        // }

        // get IE load rhs with f_I = 0 and f_E neq 0 (only edge forces)
        addVecIEVtoIE(1.0, f_IEV, 0.0, f_IE);
        addVecIEtoIEV(-1.0, f_IE, 1.0, f_IEV);
        zeroInteriorIE(f_IE);

        // solve interior+edge (IE) subdomain-parallel matrix
        solveSubdomainIE(f_IE, u_IE);

        // adjust coarse vertex (V) loads from IE solution
        addVecIEtoIEV(1.0, u_IE, 0.0, u_IEV);
        kmat_IEV->mult(-1.0, u_IEV, 0.0, f_IEV);
        addVecIEVtoVc(1.0, f_IEV, 1.0, f_V);

        // solve coarse vertex (V) problem using Schur complement system
        // if (debug) {
        //     printf("[BDDC-solve]: f_V\n");
        //     printDeviceNodeVec(Vc_nnodes[0], f_V->getPtr(0));
        // }
        solveCoarse(f_V, u_V);
        u_V->expandToLocal();
        // if (debug) {
        //     printf("[BDDC-solve]: u_V\n");
        //     printDeviceNodeVec(Vc_nnodes[0], u_V->getPtr(0));
        // }

        // compute updated IE loads from coarse vertex (V) solution
        //    uses IEV system as intermediary
        addVecVctoIEV(1.0, u_V, 0.0, temp_IEV);
        kmat_IEV->mult(-1.0, temp_IEV, 0.0, f_IEV);
        addVecIEVtoIE(1.0, f_IEV, 0.0, f_IE);

        // solve interior+edge (IE) subdomain-parallel matrix again
        u_IE->zeroAll();
        solveSubdomainIE(f_IE, u_IE);

        // harmonic extension from IE to full interface solution (EV = gam) to end precond-solve
        addVecIEtoIEV(1.0, u_IE, 1.0, u_IEV);
        addVecVctoIEV<SCALED>(1.0, u_V, 1.0, u_IEV);
        addVecIEVtoGam<SCALED>(1.0, u_IEV, 0.0, gam);

        // if (debug) {
        //     printf("[BDDC-solve]: gam\n");
        //     printDeviceNodeVec(ngam[0], gam->getPtr(0));
        // }

        return false;
    }

    void get_global_soln(SDVec *gam_soln, Vec *soln) {
        // recover global solution from interface DOF
        soln->zeroAll();

        // add E and V values from interface (gam) to IEV soln and coarse vertex (V) soln
        addVecGamtoIEV(1.0, gam_soln, 0.0, u_IEV);
        const bool SCALED = true;
        addVecIEVtoVc<SCALED>(1.0, u_IEV, 0.0, u_V);

        // E + V already solved, so what remains is to solve the I nodes of each subdomain
        // compute the updated IEV then I residual (or remaining forces)
        addVecIEVtoI(1.0, res_IEV, 0.0, f_I);
        kmat_IEV->mult(-1.0, u_IEV, 0.0, f_IEV);
        addVecIEVtoI(1.0, f_IEV, 1.0, f_I);

        // solve the interior (I) nodes with subdomain-parallel matrix
        solveSubdomainI(f_I, u_I);

        // add the solved interior (I) nodes into IE vec
        addVecItoIE(1.0, u_I, 0.0, u_IE);

        // add E DOF from IEV gam into u_IE
        // (since more convenient to go from IE + V => soln than IEV=>soln?)
        addVecIEVtoIE(1.0, u_IEV, 1.0, u_IE);

        // add IE + V parts together reducing into global soln on original FEA mesh
        addGlobalSoln(u_IE, u_V, soln);
    }

    template <class LoadMagnitude>
    void add_subdomain_fext(const LoadMagnitude &load, T load_mag, T x_inplane_frac = 0.0) {
        fext_IEV->zeroAll();
        const int elems_per_block = 8;
        addVec_globalToIEV(1.0, d_xpts, 0.0, d_IEV_xpts, 3);
        addVec_globalToIEV(1.0, d_vars, 0.0, d_IEV_vars, block_dim);

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_IEV_xpts->getExpandedNodes(g);
            int loc_nelems = part->getLocalNumElements(g);

            // can't use Vec objects here since not same number of nodes?
            // or I could create it from IEV_elem_conn partitioners? we'll see
            T *loc_xpts_ptr = d_IEV_xpts->getLocalPtr(g);
            T *load_fext_ptr = fext_IEV->getLocalPtr(g);
            int *loc_elem_comps = d_loc_elem_components[g];
            Data *loc_comp_data_ptr = d_loc_comp_data[g];

            // local element connectivity, used for both rows and columns
            int *loc_elem_conn_ptr = kmat_IEV->getLocalElemConn(g);
            // int *loc_elem_ind_map = kmat_IEV->getLocalElemIndMap(g);
            // T *loc_mat_vals = kmat_IEV->getLocalVals(g);

            dim3 block(num_quad_pts, elems_per_block);
            dim3 grid(loc_nelems);

            k_add_multigpu_fext_fast<T, elems_per_block, Assembler, LoadMagnitude>
                <<<grid, block, 0, streams[g]>>>(
                    loc_nelems, load, loc_elem_comps, loc_elem_conn_ptr, loc_elem_conn_ptr,
                    loc_xpts_ptr, loc_comp_data_ptr, load_mag, load_fext_ptr, x_inplane_frac);

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        fext_IEV->reduceFromLocal();
        fext_IEV->apply_bcs(n_IEV_owned_bcs, d_IEV_owned_bcs, n_IEV_local_bcs, d_IEV_local_bcs);
        fext_IEV->copyTo(res_IEV);
    }

   private:
    static int *copy_vec(const std::vector<int> &v) {
        if (v.empty()) return nullptr;

        int *out = new int[v.size()];
        std::memcpy(out, v.data(), v.size() * sizeof(int));
        return out;
    }

    void import_splitting() {
        // if (debug) printf("[BDDC-import_splitting] begin\n");

        // sgpu = single GPU
        sgpu_num_subdomains = split->num_subdomains;

        // Original/single-GPU splitting copied directly from split
        sgpu_I_nnodes = split->I_nnodes;
        sgpu_IE_nnodes = split->IE_nnodes;
        sgpu_IEV_nnodes = split->IEV_nnodes;
        sgpu_Vc_nnodes = split->Vc_nnodes;
        sgpu_V_nnodes = split->V_nnodes;
        sgpu_lam_nnodes = split->lam_nnodes;

        // if (debug) {
        //     printf(
        //         "[BDDC-import_splitting] "
        //         "nsub=%d I=%d IE=%d IEV=%d Vc=%d V=%d lam=%d\n",
        //         sgpu_num_subdomains, sgpu_I_nnodes, sgpu_IE_nnodes, sgpu_IEV_nnodes,
        //         sgpu_Vc_nnodes, sgpu_V_nnodes, sgpu_lam_nnodes);
        // }

        sgpu_elem_sd_ind = copy_vec(split->elem_sd_ind);
        sgpu_node_class_ind = copy_vec(split->node_class_ind);
        sgpu_node_nsd = copy_vec(split->node_nsd);

        // if (debug) printf("[BDDC-import_splitting] copied elem/node classification arrays\n");

        sgpu_IEV_sd_ptr = copy_vec(split->IEV_sd_ptr);
        sgpu_IEV_sd_ind = copy_vec(split->IEV_sd_ind);
        sgpu_IEV_nodes = copy_vec(split->IEV_nodes);
        sgpu_IEV_elem_conn = copy_vec(split->IEV_elem_conn);

        // if (debug) {
        //     printf("Fine BDDC IEV_conn: ");
        //     printVec<int>(nodes_per_elem * num_elements, sgpu_IEV_elem_conn);
        //     printf("elem_sd_ind: ");
        //     printVec<int>(num_elements, sgpu_elem_sd_ind);
        // }

        // if (debug) {
        //     printf(
        //         "[BDDC-import_splitting] copied IEV arrays "
        //         "(total_iev=%d)\n",
        //         sgpu_IEV_sd_ptr[sgpu_num_subdomains]);
        // }

        // if (debug) {
        //     printf(
        //         "[BDDC-import_splitting] build part_IEV "
        //         "(ngpus=%d nnodes=%d nelems=%d)\n",
        //         ngpus, sgpu_IEV_nnodes, num_elements);
        // }

        part_IEV =
            new Partition(ngpus, sgpu_IEV_nnodes, num_elements, nodes_per_elem, sgpu_IEV_elem_conn,
                          part->num_components, part->h_elem_components, false);

        // if (debug) {
        //     for (int g = 0; g < ngpus; g++) {
        //         printf(
        //             "[BDDC-import_splitting] part_IEV gpu %d: "
        //             "nnodes=%d nelems=%d\n",
        //             g, part_IEV->local_nnodes[g], part_IEV->local_nelems[g]);
        //     }
        // }

        // if (debug) printf("[BDDC-import_splitting] create_multigpu_splitting()\n");

        create_multigpu_splitting();

        // if (debug) printf("[BDDC-import_splitting] allocate IEV vectors\n");

        // values for d_IEV_xpts + d_IEV_vars set using vec maps + add kernels
        d_IEV_xpts = new Vec(ctx, part_IEV, 3);
        d_IEV_vars = new Vec(ctx, part_IEV, block_dim);

        // if (debug) printf("[BDDC-import_splitting] done\n");

        // mat_IEV = new Mat(ctx, part_IEV, block_dim);
    }

    void create_multigpu_splitting() {
        // if (debug) printf("[BDDC-create_multigpu_splitting] start\n");

        // determine which subdomains belong to which GPUs from the partition..
        // if (debug) printf("[BDDC-create_multigpu_splitting] assigning subdomains to GPUs\n");

        subdomain_gpu_ind = new int[sgpu_num_subdomains];
        memset(subdomain_gpu_ind, -1, sgpu_num_subdomains * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int gpu = part->find_owned_gpu_from_elem(e);
            subdomain_gpu_ind[s] = gpu;
        }

        // get local nnodes and nelems on each GPU
        // if (debug) printf("[BDDC-create_multigpu_splitting] copying local sizes\n");

        local_nnodes = new int[ngpus];
        local_nelems = new int[ngpus];

        for (int g = 0; g < ngpus; g++) {
            local_nnodes[g] = part->local_nnodes[g];
            local_nelems[g] = part->local_nelems[g];

            // if (debug) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] gpu %d: local_nnodes=%d "
            //         "local_nelems=%d\n",
            //         g, local_nnodes[g], local_nelems[g]);
            // }
        }

        // compute glob to local elem map
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] building global-to-local "
        //         "element maps\n");

        int *elem_ctr = new int[ngpus];
        memset(elem_ctr, 0, ngpus * sizeof(int));

        glob_loc_elem_map = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            glob_loc_elem_map[g] = new int[num_elements];
            memset(glob_loc_elem_map[g], -1, num_elements * sizeof(int));
        }

        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int g = subdomain_gpu_ind[s];

            // if (debug && (g < 0 || g >= ngpus)) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] ERROR: bad gpu=%d for "
            //         "e=%d s=%d\n",
            //         g, e, s);
            // }

            int ered = elem_ctr[g]++;
            glob_loc_elem_map[g][e] = ered;
        }

        // for (int g = 0; g < ngpus; g++) {
        //     if (debug) {
        //         printf(
        //             "[BDDC-create_multigpu_splitting] gpu %d elem_ctr=%d "
        //             "expected local_nelems=%d\n",
        //             g, elem_ctr[g], local_nelems[g]);
        //     }
        // }

        // compute elem_sd_ind on each local GPU
        // if (debug) printf("[BDDC-create_multigpu_splitting] building elem_sd_ind\n");

        elem_sd_ind = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            elem_sd_ind[g] = new int[local_nelems[g]];
        }

        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int g = subdomain_gpu_ind[s];
            int ered = glob_loc_elem_map[g][e];
            elem_sd_ind[g][ered] = s;
        }

        // get num subdomains in each GPU
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] counting reduced "
        //         "subdomains\n");

        num_subdomains = new int[ngpus];

        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> subdomains;

            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int s = elem_sd_ind[g][ered];
                subdomains.insert(s);
            }

            num_subdomains[g] = subdomains.size();

            // if (debug) {
            //     printf("[BDDC-create_multigpu_splitting] gpu %d num_subdomains=%d\n", g,
            //            num_subdomains[g]);
            // }
        }

        sd_cts = new int[ngpus];
        std::memset(sd_cts, 0, ngpus * sizeof(int));

        red_subdomains = new int *[ngpus];
        std::memset(red_subdomains, 0, ngpus * sizeof(int *));

        // if (debug) printf("[BDDC-create_multigpu_splitting] building red_subdomains\n");

        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> subdomains;

            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int s = elem_sd_ind[g][ered];
                subdomains.insert(s);
            }

            std::vector<int> sd_vec(subdomains.begin(), subdomains.end());
            std::sort(sd_vec.begin(), sd_vec.end());

            sd_cts[g] = static_cast<int>(sd_vec.size());
            red_subdomains[g] = new int[sd_cts[g]];

            for (int sred = 0; sred < sd_cts[g]; sred++) {
                red_subdomains[g][sred] = sd_vec[sred];
            }
        }

        // classify the nodes
        // if (debug) printf("[BDDC-create_multigpu_splitting] classifying nodes\n");

        node_class_ind = new int *[ngpus];
        node_nsd = new int *[ngpus];

        I_nnodes = new int[ngpus];
        IE_nnodes = new int[ngpus];
        IEV_nnodes = new int[ngpus];
        Vc_nnodes = new int[ngpus];
        V_nnodes = new int[ngpus];
        lam_nnodes = new int[ngpus];
        ngam = new int[ngpus];

        for (int g = 0; g < ngpus; g++) {
            node_class_ind[g] = new int[local_nnodes[g]];
            node_nsd[g] = new int[local_nnodes[g]];

            I_nnodes[g] = 0;
            IE_nnodes[g] = 0;
            IEV_nnodes[g] = 0;
            Vc_nnodes[g] = 0;
            V_nnodes[g] = 0;
            lam_nnodes[g] = 0;

            for (int l = 0; l < local_nnodes[g]; l++) {
                int n = part->h_local_nodes[g][l];
                int node_class = sgpu_node_class_ind[n];
                int nsd = sgpu_node_nsd[n];

                node_class_ind[g][l] = node_class;
                node_nsd[g][l] = nsd;

                if (node_class == IEV_INTERIOR) {
                    I_nnodes[g] += 1;
                    IE_nnodes[g] += 1;
                    IEV_nnodes[g] += 1;
                } else if (node_class == IEV_EDGE) {
                    lam_nnodes[g] += 1;
                    IE_nnodes[g] += nsd;
                    IEV_nnodes[g] += nsd;
                } else {
                    Vc_nnodes[g] += 1;
                    V_nnodes[g] += nsd;
                    IEV_nnodes[g] += nsd;
                }
            }

            ngam[g] = lam_nnodes[g] + Vc_nnodes[g];

            // if (debug) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] gpu %d: "
            //         "I=%d IE=%d IEV=%d Vc=%d V=%d lam=%d ngam=%d\n",
            //         g, I_nnodes[g], IE_nnodes[g], IEV_nnodes[g], Vc_nnodes[g], V_nnodes[g],
            //         lam_nnodes[g], ngam[g]);
            // }
        }

        // build IEV arrays
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] building IEV_nodes and "
        //         "IEV_loc_to_glob\n");

        IEV_nodes = new int *[ngpus];
        IEV_loc_to_glob = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IEV_nodes[g] = new int[IEV_nnodes[g]];
            IEV_loc_to_glob[g] = new int[IEV_nnodes[g]];
        }

        // if (debug) {
        //     printf("IEV_nodes on GPU[0]: ");
        //     printVec<int>(IEV_nnodes[0], IEV_nodes[0]);
        // }

        int *IEV_cts = new int[ngpus];
        memset(IEV_cts, 0, ngpus * sizeof(int));

        for (int s = 0; s < sgpu_num_subdomains; s++) {
            int gpu = subdomain_gpu_ind[s];

            // if (debug && (gpu < 0 || gpu >= ngpus)) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] ERROR: bad gpu=%d for "
            //         "subdomain s=%d\n",
            //         gpu, s);
            // }

            for (int iev = sgpu_IEV_sd_ptr[s]; iev < sgpu_IEV_sd_ptr[s + 1]; iev++) {
                int n = sgpu_IEV_nodes[iev];
                int iev_red = IEV_cts[gpu]++;

                int n_red = part->global_to_local[gpu][n];

                IEV_nodes[gpu][iev_red] = n_red;
                IEV_loc_to_glob[gpu][iev_red] = iev;
            }
        }

        // for (int g = 0; g < ngpus; g++) {
        //     if (debug) {
        //         printf(
        //             "[BDDC-create_multigpu_splitting] gpu %d IEV_cts=%d "
        //             "expected IEV_nnodes=%d\n",
        //             g, IEV_cts[g], IEV_nnodes[g]);
        //     }
        // }

        // fill out IEV_sd_ptr
        // if (debug) printf("[BDDC-create_multigpu_splitting] building IEV_sd_ptr\n");

        IEV_sd_ptr = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IEV_sd_ptr[g] = new int[num_subdomains[g] + 1];

            memset(IEV_sd_ptr[g], 0, (num_subdomains[g] + 1) * sizeof(int));

            for (int sred = 0; sred < num_subdomains[g]; sred++) {
                int s = red_subdomains[g][sred];
                int d_iev = sgpu_IEV_sd_ptr[s + 1] - sgpu_IEV_sd_ptr[s];

                IEV_sd_ptr[g][sred + 1] = IEV_sd_ptr[g][sred] + d_iev;
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] gpu %d "
            //         "IEV_sd_ptr end=%d\n",
            //         g, IEV_sd_ptr[g][num_subdomains[g]]);
            // }
        }

        // then fill out IEV_elem_conn
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] building "
        //         "IEV_elem_conn and elem maps\n");

        IEV_elem_conn = new int *[ngpus];
        elem_loc_to_glob = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IEV_elem_conn[g] = new int[nodes_per_elem * local_nelems[g]];
            elem_loc_to_glob[g] = new int[local_nelems[g]];
        }

        int *elem_red_cts = new int[ngpus];
        memset(elem_red_cts, 0, ngpus * sizeof(int));

        elem_glob_to_loc = new int[num_elements];

        for (int e = 0; e < num_elements; e++) {
            int g = part->h_elem_assigned_gpu[e];

            int ered = elem_red_cts[g]++;

            elem_glob_to_loc[e] = ered;
            elem_loc_to_glob[g][ered] = e;
        }

        // for (int g = 0; g < ngpus; g++) {
        //     if (debug) {
        //         printf(
        //             "[BDDC-create_multigpu_splitting] gpu %d "
        //             "elem_red_cts=%d expected local_nelems=%d\n",
        //             g, elem_red_cts[g], local_nelems[g]);
        //     }
        // }

        // now fill out the IEV_elem_conn
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] filling "
        //         "IEV_elem_conn\n");

        for (int g = 0; g < ngpus; g++) {
            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int e = elem_loc_to_glob[g][ered];

                int *sgpu_lnodes = &sgpu_IEV_elem_conn[nodes_per_elem * e];

                int *lnodes = &IEV_elem_conn[g][nodes_per_elem * ered];

                for (int l = 0; l < nodes_per_elem; l++) {
                    int n = sgpu_lnodes[l];
                    int nred = part->global_to_local[g][n];
                    lnodes[l] = nred;
                }
            }
        }

        // move IEV_elem_conn to device
        // if (debug)
        //     printf(
        //         "[BDDC-create_multigpu_splitting] moving "
        //         "IEV_elem_conn to device\n");

        d_IEV_elem_conn = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            d_IEV_elem_conn[g] = HostVec<int>(local_nelems[g] * nodes_per_elem, IEV_elem_conn[g])
                                     .createDeviceVec()
                                     .getPtr();

            // if (debug) {
            //     printf(
            //         "[BDDC-create_multigpu_splitting] gpu %d "
            //         "copied d_IEV_elem_conn\n",
            //         g);
            // }
        }

        delete[] elem_ctr;
        delete[] IEV_cts;
        delete[] elem_red_cts;

        // if (debug) printf("[BDDC-create_multigpu_splitting] done\n");
    }

    void build_IE_I_V_maps() {
        // if (debug) printf("[BDDC-build_IE_I_V_maps] begin\n");

        IE_nodes = new int *[ngpus];
        I_nodes = new int *[ngpus];

        IEVtoIE_map = new int *[ngpus];
        IEVtoIE_imap = new int *[ngpus];

        IEVtoI_map = new int *[ngpus];
        IEVtoI_imap = new int *[ngpus];

        IE_interior = new bool *[ngpus];
        IE_general_edge = new bool *[ngpus];

        d_IE_interior = new bool *[ngpus];
        d_IE_general_edge = new bool *[ngpus];

        d_IE_nodes = new int *[ngpus];
        d_IEVtoIE_imap = new int *[ngpus];
        d_IEVtoI_imap = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IE_nodes[g] = new int[IE_nnodes[g]];
            I_nodes[g] = new int[I_nnodes[g]];

            IEVtoIE_map[g] = new int[IEV_nnodes[g]];
            IEVtoIE_imap[g] = new int[IE_nnodes[g]];

            IEVtoI_map[g] = new int[IEV_nnodes[g]];
            IEVtoI_imap[g] = new int[I_nnodes[g]];

            IE_interior[g] = new bool[IE_nnodes[g]];
            IE_general_edge[g] = new bool[IE_nnodes[g]];

            std::memset(IEVtoIE_map[g], -1, IEV_nnodes[g] * sizeof(int));

            std::memset(IEVtoI_map[g], -1, IEV_nnodes[g] * sizeof(int));

            std::memset(IEVtoIE_imap[g], -1, IE_nnodes[g] * sizeof(int));

            std::memset(IEVtoI_imap[g], -1, I_nnodes[g] * sizeof(int));

            int IE_ind = 0;
            int I_ind = 0;

            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int gnode = IEV_nodes[g][iev];
                int cls = node_class_ind[g][gnode];

                bool is_I = (cls == IEV_INTERIOR || cls == IEV_DIRICHLET_EDGE);
                bool is_IE = is_I || cls == IEV_EDGE;

                if (is_IE) {
                    IE_interior[g][IE_ind] = cls == IEV_INTERIOR || cls == IEV_DIRICHLET_EDGE;

                    IE_general_edge[g][IE_ind] = cls == IEV_INTERIOR || cls == IEV_DIRICHLET_EDGE;

                    IE_nodes[g][IE_ind] = gnode;

                    IEVtoIE_map[g][iev] = IE_ind;
                    IEVtoIE_imap[g][IE_ind] = iev;

                    IE_ind++;
                }

                if (is_I) {
                    I_nodes[g][I_ind] = gnode;

                    IEVtoI_map[g][iev] = I_ind;
                    IEVtoI_imap[g][I_ind] = iev;

                    I_ind++;
                }
            }

            // printf("IE_interior %d: ", IE_nnodes[0]);
            // printVec<bool>(IE_nnodes[0], IE_interior[0]);

            d_IE_interior[g] =
                HostVec<bool>(IE_nnodes[g], IE_interior[g]).createDeviceVec().getPtr();
            d_IE_general_edge[g] =
                HostVec<bool>(IE_nnodes[g], IE_general_edge[g]).createDeviceVec().getPtr();
            d_IE_nodes[g] = HostVec<int>(IE_nnodes[g], IE_nodes[g]).createDeviceVec().getPtr();
            d_IEVtoIE_imap[g] =
                HostVec<int>(IE_nnodes[g], IEVtoIE_imap[g]).createDeviceVec().getPtr();
            d_IEVtoI_imap[g] = HostVec<int>(I_nnodes[g], IEVtoI_imap[g]).createDeviceVec().getPtr();

            // if (debug) {
            //     printf(
            //         "[BDDC-build_IE_I_V_maps] gpu %d: "
            //         "IE=%d I=%d IEV=%d\n",
            //         g, IE_ind, I_ind, IEV_nnodes[g]);
            // }
        }

        // if (debug) {
        //     printf("IE_nodes on GPU[0]: ");
        //     printVec<int>(IE_nnodes[0], IE_nodes[0]);
        //     printf("I_nodes on GPU[0]: ");
        //     printVec<int>(I_nnodes[0], I_nodes[0]);
        // }

        // if (debug) printf("[BDDC-build_IE_I_V_maps] build_Vc_and_gam_maps()\n");

        build_Vc_and_gam_maps();

        // if (debug) printf("[BDDC-build_IE_I_V_maps] done\n");
    }

    void build_Vc_and_gam_maps() {
        // if (debug) printf("[BDDC-build_Vc_and_gam_maps] begin\n");

        Vc_nodes = new int *[ngpus];
        Vc_inodes = new int *[ngpus];

        IEVtoV_imap = new int *[ngpus];
        VctoV_imap = new int *[ngpus];

        d_IEVtoV_imap = new int *[ngpus];
        d_VctoV_imap = new int *[ngpus];

        n_edge = new int[ngpus];
        ngam = new int[ngpus];

        gam_nodes = new int *[ngpus];
        d_Vc_nodes = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            // std::unordered_set<int> Vc_set;

            // for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
            //     int lnode = IEV_nodes[g][iev];

            //     if (node_class_ind[g][lnode] == IEV_VERTEX) {
            //         Vc_set.insert(lnode);
            //     }
            // }

            // std::vector<int> Vc_vec(Vc_set.begin(), Vc_set.end());
            // std::sort(Vc_vec.begin(), Vc_vec.end());

            // // if (debug) {
            // //     printf(
            // //         "[BDDC-build_Vc_and_gam_maps] gpu %d: "
            // //         "Vc_set=%d expected_Vc=%d V=%d lam=%d\n",
            // //         g, (int)Vc_vec.size(), Vc_nnodes[g], V_nnodes[g], lam_nnodes[g]);
            // // }

            // Vc_nodes[g] = new int[Vc_nnodes[g]];
            // Vc_inodes[g] = new int[local_nnodes[g]];

            // unordered set code above destroys IEV discovery order..
            std::vector<int> Vc_vec;
            Vc_vec.reserve(Vc_nnodes[g]);

            std::vector<char> seen(local_nnodes[g], 0);

            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int lnode = IEV_nodes[g][iev];

                if (node_class_ind[g][lnode] == IEV_VERTEX && !seen[lnode]) {
                    seen[lnode] = 1;
                    Vc_vec.push_back(lnode);  // preserves first occurrence order in IEV_nodes
                }
            }

            // Optional sanity check
            if ((int)Vc_vec.size() != Vc_nnodes[g]) {
                printf("ERROR gpu %d: Vc_vec.size=%d expected=%d\n", g, (int)Vc_vec.size(),
                       Vc_nnodes[g]);
                std::abort();
            }

            Vc_nodes[g] = new int[Vc_nnodes[g]];
            Vc_inodes[g] = new int[local_nnodes[g]];

            std::memset(Vc_inodes[g], -1, local_nnodes[g] * sizeof(int));

            for (int i = 0; i < (int)Vc_vec.size(); i++) {
                int lnode = Vc_vec[i];
                Vc_nodes[g][i] = lnode;
                Vc_inodes[g][lnode] = i;
            }

            IEVtoV_imap[g] = new int[V_nnodes[g]];
            VctoV_imap[g] = new int[V_nnodes[g]];

            std::memset(IEVtoV_imap[g], -1, V_nnodes[g] * sizeof(int));
            std::memset(VctoV_imap[g], -1, V_nnodes[g] * sizeof(int));

            int V_ind = 0;

            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int lnode = IEV_nodes[g][iev];

                if (node_class_ind[g][lnode] == IEV_VERTEX) {
                    IEVtoV_imap[g][V_ind] = iev;
                    VctoV_imap[g][V_ind] = Vc_inodes[g][lnode];
                    V_ind++;
                }
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-build_Vc_and_gam_maps] gpu %d: "
            //         "V_ind=%d expected_V=%d\n",
            //         g, V_ind, V_nnodes[g]);
            // }

            d_IEVtoV_imap[g] = HostVec<int>(V_nnodes[g], IEVtoV_imap[g]).createDeviceVec().getPtr();

            d_VctoV_imap[g] = HostVec<int>(V_nnodes[g], VctoV_imap[g]).createDeviceVec().getPtr();

            n_edge[g] = lam_nnodes[g];
            ngam[g] = n_edge[g] + Vc_nnodes[g];

            gam_nodes[g] = new int[ngam[g]];

            int e = 0;

            for (int inode = 0; inode < local_nnodes[g]; inode++) {
                if (node_class_ind[g][inode] == IEV_EDGE) {
                    if (e < n_edge[g]) {
                        gam_nodes[g][e++] = inode;
                    }
                }
            }

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                gam_nodes[g][n_edge[g] + i] = Vc_nodes[g][i];
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-build_Vc_and_gam_maps] gpu %d: "
            //         "edge=%d expected_edge=%d ngam=%d\n",
            //         g, e, n_edge[g], ngam[g]);
            // }

            d_Vc_nodes[g] = HostVec<int>(Vc_nnodes[g], Vc_nodes[g]).createDeviceVec().getPtr();
        }

        // if (debug) {
        //     printf("Vc_nodes on GPU[0]: ");
        //     printVec<int>(Vc_nnodes[0], Vc_nodes[0]);
        // }

        // if (debug) printf("[BDDC-build_Vc_and_gam_maps] done\n");
    }

    void build_IEV_sparsity() {
        // if (debug) printf("[BDDC-build_IEV_sparsity] begin\n");

        // build IEV kmat and then extract host matrix pointers out of it
        kmat_IEV = new Mat(ctx, part_IEV, block_dim);

        d_IEV_vals = new DeviceVec<T>[ngpus];

        IEV_rowp = new int *[ngpus];
        IEV_cols = new int *[ngpus];
        IEV_nnzb = new int[ngpus];
        IEV_rows = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IEV_rowp[g] = kmat_IEV->getHostLocalRowp(g);
            IEV_cols[g] = kmat_IEV->getHostLocalCols(g);
            IEV_nnzb[g] = kmat_IEV->getLocalNumNonzeroBlocks(g);

            IEV_rows[g] = new int[IEV_nnzb[g]];

            for (int i = 0; i < IEV_nnodes[g]; i++) {
                for (int jp = IEV_rowp[g][i]; jp < IEV_rowp[g][i + 1]; jp++) {
                    IEV_rows[g][jp] = i;
                }
            }

            d_IEV_vals[g] =
                DeviceVec<T>(kmat_IEV->getLocalNumNonzeros(g), kmat_IEV->getLocalVals(g));

            // if (debug) {
            //     printf(
            //         "[BDDC-build_IEV_sparsity] gpu %d: "
            //         "IEV_nnodes=%d nnzb=%d nnz=%d\n",
            //         g, IEV_nnodes[g], IEV_nnzb[g], kmat_IEV->getLocalNumNonzeros(g));
            // }
        }

        // if (debug) printf("[BDDC-build_IEV_sparsity] done\n");
    }

    void build_IE_and_I_sparsity() {
        // if (debug) printf("[BDDC-build_IE_and_I_sparsity] begin\n");

        IE_rowp = new int *[ngpus];
        I_rowp = new int *[ngpus];

        IE_cols = new int *[ngpus];
        I_cols = new int *[ngpus];

        I_nnzb = new int[ngpus];
        IE_nnzb = new int[ngpus];

        IE_rows = new int *[ngpus];
        I_rows = new int *[ngpus];

        d_IE_vals = new DeviceVec<T>[ngpus];
        d_I_vals = new DeviceVec<T>[ngpus];

        d_IE_vals_ptr = new T *[ngpus];
        d_I_vals_ptr = new T *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IE_rowp[g] = new int[IE_nnodes[g] + 1];
            I_rowp[g] = new int[I_nnodes[g] + 1];

            memset(IE_rowp[g], 0, (IE_nnodes[g] + 1) * sizeof(int));
            memset(I_rowp[g], 0, (I_nnodes[g] + 1) * sizeof(int));

            int IE_row = 0;
            int I_row = 0;

            for (int row = 0; row < IEV_nnodes[g]; row++) {
                int gnode_row = IEV_nodes[g][row];
                int class_row = node_class_ind[g][gnode_row];
                bool typeI_row = (class_row == IEV_INTERIOR || class_row == IEV_DIRICHLET_EDGE);
                bool typeIE_row = (typeI_row || class_row == IEV_EDGE);

                if (typeI_row) I_rowp[g][I_row + 1] = I_rowp[g][I_row];
                if (typeIE_row) IE_rowp[g][IE_row + 1] = IE_rowp[g][IE_row];

                for (int jp = IEV_rowp[g][row]; jp < IEV_rowp[g][row + 1]; jp++) {
                    int col = IEV_cols[g][jp];
                    int gnode_col = IEV_nodes[g][col];
                    int class_col = node_class_ind[g][gnode_col];
                    bool typeI_col = (class_col == IEV_INTERIOR || class_col == IEV_DIRICHLET_EDGE);
                    bool typeIE_col = (typeI_col || class_col == IEV_EDGE);

                    if (typeI_row && typeI_col) I_rowp[g][I_row + 1]++;
                    if (typeIE_row && typeIE_col) IE_rowp[g][IE_row + 1]++;
                }

                if (typeI_row) I_row++;
                if (typeIE_row) IE_row++;
            }

            I_nnzb[g] = I_rowp[g][I_nnodes[g]];
            IE_nnzb[g] = IE_rowp[g][IE_nnodes[g]];

            IE_rows[g] = new int[IE_nnzb[g]];
            I_rows[g] = new int[I_nnzb[g]];

            for (int g = 0; g < ngpus; g++) {
                for (int inode = 0; inode < IE_nnodes[g]; inode++) {
                    for (int jp = IE_rowp[g][inode]; jp < IE_rowp[g][inode + 1]; jp++) {
                        IE_rows[g][jp] = inode;
                    }
                }
                for (int inode = 0; inode < I_nnodes[g]; inode++) {
                    for (int jp = I_rowp[g][inode]; jp < I_rowp[g][inode + 1]; jp++) {
                        I_rows[g][jp] = inode;
                    }
                }
            }

            IE_cols[g] = new int[IE_nnzb[g]];
            I_cols[g] = new int[I_nnzb[g]];
            int I_ind = 0, IE_ind = 0;

            I_ind = 0;
            IE_ind = 0;
            for (int row = 0; row < IEV_nnodes[g]; row++) {
                int gnode_row = IEV_nodes[g][row];
                int class_row = node_class_ind[g][gnode_row];
                bool typeI_row = (class_row == IEV_INTERIOR || class_row == IEV_DIRICHLET_EDGE);
                bool typeIE_row = (typeI_row || class_row == IEV_EDGE);

                for (int jp = IEV_rowp[g][row]; jp < IEV_rowp[g][row + 1]; jp++) {
                    int col = IEV_cols[g][jp];
                    int gnode_col = IEV_nodes[g][col];
                    int class_col = node_class_ind[g][gnode_col];
                    bool typeI_col = (class_col == IEV_INTERIOR || class_col == IEV_DIRICHLET_EDGE);
                    bool typeIE_col = (typeI_col || class_col == IEV_EDGE);

                    if (typeI_row && typeI_col) {
                        I_cols[g][I_ind++] = IEVtoI_map[g][col];
                    }
                    if (typeIE_row && typeIE_col) {
                        IE_cols[g][IE_ind++] = IEVtoIE_map[g][col];
                    }
                }
            }

            d_IE_vals[g] = DeviceVec<T>(block_dim2 * IE_nnzb[g]);
            d_I_vals[g] = DeviceVec<T>(block_dim2 * I_nnzb[g]);

            d_IE_vals_ptr[g] = d_IE_vals[g].getPtr();
            d_I_vals_ptr[g] = d_I_vals[g].getPtr();

            // if (debug) {
            //     printf(
            //         "[BDDC-build_IE_and_I_sparsity] gpu %d: "
            //         "I=%d IE=%d I_nnzb=%d IE_nnzb=%d\n",
            //         g, I_nnodes[g], IE_nnodes[g], I_nnzb[g], IE_nnzb[g]);
            // }
        }

        // if (debug) printf("[BDDC-build_IE_and_I_sparsity] done\n");
    }

    void create_kmat_copy_maps() {
        // if (debug) printf("[BDDC-create_kmat_copy_maps] begin\n");

        // -----------------------------------------
        // IEV => IE kmat block copy map
        // -----------------------------------------
        kmat_IEnofill_map = new int *[ngpus];
        kmat_IEtoIEV_map = new int *[ngpus];
        d_kmat_IEnofill_map = new int *[ngpus];
        d_kmat_IEtoIEV_map = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            kmat_IEnofill_map[g] = new int[IE_nnzb[g]];
            kmat_IEtoIEV_map[g] = new int[IE_nnzb[g]];

            memset(kmat_IEnofill_map[g], -1, IE_nnzb[g] * sizeof(int));
            memset(kmat_IEtoIEV_map[g], -1, IE_nnzb[g] * sizeof(int));

            int nofill_ind = 0;
            int missing = 0;

            for (int i = 0; i < IE_nnodes[g]; i++) {
                int i_IEV = IEVtoIE_imap[g][i];

                for (int jp = IE_rowp[g][i]; jp < IE_rowp[g][i + 1]; jp++) {
                    int j = IE_cols[g][jp];
                    int j_IEV = IEVtoIE_imap[g][j];

                    bool found = false;

                    for (int kp = IEV_rowp[g][i_IEV]; kp < IEV_rowp[g][i_IEV + 1]; kp++) {
                        int k = IEV_cols[g][kp];

                        if (k == j_IEV) {
                            kmat_IEnofill_map[g][nofill_ind] = jp;
                            kmat_IEtoIEV_map[g][nofill_ind] = kp;
                            nofill_ind++;
                            found = true;
                            break;
                        }
                    }

                    if (!found) missing++;
                }
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-create_kmat_copy_maps] gpu %d IE: "
            //         "rows=%d nnzb=%d copied=%d missing=%d\n",
            //         g, IE_nnodes[g], IE_nnzb[g], nofill_ind, missing);
            // }

            d_kmat_IEtoIEV_map[g] =
                HostVec<int>(IE_nnzb[g], kmat_IEtoIEV_map[g]).createDeviceVec().getPtr();

            d_kmat_IEnofill_map[g] =
                HostVec<int>(IE_nnzb[g], kmat_IEnofill_map[g]).createDeviceVec().getPtr();
        }

        // -----------------------------------------
        // IEV => I kmat block copy map
        // -----------------------------------------
        kmat_Inofill_map = new int *[ngpus];
        kmat_ItoIEV_map = new int *[ngpus];
        d_kmat_Inofill_map = new int *[ngpus];
        d_kmat_ItoIEV_map = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            kmat_Inofill_map[g] = new int[I_nnzb[g]];
            kmat_ItoIEV_map[g] = new int[I_nnzb[g]];

            memset(kmat_Inofill_map[g], -1, I_nnzb[g] * sizeof(int));
            memset(kmat_ItoIEV_map[g], -1, I_nnzb[g] * sizeof(int));

            int nofill_ind = 0;
            int missing = 0;

            for (int i = 0; i < I_nnodes[g]; i++) {
                int i_IEV = IEVtoI_imap[g][i];

                for (int jp = I_rowp[g][i]; jp < I_rowp[g][i + 1]; jp++) {
                    int j = I_cols[g][jp];
                    int j_IEV = IEVtoI_imap[g][j];

                    bool found = false;

                    for (int kp = IEV_rowp[g][i_IEV]; kp < IEV_rowp[g][i_IEV + 1]; kp++) {
                        int k = IEV_cols[g][kp];

                        if (k == j_IEV) {
                            kmat_Inofill_map[g][nofill_ind] = jp;
                            kmat_ItoIEV_map[g][nofill_ind] = kp;
                            nofill_ind++;
                            found = true;
                            break;
                        }
                    }

                    if (!found) missing++;
                }
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-create_kmat_copy_maps] gpu %d I: "
            //         "rows=%d nnzb=%d copied=%d missing=%d\n",
            //         g, I_nnodes[g], I_nnzb[g], nofill_ind, missing);
            // }

            d_kmat_ItoIEV_map[g] =
                HostVec<int>(I_nnzb[g], kmat_ItoIEV_map[g]).createDeviceVec().getPtr();

            d_kmat_Inofill_map[g] =
                HostVec<int>(I_nnzb[g], kmat_Inofill_map[g]).createDeviceVec().getPtr();
        }

        // if (debug) printf("[BDDC-create_kmat_copy_maps] done\n");
    }

    void build_Svv_sparsity() {
        // if (debug) printf("[BDDC-build_Svv_sparsity] begin\n");

        Vc_node_imap = new int *[ngpus];
        Svv_rowp = new int *[ngpus];
        Svv_cols = new int *[ngpus];
        Svv_nnzb = new int[ngpus];
        Svv_rows = new int *[ngpus];

        d_Svv_vals = new DeviceVec<T>[ngpus];
        d_Svv_vals_ptr = new T *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            Vc_node_imap[g] = new int[local_nnodes[g]];
            memset(Vc_node_imap[g], -1, local_nnodes[g] * sizeof(int));

            for (int vnode = 0; vnode < Vc_nnodes[g]; vnode++) {
                int orig_lnode = Vc_nodes[g][vnode];

                if (orig_lnode < 0 || orig_lnode >= local_nnodes[g]) {
                    if (debug) {
                        printf(
                            "[BDDC-build_Svv_sparsity] ERROR gpu %d: "
                            "bad Vc orig_lnode=%d local_nnodes=%d\n",
                            g, orig_lnode, local_nnodes[g]);
                    }
                    continue;
                }

                Vc_node_imap[g][orig_lnode] = vnode;
            }

            std::vector<std::unordered_set<int>> Svv_adj(Vc_nnodes[g]);

            int *loc_elem_conn_IEV = kmat_IEV->getHostLocalElemConn(g);

            for (int isd_red = 0; isd_red < num_subdomains[g]; isd_red++) {
                int global_sd = red_subdomains[g][isd_red];

                std::unordered_set<int> sd_Vc_nodeset;

                for (int ielem = 0; ielem < local_nelems[g]; ielem++) {
                    int elem_sd = elem_sd_ind[g][ielem];

                    if (elem_sd != global_sd) continue;

                    int *iev_local_nodes = &loc_elem_conn_IEV[nodes_per_elem * ielem];

                    for (int l = 0; l < nodes_per_elem; l++) {
                        int iev_lnode = iev_local_nodes[l];

                        if (iev_lnode < 0 || iev_lnode >= part_IEV->local_nnodes[g]) {
                            // if (debug) {
                            //     printf(
                            //         "[BDDC-build_Svv_sparsity] ERROR gpu %d: "
                            //         "bad iev_lnode=%d part_IEV_nnodes=%d ielem=%d\n",
                            //         g, iev_lnode, part_IEV->local_nnodes[g], ielem);
                            // }
                            continue;
                        }

                        // Convert part_IEV local node -> single-GPU IEV node id
                        int sgpu_iev = part_IEV->h_local_nodes[g][iev_lnode];

                        if (sgpu_iev < 0 || sgpu_iev >= sgpu_IEV_nnodes) {
                            if (debug) {
                                printf(
                                    "[BDDC-build_Svv_sparsity] ERROR gpu %d: "
                                    "bad sgpu_iev=%d sgpu_IEV_nnodes=%d\n",
                                    g, sgpu_iev, sgpu_IEV_nnodes);
                            }
                            continue;
                        }

                        // Convert single-GPU IEV node id -> original global node id
                        int orig_gnode = sgpu_IEV_nodes[sgpu_iev];

                        // Convert original global node id -> original local node id on this GPU
                        int orig_lnode = part->global_to_local[g][orig_gnode];

                        if (orig_lnode < 0 || orig_lnode >= local_nnodes[g]) {
                            if (debug) {
                                printf(
                                    "[BDDC-build_Svv_sparsity] ERROR gpu %d: "
                                    "bad orig_lnode=%d orig_gnode=%d local_nnodes=%d\n",
                                    g, orig_lnode, orig_gnode, local_nnodes[g]);
                            }
                            continue;
                        }

                        int node_class = node_class_ind[g][orig_lnode];

                        if (node_class == IEV_VERTEX) {
                            int vnode = Vc_node_imap[g][orig_lnode];

                            if (vnode >= 0) {
                                sd_Vc_nodeset.insert(vnode);
                            }
                        }
                    }
                }

                std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());

                for (int i : sd_Vc_nodes) {
                    for (int j : sd_Vc_nodes) {
                        Svv_adj[i].insert(j);
                    }
                }
            }

            int *Svv_rowcts = new int[Vc_nnodes[g]];
            memset(Svv_rowcts, 0, Vc_nnodes[g] * sizeof(int));

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                Svv_rowcts[i] = static_cast<int>(Svv_adj[i].size());
            }

            Svv_rowp[g] = new int[Vc_nnodes[g] + 1];
            memset(Svv_rowp[g], 0, (Vc_nnodes[g] + 1) * sizeof(int));

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                Svv_rowp[g][i + 1] = Svv_rowp[g][i] + Svv_rowcts[i];
            }

            Svv_nnzb[g] = Svv_rowp[g][Vc_nnodes[g]];

            Svv_cols[g] = new int[Svv_nnzb[g]];

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                int jp = Svv_rowp[g][i];

                for (int j : Svv_adj[i]) {
                    Svv_cols[g][jp++] = j;
                }

                std::sort(&Svv_cols[g][Svv_rowp[g][i]], &Svv_cols[g][Svv_rowp[g][i + 1]]);
            }

            Svv_rows[g] = new int[Svv_nnzb[g]];

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                for (int jp = Svv_rowp[g][i]; jp < Svv_rowp[g][i + 1]; jp++) {
                    Svv_rows[g][jp] = i;
                }
            }

            d_Svv_vals[g] = DeviceVec<T>(Svv_nnzb[g] * block_dim2);
            d_Svv_vals_ptr[g] = d_Svv_vals[g].getPtr();

            // if (debug) {
            //     printf(
            //         "[BDDC-build_Svv_sparsity] gpu %d: "
            //         "Vc=%d nsub=%d Svv_nnzb=%d vals=%d\n",
            //         g, Vc_nnodes[g], num_subdomains[g], Svv_nnzb[g], Svv_nnzb[g] * block_dim2);
            // }

            delete[] Svv_rowcts;
        }

        // if (debug) printf("[BDDC-build_Svv_sparsity] done\n");
    }

    void build_Svv_maps() {
        // if (debug) printf("[BDDC-build_Svv_maps] begin\n");

        Svv_copy_nnzb = new int[ngpus];

        d_Svv_IEV_copyBlocks = new int *[ngpus];
        d_Svv_Vc_copyBlocks = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            Svv_copy_nnzb[g] = 0;

            std::vector<int> Svv_IEV_copyBlocks;
            std::vector<int> Svv_Vc_copyBlocks;

            for (int IEV_row = 0; IEV_row < IEV_nnodes[g]; IEV_row++) {
                int glob_row = IEV_nodes[g][IEV_row];
                int row_class = node_class_ind[g][glob_row];

                if (row_class != IEV_VERTEX) continue;

                int Vc_row = Vc_node_imap[g][glob_row];

                for (int jp = IEV_rowp[g][IEV_row]; jp < IEV_rowp[g][IEV_row + 1]; jp++) {
                    int IEV_col = IEV_cols[g][jp];
                    int glob_col = IEV_nodes[g][IEV_col];
                    int col_class = node_class_ind[g][glob_col];

                    if (col_class == IEV_VERTEX) {
                        int Vc_col = Vc_node_imap[g][glob_col];

                        for (int kp = Svv_rowp[g][Vc_row]; kp < Svv_rowp[g][Vc_row + 1]; kp++) {
                            int k = Svv_cols[g][kp];

                            if (k == Vc_col) {
                                Svv_IEV_copyBlocks.push_back(jp);
                                Svv_Vc_copyBlocks.push_back(kp);
                                Svv_copy_nnzb[g]++;
                                break;
                            }
                        }
                    }
                }
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-build_Svv_maps] gpu %d Svv copy: "
            //         "IEV=%d Vc=%d copied=%d\n",
            //         g, IEV_nnodes[g], Vc_nnodes[g], Svv_copy_nnzb[g]);
            // }

            d_Svv_IEV_copyBlocks[g] = HostVec<int>(Svv_copy_nnzb[g], Svv_IEV_copyBlocks.data())
                                          .createDeviceVec()
                                          .getPtr();

            d_Svv_Vc_copyBlocks[g] =
                HostVec<int>(Svv_copy_nnzb[g], Svv_Vc_copyBlocks.data()).createDeviceVec().getPtr();
        }

        IEVset_nnzb = new int *[ngpus];
        IEVtoSVV_nnzb = new int *[ngpus];

        d_IEVset_blocks = new int **[ngpus];
        d_IEVout_blocks = new int **[ngpus];
        d_IEVtoSVV_blocks = new int **[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IEVset_nnzb[g] = new int[MAX_NUM_VERTEX_PER_SUBDOMAIN];
            IEVtoSVV_nnzb[g] = new int[MAX_NUM_VERTEX_PER_SUBDOMAIN];

            d_IEVset_blocks[g] = new int *[MAX_NUM_VERTEX_PER_SUBDOMAIN];
            d_IEVout_blocks[g] = new int *[MAX_NUM_VERTEX_PER_SUBDOMAIN];
            d_IEVtoSVV_blocks[g] = new int *[MAX_NUM_VERTEX_PER_SUBDOMAIN];

            for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
                IEVset_nnzb[g][k] = 0;
                IEVtoSVV_nnzb[g][k] = 0;

                d_IEVset_blocks[g][k] = nullptr;
                d_IEVout_blocks[g][k] = nullptr;
                d_IEVtoSVV_blocks[g][k] = nullptr;
            }

            std::vector<int> IEVset_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];
            std::vector<int> IEVout_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];
            std::vector<int> IEVtoSVV_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];

            int active_subdomains = 0;
            int max_nsv = 0;
            int total_set_blocks = 0;
            int total_svv_blocks = 0;

            for (int isd = 0; isd < num_subdomains[g]; isd++) {
                std::vector<int> sd_iev_vertex_blocks;
                std::vector<int> sd_vc_nodes;

                for (int jp = IEV_sd_ptr[g][isd]; jp < IEV_sd_ptr[g][isd + 1]; jp++) {
                    int gnode = IEV_nodes[g][jp];

                    if (node_class_ind[g][gnode] == IEV_VERTEX) {
                        sd_iev_vertex_blocks.push_back(jp);

                        int vc_node = Vc_node_imap[g][gnode];

                        if (vc_node < 0) {
                            printf(
                                "[BDDC-build_Svv_maps] ERROR gpu %d: "
                                "vertex node %d on subdomain %d not found in Vc map\n",
                                g, gnode, isd);
                            exit(-1);
                        }

                        sd_vc_nodes.push_back(vc_node);
                    }
                }

                const int nsv = static_cast<int>(sd_iev_vertex_blocks.size());

                if (nsv == 0) continue;

                active_subdomains++;
                max_nsv = std::max(max_nsv, nsv);
                total_set_blocks += nsv;
                total_svv_blocks += nsv * nsv;

                if (nsv > MAX_NUM_VERTEX_PER_SUBDOMAIN) {
                    printf(
                        "[BDDC-build_Svv_maps] ERROR gpu %d: "
                        "subdomain %d has %d vertex slots > MAX=%d\n",
                        g, isd, nsv, MAX_NUM_VERTEX_PER_SUBDOMAIN);
                    exit(-1);
                }

                for (int k = 0; k < nsv; k++) {
                    const int iev_block = sd_iev_vertex_blocks[k];
                    const int vc_row = sd_vc_nodes[k];

                    IEVset_blocks_host[k].push_back(iev_block);

                    for (int kk = 0; kk < nsv; kk++) {
                        const int iev_block2 = sd_iev_vertex_blocks[kk];
                        const int vc_col = sd_vc_nodes[kk];

                        int svv_block = -1;

                        for (int jp = Svv_rowp[g][vc_row]; jp < Svv_rowp[g][vc_row + 1]; jp++) {
                            int m = Svv_cols[g][jp];

                            if (m == vc_col) {
                                svv_block = jp;
                                break;
                            }
                        }

                        if (svv_block < 0) {
                            printf(
                                "[BDDC-build_Svv_maps] ERROR gpu %d: "
                                "missing Svv block isd=%d vc_row=%d vc_col=%d nsv=%d\n",
                                g, isd, vc_row, vc_col, nsv);
                            exit(-1);
                        }

                        IEVout_blocks_host[k].push_back(iev_block2);
                        IEVtoSVV_blocks_host[k].push_back(svv_block);
                    }
                }
            }

            // if (debug) {
            //     printf(
            //         "[BDDC-build_Svv_maps] gpu %d subdomain maps: "
            //         "nsub=%d active=%d max_nsv=%d set=%d svv=%d\n",
            //         g, num_subdomains[g], active_subdomains, max_nsv, total_set_blocks,
            //         total_svv_blocks);
            // }

            for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
                IEVset_nnzb[g][k] = static_cast<int>(IEVset_blocks_host[k].size());
                IEVtoSVV_nnzb[g][k] = static_cast<int>(IEVtoSVV_blocks_host[k].size());

                if ((int)IEVout_blocks_host[k].size() != IEVtoSVV_nnzb[g][k]) {
                    printf(
                        "[BDDC-build_Svv_maps] ERROR gpu %d: "
                        "slot %d mismatch IEVout=%d IEVtoSVV=%d\n",
                        g, k, (int)IEVout_blocks_host[k].size(), IEVtoSVV_nnzb[g][k]);
                    exit(-1);
                }

                if (IEVset_nnzb[g][k] > 0) {
                    d_IEVset_blocks[g][k] =
                        HostVec<int>(IEVset_nnzb[g][k], IEVset_blocks_host[k].data())
                            .createDeviceVec()
                            .getPtr();
                }

                if (IEVtoSVV_nnzb[g][k] > 0) {
                    d_IEVout_blocks[g][k] =
                        HostVec<int>(IEVtoSVV_nnzb[g][k], IEVout_blocks_host[k].data())
                            .createDeviceVec()
                            .getPtr();

                    d_IEVtoSVV_blocks[g][k] =
                        HostVec<int>(IEVtoSVV_nnzb[g][k], IEVtoSVV_blocks_host[k].data())
                            .createDeviceVec()
                            .getPtr();
                }
            }

            if (debug) {
                int nonempty_slots = 0;
                int total_IEVset = 0;
                int total_IEVtoSVV = 0;

                for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
                    if (IEVset_nnzb[g][k] > 0 || IEVtoSVV_nnzb[g][k] > 0) nonempty_slots++;

                    total_IEVset += IEVset_nnzb[g][k];
                    total_IEVtoSVV += IEVtoSVV_nnzb[g][k];
                }

                // printf(
                //     "[BDDC-build_Svv_maps] gpu %d slots: "
                //     "nonempty=%d IEVset=%d IEVtoSVV=%d\n",
                //     g, nonempty_slots, total_IEVset, total_IEVtoSVV);
            }
        }

        // if (debug) printf("[BDDC-build_Svv_maps] done\n");
    }

    void build_iev_bcs() {
        // if (debug) printf("[BDDC-build_iev_bcs] begin\n");

        assembler->getLocalDeviceBCs(n_owned_bcs, n_local_bcs, d_owned_bcs, d_local_bcs);
        // if (debug) printf("[BDDC-build_iev_bcs] checkpt1\n");

        n_IEV_owned_bcs = new int[ngpus];
        n_IEV_local_bcs = new int[ngpus];

        d_IEV_owned_bcs = new int *[ngpus];
        d_IEV_local_bcs = new int *[ngpus];
        // if (debug) printf("[BDDC-build_iev_bcs] checkpt2\n");

        for (int g = 0; g < ngpus; g++) {
            // if (debug) {
            //     printf(
            //         "[BDDC-build_iev_bcs] gpu %d input: "
            //         "IEV_nnodes=%d owned_bcs=%d local_bcs=%d\n",
            //         g, IEV_nnodes[g], n_owned_bcs[g], n_local_bcs[g]);
            // }

            // -----------------------------------
            // local BCs
            // -----------------------------------
            int *h_local_bcs =
                DeviceVec<int>(n_local_bcs[g], d_local_bcs[g]).createHostVec().getPtr();

            std::vector<int> IEV_local_bcs_vec;

            for (int IEV_node = 0; IEV_node < IEV_nnodes[g]; IEV_node++) {
                int gnode = IEV_nodes[g][IEV_node];

                for (int ibc = 0; ibc < n_local_bcs[g]; ibc++) {
                    int bc = h_local_bcs[ibc];

                    int bc_node = bc / block_dim;
                    int bc_dof = bc % block_dim;

                    if (bc_node == gnode) {
                        int IEV_dof = block_dim * IEV_node + bc_dof;
                        IEV_local_bcs_vec.push_back(IEV_dof);
                    }
                }
            }

            n_IEV_local_bcs[g] = static_cast<int>(IEV_local_bcs_vec.size());

            d_IEV_local_bcs[g] = HostVec<int>(IEV_local_bcs_vec.size(), IEV_local_bcs_vec.data())
                                     .createDeviceVec()
                                     .getPtr();

            // -----------------------------------
            // owned BCs
            // -----------------------------------
            int *h_owned_bcs =
                DeviceVec<int>(n_owned_bcs[g], d_owned_bcs[g]).createHostVec().getPtr();

            std::vector<int> IEV_owned_bcs_vec;

            for (int IEV_node = 0; IEV_node < IEV_nnodes[g]; IEV_node++) {
                int gnode = IEV_nodes[g][IEV_node];

                for (int ibc = 0; ibc < n_owned_bcs[g]; ibc++) {
                    int bc = h_owned_bcs[ibc];

                    int bc_node = bc / block_dim;
                    int bc_dof = bc % block_dim;

                    if (bc_node == gnode) {
                        int IEV_dof = block_dim * IEV_node + bc_dof;
                        IEV_owned_bcs_vec.push_back(IEV_dof);
                    }
                }
            }

            n_IEV_owned_bcs[g] = static_cast<int>(IEV_owned_bcs_vec.size());

            d_IEV_owned_bcs[g] = HostVec<int>(IEV_owned_bcs_vec.size(), IEV_owned_bcs_vec.data())
                                     .createDeviceVec()
                                     .getPtr();

            // if (debug) {
            //     printf(
            //         "[BDDC-build_iev_bcs] gpu %d output: "
            //         "IEV_local_bcs=%d IEV_owned_bcs=%d\n",
            //         g, n_IEV_local_bcs[g], n_IEV_owned_bcs[g]);
            // }

            delete[] h_local_bcs;
            delete[] h_owned_bcs;
        }

        // if (debug) printf("[BDDC-build_iev_bcs] done\n");
    }

    void compute_jump_operators() {
        // rescaling like from FETI-DP but used in BDDC also
        // skips some things from FETI-DP
        edge_nsd = new int *[ngpus];
        vertex_nsd = new int *[ngpus];
        IE_nsd = new int *[ngpus];
        d_edge_nsd = new int *[ngpus];
        d_vertex_nsd = new int *[ngpus];
        d_IE_nsd = new int *[ngpus];
        d_IE_to_lam_map = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            // get the scales for edge DOF (# subdomains)
            edge_nsd[g] = new int[lam_nnodes[g]];
            for (int ilam = 0; ilam < lam_nnodes[g]; ilam++) {
                int loc_node = gam_nodes[g][ilam];
                edge_nsd[g][ilam] = node_nsd[g][loc_node];
            }

            // and similarly for vertex
            vertex_nsd[g] = new int[Vc_nnodes[g]];
            for (int iv = 0; iv < Vc_nnodes[g]; iv++) {
                int loc_node = Vc_nodes[g][iv];
                vertex_nsd[g][iv] = node_nsd[g][loc_node];
            }

            IE_nsd[g] = new int[IE_nnodes[g]];
            for (int i = 0; i < IE_nnodes[g]; i++) {
                int loc_node = IE_nodes[g][i];
                IE_nsd[g][i] = node_nsd[g][loc_node];
            }

            d_edge_nsd[g] = HostVec<int>(lam_nnodes[g], edge_nsd[g]).createDeviceVec().getPtr();
            d_vertex_nsd[g] = HostVec<int>(Vc_nnodes[g], vertex_nsd[g]).createDeviceVec().getPtr();
            d_IE_nsd[g] = HostVec<int>(IE_nnodes[g], IE_nsd[g]).createDeviceVec().getPtr();

            int *IE_to_lam_map = new int[IE_nnodes[g]];
            int *lam_nodes = new int[lam_nnodes[g]];
            // NOTE : turn the -1 on to ensure the map is correct (nz parts basically)
            memset(IE_to_lam_map, -1, IE_nnodes[g] * sizeof(int));
            // memset(IE_to_lam_map, 0, IE_nnodes * sizeof(int));
            int *temp_glob_node_tracker = new int[local_nnodes[g]];
            memset(temp_glob_node_tracker, -1, local_nnodes[g] * sizeof(int));
            int *temp_lam_ind = new int[local_nnodes[g]];
            memset(temp_lam_ind, -1, local_nnodes[g] * sizeof(int));
            int lam_ind = 0;
            for (int i = 0; i < IE_nnodes[g]; i++) {
                int glob_node = IE_nodes[g][i];
                int node_class = node_class_ind[g][glob_node];

                if (node_class == IEV_EDGE) {
                    if (temp_glob_node_tracker[glob_node] != -1) {
                        // then this edge node has been reached by previous subdomain
                        IE_to_lam_map[i] = temp_lam_ind[glob_node];
                    } else {
                        // has not been reached by previous subomdain, fill it
                        temp_lam_ind[glob_node] = lam_ind++;
                        temp_glob_node_tracker[glob_node] = 0;
                        IE_to_lam_map[i] = temp_lam_ind[glob_node];
                    }
                    lam_nodes[temp_lam_ind[glob_node]] = glob_node;
                }
            }
            // printf("IE_to_lam_map: ");
            // printVec<int>(IE_nnodes[g], IE_to_lam_map);

            d_IE_to_lam_map[g] =
                HostVec<int>(IE_nnodes[g], IE_to_lam_map).createDeviceVec().getPtr();
        }
    }

    void compute_reduced_partitions() {
        // if (debug) printf("[BDDC-compute_reduced_partitions] begin\n");

        // if (debug) {
        //     for (int g = 0; g < ngpus; g++) {
        //         printf(
        //             "[BDDC-compute_reduced_partitions] gpu %d: "
        //             "IE=%d I=%d Vc=%d gam=%d IEV=%d\n",
        //             g, IE_nnodes[g], I_nnodes[g], Vc_nnodes[g], ngam[g], IEV_nnodes[g]);
        //     }
        // }
        // whether node subsets for partition are in same order as IEV nodes
        bool same_IEV_order = true;

        // printf("[BDDC-compute_reduced_partitions] build part_IE\n");
        part_IE = new SDPartition(ngpus, num_nodes, IE_nnodes, IE_nodes, IEV_nodes, part_IEV,
                                  same_IEV_order, debug);

        // printf("[BDDC-compute_reduced_partitions] build part_I\n");
        part_I = new SDPartition(ngpus, num_nodes, I_nnodes, I_nodes, IEV_nodes, part_IEV,
                                 same_IEV_order, debug);

        // printf("[BDDC-compute_reduced_partitions] build part_V\n");
        // printf("Vc_nodes (size=%d): ", Vc_nnodes[0]);
        // printVec<int>(Vc_nnodes[0], Vc_nodes[0]);
        // printf("IEV_nodes (size=%d): ", IEV_nnodes[0]);
        // printVec<int>(IEV_nnodes[0], IEV_nodes[0]);
        part_V = new SDPartition(ngpus, num_nodes, Vc_nnodes, Vc_nodes, IEV_nodes, part_IEV,
                                 same_IEV_order, debug);

        // printf("[BDDC-compute_reduced_partitions] build part_gam\n");
        same_IEV_order = false;  // since it's all E nodes then V nodes (not in same IEV order)
        part_gam = new SDPartition(ngpus, num_nodes, ngam, gam_nodes, IEV_nodes, part_IEV,
                                   same_IEV_order, debug);

        // printf("[BDDC-compute_reduced_partitions] done\n");
    }

    void allocate_vectors() {
        // d_IEV_xpts = new Vec(ctx, part_IEV, 3);
        // d_IEV_vars = new Vec(ctx, part_IEV, block_dim);
        fext_IEV = new Vec(ctx, part_IEV, block_dim);
        fint_IEV = new Vec(ctx, part_IEV, block_dim);
        res_IEV = new Vec(ctx, part_IEV, block_dim);
        f_IEV = new Vec(ctx, part_IEV, block_dim);
        u_IEV = new Vec(ctx, part_IEV, block_dim);
        temp_IEV = new Vec(ctx, part_IEV, block_dim);

        f_IE = new SDVec(ctx, part_IE, block_dim);
        u_IE = new SDVec(ctx, part_IE, block_dim);
        u_IE3 = new SDVec(ctx, part_IE, block_dim);
        temp_IE = new SDVec(ctx, part_IE, block_dim);
        f_I = new SDVec(ctx, part_I, block_dim);
        u_I = new SDVec(ctx, part_I, block_dim);
        f_V = new SDVec(ctx, part_V, block_dim);
        u_V = new SDVec(ctx, part_V, block_dim);
        u_V3 = new SDVec(ctx, part_V, block_dim);
        temp_V = new SDVec(ctx, part_V, block_dim);
        temp_lam = new SDVec(ctx, part_gam, block_dim);
        temp_lam2 = new SDVec(ctx, part_gam, block_dim);
        d_coarse_vars = new SDVec(ctx, part_V, block_dim);
    }

    void clear_host_data() {}

    void assemble_subdomains() {
        // if (debug) printf("[BDDC-assemble_subdomains] addVec_globalToIEV methods\n");
        d_IEV_xpts->zeroAll();
        d_IEV_vars->zeroAll();
        addVec_globalToIEV(1.0, d_xpts, 0.0, d_IEV_xpts, 3);
        addVec_globalToIEV(1.0, d_vars, 0.0, d_IEV_vars, block_dim);

        kmat_IEV->zeroValues();
        // if (debug) printf("[BDDC-assemble_subdomains] add_IEV_jacobian\n");
        add_IEV_jacobian();
        // fext_IEV->zeroValues();
        // add_subdomain_fext();

        // TODO : something like this here.. but for each GPU matrix..
        // if (debug) printf("[BDDC-assemble_subdomains] apply_IEV_bcs\n");
        apply_IEV_bcs();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            cudaMemset(d_IE_vals[g].getPtr(), 0.0, block_dim2 * IE_nnzb[g] * sizeof(T));
            cudaMemset(d_I_vals[g].getPtr(), 0.0, block_dim2 * I_nnzb[g] * sizeof(T));
        }
        ctx->sync();
        // if (debug) printf("[BDDC-assemble_subdomains] copyKmat_IEVtoIE\n");
        copyKmat_IEVtoIE();
        // if (debug) printf("[BDDC-assemble_subdomains] copyKmat_IEVtoI\n");
        copyKmat_IEVtoI();
        // if (debug) printf("[BDDC-assemble_subdomains] done with method\n");
    }

    void add_IEV_jacobian() {
        const int cols_per_elem = 24;  // for 1st order element
        const int elems_per_block = 1;

        // no need to expand to local since already local?
        // d_IEV_xpts->expandToLocal();
        // d_IEV_vars->expandToLocal();

        dim3 block(num_quad_pts, cols_per_elem, elems_per_block);
        int elem_cols_per_block = cols_per_elem * elems_per_block;

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_IEV_xpts->getExpandedNodes(g);
            int loc_nelems = part_IEV->getLocalNumElements(g);

            // can't use Vec objects here since not same number of nodes?
            // or I could create it from IEV_elem_conn partitioners? we'll see
            T *loc_xpts_ptr = d_IEV_xpts->getLocalPtr(g);
            T *loc_vars_ptr = d_IEV_vars->getLocalPtr(g);
            int *loc_elem_comps = d_loc_elem_components[g];
            Data *loc_comp_data_ptr = d_loc_comp_data[g];

            // local element connectivity, used for both rows and columns
            int *loc_elem_conn_ptr = kmat_IEV->getLocalElemConn(g);
            int *loc_elem_ind_map = kmat_IEV->getLocalElemIndMap(g);
            T *loc_mat_vals = kmat_IEV->getLocalVals(g);

            int nblocks =
                (loc_nelems * cols_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;

            dim3 grid(nblocks);

            k_add_multigpu_jacobian_fast<T, elems_per_block, Assembler>
                <<<grid, block, 0, streams[g]>>>(
                    loc_num_nodes, loc_nelems, cols_per_elem, loc_elem_comps, loc_elem_conn_ptr,
                    loc_xpts_ptr, loc_vars_ptr, loc_comp_data_ptr, loc_elem_ind_map, loc_mat_vals);

            CHECK_CUDA(cudaGetLastError());
        }

        // if (debug) {
        //     int _nnz = kmat_IEV->getLocalNumNonzeros(0);
        //     int _nnzb = _nnz / block_dim2;
        //     printf("kmat_IEV vals on GPU[0] with nnzb=%d: \n", _nnzb);
        //     T *d_kmat_IEV_vals = kmat_IEV->getLocalVals(0);
        //     T *h_kmat_IEV_vals = DeviceVec<T>(_nnz, d_kmat_IEV_vals).createHostVec().getPtr();
        //     for (int i = 0; i < _nnzb; i++) {
        //         T *h_block = &h_kmat_IEV_vals[block_dim2 * i];
        //         printf("kmat_IEV: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }

    void assemble_coarse_problem() {
        // if (debug) printf("[BDDC-assemble_coarse_problem] zero Svv vals\n");
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            d_Svv_vals[g].zeroValues();
        }
        ctx->sync();

        // if (debug) printf("[BDDC-assemble_coarse_problem] copyKmat_IEVtoSvv\n");
        copyKmat_IEVtoSvv();

        // if (debug) {
        //     int _nnz = kmat_IEV->getLocalNumNonzeros(0);
        //     int _nnzb = _nnz / block_dim2;
        //     printf("kmat_IEV vals on GPU[0] with nnzb=%d: \n", _nnzb);
        //     T *d_kmat_IEV_vals = kmat_IEV->getLocalVals(0);
        //     T *h_kmat_IEV_vals = DeviceVec<T>(_nnz, d_kmat_IEV_vals).createHostVec().getPtr();
        //     for (int i = 0; i < _nnzb; i++) {
        //         T *h_block = &h_kmat_IEV_vals[block_dim2 * i];
        //         printf("kmat_IEV: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }

        // if (debug) printf("[BDDC-assemble_coarse_problem] computeSvvInverseTerm\n");
        computeSvvInverseTerm();
        // if (debug) printf("[BDDC-assemble_coarse_problem] done with computeSvvInverseTerm\n");

        // if (debug) {
        //     int _nnz = Svv_nnzb[0] * block_dim2;
        //     printf("S_VV vals on GPU[0] with nnzb=%d: \n", Svv_nnzb[0]);
        //     T *h_Svv_vals = d_Svv_vals[0].createHostVec().getPtr();
        //     for (int i = 0; i < Svv_nnzb[0]; i++) {
        //         int row = Svv_rows[0][i], col = Svv_cols[0][i];
        //         T *h_block = &h_Svv_vals[block_dim2 * i];
        //         printf("S_VV: block[%d] (%d,%d):\n", i, row, col);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
        // if (debug) printf("[BDDC-assemble_coarse_problem] done with method\n");
    }

    void apply_IEV_bcs() {
        kmat_IEV->apply_bcs(n_IEV_owned_bcs, d_IEV_owned_bcs, n_IEV_local_bcs, d_IEV_local_bcs);
    }

    void copyKmat_IEVtoIE() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int n_IE_vals = IE_nnzb[g] * block_dim2;
            dim3 block(32), grid((n_IE_vals + 31) / 32);
            k_copyMatToMat_restrict<T><<<grid, block, 0, streams[g]>>>(
                IE_nnzb[g], block_dim, d_kmat_IEtoIEV_map[g], d_kmat_IEnofill_map[g],
                d_IEV_vals[g].getPtr(), d_IE_vals[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
        }

        // if (debug) {
        //     int _nnz = IE_nnzb[0] * block_dim2;
        //     printf("kmat_IE vals on GPU[0] with nnzb=%d: \n", IE_nnzb[0]);
        //     T *h_kmat_IE_vals = d_IE_vals[0].createHostVec().getPtr();
        //     for (int i = 0; i < IE_nnzb[0]; i++) {
        //         T *h_block = &h_kmat_IE_vals[block_dim2 * i];
        //         printf("kmat_IE: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }

    void copyKmat_IEVtoI() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int n_I_vals = I_nnzb[g] * block_dim2;
            dim3 block(32), grid((n_I_vals + 31) / 32);
            k_copyMatToMat_restrict<T><<<grid, block, 0, streams[g]>>>(
                I_nnzb[g], block_dim, d_kmat_ItoIEV_map[g], d_kmat_Inofill_map[g],
                d_IEV_vals[g].getPtr(), d_I_vals[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
        }

        // if (debug) {
        //     int _nnz = I_nnzb[0] * block_dim2;
        //     printf("kmat_I vals on GPU[0] with nnzb=%d: \n", I_nnzb[0]);
        //     T *h_kmat_I_vals = d_I_vals[0].createHostVec().getPtr();
        //     for (int i = 0; i < I_nnzb[0]; i++) {
        //         T *h_block = &h_kmat_I_vals[block_dim2 * i];
        //         printf("kmat_I: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }

    void copyKmat_IEVtoSvv() {
        // TBD: may also need kmat_IEV on root GPU to ensure copy to S_VV on root GPU (can't be
        // partitioned maybe) or if there's some way I can assemble that part directly into Svv?

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int n_Svv_vals = Svv_copy_nnzb[g] * block_dim2;
            dim3 block(32), grid((n_Svv_vals + 31) / 32);
            k_copyMatToMat_restrict<T, true><<<grid, block, 0, streams[g]>>>(
                Svv_copy_nnzb[g], block_dim, d_Svv_IEV_copyBlocks[g], d_Svv_Vc_copyBlocks[g],
                d_IEV_vals[g].getPtr(), d_Svv_vals[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
        }
    }

    void printNodeVec(int nnodes, T *vec) {
        for (int i = 0; i < nnodes; i++) {
            T *node_vec = &vec[i * block_dim];
            printf("n=%d: ", i);
            printVec<T>(block_dim, node_vec);
        }
    }

    void printDeviceNodeVec(int nnodes, T *d_vec) {
        T *h_vec = DeviceVec<T>(nnodes * block_dim, d_vec).createHostVec().getPtr();
        printNodeVec(nnodes, h_vec);
    }

    void computeSvvInverseTerm() {
        // printf("MAX_NUM_VERTEX_PER_SUBDOMAIN = %d\n", MAX_NUM_VERTEX_PER_SUBDOMAIN);
        int ncols = MAX_NUM_VERTEX_PER_SUBDOMAIN * block_dim;
        for (int icol = 0; icol < ncols; icol++) {
            u_IEV->zeroAll();
            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] setVec_IEVtoV_vals\n",
            // icol);
            setVec_IEVtoV_vals(u_IEV, icol, 1.0);  // set these vals to 1.0 and all else 0
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] u_IEV vec\n", icol);
            //     T *h_uIEV = DeviceVec<T>(block_dim * IEV_nnodes[0], u_IEV->getPtr(0))
            //                     .createHostVec()
            //                     .getPtr();
            //     // printVec<T>(block_dim * IEV_nnodes[0], h_uIEV);
            //     printNodeVec(IEV_nnodes[0], h_uIEV);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] kmat_IEV->mult\n", icol);
            kmat_IEV->mult(u_IEV, f_IEV);
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] f_IEV vec\n", icol);
            //     T *h_fIEV = DeviceVec<T>(block_dim * IEV_nnodes[0], f_IEV->getPtr(0))
            //                     .createHostVec()
            //                     .getPtr();
            //     // printVec<T>(block_dim * IEV_nnodes[0], h_fIEV);
            //     printNodeVec(IEV_nnodes[0], h_fIEV);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] addVecIEVtoIE\n", icol);
            addVecIEVtoIE(1.0, f_IEV, 0.0, f_IE);
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] f_IE vec\n", icol);
            //     T *h_fIE = DeviceVec<T>(block_dim * IE_nnodes[0], f_IE->getPtr(0))
            //                    .createHostVec()
            //                    .getPtr();
            //     // printVec<T>(block_dim * IE_nnodes[0], h_fIE);
            //     printNodeVec(IE_nnodes[0], h_fIE);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] solveSubdomainIE\n", icol);
            solveSubdomainIE(f_IE, u_IE);
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] u_IE vec\n", icol);
            //     T *h_uIE = DeviceVec<T>(block_dim * IE_nnodes[0], u_IE->getPtr(0))
            //                    .createHostVec()
            //                    .getPtr();
            //     // printVec<T>(block_dim * IE_nnodes[0], h_uIE);
            //     printNodeVec(IE_nnodes[0], h_uIE);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] addVecIEtoIEV\n", icol);
            addVecIEtoIEV(1.0, u_IE, 0.0, u_IEV);
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] u_IEV2 vec\n", icol);
            //     T *h_uIEV = DeviceVec<T>(block_dim * IEV_nnodes[0], u_IEV->getPtr(0))
            //                     .createHostVec()
            //                     .getPtr();
            //     // printVec<T>(block_dim * IEV_nnodes[0], h_uIEV);
            //     printNodeVec(IEV_nnodes[0], h_uIEV);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] kmat_IEV->mult\n", icol);
            kmat_IEV->mult(-1.0, u_IEV, 0.0, f_IEV);
            // if (debug) {
            //     printf("[BDDC-computeSvvInverseTerm, icol=%d] f_IEV2 vec\n", icol);
            //     T *h_fIEV = DeviceVec<T>(block_dim * IEV_nnodes[0], f_IEV->getPtr(0))
            //                     .createHostVec()
            //                     .getPtr();
            //     // printVec<T>(block_dim * IEV_nnodes[0], h_fIEV);
            //     printNodeVec(IEV_nnodes[0], h_fIEV);
            // }

            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] addMat_IEVtoV_vals\n",
            // icol);
            addMat_IEVtoV_vals(icol, f_IEV);
            // if (debug) printf("[BDDC-computeSvvInverseTerm, icol=%d] done\n", icol);
        }
    }

    void create_cudss_solvers() {
        // build Svv as a GPUbsrmat
        // build_Svv_gpumat();

        subdomain_IE_solver = new CudssSubdomainBsrSolve<T>(ctx, IE_nnodes, block_dim, IE_rowp,
                                                            IE_cols, IE_nnzb, d_IE_vals_ptr);

        subdomain_I_solver = new CudssSubdomainBsrSolve<T>(ctx, I_nnodes, block_dim, I_rowp, I_cols,
                                                           I_nnzb, d_I_vals_ptr);

        // optional: alternate constructors to build an Svv_mat despite not having elem conn?
        // Svv_part = new LightPartitioner(ctx, sgpu_Vc_nnodes, Vc_nodes);
        // Svv_mat = GPUbsrmat<T, LightPartitioner>(ctx, Svv_part, block_dim, Svv_rowp, Svv_cols,
        //                                          Svv_nnzb, Svv_rows, d_Svv_vals);
        std::unordered_set<int> sgpu_Vc_nodes_set;

        for (int g = 0; g < ngpus; g++) {
            for (int ivc = 0; ivc < Vc_nnodes[g]; ivc++) {
                int loc_node = Vc_nodes[g][ivc];
                int glob_node = part->h_local_nodes[g][loc_node];
                sgpu_Vc_nodes_set.insert(glob_node);
            }
        }

        int *sgpu_Vc_nodes = new int[sgpu_Vc_nodes_set.size()];

        int ct = 0;
        for (int glob_node : sgpu_Vc_nodes_set) {
            sgpu_Vc_nodes[ct++] = glob_node;
        }

        if (ct != sgpu_Vc_nnodes) {
            printf("ERROR: sgpu_Vc_nodes count mismatch: got %d expected %d\n", ct, sgpu_Vc_nnodes);
            exit(-1);
        }

        int *glob_Vc_imap = new int[num_nodes];
        std::fill(glob_Vc_imap, glob_Vc_imap + num_nodes, -1);

        for (int vc = 0; vc < sgpu_Vc_nnodes; vc++) {
            int glob_node = sgpu_Vc_nodes[vc];

            if (glob_node < 0 || glob_node >= num_nodes) {
                printf("ERROR: bad glob_node=%d num_nodes=%d\n", glob_node, num_nodes);
                exit(-1);
            }

            glob_Vc_imap[glob_node] = vc;
        }

        int **red_Vc_nodes = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            red_Vc_nodes[g] = new int[Vc_nnodes[g]];

            for (int ivc = 0; ivc < Vc_nnodes[g]; ivc++) {
                int loc_node = Vc_nodes[g][ivc];
                int glob_node = part->h_local_nodes[g][loc_node];

                int red_node = glob_Vc_imap[glob_node];

                if (red_node < 0) {
                    printf("ERROR: gpu %d ivc %d loc_node=%d glob_node=%d not in glob_Vc_imap\n", g,
                           ivc, loc_node, glob_node);
                    exit(-1);
                }

                red_Vc_nodes[g][ivc] = red_node;
            }
        }
        // printf("red_Vc_nodes: ");
        // printVec<int>(Vc_nnodes[0], red_Vc_nodes[0]);

        Svv_solver =
            new CudssMgBSRSolverV2<T>(ctx, sgpu_Vc_nnodes, Vc_nnodes, red_Vc_nodes, block_dim,
                                      Svv_rowp, Svv_cols, Svv_nnzb, Svv_rows, d_Svv_vals_ptr);
    }

    // deprecated
    // void build_Svv_gpumat() {
    //     // TODO : how to best do this? alternate constructor?
    // }

    void solveSubdomainIE(SDVec *rhs_in, SDVec *sol_out) {
        rhs_in->expandToLocal();
        subdomain_IE_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
        sol_out->reduceFromLocal();
    }
    void solveSubdomainI(SDVec *rhs_in, SDVec *sol_out) {
        rhs_in->expandToLocal();
        subdomain_I_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
        sol_out->reduceFromLocal();
    }
    void solveCoarse(SDVec *rhs_in, SDVec *sol_out) {
        // TODO : prob shouldn't be using double pointers  here.. TBD
        rhs_in->expandToLocal();
        Svv_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
        sol_out->reduceFromLocal();
    }

    void setVec_IEVtoV_vals(Vec *vec_IEV, int irow, T val) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int block_row = irow / block_dim;
            int set_nnzb = IEVset_nnzb[g][block_row];
            int *d_blocks = d_IEVset_blocks[g][block_row];
            T *loc_vec_IEV = vec_IEV->getLocalPtr(g);
            if (set_nnzb == 0) continue;

            dim3 block(32);
            dim3 grid((set_nnzb + 31) / 32);
            k_setVec_IEVtoV_vals<T><<<grid, block, 0, streams[g]>>>(set_nnzb, block_dim, irow,
                                                                    d_blocks, loc_vec_IEV, val);

            CHECK_CUDA(cudaGetLastError());
        }
        vec_IEV->reduceFromLocal();
    }

    void addMat_IEVtoV_vals(const int icol, Vec *hvec) {
        // helps assembly of S_VV schur complement matrix on each local GPU partition
        hvec->expandToLocal();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int block_col = icol / block_dim;
            int set_nnzb = IEVtoSVV_nnzb[g][block_col];
            int *d_svv_blocks = d_IEVtoSVV_blocks[g][block_col];
            int *d_iev_blocks = d_IEVout_blocks[g][block_col];

            T *loc_hvec = hvec->getLocalPtr(g);
            if (set_nnzb == 0) continue;

            dim3 block(32);
            dim3 grid((set_nnzb * block_dim + 31) / 32);

            k_addMat_IEVtoV_vals<T>
                <<<grid, block, 0, streams[g]>>>(set_nnzb, block_dim, icol, d_iev_blocks,
                                                 d_svv_blocks, loc_hvec, d_Svv_vals[g].getPtr());
            CHECK_CUDA(cudaGetLastError());
        }
    }

    template <bool scaled = false>
    void addVec_globalToIEV(T a, Vec *x_global, T b, Vec *y_iev, int vars_per_node_in) {
        y_iev->scale(b);
        y_iev->expandToLocal();
        auto IE_vec = (vars_per_node_in == 3) ? u_IE3 : u_IE;
        auto V_vec = (vars_per_node_in == 3) ? u_V3 : u_V;
        IE_vec->zeroAll();
        V_vec->zeroAll();
        // if (debug) printf("[BDDC-addVec_globalToIEV] checkpt1\n");

        // add from global to IEV on each GPU partition
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_xglob = x_global->getLocalPtr(g);
            T *loc_uIE = IE_vec->getLocalPtr(g);

            // if (debug) {
            //     int *h_IE_nsd =
            //         DeviceVec<int>(this->IE_nnodes[g],
            //         this->d_IE_nsd[g]).createHostVec().getPtr();
            //     printf("h_IE_nsd (size=%d): ", this->IE_nnodes[g]);
            //     printVec<int>(this->IE_nnodes[g], h_IE_nsd);
            //     int *h_IE_nodes = DeviceVec<int>(this->IE_nnodes[g], this->d_IE_nodes[g])
            //                           .createHostVec()
            //                           .getPtr();
            //     printf("h_IE_nodes (size=%d): ", this->IE_nnodes[g]);
            //     printVec<int>(this->IE_nnodes[g], h_IE_nodes);
            // }

            int nvals = IE_nnodes[g] * vars_per_node_in;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVec_GlobalToIE<T, scaled><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], vars_per_node_in, d_IE_nodes[g], d_IE_nsd[g], loc_xglob, loc_uIE, a);
            CHECK_CUDA(cudaGetLastError());

            T *loc_uV = V_vec->getLocalPtr(g);
            // if (debug) {
            //     int *h_vertex_nsd = DeviceVec<int>(this->Vc_nnodes[g], this->d_vertex_nsd[g])
            //                             .createHostVec()
            //                             .getPtr();
            //     printf("h_vertex_nsd: ");
            //     printVec<int>(this->Vc_nnodes[g], h_vertex_nsd);
            // }

            int nvals2 = Vc_nnodes[g] * vars_per_node_in;
            dim3 grid2((nvals2 + 31) / 32);

            // scales by 0.25x like for load distribution
            k_addVec_GlobaltoVc<T, scaled>
                <<<grid2, block, 0, streams[g]>>>(Vc_nnodes[g], vars_per_node_in, d_Vc_nodes[g],
                                                  d_vertex_nsd[g], loc_xglob, loc_uV, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        // if (debug) printf("[BDDC-addVec_globalToIEV] checkpt3\n");

        // reduce across ghost nodes (so matches single GPU result)
        // TODO : should I delay reduction until only for y_iev though? TBD..
        V_vec->reduceFromLocal();
        IE_vec->reduceFromLocal();

        // if (debug) printf("[BDDC-addVec_globalToIEV] addVecIEtoIEV\n");
        addVecIEtoIEV(1.0, IE_vec, 0.0, y_iev, vars_per_node_in);
        // if (debug) printf("[BDDC-addVec_globalToIEV] addVecVctoIEV\n");
        addVecVctoIEV(1.0, V_vec, 1.0, y_iev, vars_per_node_in);
        // if (debug) printf("[BDDC-addVec_globalToIEV] done with method\n");
    }

    void addVecIEVtoIE(T a, Vec *x, T b, SDVec *y) {
        x->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int nvals = IE_nnodes[g] * block_dim;
            T *loc_x = x->getLocalPtr(g);
            T *loc_y = y->getLocalPtr(g);

            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecSmallerOut<T><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], block_dim, d_IEVtoIE_imap[g], loc_x, loc_y, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }

    void addVecIEVtoI(T a, Vec *x, T b, SDVec *y) {
        x->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int nvals = I_nnodes[g] * block_dim;
            T *loc_x = x->getLocalPtr(g);
            T *loc_y = y->getLocalPtr(g);

            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecSmallerOut<T><<<grid, block, 0, streams[g]>>>(
                I_nnodes[g], block_dim, d_IEVtoI_imap[g], loc_x, loc_y, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }

    void addVecIEtoI(T a, SDVec *x, T b, SDVec *y) {
        addVecIEtoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoI(temp_IEV, y, 1.0, b);
    }

    template <bool scaled = false>
    void addVecIEVtoVc(T a, Vec *x, T b, SDVec *y) {
        x->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getLocalPtr(g);
            T *loc_y = y->getLocalPtr(g);

            int nvals = V_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVec_IEVtoVc<T><<<grid, block, 0, streams[g]>>>(
                V_nnodes[g], block_dim, d_IEVtoV_imap[g], d_VctoV_imap[g], loc_x, loc_y, a);

            if constexpr (scaled) {
                int Vc_nvals = Vc_nnodes[g] * block_dim;
                dim3 block(32), grid((Vc_nvals + 31) / 32);
                k_subdomain_normalize_vec_inout<T><<<grid, block, 0, streams[g]>>>(
                    Vc_nnodes[g], block_dim, d_vertex_nsd[g], loc_y);
            }
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }

    void addVecGamtoIE(T a, SDVec *gam, T b, SDVec *vec_IE) {
        gam->expandToLocal();
        vec_IE->scale(b);
        vec_IE->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_gam = gam->getLocalPtr(g);
            T *loc_vecIE = vec_IE->getLocalPtr(g);

            int nvals = IE_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecGamtoIE<T><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], block_dim, d_IE_to_lam_map[g], loc_gam, loc_vecIE, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        vec_IE->reduceFromLocal();
    }

    void addVecIEtoIEV(T a, SDVec *x, T b, Vec *y, int vars_per_node = -1) {
        x->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getLocalPtr(g);
            T *loc_y = y->getLocalPtr(g);

            if (vars_per_node == -1) vars_per_node = block_dim;
            int nvals = IE_nnodes[g] * vars_per_node;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecSmallerIn<T><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], vars_per_node, d_IEVtoIE_imap[g], loc_x, loc_y, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }
    void addVecItoIEV(T a, SDVec *x, T b, Vec *y) {
        x->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getLocalPtr(g);
            T *loc_y = y->getLocalPtr(g);

            int nvals = I_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecSmallerIn<T><<<grid, block, 0, streams[g]>>>(I_nnodes[g], vars_per_node,
                                                                 d_IEVtoI_imap[g], loc_x, loc_y, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }
    void addVecItoIE(T a, SDVec *x, T b, SDVec *y) {
        addVecItoIEV(a, x, 0.0, temp_IEV);
        addVecIEVtoIE(1.0, temp_IEV, b, y);
    }
    template <bool scaled = false>
    void addVecVctoIEV(T a, SDVec *x, T b, Vec *y, int vars_per_node = -1) {
        x->copyTo(temp_V);
        temp_V->expandToLocal();
        y->scale(b);
        y->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getLocalPtr(g);
            T *loc_tempV = temp_V->getLocalPtr(g);
            cudaMemcpy(loc_tempV, loc_x, Vc_nnodes[g] * block_dim * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            if constexpr (scaled) {
                int Vc_nvals = Vc_nnodes[g] * block_dim;
                dim3 block(32), grid((Vc_nvals + 31) / 32);
                k_subdomain_normalize_vec_inout<T><<<grid, block, 0, streams[g]>>>(
                    Vc_nnodes[g], block_dim, d_vertex_nsd[g], loc_tempV);
            }

            T *loc_y = y->getLocalPtr(g);
            if (vars_per_node == -1) vars_per_node = block_dim;
            int nvals = V_nnodes[g] * vars_per_node;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVec_VctoIEV<T><<<grid, block, 0, streams[g]>>>(
                V_nnodes[g], vars_per_node, d_IEVtoV_imap[g], d_VctoV_imap[g], loc_tempV, loc_y, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        y->reduceFromLocal();
    }
    void addVecIEtoGam(T a, SDVec *vec_IE, T b, SDVec *gam) {
        vec_IE->expandToLocal();
        gam->scale(b);
        gam->expandToLocal();
        // if (debug) {
        //     printf("[BDDC-addVecIEtoGam]: vec_IE\n");
        //     printDeviceNodeVec(IE_nnodes[0], vec_IE->getPtr(0));
        //     printf("[BDDC-addVecIEtoGam]: pre_gam\n");
        //     printDeviceNodeVec(ngam[0], gam->getPtr(0));
        // }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_vecIE = vec_IE->getLocalPtr(g);
            T *loc_gam = gam->getLocalPtr(g);

            int nvals = IE_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVecIEtoGam<T><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], block_dim, d_IE_to_lam_map[g], loc_vecIE, loc_gam, a);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        gam->reduceFromLocal();

        // if (debug) {
        //     printf("[BDDC-addVecIEtoGam]: post_gam\n");
        //     printDeviceNodeVec(ngam[0], gam->getLocalPtr(0));
        //     printf("[BDDC-addVecIEtoGam]: post_gam_owned\n");
        //     printDeviceNodeVec(ngam[0], gam->getPtr(0));
        // }
    }
    void zeroInteriorIE(SDVec *x) {
        // TODO : need this?
        x->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_x = x->getLocalPtr(g);

            int nvals = IE_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_zeroInterior<T>
                <<<grid, block, 0, streams[g]>>>(IE_nnodes[g], block_dim, d_IE_interior[g], loc_x);
            CHECK_CUDA(cudaGetLastError());
        }
        // TODO : need this?
        x->reduceFromLocal();
        x->expandToLocal();
    }
    void addGlobalSoln(SDVec *u_IE, SDVec *u_V, Vec *soln) {
        u_IE->expandToLocal();
        u_V->expandToLocal();
        soln->expandToLocal();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_uIE = u_IE->getLocalPtr(g);
            T *loc_uV = u_V->getLocalPtr(g);
            T *loc_soln = soln->getLocalPtr(g);

            int nvals = IE_nnodes[g] * block_dim;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVec_IEtoGlobal<T><<<grid, block>>>(IE_nnodes[g], block_dim, d_IE_nodes[g],
                                                    d_IE_nsd[g], loc_uIE, loc_soln, 1.0);

            int nvals2 = Vc_nnodes[g] * block_dim;
            dim3 grid2((nvals2 + 31) / 32);
            k_addVec_VctoGlobal<T>
                <<<grid2, block>>>(Vc_nnodes[g], block_dim, d_Vc_nodes[g], loc_uV, loc_soln, 1.0);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();  // TODO : do I need to add this in other places?
        soln->reduceFromLocal();
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(T alpha, Vec *vec_IEV, T beta, SDVec *vec_gam) {
        vec_gam->scale(beta);
        vec_gam->expandToLocal();
        temp_lam->zeroAll();
        temp_V->zeroAll();
        // if (debug) printf("[BDDC-addVecIEVtoGam] start method\n");

        // add IEV to E part of Gam first (then V later)
        // if (debug) printf("[BDDC-addVecIEVtoGam] addVecIEVtoIE\n");
        addVecIEVtoIE(alpha, vec_IEV, 0.0, temp_IE);
        // if (debug) {
        //     printf("[BDDC-addVecIEVtoGam]: temp_IE\n");
        //     printDeviceNodeVec(IE_nnodes[0], temp_IE->getPtr(0));
        // }

        // if (debug) printf("[BDDC-addVecIEVtoGam] addVecIEtoGam\n");
        addVecIEtoGam(1.0, temp_IE, 0.0, temp_lam);
        temp_lam->expandToLocal();
        // if (debug) {
        //     printf("[BDDC-addVecIEVtoGam]: temp_lam\n");
        //     printDeviceNodeVec(lam_nnodes[0], temp_lam->getPtr(0));
        // }

        // if (debug) printf("[BDDC-addVecIEVtoGam] set_edge_values\n");
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int edge_size = lam_nnodes[g] * block_dim;
            T *loc_temp_lam = temp_lam->getPtr(g);  // DeviceVec<T> array
            if constexpr (SCALED) {
                dim3 block(32), grid((edge_size + 31) / 32);
                k_subdomain_normalize_vec_inout<T><<<grid, block, 0, streams[g]>>>(
                    lam_nnodes[g], block_dim, d_edge_nsd[g], loc_temp_lam);
            }

            T a = 1.0;
            T *loc_vec_gam = vec_gam->getLocalPtr(g);
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandles[g], edge_size, &a, loc_temp_lam, 1, loc_vec_gam, 1));

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        // if (debug) printf("[BDDC-addVecIEVtoGam] addVecIEVtoVc\n");
        addVecIEVtoVc(alpha, vec_IEV, 0.0, temp_V);
        temp_V->expandToLocal();

        // if (debug) printf("[BDDC-addVecIEVtoGam] set vertex values\n");
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_temp_V = temp_V->getLocalPtr(g);  // DeviceVec<T> array
            T *loc_vec_gam = vec_gam->getLocalPtr(g);
            T a = 1.0;
            int edge_size = lam_nnodes[g] * block_dim;
            int V_size = Vc_nnodes[g] * block_dim;
            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], V_size, &a, loc_temp_V, 1,
                                     &loc_vec_gam[edge_size], 1));

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        // if (debug) printf("[BDDC-addVecIEVtoGam] vec_gam->reduceFromLocal\n");
        vec_gam->reduceFromLocal();

        // if (debug) printf("[BDDC-addVecIEVtoGam] done with method\n");
        // then scale vec_gam
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(T alpha, SDVec *vec_gam, T beta, Vec *vec_IEV) {
        vec_gam->expandToLocal();
        vec_IEV->scale(beta);
        vec_IEV->expandToLocal();
        temp_lam->zeroAll();
        temp_V->zeroAll();

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: vec_gam\n");
        //     printDeviceNodeVec(ngam[0], vec_gam->getPtr(0));
        // }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int edge_size = lam_nnodes[g] * block_dim;
            T *loc_temp_lam = temp_lam->getLocalPtr(g);

            T a = alpha;
            T *loc_vec_gam = vec_gam->getLocalPtr(g);
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandles[g], edge_size, &a, loc_vec_gam, 1, loc_temp_lam, 1));

            if constexpr (SCALED) {
                dim3 block(32), grid((edge_size + 31) / 32);
                k_subdomain_normalize_vec_inout<T><<<grid, block, 0, streams[g]>>>(
                    lam_nnodes[g], block_dim, d_edge_nsd[g], loc_temp_lam);
            }

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        temp_lam->reduceFromLocal();

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: temp_lam\n");
        //     this->printDeviceNodeVec(lam_nnodes[0], temp_lam->getPtr(0));
        // }

        // TODO : might be a multi-GPU mistake in reduce and expand vec_IEV multiple times..
        addVecGamtoIE(1.0, temp_lam, 0.0, temp_IE);
        temp_IE->expandToLocal();
        addVecIEtoIEV(1.0, temp_IE, 0.0, vec_IEV);
        vec_IEV->expandToLocal();

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: vec_IEV\n");
        //     this->printDeviceNodeVec(IEV_nnodes[0], vec_IEV->getPtr(0));
        // }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_temp_V = temp_V->getLocalPtr(g);
            T *loc_vec_gam = vec_gam->getLocalPtr(g);
            T a = 1.0;
            int edge_size = lam_nnodes[g] * block_dim;
            int V_size = Vc_nnodes[g] * block_dim;
            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], V_size, &a, &loc_vec_gam[edge_size], 1,
                                     loc_temp_V, 1));

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        temp_V->reduceFromLocal();

        addVecVctoIEV(1.0, temp_V, 1.0, vec_IEV);
        vec_IEV->expandToLocal();
    }

   private:
    MultiGPUContext *ctx = nullptr;

    int num_elements = 0;
    int num_nodes = 0;
    int N = 0;
    int block_dim = 0;
    int block_dim2 = 0;
    int ngpus = 0;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 0;

    Assembler *assembler = nullptr;
    Mat *mat = nullptr;
    Mat *kmat_IEV = nullptr;
    Partition *part = nullptr;
    Partition *part_IEV = nullptr;
    SDPartition *part_IE = nullptr;
    SDPartition *part_I = nullptr;
    SDPartition *part_V = nullptr;
    SDPartition *part_gam = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;

    Vec *d_xpts = nullptr;
    Vec *d_vars = nullptr;
    Vec *d_IEV_xpts = nullptr;
    Vec *d_IEV_vars = nullptr;

    int **d_loc_elem_components = nullptr;
    Data **d_loc_comp_data = nullptr;

    int *n_owned_bcs = nullptr;
    int *n_local_bcs = nullptr;
    int **d_owned_bcs = nullptr;
    int **d_local_bcs = nullptr;

    int sgpu_num_subdomains = 0;
    int sgpu_I_nnodes = 0;
    int sgpu_IE_nnodes = 0;
    int sgpu_IEV_nnodes = 0;
    int sgpu_Vc_nnodes = 0;
    int sgpu_V_nnodes = 0;
    int sgpu_lam_nnodes = 0;

    int *sgpu_elem_sd_ind = nullptr;
    int *sgpu_node_class_ind = nullptr;
    int *sgpu_node_nsd = nullptr;
    int *sgpu_IEV_sd_ptr = nullptr;
    int *sgpu_IEV_sd_ind = nullptr;
    int *sgpu_IEV_nodes = nullptr;
    int *sgpu_IEV_elem_conn = nullptr;

    int *subdomain_gpu_ind = nullptr;
    int *num_subdomains = nullptr;
    int *local_nnodes = nullptr;
    int *local_nelems = nullptr;
    int *sd_cts = nullptr;
    int *elem_glob_to_loc = nullptr;

    int **glob_loc_elem_map = nullptr;
    int **red_subdomains = nullptr;
    int **elem_sd_ind = nullptr;
    int **node_class_ind = nullptr;
    int **node_nsd = nullptr;

    int *I_nnodes = nullptr;
    int *IE_nnodes = nullptr;
    int *IEV_nnodes = nullptr;
    int *Vc_nnodes = nullptr;
    int *V_nnodes = nullptr;
    int *lam_nnodes = nullptr;
    int *n_edge = nullptr;
    int *ngam = nullptr;

    int **IEV_nodes = nullptr;
    int **IEV_elem_conn = nullptr;
    int **IEV_loc_to_glob = nullptr;
    int **elem_loc_to_glob = nullptr;
    int **d_IEV_elem_conn = nullptr;
    int **IEV_sd_ptr = nullptr;

    int **IE_nodes = nullptr;
    int **I_nodes = nullptr;
    int **Vc_nodes = nullptr;
    int **Vc_inodes = nullptr;
    int **gam_nodes = nullptr;

    int **IEVtoIE_map = nullptr;
    int **IEVtoIE_imap = nullptr;
    int **IEVtoI_map = nullptr;
    int **IEVtoI_imap = nullptr;
    int **IEVtoV_imap = nullptr;
    int **VctoV_imap = nullptr;

    bool **IE_interior = nullptr;
    bool **IE_general_edge = nullptr;
    bool **d_IE_interior = nullptr;
    bool **d_IE_general_edge = nullptr;

    int **d_IE_nodes = nullptr;
    int **d_Vc_nodes = nullptr;
    int **d_IEVtoIE_imap = nullptr;
    int **d_IEVtoI_imap = nullptr;
    int **d_IEVtoV_imap = nullptr;
    int **d_VctoV_imap = nullptr;

    int **edge_nsd = nullptr;
    int **vertex_nsd = nullptr;
    int **IE_nsd = nullptr;
    int **d_IE_nsd = nullptr;
    int **d_edge_nsd = nullptr;
    int **d_vertex_nsd = nullptr;
    int **d_IE_to_lam_map = nullptr;

    DeviceVec<T> *d_IEV_vals = nullptr;
    DeviceVec<T> *d_IE_vals = nullptr;
    DeviceVec<T> *d_I_vals = nullptr;
    DeviceVec<T> *d_Svv_vals = nullptr;
    T **d_IE_vals_ptr = nullptr;
    T **d_I_vals_ptr = nullptr;
    T **d_Svv_vals_ptr = nullptr;

    int **IEV_rowp = nullptr;
    int **IEV_cols = nullptr;
    int *IEV_nnzb = nullptr;
    int **IEV_rows = nullptr;

    int **IE_rowp = nullptr;
    int **IE_cols = nullptr;
    int *IE_nnzb = nullptr;
    int **IE_rows = nullptr;
    bool debug = false;

    int **I_rowp = nullptr;
    int **I_cols = nullptr;
    int *I_nnzb = nullptr;
    int **I_rows = nullptr;

    int **kmat_IEnofill_map = nullptr;
    int **kmat_IEtoIEV_map = nullptr;
    int **d_kmat_IEnofill_map = nullptr;
    int **d_kmat_IEtoIEV_map = nullptr;

    int **kmat_Inofill_map = nullptr;
    int **kmat_ItoIEV_map = nullptr;
    int **d_kmat_Inofill_map = nullptr;
    int **d_kmat_ItoIEV_map = nullptr;

    int **Vc_node_imap = nullptr;
    int **Svv_rowp = nullptr;
    int **Svv_cols = nullptr;
    int *Svv_nnzb = nullptr;
    int **Svv_rows = nullptr;

    int *Svv_copy_nnzb = nullptr;
    int **d_Svv_IEV_copyBlocks = nullptr;
    int **d_Svv_Vc_copyBlocks = nullptr;

    int **IEVset_nnzb = nullptr;
    int **IEVtoSVV_nnzb = nullptr;
    int ***d_IEVset_blocks = nullptr;
    int ***d_IEVout_blocks = nullptr;
    int ***d_IEVtoSVV_blocks = nullptr;

    int *n_IEV_owned_bcs = nullptr;
    int *n_IEV_local_bcs = nullptr;
    int **d_IEV_owned_bcs = nullptr;
    int **d_IEV_local_bcs = nullptr;

    Vec *fext_IEV = nullptr;
    Vec *fint_IEV = nullptr;
    Vec *res_IEV = nullptr;
    Vec *f_IEV = nullptr;
    Vec *u_IEV = nullptr;
    Vec *temp_IEV = nullptr;

    IEVSplit *split = nullptr;
    SDVec *temp_IE = nullptr;
    SDVec *temp_V = nullptr;
    SDVec *f_IE = nullptr;
    SDVec *u_IE = nullptr;
    SDVec *u_IE3 = nullptr;
    SDVec *f_I = nullptr;
    SDVec *u_I = nullptr;
    SDVec *f_V = nullptr;
    SDVec *u_V = nullptr;
    SDVec *u_V3 = nullptr;
    SDVec *temp_lam = nullptr;
    SDVec *temp_lam2 = nullptr;
    SDVec *d_coarse_vars = nullptr;

    CudssSubdomainBsrSolve<T> *subdomain_IE_solver = nullptr;
    CudssSubdomainBsrSolve<T> *subdomain_I_solver = nullptr;
    CudssMgBSRSolverV2<T> *Svv_solver = nullptr;
};