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
                          IEVSplit *split_)
        : ctx(ctx_),
          part(part_),
          mat(mat_),
          cublasHandles(ctx_->cublasHandles),
          cusparseHandles(ctx_->cusparseHandles),
          streams(ctx_->streams),
          split(split_) {
        ngpus = ctx->ngpus;
        num_elements = assembler->get_num_elements();
        num_nodes = assembler->get_num_nodes();
        N = num_nodes * vars_per_node;

        block_dim = mat->getBlockDim();
        block_dim2 = block_dim * block_dim;

        d_xpts = assembler->getDeviceXpts();
        d_vars = assembler->getDeviceVars();
        d_loc_elem_components = assembler->getDeviceElemComponents();
        d_loc_comp_data = assembler->getDeviceCompData();
        assembler->getLocalDeviceBCs(n_owned_bcs, n_local_bcs, d_owned_bcs, d_local_bcs);
        MAX_NUM_VERTEX_PER_SUBDOMAIN = split_->MAX_NUM_VERTEX_PER_SUBDOMAIN;

        // setup on construction (sparsity patterns, maps, etc.)
        import_splitting();
        build_IE_I_V_maps();
        build_IEV_sparsity();
        build_IE_and_I_sparsity();
        create_kmat_copy_maps();
        build_Svv_sparsity();
        build_Svv_maps();
        build_iev_bcs();
        compute_reduced_partitions();
        allocate_vectors();
        create_cudss_solvers();
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
    T *get_IEV_xpts(int gpu) { return this->d_IEV_xpts[gpu]; }
    T *get_IEV_vars(int gpu) { return this->d_IEV_vars[gpu]; }
    SDPartition *get_part_gam() { return part_gam; }

    void update_after_assembly(Vec *vars) {
        vars->copyTo(d_vars);
        assemble_subdomains();
        subdomain_I_solver->factor();
        subdomain_IE_solver->factor();
        assemble_coarse_problem();
        Svv_solver->factor();
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, Vec *vars) {
        // compute res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV
        addVec_globalToIEV(1.0, d_xpts, 0.0, d_IEV_xpts, 3);
        addVec_globalToIEV(1.0, d_vars, 0.0, d_IEV_vars, block_dim);
        fint_IEV->zeroAll();
        d_IEV_xpts->expandToLocal();
        d_IEV_vars->expandToLocal();

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
        }

        this->fint_IEV->apply_bcs(n_IEV_owned_bcs, d_IEV_owned_bcs, n_IEV_local_bcs,
                                  d_IEV_local_bcs);
        this->fint_IEV->scale(-lambdaI);
        this->fint_IEV->copyTo(this->res_IEV);
        res_IEV->axpy(lambdaE, fext_IEV);
    }

    void get_lam_rhs(SDVec *gam_rhs) {
        // gets rhs of interface BDDC system
        gam_rhs->zeroAll();

        // harmonic extension from interface (Gam) to interior (I) nodes
        res_IEV->copyTo(f_IEV);
        addVecIEVtoI(1.0, f_IEV, 0.0, f_I);

        // solve interior subdomain-parallel matrix
        solveSubdomainI(f_I, u_I);

        // harmonic extension from interior (I) to interface (Gam)
        addVecItoIEV(1.0, u_I, 0.0, u_IEV);
        kmat_IEV->mult(-1.0, u_IEV, 1.0, f_IEV);
        addVecIEVtoGam(1.0, f_IEV, 0.0, gam_rhs);
    }

    void mat_vec(SDVec *gam_in, SDVec *gam_out) {
        // gets K_{Gam,Gam}*x_{Gam} internal residual of interface BDDC system
        gam_out->zeroAll();

        // harmonic extension from interface (Gam) to interior (I) nodes
        addVecGamtoIEV(1.0, gam_in, 0.0, u_IEV);
        kmat_IEV->mult(1.0, u_IEV, 0.0, f_IEV);
        addVecIEVtoI(1.0, f_IEV, 0.0, f_I);

        // solve interior (I) subdomain-parallel matrix
        solveSubdomainI(f_I, u_I);

        // harmonic extension from interior (I) to interface (Gam)
        addVecItoIEV(1.0, u_I, 0.0, u_IEV);
        kmat_IEV->mult(-1.0, u_IEV, 1.0, f_IEV);
        addVecIEVtoGam(1.0, f_IEV, 0.0, gam_out);
    }

    bool solve(SDVec *gam_rhs, SDVec *gam, bool check_conv = false) {
        // gets preconditioner solve for interface BDDC system M_{gam}^{-1} * y_{Gam}

        // get coarse vertex (V) loads for later
        constexpr bool SCALED = true;
        addVecGamtoIEV<SCALED>(1.0, gam_rhs, 0.0, f_IEV);
        addVecIEVtoVc<SCALED>(1.0, f_IEV, 0.0, f_V);

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
        solveCoarse(f_V, u_V);

        // compute updated IE loads from coarse vertex (V) solution
        //    uses IEV system as intermediary
        addVecVctoIEV(1.0, u_V, 0.0, temp_IEV);
        kmat_IEV->mult(-1.0, temp_IEV, 1.0, f_IEV);
        addVecIEVtoIE(1.0, f_IEV, 0.0, f_IE);

        // solve interior+edge (IE) subdomain-parallel matrix again
        u_IE->zeroAll();
        solveSubdomainIE(f_IE, u_IE);

        // harmonic extension from IE to full interface solution (EV = gam) to end precond-solve
        addVecIEtoIEV(1.0, u_IE, 1.0, u_IEV);
        addVecVctoIEV<SCALED>(1.0, u_V, 1.0, u_IEV);
        addVecIEVtoGam<SCALED>(1.0, u_IEV, 0.0, gam);

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
        // sgpu = single GPU
        sgpu_num_subdomains = split->num_subdomains;

        // Original/single-GPU splitting copied directly from split
        sgpu_I_nnodes = split->I_nnodes;
        sgpu_IE_nnodes = split->IE_nnodes;
        sgpu_IEV_nnodes = split->IEV_nnodes;
        sgpu_Vc_nnodes = split->Vc_nnodes;
        sgpu_V_nnodes = split->V_nnodes;
        sgpu_lam_nnodes = split->lam_nnodes;

        sgpu_elem_sd_ind = copy_vec(split->elem_sd_ind);
        sgpu_node_class_ind = copy_vec(split->node_class_ind);
        sgpu_node_nsd = copy_vec(split->node_nsd);

        sgpu_IEV_sd_ptr = copy_vec(split->IEV_sd_ptr);
        sgpu_IEV_sd_ind = copy_vec(split->IEV_sd_ind);
        sgpu_IEV_nodes = copy_vec(split->IEV_nodes);
        sgpu_IEV_elem_conn = copy_vec(split->IEV_elem_conn);

        // d_sgpu_IEV_elem_conn =
        //     HostVec<int>(num_elements * nodes_per_elem, sgpu_IEV_elem_conn).createDeviceVec();

        // make partition for IEV conn from IEV splitting (uses those subdomains to assign labels)
        part_IEV =
            new Partition(ngpus, sgpu_IEV_nnodes, num_elements, nodes_per_elem, sgpu_IEV_elem_conn,
                          part->num_components, part->h_elem_components, split);

        create_multigpu_splitting();

        d_IEV_xpts = new Vec(ctx, part_IEV, block_dim);
        d_IEV_vars = new Vec(ctx, part_IEV, block_dim);
        // mat_IEV = new Mat(ctx, part_IEV, block_dim);
    }

    void create_multigpu_splitting() {
        // determine which subdomains belong to which GPUs from the partition..

        subdomain_gpu_ind = new int[sgpu_num_subdomains];
        memset(subdomain_gpu_ind, -1, sgpu_num_subdomains * sizeof(int));
        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int gpu = part->find_owned_gpu_from_elem(e);
            subdomain_gpu_ind[s] = gpu;
        }

        // get local nnodes and nelems on each GPU
        local_nnodes = new int[ngpus];
        local_nelems = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            local_nnodes[g] = part->local_nnodes[g];
            local_nelems[g] = part->local_nelems[g];
        }

        // compute glob to local elem map
        int *elem_ctr = new int[ngpus];
        memset(elem_ctr, 0, ngpus * sizeof(int));
        glob_loc_elem_map = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            glob_loc_elem_map[g] = new int[local_nelems[g]];
        }
        for (int e = 0; e < num_elements; e++) {
            int s = sgpu_elem_sd_ind[e];
            int g = subdomain_gpu_ind[s];
            int ered = elem_ctr[g]++;
            glob_loc_elem_map[g][e] = ered;
        }

        // compute elem_sd_ind on each local GPU
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
        num_subdomains = new int[ngpus];
        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> subdomains;
            for (int ered = 0; ered < local_nelems[g]; ered++) {
                int s = elem_sd_ind[g][ered];
                subdomains.insert(s);
            }
            num_subdomains[g] = subdomains.size();
        }
        sd_cts = new int[ngpus];
        std::memset(sd_cts, 0, ngpus * sizeof(int));

        red_subdomains = new int *[ngpus];
        std::memset(red_subdomains, 0, ngpus * sizeof(int *));

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
        node_class_ind = new int *[ngpus];
        // node_nsd = new int *[ngpus];
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

            ngam[g] = lam_nnodes[g] + Vc_nnodes[g];  // for BDDC E+V basically (full interface)
        }

        // don't think I also need IEV_sd_ptr and IEV_sd_ind (temp arrays and we can just get this
        // from the single gpu ones)

        // so just fill out these two arrays on each local GPU
        IEV_nodes = new int *[ngpus];
        IEV_loc_to_glob = new int *[ngpus];
        // IEV_glob_to_loc = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            IEV_nodes[g] = new int[IEV_nnodes[g]];
            IEV_loc_to_glob[g] = new int[IEV_nnodes[g]];
            // IEV_glob_to_loc[g] = new int[sgpu_IEV_nnodes];
        }
        int *IEV_cts = new int[ngpus];
        memset(IEV_cts, 0, ngpus * sizeof(int));
        for (int s = 0; s < sgpu_num_subdomains; s++) {
            int gpu = subdomain_gpu_ind[s];
            for (int iev = sgpu_IEV_sd_ptr[s]; iev < sgpu_IEV_sd_ptr[s + 1]; iev++) {
                int n = sgpu_IEV_nodes[iev];
                int iev_red = IEV_cts[gpu]++;
                int n_red = part->global_to_local[gpu][n];
                IEV_nodes[gpu][iev_red] = n_red;
                IEV_loc_to_glob[gpu][iev_red] = iev;
                // IEV_glob_to_loc[gpu][iev] = iev_red;
            }
        }

        // fill out IEV_sd_ptr
        IEV_sd_ptr = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            memset(IEV_sd_ptr[g], 0, (num_subdomains[g] + 1) * sizeof(int));
            for (int sred = 0; sred < num_subdomains[g]; sred++) {
                int s = red_subdomains[g][sred];
                int d_iev = sgpu_IEV_sd_ptr[s + 1] - sgpu_IEV_sd_ptr[s];
                IEV_sd_ptr[g][sred + 1] = IEV_sd_ptr[g][sred] + d_iev;
            }
        }

        // then fill out IEV_elem_conn
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
        // now fill out the IEV_elem_conn
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
        d_IEV_elem_conn = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            d_IEV_elem_conn[g] = HostVec<int>(local_nelems[g] * nodes_per_elem, IEV_elem_conn[g])
                                     .createDeviceVec()
                                     .getPtr();
        }
    }
    void build_IE_I_V_maps() {
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

            d_IE_interior[g] =
                HostVec<bool>(IE_nnodes[g], IE_interior[g]).createDeviceVec().getPtr();
            d_IE_general_edge[g] =
                HostVec<bool>(IE_nnodes[g], IE_general_edge[g]).createDeviceVec().getPtr();
            d_IE_nodes[g] = HostVec<int>(IE_nnodes[g], IE_nodes[g]).createDeviceVec().getPtr();
            d_IEVtoIE_imap[g] =
                HostVec<int>(IE_nnodes[g], IEVtoIE_imap[g]).createDeviceVec().getPtr();
            d_IEVtoI_imap[g] = HostVec<int>(I_nnodes[g], IEVtoI_imap[g]).createDeviceVec().getPtr();

            std::memset(IEVtoIE_map[g], -1, IEV_nnodes[g] * sizeof(int));
            std::memset(IEVtoI_map[g], -1, IEV_nnodes[g] * sizeof(int));
            std::memset(IEVtoIE_imap[g], -1, IE_nnodes[g] * sizeof(int));
            std::memset(IEVtoI_imap[g], -1, I_nnodes[g] * sizeof(int));

            int IE_ind = 0;
            int I_ind = 0;

            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int gnode = IEV_nodes[g][iev];
                int cls = node_class_ind[g][gnode];
                int node_class = node_class_ind[g][iev];

                bool is_I = (cls == IEV_INTERIOR || cls == IEV_DIRICHLET_EDGE);
                bool is_IE = is_I || cls == IEV_EDGE;

                if (is_IE) {
                    IE_interior[g][IE_ind] =
                        node_class == IEV_INTERIOR || node_class == IEV_DIRICHLET_EDGE;
                    IE_general_edge[g][IE_ind] =
                        node_class == IEV_INTERIOR || node_class == IEV_DIRICHLET_EDGE;
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
        }

        build_Vc_and_gam_maps();
    }

    void build_Vc_and_gam_maps() {
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
            std::unordered_set<int> Vc_set;

            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int gnode = IEV_nodes[g][iev];
                if (node_class_ind[g][gnode] == IEV_VERTEX) {
                    Vc_set.insert(gnode);
                }
            }

            std::vector<int> Vc_vec(Vc_set.begin(), Vc_set.end());
            std::sort(Vc_vec.begin(), Vc_vec.end());

            Vc_nodes[g] = new int[Vc_vec.size()];
            for (int i = 0; i < (int)Vc_vec.size(); i++) {
                Vc_nodes[g][i] = Vc_vec[i];
            }

            std::vector<int> Vc_inodes(num_nodes, -1);
            for (int i = 0; i < (int)Vc_vec.size(); i++) {
                Vc_inodes[Vc_vec[i]] = i;
            }

            int V_ind = 0;
            for (int iev = 0; iev < IEV_nnodes[g]; iev++) {
                int gnode = IEV_nodes[g][iev];
                if (node_class_ind[g][gnode] == IEV_VERTEX) {
                    IEVtoV_imap[g][V_ind] = iev;
                    VctoV_imap[g][V_ind] = Vc_inodes[gnode];
                    V_ind++;
                }
            }

            d_IEVtoV_imap[g] = HostVec<int>(V_nnodes[g], IEVtoV_imap[g]).createDeviceVec().getPtr();
            d_VctoV_imap[g] = HostVec<int>(V_nnodes[g], VctoV_imap[g]).createDeviceVec().getPtr();

            n_edge[g] = lam_nnodes[g];
            ngam[g] = n_edge[g] + Vc_nnodes[g];
            gam_nodes[g] = new int[ngam[g]];

            int e = 0;
            for (int inode = 0; inode < part->local_nnodes[g]; inode++) {
                if (node_class_ind[g][inode] == IEV_EDGE) {
                    if (e < n_edge[g]) gam_nodes[g][e++] = inode;
                }
            }

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                gam_nodes[g][n_edge[g] + i] = Vc_nodes[g][i];
            }

            d_Vc_nodes[g] = HostVec<int>(Vc_nnodes[g], Vc_nodes[g]).createDeviceVec().getPtr();
        }
    }

    void build_IEV_sparsity() {
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
        }
    }

    void build_IE_and_I_sparsity() {
        // build IE and I reduced local matrices from IEV matrix sparsity

        // -----------------------------------------
        // reduced rowp arrays in multi-GPU environment (for kmat_IE and kmat_I)
        // -----------------------------------------
        IE_rowp = new int *[ngpus];
        I_rowp = new int *[ngpus];
        I_nnzb = new int[ngpus];
        IE_nnzb = new int[ngpus];
        IE_rows = new int *[ngpus];
        I_rows = new int *[ngpus];
        d_IE_vals = new DeviceVec<T>[ngpus];
        d_I_vals = new DeviceVec<T>[ngpus];

        for (int g = 0; g < ngpus; g++) {
            IE_rowp[g] = new int[IE_nnodes[g] + 1];
            I_rowp[g] = new int[I_nnodes[g] + 1];
            std::memset(IE_rowp[g], 0, (IE_nnodes[g] + 1) * sizeof(int));
            std::memset(I_rowp[g], 0, (I_nnodes[g] + 1) * sizeof(int));

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

            // get rows for I and IE
            I_rows[g] = new int[I_nnzb[g]];
            for (int inode = 0; inode < I_nnodes[g]; inode++) {
                for (int jp = I_rowp[g][inode]; jp < I_rowp[g][inode + 1]; jp++) {
                    I_rows[g][jp] = inode;
                }
            }
            IE_rows[g] = new int[IE_nnzb[g]];
            for (int inode = 0; inode < IE_nnodes[g]; inode++) {
                for (int jp = IE_rowp[g][inode]; jp < IE_rowp[g][inode + 1]; jp++) {
                    IE_rows[g][jp] = inode;
                }
            }

            // make matrix values
            d_IE_vals[g] = DeviceVec<T>(block_dim2 * IE_nnzb[g]);
            d_I_vals[g] = DeviceVec<T>(block_dim2 * I_nnzb[g]);
        }  // done with gpu loop
    }

    void create_kmat_copy_maps() {
        // -----------------------------------------
        // IEV => IE kmat block copy map
        // -----------------------------------------
        kmat_IEnofill_map = new int *[ngpus];
        kmat_IEtoIEV_map = new int *[ngpus];
        d_kmat_IEnofill_map = new int *[ngpus];
        d_kmat_IEtoIEV_map = new int *[ngpus];

        // no need for IE_nofill_nnzb (CuDSS will do fillin, so our code automatically has nofill)
        for (int g = 0; g < ngpus; g++) {
            kmat_IEnofill_map[g] = new int[IE_nnzb[g]];
            kmat_IEtoIEV_map[g] = new int[IE_nnzb[g]];
            memset(kmat_IEnofill_map[g], -1, IE_nnzb[g] * sizeof(int));
            memset(kmat_IEtoIEV_map[g], -1, IE_nnzb[g] * sizeof(int));
            int nofill_ind = 0;
            // also there are no permutations anymore (since CuDSS does reordering)
            for (int i = 0; i < IE_nnodes[g]; i++) {
                for (int jp = IE_rowp[g][i]; jp < IE_rowp[g][i + 1]; jp++) {
                    int j = IE_cols[g][jp];
                    // find equivalent nz block of IEV rowp
                    int i_IEV = IEVtoIE_imap[g][i];
                    int j_IEV = IEVtoIE_imap[g][j];
                    bool found = false;
                    for (int kp = IEV_rowp[g][i_IEV]; kp < IEV_rowp[g][i_IEV + 1]; kp++) {
                        int k = IEV_cols[g][kp];
                        if (k == j_IEV) {
                            kmat_IEnofill_map[g][nofill_ind] = jp;
                            kmat_IEtoIEV_map[g][nofill_ind] = kp;
                            nofill_ind++;
                            found = true;
                        }
                    }
                }
            }

            // printf("SETUP MATRIX SPARSITY 4\n");

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

        // no need for IE_nofill_nnzb (CuDSS will do fillin, so our code automatically has nofill)
        for (int g = 0; g < ngpus; g++) {
            kmat_Inofill_map[g] = new int[I_nnzb[g]];
            kmat_ItoIEV_map[g] = new int[I_nnzb[g]];
            memset(kmat_Inofill_map[g], -1, I_nnzb[g] * sizeof(int));
            memset(kmat_ItoIEV_map[g], -1, I_nnzb[g] * sizeof(int));
            int nofill_ind = 0;
            // also there are no permutations anymore (since CuDSS does reordering)
            for (int i = 0; i < I_nnodes[g]; i++) {
                for (int jp = I_rowp[g][i]; jp < I_rowp[g][i + 1]; jp++) {
                    int j = I_cols[g][jp];
                    // find equivalent nz block of IEV rowp
                    int i_IEV = IEVtoI_imap[g][i];
                    int j_IEV = IEVtoI_imap[g][j];
                    bool found = false;
                    for (int kp = IEV_rowp[g][i_IEV]; kp < IEV_rowp[g][i_IEV + 1]; kp++) {
                        int k = IEV_cols[g][kp];
                        if (k == j_IEV) {
                            kmat_Inofill_map[g][nofill_ind] = jp;
                            kmat_ItoIEV_map[g][nofill_ind] = kp;
                            nofill_ind++;
                            found = true;
                        }
                    }
                }
            }

            // printf("SETUP MATRIX SPARSITY 4\n");

            d_kmat_ItoIEV_map[g] =
                HostVec<int>(I_nnzb[g], kmat_ItoIEV_map[g]).createDeviceVec().getPtr();
            d_kmat_Inofill_map[g] =
                HostVec<int>(I_nnzb[g], kmat_Inofill_map[g]).createDeviceVec().getPtr();
        }
    }
    void build_Svv_sparsity() {
        Vc_node_imap = new int *[ngpus];
        Svv_rowp = new int *[ngpus];
        Svv_cols = new int *[ngpus];
        Svv_nnzb = new int[ngpus];
        Svv_rows = new int *[ngpus];
        d_Svv_vals = new DeviceVec<T>[ngpus];

        for (int g = 0; g < ngpus; g++) {
            // reverse map of global => reduced Vc nodes
            Vc_node_imap[g] = new int[local_nnodes[g]];
            memset(Vc_node_imap[g], -1, num_nodes * sizeof(int));
            for (int vnode = 0; vnode < Vc_nnodes[g]; vnode++) {
                int glob_node = Vc_nodes[g][vnode];
                Vc_node_imap[g][glob_node] = vnode;
            }

            // build unique adjacency per coarse row
            std::vector<std::unordered_set<int>> Svv_adj(Vc_nnodes[g]);

            for (int i_subdomain = 0; i_subdomain < num_subdomains[g]; i_subdomain++) {
                std::unordered_set<int> sd_Vc_nodeset;

                for (int ielem = 0; ielem < local_nelems[g]; ielem++) {
                    // TODO : need to somehow get this in local node numbers? (cause loc_elem_conn
                    // isn't state yet)
                    int *loc_elem_conn = kmat_IEV->getLocalElemConn(g);
                    int *local_nodes = &loc_elem_conn[nodes_per_elem * ielem];
                    int j_subdomain = elem_sd_ind[g][ielem];
                    if (i_subdomain != j_subdomain) continue;

                    for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                        int gnode = local_nodes[lnode];
                        int node_class = node_class_ind[g][gnode];
                        if (node_class == IEV_VERTEX) {
                            int vnode = Vc_node_imap[g][gnode];
                            if (vnode >= 0) {
                                sd_Vc_nodeset.insert(vnode);
                            }
                        }
                    }
                }

                std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());

                // add unique couplings for this subdomain
                for (int i : sd_Vc_nodes) {
                    for (int j : sd_Vc_nodes) {
                        Svv_adj[i].insert(j);
                    }
                }
            }

            // row counts from unique adjacency
            int *Svv_rowcts = new int[Vc_nnodes[g]];
            memset(Svv_rowcts, 0, Vc_nnodes[g] * sizeof(int));
            for (int i = 0; i < Vc_nnodes[g]; i++) {
                Svv_rowcts[i] = static_cast<int>(Svv_adj[i].size());
            }

            // fill rowp
            Svv_rowp[g] = new int[Vc_nnodes[g] + 1];
            memset(Svv_rowp[g], 0, (Vc_nnodes[g] + 1) * sizeof(int));
            for (int i = 0; i < Vc_nnodes[g]; i++) {
                Svv_rowp[g][i + 1] = Svv_rowp[g][i] + Svv_rowcts[i];
            }
            Svv_nnzb[g] = Svv_rowp[g][Vc_nnodes[g]];

            // fill cols
            Svv_cols[g] = new int[Svv_nnzb[g]];
            memset(Svv_cols[g], 0, Svv_nnzb[g] * sizeof(int));

            for (int i = 0; i < Vc_nnodes[g]; i++) {
                int jp = Svv_rowp[g][i];
                for (int j : Svv_adj[i]) {
                    Svv_cols[g][jp++] = j;
                }

                // optional but recommended: sort each row
                std::sort(&Svv_cols[g][Svv_rowp[g][i]], &Svv_cols[g][Svv_rowp[g][i + 1]]);
            }

            Svv_rows[g] = new int[Svv_nnzb[g]];
            for (int i = 0; i < Vc_nnodes[g]; i++) {
                for (int jp = Svv_rowp[g][i]; jp < Svv_rowp[g][i + 1]; jp++) {
                    Svv_rows[g][jp] = i;
                }
            }

            d_Svv_vals[g] = DeviceVec<T>(Svv_nnzb[g] * block_dim2);
        }
    }

    void build_Svv_maps() {
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
                            }
                        }
                    }
                }
            }
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

            // CONSTRUCT coarse Schur complement mat-invmat-mat maps
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

            for (int isd = 0; isd < num_subdomains[g]; isd++) {
                std::vector<int> sd_iev_vertex_blocks;
                std::vector<int> sd_vc_nodes;

                for (int jp = IEV_sd_ptr[g][isd]; jp < IEV_sd_ptr[g][isd + 1]; jp++) {
                    int gnode = IEV_nodes[g][jp];
                    if (node_class_ind[g][gnode] == IEV_VERTEX) {
                        sd_iev_vertex_blocks.push_back(jp);

                        int vc_node = -1;
                        for (int j = 0; j < Vc_nnodes[g]; j++) {
                            if (Vc_nodes[g][j] == gnode) {
                                vc_node = j;
                                break;
                            }
                        }

                        if (vc_node < 0) {
                            printf("ERROR: vertex gnode %d on subdomain %d not found in Vc_nodes\n",
                                   gnode, isd);
                            exit(-1);
                        }

                        sd_vc_nodes.push_back(vc_node);
                    }
                }

                const int nsv = static_cast<int>(sd_iev_vertex_blocks.size());
                if (nsv == 0) continue;

                if (nsv > MAX_NUM_VERTEX_PER_SUBDOMAIN) {
                    printf("ERROR: subdomain %d has %d local vertex slots (>%d)\n", isd, nsv,
                           MAX_NUM_VERTEX_PER_SUBDOMAIN);
                    exit(-1);
                }

                for (int k = 0; k < nsv; k++) {
                    const int iev_block = sd_iev_vertex_blocks[k];
                    const int vc_row = sd_vc_nodes[k];
                    // const int vc_row_perm = SVV_iperm[vc_row];

                    IEVset_blocks_host[k].push_back(iev_block);

                    for (int kk = 0; kk < nsv; kk++) {
                        const int iev_block2 = sd_iev_vertex_blocks[kk];
                        const int vc_col = sd_vc_nodes[kk];

                        int svv_block = -1;
                        for (int jp = Svv_rowp[g][vc_row]; jp < Svv_rowp[g][vc_row + 1]; jp++) {
                            int m = Svv_cols[g][jp];
                            // int m = SVV_perm[m_perm];
                            if (m == vc_col) {
                                svv_block = jp;
                                break;
                            }
                        }

                        if (svv_block < 0) {
                            printf(
                                "ERROR: could not find global Svv block for subdomain %d, row %d, "
                                "col "
                                "%d\n",
                                isd, vc_row, vc_col);
                            exit(-1);
                        }

                        IEVout_blocks_host[k].push_back(iev_block2);
                        IEVtoSVV_blocks_host[k].push_back(svv_block);
                    }
                }
            }

            for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
                IEVset_nnzb[g][k] = static_cast<int>(IEVset_blocks_host[k].size());
                IEVtoSVV_nnzb[g][k] = static_cast<int>(IEVtoSVV_blocks_host[k].size());

                if ((int)IEVout_blocks_host[k].size() != IEVtoSVV_nnzb[g][k]) {
                    printf("ERROR: slot %d mismatch: IEVout size %d != IEVtoSVV size %d\n", k,
                           (int)IEVout_blocks_host[k].size(), IEVtoSVV_nnzb[k]);
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
        }  // end of GPU loop
    }
    void build_iev_bcs() {
        assembler->getLocalDeviceBCs(n_owned_bcs, n_local_bcs, d_owned_bcs, d_local_bcs);
        n_IEV_owned_bcs = new int[ngpus];
        n_IEV_local_bcs = new int[ngpus];
        d_IEV_owned_bcs = new int *[ngpus];
        d_IEV_local_bcs = new int *[ngpus];
        for (int g = 0; g < ngpus; g++) {
            // compute the BC indices needed for kmat_IEV

            // local bcs
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
            n_IEV_local_bcs[g] = IEV_local_bcs_vec.size();
            d_IEV_local_bcs[g] = HostVec<int>(IEV_local_bcs_vec.size(), IEV_local_bcs_vec.data())
                                     .createDeviceVec()
                                     .getPtr();

            // owned bcs
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
            n_IEV_owned_bcs[g] = IEV_owned_bcs_vec.size();
            d_IEV_owned_bcs[g] = HostVec<int>(IEV_owned_bcs_vec.size(), IEV_owned_bcs_vec.data())
                                     .createDeviceVec()
                                     .getPtr();
        }
    }

    void compute_reduced_partitions() {
        // TBD : do these ones need diff partitioner not part_IEV?
        // or they need to be double pointers? prob double pointers..

        // build reduced connectivities for each first..
        part_IE = new SDPartition(ngpus, num_nodes, IE_nnodes, IE_nodes, IEV_nodes, part_IEV);
        part_I = new SDPartition(ngpus, num_nodes, I_nnodes, I_nodes, IEV_nodes, part_IEV);
        part_V = new SDPartition(ngpus, num_nodes, Vc_nnodes, Vc_nodes, IEV_nodes, part_IEV);
        part_gam = new SDPartition(ngpus, num_nodes, ngam, gam_nodes, IEV_nodes, part_IEV);
    }

    void allocate_vectors() {
        d_IEV_xpts = new Vec(ctx, part_IEV, block_dim);
        d_IEV_vars = new Vec(ctx, part_IEV, block_dim);
        fext_IEV = new Vec(ctx, part_IEV, block_dim);
        fint_IEV = new Vec(ctx, part_IEV, block_dim);
        res_IEV = new Vec(ctx, part_IEV, block_dim);
        f_IEV = new Vec(ctx, part_IEV, block_dim);
        u_IEV = new Vec(ctx, part_IEV, block_dim);
        temp_IEV = new Vec(ctx, part_IEV, block_dim);

        f_IE = new SDVec(ctx, part_IE, block_dim);
        u_IE = new SDVec(ctx, part_IE, block_dim);
        f_I = new SDVec(ctx, part_I, block_dim);
        u_I = new SDVec(ctx, part_I, block_dim);
        f_V = new SDVec(ctx, part_V, block_dim);
        u_V = new SDVec(ctx, part_V, block_dim);
        temp_lam = new SDVec(ctx, part_gam, block_dim);
        temp_lam2 = new SDVec(ctx, part_gam, block_dim);
        d_coarse_vars = new SDVec(ctx, part_V, block_dim);
    }

    void clear_host_data() {}

    void assemble_subdomains() {
        addVec_globalToIEV(1.0, d_xpts, 0.0, d_IEV_xpts, 3);
        addVec_globalToIEV(1.0, d_vars, 0.0, d_IEV_vars, block_dim);

        kmat_IEV->zeroValues();
        add_IEV_jacobian();
        // fext_IEV->zeroValues();
        // add_subdomain_fext();

        // TODO : something like this here.. but for each GPU matrix..
        apply_IEV_bcs();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            cudaMemset(d_IE_vals, 0.0, block_dim2 * IE_nnzb[g] * sizeof(T));
            cudaMemset(d_I_vals, 0.0, block_dim2 * I_nnzb[g] * sizeof(T));
        }
        ctx->sync();
        copyKmat_IEVtoIE();
        copyKmat_IEVtoI();
    }

    void add_IEV_jacobian() {
        const int cols_per_elem = 24;  // for 1st order element
        const int elems_per_block = 1;

        d_IEV_xpts->expandToLocal();
        d_IEV_vars->expandToLocal();

        dim3 block(num_quad_pts, cols_per_elem, elems_per_block);
        int elem_cols_per_block = cols_per_elem * elems_per_block;

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_num_nodes = d_IEV_xpts->getExpandedNodes(g);
            int loc_nelems = part->getLocalNumElements(g);

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

            dim3 grid(nblocks), block(32);

            k_add_multigpu_jacobian_fast<T, elems_per_block, Assembler>
                <<<grid, block, 0, streams[g]>>>(
                    loc_num_nodes, loc_nelems, cols_per_elem, loc_elem_comps, loc_elem_conn_ptr,
                    loc_xpts_ptr, loc_vars_ptr, loc_comp_data_ptr, loc_elem_ind_map, loc_mat_vals);

            CHECK_CUDA(cudaGetLastError());
        }
    }

    void assemble_coarse_problem() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            d_Svv_vals[g].zeroValues();
        }
        ctx->sync();

        copyKmat_IEVtoSvv();
        computeSvvInverseTerm();
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

    void computeSvvInverseTerm() {
        int ncols = MAX_NUM_VERTEX_PER_SUBDOMAIN * block_dim;
        for (int icol = 0; icol < ncols; icol++) {
            u_IEV->zeroAll();
            setVec_IEVtoV_vals(u_IEV, icol, 1.0);  // set these vals to 1.0 and all else 0

            kmat_IEV->mult(1.0, u_IEV, 0.0, f_IEV);
            addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
            solveSubdomainIE(f_IE, u_IE);
            addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

            kmat_IEV->mult(-1.0, u_IEV, 0.0, f_IEV);
            addMat_IEVtoV_vals(icol, f_IEV);
        }
    }

    void create_cudss_solvers() {
        // build Svv as a GPUbsrmat
        // build_Svv_gpumat();

        subdomain_IE_solver = new CudssSubdomainBsrSolve<T>(ctx, num_nodes, block_dim, IE_rowp,
                                                            IE_cols, IE_nnzb, d_IE_vals);

        subdomain_I_solver = new CudssSubdomainBsrSolve<T>(ctx, num_nodes, block_dim, I_rowp,
                                                           I_cols, I_nnzb, d_I_vals);

        // optional: alternate constructors to build an Svv_mat despite not having elem conn?
        // Svv_part = new LightPartitioner(ctx, sgpu_Vc_nnodes, Vc_nodes);
        // Svv_mat = GPUbsrmat<T, LightPartitioner>(ctx, Svv_part, block_dim, Svv_rowp, Svv_cols,
        //                                          Svv_nnzb, Svv_rows, d_Svv_vals);
        Svv_solver = new CudssMgBSRSolverV2<T>(ctx, sgpu_Vc_nnodes, Vc_nnodes, Vc_nodes, block_dim,
                                               Svv_rowp, Svv_cols, Svv_nnzb, Svv_rows, d_Svv_vals);
    }

    // deprecated
    // void build_Svv_gpumat() {
    //     // TODO : how to best do this? alternate constructor?
    // }

    void solveSubdomainIE(SDVec *rhs_in, SDVec *sol_out) {
        subdomain_IE_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
    }
    void solveSubdomainI(SDVec *rhs_in, SDVec *sol_out) {
        subdomain_I_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
    }
    void solveCoarse(SDVec *rhs_in, SDVec *sol_out) {
        // TODO : prob shouldn't be using double pointers  here.. TBD
        Svv_solver->solve(rhs_in->getLocalDoublePtr(), sol_out->getLocalDoublePtr());
    }

    void setVec_IEVtoV_vals(Vec *vec_IEV, int irow, T val) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int block_row = irow / block_dim;
            int set_nnzb = IEVset_nnzb[g][block_row];
            int *d_blocks = d_IEVset_blocks[g][block_row];
            T *loc_vec_IEV = vec_IEV->getLocalPtr(g);

            dim3 block(32);
            dim3 grid((set_nnzb + 31) / 32);
            k_setVec_IEVtoV_vals<T><<<grid, block, 0, streams[g]>>>(set_nnzb, block_dim, irow,
                                                                    d_blocks, loc_vec_IEV, val);

            CHECK_CUDA(cudaGetLastError());
        }
    }

    void addMat_IEVtoV_vals(const int icol, SDVec *hvec) {
        // helps assembly of S_VV schur complement matrix on each local GPU partition

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int block_col = icol / block_dim;
            int set_nnzb = IEVtoSVV_nnzb[g][block_col];
            int *d_svv_blocks = d_IEVtoSVV_blocks[g][block_col];
            int *d_iev_blocks = d_IEVout_blocks[g][block_col];

            T *loc_hvec = hvec->getLocalPtr(g);

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
        u_IE->zeroAll();
        u_IE->zeroAll();

        // add from global to IEV on each GPU partition
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *loc_xglob = x_global->getLocalPtr(g);
            T *loc_uIE = u_IE->getLocalPtr(g);
            T *loc_uV = u_V->getLocalPtr(g);
            T *loc_yIEV = y_iev->getLocalPtr(g);

            int nvals = IE_nnodes[g] * vars_per_node_in;
            dim3 block(32), grid((nvals + 31) / 32);
            k_addVec_GlobalToIE<T, scaled><<<grid, block, 0, streams[g]>>>(
                IE_nnodes[g], vars_per_node_in, d_IE_nodes[g], d_IE_nsd[g], loc_xglob, loc_uIE, a);

            int nvals2 = Vc_nnodes[g] * vars_per_node_in;
            dim3 grid2((nvals2 + 31) / 32);
            // scales by 0.25x like for load distribution
            k_addVec_GlobaltoVc<T, scaled>
                <<<grid2, block, 0, streams[g]>>>(Vc_nnodes[g], vars_per_node_in, d_Vc_nodes[g],
                                                  d_vertex_nsd[g], loc_xglob, loc_uV, a);
            CHECK_CUDA(cudaGetLastError());
        }
        // reduce across ghost nodes (so matches single GPU result)
        // TODO : should I delay reduction until only for y_iev though? TBD..
        u_V->reduceFromLocal();
        u_IE->reduceFromLocal();

        addVecIEtoIEV(1.0, u_IE, 0.0, y_iev, vars_per_node_in);
        addVecVctoIEV(1.0, u_V, 1.0, y_iev, vars_per_node_in);
    }

    void addVecIEVtoIE(T a, Vec *x, T b, SDVec *y) {
        y->scale(b);
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
    }

    void addVecIEVtoI(T a, Vec *x, T b, SDVec *y) {
        y->scale(b);
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
    }

    void addVecIEtoI(T a, SDVec *x, T b, SDVec *y) {
        addVecIEtoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoI(temp_IEV, y, 1.0, b);
    }

    template <bool scaled = false>
    void addVecIEVtoVc(T a, Vec *x, T b, SDVec *y) {
        y->scale(b);
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
    }

    void addVecGamtoIE(T a, SDVec *gam, T b, SDVec *vec_IE) {
        vec_IE->scale(b);
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
    }

    void addVecIEtoIEV(T a, SDVec *x, T b, Vec *y, int vars_per_node = -1) {
        y->scale(b);
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
    }
    void addVecItoIEV(T a, SDVec *x, T b, Vec *y) {
        y->scale(b);
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
    }
    void addVecItoIE(T a, SDVec *x, T b, SDVec *y) {
        addVecItoIEV(a, x, 0.0, temp_IEV);
        addVecIEVtoIE(1.0, temp_IEV, b, y);
    }
    template <bool scaled = false>
    void addVecVctoIEV(T a, SDVec *x, T b, Vec *y, int vars_per_node = -1) {
        x->copyTo(temp_V);
        y->scale(b);
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
    }
    void addVecIEtoGam(T a, SDVec *vec_IE, T b, SDVec *gam) {
        gam->scale(b);
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
    }
    void addGlobalSoln(SDVec *u_IE, SDVec *u_V, Vec *soln) {
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
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(T alpha, Vec *vec_IEV, T beta, SDVec *vec_gam) {
        vec_gam->scale(beta);

        // add IEV to E part of Gam first (then V later)
        addVecIEVtoIE(alpha, vec_IEV, 0.0, temp_IE);
        addVecIEtoGam(1.0, temp_IE, 0.0, temp_lam);

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

        addVecIEVtoVc(alpha, vec_IEV, 0.0, temp_V);

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

        // then scale vec_gam
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(T alpha, SDVec *vec_gam, T beta, Vec *vec_IEV) {
        vec_IEV->scale(beta);
        temp_lam->zeroAll();
        temp_V->zeroAll();

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

        addVecGamtoIE(1.0, temp_lam, 0.0, temp_IE);
        addVecIEtoIEV(1.0, temp_IE, 0.0, vec_IEV);

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

        addVecVctoIEV(1.0, temp_V, 0.0, vec_IEV);
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

    int **d_IE_nsd = nullptr;
    int **d_edge_nsd = nullptr;
    int **d_vertex_nsd = nullptr;
    int **d_IE_to_lam_map = nullptr;

    DeviceVec<T> *d_IEV_vals = nullptr;
    DeviceVec<T> *d_IE_vals = nullptr;
    DeviceVec<T> *d_I_vals = nullptr;
    DeviceVec<T> *d_Svv_vals = nullptr;

    int **IEV_rowp = nullptr;
    int **IEV_cols = nullptr;
    int *IEV_nnzb = nullptr;
    int **IEV_rows = nullptr;

    int **IE_rowp = nullptr;
    int **IE_cols = nullptr;
    int *IE_nnzb = nullptr;
    int **IE_rows = nullptr;

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
    SDVec *f_I = nullptr;
    SDVec *u_I = nullptr;
    SDVec *f_V = nullptr;
    SDVec *u_V = nullptr;
    SDVec *temp_lam = nullptr;
    SDVec *temp_lam2 = nullptr;
    SDVec *d_coarse_vars = nullptr;

    CudssSubdomainBsrSolve<T> *subdomain_IE_solver = nullptr;
    CudssSubdomainBsrSolve<T> *subdomain_I_solver = nullptr;
    CudssMgBSRSolverV2<T> *Svv_solver = nullptr;
};