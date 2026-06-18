#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_fetidp.cuh"
#include "cuda_utils.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "linalg/svd_utils.h"
#include "mesh/exploded_vtk_writer.h"
#include "mesh/vtk_writer.h"
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/solvers/solve_utils.h"

enum NodeClass { INTERIOR = 0, DIRICHLET_EDGE = 1, EDGE = 2, VERTEX = 3 };

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_>
class FetidpSolver : public BaseSolver {
   public:
    using ShellAssembler = ShellAssembler_;
    using Director = typename ShellAssembler::Director;
    using Basis = typename ShellAssembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename ShellAssembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Vec = Vec_<T>;
    using Mat = Mat_<Vec_<T>>;
    using BsrMatType = BsrMat<DeviceVec<T>>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    FetidpSolver() = default;

    FetidpSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                 ShellAssembler &assembler_, BsrMatType &kmat_, bool print_timing_ = false)
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          assembler(assembler_),
          subdomainISolver(nullptr),
          subdomainIESolver(nullptr),
          coarseSolver(nullptr),
          elem_sd_ind(nullptr),
          elem_conn(nullptr),
          node_sd_cols(nullptr),
          node_elem_ct(nullptr),
          node_elem_rowp(nullptr),
          kmat(&kmat_),
          node_class_ind(nullptr) {
        num_elements = assembler.get_num_elements();
        num_nodes = assembler.get_num_nodes();
        N = num_nodes * vars_per_node;
        print_timing = print_timing_;

        elem_conn = assembler.getConn().createHostVec().getPtr();

        auto kmat_bsr_data = kmat->getBsrData();
        kmat_nnzb = kmat_bsr_data.nnzb;
        block_dim = kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;

        d_xpts = assembler.getXpts();
        d_vars = assembler.getVars();
        d_elem_components = assembler.getElemComponents();
        d_compData = assembler.getCompData();
        MAX_NUM_VERTEX_PER_SUBDOMAIN = 4;
        S_VV_MLIEV = nullptr;  // unused in 2-level BDDC

        elem_sd_ind = nullptr;
        node_sd_cols = nullptr;
        node_elem_ct = nullptr;
        node_elem_rowp = nullptr;
        node_class_ind = nullptr;

        IEV_nodes = nullptr;
        IE_nodes = nullptr;
        I_nodes = nullptr;
        IE_rowp = nullptr;
        I_rowp = nullptr;
        IE_cols = nullptr;
        I_cols = nullptr;
        IEV_elem_conn = nullptr;
        IEV_sd_ptr = nullptr;
        IEV_sd_ind = nullptr;
        IEVtoIE_map = nullptr;
        IEVtoI_map = nullptr;

        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        using clock = std::chrono::high_resolution_clock;
        using sec = std::chrono::duration<double>;

        // printf("update after assembly\n");

        std::chrono::time_point<clock> t_begin, t0, t1, t_end;

        double t_assemble_subdomains = 0.0;
        double t_factor_IE = 0.0;
        double t_factor_I = 0.0;
        double t_assemble_coarse = 0.0;
        double t_factor_coarse = 0.0;
        double t_total = 0.0;

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t_begin = clock::now();
        }

        // copy vars to d_vars (NL update)
        vars.copyValuesTo(d_vars);

        // -----------------------------------------
        // 1. Assemble subdomain matrices
        // -----------------------------------------
        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t0 = clock::now();
        }

        assemble_subdomains();

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t1 = clock::now();
            t_assemble_subdomains = sec(t1 - t0).count();
        }

        // -----------------------------------------
        // 2. Factor IE matrix
        // -----------------------------------------
        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t0 = clock::now();
        }

        subdomainIESolver->factor();

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t1 = clock::now();
            t_factor_IE = sec(t1 - t0).count();
        }

        // -----------------------------------------
        // 3. Factor I matrix
        // -----------------------------------------
        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t0 = clock::now();
        }

        subdomainISolver->factor();

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t1 = clock::now();
            t_factor_I = sec(t1 - t0).count();
        }

        // -----------------------------------------
        // 4. Assemble coarse problem
        // -----------------------------------------
        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t0 = clock::now();
        }

        assemble_coarse_problem();

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t1 = clock::now();
            t_assemble_coarse = sec(t1 - t0).count();
        }

        // -----------------------------------------
        // 5. Factor coarse solver
        // -----------------------------------------
        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t0 = clock::now();
        }

        coarseSolver->factor();

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t1 = clock::now();
            t_factor_coarse = sec(t1 - t0).count();
        }

        if (print_timing) {
            CHECK_CUDA(cudaDeviceSynchronize());
            t_end = clock::now();
            t_total = sec(t_end - t_begin).count();

            printf("\nupdate_after_assembly timing breakdown:\n");
            printf("  assemble_subdomains   : %.6e s\n", t_assemble_subdomains);
            printf("  factor IE             : %.6e s\n", t_factor_IE);
            printf("  factor I              : %.6e s\n", t_factor_I);
            printf("  assemble coarse       : %.6e s\n", t_assemble_coarse);
            printf("  factor coarse         : %.6e s\n", t_factor_coarse);
            printf("  total                 : %.6e s\n", t_total);

            double tracked = t_assemble_subdomains + t_factor_IE + t_factor_I + t_assemble_coarse +
                             t_factor_coarse;

            printf("  tracked subtotal      : %.6e s\n", tracked);
            printf("  untracked overhead    : %.6e s\n\n", t_total - tracked);
        }
    }
    void factor()
        override {  // needs to be called in the update_after_assembly.. cause several factor steps
    }
    void set_print(bool print) {}
    void set_rel_tol(T rtol) {}
    void set_abs_tol(T atol) {}
    int get_num_iterations() { return 1; }
    void set_cycle_type(std::string cycle_) {}
    void free() {  // TODO
    }

    int get_num_IEV_nodes() { return this->IEV_nnodes; }
    DeviceVec<int> get_IEV_conn() { return this->d_IEV_elem_conn; }
    DeviceVec<T> get_IEV_xpts() { return this->d_IEV_xpts; }
    DeviceVec<T> get_IEV_vars() { return this->d_IEV_vars; }

    ~FetidpSolver() { clear_host_data(); }
    int getLambdaSize() const { return lam_nnodes * block_dim; }

    void setup_unstructured_subdomains(int target_sd_size = 16) {
        clear_structured_host_data();

        // node-element adjacency
        int ne_nnz = 0;
        int *ne_cts = new int[num_nodes];
        memset(ne_cts, 0, num_nodes * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                ne_cts[gnode]++;
                ne_nnz++;
            }
        }
        int *ne_ptr = new int[num_nodes + 1];
        memset(ne_ptr, 0, (num_nodes + 1) * sizeof(int));
        for (int inode = 0; inode < num_nodes; inode++) {
            ne_ptr[inode + 1] = ne_ptr[inode] + ne_cts[inode];
        }
        memset(ne_cts, 0, num_nodes * sizeof(int));
        int *ne_cols = new int[ne_nnz];
        memset(ne_cols, 0, ne_nnz * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = ne_ptr[gnode] + ne_cts[gnode];
                ne_cols[offset] = ielem;
                ne_cts[gnode]++;
                ne_nnz++;
            }
        }
        // DEBUG node-element adjacency
        // for (int inode = 0; inode < num_nodes; inode++) {
        //     printf("node %d: ", inode);
        //     for (int jp = ne_ptr[inode]; jp < ne_ptr[inode + 1]; jp++) {
        //         int ielem = ne_cols[jp];
        //         printf("%d,", ielem);
        //     }
        //     printf("\n");
        // }

        // then build element-element adjacency
        // elem => local nodes on elem => adj elem through elem-elem adjacency
        int *ee_cts = new int[num_elements];
        int ee_nnz = 0;
        memset(ee_cts, 0, num_elements * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;
                    ee_cts[ielem]++;
                    ee_nnz++;
                }
            }
        }
        // printf("here1\n");
        int *ee_ptr = new int[num_elements + 1];
        memset(ee_ptr, 0, (num_elements + 1) * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            ee_ptr[ielem + 1] = ee_ptr[ielem] + ee_cts[ielem];
        }
        memset(ee_cts, 0, num_elements * sizeof(int));
        int *ee_cols = new int[ee_nnz];
        memset(ee_cols, 0, ee_nnz * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;
                    int offset = ee_ptr[ielem] + ee_cts[ielem];
                    ee_cols[offset] = jelem;
                    ee_cts[ielem]++;
                }
            }
        }
        // DEBUG element-element adjacency
        // for (int ielem = 0; ielem < num_elements; ielem++) {
        //     printf("elem %d: ", ielem);
        //     for (int jp = ee_ptr[ielem]; jp < ee_ptr[ielem + 1]; jp++) {
        //         int jelem = ee_cols[jp];
        //         printf("%d,", jelem);
        //     }
        //     printf("\n");
        // }

        // -------------------------------------------
        // begin assigning subdomains
        // -------------------------------------------
        elem_sd_ind = new int[num_elements];
        bool *visited = new bool[num_elements];
        memset(elem_sd_ind, 0, num_elements * sizeof(int));
        memset(visited, 0, num_elements * sizeof(bool));
        int subdomain_ind = 0;
        bool all_visited = false;
        while (!all_visited) {
            // printf("in while loop 1\n");
            // find an un-assigned element
            int elem = -1;
            for (int _elem = 0; _elem < num_elements; _elem++) {
                if (!visited[_elem]) {
                    elem = _elem;
                    break;
                }
            }
            if (elem == -1) break;

            // initialize subdomain
            std::vector<int> sd_elems;
            sd_elems.push_back(elem);
            elem_sd_ind[elem] = subdomain_ind;
            visited[elem] = true;
            // printf("start subdomain %d, with elem %d\n", subdomain_ind, elem);

            // fill out the rest of the elements in this subdomain
            while (sd_elems.size() < target_sd_size) {
                // printf("in while loop 2\n");
                // build a unique element frontier (of element neighbors)
                std::unordered_set<int> frontier_set;

                for (auto _elem : sd_elems) {
                    // printf("_elem %d\n", _elem);
                    for (int jp = ee_ptr[_elem]; jp < ee_ptr[_elem + 1]; jp++) {
                        int neighbor_elem = ee_cols[jp];
                        if (visited[neighbor_elem]) continue;
                        frontier_set.insert(neighbor_elem);
                    }
                }

                // convert to vector
                std::vector<int> frontier(frontier_set.begin(), frontier_set.end());
                // printf("subdomain %d frontier: ", subdomain_ind);
                // printVec<int>(frontier.size(), frontier.data());

                if (frontier.size() == 0) break;

                // rank / sort the frontier by # subdomain corners (not same as vertices)
                // a subdomain corner is a node that has only one element in the subdomain
                int nfrontier = frontier.size();

                // total # corners in the proposed subdomain if this candidate element is added
                int *candidate_corner_count = new int[nfrontier];
                memset(candidate_corner_count, 0, nfrontier * sizeof(int));

                // # corners introduced by this candidate frontier element
                int *candidate_corner_delta = new int[nfrontier];
                memset(candidate_corner_delta, 0, nfrontier * sizeof(int));

                int ii = 0;
                for (auto frontier_elem : frontier) {
                    std::vector<int> proposed_sd_elems;
                    proposed_sd_elems.push_back(frontier_elem);
                    for (auto _elem : sd_elems) {
                        proposed_sd_elems.push_back(_elem);
                    }

                    // get all nodes in the proposed subdomain unique
                    std::unordered_set<int> proposed_sd_nodes;
                    for (auto sd_elem : proposed_sd_elems) {
                        int *local_elem_conn = &elem_conn[nodes_per_elem * sd_elem];
                        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                            int gnode = local_elem_conn[lnode];
                            proposed_sd_nodes.insert(gnode);
                        }
                    }

                    // now determine how many of the proposed_sd_nodes are corners
                    int total_corners = 0;
                    int added_corners = 0;  // number of corners contributed by frontier_elem

                    for (auto gnode : proposed_sd_nodes) {
                        int nelems_in_subdomain = 0;
                        bool candidate_elem_contains_node = false;

                        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                            int jelem = ne_cols[jp];

                            if (jelem == frontier_elem) {
                                candidate_elem_contains_node = true;
                            }

                            for (auto _elem : proposed_sd_elems) {
                                if (jelem == _elem) {
                                    nelems_in_subdomain++;
                                    break;
                                }
                            }
                        }

                        if (nelems_in_subdomain == 1) {
                            total_corners++;
                            if (candidate_elem_contains_node) {
                                added_corners++;
                            }
                        }
                    }

                    candidate_corner_count[ii] = total_corners;
                    candidate_corner_delta[ii] = added_corners;
                    ii++;
                }
                // end of computing frontier candidate scores (# corners)

                // sort by candidate corner count
                // pair each elem with its score
                std::vector<std::tuple<int, int, int>> frontier_candidates;
                frontier_candidates.reserve(nfrontier);

                for (int i = 0; i < nfrontier; i++) {
                    frontier_candidates.emplace_back(frontier[i], candidate_corner_count[i],
                                                     candidate_corner_delta[i]);
                }

                // sort by score (smaller first)
                std::sort(
                    frontier_candidates.begin(), frontier_candidates.end(),
                    [](const std::tuple<int, int, int> &a, const std::tuple<int, int, int> &b) {
                        return std::get<1>(a) < std::get<1>(b);
                    });

                // write back sorted frontier
                for (int i = 0; i < nfrontier; i++) {
                    frontier[i] = std::get<0>(frontier_candidates[i]);
                    candidate_corner_count[i] = std::get<1>(frontier_candidates[i]);
                    candidate_corner_delta[i] = std::get<2>(frontier_candidates[i]);
                }

                // add all frontier elems in the right order to subdomain
                bool added_any = false;
                for (int i = 0; i < nfrontier; i++) {
                    int frontier_elem = frontier[i];
                    if (sd_elems.size() >= target_sd_size) break;
                    if (visited[frontier_elem]) continue;

                    // this element would introduce too many new corners
                    if (candidate_corner_delta[i] > 2) continue;

                    // printf("\tnfrontier %d, subdomain %d, adding elem %d\n", nfrontier,
                    //        subdomain_ind, frontier_elem);
                    sd_elems.push_back(frontier_elem);
                    elem_sd_ind[frontier_elem] = subdomain_ind;
                    visited[frontier_elem] = true;
                    added_any = true;
                }

                if (!added_any) break;

                delete[] candidate_corner_count;
                delete[] candidate_corner_delta;
            }
            // end of while loop to build the subdomain

            // printf("done with subdomain %d, #elems = %d == target_sd_size %d\n", subdomain_ind,
            //        sd_elems.size(), target_sd_size);
            // printf("\tsd_elems: ");
            // printVec<int>(sd_elems.size(), sd_elems.data());
            subdomain_ind++;

            // check if all visited
            all_visited = true;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (!visited[ielem]) {
                    all_visited = false;
                    break;
                }
            }
        }
        num_subdomains = subdomain_ind;
        int n_subdomain = num_subdomains;

        printf("initial elem_sd_ind: ");
        printVec<int>(num_elements, elem_sd_ind);

        auto h_vars0 = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars0, elem_sd_ind,
                                                            "subdomains", "out/elem_sd_ind0.vtk");

        // Phase 2 : merge very small subdomains if it reduces total vertices
        // printf("begin phase 2\n");
        int *subdomain_cts = new int[n_subdomain];
        memset(subdomain_cts, 0, n_subdomain * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            subdomain_ind = elem_sd_ind[ielem];
            subdomain_cts[subdomain_ind]++;
        }
        int *subdomain_ptr = new int[n_subdomain + 1];
        memset(subdomain_ptr, 0, (n_subdomain + 1) * sizeof(int));
        for (int isd = 0; isd < n_subdomain; isd++) {
            subdomain_ptr[isd + 1] = subdomain_ptr[isd] + subdomain_cts[isd];
        }
        memset(subdomain_cts, 0, n_subdomain * sizeof(int));
        int *subdomain_cols = new int[num_elements];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            int offset = subdomain_ptr[isd] + subdomain_cts[isd];
            subdomain_cols[offset] = ielem;
            subdomain_cts[isd]++;
        }
        for (int isd = 0; isd < n_subdomain; isd++) {
            int nelems_sd = subdomain_cts[isd];
            if (nelems_sd < target_sd_size) {
                // find adjacent subdomains
                std::vector<int> sd_elems;
                for (int jp = subdomain_ptr[isd]; jp < subdomain_ptr[isd + 1]; jp++) {
                    int ielem = subdomain_cols[jp];
                    sd_elems.push_back(ielem);
                }
                // printf("sd[%d]-elems: ", isd);
                // printVec<int>(sd_elems.size(), sd_elems.data());
                std::unordered_set<int> adj_subdomains_set;
                for (auto _elem : sd_elems) {
                    // adjacent elems to find adjacent subdomains
                    for (int jp = ee_ptr[_elem]; jp < ee_ptr[_elem + 1]; jp++) {
                        int jelem = ee_cols[jp];
                        int jsd = elem_sd_ind[jelem];
                        if (jsd == isd) continue;
                        adj_subdomains_set.insert(jsd);
                    }
                }

                std::vector<int> adj_subdomains(adj_subdomains_set.begin(),
                                                adj_subdomains_set.end());
                // printf("adj subdomains[%d]: ", isd);
                // printVec<int>(adj_subdomains.size(), adj_subdomains.data());

                // TODO : combine with nearest subdomain to get min vertices
                // for now just combine with the nearest subdomain
                int jsd = adj_subdomains[0];
                for (auto _elem : sd_elems) {
                    elem_sd_ind[_elem] = jsd;
                }
            }
        }
        // printf("done with phase 2\n");

        // now get unique list of subdomains (and adjust isd to new max)
        std::unordered_set<int> sd_set;
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            sd_set.insert(isd);
        }
        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        int *sd_imap = new int[n_subdomain];
        for (int isd = 0; isd < n_subdomain; isd++) {
            sd_imap[isd] = -1;
        }

        // map old subdomain id -> compact new id
        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];
            sd_imap[old_isd] = new_isd;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int old_isd = elem_sd_ind[ielem];
            elem_sd_ind[ielem] = sd_imap[old_isd];
        }

        num_subdomains = sd_vec.size();
        delete[] sd_imap;
        auto h_vars1 = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars1, elem_sd_ind,
                                                            "subdomains", "out/elem_sd_ind1.vtk");

        // classify nodes, etc.
        // -----------------------------------------
        // node -> subdomain incidence
        // -----------------------------------------
        // printf("node -> subdomain incidence\n");
        node_elem_nnz = 0;
        node_elem_rowp = new int[num_nodes + 1];
        node_elem_ct = new int[num_nodes];
        std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
        std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        int *temp_node_elem = new int[num_nodes];
        node_sd_cols = new int[node_elem_nnz];
        std::memset(temp_node_elem, 0, num_nodes * sizeof(int));
        std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int subdomain_ind = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;

        // -----------------------------------------
        // classify nodes
        // -----------------------------------------
        // printf("classify nodes\n");
        node_class_ind = new int[num_nodes];
        std::memset(node_class_ind, 0, num_nodes * sizeof(int));
        node_nsd = new int[num_nodes];
        I_nnodes = 0, IE_nnodes = 0, IEV_nnodes = 0;
        Vc_nnodes = 0, V_nnodes = 0, lam_nnodes = 0;

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int nsd = static_cast<int>(node_sds.size());
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;
                I_nnodes++, IE_nnodes++, IEV_nnodes++;
            } else if (nsd == 2) {
                node_class_ind[inode] = EDGE;
                lam_nnodes++;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = VERTEX;
                Vc_nnodes++;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }

        // printf("before comp_node_nsd / comp_node_class VTK\n");
        T *h_soln = new T[num_nodes * block_dim];
        memset(h_soln, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_soln[6 * i] = node_class_ind[i];
        }
        auto h_soln_debug = HostVec<T>(num_nodes * block_dim, h_soln);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_soln_debug,
                                               "out/comp_node_class_ind.vtk");

        // get MAX vertices per subdomain
        // -----------------------------------------

        int *nvertex = new int[num_subdomains];
        MAX_NUM_VERTEX_PER_SUBDOMAIN = 0;
        memset(nvertex, 0, num_subdomains * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int node_class = node_class_ind[gnode];
                if (node_class == VERTEX) {
                    nvertex[isd]++;
                }
            }
        }

        for (int isd = 0; isd < num_subdomains; isd++) {
            int sd_nvertex = nvertex[isd];
            MAX_NUM_VERTEX_PER_SUBDOMAIN = std::max(MAX_NUM_VERTEX_PER_SUBDOMAIN, sd_nvertex);
        }

        printf("MAX_NUM_VERTEX_PER_SUBDOMAIN %d\n", MAX_NUM_VERTEX_PER_SUBDOMAIN);
        // // MAX_NUM_VERTEX_PER_SUBDOMAIN = 11;
        // MAX_NUM_VERTEX_PER_SUBDOMAIN = 6;

        // -----------------------------------------
        // build duplicated IEV nodal layout
        // -----------------------------------------
        // printf("build duplicated IEV nodal layout\n");
        IEV_sd_ptr = new int[num_subdomains + 1];
        IEV_sd_ind = new int[IEV_nnodes];
        IEV_nodes = new int[IEV_nnodes];

        std::memset(IEV_sd_ptr, 0, (num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[num_nodes];

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::memset(temp_completion, 0, num_nodes * sizeof(int));
            IEV_sd_ptr[i_subdomain + 1] = IEV_sd_ptr[i_subdomain];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != i_subdomain) continue;

                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    if (temp_completion[gnode]) continue;

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = i_subdomain;
                    IEV_sd_ptr[i_subdomain + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;

        // printf("IEV_sd_ptr: ");
        // printVec<int>(num_subdomains + 1, IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_nnodes, IEV_sd_ind);

        // -----------------------------------------
        // build IEV element connectivity
        // -----------------------------------------
        // printf("build IEV element connectivity\n");
        IEV_elem_conn = new int[num_elements * nodes_per_elem];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int i_subdomain = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[i_subdomain]; jp < IEV_sd_ptr[i_subdomain + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                }
                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }

        // printf("IEV_ind %d\n", IEV_ind);
        // printf("IEV_nodes %d: ", IEV_nnodes);
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("Fine BDDC IEV_conn: ");
        // printVec<int>(nodes_per_elem * num_elements, IEV_elem_conn);

        // for (int iev = 0; iev < IEV_nnodes; iev++) {
        //     int isd = IEV_sd_ind[iev];
        //     int gnode = IEV_nodes[iev];
        //     int iclass = node_class_ind[gnode];
        //     printf("iev %d, isd %d, gnode %d, class %d\n", iev, isd, gnode, iclass);
        // }

        // -----------------------------------------
        // IE and I nodal lists
        // -----------------------------------------
        printf("IE and I nodal lists, IEV_nnodes(%d) and IE_nnodes(%d) and I_nnodes(%d)\n",
               IEV_nnodes, IE_nnodes, I_nnodes);
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];
        std::memset(IE_nodes, 0, IE_nnodes * sizeof(int));
        std::memset(I_nodes, 0, I_nnodes * sizeof(int));
        IE_interior = new bool[IE_nnodes];
        IE_general_edge = new bool[IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                I_nodes[I_ind++] = gnode;
            }
        }
        d_IE_interior = HostVec<bool>(IE_nnodes, IE_interior).createDeviceVec().getPtr();
        d_IE_general_edge = HostVec<bool>(IE_nnodes, IE_general_edge).createDeviceVec().getPtr();
        d_IE_nodes = HostVec<int>(IE_nnodes, IE_nodes).createDeviceVec().getPtr();

        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("I_nodes %d: ", I_nnodes);
        // printVec<int>(I_nnodes, I_nodes);

        // -----------------------------------------
        // build IEV sparsity from duplicated connectivity
        // -----------------------------------------
        // printf("build IEV sparsity from duplicated connectivity\n");
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);
        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;
        IEV_rows = new int[IEV_nnzb];
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            for (int jp = IEV_rowp[inode]; jp < IEV_rowp[inode + 1]; jp++) {
                IEV_rows[jp] = inode;
            }
        }
        // printf("IEV_rowp with nnzb %d: ", IEV_nnzb);
        // printVec<int>(IEV_nnodes + 1, IEV_rowp);
        // printf("IEV_cols: ");
        // printVec<int>(IEV_nnzb, IEV_cols);

        // -----------------------------------------
        // reduced rowp arrays
        // -----------------------------------------
        // printf("reduced rowp arrays\n");
        IE_rowp = new int[IE_nnodes + 1];
        I_rowp = new int[I_nnodes + 1];
        std::memset(IE_rowp, 0, (IE_nnodes + 1) * sizeof(int));
        std::memset(I_rowp, 0, (I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) I_rowp[I_row + 1] = I_rowp[I_row];
            if (typeIE_row) IE_rowp[IE_row + 1] = IE_rowp[IE_row];

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        I_nnzb = I_rowp[I_nnodes];
        IE_nnzb = IE_rowp[IE_nnodes];

        // printf("I_rowp nnzb %d: ", I_nnzb);
        // printVec<int>(I_nnodes + 1, I_rowp);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_rowp nnzb %d: ", IE_nnzb);
        // printVec<int>(IE_nnodes + 1, IE_rowp);

        IE_rows = new int[IE_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }

        // -----------------------------------------
        // IEV -> IE map
        // -----------------------------------------
        // printf("IEV -> IE map\n");
        IEVtoIE_map = new int[IEV_nnodes];
        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        IEVtoIE_imap = new int[IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IEVtoIE_imap[IE_ind] = inode;
                IEVtoIE_map[inode] = IE_ind++;
            }
        }

        // printf("IEVtoIE_map: ");
        // printVec<int>(IEV_nnodes, IEVtoIE_map);

        // put on device
        d_IEVtoIE_imap = HostVec<int>(IE_nnodes, IEVtoIE_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV -> I map
        // -----------------------------------------
        // printf("IEV -> I map\n");
        IEVtoI_map = new int[IEV_nnodes];
        IEVtoI_imap = new int[I_nnodes];
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IEVtoI_imap[I_ind] = inode;
                IEVtoI_map[inode] = I_ind++;
            }
        }

        // printf("IEVtoI_map: ");
        // printVec<int>(IEV_nnodes, IEVtoI_map);

        // put on device
        d_IEVtoI_imap = HostVec<int>(I_nnodes, IEVtoI_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // reduced column arrays
        // -----------------------------------------
        // printf("reduced column arrays\n");
        IE_cols = new int[IE_nnzb];
        I_cols = new int[I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    I_cols[I_ind++] = IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    IE_cols[IE_ind++] = IEVtoIE_map[col];
                }
            }
        }

        // printf("I_cols: ");
        // printVec<int>(I_nnzb, I_cols);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();

        // -----------------------------------------
        // IEV -> Vc map (coarse non-repeated vertices)
        // -----------------------------------------
        // printf("IEV -> Vc map (coarse non-repeated vertices)\n");

        // make a list of the reduced Vc_ind (takes global node and figures out it's Vc_ind)
        std::unordered_set<int> Vc_nodeset;
        for (int i = 0; i < IEV_nnodes; i++) {
            int gnode = IEV_nodes[i];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());
        // printf("Vc_nodes %d: ", Vc_nodes_vec.size());
        // printVec<int>(Vc_nodes_vec.size(), Vc_nodes_vec.data());

        int *Vc_inodes = new int[num_nodes];  // takes Vc global node => Vc red node
        memset(Vc_inodes, -1, num_nodes * sizeof(int));
        for (int i = 0; i < Vc_nodes_vec.size(); i++) {
            int j = Vc_nodes_vec[i];
            Vc_inodes[j] = i;
        }

        // printf("Vc_inodes: ");
        // printVec<int>(num_nodes, Vc_inodes);

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();

        // printf("Vc_nodes: ");
        // printVec<int>(Vc_nnodes, Vc_nodes);

        // set VcToV_imap and IEVtoV_imap now
        VctoV_imap = new int[V_nnodes];
        std::memset(VctoV_imap, -1, V_nnodes * sizeof(int));
        IEVtoV_imap = new int[V_nnodes];
        std::memset(IEVtoV_imap, -1, V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                VctoV_imap[V_ind] = Vc_ind;
                IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }

        // printf("IEVtoV_imap %d: ", V_nnodes);
        // printVec<int>(V_nnodes, IEVtoV_imap);

        // put on device
        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();

        // Build all remaining maps/permutations:
        //  - jump operator ownership/sign maps

        // printf("compute jump operators\n");
        _compute_jump_operators(false);
        allocate_workspace();

        // ---------------------------------------------------
        // Save ORIGINAL nofill sparsity for IE and I
        // before fill-in / AMD reordering
        // ---------------------------------------------------
        IE_bsr_data = BsrData(IE_nnodes, block_dim, IE_nnzb, IE_rowp, IE_cols);
        IE_bsr_data.rows = IE_rows;
        IE_nofill_nnzb = IE_nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        // sparsity before the permutation
        // printf("\nIE SPARSITY BEFORE PERMUTATION\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("(");
        //     int grow = IE_nodes[IE_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
        //         int j = IE_cols[jp];
        //         int gcol = IE_nodes[IE_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");

        // printf("pre-perm IE_rowp: ");
        // printVec<int>(IE_nnodes + 1, IE_rowp);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("\n");

        I_bsr_data = BsrData(I_nnodes, block_dim, I_nnzb, I_rowp, I_cols);
        I_bsr_data.rows = I_rows;
        I_nofill_nnzb = I_nnzb;

        // DEBUG
        // printf("num_subdomains %d\n", num_subdomains);
        // printf("I_nnodes %d\n", I_nnodes);
        // printf("IE_nnodes %d\n", IE_nnodes);
        // printf("IEV_nnodes %d\n", IEV_nnodes);
        // printf("Vc_nnodes %d\n", Vc_nnodes);
        // printf("V_nnodes %d\n", V_nnodes);
        // printf("lam_nnodes %d\n", lam_nnodes);
        // printf("elem_sd_ind: ");
        // printVec<int>(num_elements, elem_sd_ind);
        // printf("node_class_ind: ");
        // printVec<int>(num_nodes, node_class_ind);
        // printf("node_nsd: ");
        // printVec<int>(num_nodes, node_nsd);
        // printf("IEV_sd_ptr: ");
        // printVec<int>(num_subdomains + 1, IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_nnodes, IEV_sd_ind);
        // printf("IEV_nodes: ");
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("IEV_elem_conn: ");
        // printVec<int>(4 * num_elements, IEV_elem_conn);
    }

    void setup_structured_subdomains(int nxe_, int nye_, int nxs_, int nys_,
                                     bool close_hoop = false, bool track_dirichlet = false) {
        clear_structured_host_data();

        nxe = nxe_;
        nye = nye_;
        nxs = nxs_;
        nys = nys_;

        int order = Basis::order;
        nx = order * nxe + 1;
        ny = close_hoop ? order * nye : (order * nye + 1);

        num_subdomains = nxs * nys;

        const int nxse = (nxe + nxs - 1) / nxs;
        const int nyse = (nye + nys - 1) / nys;

        // -----------------------------------------
        // classify elements into subdomains
        // -----------------------------------------
        elem_sd_ind = new int[num_elements];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int ixe = ielem % nxe;
            int iye = ielem / nxe;
            int ixs = ixe / nxse;
            int iys = iye / nyse;

            if (ixs >= nxs) ixs = nxs - 1;
            if (iys >= nys) iys = nys - 1;

            elem_sd_ind[ielem] = ixs + iys * nxs;
        }

        // -----------------------------------------
        // node -> subdomain incidence
        // -----------------------------------------
        node_elem_nnz = 0;
        node_elem_rowp = new int[num_nodes + 1];
        node_elem_ct = new int[num_nodes];
        std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
        std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        int *temp_node_elem = new int[num_nodes];
        node_sd_cols = new int[node_elem_nnz];
        std::memset(temp_node_elem, 0, num_nodes * sizeof(int));
        std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int subdomain_ind = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;

        // -----------------------------------------
        // classify nodes
        // -----------------------------------------
        node_class_ind = new int[num_nodes];
        std::memset(node_class_ind, 0, num_nodes * sizeof(int));
        node_nsd = new int[num_nodes];
        std::memset(node_nsd, 0, num_nodes * sizeof(int));

        nnodes_interior = 0;
        nnodes_edge = 0;
        nnodes_vertex = 0;
        nnodes_dirichlet_edge = 0;

        printf(
            "WARNING: for unstructured meshes, this node classification may fail at junctions.\n");

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int ix = inode % nx;
            int iy = inode / nx;
            bool on_x = (ix == 0) || (ix == nx - 1);
            bool on_y = ((iy == 0) || (iy == ny - 1)) && !close_hoop;
            bool on_bndry = on_x || on_y;

            int nsd = static_cast<int>(node_sds.size());
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;
                nnodes_interior++;
            } else if (nsd == 2) {
                if (on_bndry && track_dirichlet) {
                    node_class_ind[inode] = DIRICHLET_EDGE;
                    nnodes_dirichlet_edge++;
                } else {
                    node_class_ind[inode] = EDGE;
                    nnodes_edge++;
                }
            } else {
                node_class_ind[inode] = VERTEX;
                nnodes_vertex++;
            }
        }

        // TODO:
        // This count assumes structured duplication counts:
        //   interior            -> 1 copy
        //   dirichlet-edge      -> 2 copies
        //   edge                -> 2 copies
        //   vertex              -> 4 copies
        // Revisit for more general cases.
        I_nnodes = nnodes_interior + 2 * nnodes_dirichlet_edge;
        IE_nnodes = I_nnodes + 2 * nnodes_edge;
        IEV_nnodes = IE_nnodes + 4 * nnodes_vertex;
        Vc_nnodes = nnodes_vertex;     // vertex non-repeated (here for coarse system)
        V_nnodes = 4 * nnodes_vertex;  // vertex repeated
        lam_nnodes = nnodes_edge;      // FETI lagrange multipliers

        // printf("nsub=%d, I=%d, IE=%d, IEV=%d, Vc=%d, V=%d, lam=%d\n", num_subdomains, I_nnodes,
        //        IE_nnodes, IEV_nnodes, Vc_nnodes, V_nnodes, lam_nnodes);

        // -----------------------------------------
        // build duplicated IEV nodal layout
        // -----------------------------------------
        IEV_sd_ptr = new int[num_subdomains + 1];
        IEV_sd_ind = new int[IEV_nnodes];
        IEV_nodes = new int[IEV_nnodes];

        std::memset(IEV_sd_ptr, 0, (num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[num_nodes];

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::memset(temp_completion, 0, num_nodes * sizeof(int));
            IEV_sd_ptr[i_subdomain + 1] = IEV_sd_ptr[i_subdomain];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != i_subdomain) continue;

                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    if (temp_completion[gnode]) continue;

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = i_subdomain;
                    IEV_sd_ptr[i_subdomain + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;

        // printf("IEV_nodes: ");
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("IEV_sd_ptr: ");
        // printVec<int>(num_subdomains + 1, IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_nnodes, IEV_sd_ind);

        // -----------------------------------------
        // build IEV element connectivity
        // -----------------------------------------
        IEV_elem_conn = new int[num_elements * nodes_per_elem];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int i_subdomain = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[i_subdomain]; jp < IEV_sd_ptr[i_subdomain + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                }
                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }

        // printf("IEV_ind %d\n", IEV_ind);
        // printf("IEV_nodes %d: ", IEV_nnodes);
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("Fine BDDC IEV_conn: ");
        // printVec<int>(nodes_per_elem * num_elements, IEV_elem_conn);
        // printf("elem_sd_ind: ");
        // printVec<int>(num_elements, elem_sd_ind);

        // for (int iev = 0; iev < IEV_nnodes; iev++) {
        //     int isd = IEV_sd_ind[iev];
        //     int gnode = IEV_nodes[iev];
        //     int iclass = node_class_ind[gnode];
        //     printf("iev %d, isd %d, gnode %d, class %d\n", iev, isd, gnode, iclass);
        // }

        // -----------------------------------------
        // IE and I nodal lists
        // -----------------------------------------
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];
        std::memset(IE_nodes, 0, IE_nnodes * sizeof(int));
        std::memset(I_nodes, 0, I_nnodes * sizeof(int));
        IE_interior = new bool[IE_nnodes];
        IE_general_edge = new bool[IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                I_nodes[I_ind++] = gnode;
            }
        }
        d_IE_interior = HostVec<bool>(IE_nnodes, IE_interior).createDeviceVec().getPtr();
        d_IE_general_edge = HostVec<bool>(IE_nnodes, IE_general_edge).createDeviceVec().getPtr();
        d_IE_nodes = HostVec<int>(IE_nnodes, IE_nodes).createDeviceVec().getPtr();

        // printf("IE_interior %d: ", IE_nnodes);
        // printVec<bool>(IE_nnodes, IE_interior);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("I_nodes %d: ", I_nnodes);
        // printVec<int>(I_nnodes, I_nodes);

        // -----------------------------------------
        // build IEV sparsity from duplicated connectivity
        // -----------------------------------------
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);
        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;
        IEV_rows = new int[IEV_nnzb];
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            for (int jp = IEV_rowp[inode]; jp < IEV_rowp[inode + 1]; jp++) {
                IEV_rows[jp] = inode;
            }
        }
        // printf("IEV_rowp with nnzb %d: ", IEV_nnzb);
        // printVec<int>(IEV_nnodes + 1, IEV_rowp);
        // printf("IEV_cols: ");
        // printVec<int>(IEV_nnzb, IEV_cols);

        // -----------------------------------------
        // reduced rowp arrays
        // -----------------------------------------
        IE_rowp = new int[IE_nnodes + 1];
        I_rowp = new int[I_nnodes + 1];
        std::memset(IE_rowp, 0, (IE_nnodes + 1) * sizeof(int));
        std::memset(I_rowp, 0, (I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) I_rowp[I_row + 1] = I_rowp[I_row];
            if (typeIE_row) IE_rowp[IE_row + 1] = IE_rowp[IE_row];

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        I_nnzb = I_rowp[I_nnodes];
        IE_nnzb = IE_rowp[IE_nnodes];

        // printf("I_rowp nnzb %d: ", I_nnzb);
        // printVec<int>(I_nnodes + 1, I_rowp);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_rowp nnzb %d: ", IE_nnzb);
        // printVec<int>(IE_nnodes + 1, IE_rowp);

        IE_rows = new int[IE_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }

        // -----------------------------------------
        // IEV -> IE map
        // -----------------------------------------
        IEVtoIE_map = new int[IEV_nnodes];
        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        IEVtoIE_imap = new int[IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IEVtoIE_imap[IE_ind] = inode;
                IEVtoIE_map[inode] = IE_ind++;
            }
        }

        // printf("IEVtoIE_map: ");
        // printVec<int>(IEV_nnodes, IEVtoIE_map);

        // put on device
        d_IEVtoIE_imap = HostVec<int>(IE_nnodes, IEVtoIE_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV -> I map
        // -----------------------------------------
        IEVtoI_map = new int[IEV_nnodes];
        IEVtoI_imap = new int[I_nnodes];
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IEVtoI_imap[I_ind] = inode;
                IEVtoI_map[inode] = I_ind++;
            }
        }

        // printf("IEVtoI_map: ");
        // printVec<int>(IEV_nnodes, IEVtoI_map);

        // put on device
        d_IEVtoI_imap = HostVec<int>(I_nnodes, IEVtoI_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // reduced column arrays
        // -----------------------------------------
        IE_cols = new int[IE_nnzb];
        I_cols = new int[I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    I_cols[I_ind++] = IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    IE_cols[IE_ind++] = IEVtoIE_map[col];
                }
            }
        }

        // printf("I_cols: ");
        // printVec<int>(I_nnzb, I_cols);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();

        // -----------------------------------------
        // IEV -> Vc map (coarse non-repeated vertices)
        // -----------------------------------------

        // make a list of the reduced Vc_ind (takes global node and figures out it's Vc_ind)
        std::unordered_set<int> Vc_nodeset;
        for (int i = 0; i < IEV_nnodes; i++) {
            int gnode = IEV_nodes[i];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());
        // printf("Vc_nodes %d: ", Vc_nodes_vec.size());
        // printVec<int>(Vc_nodes_vec.size(), Vc_nodes_vec.data());

        int *Vc_inodes = new int[num_nodes];  // takes Vc global node => Vc red node
        memset(Vc_inodes, -1, num_nodes * sizeof(int));
        for (int i = 0; i < Vc_nodes_vec.size(); i++) {
            int j = Vc_nodes_vec[i];
            Vc_inodes[j] = i;
        }

        // printf("Vc_inodes: ");
        // printVec<int>(num_nodes, Vc_inodes);

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();

        // set VcToV_imap and IEVtoV_imap now
        VctoV_imap = new int[V_nnodes];
        std::memset(VctoV_imap, -1, V_nnodes * sizeof(int));
        IEVtoV_imap = new int[V_nnodes];
        std::memset(IEVtoV_imap, -1, V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                VctoV_imap[V_ind] = Vc_ind;
                IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }

        // printf("IEVtoV_imap %d: ", V_nnodes);
        // printVec<int>(V_nnodes, IEVtoV_imap);

        // put on device
        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();

        // Build all remaining maps/permutations:
        //  - jump operator ownership/sign maps

        _compute_jump_operators();
        allocate_workspace();

        // ---------------------------------------------------
        // Save ORIGINAL nofill sparsity for IE and I
        // before fill-in / AMD reordering
        // ---------------------------------------------------
        IE_bsr_data = BsrData(IE_nnodes, block_dim, IE_nnzb, IE_rowp, IE_cols);
        IE_bsr_data.rows = IE_rows;
        IE_nofill_nnzb = IE_nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        // sparsity before the permutation
        // printf("\nIE SPARSITY BEFORE PERMUTATION\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("(");
        //     int grow = IE_nodes[IE_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
        //         int j = IE_cols[jp];
        //         int gcol = IE_nodes[IE_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");

        // printf("pre-perm IE_rowp: ");
        // printVec<int>(IE_nnodes + 1, IE_rowp);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("\n");

        I_bsr_data = BsrData(I_nnodes, block_dim, I_nnzb, I_rowp, I_cols);
        I_bsr_data.rows = I_rows;
        I_nofill_nnzb = I_nnzb;
    }

    void setup_tacs_component_subdomains(int nxse_, int nyse_, int MOD_WRAPAROUND = -1,
                                         T wrap_frac = 1.0, bool compute_jump = true,
                                         bool track_dirichlet = false) {
        // bool my_debug = true;
        bool my_debug = false;
        if (my_debug) {
            _setup_tacs_component_subdomains_debug(nxse_, nyse_, MOD_WRAPAROUND, wrap_frac,
                                                   compute_jump, track_dirichlet);
        } else {
            _setup_tacs_component_subdomains_nodebug(nxse_, nyse_, MOD_WRAPAROUND, wrap_frac,
                                                     compute_jump, track_dirichlet);
        }
    }

    void _setup_tacs_component_subdomains_debug(int nxse_, int nyse_, int MOD_WRAPAROUND = -1,
                                                T wrap_frac = 1.0, bool compute_jump = true,
                                                bool track_dirichlet = false) {
        bool print_debug = true;
        auto dbg = [&](const char *msg) {
            if (print_debug) {
                printf("[SD-DBG] %s\n", msg);
                fflush(stdout);
            }
        };

        auto dbg_val = [&](const char *name, int val) {
            if (print_debug) {
                printf("[SD-DBG] %s = %d\n", name, val);
                fflush(stdout);
            }
        };

        auto dbg_val2 = [&](const char *name, double val) {
            if (print_debug) {
                printf("[SD-DBG] %s = %.6e\n", name, val);
                fflush(stdout);
            }
        };

        auto print_int_sample = [&](const char *name, const int *arr, int n, int max_print = 8) {
            if (!print_debug) return;
            printf("[SD-DBG] %s sample (n=%d): ", name, n);
            int m = (n < max_print ? n : max_print);
            for (int i = 0; i < m; i++) {
                printf("%d ", arr[i]);
            }
            if (n > m) printf("...");
            printf("\n");
            fflush(stdout);
        };

        auto print_bool_sample = [&](const char *name, const bool *arr, int n, int max_print = 8) {
            if (!print_debug) return;
            printf("[SD-DBG] %s sample (n=%d): ", name, n);
            int m = (n < max_print ? n : max_print);
            for (int i = 0; i < m; i++) {
                printf("%d ", (int)arr[i]);
            }
            if (n > m) printf("...");
            printf("\n");
            fflush(stdout);
        };

        auto print_t_sample = [&](const char *name, const T *arr, int n, int max_print = 8) {
            if (!print_debug) return;
            printf("[SD-DBG] %s sample (n=%d): ", name, n);
            int m = (n < max_print ? n : max_print);
            for (int i = 0; i < m; i++) {
                printf("%.4e ", (double)arr[i]);
            }
            if (n > m) printf("...");
            printf("\n");
            fflush(stdout);
        };

        dbg("ENTER setup_tacs_component_subdomains_debug");
        dbg_val("nxse_", nxse_);
        dbg_val("nyse_", nyse_);
        dbg_val("MOD_WRAPAROUND", MOD_WRAPAROUND);
        dbg_val2("wrap_frac", wrap_frac);
        dbg_val("compute_jump", (int)compute_jump);
        dbg_val("num_nodes", num_nodes);
        dbg_val("num_elements", num_elements);
        dbg_val("nodes_per_elem", nodes_per_elem);
        dbg_val("block_dim", block_dim);

        clear_structured_host_data();
        dbg("after clear_structured_host_data");

        int *nodal_num_wing_comps, *node_wing_geom_ind;
        dbg("before get_nodal_geom_indices");
        WingboxMultiColoring<ShellAssembler>::get_nodal_geom_indices(
            assembler, nodal_num_wing_comps, node_wing_geom_ind);
        dbg("after get_nodal_geom_indices");
        print_int_sample("node_wing_geom_ind", node_wing_geom_ind, num_nodes);
        print_int_sample("nodal_num_wing_comps", nodal_num_wing_comps, num_nodes);

        int *node_nelems = new int[num_nodes];
        memset(node_nelems, 0, num_nodes * sizeof(int));
        dbg("before node_nelems accumulation");
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                if (gnode < 0 || gnode >= num_nodes) {
                    printf("[SD-DBG] ERROR bad gnode=%d at ielem=%d lnode=%d\n", gnode, ielem,
                           lnode);
                    fflush(stdout);
                }
                node_nelems[gnode]++;
            }
        }
        dbg("after node_nelems accumulation");
        print_int_sample("node_nelems", node_nelems, num_nodes);

        dbg("before assembler.getBCs()");
        auto d_bcs = assembler.getBCs();
        int n_orig_bcs = d_bcs.getSize();
        int *h_bcs = d_bcs.createHostVec().getPtr();
        dbg("after assembler.getBCs()");
        dbg_val("n_orig_bcs", n_orig_bcs);
        print_int_sample("h_bcs", h_bcs, n_orig_bcs);

        bool *dirichlet_ind = new bool[num_nodes];
        memset(dirichlet_ind, false, num_nodes * sizeof(bool));
        dbg("before dirichlet_ind fill");
        for (int ibc = 0; ibc < n_orig_bcs; ibc++) {
            int bc_node = h_bcs[ibc] / block_dim;
            if (bc_node < 0 || bc_node >= num_nodes) {
                printf("[SD-DBG] ERROR bad bc_node=%d from h_bcs[%d]=%d\n", bc_node, ibc,
                       h_bcs[ibc]);
                fflush(stdout);
            }
            dirichlet_ind[bc_node] = true;
        }
        dbg("after dirichlet_ind fill");
        print_bool_sample("dirichlet_ind", dirichlet_ind, num_nodes);

        int *node_bndry_ind = new int[num_nodes];
        dbg("before node_bndry_ind fill");
        for (int inode = 0; inode < num_nodes; inode++) {
            node_bndry_ind[inode] = 0;
            int nelems_attached = node_nelems[inode];
            if (nelems_attached == 2) {
                node_bndry_ind[inode] = 1;
            } else if (nelems_attached == 1) {
                node_bndry_ind[inode] = 2;
            }
        }
        dbg("after node_bndry_ind fill");
        print_int_sample("node_bndry_ind", node_bndry_ind, num_nodes);

        dbg("before write node geom vtk");
        T *h_wgeom_ind = new T[num_nodes * block_dim];
        memset(h_wgeom_ind, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_wgeom_ind[6 * i] = node_wing_geom_ind[i];
        }
        auto h_wgeom_vec = HostVec<T>(num_nodes * block_dim, h_wgeom_ind);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_wgeom_vec,
                                               "out/wing_node_geom_ind.vtk");
        dbg("after write node geom vtk");

        int num_comps = assembler.get_num_components();
        dbg_val("num_comps", num_comps);

        dbg("before getElemComponents/createHostVec");
        int *h_elem_comps = assembler.getElemComponents().createHostVec().getPtr();
        dbg("after getElemComponents/createHostVec");
        print_int_sample("h_elem_comps", h_elem_comps, num_elements);

        dbg("before d_xpts host copy");
        T *h_xpts = d_xpts.createHostVec().getPtr();
        dbg("after d_xpts host copy");
        print_t_sample("h_xpts", h_xpts, 3 * ((num_nodes < 3) ? num_nodes : 3), 9);

        dbg("before element centroid computation");
        T *elem_centroids = new T[3 * num_elements];
        memset(elem_centroids, 0, 3 * num_elements * sizeof(T));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                T *xpt = &h_xpts[3 * gnode];
                elem_centroids[3 * ielem + 0] += xpt[0];
                elem_centroids[3 * ielem + 1] += xpt[1];
                elem_centroids[3 * ielem + 2] += xpt[2];
            }

            elem_centroids[3 * ielem + 0] /= nodes_per_elem;
            elem_centroids[3 * ielem + 1] /= nodes_per_elem;
            elem_centroids[3 * ielem + 2] /= nodes_per_elem;
        }
        dbg("after element centroid computation");
        print_t_sample("elem_centroids", elem_centroids,
                       3 * ((num_elements < 3) ? num_elements : 3), 9);

        elem_sd_ind = new int[num_elements];
        memset(elem_sd_ind, 0, num_elements * sizeof(int));
        int *debug_lelem_sd_ind = new int[num_elements];
        memset(debug_lelem_sd_ind, 0, num_elements * sizeof(int));
        int i_subdomain = 0;

        int *elem2elem_row_cts = new int[num_elements];
        int *comp_num_elems = new int[num_comps];
        memset(elem2elem_row_cts, 0, num_elements * sizeof(int));
        memset(comp_num_elems, 0, num_comps * sizeof(int));

        dbg("before elem2elem_row_cts + comp_num_elems loop");
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int icomp = h_elem_comps[ielem];
            if (icomp < 0 || icomp >= num_comps) {
                printf("[SD-DBG] ERROR bad icomp=%d at ielem=%d\n", icomp, ielem);
                fflush(stdout);
            }
            comp_num_elems[icomp]++;

            bool elem_is_interior = true;
            bool elem_is_edge = true;
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int node_class = node_wing_geom_ind[gnode];
                int bndry_class = node_bndry_ind[gnode];
                if (node_class > 0) elem_is_interior = false;
                if (node_class > 1) elem_is_edge = false;
                if (bndry_class > 0) elem_is_interior = false;
                if (bndry_class > 1) elem_is_edge = false;
                if (bndry_class == 1 && node_class == 1) elem_is_edge = false;
            }

            if (elem_is_interior) {
                elem2elem_row_cts[ielem] = 4;
            } else if (elem_is_edge) {
                elem2elem_row_cts[ielem] = 3;
            } else {
                elem2elem_row_cts[ielem] = 2;
            }
        }
        dbg("after elem2elem_row_cts + comp_num_elems loop");
        print_int_sample("comp_num_elems", comp_num_elems, num_comps);
        print_int_sample("elem2elem_row_cts", elem2elem_row_cts, num_elements);

        int **elem2elem_ielem = new int *[num_comps];
        int **elem2elem_rowp = new int *[num_comps];
        int **elem2elem_cols = new int *[num_comps];
        int *elem2elem_nnz = new int[num_comps];

        for (int icomp = 0; icomp < num_comps; icomp++) {
            elem2elem_ielem[icomp] = nullptr;
            elem2elem_rowp[icomp] = nullptr;
            elem2elem_cols[icomp] = nullptr;
            elem2elem_nnz[icomp] = 0;
        }

        dbg("before component loop");
        for (int icomp = 0; icomp < num_comps; icomp++) {
            if (print_debug) {
                printf("\n[SD-DBG] ===== COMPONENT %d / %d =====\n", icomp, num_comps - 1);
                fflush(stdout);
            }

            int _num_elems = comp_num_elems[icomp];
            dbg_val("_num_elems", _num_elems);

            int *e2e_ielem = new int[_num_elems];
            int *e2e_rowp = new int[_num_elems + 1];
            elem2elem_ielem[icomp] = e2e_ielem;
            elem2elem_rowp[icomp] = e2e_rowp;

            e2e_rowp[0] = 0;

            dbg("before reduced component element list");
            int elem_ct = 0;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int jcomp = h_elem_comps[ielem];
                if (icomp != jcomp) continue;

                e2e_ielem[elem_ct] = ielem;
                e2e_rowp[elem_ct + 1] = e2e_rowp[elem_ct] + elem2elem_row_cts[ielem];
                elem_ct++;
            }
            dbg("after reduced component element list");
            print_int_sample("e2e_ielem", e2e_ielem, _num_elems);
            print_int_sample("e2e_rowp", e2e_rowp, _num_elems + 1);

            if (elem_ct != _num_elems) {
                printf("ERROR: icomp %d expected _num_elems %d but got elem_ct %d\n", icomp,
                       _num_elems, elem_ct);
                fflush(stdout);
                exit(1);
            }

            int nnz = e2e_rowp[_num_elems];
            elem2elem_nnz[icomp] = nnz;
            dbg_val("initial nnz", nnz);

            int *e2e_cols = new int[nnz];
            elem2elem_cols[icomp] = e2e_cols;

            dbg("before e2e_cols fill");
            int ind = 0;
            for (int i2 = 0; i2 < _num_elems * _num_elems; i2++) {
                int i = i2 / _num_elems, j = i2 % _num_elems;
                if (i == j) continue;

                int ielem = e2e_ielem[i];
                int *li_conn = &elem_conn[nodes_per_elem * ielem];
                int jelem = e2e_ielem[j];
                int *lj_conn = &elem_conn[nodes_per_elem * jelem];

                int num_match = 0;
                for (int lnode2 = 0; lnode2 < nodes_per_elem * nodes_per_elem; lnode2++) {
                    int lnodei = lnode2 % nodes_per_elem;
                    int lnodej = lnode2 / nodes_per_elem;
                    int inode = li_conn[lnodei], jnode = lj_conn[lnodej];
                    if (inode == jnode) num_match++;
                }

                if (num_match > 1) {
                    if (ind >= nnz) {
                        printf("ERROR: e2e_cols overflow for icomp %d\n", icomp);
                        fflush(stdout);
                        exit(1);
                    }
                    e2e_cols[ind++] = j;
                }
            }
            dbg("after e2e_cols fill");

            if (ind != nnz) {
                printf("WARNING: icomp %d expected nnz %d but filled %d\n", icomp, nnz, ind);
                fflush(stdout);
                nnz = ind;
                elem2elem_nnz[icomp] = ind;
            }
            print_int_sample("e2e_cols", e2e_cols, nnz);

            dbg("before xpt bounds + xcg");
            T xpt_min[3] = {1e20, 1e20, 1e20};
            T xpt_max[3] = {-1e20, -1e20, -1e20};

            T xcg[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *xpt_elem = &elem_centroids[3 * ielem];

                xcg[0] += xpt_elem[0];
                xcg[1] += xpt_elem[1];
                xcg[2] += xpt_elem[2];

                int *lconn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = lconn[lnode];
                    T *xpt = &h_xpts[3 * gnode];
                    for (int dim = 0; dim < 3; dim++) {
                        xpt_min[dim] = (xpt[dim] < xpt_min[dim]) ? xpt[dim] : xpt_min[dim];
                        xpt_max[dim] = (xpt[dim] > xpt_max[dim]) ? xpt[dim] : xpt_max[dim];
                    }
                }
            }
            xcg[0] /= _num_elems;
            xcg[1] /= _num_elems;
            xcg[2] /= _num_elems;
            dbg("after xpt bounds + xcg");
            print_t_sample("xpt_min", xpt_min, 3, 3);
            print_t_sample("xpt_max", xpt_max, 3, 3);
            print_t_sample("xcg", xcg, 3, 3);

            T xpt_span[3];
            for (int dim = 0; dim < 3; dim++) {
                xpt_span[dim] = xpt_max[dim] - xpt_min[dim];
            }
            print_t_sample("xpt_span", xpt_span, 3, 3);

            auto vec_dot = [](const T *a, const T *b) -> T {
                return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            };

            auto vec_norm = [&](const T *a) -> T { return sqrt(vec_dot(a, a)); };

            auto normalize = [&](T *a) {
                T nrm = vec_norm(a);
                if (nrm > 1e-30) {
                    a[0] /= nrm;
                    a[1] /= nrm;
                    a[2] /= nrm;
                }
            };

            dbg("before geom_normal");
            T geom_normal[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                int *lconn = &elem_conn[nodes_per_elem * ielem];

                T *x0 = &h_xpts[3 * lconn[0]];
                T *x1 = &h_xpts[3 * lconn[1]];
                T *x2 = &h_xpts[3 * lconn[2]];
                T *x3 = &h_xpts[3 * lconn[3]];

                T e01[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
                T e02[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
                T e03[3] = {x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]};

                T n1[3], n2[3];
                A2D::VecCrossCore<T>(e01, e02, n1);
                A2D::VecCrossCore<T>(e02, e03, n2);

                geom_normal[0] += n1[0] + n2[0];
                geom_normal[1] += n1[1] + n2[1];
                geom_normal[2] += n1[2] + n2[2];
            }

            if (vec_norm(geom_normal) < 1e-12) {
                int min_direc = 0;
                T min_span = xpt_span[0];
                for (int dim = 1; dim < 3; dim++) {
                    if (xpt_span[dim] < min_span) {
                        min_span = xpt_span[dim];
                        min_direc = dim;
                    }
                }
                geom_normal[0] = 0.0;
                geom_normal[1] = 0.0;
                geom_normal[2] = 0.0;
                geom_normal[min_direc] = 1.0;
            }
            normalize(geom_normal);
            dbg("after geom_normal");
            print_t_sample("geom_normal", geom_normal, 3, 3);

            dbg("before inertia tensor");
            T I[3][3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    I[r][c] = 0.0;
                }
            }

            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *x = &elem_centroids[3 * ielem];

                T dx = x[0] - xcg[0];
                T dy = x[1] - xcg[1];
                T dz = x[2] - xcg[2];

                I[0][0] += dy * dy + dz * dz;
                I[1][1] += dx * dx + dz * dz;
                I[2][2] += dx * dx + dy * dy;
                I[0][1] -= dx * dy;
                I[0][2] -= dx * dz;
                I[1][2] -= dy * dz;
            }
            I[1][0] = I[0][1];
            I[2][0] = I[0][2];
            I[2][1] = I[1][2];
            dbg("after inertia tensor");

            dbg("before eig3x3_exact_givens");
            T evals[3];
            T VT[9];
            T A_I[9] = {
                I[0][0], I[0][1], I[0][2], I[1][0], I[1][1], I[1][2], I[2][0], I[2][1], I[2][2],
            };
            eig3x3_exact_givens<T, 12, false, true>(A_I, evals, VT);
            dbg("after eig3x3_exact_givens");
            print_t_sample("evals", evals, 3, 3);

            T axis0[3] = {VT[0], VT[1], VT[2]};
            T axis1[3] = {VT[3], VT[4], VT[5]};
            T axis2[3] = {VT[6], VT[7], VT[8]};
            normalize(axis0);
            normalize(axis1);
            normalize(axis2);

            T inertial_normal[3] = {axis0[0], axis0[1], axis0[2]};
            T inertial_axis_a[3] = {axis1[0], axis1[1], axis1[2]};
            T inertial_axis_b[3] = {axis2[0], axis2[1], axis2[2]};

            if (vec_dot(inertial_normal, geom_normal) < 0.0) {
                inertial_normal[0] *= -1.0;
                inertial_normal[1] *= -1.0;
                inertial_normal[2] *= -1.0;
            }

            T ex[3] = {1.0, 0.0, 0.0};
            T ey[3] = {0.0, 1.0, 0.0};
            T ez[3] = {0.0, 0.0, 1.0};

            T ref_axis1[3], ref_axis2[3], ref_axis3[3];
            ref_axis3[0] = inertial_normal[0];
            ref_axis3[1] = inertial_normal[1];
            ref_axis3[2] = inertial_normal[2];

            T dax = fabs(vec_dot(inertial_axis_a, ex));
            T day = fabs(vec_dot(inertial_axis_a, ey));
            T daz = fabs(vec_dot(inertial_axis_a, ez));

            if (dax >= day && dax >= daz) {
                ref_axis1[0] = 1.0;
                ref_axis1[1] = 0.0;
                ref_axis1[2] = 0.0;
            } else if (day >= dax && day >= daz) {
                ref_axis1[0] = 0.0;
                ref_axis1[1] = 1.0;
                ref_axis1[2] = 0.0;
            } else {
                ref_axis1[0] = 0.0;
                ref_axis1[1] = 0.0;
                ref_axis1[2] = 1.0;
            }

            T dbx = fabs(vec_dot(inertial_axis_b, ex));
            T dby = fabs(vec_dot(inertial_axis_b, ey));
            T dbz = fabs(vec_dot(inertial_axis_b, ez));

            if (dbx >= dby && dbx >= dbz) {
                ref_axis2[0] = 1.0;
                ref_axis2[1] = 0.0;
                ref_axis2[2] = 0.0;
            } else if (dby >= dbx && dby >= dbz) {
                ref_axis2[0] = 0.0;
                ref_axis2[1] = 1.0;
                ref_axis2[2] = 0.0;
            } else {
                ref_axis2[0] = 0.0;
                ref_axis2[1] = 0.0;
                ref_axis2[2] = 1.0;
            }

            print_t_sample("ref_axis1", ref_axis1, 3, 3);
            print_t_sample("ref_axis2", ref_axis2, 3, 3);
            print_t_sample("ref_axis3", ref_axis3, 3, 3);

            dbg("before min_lelem search");
            int min_lelem = -1;
            T min_xi = 1e30, min_eta = 1e30;
            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *xpt_elem = &elem_centroids[3 * ielem];
                T xi = A2D::VecDotCore<T, 3>(xpt_elem, ref_axis1);
                T eta = A2D::VecDotCore<T, 3>(xpt_elem, ref_axis2);
                if (xi + eta < min_xi + min_eta) {
                    min_lelem = i;
                    min_xi = xi;
                    min_eta = eta;
                }
            }
            dbg("after min_lelem search");
            dbg_val("min_lelem", min_lelem);
            dbg_val2("min_xi", min_xi);
            dbg_val2("min_eta", min_eta);

            dbg("before allocate ixe/iye/assigned");
            int *ixe_indices = new int[_num_elems];
            int *iye_indices = new int[_num_elems];
            bool *assigned = new bool[_num_elems];

            for (int i = 0; i < _num_elems; i++) {
                ixe_indices[i] = 0;
                iye_indices[i] = 0;
                assigned[i] = false;
            }

            ixe_indices[min_lelem] = 0;
            iye_indices[min_lelem] = 0;
            assigned[min_lelem] = true;
            int n_assigned = 1;
            int i = min_lelem;
            dbg("after allocate ixe/iye/assigned");

            dbg("before while(n_assigned < _num_elems)");
            int while_iter = 0;
            while (n_assigned < _num_elems) {
                while_iter++;
                if (print_debug && (while_iter <= 10 || while_iter % 100 == 0)) {
                    printf("[SD-DBG] component %d while_iter=%d n_assigned=%d/%d seed_i=%d\n",
                           icomp, while_iter, n_assigned, _num_elems, i);
                    fflush(stdout);
                }

                bool any_unfilled = false;
                bool assigned_new = false;

                for (int jp = e2e_rowp[i]; jp < e2e_rowp[i + 1]; jp++) {
                    int j = e2e_cols[jp];
                    if (assigned[j]) continue;

                    any_unfilled = true;

                    int ielem = e2e_ielem[i], jelem = e2e_ielem[j];
                    T *xpti = &elem_centroids[3 * ielem];
                    T *xptj = &elem_centroids[3 * jelem];

                    T dx[3];
                    for (int dim = 0; dim < 3; dim++) {
                        dx[dim] = xptj[dim] - xpti[dim];
                    }

                    T xi_dist = A2D::VecDotCore<T, 3>(dx, ref_axis1);
                    T eta_dist = A2D::VecDotCore<T, 3>(dx, ref_axis2);

                    if (fabs(xi_dist) > fabs(eta_dist)) {
                        int sign = (xi_dist > 0.0) ? 1 : -1;
                        ixe_indices[j] = ixe_indices[i] + sign;
                        iye_indices[j] = iye_indices[i];
                    } else {
                        int sign = (eta_dist > 0.0) ? 1 : -1;
                        ixe_indices[j] = ixe_indices[i];
                        iye_indices[j] = iye_indices[i] + sign;
                    }

                    assigned[j] = true;
                    n_assigned++;
                    i = j;
                    assigned_new = true;
                    break;
                }

                if (!assigned_new) {
                    bool found_seed = false;
                    for (int j = 0; j < _num_elems; j++) {
                        bool has_unfilled_adj = false;
                        for (int kp = e2e_rowp[j]; kp < e2e_rowp[j + 1]; kp++) {
                            int k = e2e_cols[kp];
                            if (!assigned[k]) has_unfilled_adj = true;
                        }
                        if (has_unfilled_adj) {
                            i = j;
                            found_seed = true;
                            break;
                        }
                    }
                    if (!found_seed) {
                        dbg("breaking while loop because !found_seed");
                        break;
                    }
                }
            }
            dbg("after while(n_assigned < _num_elems)");
            dbg_val("final n_assigned", n_assigned);
            print_int_sample("ixe_indices", ixe_indices, _num_elems);
            print_int_sample("iye_indices", iye_indices, _num_elems);

            int ixe_min = ixe_indices[0], ixe_max = ixe_indices[0];
            int iye_min = iye_indices[0], iye_max = iye_indices[0];
            for (int i2 = 1; i2 < _num_elems; i2++) {
                ixe_min = (ixe_indices[i2] < ixe_min) ? ixe_indices[i2] : ixe_min;
                ixe_max = (ixe_indices[i2] > ixe_max) ? ixe_indices[i2] : ixe_max;
                iye_min = (iye_indices[i2] < iye_min) ? iye_indices[i2] : iye_min;
                iye_max = (iye_indices[i2] > iye_max) ? iye_indices[i2] : iye_max;
            }

            for (int i2 = 0; i2 < _num_elems; i2++) {
                ixe_indices[i2] -= ixe_min;
                iye_indices[i2] -= iye_min;
            }

            int nxs_comp = (ixe_max - ixe_min + nxse_ - 1) / nxse_;
            int nys_comp = (iye_max - iye_min + nyse_ - 1) / nyse_;

            if (MOD_WRAPAROUND != -1) {
                dbg("before wraparound index shift");
                for (int i2 = 0; i2 < _num_elems; i2++) {
                    ixe_indices[i2] += MOD_WRAPAROUND;
                    iye_indices[i2] += MOD_WRAPAROUND;
                }

                nxs_comp = (ixe_max - ixe_min + MOD_WRAPAROUND + nxse_ - 1) / nxse_;
                nys_comp = (iye_max - iye_min + MOD_WRAPAROUND + nyse_ - 1) / nyse_;
                dbg("after wraparound index shift");
            }

            dbg_val("ixe_min", ixe_min);
            dbg_val("ixe_max", ixe_max);
            dbg_val("iye_min", iye_min);
            dbg_val("iye_max", iye_max);
            dbg_val("nxs_comp", nxs_comp);
            dbg_val("nys_comp", nys_comp);

            dbg("before elem_sd_ind write for component");
            for (int i2 = 0; i2 < _num_elems; i2++) {
                int ielem = e2e_ielem[i2];
                int ixe = ixe_indices[i2];
                int iye = iye_indices[i2];

                int ixs = ixe / nxse_;
                int iys = iye / nyse_;
                elem_sd_ind[ielem] = i_subdomain + ixs + iys * nxs_comp;
                debug_lelem_sd_ind[ielem] = elem_sd_ind[ielem] - i_subdomain;
            }
            dbg("after elem_sd_ind write for component");
            print_int_sample("debug_lelem_sd_ind partial", debug_lelem_sd_ind, num_elements);

            i_subdomain += nxs_comp * nys_comp;
            dbg_val("running i_subdomain", i_subdomain);

            delete[] ixe_indices;
            delete[] iye_indices;
            delete[] assigned;
            dbg("component cleanup complete");
        }

        dbg("after component loop");
        num_subdomains = i_subdomain;
        dbg_val("num_subdomains", num_subdomains);
        print_int_sample("elem_sd_ind", elem_sd_ind, num_elements);

        dbg("before printToVTK comp_sd0");
        auto h_vars0 = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars0, elem_sd_ind,
                                                            "subdomains", "out/comp_sd0.vtk");
        dbg("after printToVTK comp_sd0");

        if (MOD_WRAPAROUND != -1) {
            dbg("ENTER wraparound postprocessing");

            MAX_NUM_VERTEX_PER_SUBDOMAIN = 6;

            node_elem_nnz = 0;
            node_elem_rowp = new int[num_nodes + 1];
            node_elem_ct = new int[num_nodes];
            std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
            std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

            dbg("before wraparound node->subdomain incidence count");
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    node_elem_ct[gnode]++;
                    node_elem_nnz++;
                }
            }
            dbg("after wraparound node->subdomain incidence count");
            dbg_val("node_elem_nnz", node_elem_nnz);

            for (int inode = 0; inode < num_nodes; inode++) {
                node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
            }
            print_int_sample("node_elem_rowp", node_elem_rowp, num_nodes + 1);

            int *temp_node_elem0 = new int[num_nodes];
            node_sd_cols = new int[node_elem_nnz];
            std::memset(temp_node_elem0, 0, num_nodes * sizeof(int));
            std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

            dbg("before wraparound node_sd_cols fill");
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                int subdomain_ind = elem_sd_ind[ielem];

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    int offset = node_elem_rowp[gnode] + temp_node_elem0[gnode];
                    node_sd_cols[offset] = subdomain_ind;
                    temp_node_elem0[gnode]++;
                }
            }
            delete[] temp_node_elem0;
            dbg("after wraparound node_sd_cols fill");
            print_int_sample("node_sd_cols", node_sd_cols, node_elem_nnz);

            int *node_sd_bndry_ind = new int[num_nodes];
            std::memset(node_sd_bndry_ind, 0, num_nodes * sizeof(int));

            dbg("before node_sd_bndry_ind classification");
            for (int inode = 0; inode < num_nodes; inode++) {
                int start = node_elem_rowp[inode];
                int end = node_elem_rowp[inode + 1];

                if (start == end) {
                    node_sd_bndry_ind[inode] = 0;
                    continue;
                }

                int sd0 = node_sd_cols[start];
                int count_in_sd = 0;
                for (int j = start; j < end; j++) {
                    if (node_sd_cols[j] == sd0) {
                        count_in_sd++;
                    }
                }

                if (count_in_sd == 4) {
                    node_sd_bndry_ind[inode] = 0;
                } else if (count_in_sd == 2) {
                    node_sd_bndry_ind[inode] = 1;
                } else if (count_in_sd == 1) {
                    node_sd_bndry_ind[inode] = 2;
                } else {
                    printf("warning: inode %d has count_in_sd = %d in subdomain %d\n", inode,
                           count_in_sd, sd0);
                    fflush(stdout);
                    node_sd_bndry_ind[inode] = 2;
                }
            }
            dbg("after node_sd_bndry_ind classification");
            print_int_sample("node_sd_bndry_ind", node_sd_bndry_ind, num_nodes);

            std::vector<std::vector<int>> sd_groups;
            dbg("before sd_groups build");
            for (int inode = 0; inode < num_nodes; inode++) {
                int _sd_bndry = node_sd_bndry_ind[inode];
                int _comp_bndry = node_wing_geom_ind[inode];

                if (_sd_bndry == 1 && _comp_bndry == 1) {
                    int start = node_elem_rowp[inode];
                    int end = node_elem_rowp[inode + 1];

                    std::vector<int> group;
                    group.reserve(end - start);

                    for (int j = start; j < end; j++) {
                        group.push_back(node_sd_cols[j]);
                    }

                    std::sort(group.begin(), group.end());
                    group.erase(std::unique(group.begin(), group.end()), group.end());

                    if (group.size() > 1) {
                        sd_groups.push_back(group);
                    }
                }
            }
            dbg("after sd_groups build");
            dbg_val("sd_groups.size()", (int)sd_groups.size());

            std::sort(sd_groups.begin(), sd_groups.end());
            sd_groups.erase(std::unique(sd_groups.begin(), sd_groups.end()), sd_groups.end());
            dbg_val("unique sd_groups.size()", (int)sd_groups.size());

            int *elem_sd_cts = new int[num_subdomains];
            memset(elem_sd_cts, 0, num_subdomains * sizeof(int));
            int *elem_sd_rowp = new int[num_subdomains + 1];
            memset(elem_sd_rowp, 0, (num_subdomains + 1) * sizeof(int));

            dbg("before elem_sd_cts count");
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int i_subdomain_local = elem_sd_ind[ielem];
                elem_sd_cts[i_subdomain_local]++;
            }
            dbg("after elem_sd_cts count");

            for (int i_sd = 0; i_sd < num_subdomains; i_sd++) {
                elem_sd_rowp[i_sd + 1] = elem_sd_rowp[i_sd] + elem_sd_cts[i_sd];
            }
            print_int_sample("elem_sd_rowp", elem_sd_rowp, num_subdomains + 1);

            int *elem_sd_cols = new int[num_elements];
            memset(elem_sd_cols, 0, num_elements * sizeof(int));
            memset(elem_sd_cts, 0, num_subdomains * sizeof(int));

            dbg("before elem_sd_cols fill");
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int i_subdomain_local = elem_sd_ind[ielem];
                int offset = elem_sd_rowp[i_subdomain_local] + elem_sd_cts[i_subdomain_local];
                elem_sd_cols[offset] = ielem;
                elem_sd_cts[i_subdomain_local]++;
            }
            dbg("after elem_sd_cols fill");

            int ct = 0;
            int n_wrap = sd_groups.size() * wrap_frac;
            dbg_val("n_wrap", n_wrap);

            dbg("before applying wrap groups");
            for (auto group : sd_groups) {
                int first_sd_ind = group[0];
                if (ct >= n_wrap) continue;
                ct++;

                int isd0 = group[0];
                int ielem0 = elem_sd_cols[elem_sd_rowp[isd0]];
                if (elem_sd_ind[ielem0] != first_sd_ind) first_sd_ind = elem_sd_ind[ielem0];

                for (auto isd : group) {
                    for (int jp = elem_sd_rowp[isd]; jp < elem_sd_rowp[isd + 1]; jp++) {
                        int ielem = elem_sd_cols[jp];
                        elem_sd_ind[ielem] = first_sd_ind;
                    }
                }
            }
            dbg("after applying wrap groups");

            std::vector<int> subdomains;
            subdomains.reserve(num_elements);

            for (int ielem = 0; ielem < num_elements; ielem++) {
                subdomains.push_back(elem_sd_ind[ielem]);
            }

            int num_subdomains_0 = num_subdomains;
            std::sort(subdomains.begin(), subdomains.end());
            subdomains.erase(std::unique(subdomains.begin(), subdomains.end()), subdomains.end());
            num_subdomains = subdomains.size();
            dbg_val("num_subdomains after wrap reduction", num_subdomains);

            int *subdomain_iperm = new int[num_subdomains_0];
            memset(subdomain_iperm, -1, num_subdomains_0 * sizeof(int));

            for (int isd = 0; isd < num_subdomains; isd++) {
                int old_sd = subdomains[isd];
                subdomain_iperm[old_sd] = isd;
            }

            dbg("before elem_sd_ind renumber");
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int old_sd = elem_sd_ind[ielem];
                int new_red_sd = subdomain_iperm[old_sd];
                elem_sd_ind[ielem] = new_red_sd;
            }
            dbg("after elem_sd_ind renumber");
            print_int_sample("elem_sd_ind after wrap", elem_sd_ind, num_elements);
        }

        dbg("before print VTKs after subdomain assignment");
        auto h_vars = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars, elem_sd_ind,
                                                            "subdomains", "out/comp_sd.vtk");
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars, debug_lelem_sd_ind,
                                                            "subdomains", "out/comp_lsd.vtk");

        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, LOWER_SKIN>(
            assembler, h_vars, "out/comp_lskin.vtk", num_subdomains, elem_sd_ind);
        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, UPPER_SKIN>(
            assembler, h_vars, "out/comp_uskin.vtk", num_subdomains, elem_sd_ind);
        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, INT_STRUCT>(
            assembler, h_vars, "out/comp_intstruct.vtk", num_subdomains, elem_sd_ind);
        dbg("after print VTKs after subdomain assignment");

        dbg("before final node->subdomain incidence rebuild");
        node_elem_nnz = 0;
        node_elem_rowp = new int[num_nodes + 1];
        node_elem_ct = new int[num_nodes];
        std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
        std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        int *temp_node_elem = new int[num_nodes];
        node_sd_cols = new int[node_elem_nnz];
        std::memset(temp_node_elem, 0, num_nodes * sizeof(int));
        std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int subdomain_ind = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;
        dbg("after final node->subdomain incidence rebuild");

        dbg("before node classification");
        node_class_ind = new int[num_nodes];
        std::memset(node_class_ind, 0, num_nodes * sizeof(int));

        printf(
            "WARNING: for unstructured meshes, this node classification may fail at junctions.\n");
        fflush(stdout);

        node_nsd = new int[num_nodes];
        I_nnodes = 0, IE_nnodes = 0, IEV_nnodes = 0;
        Vc_nnodes = 0, V_nnodes = 0, lam_nnodes = 0;

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int nsd = static_cast<int>(node_sds.size());
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;
                I_nnodes++, IE_nnodes++, IEV_nnodes++;
            } else if (dirichlet_ind[inode] && track_dirichlet) {
                node_class_ind[inode] = INTERIOR;
                if (node_bndry_ind[inode] > 0) {
                    node_class_ind[inode] = DIRICHLET_EDGE;
                    I_nnodes += nsd;
                    IE_nnodes += nsd;
                    IEV_nnodes += nsd;
                }
            } else if (nsd == 2 || node_wing_geom_ind[inode] == 1) {
                node_class_ind[inode] = EDGE;
                lam_nnodes++;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = VERTEX;
                Vc_nnodes++;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }
        dbg("after node classification");
        dbg_val("I_nnodes", I_nnodes);
        dbg_val("IE_nnodes", IE_nnodes);
        dbg_val("IEV_nnodes", IEV_nnodes);
        dbg_val("Vc_nnodes", Vc_nnodes);
        dbg_val("V_nnodes", V_nnodes);
        dbg_val("lam_nnodes", lam_nnodes);
        print_int_sample("node_nsd", node_nsd, num_nodes);
        print_int_sample("node_class_ind", node_class_ind, num_nodes);

        dbg("before comp_node_nsd / comp_node_class VTK");
        T *h_soln = new T[num_nodes * block_dim];
        memset(h_soln, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_soln[6 * i] = node_nsd[i];
        }
        auto h_soln_debug = HostVec<T>(num_nodes * block_dim, h_soln);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_soln_debug, "out/comp_node_nsd.vtk");

        T *h_soln2 = new T[num_nodes * block_dim];
        memset(h_soln2, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_soln2[6 * i] = node_class_ind[i];
        }
        auto h_soln_debug2 = HostVec<T>(num_nodes * block_dim, h_soln2);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_soln_debug2, "out/comp_node_class.vtk");
        dbg("after comp_node_nsd / comp_node_class VTK");

        dbg("before duplicated IEV nodal layout");
        IEV_sd_ptr = new int[num_subdomains + 1];
        IEV_sd_ind = new int[IEV_nnodes];
        IEV_nodes = new int[IEV_nnodes];
        std::memset(IEV_sd_ptr, 0, (num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[num_nodes];

        for (int i_subdomain_local = 0; i_subdomain_local < num_subdomains; i_subdomain_local++) {
            std::memset(temp_completion, 0, num_nodes * sizeof(int));
            IEV_sd_ptr[i_subdomain_local + 1] = IEV_sd_ptr[i_subdomain_local];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != i_subdomain_local) continue;

                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    if (temp_completion[gnode]) continue;

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = i_subdomain_local;
                    IEV_sd_ptr[i_subdomain_local + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;
        dbg("after duplicated IEV nodal layout");
        dbg_val("IEV_ind", IEV_ind);
        print_int_sample("IEV_sd_ptr", IEV_sd_ptr, num_subdomains + 1);
        print_int_sample("IEV_nodes", IEV_nodes, IEV_nnodes);
        print_int_sample("IEV_sd_ind", IEV_sd_ind, IEV_nnodes);

        dbg("before IEV element connectivity");
        IEV_elem_conn = new int[num_elements * nodes_per_elem];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int i_subdomain_local = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[i_subdomain_local]; jp < IEV_sd_ptr[i_subdomain_local + 1];
                     jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                    fflush(stdout);
                }
                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }
        dbg("after IEV element connectivity");
        print_int_sample("IEV_elem_conn", IEV_elem_conn, num_elements * nodes_per_elem);

        dbg("before IE / I nodal lists");
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];
        std::memset(IE_nodes, 0, IE_nnodes * sizeof(int));
        std::memset(I_nodes, 0, I_nnodes * sizeof(int));
        IE_interior = new bool[IE_nnodes];
        IE_general_edge = new bool[IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                I_nodes[I_ind++] = gnode;
            }
        }
        dbg("after IE / I nodal lists");
        print_int_sample("IE_nodes", IE_nodes, IE_nnodes);
        print_int_sample("I_nodes", I_nodes, I_nnodes);
        print_bool_sample("IE_interior", IE_interior, IE_nnodes);
        print_bool_sample("IE_general_edge", IE_general_edge, IE_nnodes);

        d_IE_interior = HostVec<bool>(IE_nnodes, IE_interior).createDeviceVec().getPtr();
        d_IE_general_edge = HostVec<bool>(IE_nnodes, IE_general_edge).createDeviceVec().getPtr();
        d_IE_nodes = HostVec<int>(IE_nnodes, IE_nodes).createDeviceVec().getPtr();
        dbg("after IE metadata moved to device");

        dbg("before IEV_bsr_data");
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);
        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;
        IEV_rows = new int[IEV_nnzb];
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            for (int jp = IEV_rowp[inode]; jp < IEV_rowp[inode + 1]; jp++) {
                IEV_rows[jp] = inode;
            }
        }
        dbg("after IEV_bsr_data");
        dbg_val("IEV_nnzb", IEV_nnzb);
        print_int_sample("IEV_rowp", IEV_rowp, IEV_nnodes + 1);
        print_int_sample("IEV_cols", IEV_cols, IEV_nnzb);

        dbg("before reduced rowp arrays");
        IE_rowp = new int[IE_nnodes + 1];
        I_rowp = new int[I_nnodes + 1];
        std::memset(IE_rowp, 0, (IE_nnodes + 1) * sizeof(int));
        std::memset(I_rowp, 0, (I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) I_rowp[I_row + 1] = I_rowp[I_row];
            if (typeIE_row) IE_rowp[IE_row + 1] = IE_rowp[IE_row];

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        I_nnzb = I_rowp[I_nnodes];
        IE_nnzb = IE_rowp[IE_nnodes];
        dbg("after reduced rowp arrays");
        dbg_val("I_nnzb", I_nnzb);
        dbg_val("IE_nnzb", IE_nnzb);
        print_int_sample("I_rowp", I_rowp, I_nnodes + 1);
        print_int_sample("IE_rowp", IE_rowp, IE_nnodes + 1);

        IE_rows = new int[IE_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }
        dbg("after reduced rows arrays");

        dbg("before IEV->IE map");
        IEVtoIE_map = new int[IEV_nnodes];
        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        IEVtoIE_imap = new int[IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IEVtoIE_imap[IE_ind] = inode;
                IEVtoIE_map[inode] = IE_ind++;
            }
        }
        dbg("after IEV->IE map");
        print_int_sample("IEVtoIE_map", IEVtoIE_map, IEV_nnodes);
        print_int_sample("IEVtoIE_imap", IEVtoIE_imap, IE_nnodes);

        d_IEVtoIE_imap = HostVec<int>(IE_nnodes, IEVtoIE_imap).createDeviceVec().getPtr();
        dbg("after d_IEVtoIE_imap");

        dbg("before IEV->I map");
        IEVtoI_map = new int[IEV_nnodes];
        IEVtoI_imap = new int[I_nnodes];
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IEVtoI_imap[I_ind] = inode;
                IEVtoI_map[inode] = I_ind++;
            }
        }
        dbg("after IEV->I map");
        print_int_sample("IEVtoI_map", IEVtoI_map, IEV_nnodes);
        print_int_sample("IEVtoI_imap", IEVtoI_imap, I_nnodes);

        d_IEVtoI_imap = HostVec<int>(I_nnodes, IEVtoI_imap).createDeviceVec().getPtr();
        dbg("after d_IEVtoI_imap");

        dbg("before reduced column arrays");
        IE_cols = new int[IE_nnzb];
        I_cols = new int[I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    I_cols[I_ind++] = IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    IE_cols[IE_ind++] = IEVtoIE_map[col];
                }
            }
        }
        dbg("after reduced column arrays");
        print_int_sample("I_cols", I_cols, I_nnzb);
        print_int_sample("IE_cols", IE_cols, IE_nnzb);

        dbg("before d_IEV_elem_conn");
        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();
        dbg("after d_IEV_elem_conn");

        dbg("before Vc setup");
        std::unordered_set<int> Vc_nodeset;
        for (int i2 = 0; i2 < IEV_nnodes; i2++) {
            int gnode = IEV_nodes[i2];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());

        int *Vc_inodes = new int[num_nodes];
        memset(Vc_inodes, -1, num_nodes * sizeof(int));
        for (int i2 = 0; i2 < (int)Vc_nodes_vec.size(); i2++) {
            int j = Vc_nodes_vec[i2];
            Vc_inodes[j] = i2;
        }

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();
        dbg("after Vc setup");
        print_int_sample("Vc_nodes", Vc_nodes, Vc_nnodes);

        int nwrap_nodes = 0;
        for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
            int glob_node = Vc_nodes[ivc];
            int wing_node_class = node_wing_geom_ind[glob_node];
            if (wing_node_class == 0) {
                nwrap_nodes++;
            }
        }
        T wrap_node_frac = 1.0 * nwrap_nodes / Vc_nnodes;
        printf("frac of nodes on interior (for wrap subdomains): %d/%d = %.4f\n", nwrap_nodes,
               Vc_nnodes, wrap_node_frac);
        fflush(stdout);

        dbg("before VctoV/IEVtoV maps");
        VctoV_imap = new int[V_nnodes];
        std::memset(VctoV_imap, -1, V_nnodes * sizeof(int));
        IEVtoV_imap = new int[V_nnodes];
        std::memset(IEVtoV_imap, -1, V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                VctoV_imap[V_ind] = Vc_ind;
                IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }
        dbg("after VctoV/IEVtoV maps");
        print_int_sample("VctoV_imap", VctoV_imap, V_nnodes);
        print_int_sample("IEVtoV_imap", IEVtoV_imap, V_nnodes);

        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();
        dbg("after d_IEVtoV_imap / d_VctoV_imap");

        dbg("before _compute_jump_operators");
        bool square_domain = compute_jump;
        _compute_jump_operators(square_domain);
        dbg("after _compute_jump_operators");

        dbg("before allocate_workspace");
        allocate_workspace();
        dbg("after allocate_workspace");

        dbg("before IE_bsr_data / I_bsr_data");
        IE_bsr_data = BsrData(IE_nnodes, block_dim, IE_nnzb, IE_rowp, IE_cols);
        IE_bsr_data.rows = IE_rows;
        IE_nofill_nnzb = IE_nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        I_bsr_data = BsrData(I_nnodes, block_dim, I_nnzb, I_rowp, I_cols);
        I_bsr_data.rows = I_rows;
        I_nofill_nnzb = I_nnzb;
        dbg("after IE_bsr_data / I_bsr_data");

        dbg("EXIT setup_tacs_component_subdomains_debug");
    }

    void _setup_tacs_component_subdomains_nodebug(int nxse_, int nyse_, int MOD_WRAPAROUND = -1,
                                                  T wrap_frac = 1.0, bool compute_jump = true,
                                                  bool track_dirichlet = false) {
        clear_structured_host_data();

        // printf("SETUP_WING_SUBDOMAINS : get_nodal_geom_indices\n");
        int *nodal_num_wing_comps, *node_wing_geom_ind;
        WingboxMultiColoring<ShellAssembler>::get_nodal_geom_indices(
            assembler, nodal_num_wing_comps, node_wing_geom_ind);

        // modify node_wing_geom_ind for plate + cylinder fuselage cases
        // add edges and vertices if they only belong to one or two elements
        // for plate + cylinder fuselage cases (outer bndry)
        int *node_nelems = new int[num_nodes];
        memset(node_nelems, 0, num_nodes * sizeof(int));
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_nelems[gnode]++;
            }
        }

        // compute the BC indices needed for kmat_IEV
        auto d_bcs = assembler.getBCs();
        int n_orig_bcs = d_bcs.getSize();
        int *h_bcs = d_bcs.createHostVec().getPtr();
        // printf("h_bcs: ");
        // printVec<int>(n_orig_bcs, h_bcs);

        // get from Dirichlet bcs
        bool *dirichlet_ind = new bool[num_nodes];
        memset(dirichlet_ind, false, num_nodes * sizeof(bool));
        for (int ibc = 0; ibc < n_orig_bcs; ibc++) {
            int bc_node = h_bcs[ibc] / block_dim;
            dirichlet_ind[bc_node] = true;
        }
        // printf("dirichlet ind: ");
        // printVec<bool>(num_nodes, dirichlet_ind);

        int *node_bndry_ind = new int[num_nodes];
        for (int inode = 0; inode < num_nodes; inode++) {
            node_bndry_ind[inode] = 0;
            int nelems_attached = node_nelems[inode];
            // outer bndry nodes changed to edge + vertices
            if (nelems_attached == 2) {
                node_bndry_ind[inode] = 1;  // change it to edge node if was labeled interior
            } else if (nelems_attached == 1) {
                node_bndry_ind[inode] = 2;  // change to vertex node
            }
        }

        T *h_wgeom_ind = new T[num_nodes * block_dim];
        memset(h_wgeom_ind, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_wgeom_ind[6 * i] = node_wing_geom_ind[i];
        }
        auto h_wgeom_vec = HostVec<T>(num_nodes * block_dim, h_wgeom_ind);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_wgeom_vec,
                                               "out/wing_node_geom_ind.vtk");

        int num_comps = assembler.get_num_components();
        int *h_elem_comps = assembler.getElemComponents().createHostVec().getPtr();
        T *h_xpts = d_xpts.createHostVec().getPtr();

        // compute the centroid of each element
        // printf("SETUP_WING_SUBDOMAINS : get_elem_centroids\n");
        T *elem_centroids = new T[3 * num_elements];
        memset(elem_centroids, 0, 3 * num_elements * sizeof(T));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                T *xpt = &h_xpts[3 * gnode];
                elem_centroids[3 * ielem + 0] += xpt[0];
                elem_centroids[3 * ielem + 1] += xpt[1];
                elem_centroids[3 * ielem + 2] += xpt[2];
            }

            elem_centroids[3 * ielem + 0] /= nodes_per_elem;
            elem_centroids[3 * ielem + 1] /= nodes_per_elem;
            elem_centroids[3 * ielem + 2] /= nodes_per_elem;
        }

        elem_sd_ind = new int[num_elements];
        memset(elem_sd_ind, 0, num_elements * sizeof(int));
        int *debug_lelem_sd_ind = new int[num_elements];  // elem_sd_ind in local components
        memset(debug_lelem_sd_ind, 0, num_elements * sizeof(int));
        int i_subdomain = 0;

        // printf("SETUP_WING_SUBDOMAINS : elem2elem_row_cts + comp_num_elems\n");
        int *elem2elem_row_cts = new int[num_elements];
        int *comp_num_elems = new int[num_comps];
        memset(elem2elem_row_cts, 0, num_elements * sizeof(int));
        memset(comp_num_elems, 0, num_comps * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int icomp = h_elem_comps[ielem];
            comp_num_elems[icomp]++;

            bool elem_is_interior = true;
            bool elem_is_edge = true;
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int node_class = node_wing_geom_ind[gnode];
                int bndry_class = node_bndry_ind[gnode];
                if (node_class > 0) elem_is_interior = false;
                if (node_class > 1) elem_is_edge = false;
                if (bndry_class > 0) elem_is_interior = false;
                if (bndry_class > 1) elem_is_edge = false;
                // on bndry + inter-subdomain edge also counts like a vertex
                if (bndry_class == 1 && node_class == 1) elem_is_edge = false;
            }

            if (elem_is_interior) {
                elem2elem_row_cts[ielem] = 4;
            } else if (elem_is_edge) {
                elem2elem_row_cts[ielem] = 3;
            } else {
                elem2elem_row_cts[ielem] = 2;
            }
        }

        // printf("comp_num_elems (%d num_comps): ", num_comps);
        // printVec<int>(num_comps, comp_num_elems);

        int **elem2elem_ielem = new int *[num_comps];
        int **elem2elem_rowp = new int *[num_comps];
        int **elem2elem_cols = new int *[num_comps];
        int *elem2elem_nnz = new int[num_comps];

        for (int icomp = 0; icomp < num_comps; icomp++) {
            elem2elem_ielem[icomp] = nullptr;
            elem2elem_rowp[icomp] = nullptr;
            elem2elem_cols[icomp] = nullptr;
            elem2elem_nnz[icomp] = 0;
        }

        for (int icomp = 0; icomp < num_comps; icomp++) {
            int _num_elems = comp_num_elems[icomp];

            int *e2e_ielem = new int[_num_elems];
            int *e2e_rowp = new int[_num_elems + 1];
            elem2elem_ielem[icomp] = e2e_ielem;
            elem2elem_rowp[icomp] = e2e_rowp;

            e2e_rowp[0] = 0;

            // build reduced element list and row pointers only for this component
            int elem_ct = 0;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int jcomp = h_elem_comps[ielem];
                if (icomp != jcomp) {
                    continue;
                }

                e2e_ielem[elem_ct] = ielem;
                e2e_rowp[elem_ct + 1] = e2e_rowp[elem_ct] + elem2elem_row_cts[ielem];
                elem_ct++;
            }

            if (elem_ct != _num_elems) {
                printf("ERROR: icomp %d expected _num_elems %d but got elem_ct %d\n", icomp,
                       _num_elems, elem_ct);
                exit(1);
            }

            int nnz = e2e_rowp[_num_elems];
            elem2elem_nnz[icomp] = nnz;

            // printf("\n\nicomp %d\n", icomp);
            // printf("e2e_ielem (_num_elems %d): ", _num_elems);
            // printVec<int>(_num_elems, e2e_ielem);
            // printf("e2e_rowp: ");
            // printVec<int>(_num_elems + 1, e2e_rowp);

            int *e2e_cols = new int[nnz];
            elem2elem_cols[icomp] = e2e_cols;

            int ind = 0;
            for (int i2 = 0; i2 < _num_elems * _num_elems; i2++) {
                int i = i2 / _num_elems, j = i2 % _num_elems;
                if (i == j) continue;

                int ielem = e2e_ielem[i];
                int *li_conn = &elem_conn[nodes_per_elem * ielem];
                int jelem = e2e_ielem[j];
                int *lj_conn = &elem_conn[nodes_per_elem * jelem];

                int num_match = 0;
                for (int lnode2 = 0; lnode2 < nodes_per_elem * nodes_per_elem; lnode2++) {
                    int lnodei = lnode2 % nodes_per_elem;
                    int lnodej = lnode2 / nodes_per_elem;
                    int inode = li_conn[lnodei], jnode = lj_conn[lnodej];
                    if (inode == jnode) num_match++;
                }

                if (num_match > 1) {
                    if (ind >= nnz) {
                        printf("ERROR: e2e_cols overflow for icomp %d\n", icomp);
                        exit(1);
                    }
                    e2e_cols[ind++] = j;
                }
            }

            if (ind != nnz) {
                printf("WARNING: icomp %d expected nnz %d but filled %d\n", icomp, nnz, ind);
                nnz = ind;
                elem2elem_nnz[icomp] = ind;
            }

            // printf("e2e_cols with nnz %d: ", nnz);
            // printVec<int>(nnz, e2e_cols);

            // T xpt_min[3] = {1e20, 1e20, 1e20};
            // T xpt_max[3] = {-1e20, -1e20, -1e20};

            // for (int i = 0; i < _num_elems; i++) {
            //     int ielem = e2e_ielem[i];
            //     int *lconn = &elem_conn[nodes_per_elem * ielem];
            //     for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
            //         int gnode = lconn[lnode];
            //         T *xpt = &h_xpts[3 * gnode];
            //         for (int dim = 0; dim < 3; dim++) {
            //             xpt_min[dim] = (xpt[dim] < xpt_min[dim]) ? xpt[dim] :
            //             xpt_min[dim]; xpt_max[dim] = (xpt[dim] > xpt_max[dim]) ? xpt[dim]
            //             : xpt_max[dim];
            //         }
            //     }
            // }

            // printf("xpt_min: ");
            // printVec<T>(3, xpt_min);
            // printf("xpt_max: ");
            // printVec<T>(3, xpt_max);

            // T xpt_span[3];
            // T min_span = 1e30;
            // int min_direc = -1;
            // for (int dim = 0; dim < 3; dim++) {
            //     xpt_span[dim] = xpt_max[dim] - xpt_min[dim];
            //     if (xpt_span[dim] < min_span) {
            //         min_span = xpt_span[dim];
            //         min_direc = dim;
            //     }
            // }

            // printf("xpt_span: ");
            // printVec<T>(3, xpt_span);

            // T ref_axis1[3] = {0.0, 0.0, 0.0};
            // T ref_axis2[3] = {0.0, 0.0, 0.0};

            // if (min_direc == 0) {
            //     ref_axis1[1] = 1.0;
            //     ref_axis2[2] = 1.0;
            // } else if (min_direc == 1) {
            //     ref_axis1[0] = 1.0;
            //     ref_axis2[2] = 1.0;
            // } else {
            //     ref_axis1[0] = 1.0;
            //     ref_axis2[1] = 1.0;
            // }

            // printf("ref_axis1: ");
            // printVec<T>(3, ref_axis1);
            // printf("ref_axis2: ");
            // printVec<T>(3, ref_axis2);

            T xpt_min[3] = {1e20, 1e20, 1e20};
            T xpt_max[3] = {-1e20, -1e20, -1e20};

            // ---------------------------------------------
            // First pass: bounds + component centroid
            // ---------------------------------------------
            T xcg[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *xpt_elem = &elem_centroids[3 * ielem];

                xcg[0] += xpt_elem[0];
                xcg[1] += xpt_elem[1];
                xcg[2] += xpt_elem[2];

                int *lconn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = lconn[lnode];
                    T *xpt = &h_xpts[3 * gnode];
                    for (int dim = 0; dim < 3; dim++) {
                        xpt_min[dim] = (xpt[dim] < xpt_min[dim]) ? xpt[dim] : xpt_min[dim];
                        xpt_max[dim] = (xpt[dim] > xpt_max[dim]) ? xpt[dim] : xpt_max[dim];
                    }
                }
            }
            xcg[0] /= _num_elems;
            xcg[1] /= _num_elems;
            xcg[2] /= _num_elems;

            // printf("xpt_min: ");
            // printVec<T>(3, xpt_min);
            // printf("xpt_max: ");
            // printVec<T>(3, xpt_max);

            T xpt_span[3];
            for (int dim = 0; dim < 3; dim++) {
                xpt_span[dim] = xpt_max[dim] - xpt_min[dim];
            }
            // printf("xpt_span: ");
            // printVec<T>(3, xpt_span);

            // ---------------------------------------------
            // Helpers
            // ---------------------------------------------
            auto vec_dot = [](const T *a, const T *b) -> T {
                return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            };

            auto vec_norm = [&](const T *a) -> T { return sqrt(vec_dot(a, a)); };

            auto normalize = [&](T *a) {
                T nrm = vec_norm(a);
                if (nrm > 1e-30) {
                    a[0] /= nrm;
                    a[1] /= nrm;
                    a[2] /= nrm;
                }
            };

            auto abs_dot = [&](const T *a, const T *b) -> T { return fabs(vec_dot(a, b)); };

            auto swap_scalar = [](T &a, T &b) {
                T tmp = a;
                a = b;
                b = tmp;
            };

            auto swap_vec3 = [](T *a, T *b) {
                for (int k = 0; k < 3; k++) {
                    T tmp = a[k];
                    a[k] = b[k];
                    b[k] = tmp;
                }
            };

            // ---------------------------------------------
            // Approximate geometric normal from element faces
            // ---------------------------------------------
            T geom_normal[3] = {0.0, 0.0, 0.0};

            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                int *lconn = &elem_conn[nodes_per_elem * ielem];

                T *x0 = &h_xpts[3 * lconn[0]];
                T *x1 = &h_xpts[3 * lconn[1]];
                T *x2 = &h_xpts[3 * lconn[2]];
                T *x3 = &h_xpts[3 * lconn[3]];

                T e01[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
                T e02[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
                T e03[3] = {x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]};

                T n1[3], n2[3];
                A2D::VecCrossCore<T>(e01, e02, n1);
                A2D::VecCrossCore<T>(e02, e03, n2);

                geom_normal[0] += n1[0] + n2[0];
                geom_normal[1] += n1[1] + n2[1];
                geom_normal[2] += n1[2] + n2[2];
            }

            if (vec_norm(geom_normal) < 1e-12) {
                int min_direc = 0;
                T min_span = xpt_span[0];
                for (int dim = 1; dim < 3; dim++) {
                    if (xpt_span[dim] < min_span) {
                        min_span = xpt_span[dim];
                        min_direc = dim;
                    }
                }
                geom_normal[0] = 0.0;
                geom_normal[1] = 0.0;
                geom_normal[2] = 0.0;
                geom_normal[min_direc] = 1.0;
            }
            normalize(geom_normal);

            // printf("geom_normal: ");
            // printVec<T>(3, geom_normal);

            // ---------------------------------------------
            // Build rigid-body inertia tensor about centroid
            // using lumped unit masses at element centroids
            // ---------------------------------------------
            T I[3][3];
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    I[r][c] = 0.0;
                }
            }

            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *x = &elem_centroids[3 * ielem];

                T dx = x[0] - xcg[0];
                T dy = x[1] - xcg[1];
                T dz = x[2] - xcg[2];

                I[0][0] += dy * dy + dz * dz;
                I[1][1] += dx * dx + dz * dz;
                I[2][2] += dx * dx + dy * dy;

                I[0][1] -= dx * dy;
                I[0][2] -= dx * dz;
                I[1][2] -= dy * dz;
            }
            I[1][0] = I[0][1];
            I[2][0] = I[0][2];
            I[2][1] = I[1][2];

            // printf("Inertia tensor:\n");
            // for (int r = 0; r < 3; r++) {
            //     printf("%.6e %.6e %.6e\n", I[r][0], I[r][1], I[r][2]);
            // }

            // ---------------------------------------------
            // Diagonalize symmetric 3x3 inertia tensor
            // using existing exact Givens routine
            // ---------------------------------------------
            T evals[3];
            T VT[9];

            // copy I into flat row-major array because eig3x3_exact_givens modifies A in
            // place
            T A_I[9] = {
                I[0][0], I[0][1], I[0][2], I[1][0], I[1][1], I[1][2], I[2][0], I[2][1], I[2][2],
            };

            // use exact sorting, not smoothed sorting
            eig3x3_exact_givens<T, 12, false, true>(A_I, evals, VT);

            // rows of VT are eigenvectors
            T axis0[3] = {VT[0], VT[1], VT[2]};
            T axis1[3] = {VT[3], VT[4], VT[5]};
            T axis2[3] = {VT[6], VT[7], VT[8]};
            normalize(axis0);
            normalize(axis1);
            normalize(axis2);

            // eig3x3_exact_givens with can_swap=true, smoothed=false sorts DESCENDING
            // so:
            //   evals[0] = largest principal inertia
            //   evals[1] = middle
            //   evals[2] = smallest
            //
            // for a thin plate/shell face, the largest inertia axis should be the
            // normal-like one
            T inertial_normal[3] = {axis0[0], axis0[1], axis0[2]};
            T inertial_axis_a[3] = {axis1[0], axis1[1], axis1[2]};
            T inertial_axis_b[3] = {axis2[0], axis2[1], axis2[2]};

            // orient normal consistently with geometric normal
            if (vec_dot(inertial_normal, geom_normal) < 0.0) {
                inertial_normal[0] *= -1.0;
                inertial_normal[1] *= -1.0;
                inertial_normal[2] *= -1.0;
            }

            T ex[3] = {1.0, 0.0, 0.0};
            T ey[3] = {0.0, 1.0, 0.0};
            T ez[3] = {0.0, 0.0, 1.0};

            T ref_axis1[3], ref_axis2[3], ref_axis3[3];

            // keep normal as the inertial normal
            ref_axis3[0] = inertial_normal[0];
            ref_axis3[1] = inertial_normal[1];
            ref_axis3[2] = inertial_normal[2];

            // pick ref_axis1 as whichever Cartesian axis is closest to inertial_axis_a
            T dax = fabs(vec_dot(inertial_axis_a, ex));
            T day = fabs(vec_dot(inertial_axis_a, ey));
            T daz = fabs(vec_dot(inertial_axis_a, ez));

            if (dax >= day && dax >= daz) {
                ref_axis1[0] = 1.0;
                ref_axis1[1] = 0.0;
                ref_axis1[2] = 0.0;
            } else if (day >= dax && day >= daz) {
                ref_axis1[0] = 0.0;
                ref_axis1[1] = 1.0;
                ref_axis1[2] = 0.0;
            } else {
                ref_axis1[0] = 0.0;
                ref_axis1[1] = 0.0;
                ref_axis1[2] = 1.0;
            }

            // pick ref_axis2 as whichever Cartesian axis is closest to inertial_axis_b
            T dbx = fabs(vec_dot(inertial_axis_b, ex));
            T dby = fabs(vec_dot(inertial_axis_b, ey));
            T dbz = fabs(vec_dot(inertial_axis_b, ez));

            if (dbx >= dby && dbx >= dbz) {
                ref_axis2[0] = 1.0;
                ref_axis2[1] = 0.0;
                ref_axis2[2] = 0.0;
            } else if (dby >= dbx && dby >= dbz) {
                ref_axis2[0] = 0.0;
                ref_axis2[1] = 1.0;
                ref_axis2[2] = 0.0;
            } else {
                ref_axis2[0] = 0.0;
                ref_axis2[1] = 0.0;
                ref_axis2[2] = 1.0;
            }

            // printf("principal inertias (descending): %.6e %.6e %.6e\n", evals[0],
            // evals[1],
            //        evals[2]);
            // printf("inertial_normal: ");
            // printVec<T>(3, inertial_normal);
            // printf("inertial_axis_a: ");
            // printVec<T>(3, inertial_axis_a);
            // printf("inertial_axis_b: ");
            // printVec<T>(3, inertial_axis_b);
            // printf("ref_axis1: ");
            // printVec<T>(3, ref_axis1);
            // printf("ref_axis2: ");
            // printVec<T>(3, ref_axis2);
            // printf("ref_axis3: ");
            // printVec<T>(3, ref_axis3);

            int min_lelem = -1;
            T min_xi = 1e30, min_eta = 1e30;
            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                T *xpt_elem = &elem_centroids[3 * ielem];
                T xi = A2D::VecDotCore<T, 3>(xpt_elem, ref_axis1);
                T eta = A2D::VecDotCore<T, 3>(xpt_elem, ref_axis2);
                // printf("i %d, xi %.4e, eta %.4e\n", i, xi, eta);

                if (xi + eta < min_xi + min_eta) {
                    min_lelem = i;
                    min_xi = xi;
                    min_eta = eta;
                }
            }

            // printf("min_lelem %d, min_xi %.4e, min_eta %.4e\n", min_lelem, min_xi,
            // min_eta);

            // printf("pre ixe_indices, %d\n", _num_elems);
            int *ixe_indices = new int[_num_elems];
            int *iye_indices = new int[_num_elems];
            // printf("pre assigned int* assign\n");
            bool *assigned = new bool[_num_elems];
            // printf("mem allocate\n");

            for (int i = 0; i < _num_elems; i++) {
                ixe_indices[i] = 0;
                iye_indices[i] = 0;
                assigned[i] = false;
            }

            // printf("post memset\n");
            ixe_indices[min_lelem] = 0;
            iye_indices[min_lelem] = 0;
            // printf("assigned bool pre\n");
            assigned[min_lelem] = true;
            int n_assigned = 1;
            int i = min_lelem;

            // printf("Pre while loop\n");
            while (n_assigned < _num_elems) {
                bool any_unfilled = false;
                bool assigned_new = false;
                // printf("while loop\n");

                for (int jp = e2e_rowp[i]; jp < e2e_rowp[i + 1]; jp++) {
                    int j = e2e_cols[jp];
                    if (assigned[j]) continue;

                    any_unfilled = true;

                    int ielem = e2e_ielem[i], jelem = e2e_ielem[j];
                    T *xpti = &elem_centroids[3 * ielem];
                    T *xptj = &elem_centroids[3 * jelem];

                    T dx[3];
                    for (int dim = 0; dim < 3; dim++) {
                        dx[dim] = xptj[dim] - xpti[dim];
                    }

                    T xi_dist = A2D::VecDotCore<T, 3>(dx, ref_axis1);
                    T eta_dist = A2D::VecDotCore<T, 3>(dx, ref_axis2);

                    // printf("e2e pair (%d,%d), (xi,eta)=(%.4e,%.4e)\n", i, j, xi_dist,
                    // eta_dist);

                    if (fabs(xi_dist) > fabs(eta_dist)) {
                        int sign = (xi_dist > 0.0) ? 1 : -1;
                        ixe_indices[j] = ixe_indices[i] + sign;
                        iye_indices[j] = iye_indices[i];
                    } else {
                        int sign = (eta_dist > 0.0) ? 1 : -1;
                        ixe_indices[j] = ixe_indices[i];
                        iye_indices[j] = iye_indices[i] + sign;
                    }

                    assigned[j] = true;
                    n_assigned++;
                    i = j;
                    assigned_new = true;
                    break;
                }

                if (!assigned_new) {
                    // printf("assigned new %d, any_unfilled %d\n", assigned_new,
                    // any_unfilled); printf("\tassigned: "); printVec<bool>(_num_elems,
                    // assigned);
                    bool found_seed = false;
                    for (int j = 0; j < _num_elems; j++) {
                        bool has_unfilled_adj = false;
                        for (int kp = e2e_rowp[j]; kp < e2e_rowp[j + 1]; kp++) {
                            int k = e2e_cols[kp];
                            if (!assigned[k]) has_unfilled_adj = true;
                        }
                        if (has_unfilled_adj) {
                            // printf("\tfound seed True\n", assigned_new, any_unfilled);
                            i = j;
                            found_seed = true;
                            break;
                        }
                    }
                    if (!found_seed) {
                        break;
                    }
                }
            }

            // printf("ixe_indices: ");
            // printVec<int>(_num_elems, ixe_indices);
            // printf("iye_indices: ");
            // printVec<int>(_num_elems, iye_indices);

            int ixe_min = ixe_indices[0], ixe_max = ixe_indices[0];
            int iye_min = iye_indices[0], iye_max = iye_indices[0];
            for (int i = 1; i < _num_elems; i++) {
                ixe_min = (ixe_indices[i] < ixe_min) ? ixe_indices[i] : ixe_min;
                ixe_max = (ixe_indices[i] > ixe_max) ? ixe_indices[i] : ixe_max;
                iye_min = (iye_indices[i] < iye_min) ? iye_indices[i] : iye_min;
                iye_max = (iye_indices[i] > iye_max) ? iye_indices[i] : iye_max;
            }

            // ensures ixe, iye indices are (0,0) at corner
            for (int i = 0; i < _num_elems; i++) {
                ixe_indices[i] -= ixe_min;
                iye_indices[i] -= iye_min;
            }
            // compute number of subdomains each direction (in this component)
            int nxs_comp = (ixe_max - ixe_min + nxse_ - 1) / nxse_;
            int nys_comp = (iye_max - iye_min + nyse_ - 1) / nyse_;

            if (MOD_WRAPAROUND != -1) {
                // then modify the ixe, iye indices with modulo, so not (0,0) at corner
                // part of code to ensure wraparound subdomains
                for (int i = 0; i < _num_elems; i++) {
                    ixe_indices[i] += MOD_WRAPAROUND;
                    iye_indices[i] += MOD_WRAPAROUND;
                }

                // compute number of subdomains each direction (modify it)
                nxs_comp = (ixe_max - ixe_min + MOD_WRAPAROUND + nxse_ - 1) / nxse_;
                nys_comp = (iye_max - iye_min + MOD_WRAPAROUND + nyse_ - 1) / nyse_;
            }

            for (int i = 0; i < _num_elems; i++) {
                int ielem = e2e_ielem[i];
                int ixe = ixe_indices[i];
                int iye = iye_indices[i];

                int ixs = ixe / nxse_;
                int iys = iye / nyse_;
                elem_sd_ind[ielem] = i_subdomain + ixs + iys * nxs_comp;
                debug_lelem_sd_ind[ielem] = elem_sd_ind[ielem] - i_subdomain;
            }

            i_subdomain += nxs_comp * nys_comp;

            delete[] ixe_indices;
            delete[] iye_indices;
            delete[] assigned;
        }  // end of component loop

        num_subdomains = i_subdomain;
        // printf("num_subdomains %d\n", num_subdomains);

        auto h_vars0 = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars0, elem_sd_ind,
                                                            "subdomains", "out/comp_sd0.vtk");

        if (MOD_WRAPAROUND != -1) {
            // now make wraparound subdomains if they meet on subdomain boundaries
            MAX_NUM_VERTEX_PER_SUBDOMAIN = 6;  // for multi-patch structures

            // -----------------------------------------
            // node -> subdomain incidence (with repeats)
            // one entry per incident element
            // -----------------------------------------
            node_elem_nnz = 0;
            node_elem_rowp = new int[num_nodes + 1];
            node_elem_ct = new int[num_nodes];
            std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
            std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    node_elem_ct[gnode]++;
                    node_elem_nnz++;
                }
            }

            for (int inode = 0; inode < num_nodes; inode++) {
                node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
            }

            int *temp_node_elem0 = new int[num_nodes];
            node_sd_cols = new int[node_elem_nnz];
            std::memset(temp_node_elem0, 0, num_nodes * sizeof(int));
            std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                int subdomain_ind = elem_sd_ind[ielem];

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    int offset = node_elem_rowp[gnode] + temp_node_elem0[gnode];
                    node_sd_cols[offset] = subdomain_ind;
                    temp_node_elem0[gnode]++;
                }
            }
            delete[] temp_node_elem0;

            // -----------------------------------------
            // classify node relative to a touching subdomain:
            // 0 = interior, 1 = edge, 2 = vertex
            //
            // Since the mesh is structured, assume the classification is
            // consistent across touching subdomains for the purposes here.
            // -----------------------------------------
            int *node_sd_bndry_ind = new int[num_nodes];
            std::memset(node_sd_bndry_ind, 0, num_nodes * sizeof(int));

            for (int inode = 0; inode < num_nodes; inode++) {
                int start = node_elem_rowp[inode];
                int end = node_elem_rowp[inode + 1];

                if (start == end) {
                    node_sd_bndry_ind[inode] = 0;
                    continue;
                }

                // just use the first touching subdomain
                int sd0 = node_sd_cols[start];

                // count how many incident elements from this subdomain contain the node
                int count_in_sd = 0;
                for (int j = start; j < end; j++) {
                    if (node_sd_cols[j] == sd0) {
                        count_in_sd++;
                    }
                }

                if (count_in_sd == 4) {
                    node_sd_bndry_ind[inode] = 0;  // interior
                } else if (count_in_sd == 2) {
                    node_sd_bndry_ind[inode] = 1;  // edge
                } else if (count_in_sd == 1) {
                    node_sd_bndry_ind[inode] = 2;  // vertex
                } else {
                    // unexpected for structured quad subdomains
                    printf("warning: inode %d has count_in_sd = %d in subdomain %d\n", inode,
                           count_in_sd, sd0);
                    node_sd_bndry_ind[inode] = 2;
                }
            }

            // now for any nodes which are on subdomain EDGE boundaries, make list of
            // subdomain groups that we're going to join together
            std::vector<std::vector<int>> sd_groups;

            for (int inode = 0; inode < num_nodes; inode++) {
                int _sd_bndry = node_sd_bndry_ind[inode];
                int _comp_bndry = node_wing_geom_ind[inode];

                if (_sd_bndry == 1 && _comp_bndry == 1) {
                    // node must be on a subdomain edge, not a subdomain vertex
                    // AND on a component edge
                    int start = node_elem_rowp[inode];
                    int end = node_elem_rowp[inode + 1];

                    std::vector<int> group;
                    group.reserve(end - start);

                    // collect all incident subdomains
                    for (int j = start; j < end; j++) {
                        group.push_back(node_sd_cols[j]);
                    }

                    // unique them
                    std::sort(group.begin(), group.end());
                    group.erase(std::unique(group.begin(), group.end()), group.end());

                    // only keep actual interface groups
                    if (group.size() > 1) {
                        // printf("sd group: ");
                        // printVec<int>(group.size(), group.data());
                        sd_groups.push_back(group);
                    }
                }
            }

            // optional: remove repeated groups
            std::sort(sd_groups.begin(), sd_groups.end());
            sd_groups.erase(std::unique(sd_groups.begin(), sd_groups.end()), sd_groups.end());

            // now modify elem_sd_ind by combining certain subdomain pairs on TACS component
            // interfaces OR patch boundaries for (int inode)
            // first make a rowp, cols of the current elems in each subdomain
            int *elem_sd_cts = new int[num_subdomains];
            memset(elem_sd_cts, 0, num_subdomains * sizeof(int));
            int *elem_sd_rowp = new int[num_subdomains + 1];
            memset(elem_sd_rowp, 0, (num_subdomains + 1) * sizeof(int));
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int i_subdomain = elem_sd_ind[ielem];
                // start by just putting row cts in i_subdomain+1 entry (then later we'll add up to
                // get full rowp offsets)
                elem_sd_cts[i_subdomain]++;
            }

            for (int i_sd = 0; i_sd < num_subdomains; i_sd++) {
                elem_sd_rowp[i_sd + 1] = elem_sd_rowp[i_sd] + elem_sd_cts[i_sd];
            }

            int *elem_sd_cols = new int[num_elements];
            memset(elem_sd_cols, 0, num_elements * sizeof(int));
            // reuse cts to keep track of filling up rowp + cols
            memset(elem_sd_cts, 0, num_subdomains * sizeof(int));
            for (int ielem = 0; ielem < num_elements; ielem++) {
                int i_subdomain = elem_sd_ind[ielem];
                int offset = elem_sd_rowp[i_subdomain] + elem_sd_cts[i_subdomain];
                elem_sd_cols[offset] = ielem;
                elem_sd_cts[i_subdomain]++;
            }

            // the whole reason we did that above part is to avoid num_elements^2 operations
            // now use this sparse element list to more cheaply join together subdomain groups
            // printf("orig elem_sd_ind: ");
            // printVec<int>(num_elements, elem_sd_ind);
            int ct = 0;
            int n_wrap =
                sd_groups.size() * wrap_frac;  // how many subdomains we're going to wrap (ideal is
                                               // all of them, but allow fraction here to show in
                                               // paper that this is huge affect on runtime)
            for (auto group : sd_groups) {
                int first_sd_ind = group[0];
                if (ct >= n_wrap) continue;  // if more than n_wrap then we exit
                ct++;

                // for middle cases: adjust first_sd_ind in case it was changed previously
                int isd0 = group[0];
                int ielem0 = elem_sd_cols[elem_sd_rowp[isd0]];
                if (elem_sd_ind[ielem0] != first_sd_ind) first_sd_ind = elem_sd_ind[ielem0];

                // printf("sd group write: ");
                // printVec<int>(group.size(), group.data());
                for (auto isd : group) {
                    // now loop through elements in this subdomain and write them to the first sd
                    // ind
                    for (int jp = elem_sd_rowp[isd]; jp < elem_sd_rowp[isd + 1]; jp++) {
                        int ielem = elem_sd_cols[jp];
                        // printf("modify ielem %d, from isd %d to %d\n", ielem, elem_sd_ind[ielem],
                        //        first_sd_ind);
                        elem_sd_ind[ielem] = first_sd_ind;
                    }
                }
            }

            // printf("elem_sd_ind: ");
            // printVec<int>(num_elements, elem_sd_ind);

            // now get full # of unique subdomains
            // now get full # of unique subdomains
            std::vector<int> subdomains;
            subdomains.reserve(num_elements);

            for (int ielem = 0; ielem < num_elements; ielem++) {
                subdomains.push_back(elem_sd_ind[ielem]);
            }

            // make unique
            int num_subdomains_0 = num_subdomains;
            std::sort(subdomains.begin(), subdomains.end());
            subdomains.erase(std::unique(subdomains.begin(), subdomains.end()), subdomains.end());
            num_subdomains = subdomains.size();  // update subdomain number now

            // printf("subdomains %d: ", subdomains.size());
            // printVec<int>(subdomains.size(), subdomains.data());

            // now make the elem_sd_ind 0 to (new_num_subdomains-1)
            int *subdomain_iperm =
                new int[num_subdomains_0];  // inverse map : old_sd => new_reduced_sd
            memset(subdomain_iperm, -1, num_subdomains_0 * sizeof(int));

            for (int isd = 0; isd < num_subdomains; isd++) {
                int old_sd = subdomains[isd];
                // printf("isd %d => old_sd %d\n", isd, old_sd);
                subdomain_iperm[old_sd] = isd;
            }
            // printf("subdomain_iperm: ");
            // printVec<int>(num_subdomains_0, subdomain_iperm);

            for (int ielem = 0; ielem < num_elements; ielem++) {
                int old_sd = elem_sd_ind[ielem];
                int new_red_sd = subdomain_iperm[old_sd];
                elem_sd_ind[ielem] = new_red_sd;
            }

            // printf("elem_sd_ind: ");
            // printVec<int>(num_elements, elem_sd_ind);

            // TODO : reset node_sd_cols, node_sd_rowp, node_elem_ct pointers (delete)
            // for new set of subdomains, as we recompute them again later
        }

        // printf("printToVTK elemVec\n");
        auto h_vars = assembler.createVarsVec().createHostVec();
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars, elem_sd_ind,
                                                            "subdomains", "out/comp_sd.vtk");
        printToVTK_elemVec<int, ShellAssembler, HostVec<T>>(assembler, h_vars, debug_lelem_sd_ind,
                                                            "subdomains", "out/comp_lsd.vtk");

        // for visualization of exploded subdomain divisions (to show method in ppt slides and
        // thesis)
        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, LOWER_SKIN>(
            assembler, h_vars, "out/comp_lskin.vtk", num_subdomains, elem_sd_ind);
        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, UPPER_SKIN>(
            assembler, h_vars, "out/comp_uskin.vtk", num_subdomains, elem_sd_ind);
        explodedSubdomainsPrintToVTK<ShellAssembler, HostVec<T>, INT_STRUCT>(
            assembler, h_vars, "out/comp_intstruct.vtk", num_subdomains, elem_sd_ind);
        // printf("\tdone printToVTK elemVec");

        // -----------------------------------------
        // node -> subdomain incidence
        // -----------------------------------------
        node_elem_nnz = 0;
        node_elem_rowp = new int[num_nodes + 1];
        node_elem_ct = new int[num_nodes];
        std::memset(node_elem_rowp, 0, (num_nodes + 1) * sizeof(int));
        std::memset(node_elem_ct, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        int *temp_node_elem = new int[num_nodes];
        node_sd_cols = new int[node_elem_nnz];
        std::memset(temp_node_elem, 0, num_nodes * sizeof(int));
        std::memset(node_sd_cols, 0, node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int subdomain_ind = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;

        // -----------------------------------------
        // classify nodes
        // -----------------------------------------
        node_class_ind = new int[num_nodes];
        std::memset(node_class_ind, 0, num_nodes * sizeof(int));

        printf(
            "WARNING: for unstructured meshes, this node classification may fail at "
            "junctions.\n");

        node_nsd = new int[num_nodes];
        I_nnodes = 0, IE_nnodes = 0, IEV_nnodes = 0;
        Vc_nnodes = 0, V_nnodes = 0, lam_nnodes = 0;

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int nsd = static_cast<int>(node_sds.size());
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;
                I_nnodes++, IE_nnodes++, IEV_nnodes++;
            } else if (dirichlet_ind[inode] && track_dirichlet) {
                node_class_ind[inode] = INTERIOR;
                if (node_bndry_ind[inode] > 0) {
                    node_class_ind[inode] = DIRICHLET_EDGE;
                    // acts like an edge node, repeated twice
                    I_nnodes += nsd;
                    IE_nnodes += nsd;
                    IEV_nnodes += nsd;
                }
            } else if (nsd == 2 || node_wing_geom_ind[inode] == 1) {
                // TODO: this logic is not quite right for wing..
                node_class_ind[inode] = EDGE;
                lam_nnodes++;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = VERTEX;
                Vc_nnodes++;  //, lam_nnodes++;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }

        // print to VTK the nodal num subdomains
        T *h_soln = new T[num_nodes * block_dim];
        memset(h_soln, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_soln[6 * i] = node_nsd[i];
        }
        auto h_soln_debug = HostVec<T>(num_nodes * block_dim, h_soln);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_soln_debug, "out/comp_node_nsd.vtk");

        // print to VTK the IEV labels..
        T *h_soln2 = new T[num_nodes * block_dim];
        memset(h_soln2, 0.0, num_nodes * block_dim * sizeof(T));
        for (int i = 0; i < num_nodes; i++) {
            h_soln2[6 * i] = node_class_ind[i];
        }
        auto h_soln_debug2 = HostVec<T>(num_nodes * block_dim, h_soln2);
        printToVTK<ShellAssembler, HostVec<T>>(assembler, h_soln_debug2, "out/comp_node_class.vtk");
        // printf("done with classify nodes\n");

        // -----------------------------------------
        // build duplicated IEV nodal layout
        // -----------------------------------------
        IEV_sd_ptr = new int[num_subdomains + 1];
        IEV_sd_ind = new int[IEV_nnodes];
        IEV_nodes = new int[IEV_nnodes];

        std::memset(IEV_sd_ptr, 0, (num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[num_nodes];

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::memset(temp_completion, 0, num_nodes * sizeof(int));
            IEV_sd_ptr[i_subdomain + 1] = IEV_sd_ptr[i_subdomain];
            // printf("isd %d\n", i_subdomain);

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != i_subdomain) continue;

                int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];
                    if (temp_completion[gnode]) continue;

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = i_subdomain;
                    IEV_sd_ptr[i_subdomain + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;

        // printf("IEV_sd_ptr: ");
        // printVec<int>(num_subdomains + 1, IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_nnodes, IEV_sd_ind);
        // printf("IEV_nodes: ");
        // printVec<int>(IEV_nnodes, IEV_nodes);

        // -----------------------------------------
        // build IEV element connectivity
        // -----------------------------------------
        IEV_elem_conn = new int[num_elements * nodes_per_elem];
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int i_subdomain = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[i_subdomain]; jp < IEV_sd_ptr[i_subdomain + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                }
                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }

        // printf("IEV_ind %d\n", IEV_ind);
        // printf("IEV_nodes %d: ", IEV_nnodes);
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("IEV_conn: ");
        // printVec<int>(nodes_per_elem * num_elements, IEV_elem_conn);

        // for (int iev = 0; iev < IEV_nnodes; iev++) {
        //     int isd = IEV_sd_ind[iev];
        //     int gnode = IEV_nodes[iev];
        //     int iclass = node_class_ind[gnode];
        //     printf("iev %d, isd %d, gnode %d, class %d\n", iev, isd, gnode, iclass);
        // }

        // -----------------------------------------
        // IE and I nodal lists
        // -----------------------------------------
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];
        std::memset(IE_nodes, 0, IE_nnodes * sizeof(int));
        std::memset(I_nodes, 0, I_nnodes * sizeof(int));
        IE_interior = new bool[IE_nnodes];
        IE_general_edge = new bool[IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                I_nodes[I_ind++] = gnode;
            }
        }
        d_IE_interior = HostVec<bool>(IE_nnodes, IE_interior).createDeviceVec().getPtr();
        d_IE_general_edge = HostVec<bool>(IE_nnodes, IE_general_edge).createDeviceVec().getPtr();
        d_IE_nodes = HostVec<int>(IE_nnodes, IE_nodes).createDeviceVec().getPtr();

        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("I_nodes %d: ", I_nnodes);
        // printVec<int>(I_nnodes, I_nodes);

        // -----------------------------------------
        // build IEV sparsity from duplicated connectivity
        // -----------------------------------------
        // printf("build BSR data\n");
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);
        // printf("\tdone build BSR data\n");
        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;
        IEV_rows = new int[IEV_nnzb];
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            for (int jp = IEV_rowp[inode]; jp < IEV_rowp[inode + 1]; jp++) {
                IEV_rows[jp] = inode;
            }
        }
        // printf("IEV_rowp with nnzb %d: ", IEV_nnzb);
        // printVec<int>(IEV_nnodes + 1, IEV_rowp);
        // printf("IEV_cols: ");
        // printVec<int>(IEV_nnzb, IEV_cols);

        // -----------------------------------------
        // reduced rowp arrays
        // -----------------------------------------
        IE_rowp = new int[IE_nnodes + 1];
        I_rowp = new int[I_nnodes + 1];
        std::memset(IE_rowp, 0, (IE_nnodes + 1) * sizeof(int));
        std::memset(I_rowp, 0, (I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) I_rowp[I_row + 1] = I_rowp[I_row];
            if (typeIE_row) IE_rowp[IE_row + 1] = IE_rowp[IE_row];

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        I_nnzb = I_rowp[I_nnodes];
        IE_nnzb = IE_rowp[IE_nnodes];

        // printf("I_rowp nnzb %d: ", I_nnzb);
        // printVec<int>(I_nnodes + 1, I_rowp);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_rowp nnzb %d: ", IE_nnzb);
        // printVec<int>(IE_nnodes + 1, IE_rowp);

        IE_rows = new int[IE_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }

        // -----------------------------------------
        // IEV -> IE map
        // -----------------------------------------
        IEVtoIE_map = new int[IEV_nnodes];
        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        IEVtoIE_imap = new int[IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                IEVtoIE_imap[IE_ind] = inode;
                IEVtoIE_map[inode] = IE_ind++;
            }
        }

        // printf("IEVtoIE_map: ");
        // printVec<int>(IEV_nnodes, IEVtoIE_map);

        // put on device
        d_IEVtoIE_imap = HostVec<int>(IE_nnodes, IEVtoIE_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV -> I map
        // -----------------------------------------
        IEVtoI_map = new int[IEV_nnodes];
        IEVtoI_imap = new int[I_nnodes];
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IEVtoI_imap[I_ind] = inode;
                IEVtoI_map[inode] = I_ind++;
            }
        }

        // printf("IEVtoI_map: ");
        // printVec<int>(IEV_nnodes, IEVtoI_map);

        // put on device
        d_IEVtoI_imap = HostVec<int>(I_nnodes, IEVtoI_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // reduced column arrays
        // -----------------------------------------
        IE_cols = new int[IE_nnzb];
        I_cols = new int[I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < IEV_nnodes; row++) {
            int gnode_row = IEV_nodes[row];
            int class_row = node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = IEV_rowp[row]; jp < IEV_rowp[row + 1]; jp++) {
                int col = IEV_cols[jp];
                int gnode_col = IEV_nodes[col];
                int class_col = node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    I_cols[I_ind++] = IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    IE_cols[IE_ind++] = IEVtoIE_map[col];
                }
            }
        }

        // printf("I_cols: ");
        // printVec<int>(I_nnzb, I_cols);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();

        // -----------------------------------------
        // IEV -> Vc map (coarse non-repeated vertices)
        // -----------------------------------------

        // make a list of the reduced Vc_ind (takes global node and figures out it's Vc_ind)
        std::unordered_set<int> Vc_nodeset;
        for (int i = 0; i < IEV_nnodes; i++) {
            int gnode = IEV_nodes[i];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());
        // printf("Vc_nodes %d: ", Vc_nodes_vec.size());
        // printVec<int>(Vc_nodes_vec.size(), Vc_nodes_vec.data());

        int *Vc_inodes = new int[num_nodes];  // takes Vc global node => Vc red node
        memset(Vc_inodes, -1, num_nodes * sizeof(int));
        for (int i = 0; i < Vc_nodes_vec.size(); i++) {
            int j = Vc_nodes_vec[i];
            Vc_inodes[j] = i;
        }

        // printf("Vc_inodes: ");
        // printVec<int>(num_nodes, Vc_inodes);

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();

        // compute what fraction of nodes are wrapped around multi-patch junctions
        int nwrap_nodes = 0;
        for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
            int glob_node = Vc_nodes[ivc];
            int wing_node_class = node_wing_geom_ind[glob_node];
            if (wing_node_class == 0) {
                nwrap_nodes++;  // then on interior
            }
        }
        T wrap_node_frac = 1.0 * nwrap_nodes / Vc_nnodes;
        printf("frac of nodes on interior (for wrap subdomains): %d/%d = %.4f\n", nwrap_nodes,
               Vc_nnodes, wrap_node_frac);

        // printf("post move to device of Vc_nodes\n");

        // set VcToV_imap and IEVtoV_imap now
        VctoV_imap = new int[V_nnodes];
        std::memset(VctoV_imap, -1, V_nnodes * sizeof(int));
        IEVtoV_imap = new int[V_nnodes];
        std::memset(IEVtoV_imap, -1, V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < IEV_nnodes; inode++) {
            int gnode = IEV_nodes[inode];
            int node_class = node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                VctoV_imap[V_ind] = Vc_ind;
                IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }

        // printf("IEVtoV_imap %d: ", V_nnodes);
        // printVec<int>(V_nnodes, IEVtoV_imap);

        // put on device
        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();

        // Build all remaining maps/permutations:
        //  - jump operator ownership/sign maps

        // printf("compute jump operators\n");
        bool square_domain = compute_jump;  // should be false
        _compute_jump_operators(square_domain);
        // printf("allocate workspace\n");
        allocate_workspace();
        // printf("\tdone with allocate workspace\n");

        // ---------------------------------------------------
        // Save ORIGINAL nofill sparsity for IE and I
        // before fill-in / AMD reordering
        // ---------------------------------------------------
        IE_bsr_data = BsrData(IE_nnodes, block_dim, IE_nnzb, IE_rowp, IE_cols);
        IE_bsr_data.rows = IE_rows;
        IE_nofill_nnzb = IE_nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        // sparsity before the permutation
        // printf("\nIE SPARSITY BEFORE PERMUTATION\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("(");
        //     int grow = IE_nodes[IE_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
        //         int j = IE_cols[jp];
        //         int gcol = IE_nodes[IE_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");

        // printf("pre-perm IE_rowp: ");
        // printVec<int>(IE_nnodes + 1, IE_rowp);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("\n");

        I_bsr_data = BsrData(I_nnodes, block_dim, I_nnzb, I_rowp, I_cols);
        I_bsr_data.rows = I_rows;
        I_nofill_nnzb = I_nnzb;
        // printf("\tdone with setup wing subdomains\n");
    }

    void setup_coarse_structured_subdomains(const int nxe_, const int nye_, const int nxs_,
                                            const int nys_, const int nxs2_, const int nys2_,
                                            const bool close_hoop, int &coarse_num_elements,
                                            int &coarse_num_nodes, int &coarse_elem_nnz,
                                            int *&coarse_elem_ptr, int *&coarse_elem_conn,
                                            int *&coarse_elem_sd_ind) {
        // build a subdomain splitting for the coarse BDDC solver using two hierarchical subdomain
        // splittings on a structured mesh (like plate / cylinder)
        int *elem_sd_ind1 = new int[num_elements];
        int *elem_sd_ind2 = new int[num_elements];

        // calls two levels of refinement fine grid subdomain splittings first
        printf("first setup structured subdomains\n");
        this->setup_structured_subdomains(nxe_, nye_, nxs_, nys_, close_hoop);
        printf("\tdone with first setup structured subdomains\n");
        for (int i = 0; i < num_elements; i++) {
            elem_sd_ind1[i] = elem_sd_ind[i];
        }
        coarse_num_nodes = Vc_nnodes;
        coarse_num_elements = num_subdomains;
        // printf("coarse_num_nodes %d, coarse_num_elements %d\n", coarse_num_nodes,
        //        coarse_num_elements);
        // printf("elem_sd_ind1: ");
        // printVec<int>(num_elements, elem_sd_ind1);

        // build coarse element connectivity from elem_sd_ind1 (first refinement level)
        int *coarse_elem_cts = new int[coarse_num_elements];
        memset(coarse_elem_cts, 0, coarse_num_elements * sizeof(int));
        for (int i = 0; i < num_elements; i++) {
            int i_subdomain = elem_sd_ind1[i];
            for (int local_node = 0; local_node < nodes_per_elem; local_node++) {
                int gnode = elem_conn[nodes_per_elem * i + local_node];
                int node_class = node_class_ind[gnode];
                if (node_class == VERTEX) {
                    coarse_elem_cts[i_subdomain]++;
                }
            }
        }
        // printf("coarse_elem_cts: ");
        // printVec<int>(coarse_num_elements, coarse_elem_cts);
        coarse_elem_ptr = new int[coarse_num_elements + 1];
        memset(coarse_elem_ptr, 0, (coarse_num_elements + 1) * sizeof(int));
        for (int ic = 0; ic < coarse_num_elements; ic++) {
            coarse_elem_ptr[ic + 1] = coarse_elem_ptr[ic] + coarse_elem_cts[ic];
        }
        // printf("coarse_elem_ptr (nelems %d): ", coarse_num_elements);
        // printVec<int>(coarse_num_elements + 1, coarse_elem_ptr);
        coarse_elem_nnz = coarse_elem_ptr[coarse_num_elements];
        printf("celem_nnz %d\n", coarse_elem_nnz);
        coarse_elem_conn = new int[coarse_elem_nnz];
        memset(coarse_elem_conn, 0, coarse_elem_nnz * sizeof(int));
        // to help track current progress in nz pattern filling
        memset(coarse_elem_cts, 0, coarse_num_elements * sizeof(int));
        // reverse map of global => reduced Vc nodes
        // same as Vc_node_imap (but needed here before called in setup_matrix_sparsity)
        int *Vc_imap = new int[num_nodes];
        memset(Vc_imap, -1, num_nodes * sizeof(int));
        for (int vnode = 0; vnode < Vc_nnodes; vnode++) {
            int glob_node = Vc_nodes[vnode];
            Vc_imap[glob_node] = vnode;
        }

        for (int i = 0; i < num_elements; i++) {
            int i_subdomain = elem_sd_ind1[i];
            for (int local_node = 0; local_node < nodes_per_elem; local_node++) {
                int gnode = elem_conn[nodes_per_elem * i + local_node];
                int node_class = node_class_ind[gnode];
                if (node_class == VERTEX) {
                    int ivc = Vc_imap[gnode];  // reduced coarse node
                    int offset = coarse_elem_cts[i_subdomain] + coarse_elem_ptr[i_subdomain];
                    // printf("fill conn[%d] = %d on i_subdomain %d (out of %d nnz)\n", offset, ivc,
                    //        i_subdomain, coarse_elem_nnz);
                    coarse_elem_conn[offset] = ivc;
                    coarse_elem_cts[i_subdomain]++;
                }
            }
        }

        // DEBUG
        // printf("coarse_elem_conn (nnz %d): ", coarse_elem_nnz);
        // printVec<int>(coarse_elem_nnz, coarse_elem_conn);
        // for (int i_subdomain = 0; i_subdomain < this->num_subdomains; i_subdomain++) {
        //     printf("subdomain %d (elem_conn): ", i_subdomain);
        //     for (int elemp = coarse_elem_ptr[i_subdomain]; elemp < coarse_elem_ptr[i_subdomain +
        //     1];
        //          elemp++) {
        //         int ivc = coarse_elem_conn[elemp];
        //         printf("%d ", ivc);
        //     }
        //     printf("\n");
        // }

        // // build coarse xpts
        // T *h_xpts = d_xpts.createHostVec().getPtr();
        // T *h_coarse_xpts = new T[3 * coarse_num_nodes];
        // for (int ivc = 0; ivc < coarse_num_nodes; ivc++) {
        //     int glob_node = Vc_nodes[ivc];
        //     for (int idim = 0; idim < 3; idim++) {
        //         h_coarse_xpts[3 * ivc + idim] = h_xpts[3 * glob_node + idim];
        //     }
        // }
        // d_coarse_xpts = HostVec<T>(3 * coarse_num_nodes, h_coarse_xpts).createDeviceVec();

        // then run greater refined subdomain splitting
        // printf("second setup structured subdomains\n");
        this->setup_structured_subdomains(nxe_, nye_, nxs2_, nys2_, close_hoop);
        // printf("\tdone with second setup structured subdomains\n");
        for (int i = 0; i < num_elements; i++) {
            elem_sd_ind2[i] = elem_sd_ind[i];
        }
        // printf("elem_sd_ind2: ");
        // printVec<int>(num_elements, elem_sd_ind2);

        // then build coarse_elem_sd_ind
        coarse_elem_sd_ind = new int[coarse_num_elements];
        // int coarse_num_subdomains = this->num_subdomains;
        // coarse_fine_sd_map = new int[coarse_num_subdomains];
        for (int i = 0; i < num_elements; i++) {
            int celem = elem_sd_ind1[i];
            int c_subdomain = elem_sd_ind2[i];
            // all elements in a subdomain have same value (hierarchical, so don't need unique
            // checks here) fine to overwrite (from fine grid)
            coarse_elem_sd_ind[celem] = c_subdomain;
            // coarse_fine_sd_map[c_subdomain] = celem;
        }
        // printf("coarse_elem_sd_ind: ");
        // printVec<int>(coarse_num_elements, coarse_elem_sd_ind);
    }

    void setup_coarse_tacs_component_subdomains(const int nxse_, const int nyse_, const int nxse2_,
                                                const int nyse2_, const int MOD_WRAPAROUND,
                                                const T wrap_frac, const bool compute_jump,
                                                int &coarse_num_elements, int &coarse_num_nodes,
                                                int &coarse_elem_nnz, int *&coarse_elem_ptr,
                                                int *&coarse_elem_conn, int *&coarse_elem_sd_ind) {
        // build a subdomain splitting for the coarse BDDC solver using two hierarchical subdomain
        // splittings on a structured mesh (like plate / cylinder)
        int *elem_sd_ind1 = new int[num_elements];
        int *elem_sd_ind2 = new int[num_elements];

        // calls two levels of refinement fine grid subdomain splittings first
        printf("first setup tacs component subdomains\n");
        this->setup_tacs_component_subdomains(nxse_, nyse_, MOD_WRAPAROUND, wrap_frac,
                                              compute_jump);
        printf("\tdone with first setup tacs component subdomains\n");
        for (int i = 0; i < num_elements; i++) {
            elem_sd_ind1[i] = elem_sd_ind[i];
        }
        coarse_num_nodes = Vc_nnodes;
        coarse_num_elements = num_subdomains;
        printf("coarse_num_nodes %d, coarse_num_elements %d\n", coarse_num_nodes,
               coarse_num_elements);
        printf("elem_sd_ind1: ");
        printVec<int>(num_elements, elem_sd_ind1);

        // build coarse element connectivity from elem_sd_ind1 (first refinement level)
        int *coarse_elem_cts = new int[coarse_num_elements];
        memset(coarse_elem_cts, 0, coarse_num_elements * sizeof(int));
        for (int i = 0; i < num_elements; i++) {
            int i_subdomain = elem_sd_ind1[i];
            for (int local_node = 0; local_node < nodes_per_elem; local_node++) {
                int gnode = elem_conn[nodes_per_elem * i + local_node];
                int node_class = node_class_ind[gnode];
                if (node_class == VERTEX) {
                    coarse_elem_cts[i_subdomain]++;
                }
            }
        }
        printf("coarse_elem_cts: ");
        printVec<int>(coarse_num_elements, coarse_elem_cts);
        coarse_elem_ptr = new int[coarse_num_elements + 1];
        memset(coarse_elem_ptr, 0, (coarse_num_elements + 1) * sizeof(int));
        for (int ic = 0; ic < coarse_num_elements; ic++) {
            coarse_elem_ptr[ic + 1] = coarse_elem_ptr[ic] + coarse_elem_cts[ic];
        }
        printf("coarse_elem_ptr (nelems %d): ", coarse_num_elements);
        printVec<int>(coarse_num_elements + 1, coarse_elem_ptr);
        coarse_elem_nnz = coarse_elem_ptr[coarse_num_elements];
        printf("coarse_elem_nnz %d\n", coarse_elem_nnz);
        coarse_elem_conn = new int[coarse_elem_nnz];
        memset(coarse_elem_conn, 0, coarse_elem_nnz * sizeof(int));
        // to help track current progress in nz pattern filling
        memset(coarse_elem_cts, 0, coarse_num_elements * sizeof(int));
        // reverse map of global => reduced Vc nodes
        // same as Vc_node_imap (but needed here before called in setup_matrix_sparsity)
        int *Vc_imap = new int[num_nodes];
        memset(Vc_imap, -1, num_nodes * sizeof(int));
        for (int vnode = 0; vnode < Vc_nnodes; vnode++) {
            int glob_node = Vc_nodes[vnode];
            Vc_imap[glob_node] = vnode;
        }

        for (int i = 0; i < num_elements; i++) {
            int i_subdomain = elem_sd_ind1[i];
            for (int local_node = 0; local_node < nodes_per_elem; local_node++) {
                int gnode = elem_conn[nodes_per_elem * i + local_node];
                int node_class = node_class_ind[gnode];
                if (node_class == VERTEX) {
                    int ivc = Vc_imap[gnode];  // reduced coarse node
                    int offset = coarse_elem_cts[i_subdomain] + coarse_elem_ptr[i_subdomain];
                    // printf("fill conn[%d] = %d (out of %d nnz)\n", offset, ivc, celem_nnz);
                    coarse_elem_conn[offset] = ivc;
                    coarse_elem_cts[i_subdomain]++;
                }
            }
        }
        printf("coarse_elem_conn (nnz %d): ", coarse_elem_nnz);
        printVec<int>(coarse_elem_nnz, coarse_elem_conn);

        // // build coarse xpts
        // T *h_xpts = d_xpts.createHostVec().getPtr();
        // T *h_coarse_xpts = new T[3 * coarse_num_nodes];
        // for (int ivc = 0; ivc < coarse_num_nodes; ivc++) {
        //     int glob_node = Vc_nodes[ivc];
        //     for (int idim = 0; idim < 3; idim++) {
        //         h_coarse_xpts[3 * ivc + idim] = h_xpts[3 * glob_node + idim];
        //     }
        // }
        // d_coarse_xpts = HostVec<T>(3 * coarse_num_nodes, h_coarse_xpts).createDeviceVec();

        // then run greater refined subdomain splitting
        printf("second setup tacs component subdomains\n");
        this->setup_tacs_component_subdomains(nxse2_, nyse2_, MOD_WRAPAROUND, wrap_frac,
                                              compute_jump);
        printf("\tdone with second setup tacs component subdomains\n");
        for (int i = 0; i < num_elements; i++) {
            elem_sd_ind2[i] = elem_sd_ind[i];
        }
        printf("elem_sd_ind2: ");
        printVec<int>(num_elements, elem_sd_ind2);

        // then build coarse_elem_sd_ind
        coarse_elem_sd_ind = new int[coarse_num_elements];
        for (int i = 0; i < num_elements; i++) {
            int celem = elem_sd_ind1[i];
            int c_subdomain = elem_sd_ind2[i];
            // all elements in a subdomain have same value (hierarchical, so don't need unique
            // checks here) fine to overwrite (from fine grid)
            coarse_elem_sd_ind[celem] = c_subdomain;
        }
        printf("coarse_elem_sd_ind: ");
        printVec<int>(coarse_num_elements, coarse_elem_sd_ind);
    }

    void setup_matrix_sparsity() {
        // USER must call this routine..
        printf(
            "NOTE : FETI-DP doesn't support permutations yet on subdomains.. TBD later on "
            "that\n");
        printf(
            "\tJust does full fillin currently of each matrix used for inner linear "
            "solves\n");

        // do fillin of IE and I matrices (later also do coarse matrix)
        IE_rowp = IE_bsr_data.rowp, IE_cols = IE_bsr_data.cols, IE_nnzb = IE_bsr_data.nnzb;
        IE_perm = IE_bsr_data.perm, IE_iperm = IE_bsr_data.iperm;

        // host
        I_rowp = I_bsr_data.rowp, I_cols = I_bsr_data.cols, I_nnzb = I_bsr_data.nnzb;
        I_perm = I_bsr_data.perm, I_iperm = I_bsr_data.iperm;

        // printf("post-perm IE_rowp: ");
        // printVec<int>(IE_nnodes + 1, IE_rowp);
        // printf("IE_perm: ");
        // printVec<int>(IE_nnodes, IE_perm);
        // printf("IE_iperm: ");
        // printVec<int>(IE_nnodes, IE_iperm);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        // sparsity after the permutation
        // printf("\nIE SPARSITY AFTER PERMUTATION\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("(");
        //     int grow = IE_nodes[IE_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
        //         int j = IE_cols[jp];
        //         int gcol = IE_nodes[IE_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");

        // printf("SETUP MATRIX SPARSITY 1\n");

        // recompute rows after potential fillin
        delete[] IE_rows, I_rows;
        IE_rows = new int[IE_nnzb];
        I_rows = new int[I_nnzb];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
                IE_rows[jp] = inode;
            }
        }
        for (int inode = 0; inode < I_nnodes; inode++) {
            for (int jp = I_rowp[inode]; jp < I_rowp[inode + 1]; jp++) {
                I_rows[jp] = inode;
            }
        }
        IE_bsr_data.rows = IE_rows;
        I_bsr_data.rows = I_rows;

        // printf("SETUP MATRIX SPARSITY 2\n");

        // now move sparsity to the device
        // and no fillin for IEV matrix cause it isn't used for linear solves
        IEV_bsr_data.rows = IEV_rows;
        d_IEV_bsr_data = IEV_bsr_data.createDeviceBsrData();
        d_IEV_vals = DeviceVec<T>(block_dim2 * IEV_nnzb);
        kmat_IEV = new BsrMatType(d_IEV_bsr_data, d_IEV_vals);

        d_IE_bsr_data = IE_bsr_data.createDeviceBsrData();
        d_IE_vals = DeviceVec<T>(block_dim2 * IE_nnzb);
        kmat_IE = new BsrMatType(d_IE_bsr_data, d_IE_vals);

        d_I_bsr_data = I_bsr_data.createDeviceBsrData();
        d_I_vals = DeviceVec<T>(block_dim2 * I_nnzb);
        kmat_I = new BsrMatType(d_I_bsr_data, d_I_vals);

        d_IE_perm = d_IE_bsr_data.perm, d_IE_iperm = d_IE_bsr_data.iperm;
        d_I_perm = d_I_bsr_data.perm, d_I_iperm = d_I_bsr_data.iperm;

        // printf("SETUP MATRIX SPARSITY 3\n");

        // -----------------------------------------
        // IEV => IE kmat block copy map
        // -----------------------------------------

        kmat_IEnofill_map = new int[IE_nofill_nnzb];
        kmat_IEtoIEV_map = new int[IE_nofill_nnzb];
        memset(kmat_IEnofill_map, -1, IE_nofill_nnzb * sizeof(int));
        memset(kmat_IEtoIEV_map, -1, IE_nofill_nnzb * sizeof(int));
        int nofill_ind = 0;
        for (int i = 0; i < IE_nnodes; i++) {
            int i_perm = IE_iperm[i];  // VIS to solve order
            for (int jp = IE_rowp[i_perm]; jp < IE_rowp[i_perm + 1]; jp++) {
                int j_perm = IE_cols[jp];
                int j = IE_perm[j_perm];  // solve to VIS order
                // find equivalent nz block of IEV rowp
                int i_IEV = IEVtoIE_imap[i];
                int j_IEV = IEVtoIE_imap[j];
                bool found = false;
                for (int kp = IEV_rowp[i_IEV]; kp < IEV_rowp[i_IEV + 1]; kp++) {
                    int k = IEV_cols[kp];
                    if (k == j_IEV) {
                        kmat_IEnofill_map[nofill_ind] = jp;
                        kmat_IEtoIEV_map[nofill_ind] = kp;

                        int gr_IE = IE_nodes[i], gc_IE = IE_nodes[j];
                        int gr_IEV = IEV_nodes[i_IEV], gc_IEV = IEV_nodes[j_IEV];
                        // printf("nofill ind %d, IE (%d,%d) block %d, IEV (%d,%d) block
                        // %d\n",
                        //        nofill_ind, gr_IE, gc_IE, jp, gr_IEV, gc_IEV, kp);
                        nofill_ind++;
                        found = true;
                    }
                }
                // would need more advanced check here since fillin means some blocks in IE
                // won't exist in IEV
                // if (!found) {
                //     printf("IE block %d not found in IEV matrix\n", jp);
                // }
            }
        }

        // printf("SETUP MATRIX SPARSITY 4\n");

        d_kmat_IEtoIEV_map =
            HostVec<int>(IE_nofill_nnzb, kmat_IEtoIEV_map).createDeviceVec().getPtr();
        d_kmat_IEnofill_map =
            HostVec<int>(IE_nofill_nnzb, kmat_IEnofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV => I kmat block copy map
        // -----------------------------------------

        // printf("SETUP MATRIX SPARSITY 5\n");

        kmat_Inofill_map = new int[I_nofill_nnzb];
        kmat_ItoIEV_map = new int[I_nofill_nnzb];
        memset(kmat_Inofill_map, -1, I_nofill_nnzb * sizeof(int));
        memset(kmat_ItoIEV_map, -1, I_nofill_nnzb * sizeof(int));
        nofill_ind = 0;
        for (int i = 0; i < I_nnodes; i++) {
            int i_perm = I_iperm[i];  // VIS to solve order
            for (int jp = I_rowp[i_perm]; jp < I_rowp[i_perm + 1]; jp++) {
                int j_perm = I_cols[jp];
                int j = I_perm[j_perm];
                // find equivalent nz block of IEV rowp
                int i_IEV = IEVtoI_imap[i];
                int j_IEV = IEVtoI_imap[j];
                bool found = false;
                for (int kp = IEV_rowp[i_IEV]; kp < IEV_rowp[i_IEV + 1]; kp++) {
                    int k = IEV_cols[kp];
                    if (k == j_IEV) {
                        kmat_Inofill_map[nofill_ind] = jp;
                        kmat_ItoIEV_map[nofill_ind] = kp;
                        nofill_ind++;
                        found = true;
                    }
                }
                // would need more advanced check here since fillin means some blocks in IE
                // won't exist in IEV
                // if (!found) {
                //     printf("IE block %d not found in IEV matrix\n", jp);
                // }
            }
        }

        // printf("SETUP MATRIX SPARSITY 6\n");

        d_kmat_ItoIEV_map = HostVec<int>(I_nofill_nnzb, kmat_ItoIEV_map).createDeviceVec().getPtr();
        d_kmat_Inofill_map =
            HostVec<int>(I_nofill_nnzb, kmat_Inofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // get S_VV matrix sparsity / nonzero pattern (nofill first)
        // -----------------------------------------

        // printf("SETUP MATRIX SPARSITY 7\n");

        // reverse map of global => reduced Vc nodes
        Vc_node_imap = new int[num_nodes];
        memset(Vc_node_imap, -1, num_nodes * sizeof(int));
        for (int vnode = 0; vnode < Vc_nnodes; vnode++) {
            int glob_node = Vc_nodes[vnode];
            Vc_node_imap[glob_node] = vnode;
        }

        // printf("SETUP MATRIX SPARSITY 8\n");

        // printf("Vc_node_imap: ");
        // printVec<int>(num_nodes, Vc_node_imap);
        // printf("Vc_nnodes %d\n", Vc_nnodes);

        // printf("elem_sd_ind: ");
        // printVec<int>(num_elements, elem_sd_ind);

        // build unique adjacency per coarse row
        std::vector<std::unordered_set<int>> Svv_adj(Vc_nnodes);

        for (int i_subdomain = 0; i_subdomain < num_subdomains; i_subdomain++) {
            std::unordered_set<int> sd_Vc_nodeset;

            for (int ielem = 0; ielem < num_elements; ielem++) {
                int *local_nodes = &elem_conn[nodes_per_elem * ielem];
                int j_subdomain = elem_sd_ind[ielem];
                if (i_subdomain != j_subdomain) continue;

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_nodes[lnode];
                    int node_class = node_class_ind[gnode];
                    if (node_class == VERTEX) {
                        int vnode = Vc_node_imap[gnode];
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

        // printf("SETUP MATRIX SPARSITY 9\n");

        // row counts from unique adjacency
        int *Svv_rowcts = new int[Vc_nnodes];
        memset(Svv_rowcts, 0, Vc_nnodes * sizeof(int));
        for (int i = 0; i < Vc_nnodes; i++) {
            Svv_rowcts[i] = static_cast<int>(Svv_adj[i].size());
        }

        // fill rowp
        Svv_rowp = new int[Vc_nnodes + 1];
        memset(Svv_rowp, 0, (Vc_nnodes + 1) * sizeof(int));
        for (int i = 0; i < Vc_nnodes; i++) {
            Svv_rowp[i + 1] = Svv_rowp[i] + Svv_rowcts[i];
        }
        Svv_nnzb = Svv_rowp[Vc_nnodes];

        // fill cols
        Svv_cols = new int[Svv_nnzb];
        memset(Svv_cols, 0, Svv_nnzb * sizeof(int));

        // printf("SETUP MATRIX SPARSITY 10\n");

        for (int i = 0; i < Vc_nnodes; i++) {
            int jp = Svv_rowp[i];
            for (int j : Svv_adj[i]) {
                Svv_cols[jp++] = j;
            }

            // optional but recommended: sort each row
            std::sort(&Svv_cols[Svv_rowp[i]], &Svv_cols[Svv_rowp[i + 1]]);
        }

        // printf("Svv_rowp with nnzb %d: ", Svv_nnzb);
        // printVec<int>(Vc_nnodes + 1, Svv_rowp);
        // printf("Svv_cols: ");
        // printVec<int>(Svv_nnzb, Svv_cols);

        // now go back through and put cols in
        // use temp_Svv_fill to help keep track of putting nonzero entries in the Svv
        // sparsity int *temp_Svv_fill = new int[Vc_nnodes]; memset(temp_Svv_fill, 0,
        // Vc_nnodes * sizeof(int)); Svv_cols = new int[Svv_nnzb]; memset(Svv_cols, 0,
        // Svv_nnzb * sizeof(int)); for (int i_subdomain = 0; i_subdomain < num_subdomains;
        // i_subdomain++) {
        //     std::unordered_set<int> sd_Vc_nodeset;

        //     for (int ielem = 0; ielem < num_elements; ielem++) {
        //         int *local_nodes = &elem_conn[nodes_per_elem * ielem];
        //         int j_subdomain = elem_sd_ind[ielem];
        //         if (i_subdomain != j_subdomain) continue;

        //         for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
        //             int gnode = local_nodes[lnode];
        //             int node_class = node_class_ind[gnode];
        //             if (node_class == VERTEX) {
        //                 int vnode = Vc_node_imap[gnode];
        //                 sd_Vc_nodeset.insert(vnode);
        //             }
        //         }
        //     }

        //     // convert to vector
        //     std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());
        //     // see prevoius block on why each subdomain "macro-element" is fully dense in
        //     macro-elem
        //     // but no connections outside macro-elems
        //     for (int i : sd_Vc_nodes) {
        //         for (int j : sd_Vc_nodes) {
        //             Svv_cols[temp_Svv_fill[i]++] = j;
        //         }
        //     }
        // }

        // printf("Svv_rowp with nnzb %d: ", Svv_nnzb);
        // printVec<int>(Vc_nnodes + 1, Svv_rowp);
        // printf("Svv_cols: ");
        // printVec<int>(Svv_nnzb, Svv_cols);

        Svv_rows = new int[Svv_nnzb];
        for (int i = 0; i < Vc_nnodes; i++) {
            for (int jp = Svv_rowp[i]; jp < Svv_rowp[i + 1]; jp++) {
                Svv_rows[jp] = i;
            }
        }

        // printf("SETUP MATRIX SPARSITY 11\n");

        // -----------------------------------------
        // build Svv matrix sparsity and do fillin
        // -----------------------------------------
        // small matrix typically

        // Svv_bsr_data = BsrData(Svv_nodes, block_dim, Vc_nnzb, Svv_rowp, Svv_cols);
        Svv_bsr_data = BsrData(Vc_nnodes, block_dim, Svv_nnzb, Svv_rowp, Svv_cols);
        Svv_bsr_data.rows = Svv_rows;
        Svv_nofill_nnzb = Svv_nnzb;
        SVV_perm = Svv_bsr_data.perm, SVV_iperm = Svv_bsr_data.iperm;

        // sparsity before the permutation
        // printf("\nSvv SPARSITY BEFORE PERMUTATION\n");
        // for (int inode = 0; inode < Vc_nnodes; inode++) {
        //     printf("(");
        //     int grow = Vc_nodes[SVV_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = Svv_rowp[inode]; jp < Svv_rowp[inode + 1]; jp++) {
        //         int j = Svv_cols[jp];
        //         int gcol = Vc_nodes[SVV_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");
    }

    void setup_coarse_matrix_sparsity() {
        Svv_rowp = Svv_bsr_data.rowp, Svv_cols = Svv_bsr_data.cols, Svv_nnzb = Svv_bsr_data.nnzb;
        SVV_perm = Svv_bsr_data.perm, SVV_iperm = Svv_bsr_data.iperm;

        // recompute rows after potential fillin
        delete[] Svv_rows;
        Svv_rows = new int[Svv_nnzb];
        for (int i = 0; i < Vc_nnodes; i++) {
            for (int jp = Svv_rowp[i]; jp < Svv_rowp[i + 1]; jp++) {
                Svv_rows[jp] = i;
            }
        }
        Svv_bsr_data.rows = Svv_rows;
        this->d_coarse_vars = DeviceVec<T>(block_dim * Vc_nnodes);

        d_Svv_bsr_data = Svv_bsr_data.createDeviceBsrData();
        d_Svv_vals = DeviceVec<T>(block_dim2 * Svv_nnzb);
        S_VV = new BsrMatType(d_Svv_bsr_data, d_Svv_vals);
        d_SVV_perm = d_Svv_bsr_data.perm, d_SVV_iperm = d_Svv_bsr_data.iperm;

        // -----------------------------------------
        // IEV => V kmat block copy map (for A_{VV} copy in S_{VV})
        // -----------------------------------------

        Svv_copy_nnzb = 0;
        std::vector<int> Svv_IEV_copyBlocks;
        std::vector<int> Svv_Vc_copyBlocks;
        for (int IEV_row = 0; IEV_row < IEV_nnodes; IEV_row++) {
            int glob_row = IEV_nodes[IEV_row];
            int row_class = node_class_ind[glob_row];
            if (row_class != VERTEX) continue;
            int Vc_row = Vc_node_imap[glob_row];
            int Vc_rowperm = SVV_iperm[Vc_row];

            for (int jp = IEV_rowp[IEV_row]; jp < IEV_rowp[IEV_row + 1]; jp++) {
                int IEV_col = IEV_cols[jp];
                int glob_col = IEV_nodes[IEV_col];
                int col_class = node_class_ind[glob_col];
                if (col_class == VERTEX) {
                    int Vc_col = Vc_node_imap[glob_col];
                    for (int kp = Svv_rowp[Vc_rowperm]; kp < Svv_rowp[Vc_rowperm + 1]; kp++) {
                        int k_perm = Svv_cols[kp];
                        int k = SVV_perm[k_perm];
                        if (k == Vc_col) {
                            Svv_IEV_copyBlocks.push_back(jp);
                            Svv_Vc_copyBlocks.push_back(kp);
                            Svv_copy_nnzb++;
                        }
                    }
                }
            }
        }
        d_Svv_IEV_copyBlocks =
            HostVec<int>(Svv_copy_nnzb, Svv_IEV_copyBlocks.data()).createDeviceVec().getPtr();
        d_Svv_Vc_copyBlocks =
            HostVec<int>(Svv_copy_nnzb, Svv_Vc_copyBlocks.data()).createDeviceVec().getPtr();

        // CONSTRUCT coarse Schur complement mat-invmat-mat maps
        for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            IEVset_nnzb[k] = 0;
            IEVtoSVV_nnzb[k] = 0;
            d_IEVset_blocks[k] = nullptr;
            d_IEVout_blocks[k] = nullptr;
            d_IEVtoSVV_blocks[k] = nullptr;
        }

        std::vector<int> IEVset_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];
        std::vector<int> IEVout_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];
        std::vector<int> IEVtoSVV_blocks_host[MAX_NUM_VERTEX_PER_SUBDOMAIN];

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::vector<int> sd_iev_vertex_blocks;
            std::vector<int> sd_vc_nodes;

            for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                int gnode = IEV_nodes[jp];
                if (node_class_ind[gnode] == VERTEX) {
                    sd_iev_vertex_blocks.push_back(jp);

                    int vc_node = -1;
                    for (int j = 0; j < Vc_nnodes; j++) {
                        if (Vc_nodes[j] == gnode) {
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
                const int vc_row_perm = SVV_iperm[vc_row];

                IEVset_blocks_host[k].push_back(iev_block);

                for (int kk = 0; kk < nsv; kk++) {
                    const int iev_block2 = sd_iev_vertex_blocks[kk];
                    const int vc_col = sd_vc_nodes[kk];

                    int svv_block = -1;
                    for (int jp = Svv_rowp[vc_row_perm]; jp < Svv_rowp[vc_row_perm + 1]; jp++) {
                        int m_perm = Svv_cols[jp];
                        int m = SVV_perm[m_perm];
                        if (m == vc_col) {
                            svv_block = jp;
                            break;
                        }
                    }

                    if (svv_block < 0) {
                        printf(
                            "ERROR: could not find global Svv block for subdomain %d, row %d, col "
                            "%d\n",
                            isd, vc_row, vc_col);
                        exit(-1);
                    }

                    IEVout_blocks_host[k].push_back(iev_block2);
                    IEVtoSVV_blocks_host[k].push_back(svv_block);
                }
            }
        }

        // ------------------------------------------------------------
        // DEBUG: validate regular Schur complement maps on host
        // ------------------------------------------------------------
        // printf("DEBUG regular coarse Schur maps\n");
        for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            // printf("DEBUG regular slot %d: set nnzb host %d, out nnzb host %d, svv nnzb host
            // %d\n",
            //        k, (int)IEVset_blocks_host[k].size(), (int)IEVout_blocks_host[k].size(),
            //        (int)IEVtoSVV_blocks_host[k].size());

            if ((int)IEVout_blocks_host[k].size() != (int)IEVtoSVV_blocks_host[k].size()) {
                printf(
                    "ERROR: regular slot %d mismatch before device copy: out size %d != svv size "
                    "%d\n",
                    k, (int)IEVout_blocks_host[k].size(), (int)IEVtoSVV_blocks_host[k].size());
                exit(-1);
            }

            for (int p = 0; p < (int)IEVset_blocks_host[k].size(); p++) {
                int iev_set_block = IEVset_blocks_host[k][p];
                if (iev_set_block < 0 || iev_set_block >= IEV_nnodes) {
                    printf(
                        "ERROR: invalid regular set block at slot %d, p %d: iev block %d not in "
                        "[0,%d)\n",
                        k, p, iev_set_block, IEV_nnodes);
                    exit(-1);
                }
            }

            for (int p = 0; p < (int)IEVout_blocks_host[k].size(); p++) {
                int iev_out_block = IEVout_blocks_host[k][p];
                int svv_block = IEVtoSVV_blocks_host[k][p];

                if (iev_out_block < 0 || iev_out_block >= IEV_nnodes) {
                    printf(
                        "ERROR: invalid regular out block at slot %d, p %d: iev block %d not in "
                        "[0,%d)\n",
                        k, p, iev_out_block, IEV_nnodes);
                    exit(-1);
                }

                if (svv_block < 0 || svv_block >= Svv_nnzb) {
                    printf(
                        "ERROR: invalid regular svv block at slot %d, p %d: svv block %d not in "
                        "[0,%d)\n",
                        k, p, svv_block, Svv_nnzb);
                    exit(-1);
                }

                int row = Svv_rows[svv_block];
                if (row < 0 || row >= Vc_nnodes) {
                    printf(
                        "ERROR: invalid regular svv row decode at slot %d, p %d: block %d -> row "
                        "%d\n",
                        k, p, svv_block, row);
                    exit(-1);
                }

                if (!(Svv_rowp[row] <= svv_block && svv_block < Svv_rowp[row + 1])) {
                    printf(
                        "ERROR: invalid regular svv storage location at slot %d, p %d: "
                        "block %d not in decoded row %d range [%d,%d)\n",
                        k, p, svv_block, row, Svv_rowp[row], Svv_rowp[row + 1]);
                    exit(-1);
                }

                int col = Svv_cols[svv_block];
                if (col < 0 || col >= Vc_nnodes) {
                    printf(
                        "ERROR: invalid regular svv col decode at slot %d, p %d: block %d -> col "
                        "%d\n",
                        k, p, svv_block, col);
                    exit(-1);
                }
            }

            // if (!IEVout_blocks_host[k].empty()) {
            //     printf("  first few regular host blocks for slot %d:\n", k);
            //     for (int p = 0; p < (int)IEVout_blocks_host[k].size() && p < 12; p++) {
            //         printf("    p %d: iev %d, svv %d\n", p, IEVout_blocks_host[k][p],
            //                IEVtoSVV_blocks_host[k][p]);
            //     }
            // }
        }

        for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            IEVset_nnzb[k] = static_cast<int>(IEVset_blocks_host[k].size());
            IEVtoSVV_nnzb[k] = static_cast<int>(IEVtoSVV_blocks_host[k].size());

            if ((int)IEVout_blocks_host[k].size() != IEVtoSVV_nnzb[k]) {
                printf("ERROR: slot %d mismatch: IEVout size %d != IEVtoSVV size %d\n", k,
                       (int)IEVout_blocks_host[k].size(), IEVtoSVV_nnzb[k]);
                exit(-1);
            }

            if (IEVset_nnzb[k] > 0) {
                d_IEVset_blocks[k] = HostVec<int>(IEVset_nnzb[k], IEVset_blocks_host[k].data())
                                         .createDeviceVec()
                                         .getPtr();
            }

            if (IEVtoSVV_nnzb[k] > 0) {
                d_IEVout_blocks[k] = HostVec<int>(IEVtoSVV_nnzb[k], IEVout_blocks_host[k].data())
                                         .createDeviceVec()
                                         .getPtr();

                d_IEVtoSVV_blocks[k] =
                    HostVec<int>(IEVtoSVV_nnzb[k], IEVtoSVV_blocks_host[k].data())
                        .createDeviceVec()
                        .getPtr();
            }
        }

        // ------------------------------------------------------------
        // DEBUG: validate regular Schur complement maps on device
        // ------------------------------------------------------------
        for (int k = 0; k < MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            // printf(
            //     "DEBUG regular slot %d final: IEVset_nnzb %d, IEVtoSVV_nnzb %d, d_set %p, d_out "
            //     "%p, d_svv %p\n",
            //     k, IEVset_nnzb[k], IEVtoSVV_nnzb[k], (void *)d_IEVset_blocks[k],
            //     (void *)d_IEVout_blocks[k], (void *)d_IEVtoSVV_blocks[k]);

            if (IEVset_nnzb[k] > 0) {
                std::vector<int> h_dbg_set(IEVset_nnzb[k]);
                CHECK_CUDA(cudaMemcpy(h_dbg_set.data(), d_IEVset_blocks[k],
                                      IEVset_nnzb[k] * sizeof(int), cudaMemcpyDeviceToHost));

                for (int p = 0; p < IEVset_nnzb[k]; p++) {
                    if (h_dbg_set[p] < 0 || h_dbg_set[p] >= IEV_nnodes) {
                        printf(
                            "ERROR: device regular set block invalid at slot %d, p %d: %d not in "
                            "[0,%d)\n",
                            k, p, h_dbg_set[p], IEV_nnodes);
                        exit(-1);
                    }
                }
            }

            if (IEVtoSVV_nnzb[k] > 0) {
                std::vector<int> h_dbg_out(IEVtoSVV_nnzb[k]);
                std::vector<int> h_dbg_svv(IEVtoSVV_nnzb[k]);

                CHECK_CUDA(cudaMemcpy(h_dbg_out.data(), d_IEVout_blocks[k],
                                      IEVtoSVV_nnzb[k] * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_dbg_svv.data(), d_IEVtoSVV_blocks[k],
                                      IEVtoSVV_nnzb[k] * sizeof(int), cudaMemcpyDeviceToHost));

                for (int p = 0; p < IEVtoSVV_nnzb[k]; p++) {
                    if (h_dbg_out[p] < 0 || h_dbg_out[p] >= IEV_nnodes) {
                        printf(
                            "ERROR: device regular out block invalid at slot %d, p %d: %d not in "
                            "[0,%d)\n",
                            k, p, h_dbg_out[p], IEV_nnodes);
                        exit(-1);
                    }
                    if (h_dbg_svv[p] < 0 || h_dbg_svv[p] >= Svv_nnzb) {
                        printf(
                            "ERROR: device regular svv block invalid at slot %d, p %d: %d not in "
                            "[0,%d)\n",
                            k, p, h_dbg_svv[p], Svv_nnzb);
                        exit(-1);
                    }
                }

                // printf("  first few regular device blocks for slot %d:\n", k);
                // for (int p = 0; p < IEVtoSVV_nnzb[k] && p < 12; p++) {
                //     printf("    p %d: iev %d, svv %d\n", p, h_dbg_out[p], h_dbg_svv[p]);
                // }
            }
        }

        // compute the BC indices needed for kmat_IEV
        auto d_bcs = assembler.getBCs();
        int n_orig_bcs = d_bcs.getSize();
        int *h_bcs = d_bcs.createHostVec().getPtr();

        std::vector<int> IEV_bcs_vec;
        for (int IEV_node = 0; IEV_node < IEV_nnodes; IEV_node++) {
            int gnode = IEV_nodes[IEV_node];
            for (int ibc = 0; ibc < n_orig_bcs; ibc++) {
                int bc = h_bcs[ibc];
                int bc_node = bc / block_dim;
                int bc_dof = bc % block_dim;
                if (bc_node == gnode) {
                    int IEV_dof = block_dim * IEV_node + bc_dof;
                    IEV_bcs_vec.push_back(IEV_dof);
                }
            }
        }

        d_IEV_bcs = HostVec<int>(IEV_bcs_vec.size(), IEV_bcs_vec.data()).createDeviceVec();
    }

    void setup_MLIEV_coarse_matrix_sparsity(BsrMatType *S_VV_MLIEV_, const int *coarse_IEV_nodes,
                                            const int *coarse_IEV_sd_ptr,
                                            const int *coarse_IEV_sd_ind,
                                            const int *coarse_elem_sd_ind) {
        // setup multilevel IEV coarse matrix sparsity (for 3-levels and coarse matrix with
        // IEV-splitting)

        this->S_VV_MLIEV = S_VV_MLIEV_;
        this->d_Svv_MLIEV_bsr_data = this->S_VV_MLIEV->getBsrData();
        this->Svv_MLIEV_bsr_data = this->d_Svv_MLIEV_bsr_data.createHostBsrData();
        d_Svv_MLIEV_vals = this->S_VV_MLIEV->getVec();

        // used in multilevel BDDC (since coarse solver uses Svv matrix with different sparsity)
        int coarse_IEV_nnodes = this->d_Svv_MLIEV_bsr_data.nnodes;
        this->Svv_MLIEV_rowp = this->Svv_MLIEV_bsr_data.rowp;
        this->Svv_MLIEV_cols = this->Svv_MLIEV_bsr_data.cols;
        this->Svv_MLIEV_nnzb = this->Svv_MLIEV_bsr_data.nnzb;
        this->SVV_MLIEV_perm = this->Svv_MLIEV_bsr_data.perm;
        this->SVV_MLIEV_iperm = this->Svv_MLIEV_bsr_data.iperm;

        // -------------------------------------------------------------------------
        // Build matched fine-IEV <-> coarse-IEV pairs.
        //
        // Not every fine IEV node has a coarse IEV match.
        // Multiple fine IEV nodes may map to the same coarse IEV node.
        // -------------------------------------------------------------------------
        std::vector<int> fine_match_iev;
        std::vector<int> coarse_match_iev;

        for (int IEV_ind = 0; IEV_ind < this->IEV_nnodes; IEV_ind++) {
            int fine_glob_node = this->IEV_nodes[IEV_ind];
            int fine_subdomain = this->IEV_sd_ind[IEV_ind];
            int coarse_subdomain = coarse_elem_sd_ind[fine_subdomain];

            for (int coarse_IEV_ind = coarse_IEV_sd_ptr[coarse_subdomain];
                 coarse_IEV_ind < coarse_IEV_sd_ptr[coarse_subdomain + 1]; coarse_IEV_ind++) {
                int coarse_glob_node = this->Vc_nodes[coarse_IEV_nodes[coarse_IEV_ind]];

                if (fine_glob_node == coarse_glob_node) {
                    fine_match_iev.push_back(IEV_ind);
                    coarse_match_iev.push_back(coarse_IEV_ind);
                    break;
                }
            }
        }

        // lookup only for matched fine IEV nodes
        std::unordered_map<int, int> fine_to_coarse_match_lookup;
        for (int match_ind = 0; match_ind < static_cast<int>(fine_match_iev.size()); match_ind++) {
            fine_to_coarse_match_lookup[fine_match_iev[match_ind]] = coarse_match_iev[match_ind];
        }

        // recompute rows after potential fillin
        delete[] this->Svv_MLIEV_rows;
        this->Svv_MLIEV_rows = new int[this->Svv_MLIEV_nnzb];
        for (int i = 0; i < coarse_IEV_nnodes; i++) {
            for (int jp = this->Svv_MLIEV_rowp[i]; jp < this->Svv_MLIEV_rowp[i + 1]; jp++) {
                this->Svv_MLIEV_rows[jp] = i;
            }
        }
        this->Svv_MLIEV_bsr_data.rows = this->Svv_MLIEV_rows;

        // -------------------------------------------------------------------------
        // Decode coarse S_VV_MLIEV block map using coarse IEV row/col indices directly.
        // -------------------------------------------------------------------------
        auto pair_key = [](int i, int j) -> long long {
            return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
        };

        std::unordered_map<long long, int> ml_block_lookup;

        for (int inode = 0; inode < coarse_IEV_nnodes; inode++) {
            int coarse_IEV_row = this->SVV_MLIEV_perm[inode];

            for (int jp = this->Svv_MLIEV_rowp[inode]; jp < this->Svv_MLIEV_rowp[inode + 1]; jp++) {
                int j = this->Svv_MLIEV_cols[jp];
                int coarse_IEV_col = this->SVV_MLIEV_perm[j];
                ml_block_lookup[pair_key(coarse_IEV_row, coarse_IEV_col)] = jp;
            }
        }

        this->d_SVV_MLIEV_perm = this->d_Svv_MLIEV_bsr_data.perm;
        this->d_SVV_MLIEV_iperm = this->d_Svv_MLIEV_bsr_data.iperm;

        // -----------------------------------------
        // IEV => coarse-IEV kmat block copy map
        // (for A_VV copy into S_VV^{MLIEV})
        // -----------------------------------------
        this->Svv_MLIEV_copy_nnzb = 0;
        std::vector<int> Svv_IEV_copyBlocks;
        std::vector<int> Svv_Vc_copyBlocks;

        for (int IEV_row = 0; IEV_row < this->IEV_nnodes; IEV_row++) {
            int glob_row = this->IEV_nodes[IEV_row];
            if (this->node_class_ind[glob_row] != VERTEX) continue;

            auto row_it = fine_to_coarse_match_lookup.find(IEV_row);
            if (row_it == fine_to_coarse_match_lookup.end()) continue;
            int coarse_IEV_row = row_it->second;

            for (int jp = this->IEV_rowp[IEV_row]; jp < this->IEV_rowp[IEV_row + 1]; jp++) {
                int IEV_col = this->IEV_cols[jp];
                int glob_col = this->IEV_nodes[IEV_col];
                if (this->node_class_ind[glob_col] != VERTEX) continue;

                auto col_it = fine_to_coarse_match_lookup.find(IEV_col);
                if (col_it == fine_to_coarse_match_lookup.end()) continue;
                int coarse_IEV_col = col_it->second;

                auto it = ml_block_lookup.find(pair_key(coarse_IEV_row, coarse_IEV_col));
                if (it != ml_block_lookup.end()) {
                    Svv_IEV_copyBlocks.push_back(jp);
                    Svv_Vc_copyBlocks.push_back(it->second);
                    this->Svv_MLIEV_copy_nnzb++;
                }
            }
        }

        this->d_Svv_IEV_copyBlocks =
            HostVec<int>(this->Svv_MLIEV_copy_nnzb, Svv_IEV_copyBlocks.data())
                .createDeviceVec()
                .getPtr();
        this->d_Svv_Vc_copyBlocks =
            HostVec<int>(this->Svv_MLIEV_copy_nnzb, Svv_Vc_copyBlocks.data())
                .createDeviceVec()
                .getPtr();

        // CONSTRUCT coarse Schur complement mat-invmat-mat maps
        for (int k = 0; k < this->MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            this->ML_IEVset_nnzb[k] = 0;
            this->ML_IEVtoSVV_nnzb[k] = 0;
            this->d_ML_IEVset_blocks[k] = nullptr;
            this->d_ML_IEVout_blocks[k] = nullptr;
            this->d_ML_IEVtoSVV_blocks[k] = nullptr;
        }

        std::vector<std::vector<int>> ML_IEVset_blocks_host(this->MAX_NUM_VERTEX_PER_SUBDOMAIN);
        std::vector<std::vector<int>> ML_IEVout_blocks_host(this->MAX_NUM_VERTEX_PER_SUBDOMAIN);
        std::vector<std::vector<int>> ML_IEVtoSVV_blocks_host(this->MAX_NUM_VERTEX_PER_SUBDOMAIN);

        for (int isd = 0; isd < this->num_subdomains; isd++) {
            // EXACTLY like old method: collect all fine-grid local vertex slots first
            std::vector<int>
                sd_iev_vertex_blocks;  // repeated fine IEV vertex blocks on this subdomain
            std::vector<int>
                sd_coarse_iev_blocks;  // coarse IEV match for that fine local slot, or -1

            for (int jp = this->IEV_sd_ptr[isd]; jp < this->IEV_sd_ptr[isd + 1]; jp++) {
                int gnode = this->IEV_nodes[jp];
                if (this->node_class_ind[gnode] != VERTEX) continue;

                sd_iev_vertex_blocks.push_back(jp);

                auto it_match = fine_to_coarse_match_lookup.find(jp);
                if (it_match == fine_to_coarse_match_lookup.end()) {
                    sd_coarse_iev_blocks.push_back(-1);  // unmatched fine vertex slot
                } else {
                    sd_coarse_iev_blocks.push_back(it_match->second);
                }
            }

            const int nsv = static_cast<int>(sd_iev_vertex_blocks.size());
            if (nsv == 0) continue;

            // keep k as the ORIGINAL fine local vertex slot
            for (int k = 0; k < nsv; k++) {
                const int iev_block_row = sd_iev_vertex_blocks[k];
                const int coarse_IEV_row = sd_coarse_iev_blocks[k];

                // if this local fine-vertex slot has no coarse match, skip it
                if (coarse_IEV_row < 0) continue;

                // one basis injection for this fine-subdomain-local slot
                ML_IEVset_blocks_host[k].push_back(iev_block_row);

                // only couple to other local fine-vertex slots on THIS SAME fine subdomain
                // that also have a coarse match
                for (int kk = 0; kk < nsv; kk++) {
                    const int iev_block_col = sd_iev_vertex_blocks[kk];
                    const int coarse_IEV_col = sd_coarse_iev_blocks[kk];

                    if (coarse_IEV_col < 0) continue;

                    auto it = ml_block_lookup.find(pair_key(coarse_IEV_row, coarse_IEV_col));
                    if (it == ml_block_lookup.end()) {
                        printf(
                            "ERROR: could not find coarse Svv_MLIEV block for fine sd %d, "
                            "fine local slots (%d,%d), coarse row %d, coarse col %d\n",
                            isd, k, kk, coarse_IEV_row, coarse_IEV_col);
                        exit(-1);
                    }

                    ML_IEVout_blocks_host[k].push_back(iev_block_col);
                    ML_IEVtoSVV_blocks_host[k].push_back(it->second);
                }
            }
        }

        // -------------------------------------------------------------------------
        // DEBUG CHECKS:
        // 1) every Svv block index must be a valid BSR storage location in S_VV_MLIEV
        // 2) every out-block index must be a valid fine IEV block id
        // -------------------------------------------------------------------------
        for (int k = 0; k < this->MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            if (ML_IEVout_blocks_host[k].size() != ML_IEVtoSVV_blocks_host[k].size()) {
                printf("ERROR: slot %d mismatch before device copy: out size %d != svv size %d\n",
                       k, (int)ML_IEVout_blocks_host[k].size(),
                       (int)ML_IEVtoSVV_blocks_host[k].size());
                exit(-1);
            }

            for (int p = 0; p < (int)ML_IEVtoSVV_blocks_host[k].size(); p++) {
                int svv_block = ML_IEVtoSVV_blocks_host[k][p];
                int out_block = ML_IEVout_blocks_host[k][p];

                if (svv_block < 0 || svv_block >= this->Svv_MLIEV_nnzb) {
                    printf("ERROR: invalid Svv block index at slot %d, p %d: %d not in [0,%d)\n", k,
                           p, svv_block, this->Svv_MLIEV_nnzb);
                    exit(-1);
                }

                if (out_block < 0 || out_block >= this->IEV_nnodes) {
                    printf(
                        "ERROR: invalid fine IEV out block index at slot %d, p %d: %d not in "
                        "[0,%d)\n",
                        k, p, out_block, this->IEV_nnodes);
                    exit(-1);
                }

                // extra debug: verify the BSR block index really belongs to a valid row/col entry
                int bsr_row = this->Svv_MLIEV_rows[svv_block];
                if (bsr_row < 0 || bsr_row >= coarse_IEV_nnodes) {
                    printf("ERROR: invalid Svv row decode at slot %d, p %d: block %d -> row %d\n",
                           k, p, svv_block, bsr_row);
                    exit(-1);
                }

                if (!(this->Svv_MLIEV_rowp[bsr_row] <= svv_block &&
                      svv_block < this->Svv_MLIEV_rowp[bsr_row + 1])) {
                    printf(
                        "ERROR: Svv block %d at slot %d, p %d is not inside decoded row %d range "
                        "[%d,%d)\n",
                        svv_block, k, p, bsr_row, this->Svv_MLIEV_rowp[bsr_row],
                        this->Svv_MLIEV_rowp[bsr_row + 1]);
                    exit(-1);
                }

                int bsr_col = this->Svv_MLIEV_cols[svv_block];
                if (bsr_col < 0 || bsr_col >= coarse_IEV_nnodes) {
                    printf("ERROR: invalid Svv col decode at slot %d, p %d: block %d -> col %d\n",
                           k, p, svv_block, bsr_col);
                    exit(-1);
                }
            }

            for (int p = 0; p < (int)ML_IEVset_blocks_host[k].size(); p++) {
                int set_block = ML_IEVset_blocks_host[k][p];
                if (set_block < 0 || set_block >= this->IEV_nnodes) {
                    printf(
                        "ERROR: invalid fine IEV set block index at slot %d, p %d: %d not in "
                        "[0,%d)\n",
                        k, p, set_block, this->IEV_nnodes);
                    exit(-1);
                }
            }
        }

        for (int k = 0; k < this->MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            this->ML_IEVset_nnzb[k] = static_cast<int>(ML_IEVset_blocks_host[k].size());
            this->ML_IEVtoSVV_nnzb[k] = static_cast<int>(ML_IEVtoSVV_blocks_host[k].size());

            if ((int)ML_IEVout_blocks_host[k].size() != this->ML_IEVtoSVV_nnzb[k]) {
                printf("ERROR: slot %d mismatch: ML_IEVout size %d != ML_IEVtoSVV size %d\n", k,
                       (int)ML_IEVout_blocks_host[k].size(), this->ML_IEVtoSVV_nnzb[k]);
                exit(-1);
            }

            if (this->ML_IEVset_nnzb[k] > 0) {
                this->d_ML_IEVset_blocks[k] =
                    HostVec<int>(this->ML_IEVset_nnzb[k], ML_IEVset_blocks_host[k].data())
                        .createDeviceVec()
                        .getPtr();
            }

            if (this->ML_IEVtoSVV_nnzb[k] > 0) {
                this->d_ML_IEVout_blocks[k] =
                    HostVec<int>(this->ML_IEVtoSVV_nnzb[k], ML_IEVout_blocks_host[k].data())
                        .createDeviceVec()
                        .getPtr();

                this->d_ML_IEVtoSVV_blocks[k] =
                    HostVec<int>(this->ML_IEVtoSVV_nnzb[k], ML_IEVtoSVV_blocks_host[k].data())
                        .createDeviceVec()
                        .getPtr();
            }
        }

        // ------------------------------------------------------------
        // DEBUG: validate ML Schur complement maps before device copy
        // ------------------------------------------------------------
        for (int k = 0; k < this->MAX_NUM_VERTEX_PER_SUBDOMAIN; k++) {
            printf("DEBUG ML slot %d: set nnzb host %d, out nnzb host %d, svv nnzb host %d\n", k,
                   (int)ML_IEVset_blocks_host[k].size(), (int)ML_IEVout_blocks_host[k].size(),
                   (int)ML_IEVtoSVV_blocks_host[k].size());

            if (ML_IEVout_blocks_host[k].size() != ML_IEVtoSVV_blocks_host[k].size()) {
                printf("ERROR: ML slot %d mismatch: out size %d != svv size %d\n", k,
                       (int)ML_IEVout_blocks_host[k].size(),
                       (int)ML_IEVtoSVV_blocks_host[k].size());
                exit(-1);
            }

            for (int p = 0; p < (int)ML_IEVset_blocks_host[k].size(); p++) {
                int iev_set_block = ML_IEVset_blocks_host[k][p];
                if (iev_set_block < 0 || iev_set_block >= this->IEV_nnodes) {
                    printf(
                        "ERROR: invalid ML set block at slot %d, p %d: iev block %d not in "
                        "[0,%d)\n",
                        k, p, iev_set_block, this->IEV_nnodes);
                    exit(-1);
                }
            }

            for (int p = 0; p < (int)ML_IEVout_blocks_host[k].size(); p++) {
                int iev_out_block = ML_IEVout_blocks_host[k][p];
                int svv_block = ML_IEVtoSVV_blocks_host[k][p];

                if (iev_out_block < 0 || iev_out_block >= this->IEV_nnodes) {
                    printf(
                        "ERROR: invalid ML out block at slot %d, p %d: iev block %d not in "
                        "[0,%d)\n",
                        k, p, iev_out_block, this->IEV_nnodes);
                    exit(-1);
                }

                if (svv_block < 0 || svv_block >= this->Svv_MLIEV_nnzb) {
                    printf(
                        "ERROR: invalid ML svv block at slot %d, p %d: svv block %d not in "
                        "[0,%d)\n",
                        k, p, svv_block, this->Svv_MLIEV_nnzb);
                    exit(-1);
                }

                int row = this->Svv_MLIEV_rows[svv_block];
                if (row < 0 || row >= coarse_IEV_nnodes) {
                    printf(
                        "ERROR: invalid ML svv row decode at slot %d, p %d: block %d -> row %d\n",
                        k, p, svv_block, row);
                    exit(-1);
                }

                if (!(this->Svv_MLIEV_rowp[row] <= svv_block &&
                      svv_block < this->Svv_MLIEV_rowp[row + 1])) {
                    printf(
                        "ERROR: invalid ML svv storage location at slot %d, p %d: "
                        "block %d not in decoded row %d range [%d,%d)\n",
                        k, p, svv_block, row, this->Svv_MLIEV_rowp[row],
                        this->Svv_MLIEV_rowp[row + 1]);
                    exit(-1);
                }

                int col = this->Svv_MLIEV_cols[svv_block];
                if (col < 0 || col >= coarse_IEV_nnodes) {
                    printf(
                        "ERROR: invalid ML svv col decode at slot %d, p %d: "
                        "block %d -> col %d\n",
                        k, p, svv_block, col);
                    exit(-1);
                }
            }
        }
    }

    void assemble_subdomains() {
        // TODO:
        // build workspace vectors / allocate reduced matrices if needed

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(d_vars, d_IEV_vars, block_dim, 1.0, 0.0);

        // // DEBUG
        // T *h_xpts = d_xpts.createHostVec().getPtr();
        // printf("orig_xpts: ");
        // printVec<T>(3 * num_nodes, h_xpts);
        // T *h_IEV_xpts = d_IEV_xpts.createHostVec().getPtr();
        // // printf("IEV_xpts: ");
        // // printVec<T>(3 * IEV_nnodes, h_IEV_xpts);
        // printf("\nIEV_xpts\n");
        // for (int inode = 0; inode < IEV_nnodes; inode++) {
        //     printf("IEV %d, gnode %d, ", inode, IEV_nodes[inode]);
        //     printVec<T>(3, &h_IEV_xpts[3 * inode]);
        // }

        // TODO: kmat_IEV must be allocated before this
        kmat_IEV->zeroValues();

        // int cols_per_elem = (Quadrature::num_quad_pts <= 4) ? 24 : 9;
        const int elems_per_block = 1;
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);
        dim3 block(num_quad_pts, cols_per_elem, 1);
        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nblocks = (num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        k_add_jacobian_fast<T, elems_per_block, ShellAssembler, Data, Vec_, BsrMatType>
            <<<grid, block>>>(IEV_nnodes, num_elements, cols_per_elem, d_elem_components,
                              d_IEV_elem_conn, d_IEV_elem_conn, d_IEV_xpts, d_IEV_vars, d_compData,
                              *kmat_IEV);

        // printf("kmat_IEV vals on GPU[0]: ");
        // int nnz = IEV_nnzb * block_dim2;
        // T *h_kmat_IEV_vals = kmat_IEV->createHostVec().getPtr();
        // for (int i = 0; i < IEV_nnzb; i++) {
        //     T *h_block = &h_kmat_IEV_vals[block_dim2 * i];
        //     printf("kmat_IEV: block[%d]:\n", i);

        //     for (int j = 0; j < block_dim; j++) {
        //         printVec<T>(block_dim, &h_block[block_dim * j]);
        //     }
        // }

        // apply bcs to the kmat_IEV (before copying to IE and I subdomain matrices)
        kmat_IEV->apply_bcs(d_IEV_bcs);

        // copy entries from kmat_IEV to kmat_IE, kmat_I and S_VV matrices
        kmat_IE->zeroValues();
        kmat_I->zeroValues();
        copyKmat_IEVtoIE();
        copyKmat_IEVtoI();
    }

    void assemble_coarse_problem() {
        // now also compute the Schur complement inverse term in S_VV
        if (print_timing) {
            _assemble_coarse_problem_timing();
        } else {
            // non timed version
            S_VV->zeroValues();
            if (S_VV_MLIEV) S_VV_MLIEV->zeroValues();  // for 3+ level BDDC
            // printf("copyKmat to Svv\n");
            copyKmat_IEVtoSvv();
            // printf("compute Svv inverse term\n");
            computeSvvInverseTerm();
            CHECK_CUDA(cudaDeviceSynchronize());
            // printf("\tdone with Svv inverse term\n");
        }

        // bool debug = true;
        // if (debug) {
        //     printf("S_VV vals on GPU[0] with nnzb=%d: \n", Svv_nofill_nnzb);
        //     T *h_Svv_vals = d_Svv_vals.createHostVec().getPtr();
        //     // int *Svv_nofill_map =
        //     //     DeviceVec<int>(Svv_nofill_nnzb, d_Svv_Vc_copyBlocks).createHostVec().getPtr();
        //     for (int i = 0; i < Svv_nofill_nnzb; i++) {
        //         // int i2 = Svv_nofill_map[i];
        //         int row = Svv_rows[i], col = Svv_cols[i];
        //         T *h_block = &h_Svv_vals[block_dim2 * i];
        //         printf("S_VV: block[%d] (%d,%d):\n", i, row, col);
        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }
    void _assemble_coarse_problem_timing() {
        cudaEvent_t s1, e1, s2, e2;
        float t1 = 0.0f, t2 = 0.0f;

        if (print_timing) {
            cudaEventCreate(&s1);
            cudaEventCreate(&e1);
            cudaEventCreate(&s2);
            cudaEventCreate(&e2);
        }

        // now also compute the Schur complement inverse term in S_VV
        S_VV->zeroValues();

        if (print_timing) cudaEventRecord(s1);
        copyKmat_IEVtoSvv();
        if (print_timing) {
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t1, s1, e1);
        }

        if (print_timing) cudaEventRecord(s2);
        computeSvvInverseTerm();
        if (print_timing) {
            cudaEventRecord(e2);
            cudaEventSynchronize(e2);
            cudaEventElapsedTime(&t2, s2, e2);
            CHECK_CUDA(cudaDeviceSynchronize());

            printf("\tcopyKmat_IEVtoSvv: %.6f ms\n", t1);
            printf("\tcomputeSvvInverseTerm: %.6f ms\n", t2);

            cudaEventDestroy(s1);
            cudaEventDestroy(e1);
            cudaEventDestroy(s2);
            cudaEventDestroy(e2);
        }
    }

    template <class LoadMagnitude, int elems_per_block = 8>
    void add_subdomain_fext(const LoadMagnitude &load, T load_mag, T x_inplane_frac = 0.0) {
        fext_IEV.zeroValues();

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);

        dim3 block(num_quad_pts, elems_per_block);
        dim3 grid(num_elements);

        k_add_fext_fast<T, elems_per_block, ShellAssembler, Data, LoadMagnitude, Vec_>
            <<<grid, block>>>(num_elements, load, d_elem_components, d_IEV_elem_conn,
                              d_IEV_elem_conn, d_IEV_xpts, d_compData, load_mag, fext_IEV,
                              x_inplane_frac);

        // CHECK_CUDA(cudaDeviceSynchronize());

        fext_IEV.apply_bcs(d_IEV_bcs);
        fext_IEV.copyValuesTo(res_IEV);  // for linear problems

        // T *h_fext_IEV = fext_IEV.createHostVec().getPtr();
        // printf("h_fext_IEV: \n");
        // for (int iIEV = 0; iIEV < IEV_nnodes; iIEV++) {
        //     int iglob = IEV_nodes[iIEV];
        //     printf("iIEV %d, glob node %d: ", iIEV, iglob);
        //     for (int idof = 2; idof < 5; idof++) {
        //         int IEV_dof = block_dim * iIEV + idof;
        //         printf("%.6e,", h_fext_IEV[IEV_dof]);
        //     }
        //     printf("\n");
        // }
    }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           BaseSolver *coarseSolver_) {
        subdomainIESolver = subdomainIESolver_;
        subdomainISolver = subdomainISolver_;
        coarseSolver = coarseSolver_;
    }

    void set_global_rhs(DeviceVec<T> &rhs) {
        // const bool scaled = true;
        // addVec_globalToIEV<scaled>(rhs, fext_IEV, block_dim, 1.0, 0.0);
        // fext_IEV.apply_bcs(d_IEV_bcs);
    }

    void set_IEV_linear_rhs(DeviceVec<T> &vars) {
        // set fext_IEV back into res_IEV

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(vars, d_IEV_vars, block_dim, 1.0, 0.0);
        fext_IEV.copyValuesTo(res_IEV);
    }

    void set_IEV_adjoint_rhs(DeviceVec<T> &vars, DeviceVec<T> &adj_rhs_IEV, T a = 1.0) {
        // res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV

        // printf("set_IEV_residual\n");
        adj_rhs_IEV.apply_bcs(d_IEV_bcs);

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(vars, d_IEV_vars, block_dim, 1.0, 0.0);
        CHECK_CUBLAS(
            cublasDscal(cublasHandle, block_dim * IEV_nnodes, &a, adj_rhs_IEV.getPtr(), 1));
        adj_rhs_IEV.copyValuesTo(res_IEV);
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, DeviceVec<T> vars) {
        // res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV

        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(vars, d_IEV_vars, block_dim, 1.0, 0.0);

        fint_IEV.zeroValues();

        dim3 block(num_quad_pts, elems_per_block);
        dim3 grid(num_elements);

        k_add_residual_fast<T, elems_per_block, ShellAssembler, Data, Vec_>
            <<<grid, block>>>(IEV_nnodes, num_elements, d_elem_components, d_IEV_elem_conn,
                              d_IEV_elem_conn, d_IEV_xpts, d_IEV_vars, d_compData, fint_IEV);
        fint_IEV.apply_bcs(d_IEV_bcs);

        T a = -lambdaI;
        CHECK_CUBLAS(cublasDscal(cublasHandle, block_dim * IEV_nnodes, &a, fint_IEV.getPtr(), 1));

        fint_IEV.copyValuesTo(res_IEV);
        a = lambdaE;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, block_dim * IEV_nnodes, &a, fext_IEV.getPtr(), 1,
                                 res_IEV.getPtr(), 1));
    }

    void get_lam_rhs(DeviceVec<T> &lam_rhs) {
        // rhs = + fine term - coarse correction
        lam_rhs.zeroValues();

        // this is the standard RHS for linear systems global to IEV

        // ---------------------------------
        // fine term:
        //   f_IE = restricted external load on IE (or residual)
        //   u_IE = A_IE^{-1} f_IE
        //   lam_rhs += B * u_E
        //   also build repeated IEV copy of u_IE for coarse rhs
        // ---------------------------------
        printf("FineFetidp :: pre-synchronize\n");
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("FineFetidp::get_lam_rhs - pre solveSubdomainIE\n");

        addVecIEVtoIE(res_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        CHECK_CUDA(cudaDeviceSynchronize());
        printf("FineFetidp::get_lam_rhs - post solveSubdomainIE\n");
        T *h_u_IE = u_IE.createHostVec().getPtr();
        int nvals = u_IE.getSize();
        printf("FineFetidp::get_lam_rhs - h_u_IE\n");
        printVec<T>(nvals, h_u_IE);

        // build repeated IE/V representation from full IE solution
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        // only edge/interface part contributes to lambda rhs
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_rhs, 1.0, 0.0);

        // ---------------------------------
        // coarse rhs:
        //   g_V = f_V - A_{V,IE} u_IE
        // using repeated IEV representation
        // ---------------------------------
        addVecIEVtoVc(res_IEV, f_V, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        // T *h_fc = f_V.createHostVec().getPtr();
        // printf("h_fc:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fc[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // ---------------------------------
        // coarse correction:
        //   u_V = S_VV^{-1} g_V
        //   rhs_IE = A_{IE,V} u_V
        //   corr_IE = A_IE^{-1} rhs_IE
        //   lam_rhs -= B * corr_E
        // ---------------------------------
        // // temp debug
        // f_V.zeroValues();
        // f_V.add_value(2, 1.0);

        // T *h_fc2 = f_V.createHostVec().getPtr();
        // printf("h_fc (ei,i=2):\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fc2[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // // check matrix-vec multiply
        // sparseMatVec(*S_VV, f_V, 1.0, 0.0, u_V);
        // T *h_rV = u_V.createHostVec().getPtr();
        // u_V.zeroValues();
        // printf("h_rV (S_VV*ei):\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_rV[vc_dof]);
        //     }
        //     printf("\n");
        // }

        solveCoarse(f_V, u_V);

        // T *h_uc = u_V.createHostVec().getPtr();
        // printf("h_uc:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_uc[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // // check residual
        // sparseMatVec(*S_VV, u_V, -1.0, 1.0, f_V);
        // T *h_fv_res = f_V.createHostVec().getPtr();
        // printf("h_fv_res:\n");
        // for (int ivc = 0; ivc < Vc_nnodes; ivc++) {
        //     int iglob = Vc_nodes[ivc];
        //     printf("ivc %d, glob node %d: ", ivc, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int vc_dof = block_dim * ivc + idof;
        //         printf("%.6e,", h_fv_res[vc_dof]);
        //     }
        //     printf("\n");
        // }

        // addVecVctoIEV only writes selected entries, so clear first
        // u_IEV.zeroValues();
        addVecVctoIEV(u_V, u_IEV, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_rhs, -1.0, 1.0);

        // print lambda rhs
        // T *h_lam_rhs = lam_rhs.createHostVec().getPtr();
        // printf("h_lam_rhs:\n");
        // for (int ilam = 0; ilam < lam_nnodes; ilam++) {
        //     int iglob = lam_nodes[ilam];
        //     printf("ilam %d, glob node %d: ", ilam, iglob);
        //     for (int idof = 0; idof < block_dim; idof++) {
        //         int lam_dof = block_dim * ilam + idof;
        //         printf("%.6e,", h_lam_rhs[lam_dof]);
        //     }
        //     printf("\n");
        // }
    }

    void mat_vec(const DeviceVec<T> &lam_in, DeviceVec<T> &lam_out) {
        lam_out.zeroValues();

        addVecLamtoIE(lam_in, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        u_IEV.zeroValues();
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 1.0);
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_out, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 0.0);

        solveCoarse(f_V, u_V);

        u_IEV.zeroValues();
        addVecVctoIEV(u_V, u_IEV, 1.0, 1.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);
        zeroInteriorIE(u_IE);
        addVecIEtoLam(u_IE, lam_out, 1.0, 1.0);
    }

    bool solve(DeviceVec<T> lam_rhs, DeviceVec<T> lam, bool check_conv = false) {
        lam.zeroValues();

        addVecLamtoIE(lam_rhs, u_IE, 0.5, 0.0);  // 1/2 cause B_Ddelta scaled operators
        zeroInteriorIE(u_IE);
        sparseMatVec(*kmat_IE, u_IE, -1.0, 0.0, f_IE);
        // sparseMatVec(*kmat_IE, u_IE, 1.0, 0.0, f_IE);

        addVecIEtoI(f_IE, f_I, 1.0, 0.0);
        solveSubdomainI(f_I, u_I);

        addVecItoIE(u_I, u_IE, 1.0, 1.0);

        sparseMatVec(*kmat_IE, u_IE, 1.0, 0.0, f_IE);
        zeroInteriorIE(f_IE);

        addVecIEtoLam(f_IE, lam, 0.5, 0.0);  // 1/2 cause B_Ddelta scaled operators
        return false;                        // fail = false
    }

    void get_global_soln(const DeviceVec<T> &lam, DeviceVec<T> &soln) {
        // 1) compute coarse grid rhs g_V
        addVecIEVtoIE(res_IEV, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        addVecIEVtoVc(res_IEV, f_V, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        addVecLamtoIE(lam, f_IE, 1.0, 0.0);
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);

        // 2) solve coarse problem
        solveCoarse(f_V, u_V);

        // 3) solve interior problems
        addVecIEVtoIE(res_IEV, f_IE, 1.0, 0.0);

        addVecVctoIEV(u_V, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
        addVecIEVtoIE(f_IEV, f_IE, -1.0, 1.0);

        addVecLamtoIE(lam, f_IE, -1.0, 1.0);

        solveSubdomainIE(f_IE, u_IE);

        soln.zeroValues();
        addGlobalSoln(u_IE, u_V, soln);
    }

    void debug_IEV_matrices(bool print_IEV = true, bool print_IE = true, bool print_I = true) {
        // check matrices
        if (print_IEV) {
            auto kmat_IEV_vec = kmat_IEV->getVec().createHostVec();
            T *h_kmat_IEV = kmat_IEV_vec.getPtr();
            int IEV_nvals = kmat_IEV_vec.getSize();
            auto _IEV_bsr_data = kmat_IEV->getBsrData();
            int *h_IEV_rows = DeviceVec<int>(IEV_nnzb, _IEV_bsr_data.rows).createHostVec().getPtr();
            int *h_IEV_cols = DeviceVec<int>(IEV_nnzb, _IEV_bsr_data.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_IEV %d\n", IEV_nvals);
            for (int iblock = 0; iblock < IEV_nnzb; iblock++) {
                T *IEV_block = &h_kmat_IEV[36 * iblock];
                int row = h_IEV_rows[iblock], col = h_IEV_cols[iblock];
                int grow = IEV_nodes[row], gcol = IEV_nodes[col];
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = IEV_block[6 * iy + ix];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }

        if (print_IE) {
            // prelim check, assumes not perm here
            // for (int i = 0; i < IE_nnodes; i++) {
            //     for (int jp = IE_rowp[i]; jp < IE_rowp[i + 1]; jp++) {
            //         int j = IE_cols[jp];
            //         int gr = IE_nodes[i], gc = IE_nodes[j];
            //         printf("IE block %d, glob (%d,%d)\n", jp, gr, gc);
            //     }
            // }

            auto kmat_IE_vec = kmat_IE->getVec().createHostVec();
            T *h_kmat_IE = kmat_IE_vec.getPtr();
            int IE_nvals = kmat_IE_vec.getSize();
            auto _IEV_bsr_data2 = kmat_IE->getBsrData();
            int *h_IE_rows = DeviceVec<int>(IE_nnzb, _IEV_bsr_data2.rows).createHostVec().getPtr();
            int *h_IE_cols = DeviceVec<int>(IE_nnzb, _IEV_bsr_data2.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_IE %d\n", IE_nvals);
            for (int iblock = 0; iblock < IE_nnzb; iblock++) {
                T *IE_block = &h_kmat_IE[36 * iblock];
                int row_perm = h_IE_rows[iblock], col_perm = h_IE_cols[iblock];
                int row = IE_perm[row_perm], col = IE_perm[col_perm];
                int grow = IE_nodes[row], gcol = IE_nodes[col];
                // printf("row %d, col %d\n", row, col);
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = IE_block[6 * iy + ix];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }

        if (print_I) {
            // prelim check, assumes not perm here
            for (int i = 0; i < I_nnodes; i++) {
                for (int jp = I_rowp[i]; jp < I_rowp[i + 1]; jp++) {
                    int j = I_cols[jp];
                    int gr = I_nodes[i], gc = I_nodes[j];
                    printf("I block %d, glob (%d,%d)\n", jp, gr, gc);
                }
            }

            auto kmat_I_vec = kmat_I->getVec().createHostVec();
            T *h_kmat_I = kmat_I_vec.getPtr();
            int I_nvals = kmat_I_vec.getSize();
            auto _IV_bsr_data2 = kmat_I->getBsrData();
            int *h_I_rows = DeviceVec<int>(I_nnzb, _IV_bsr_data2.rows).createHostVec().getPtr();
            int *h_I_cols = DeviceVec<int>(I_nnzb, _IV_bsr_data2.cols).createHostVec().getPtr();
            printf("\n\nh_kmat_I %d\n", I_nvals);
            for (int iblock = 0; iblock < I_nnzb; iblock++) {
                T *I_block = &h_kmat_I[36 * iblock];
                int row = h_I_rows[iblock], col = h_I_cols[iblock];
                int grow = I_nodes[row], gcol = I_nodes[col];
                // printf("row %d, col %d\n", row, col);
                printf("block %d (%d,%d)\n", iblock, grow, gcol);
                for (int i = 0; i < 9; i++) {
                    int ix = 2 + i % 3;
                    int iy = 2 + i / 3;
                    T val = I_block[6 * iy + ix];
                    printf("%.6e,", val);
                    if (i % 3 == 2) printf("\n");
                }
                printf("\n");
            }
        }
    }

    void debug_SVV_matrix() {
        // check matrices
        auto temp_vec = S_VV->getVec().createHostVec();
        T *h_Svv_vals = temp_vec.getPtr();
        int Svv_nvals = temp_vec.getSize();
        auto _SVV_bsr_data = S_VV->getBsrData();
        // printf("h_Svv_vals: ");
        // printVec<T>(36, h_Svv_vals);
        int *h_SVV_rows = DeviceVec<int>(Svv_nnzb, _SVV_bsr_data.rows).createHostVec().getPtr();
        int *h_SVV_cols = DeviceVec<int>(Svv_nnzb, _SVV_bsr_data.cols).createHostVec().getPtr();
        printf("\n\nSVV_mat %d\n", Svv_nvals);
        for (int iblock = 0; iblock < Svv_nnzb; iblock++) {
            T *SVV_block = &h_Svv_vals[36 * iblock];
            int row = h_SVV_rows[iblock], col = h_SVV_cols[iblock];
            int grow = Vc_nodes[row], gcol = Vc_nodes[col];
            printf("block %d (%d,%d)\n", iblock, grow, gcol);
            for (int i = 0; i < 9; i++) {
                int ix = 2 + i % 3;
                int iy = 2 + i / 3;
                T val = SVV_block[6 * iy + ix];
                printf("%.6e,", val);
                if (i % 3 == 2) printf("\n");
            }
            // for (int i = 0; i < 36; i++) {
            //     int ix = i % 6;
            //     int iy = i / 6;
            //     T val = SVV_block[6 * ix + iy];
            //     printf("%.6e,", val);
            //     if (i % 6 == 5) printf("\n");
            // }
            printf("\n");
        }
    }

   protected:
    void clear_host_data() { clear_structured_host_data(); }

    void clear_structured_host_data() {
        if (elem_sd_ind) {
            delete[] elem_sd_ind;
            elem_sd_ind = nullptr;
        }
        if (node_sd_cols) {
            delete[] node_sd_cols;
            node_sd_cols = nullptr;
        }
        if (node_elem_ct) {
            delete[] node_elem_ct;
            node_elem_ct = nullptr;
        }
        if (node_elem_rowp) {
            delete[] node_elem_rowp;
            node_elem_rowp = nullptr;
        }
        if (node_class_ind) {
            delete[] node_class_ind;
            node_class_ind = nullptr;
        }

        if (IEV_nodes) {
            delete[] IEV_nodes;
            IEV_nodes = nullptr;
        }
        if (IE_nodes) {
            delete[] IE_nodes;
            IE_nodes = nullptr;
        }
        if (I_nodes) {
            delete[] I_nodes;
            I_nodes = nullptr;
        }
        if (IE_rowp) {
            delete[] IE_rowp;
            IE_rowp = nullptr;
        }
        if (I_rowp) {
            delete[] I_rowp;
            I_rowp = nullptr;
        }
        if (IE_cols) {
            delete[] IE_cols;
            IE_cols = nullptr;
        }
        if (I_cols) {
            delete[] I_cols;
            I_cols = nullptr;
        }
        if (IEV_elem_conn) {
            delete[] IEV_elem_conn;
            IEV_elem_conn = nullptr;
        }
        if (IEV_sd_ptr) {
            delete[] IEV_sd_ptr;
            IEV_sd_ptr = nullptr;
        }
        if (IEV_sd_ind) {
            delete[] IEV_sd_ind;
            IEV_sd_ind = nullptr;
        }
        if (IEVtoIE_map) {
            delete[] IEVtoIE_map;
            IEVtoIE_map = nullptr;
        }
        if (IEVtoI_map) {
            delete[] IEVtoI_map;
            IEVtoI_map = nullptr;
        }
    }

    void copyKmat_IEVtoIE() {
        // Copy / restrict kmat_IEV to kmat_IE using:
        //   IEVtoIE_map, IE_rowp, IE_cols
        int n_IE_vals = IE_nofill_nnzb * block_dim2;
        dim3 block(32), grid((n_IE_vals + 31) / 32);
        k_copyMatToMat_restrict<T><<<grid, block>>>(IE_nofill_nnzb, block_dim, d_kmat_IEtoIEV_map,
                                                    d_kmat_IEnofill_map, d_IEV_vals.getPtr(),
                                                    d_IE_vals.getPtr());

        // printf("IEV_nnzb %d, IE_nnzb %d, IE_nofill_nnzb %d\n", IEV_nnzb, IE_nnzb,
        // IE_nofill_nnzb);

        // printf("h_kmat_IEtoIEV_map: ");
        // printVec<int>(IE_nofill_nnzb, IEVtoIE_imap);
        // printf("kmat_IEnofill_map: ");
        // printVec<int>(IE_nofill_nnzb, kmat_IEnofill_map);

        // bool debug = true;
        // if (debug) {
        //     int _nnz = IE_nofill_nnzb * block_dim2;
        //     printf("kmat_IE vals on GPU[0] with nnzb=%d: \n", IE_nofill_nnzb);
        //     T *h_kmat_IE_vals = d_IE_vals.createHostVec().getPtr();
        //     for (int i = 0; i < IE_nofill_nnzb; i++) {
        //         // because the IE pattern here has fillin (so nofill match to new code requires
        //         this
        //         // map)
        //         int i2 = kmat_IEnofill_map[i];
        //         T *h_block = &h_kmat_IE_vals[block_dim2 * i2];
        //         printf("kmat_IE: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }

    void copyKmat_IEVtoI() {
        // Copy / restrict kmat_IEV to kmat_I using:
        //   IEVtoI_map, I_rowp, I_cols
        int n_I_vals = I_nofill_nnzb * block_dim2;
        dim3 block(32), grid((n_I_vals + 31) / 32);
        k_copyMatToMat_restrict<T><<<grid, block>>>(I_nofill_nnzb, block_dim, d_kmat_ItoIEV_map,
                                                    d_kmat_Inofill_map, d_IEV_vals.getPtr(),
                                                    d_I_vals.getPtr());

        // bool debug = true;
        // if (debug) {
        //     int _nnz = I_nofill_nnzb * block_dim2;
        //     printf("kmat_I vals on GPU[0] with nnzb=%d: \n", I_nofill_nnzb);
        //     T *h_kmat_I_vals = d_I_vals.createHostVec().getPtr();
        //     for (int i = 0; i < I_nofill_nnzb; i++) {
        //         // because the IE pattern here has fillin (so nofill match to new code requires
        //         this
        //         // map)
        //         int i2 = kmat_Inofill_map[i];
        //         T *h_block = &h_kmat_I_vals[block_dim2 * i2];
        //         printf("kmat_I: block[%d]:\n", i);

        //         for (int j = 0; j < block_dim; j++) {
        //             printVec<T>(block_dim, &h_block[block_dim * j]);
        //         }
        //     }
        // }
    }

    void copyKmat_IEVtoSvv() {
        // copy Avv part from kmat_IEV to Svv
        // note need Svv_copy_nnzb (not Svv_nnzb) cause Vc are duplicate nodes in IEV
        // multiple blocks copied into the same node in S_VV
        int n_Svv_vals = Svv_copy_nnzb * block_dim2;
        dim3 block(32), grid((n_Svv_vals + 31) / 32);
        k_copyMatToMat_restrict<T, true><<<grid, block>>>(Svv_copy_nnzb, block_dim,
                                                          d_Svv_IEV_copyBlocks, d_Svv_Vc_copyBlocks,
                                                          d_IEV_vals.getPtr(), d_Svv_vals.getPtr());

        // for 3+ BDDC levels, also copy the Avv part into S_VV_MLIEV
        bool MLIEV_isnot_null = S_VV_MLIEV != nullptr;
        // printf("MLIEV_isnot_null %d\n", MLIEV_isnot_null);
        if (S_VV_MLIEV != nullptr) {
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("copyKmatIEV to MLIEV\n");

            int n_Svv_MLIEV_vals = Svv_MLIEV_copy_nnzb * block_dim2;
            dim3 block2(32), grid2((n_Svv_MLIEV_vals + 31) / 32);

            k_copyMatToMat_restrict<T, true><<<grid2, block2>>>(
                Svv_MLIEV_copy_nnzb, block_dim, d_Svv_IEV_copyBlocks, d_Svv_Vc_copyBlocks,
                d_IEV_vals.getPtr(), d_Svv_MLIEV_vals.getPtr());

            CHECK_CUDA(cudaDeviceSynchronize());
            printf("\tdone with copyKmatIEV to MLIEV\n");
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
        // need 24 IE subdomain solves (tops) for quad-macro elements of struct mesh
        // to compute the -A_{V,IE} * A_{IE,IE}^{-1} * A_{IE,V} += > S_{VV} second Schur
        // complement inverse term as part of coarse matrix assembly for vertices
        // seems quite expensive but remember we do 2 IE solves and 1 I subdomain solve per
        // Krylov step so this is similar expense to like 8 Krylov steps (not too bad, but
        // not trivial)
        // printf("MAX_NUM_VERTEX_PER_SUBDOMAIN %d\n", MAX_NUM_VERTEX_PER_SUBDOMAIN);
        int ncols = MAX_NUM_VERTEX_PER_SUBDOMAIN * block_dim;
        for (int icol = 0; icol < ncols; icol++) {
            u_IEV.zeroValues();
            setVec_IEVtoV_vals(u_IEV, icol, 1.0);  // set these vals to 1.0 and all else 0
            // printf("computeSvvInverseTerm on icol=%d: u_IEV vec\n", icol);
            // T *h_uIEV = u_IEV.createHostVec().getPtr();
            // printVec<T>(block_dim * IEV_nnodes, h_uIEV);
            // printNodeVec(IEV_nnodes, h_uIEV);

            sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);
            // printf("computeSvvInverseTerm on icol=%d: f_IEV vec\n", icol);
            // T *h_fIEV = f_IEV.createHostVec().getPtr();
            // // printVec<T>(block_dim * IEV_nnodes, h_fIEV);
            // printNodeVec(IEV_nnodes, h_fIEV);

            addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
            // printf("computeSvvInverseTerm on icol=%d: f_IE vec\n", icol);
            // T *h_fIE = f_IE.createHostVec().getPtr();
            // // printVec<T>(block_dim * IE_nnodes, h_fIE);
            // printNodeVec(IE_nnodes, h_fIE);

            solveSubdomainIE(f_IE, u_IE);
            // printf("computeSvvInverseTerm on icol=%d: u_IE vec\n", icol);
            // T *h_uIE = u_IE.createHostVec().getPtr();
            // // printVec<T>(block_dim * IE_nnodes, h_uIE);
            // printNodeVec(IE_nnodes, h_uIE);

            addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
            // printf("computeSvvInverseTerm on icol=%d: u_IEV2 vec\n", icol);
            // T *h_uIEV2 = u_IEV.createHostVec().getPtr();
            // // printVec<T>(block_dim * IEV_nnodes, h_uIEV2);
            // printNodeVec(IEV_nnodes, h_uIEV2);

            sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);
            // printf("computeSvvInverseTerm on icol=%d: f_IEV2 vec\n", icol);
            // T *h_fIEV2 = f_IEV.createHostVec().getPtr();
            // printVec<T>(block_dim * IEV_nnodes, h_fIEV2);
            // printNodeVec(IEV_nnodes, h_fIEV2);
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("after sparseMatVec #2 icol %d\n", icol);

            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("computeSvvInverseTerm pre-addMat sync ok icol %d\n", icol);

            // printf("computeSvvInverseTerm entering addMat_IEVtoV_v?als icol %d\n", icol);
            addMat_IEVtoV_vals(icol, f_IEV);
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("computeSvvInverseTerm finished addMat_IEVtoV_vals icol %d\n", icol);
        }
    }

    void setVec_IEVtoV_vals(DeviceVec<T> &vec_IEV, int irow, T val) {
        int block_row = irow / block_dim;
        int set_nnzb = IEVset_nnzb[block_row];
        int *d_blocks = d_IEVset_blocks[block_row];

        dim3 block(32);
        dim3 grid((set_nnzb + 31) / 32);
        k_setVec_IEVtoV_vals<T>
            <<<grid, block>>>(set_nnzb, block_dim, irow, d_blocks, vec_IEV.getPtr(), val);
    }

    // void addMat_IEVtoV_vals(const int icol, DeviceVec<T> hvec) {
    //     // -----------------------------------------
    //     // standard 2-level BDDC assembly into S_VV
    //     // -----------------------------------------
    //     int block_col = icol / block_dim;
    //     int set_nnzb = IEVtoSVV_nnzb[block_col];
    //     int *d_svv_blocks = d_IEVtoSVV_blocks[block_col];
    //     int *d_iev_blocks = d_IEVout_blocks[block_col];

    //     dim3 block(32);
    //     dim3 grid((set_nnzb * block_dim + 31) / 32);
    //     k_addMat_IEVtoV_vals<T><<<grid, block>>>(set_nnzb, block_dim, icol, d_iev_blocks,
    //                                              d_svv_blocks, hvec.getPtr(),
    //                                              d_Svv_vals.getPtr());

    //     // -----------------------------------------
    //     // for 3+ BDDC levels, also assemble into
    //     // S_VV_MLIEV using the MLIEV sparsity pattern
    //     // -----------------------------------------
    //     // if (S_VV_MLIEV != nullptr) {
    //     //     CHECK_CUDA(cudaDeviceSynchronize());
    //     //     printf("addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);

    //     //     int ML_block_col = icol / block_dim;
    //     //     int ML_set_nnzb = ML_IEVtoSVV_nnzb[ML_block_col];
    //     //     int *d_ML_svv_blocks = d_ML_IEVtoSVV_blocks[ML_block_col];
    //     //     int *d_ML_iev_blocks = d_ML_IEVout_blocks[ML_block_col];
    //     //     // printf("uses MLIEV_set_nnzb %d\n", ML_set_nnzb);

    //     //     dim3 ML_grid((ML_set_nnzb * block_dim + 31) / 32);
    //     //     k_addMat_IEVtoV_vals<T><<<ML_grid, block>>>(ML_set_nnzb, block_dim, icol,
    //     //                                                 d_ML_iev_blocks, d_ML_svv_blocks,
    //     //                                                 hvec.getPtr(),
    //     //                                                 d_Svv_MLIEV_vals.getPtr());

    //     //     CHECK_CUDA(cudaDeviceSynchronize());
    //     //     printf("\tdone with addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);
    //     // }

    //     if (S_VV_MLIEV != nullptr) {
    //         CHECK_CUDA(cudaDeviceSynchronize());
    //         printf("addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);

    //         int ML_block_col = icol / block_dim;
    //         int ML_set_nnzb = ML_IEVtoSVV_nnzb[ML_block_col];
    //         int *d_ML_svv_blocks = d_ML_IEVtoSVV_blocks[ML_block_col];
    //         int *d_ML_iev_blocks = d_ML_IEVout_blocks[ML_block_col];

    //         printf("  ML_block_col %d, ML_set_nnzb %d, d_ML_iev_blocks %p, d_ML_svv_blocks %p\n",
    //                ML_block_col, ML_set_nnzb, (void *)d_ML_iev_blocks, (void *)d_ML_svv_blocks);

    //         if (ML_set_nnzb > 0) {
    //             std::vector<int> h_dbg_iev(ML_set_nnzb);
    //             std::vector<int> h_dbg_svv(ML_set_nnzb);

    //             CHECK_CUDA(cudaMemcpy(h_dbg_iev.data(), d_ML_iev_blocks, ML_set_nnzb *
    //             sizeof(int),
    //                                   cudaMemcpyDeviceToHost));
    //             CHECK_CUDA(cudaMemcpy(h_dbg_svv.data(), d_ML_svv_blocks, ML_set_nnzb *
    //             sizeof(int),
    //                                   cudaMemcpyDeviceToHost));

    //             for (int i = 0; i < ML_set_nnzb; i++) {
    //                 if (h_dbg_iev[i] < 0 || h_dbg_iev[i] >= IEV_nnodes) {
    //                     printf(
    //                         "ERROR: device ML iev block invalid at slot %d, i %d: %d not in "
    //                         "[0,%d)\n",
    //                         ML_block_col, i, h_dbg_iev[i], IEV_nnodes);
    //                     exit(-1);
    //                 }
    //                 if (h_dbg_svv[i] < 0 || h_dbg_svv[i] >= Svv_MLIEV_nnzb) {
    //                     printf(
    //                         "ERROR: device ML svv block invalid at slot %d, i %d: %d not in "
    //                         "[0,%d)\n",
    //                         ML_block_col, i, h_dbg_svv[i], Svv_MLIEV_nnzb);
    //                     exit(-1);
    //                 }
    //             }

    //             printf("  first few ML device blocks for slot %d:\n", ML_block_col);
    //             for (int i = 0; i < ML_set_nnzb && i < 12; i++) {
    //                 printf("    i %d: iev %d, svv %d\n", i, h_dbg_iev[i], h_dbg_svv[i]);
    //             }
    //         }

    //         dim3 block(32);
    //         dim3 ML_grid((ML_set_nnzb * block_dim + 31) / 32);
    //         k_addMat_IEVtoV_vals<T><<<ML_grid, block>>>(ML_set_nnzb, block_dim, icol,
    //                                                     d_ML_iev_blocks, d_ML_svv_blocks,
    //                                                     hvec.getPtr(),
    //                                                     d_Svv_MLIEV_vals.getPtr());

    //         CHECK_CUDA(cudaDeviceSynchronize());
    //         printf("\tdone with addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);
    //     }
    // }

    void addMat_IEVtoV_vals(const int icol, DeviceVec<T> hvec) {
        // printf("addMat_IEVtoV_vals ENTER icol %d\n", icol);

        // -----------------------------------------
        // standard 2-level BDDC assembly into S_VV
        // -----------------------------------------
        int block_col = icol / block_dim;
        int set_nnzb = IEVtoSVV_nnzb[block_col];
        int *d_svv_blocks = d_IEVtoSVV_blocks[block_col];
        int *d_iev_blocks = d_IEVout_blocks[block_col];

        // printf("  regular part: block_col %d, set_nnzb %d, d_iev_blocks %p, d_svv_blocks %p\n",
        //        block_col, set_nnzb, (void *)d_iev_blocks, (void *)d_svv_blocks);

        if (set_nnzb > 0) {
            std::vector<int> h_dbg_iev(set_nnzb);
            std::vector<int> h_dbg_svv(set_nnzb);

            CHECK_CUDA(cudaMemcpy(h_dbg_iev.data(), d_iev_blocks, set_nnzb * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_dbg_svv.data(), d_svv_blocks, set_nnzb * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < set_nnzb; i++) {
                if (h_dbg_iev[i] < 0 || h_dbg_iev[i] >= IEV_nnodes) {
                    printf("ERROR: regular iev block invalid at slot %d, i %d: %d not in [0,%d)\n",
                           block_col, i, h_dbg_iev[i], IEV_nnodes);
                    exit(-1);
                }
                if (h_dbg_svv[i] < 0 || h_dbg_svv[i] >= Svv_nnzb) {
                    printf("ERROR: regular svv block invalid at slot %d, i %d: %d not in [0,%d)\n",
                           block_col, i, h_dbg_svv[i], Svv_nnzb);
                    exit(-1);
                }
            }

            // printf("  first few regular device blocks for slot %d:\n", block_col);
            // for (int i = 0; i < set_nnzb && i < 12; i++) {
            //     printf("    i %d: iev %d, svv %d\n", i, h_dbg_iev[i], h_dbg_svv[i]);
            // }
        }

        dim3 block(32);
        dim3 grid((set_nnzb * block_dim + 31) / 32);
        // printf("  launching regular k_addMat_IEVtoV_vals, grid.x %d block.x %d\n", (int)grid.x,
        //        (int)block.x);

        k_addMat_IEVtoV_vals<T><<<grid, block>>>(set_nnzb, block_dim, icol, d_iev_blocks,
                                                 d_svv_blocks, hvec.getPtr(), d_Svv_vals.getPtr());

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("  done with regular addMat_IEVtoV_vals icol %d\n", icol);

        // -----------------------------------------
        // for 3+ BDDC levels, also assemble into
        // S_VV_MLIEV using the MLIEV sparsity pattern
        // -----------------------------------------
        if (S_VV_MLIEV != nullptr) {
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);

            int ML_block_col = icol / block_dim;
            int ML_set_nnzb = ML_IEVtoSVV_nnzb[ML_block_col];
            int *d_ML_svv_blocks = d_ML_IEVtoSVV_blocks[ML_block_col];
            int *d_ML_iev_blocks = d_ML_IEVout_blocks[ML_block_col];

            printf("  ML_block_col %d, ML_set_nnzb %d, d_ML_iev_blocks %p, d_ML_svv_blocks %p\n",
                   ML_block_col, ML_set_nnzb, (void *)d_ML_iev_blocks, (void *)d_ML_svv_blocks);

            if (ML_set_nnzb > 0) {
                std::vector<int> h_dbg_iev(ML_set_nnzb);
                std::vector<int> h_dbg_svv(ML_set_nnzb);

                CHECK_CUDA(cudaMemcpy(h_dbg_iev.data(), d_ML_iev_blocks, ML_set_nnzb * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_dbg_svv.data(), d_ML_svv_blocks, ML_set_nnzb * sizeof(int),
                                      cudaMemcpyDeviceToHost));

                for (int i = 0; i < ML_set_nnzb; i++) {
                    if (h_dbg_iev[i] < 0 || h_dbg_iev[i] >= IEV_nnodes) {
                        printf(
                            "ERROR: device ML iev block invalid at slot %d, i %d: %d not in "
                            "[0,%d)\n",
                            ML_block_col, i, h_dbg_iev[i], IEV_nnodes);
                        exit(-1);
                    }
                    if (h_dbg_svv[i] < 0 || h_dbg_svv[i] >= Svv_MLIEV_nnzb) {
                        printf(
                            "ERROR: device ML svv block invalid at slot %d, i %d: %d not in "
                            "[0,%d)\n",
                            ML_block_col, i, h_dbg_svv[i], Svv_MLIEV_nnzb);
                        exit(-1);
                    }
                }

                printf("  first few ML device blocks for slot %d:\n", ML_block_col);
                for (int i = 0; i < ML_set_nnzb && i < 12; i++) {
                    printf("    i %d: iev %d, svv %d\n", i, h_dbg_iev[i], h_dbg_svv[i]);
                }
            }

            dim3 ML_grid((ML_set_nnzb * block_dim + 31) / 32);
            printf("  launching ML k_addMat_IEVtoV_vals, grid.x %d block.x %d\n", (int)ML_grid.x,
                   (int)block.x);

            k_addMat_IEVtoV_vals<T><<<ML_grid, block>>>(ML_set_nnzb, block_dim, icol,
                                                        d_ML_iev_blocks, d_ML_svv_blocks,
                                                        hvec.getPtr(), d_Svv_MLIEV_vals.getPtr());

            CHECK_CUDA(cudaDeviceSynchronize());
            printf("\tdone with addMat_IEVtoV_vals: MLIEV part icol %d\n", icol);
        }

        // printf("addMat_IEVtoV_vals EXIT icol %d\n", icol);
    }

    template <bool scaled = false>
    void addVec_globalToIEV(Vec &x_global, Vec &y_iev, int vars_per_node_in, T a, T b) {
        // Scatter a global nodal vector into duplicated IEV layout.
        // Likely based on IEV_nodes and vars_per_node_in.

        CHECK_CUBLAS(cublasDscal(cublasHandle, y_iev.getSize(), &b, y_iev.getPtr(), 1));

        // // DEBUG:
        // int *h_IE_nodes = DeviceVec<int>(IE_nnodes, d_IE_nodes).createHostVec().getPtr();
        // printf("h_IE_nodes: ");
        // printVec<int>(IE_nnodes, h_IE_nodes);
        // int *h_Vc_nodes = DeviceVec<int>(Vc_nnodes, d_Vc_nodes).createHostVec().getPtr();
        // printf("h_Vc_nodes: ");
        // printVec<int>(Vc_nnodes, h_Vc_nodes);
        // T *h_x_glob = x_global.createHostVec().getPtr();
        // printf("h_x_glob: ");
        // printVec<T>(x_global.getSize(), h_x_glob);

        u_IE.zeroValues();
        u_V.zeroValues();

        int nvals = IE_nnodes * vars_per_node_in;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_GlobalToIE<T, scaled><<<grid, block>>>(
            IE_nnodes, vars_per_node_in, d_IE_nodes, d_IE_nsd, x_global.getPtr(), u_IE.getPtr(), a);

        int nvals2 = Vc_nnodes * vars_per_node_in;
        dim3 grid2((nvals2 + 31) / 32);
        // scales by 0.25x like for load distribution
        k_addVec_GlobaltoVc<T, scaled><<<grid2, block>>>(Vc_nnodes, vars_per_node_in, d_Vc_nodes,
                                                         d_vertex_nsd, x_global.getPtr(),
                                                         u_V.getPtr(), a);
        // CHECK_CUBLAS(cublasDscal(cublasHandle, u_IE.getSize(), &a, u_IE.getPtr(), 1));

        // T *h_u_IE = u_IE.createHostVec().getPtr();
        // printf("\nh_u_IE\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("IE %d, gnode %d, ", inode, IE_nodes[inode]);
        //     printVec<T>(3, &h_u_IE[3 * inode]);
        // }

        // T *h_u_V = u_V.createHostVec().getPtr();
        // printf("\nh_u_V\n");
        // for (int inode = 0; inode < Vc_nnodes; inode++) {
        //     printf("IE %d, gnode %d, ", inode, Vc_nodes[inode]);
        //     printVec<T>(3, &h_u_V[3 * inode]);
        // }

        // helper routines
        addVecIEtoIEV(u_IE, y_iev, 1.0, 0.0, vars_per_node_in);
        addVecVctoIEV(u_V, y_iev, 1.0, 1.0, vars_per_node_in);
    }

    void addVecIEVtoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict I/E entries from IEV into IE
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerOut<T>
            <<<grid, block>>>(IE_nnodes, block_dim, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecItoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict interior part from IE into I
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = I_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerIn<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecIEVtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // restrict interior part from IE into I
        // map(a * x) + b * y => y

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = I_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerOut<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    template <bool scaled = false>
    void addVecIEVtoVc(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // gather/scatter assembled primal V entries from IEV into Vc (coarse vertex,
        // non-repeated) map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = V_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_IEVtoVc<T><<<grid, block>>>(V_nnodes, block_dim, d_IEVtoV_imap, d_VctoV_imap,
                                             x.getPtr(), y.getPtr(), a);

        if constexpr (scaled) {
            int Vc_nvals = Vc_nnodes * block_dim;
            dim3 block(32), grid((Vc_nvals + 31) / 32);
            k_subdomain_normalize_vec_inout<T>
                <<<grid, block>>>(this->Vc_nnodes, this->block_dim, this->d_vertex_nsd, y.getPtr());
        }
    }

    void addVecIEtoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        // scatter IE entries into IEV slots
        // map(a * x) + b * y => y
        if (vars_per_node == -1) vars_per_node = block_dim;
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = IE_nnodes * vars_per_node;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecSmallerIn<T>
            <<<grid, block>>>(IE_nnodes, vars_per_node, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);

        // int *h_IEVtoIE_imap = DeviceVec<int>(IE_nnodes,
        // d_IEVtoIE_imap).createHostVec().getPtr(); printf("h_IEVtoIE_imap: ");
        // printVec<int>(IE_nnodes, h_IEVtoIE_imap);
        // printf("Check global nodes match\n");
        // for (int IE_node = 0; IE_node < IE_nnodes; IE_node++) {
        //     int gnode1 = IE_nodes[IE_node];
        //     int IEV_node = h_IEVtoIE_imap[IE_node];
        //     int gnode2 = IEV_nodes[IEV_node];
        //     printf("ind %d, IEV_node %d, gnode1 %d, gnode2 %d\n", IE_node, IEV_node,
        //     gnode1,
        //            gnode2);
        // }
    }

    template <bool scaled = false>
    void addVecVctoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        // inject coarse/global V entries into local IEV primal slots
        // map(a * x) + b * y => y

        cudaMemcpy(this->temp_V.getPtr(), x.getPtr(), this->Vc_nnodes * this->block_dim * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        if constexpr (scaled) {
            int Vc_nvals = Vc_nnodes * block_dim;
            dim3 block(32), grid((Vc_nvals + 31) / 32);
            k_subdomain_normalize_vec_inout<T><<<grid, block>>>(
                this->Vc_nnodes, this->block_dim, this->d_vertex_nsd, this->temp_V.getPtr());
        }

        if (vars_per_node == -1) vars_per_node = block_dim;
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));
        int nvals = V_nnodes * vars_per_node;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_VctoIEV<T><<<grid, block>>>(V_nnodes, vars_per_node, d_IEVtoV_imap, d_VctoV_imap,
                                             this->temp_V.getPtr(), y.getPtr(), a);

        // int *h_IEVtoV_imap = DeviceVec<int>(V_nnodes,
        // d_IEVtoV_imap).createHostVec().getPtr(); printf("h_IEVtoV_imap: ");
        // printVec<int>(V_nnodes, h_IEVtoV_imap);
        // int *h_VctoV_imap = DeviceVec<int>(V_nnodes,
        // d_VctoV_imap).createHostVec().getPtr(); printf("h_VctoV_imap: ");
        // printVec<int>(V_nnodes, h_VctoV_imap);
    }

    void addVecItoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // scatter interior part from I into IE
        // map(a * x) + b * y => y

        // reuse previous routines instead of adding a new map..
        // don't worry this routine is only called once in main solve so not too much extra
        // overhead
        addVecItoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoIE(temp_IEV, y, 1.0, b);
    }

    void addVecIEtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        // scatter interior part from I into IE
        // map(a * x) + b * y => y

        // reuse previous routines instead of adding a new map..
        // don't worry this routine is only called once in main solve so not too much extra
        // overhead
        addVecIEtoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoI(temp_IEV, y, 1.0, b);
    }

    void zeroInteriorIE(DeviceVec<T> &x) {
        // keep edge part, zero interior part
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_zeroInterior<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_interior, x.getPtr());
    }

    void addVecLamtoIE(const DeviceVec<T> &lam, DeviceVec<T> &vec_IE, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecLamtoIE<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            lam.getPtr(), vec_IE.getPtr(), a);
    }

    void addVecIEtoLam(const DeviceVec<T> &vec_IE, DeviceVec<T> &lam, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(cublasHandle, lam.getSize(), &b, lam.getPtr(), 1));
        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecIEtoLam<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            vec_IE.getPtr(), lam.getPtr(), a);
    }

    void addGlobalSoln(const DeviceVec<T> &u_IE, const DeviceVec<T> &u_V, DeviceVec<T> &soln) {
        //  - scatter interior directly to global
        //  - average duplicated edge values into global edge dofs
        //  - inject global primal values

        int nvals = IE_nnodes * block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVec_IEtoGlobal<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_nodes, d_IE_nsd,
                                                u_IE.getPtr(), soln.getPtr(), 1.0);

        int nvals2 = Vc_nnodes * block_dim;
        dim3 grid2((nvals2 + 31) / 32);
        k_addVec_VctoGlobal<T>
            <<<grid2, block>>>(Vc_nnodes, block_dim, d_Vc_nodes, u_V.getPtr(), soln.getPtr(), 1.0);
    }

    void sparseMatVec(BsrMatType &A, Vec &x, T alpha, T beta, Vec &y) {
        // y <- alpha * A * x + beta * y
        auto bsr_data = A.getBsrData();
        int mb = bsr_data.mb;
        int nb = bsr_data.nb;
        int nnzb = bsr_data.nnzb;
        T *d_vals = A.getVec().getPtr();
        int *d_rowp = bsr_data.rowp;
        int *d_cols = bsr_data.cols;
        int *perm = bsr_data.perm, *iperm = bsr_data.iperm;

        // cudaEvent_t start, stop;
        // float elapsed_ms = 0.0f;
        // if (print_timing) {
        //     CHECK_CUDA(cudaEventCreate(&start));
        //     CHECK_CUDA(cudaEventCreate(&stop));
        //     CHECK_CUDA(cudaEventRecord(start));
        // }

        // permute x input vec from VIS to solve order
        x.permuteData(block_dim, iperm);
        y.permuteData(block_dim, iperm);

        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
            &alpha, descrK, d_vals, d_rowp, d_cols, block_dim, x.getPtr(), &beta, y.getPtr()));

        // permute y output vec from solve to VIS order
        x.permuteData(block_dim, perm);
        y.permuteData(block_dim, perm);

        // if (print_timing) {
        //     CHECK_CUDA(cudaEventRecord(stop));
        //     CHECK_CUDA(cudaEventSynchronize(stop));
        //     CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        //     CHECK_CUDA(cudaDeviceSynchronize());

        //     printf("\t sparseMatVec time: %.6f ms\n", elapsed_ms);

        //     CHECK_CUDA(cudaEventDestroy(start));
        //     CHECK_CUDA(cudaEventDestroy(stop));
        // }
    }

    void sparseTransposeMatVec(BsrMatType &A, Vec &x, T alpha, T beta, Vec &y) {
        // BSR mat-vec operation
        // y <- alpha * A^T * x + beta * y
        // NOTE : this method is currently unused

        auto bsr_data = A.getBsrData();
        int mb = bsr_data.mb;
        int nb = bsr_data.nb;
        int nnzb = bsr_data.nnzb;
        T *d_vals = A.getVec().getPtr();
        int *d_rows = bsr_data.rows;
        int *d_rowp = bsr_data.rowp;
        int *d_cols = bsr_data.cols;
        int *perm = bsr_data.perm, *iperm = bsr_data.iperm;

        cudaEvent_t start, stop;
        float elapsed_ms = 0.0f;
        if (print_timing) {
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
        }

        // permute x input vec from VIS to solve order
        x.permuteData(block_dim, iperm);
        y.permuteData(block_dim, iperm);

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &beta, y.getPtr(), 1));
        const int nprods = nnzb * block_dim * block_dim;
        dim3 block(32), grid((nprods + 31) / 32);
        k_bsrmv_transpose_ax<T><<<grid, block>>>(nnzb, block_dim, d_rows, d_cols, d_vals,
                                                 x.getPtr(), alpha, y.getPtr());
        CHECK_CUDA(cudaGetLastError());

        // permute y output vec from solve to VIS order
        x.permuteData(block_dim, perm);
        y.permuteData(block_dim, perm);

        if (print_timing) {
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            CHECK_CUDA(cudaDeviceSynchronize());

            printf("\t sparseTransposeMatVec time: %.6f ms\n", elapsed_ms);

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
    }

    void solveSubdomainIE(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainIESolver) {
            printf("ERROR: subdomain IE solver is null\n");
            return;
        }
        // printf("inside subdomainIE\n");

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("post synchronize\n");

        auto _bsr_data = kmat_IE->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        // bool perm_notnull = d_perm != nullptr;
        // bool iperm_notnull = d_iperm != nullptr;
        // printf("perm_notnull %d, iperm_notnull %d\n", perm_notnull, iperm_notnull);

        // T *h_rhs_in = rhs_in.createHostVec().getPtr();
        // int n_vals = rhs_in.getSize();
        // printf("h_rhs_in(%d): ", n_vals);
        // printVec<T>(n_vals, h_rhs_in);
        // T *h_rhs_IE_perm = rhs_IE_perm.createHostVec().getPtr();
        // int n_vals_IE = rhs_IE_perm.getSize();
        // printf("h_rhs_IE_perm(%d): ", n_vals_IE);
        // printVec<T>(n_vals_IE, h_rhs_IE_perm);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("copyValuesTo rhs_perm\n");
        rhs_in.copyValuesTo(rhs_IE_perm);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("permute rhs_IE_perm\n");

        rhs_IE_perm.permuteData(block_dim, d_iperm);

        // printf("subdomainIEsolver->solve\n");
        subdomainIESolver->solve(rhs_IE_perm, sol_IE_perm);
        // printf("\tdone with subdomainIEsolver->solve\n");

        sol_IE_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveSubdomainI(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainISolver) {
            printf("ERROR: subdomain I solver is null\n");
            return;
        }

        auto _bsr_data = kmat_I->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(rhs_I_perm);
        rhs_I_perm.permuteData(block_dim, d_iperm);

        subdomainISolver->solve(rhs_I_perm, sol_I_perm);

        sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveCoarse(Vec &rhs_in, Vec &sol_out) {
        if (!coarseSolver) {
            printf("ERROR: coarse solver is null\n");
            return;
        }

        auto _bsr_data = S_VV->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(rhs_Vc_perm);
        rhs_Vc_perm.permuteData(block_dim, d_iperm);

        // for direct solver it doesn't matter for true or not
        // and if Krylov solver you do want to check conv
        bool check_conv = true;
        coarseSolver->solve(rhs_Vc_perm, sol_Vc_perm, check_conv);
        // coarseSolver->solve(rhs_in, sol_out);

        sol_Vc_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void _compute_jump_operators(bool square_domain = true) {
        // compute +1/0/-1 coefficients from u_IE to lam (with 0 for I and +1/-1 for E
        // edges)

        // compute IE to lam map
        IE_to_lam_map = new int[IE_nnodes];
        lam_nodes = new int[lam_nnodes];
        // NOTE : turn the -1 on to ensure the map is correct (nz parts basically)
        memset(IE_to_lam_map, -1, IE_nnodes * sizeof(int));
        // memset(IE_to_lam_map, 0, IE_nnodes * sizeof(int));
        int *temp_glob_node_tracker = new int[num_nodes];
        memset(temp_glob_node_tracker, -1, num_nodes * sizeof(int));
        int *temp_lam_ind = new int[num_nodes];
        memset(temp_lam_ind, -1, num_nodes * sizeof(int));
        int lam_ind = 0;
        for (int i = 0; i < IE_nnodes; i++) {
            int glob_node = IE_nodes[i];
            int node_class = node_class_ind[glob_node];

            if (node_class == EDGE) {
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
            //  else {
            //     IE_to_lam_map[i] = -1;
            // }
        }
        // printf("IE_to_lam_map: ");
        // printVec<int>(IE_nnodes, IE_to_lam_map);
        // printf("lam_nodes %d: ", lam_nnodes);
        // printVec<int>(lam_nnodes, lam_nodes);

        d_IE_to_lam_map = HostVec<int>(IE_nnodes, IE_to_lam_map).createDeviceVec().getPtr();

        // get the scales for edge DOF (# subdomains)
        edge_nsd = new int[lam_nnodes];
        for (int ilam = 0; ilam < lam_nnodes; ilam++) {
            int glob_node = lam_nodes[ilam];
            edge_nsd[ilam] = node_nsd[glob_node];
        }

        // and similarly for vertex
        vertex_nsd = new int[Vc_nnodes];
        for (int iv = 0; iv < Vc_nnodes; iv++) {
            int glob_node = Vc_nodes[iv];
            vertex_nsd[iv] = node_nsd[glob_node];
        }

        IE_nsd = new int[IE_nnodes];
        for (int i = 0; i < IE_nnodes; i++) {
            int glob_node = IE_nodes[i];
            IE_nsd[i] = node_nsd[glob_node];
        }

        d_edge_nsd = HostVec<int>(lam_nnodes, edge_nsd).createDeviceVec().getPtr();
        d_vertex_nsd = HostVec<int>(Vc_nnodes, vertex_nsd).createDeviceVec().getPtr();
        d_IE_nsd = HostVec<int>(IE_nnodes, IE_nsd).createDeviceVec().getPtr();

        if (!square_domain) return;  //

        IE_to_lam_vec = new T[IE_nnodes];
        for (int inode = 0; inode < IE_nnodes; inode++) {
            int glob_node = IE_nodes[inode];
            int IEV_node = IEVtoIE_imap[inode];
            int node_class = node_class_ind[glob_node];
            int sd_ind = IEV_sd_ind[IEV_node];
            int isx = sd_ind % nxs, isy = sd_ind / nxs;
            // use red-black coloring to determine even/odd parity and signs of edge fluxes
            int rb_color = isx + isy;
            // printf("IE_node %d, IEV_node %d, glob_node %d, node_class %d, sd_ind %d\n",
            // inode,
            //        IEV_node, glob_node, node_class, sd_ind);
            // printf("sd_ind %d, sd (%d,%d), rb_color %d\n", sd_ind, isx, isy, rb_color);

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                IE_to_lam_vec[inode] = 0.0;
            } else if (node_class == EDGE) {
                IE_to_lam_vec[inode] = (rb_color % 2 == 0) ? 1.0 : -1.0;
            }
        }
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_to_lam_vec: ");
        // printVec<T>(IE_nnodes, IE_to_lam_vec);

        d_IE_to_lam_vec = HostVec<T>(IE_nnodes, IE_to_lam_vec).createDeviceVec().getPtr();
    }

    static int find_block_index(int irow, int jcol, const int *rowp, const int *cols) {
        for (int jp = rowp[irow]; jp < rowp[irow + 1]; jp++) {
            if (cols[jp] == jcol) {
                return jp;
            }
        }
        return -1;
    }

    void allocate_workspace() {
        // duplicated IEV vectors
        d_IEV_xpts = Vec(IEV_nnodes * 3);
        d_IEV_vars = Vec(IEV_nnodes * block_dim);

        fext_IEV = Vec(IEV_nnodes * block_dim);
        res_IEV = Vec(IEV_nnodes * block_dim);
        fint_IEV = Vec(IEV_nnodes * block_dim);
        f_IEV = Vec(IEV_nnodes * block_dim);
        u_IEV = Vec(IEV_nnodes * block_dim);
        temp_IEV = Vec(IEV_nnodes * block_dim);
        temp_V = Vec(Vc_nnodes * block_dim);

        // IE vectors
        f_IE = Vec(IE_nnodes * block_dim);
        u_IE = Vec(IE_nnodes * block_dim);
        temp_IE = Vec(IE_nnodes * block_dim);
        rhs_IE_perm = Vec(IE_nnodes * block_dim);
        sol_IE_perm = Vec(IE_nnodes * block_dim);

        // I vectors
        f_I = Vec(I_nnodes * block_dim);
        u_I = Vec(I_nnodes * block_dim);
        rhs_I_perm = Vec(I_nnodes * block_dim);
        sol_I_perm = Vec(I_nnodes * block_dim);
        temp_I = Vec(I_nnodes * block_dim);

        // coarse vertex vectors
        f_V = Vec(Vc_nnodes * block_dim);
        u_V = Vec(Vc_nnodes * block_dim);
        rhs_Vc_perm = Vec(Vc_nnodes * block_dim);
        sol_Vc_perm = Vec(Vc_nnodes * block_dim);
    }

   public:
    int nxe, nye, nxs, nys;
    int nx, ny;
    int num_nodes, num_elements, N;
    int nnxs, nnys, num_subdomains;

    BsrMatType *kmat, *kmat_IEV, *kmat_IE, *kmat_I, *B_delta, *B_Ddelta, *S_VV, *S_VV_MLIEV;
    Vec f_IEV, f_IE, f_I, f_V;
    Vec u_IEV, u_IE, u_I, u_V;
    Vec temp_IEV, temp_IE, temp_V, temp_I;
    BsrData IEV_bsr_data, IE_bsr_data, I_bsr_data;
    BsrData d_IEV_bsr_data, d_IE_bsr_data, d_I_bsr_data;
    BsrData Svv_bsr_data, d_Svv_bsr_data;
    BsrData Svv_MLIEV_bsr_data, d_Svv_MLIEV_bsr_data;
    int Svv_nofill_nnzb;
    int IE_nofill_nnzb, I_nofill_nnzb;
    int IEV_nnzb, IE_nnzb, I_nnzb;

   protected:
    ShellAssembler assembler;
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;

    BaseSolver *subdomainISolver, *subdomainIESolver, *coarseSolver;

    int *elem_sd_ind, *elem_conn;
    int node_elem_nnz;
    int *node_sd_cols, *node_elem_ct, *node_elem_rowp, *node_class_ind;
    int nnodes_interior, nnodes_edge, nnodes_vertex, nnodes_dirichlet_edge;

    int *IEV_nodes, *IE_nodes, *I_nodes;
    int *Vc_nodes;
    int *d_IE_nodes, *d_Vc_nodes;
    int *IEV_rowp, *IE_rowp, *I_rowp;
    int *IEV_rows, *IE_rows, *I_rows;
    int *IEV_cols, *IE_cols, *I_cols;
    int IEV_nnodes, IE_nnodes, I_nnodes;
    int *IEV_elem_conn;
    int *IEV_sd_ptr, *IEV_sd_ind;
    int *IEVtoIE_map, *IEVtoI_map;
    int *IEVtoIE_imap, *IEVtoI_imap;
    int *d_IEVtoIE_imap, *d_IEVtoI_imap;
    int Vc_nnodes, V_nnodes;
    int *VctoV_imap, *d_VctoV_imap;
    int *IEVtoV_imap, *d_IEVtoV_imap;
    bool *IE_interior, *d_IE_interior;
    bool *IE_general_edge, *d_IE_general_edge;
    int *Vc_node_imap;
    bool print_timing;
    DeviceVec<T> d_coarse_vars;

    // optional cleanup of saved host-side nofill patterns
    int *IE_rowp_nofill, *IE_cols_nofill;
    int *I_rowp_nofill, *I_cols_nofill;
    int *Svv_rowp_nofill, *Svv_cols_nofill;

    // perm maps
    int *IE_perm, *IE_iperm;
    int *I_perm, *I_iperm;
    int *SVV_perm, *SVV_iperm;
    int *SVV_MLIEV_perm, *SVV_MLIEV_iperm;
    int *d_IE_perm, *d_IE_iperm;
    int *d_I_perm, *d_I_iperm;
    int *d_SVV_perm, *d_SVV_iperm;
    int *d_SVV_MLIEV_perm, *d_SVV_MLIEV_iperm;

    int *kmat_ItoIEV_map, *d_kmat_ItoIEV_map;
    int *kmat_IEtoIEV_map, *d_kmat_IEtoIEV_map;
    int *kmat_Inofill_map, *d_kmat_Inofill_map;
    int *kmat_IEnofill_map, *d_kmat_IEnofill_map;
    int lam_nnodes;
    T *IE_to_lam_vec, *d_IE_to_lam_vec;
    int *IE_to_lam_map, *d_IE_to_lam_map;
    int *lam_nodes;

    int Svv_nnzb, Svv_nodes;
    int *Svv_rowp, *Svv_rows, *Svv_cols;
    int *d_Svv_rowp, *d_Svv_rows, *d_Svv_cols;
    Vec d_Svv_vals;
    int Svv_copy_nnzb;
    int *d_Svv_Vc_copyBlocks, *d_Svv_IEV_copyBlocks;

    int Svv_MLIEV_nnzb, Svv_MLIEV_nodes;
    int *Svv_MLIEV_rowp, *Svv_MLIEV_rows, *Svv_MLIEV_cols;
    int *d_Svv_MLIEV_rowp, *d_Svv_MLIEV_rows, *d_Svv_MLIEV_cols;
    Vec d_Svv_MLIEV_vals;
    int Svv_MLIEV_copy_nnzb;
    int *d_Svv_MLIEV_Vc_copyBlocks, *d_Svv_MLIEV_IEV_copyBlocks;

    int *node_nsd, *edge_nsd, *vertex_nsd, *IE_nsd;
    int *d_node_nsd, *d_edge_nsd, *d_vertex_nsd, *d_IE_nsd;

    // // up to 4 local vertex slots per subdomain (structured quad partition case)
    int *IEVset_blocks[14], *d_IEVset_blocks[14];      // for setVec_IEVtoV_vals
    int *IEVout_blocks[14], *d_IEVout_blocks[14];      // for addMat_IEVtoV_vals read side
    int *IEVtoSVV_blocks[14], *d_IEVtoSVV_blocks[14];  // for addMat_IEVtoV_vals write side
    int MAX_NUM_VERTEX_PER_SUBDOMAIN;

    // and for multilevel
    int *ML_IEVset_blocks[14], *d_ML_IEVset_blocks[14];      // for setVec_IEVtoV_vals
    int *ML_IEVout_blocks[14], *d_ML_IEVout_blocks[14];      // for addMat_IEVtoV_vals read side
    int *ML_IEVtoSVV_blocks[14], *d_ML_IEVtoSVV_blocks[14];  // for addMat_IEVtoV_vals write side

    int IEVset_nnzb[14], ML_IEVset_nnzb[14];
    // int IEVout_nnzb[4];
    int IEVtoSVV_nnzb[14], ML_IEVtoSVV_nnzb[14];
    // int *IEVtoV_nnzb;       // length 4
    // int **d_IEVtoV_blocks;  // length 4, each entry is device ptr to block list
    // int *IEVtoSVV_nnzb;       // length 4
    // int **d_IEVtoSVV_blocks;  // length 4, each entry is device ptr to block list

    int block_dim, block_dim2;
    Vec d_IEV_vals, d_IE_vals, d_I_vals;
    int kmat_nnzb;
    Vec d_xpts, d_vars;
    Vec d_IEV_xpts, d_IEV_vars;
    DeviceVec<int> d_IEV_elem_conn;
    DeviceVec<int> d_elem_components;
    DeviceVec<Data> d_compData;
    Vec rhs_IE_perm, sol_IE_perm;
    Vec rhs_I_perm, sol_I_perm;
    Vec rhs_Vc_perm, sol_Vc_perm;
    Vec fext_IEV;  // external loads
    DeviceVec<int> d_IEV_bcs;
    DeviceVec<T> res_IEV, fint_IEV;

    cusparseMatDescr_t descrK;

    // TODO:
    // These are assumed to exist in your codebase / future implementation:
    // int *d_IEV_elem_conn;
    // int *elem_components;
    // int *d_compData;
    // static constexpr int elems_per_block = ...;
};