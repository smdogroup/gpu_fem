#pragma once

#include <cstdlib>   // For rand()
#include <cstring>   // For memset
#include <iterator>  // For std::advance
#include <set>
#include <vector>

// basic utils
#include "cuda_utils.h"
#include "lapacke.h"
#include "linalg/vec.h"
#include "multigrid/solvers/solve_utils.h"

// include from GMG multigrid sections
#include "multigrid/prolongation/_unstructured.cuh"  // for transpose mat-vec product
#include "multigrid/solvers/direct/cusp_directLU.h"

// local sa amg imports
#include "_rigid_modes.cuh"
#include "fake_assembler.h"
#include "sa_amg.cuh"

template <typename T, class FAKE_ASSEMBLER, class Smoother, bool ORTHOG_PROJECTOR = true>
class SmoothAggregationAMG : public BaseSolver {
    /* based on python code in _py_demo/_src/bsr_aggregation.py */
   public:
    using Assembler = FAKE_ASSEMBLER;
    using CoarseMG = SmoothAggregationAMG<T, FAKE_ASSEMBLER, Smoother, ORTHOG_PROJECTOR>;
    using CoarseDirect = CusparseMGDirectLU<T, Assembler>;

    SmoothAggregationAMG(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                         Smoother *smoother_, int nnodes_, BsrMat<DeviceVec<T>> kmat_,
                         BsrMat<DeviceVec<T>> kmat_free_, DeviceVec<T> rigid_body_modes_,
                         DeviceVec<int> d_bcs_, int coarse_node_threshold_ = 6000,
                         T sparse_threshold_ = 0.15, T omegaJac_ = 0.3, int nsmooth_ = 1,
                         int level_ = 0, int rbm_nsmooth_ = 1, int prol_nsmooth_ = 3,
                         std::string coarsening_type_ = "standard")
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          smoother(smoother_),
          kmat(kmat_),
          kmat_free(kmat_free_),
          nnodes(nnodes_),
          rigid_body_modes(rigid_body_modes_),
          coarse_node_threshold(coarse_node_threshold_),
          sparse_threshold(sparse_threshold_),
          level(level_),
          nsmooth(nsmooth_) {
        // get data out of kmat
        auto d_kmat_bsr_data = kmat.getBsrData();
        d_kmat_vals = kmat.getVec().getPtr();
        d_kmat_free_vals = kmat_free.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_rows = d_kmat_bsr_data.rows;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        block_dim = d_kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;
        N = nnodes * block_dim;
        omegaJac = omegaJac_;

        d_bcs = d_bcs_;

        // setup phase (first version)
        // printf("1 - AMG initCuda() with nnodes = %d\n", nnodes);
        initCuda();
        // printf("2 - AMG form node aggregates\n");
        form_node_aggregates();

        is_coarse_mg = num_aggregates > coarse_node_threshold;
        // printf("\tis_coarse_mg %d: num_agg %d vs coarse threshold %d\n", is_coarse_mg,
        //    num_aggregates, coarse_node_threshold);

        // printf("3 - AMG get prolong nz pattern\n");
        compute_prolongation_nz_pattern();
        // printf("4 - AMG compute coarse grid nz pattern\n");
        compute_coarse_grid_nz_pattern();
        // printf("\tdone with AMG init\n");
        // _done_post_apply_bcs = false;
        compute_coarse_problem();
        // d_bcs = DeviceVec<int>(0);  // no bcs default
    }

    int get_total_nnzb() {
        int c_nnzb = P_nnzb * 2 + kmat_nnzb;
        if (is_coarse_mg) {
            c_nnzb += coarse_mg->get_total_nnzb();
        } else {
            c_nnzb += coarse_direct->get_nnzb();
        }
        return c_nnzb;
    }
    T get_operator_complexity(int nofill_nnzb) {
        return T(get_total_nnzb()) * 1.0 / T(nofill_nnzb);
    }

    void compute_coarse_problem() {
        // after applying BCs on kmat now compute the tentative prolongator and other values
        // meaning kmat no bcs used for aggregate formation
        // old arg: DeviceVec<int> d_bcs_
        // d_bcs = d_bcs_;
        // printf("\nPOST_APPLY_BCS\n");
        // printf("1 - AMG compute prolong values\n");
        compute_prolongator_values();
        // printf("2 - AMG compute coarse grid values\n");
        compute_coarse_grid_values();
        // _done_post_apply_bcs = true;
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // TODO
    }
    void factor() {}
    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one
    void set_cycle_type(std::string cycle_) {}
    void set_matrix_nsmooth(int nsmooth_) {}
    void set_rbm_nsmooth(int nsmooth_) {}

    void build_coarse_system(Assembler coarse_assembler, Smoother *coarse_smoother) {
        // need to build the coarse smoother from coarse_kmat and then pass that in here..

        // printf("level %d, building coarse system %d=is_coarse_mg\n", level, is_coarse_mg);

        // assert(_done_post_apply_bcs);  // make sure you call post_apply_bcs method after doing
        // bcs
        // printf("build coarse grid system with num_aggregates %d\n", num_aggregates);
        // pointer for either solver and store bool of which one we use
        if (!is_coarse_mg) {
            // then instead build coarse direct solver
            // printf("\tbuild coarse direct solver\n");
            coarse_direct =
                new CoarseDirect(cublasHandle, cusparseHandle, coarse_assembler, coarse_kmat);
        } else {
            // then build coarse AMG solver and new coarse smoother
            // printf("\tbuild coarse AMG solver\n");
            auto no_bcs = DeviceVec<int>(0);
            coarse_mg =
                new CoarseMG(cublasHandle, cusparseHandle, coarse_smoother, num_aggregates,
                             coarse_kmat, coarse_free_kmat, d_Bc_vec, no_bcs, coarse_node_threshold,
                             sparse_threshold, omegaJac, nsmooth, level + 1);
            //
            // coarse_mg->post_apply_bcs(no_bcs);
        }
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // solve this multigrid level (V-cycle)

        // setup rhs and soln with init guess of 0
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("1 - start AMG solve\n");
        cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_inner_soln, 0.0, N * sizeof(T));  // re-zero the solution

        // pre-smooth defect
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("2 - AMG pre-smoothing\n");
        this->smoother->smoothDefect(d_rhs_vec, d_inner_soln_vec, nsmooth);

        // restrict
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("3 - AMG restriction\n");
        d_coarse_rhs_vec.zeroValues();  // zero before add new result
        int nprods = P_nnzb * block_dim2;
        dim3 block0(32), grid0((nprods + 31) / 32);
        k_bsrmv_transpose<T><<<grid0, block0>>>(P_nnzb, block_dim, d_prolong_rows, d_prolong_cols,
                                                d_prolong_vals, d_rhs_vec.getPtr(),
                                                d_coarse_rhs_vec.getPtr());

        // DEBUG: check coarse rhs vec
        // T *h_coarse_rhs = d_coarse_rhs_vec.createHostVec().getPtr();
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     printf("h_coarse_rhs (iagg %d): ", iagg);
        //     printVec<T>(6, &h_coarse_rhs[6 * iagg]);
        // }

        // coarse solve
        if (!is_coarse_mg) {  // direct solve
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("4 - AMG coarse direct solve\n");
            this->coarse_direct->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        } else {
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("4 - AMG pass to coarser AMG solver\n");
            this->coarse_mg->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        }

        // DEBUG: check coarse rhs vec
        // printf("\n\n");
        // T *h_coarse_soln = d_coarse_soln_vec.createHostVec().getPtr();
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     printf("h_coarse_soln (iagg %d): ", iagg);
        //     printVec<T>(6, &h_coarse_soln[6 * iagg]);
        // }

        // prolongation
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("5 - AMG prolongate coarse to fine\n");
        T a = 1.0, b = 0.0;
        int mb = nnodes, nb = num_aggregates;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, P_nnzb, &a,
                                      descrKmat, d_prolong_vals, d_prolong_rowp, d_prolong_cols,
                                      block_dim, d_coarse_soln_vec.getPtr(), &b, d_temp));
        // add to previous inner soln (see bsr_aggregation.py)
        a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_inner_soln, 1));

        // update rhs for defect
        a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_rhs));

        // post-smooth
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("6 - AMG post-smooth\n");
        this->smoother->smoothDefect(d_rhs_vec, d_inner_soln_vec, nsmooth);

        // copy internal soln to external solution of the solve method
        cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);

        return false;
    }

    BsrData get_coarse_bsr_data() { return coarse_kmat_bsr_data; }
    int get_num_aggregates() { return num_aggregates; }
    bool get_coarse_mg() { return is_coarse_mg; }
    BsrMat<DeviceVec<T>> get_coarse_kmat() { return coarse_kmat; }

    // public data
    // --------------------
    Smoother *smoother;
    bool is_coarse_mg;
    CoarseMG *coarse_mg;
    CoarseDirect *coarse_direct;

   private:
    void initCuda() {
        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        // get some host pointers
        h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();
        // compute the host kmat diagonal pointer
        int *h_kmat_diagp = new int[nnodes];
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
                int block_col = h_kmat_cols[jp];
                // printf("row %d, col %d\n", block_row, block_col);
                if (block_row == block_col) {
                    h_kmat_diagp[block_row] = jp;
                }
            }
        }
        d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

        // aggregation sparsities
        d_diag_norms = DeviceVec<T>(nnodes).getPtr();
        d_strength_indicator = DeviceVec<bool>(kmat_nnzb).getPtr();

        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_z = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();

        // for linear solver / precond use
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();
    }

    void form_node_aggregates() {
        // if (level > 0) {
        //     printf("\n\n");
        //     printf("CHECK fine grid kmat values\n");
        //     T *h_Avals = DeviceVec<T>(kmat_nnzb * 36, d_kmat_vals).createHostVec().getPtr();
        //     // printf("tentative prolongator: ");
        //     // printVec<T>(36 * P_nnzb, h_prolong_vals);
        //     for (int inode = 0; inode < nnodes; inode++) {
        //         for (int jp = h_kmat_rowp[inode]; jp < h_kmat_rowp[inode + 1]; jp++) {
        //             int jnode = h_kmat_cols[jp];
        //             printf("A mat on (node %d, node %d): \n", inode, jnode);
        //             for (int irow = 0; irow < 6; irow++) {
        //                 printVec<T>(6, &h_Avals[36 * jp + 6 * irow]);
        //             }
        //         }
        //     }
        // }

        // 1) compute strength pattern on GPU
        k_get_diag_norms<T>
            <<<nnodes, 32>>>(nnodes, d_kmat_diagp, block_dim, d_kmat_free_vals, d_diag_norms);
        // CHECK_CUDA(cudaGetLastError());

        k_compute_strength_bools<T><<<kmat_nnzb, 32>>>(kmat_nnzb, block_dim, d_diag_norms,
                                                       d_kmat_rows, d_kmat_cols, d_kmat_free_vals,
                                                       sparse_threshold, d_strength_indicator);

        // CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2) compute strength indices to host
        h_strength_indicator =
            DeviceVec<bool>(kmat_nnzb, d_strength_indicator).createHostVec().getPtr();
        // printf("h_strength_indicator: ");
        // printVec<bool>(kmat_nnzb, h_strength_indicator);
        int strength_nnz = 0;
        for (int iblock = 0; iblock < kmat_nnzb; iblock++) {
            if (h_strength_indicator[iblock]) strength_nnz++;
        }
        h_strength_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_strength_cols = HostVec<int>(strength_nnz).getPtr();
        for (int i = 0; i < nnodes; i++) {
            h_strength_rowp[i + 1] = h_strength_rowp[i];  // update from sum of last one..
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                int j = h_kmat_cols[jp];
                if (h_strength_indicator[jp]) {
                    h_strength_cols[h_strength_rowp[i + 1]] = j;
                    h_strength_rowp[i + 1]++;
                }
            }
        }

        // printf("strength conn:");
        // for (int i = 0; i < nnodes; i++) {
        //     printf("node %d: ", i);
        //     int start = h_strength_rowp[i];
        //     int end = h_strength_rowp[i + 1];
        //     printVec<int>(end - start, &h_strength_cols[start]);
        // }

        // 3) do greedy serial aggregation pattern on host
        _greedy_serial_aggregation();

        // then init coarse level cuda
        Nc = num_aggregates * block_dim;
        d_coarse_rhs_vec = DeviceVec<T>(Nc);
        d_coarse_rhs = d_coarse_rhs_vec.getPtr();
        d_coarse_soln_vec = DeviceVec<T>(Nc);
        d_coarse_soln = d_coarse_soln_vec.getPtr();
    }

    void _greedy_serial_aggregation() {
        // Assume HostVec<T> is defined appropriately and nnodes, h_strength_rowp, h_strength_cols
        // are valid and accessible.
        h_aggregate_ind = HostVec<int>(nnodes).getPtr();
        memset(h_aggregate_ind, -1, nnodes * sizeof(int));

        num_aggregates = 0;
        // First phase: assign aggregates based on unpicked strong neighbors.
        for (int i = 0; i < nnodes; i++) {
            // check that all strong neighbors are unpicked
            bool any_picked = false;
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                int j = h_strength_cols[jp];
                if (h_aggregate_ind[j] != -1) {
                    any_picked = true;
                    break;
                }
            }
            if (!any_picked) {
                // Only if not any picked, create a new node aggregate (including the node itself)
                for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                    int j = h_strength_cols[jp];
                    h_aggregate_ind[j] = num_aggregates;
                }
                num_aggregates++;
            }
        }
        // printf("phase 1 - h_aggregate_ind: ");
        // printVec<int>(nnodes, h_aggregate_ind);
        // printf("num_aggregates %d\n", num_aggregates);

        // Second phase: assign all remaining nodes to a nearby aggregate
        for (int i = 0; i < nnodes; i++) {
            if (h_aggregate_ind[i] != -1) continue;  // Node already assigned, skip it

            // Collect aggregates from strong neighbors
            std::set<int> nearby_aggregates;
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                int j = h_strength_cols[jp];
                if (h_aggregate_ind[j] != -1) {  // is picked
                    // Insert the aggregate index of neighbor j
                    nearby_aggregates.insert(h_aggregate_ind[j]);
                }
            }

            if (!nearby_aggregates.empty()) {
                // Randomly choose one aggregate from the set
                int set_size = nearby_aggregates.size();
                int random_index =
                    rand() % set_size;  // using rand() for simplicity; consider C++11 random
                                        // generators for production code
                auto it = nearby_aggregates.begin();
                std::advance(it, random_index);
                int chosen_aggregate = *it;
                h_aggregate_ind[i] = chosen_aggregate;
            } else {
                // No nearby aggregate; assign as a new aggregate.
                h_aggregate_ind[i] = num_aggregates++;
            }
        }
        // printf("final - h_aggregate_ind: ");
        // printVec<int>(nnodes, h_aggregate_ind);

        d_aggregate_ind = HostVec<int>(nnodes, h_aggregate_ind).createDeviceVec().getPtr();
    }

    void compute_prolongation_nz_pattern() {
        //--------------------------------------------------------------------------
        // Step 1: Build the tentative prolongator pattern.
        // For this example, each row i is assigned a single tentative entry: i.
        //--------------------------------------------------------------------------
        h_tentative_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_tentative_cols = HostVec<int>(nnodes).getPtr();

        h_tentative_rowp[0] = 0;
        for (int i = 0; i < nnodes; i++) {
            h_tentative_cols[i] = h_aggregate_ind[i];
            h_tentative_rowp[i + 1] = i + 1;
        }

        // printf("h_tentative_rowp: ");
        // printVec<int>(nnodes + 1, h_tentative_rowp);
        // printf("h_tentative_cols: ");
        // printVec<int>(nnodes, h_tentative_cols);

        //--------------------------------------------------------------------------
        // Step 2: Compute the smoothed A*P prolongation pattern.
        // We will use a temporary vector to build the final CSR arrays.
        // For each row i in the prolongator, the pattern is given by:
        //     P(i) = tentative(i) ∪ (⋃ for j in row i of kmat) tentative(j)
        // We use an std::set<int> to guarantee uniqueness.
        //--------------------------------------------------------------------------
        std::vector<int> prolong_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> prolong_cols;                 // column indices for P

        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;

            // Add the tentative prolongation pattern of row i (usually the "diagonal" entry).
            for (int kp = h_tentative_rowp[i]; kp < h_tentative_rowp[i + 1]; kp++) {
                uniqueIndices.insert(h_tentative_cols[kp]);
            }

            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                int j = h_kmat_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that
                // is j).
                for (int kp = h_tentative_rowp[j]; kp < h_tentative_rowp[j + 1]; kp++) {
                    uniqueIndices.insert(h_tentative_cols[kp]);
                }
            }

            // The number of entries for row i is the size of the set.
            prolong_rowp[i + 1] = prolong_rowp[i] + uniqueIndices.size();

            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                prolong_cols.push_back(col);
            }
        }

        //--------------------------------------------------------------------------
        // Step 3: Finalize the prolongator pattern.
        // P_nnzb is the total number of nonzeros in the prolongation operator.
        //--------------------------------------------------------------------------
        P_nnzb = prolong_cols.size();

        // Allocate and copy the final CSR arrays for the prolongator.
        h_prolong_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_prolong_rows = HostVec<int>(P_nnzb).getPtr();
        h_prolong_cols = HostVec<int>(P_nnzb).getPtr();

        memcpy(h_prolong_rowp, prolong_rowp.data(), (nnodes + 1) * sizeof(int));
        memcpy(h_prolong_cols, prolong_cols.data(), P_nnzb * sizeof(int));

        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                h_prolong_rows[jp] = i;
            }
        }

        // printf("prolong fillin from nnodes %d to nnzb %d\n", nnodes, P_nnzb);
        // for (int i = 0; i < nnodes; i++) {
        //     int tstart = h_tentative_rowp[i], tend = h_tentative_rowp[i + 1];
        //     printf("\nT node %d: ", i);
        //     printVec<int>(tend - tstart, &h_tentative_cols[tstart]);
        //     int pstart = h_prolong_rowp[i], pend = h_prolong_rowp[i + 1];
        //     printf("P node %d: ", i);
        //     printVec<int>(pend - pstart, &h_prolong_cols[pstart]);
        // }
        // printf("h_prolong_rowp: ");
        // printVec<int>(nnodes + 1, h_prolong_rowp);
        // printf("h_prolong_cols: ");
        // printVec<int>(P_nnzb, h_prolong_cols);

        d_prolong_rowp = HostVec<int>(nnodes + 1, h_prolong_rowp).createDeviceVec().getPtr();
        d_prolong_rows = HostVec<int>(P_nnzb, h_prolong_rows).createDeviceVec().getPtr();
        d_prolong_cols = HostVec<int>(P_nnzb, h_prolong_cols).createDeviceVec().getPtr();
        d_prolong_vals = DeviceVec<T>(P_nnzb * block_dim2).getPtr();
        d_Z_vals = DeviceVec<T>(P_nnzb * block_dim2).getPtr();

        // 4) compute the block locations of each part of tentative prolongator
        h_tentative_block_map = HostVec<int>(nnodes).getPtr();
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                if (j == h_aggregate_ind[i]) {
                    h_tentative_block_map[i] = jp;
                }
            }
        }
        d_tentative_block_map =
            HostVec<int>(nnodes, h_tentative_block_map).createDeviceVec().getPtr();
    }

    template <bool startup = true>
    void _compute_diag_vals() {
        // first need to construct rowp and cols for diagonal (fairly easy)

        // startup section
        int ndiag_vals = block_dim * block_dim * nnodes;
        // printf("diag vals part 1: startup\n");
        if constexpr (startup) {
            int *h_diag_rowp = new int[nnodes + 1];
            diag_inv_nnzb = nnodes;
            int *h_diag_cols = new int[nnodes];
            h_diag_rowp[0] = 0;

            for (int i = 0; i < nnodes; i++) {
                h_diag_rowp[i + 1] = i + 1;
                h_diag_cols[i] = i;
            }

            // now copy to device
            d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
            d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

            // create the bsr data object on device
            d_diag_bsr_data = BsrData(nnodes, block_dim, diag_inv_nnzb, d_diag_rowp, d_diag_cols,
                                      nullptr, nullptr, false);
            delete[] h_diag_rowp;
            delete[] h_diag_cols;

            // now allocate DeviceVec for the values
            d_diag_vec = DeviceVec<T>(ndiag_vals);
            d_diag_LU_vals = d_diag_vec.getPtr();  // just copy these pointers..
        }                                          // end of startup
        // CHECK_CUDA(cudaDeviceSynchronize());

        // regular jacobi preconditioner
        // printf("diag vals part 2: copy diag values from kmat\n");
        //  zero previous values (to get new Dinv, in case optimization or nonlinear problem)
        d_diag_vec.zeroValues();  // this is vector for the opinter d_diag_LU_vals (confusing, can
                                  // fix later
        k_copyBlockDiagFromBsrMat<T><<<(ndiag_vals + 31) / 32, 32>>>(
            nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // ilu0 factoriation
        // printf("diag vals part 3: perform ILU0 startup\n");
        if constexpr (startup) {
            // create M matrix object (for full numeric factorization)
            cusparseCreateMatDescr(&descr_M);
            cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseCreateBsrilu02Info(&info_M);

            // init L matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_L);
            cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
            cusparseCreateBsrsv2Info(&info_L);

            // init U matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_U);
            cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
            cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseCreateBsrsv2Info(&info_U);

            // symbolic and numeric factorizations
            CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, &pBufferSize_M));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, &pBufferSize_L));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, &pBufferSize_U));
            pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
            // cudaMalloc((void **)&pBuffer, pBufferSize);
            cudaMalloc(&pBuffer, pBufferSize);

            // perform ILU symbolic factorization on L
            CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, policy_M, pBuffer));
            status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
            if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
                printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
            }

            // analyze sparsity patern of L for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, policy_L, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());

            // analyze sparsity pattern of U for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, policy_U, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("diag vals part 4 : ILU0 numeric factorization\n");
        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                                         info_M, policy_M, pBuffer));
        // CHECK_CUDA(cudaDeviceSynchronize());
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        // startup part of Dinv linear operator
        if constexpr (startup) {
            d_dinv_vec = DeviceVec<T>(ndiag_vals);
            d_dinv_vals = d_dinv_vec.getPtr();
        }

        // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
        // to get effective matrix.. need six temp vectors..
        // printf("diag vals part 5: compute Dinv by applying triang solves 6 times\n");
        for (int i = 0; i < block_dim; i++) {
            // set d_temp to ei (one of e1 through e6 per block)
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            dim3 block(32);
            dim3 grid((nnodes + 31) / 32);
            k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

            // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
                pBuffer));  // prob only need U^-1 part for block diag.. TBD

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

            // now copy temp2 into columns of new operator
            dim3 grid2((N + 31) / 32);
            k_setLUinv_operator<T>
                <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vec.getPtr());
        }  // this works!
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* CG-lanczos spectral radius section */
    void _compute_spectral_radius() {
        // temporarily rename some temp vecs/pointers for CG style coefficients
        T *d_x = d_inner_soln;
        T *d_p = d_temp;
        T *d_w = d_temp2;
        // lastly d_z already covered

        /* first run n_lanczos steps of CG (with only jacobi preconditioner) */
        // code reused from PCG (since don't want duplicate memory by extra PCG object, and
        // BaseSolver makes it so I can't easily call it as jacobi precond) I also don't have the
        // grid object to easily make PCG, anyways could generalize / cleanup later, just get this
        // working for now
        cudaMemset(d_x, 0.0, N * sizeof(T));
        cudaMemcpy(d_resid, d_lanczos_loads_vec.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        T rho_prev, rho;  // coefficients that we need to remember
        // inner loop
        for (int j = 0; j < N_LANCZOS; j++) {
            // compute z = Dinv*r
            T a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          diag_inv_nnzb, &a, descrKmat, d_dinv_vals, d_diag_rowp,
                                          d_diag_cols, block_dim, d_resid, &b, d_z));
            // compute dot products, rho = <r, z>
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));
            if (j == 0) {
                // first iteration, p := z
                cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // compute beta and record it
                beta_vals[j - 1] = rho / rho_prev;
                // p_new = z + beta * p in two steps
                a = beta_vals[j - 1];  // p *= beta scalar
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
                a = 1.0;  // p += z
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));
            }
            // store rho for next iteration (prev), only used in this part
            rho_prev = rho;
            // compute w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_p, &b, d_w));
            // compute alpha = <r,z> / <w,p> = rho / <w,p>
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            alpha_vals[j] = rho / wp0;
            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha_vals[j], d_p, 1, d_x, 1));
            // r -= alpha * w
            a = -alpha_vals[j];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));
        }
        // then record the last CG coefficient
        // z = Dinv*r
        T a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                      diag_inv_nnzb, &a, descrKmat, d_dinv_vals, d_diag_rowp,
                                      d_diag_cols, block_dim, d_resid, &b, d_z));
        // compute rho = <r, z>
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));
        // compute last beta
        beta_vals[N_LANCZOS - 1] = rho / rho_prev;

        /* now compute equivalent lanczos coefficients */
        for (int j = 0; j < N_LANCZOS; j++) {
            delta_vals[j] = (j == 0) ? (1.0 / alpha_vals[j])
                                     : (1.0 / alpha_vals[j] + beta_vals[j - 1] / alpha_vals[j - 1]);
            eta_vals[j] = sqrt(beta_vals[j]) / alpha_vals[j];
        }

        /* now get spectral radius from LAPACKe small tridiag matrix eigval solve on the host */
        int info = LAPACKE_dstev(LAPACK_ROW_MAJOR,  // matrices stored row-major in C++
                                 'N',               // compute eigenvalues only
                                 N_LANCZOS,
                                 delta_vals,  // diagonal
                                 eta_vals,    // off-diagonal
                                 nullptr,     // no eigenvectors
                                 N_LANCZOS);
        // max eigenvalue (as it overwrites eigvals into delta_vals in-place)
        T max_eigval = delta_vals[0];
        for (int i = 1; i < N_LANCZOS; i++) {
            if (delta_vals[i] > max_eigval) max_eigval = delta_vals[i];
        }
        // and set this as spectral radius estimate (recommend omega = 0.9 or something so we are
        // consrevative)
        spectral_radius = max_eigval;
        // print current max spectral radius for DEBUG
        if (debug) printf("spectral radius %.8e\n", spectral_radius);
    }

    void compute_prolongator_values() {
        // 1) compute tentative prolongator with QR factorization for each aggregate
        // printf("4.1 - compute prolongator: QR factorization\n");
        _graham_schmidt_QR_factorization();
        _apply_dirichlet_bcs();
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2) compute spectral radius of fine grid matrix
        // printf("4.2 - compute prolongator: compute diag values\n");
        _compute_diag_vals();
        CHECK_CUDA(cudaDeviceSynchronize());
        // _compute_spectral_radius(); // don't do this for now..

        // 3) compute smoothed prolongator
        // compute -omega/rho(Dinv*A) * beta_k * A*P into Z first (scaled prolong defect matrix)
        // printf("4.3 - compute prolongator: smooth prolongator\n");
        compute_matmat_prod_nz_pattern();
        _smooth_prolongator();
        _apply_dirichlet_bcs();
    }

    void _apply_dirichlet_bcs() {
        dim3 block(32);
        int nbcs = d_bcs.getSize();
        // printf("nbcs = %d\n", nbcs);
        if (nbcs == 0) return;
        dim3 grid((nbcs + 31) / 32);
        // printf("applying dirichlet bcs with %d #bcs\n", nbcs);
        // int *h_bcs = d_bcs.createHostVec().getPtr();
        // printf("h_bcs: ");
        // printVec<int>(nbcs, h_bcs);

        // launch two kernels asynchronously
        apply_mat_bcs_P_kernel<T, DeviceVec>
            <<<grid, block>>>(d_bcs, block_dim, d_prolong_rowp, d_prolong_cols, d_prolong_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void _graham_schmidt_QR_factorization() {
        // compute tentative prolongator using QR factorization designed explicitly for N x 6 matrix
        // loop over each of 6 columns doing Graham-Schmidt decomp to find P * B = Bc
        // where B is known and the rigid body modes of fine grid

        d_Bc_vec = DeviceVec<T>(block_dim2 * num_aggregates);
        d_aggregate_norms2 = DeviceVec<T>(num_aggregates).getPtr();

        // copy fine grid rigid body modes into tentative prolongator (with fillin pattern)
        int nvals_rbm = block_dim2 * nnodes;  // nvals in tentative prolong + rigid_body_modes array
        k_copy_rbm_into_tentative_prolongator<T><<<(nvals_rbm + 31) / 32, 32>>>(
            nnodes, block_dim, d_tentative_block_map, rigid_body_modes.getPtr(), d_prolong_vals);

        // T *h_prolong_vals0 = DeviceVec<T>(P_nnzb * 36, d_prolong_vals).createHostVec().getPtr();
        // // printf("tentative prolongator: ");
        // // printVec<T>(36 * P_nnzb, h_prolong_vals);
        // for (int inode = 0; inode < nnodes; inode++) {
        //     for (int jp = h_prolong_rowp[inode]; jp < h_prolong_rowp[inode + 1]; jp++) {
        //         int iagg = h_prolong_cols[jp];
        //         printf("tentative prolong on (node %d, agg %d): \n", inode, iagg);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_prolong_vals0[36 * jp + 6 * irow]);
        //         }
        //     }
        // }

        // perform block-norm of each
        for (int imode = 0; imode < block_dim; imode++) {
            // compute orthogonalization against previous modes
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("QR mode %d / 6\n", imode);
            for (int jmode = 0; jmode < imode; jmode++) {
                // compute inner products with previous modes and store in d_Bc_vec (imode, jmode)
                // row and col of that aggregate
                // printf("\tQR inner prod (%d,%d)\n", imode, jmode);
                k_compute_GS_inner_product<T><<<(N + 31) / 32, 32>>>(
                    imode, jmode, nnodes, block_dim, d_aggregate_ind, d_tentative_block_map,
                    d_prolong_vals, d_Bc_vec.getPtr());

                // subtract inner product multiple of previous mode in Graham-Schmidt (to
                // orthogonalize against previous modes)
                // CHECK_CUDA(cudaDeviceSynchronize());
                // printf("\tQR subtract projector (%d,%d)\n", imode, jmode);
                k_remove_GS_projector_mode<T><<<(N + 31) / 32, 32>>>(
                    imode, jmode, nnodes, block_dim, d_aggregate_ind, d_tentative_block_map,
                    d_Bc_vec.getPtr(), d_prolong_vals);
            }

            // compute norms in each aggregate and then normalize P
            // CHECK_CUDA(cudaDeviceSynchronize());
            cudaMemset(d_aggregate_norms2, 0.0,
                       num_aggregates * sizeof(T));  // re-zero d_aggregate_norms2
            // printf("\tQR ||mode %d||^2 with N = %d\n", imode, N);
            k_compute_aggregate_norms2<T>
                <<<(N + 31) / 32, 32>>>(imode, nnodes, block_dim, d_aggregate_ind,
                                        d_tentative_block_map, d_prolong_vals, d_aggregate_norms2);
            // CHECK_CUDA(cudaDeviceSynchronize());
            // T *h_aggregate_norms2 =
            //     DeviceVec<T>(num_aggregates, d_aggregate_norms2).createHostVec().getPtr();
            // printf("h_aggregate_norms2: ");
            // printVec<T>(num_aggregates, h_aggregate_norms2);

            // printf("\tQR get sqrt norm %d with num agg %d\n", imode, num_aggregates);
            k_compute_sqrt_norms<T><<<(num_aggregates + 31) / 32, 32>>>(
                imode, num_aggregates, block_dim, d_aggregate_norms2,
                d_Bc_vec.getPtr());  // and store norm2 in Bc norm
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("\tQR normalize mode %d in P\n", imode);
            k_normalize_tentative_modes<T>
                <<<(N + 31) / 32, 32>>>(imode, nnodes, block_dim, d_aggregate_ind,
                                        d_tentative_block_map, d_Bc_vec.getPtr(), d_prolong_vals);
        }

        // T *h_B_vals = rigid_body_modes.createHostVec().getPtr();
        // for (int inode = 0; inode < nnodes; inode++) {
        //     printf("Bf agg %d: \n", inode);
        //     for (int irow = 0; irow < 6; irow++) {
        //         printVec<T>(6, &h_B_vals[36 * inode + 6 * irow]);
        //     }
        // }

        // // check the computed values..
        // printf("\n\n");
        // T *h_Bc_vals = d_Bc_vec.createHostVec().getPtr();
        // // printf("coarse rigid body modes: ");
        // // printVec<T>(36 * num_aggregates, h_Bc_vals);
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     printf("Bc agg %d: \n", iagg);
        //     for (int irow = 0; irow < 6; irow++) {
        //         printVec<T>(6, &h_Bc_vals[36 * iagg + 6 * irow]);
        //     }
        // }

        // _apply_dirichlet_bcs();  // DEBUG
        // printf("\n\n");
        // T *h_prolong_vals = DeviceVec<T>(P_nnzb * 36, d_prolong_vals).createHostVec().getPtr();
        // for (int inode = 0; inode < nnodes; inode++) {
        //     for (int jp = h_prolong_rowp[inode]; jp < h_prolong_rowp[inode + 1]; jp++) {
        //         int iagg = h_prolong_cols[jp];
        //         printf("tentative prolong on (node %d, agg %d): \n", inode, iagg);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_prolong_vals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }
    }

    void _smooth_prolongator() {
        // Z_mat->zeroValues();
        cudaMemset(d_Z_vals, 0.0, P_nnzb * block_dim2 * sizeof(T));
        dim3 PKP_block(216), PKP_grid(nnzb_prod);
        T a = -omegaJac;
        // T a = -omega / spectral_radius; // if just jacobi
        // printf("4.3.1 - smoothprolong: K*P => Z\n");
        k_compute_mat_mat_prod<T><<<PKP_grid, PKP_block>>>(nnzb_prod, block_dim, a, d_K_prodBlocks,
                                                           d_P_prodBlocks, d_Z_prodBlocks,
                                                           d_kmat_vals, d_prolong_vals, d_Z_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // compute Dinv*Z into Z in-place (equiv to Dinv*scale*A*P => Z)
        dim3 DP_block(216), DP_grid(P_nnzb);
        // printf("4.3.2 - smoothprolong: Dinv*Z => Z\n");
        k_compute_Dinv_P_mmprod<T>
            <<<DP_grid, DP_block>>>(P_nnzb, block_dim, d_dinv_vals, d_prolong_rows, d_Z_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // do orthogonal projector on Z (only really needed for coarse-grid galerkin AMG,
        // not smooth GMG)
        if constexpr (ORTHOG_PROJECTOR) {
            // compute free variables
            auto free_var_vec = DeviceVec<bool>(N);
            free_var_vec.setFullVecToConstValue(true);  // set all to default true meaning free var
            d_free_dof = free_var_vec.getPtr();

            dim3 OP_block(32), OP_grid(nnodes);
            // d_SU_vals.zeroValues();
            int ndiag_vals = block_dim * block_dim * nnodes;
            d_SU_vals = DeviceVec<T>(ndiag_vals);
            // compute SU matrix
            // printf("4.3.3 - smoothprolong: compute SU vals\n");
            k_orthog_projector_computeSU<T><<<OP_grid, OP_block>>>(
                nnodes, block_dim, d_Bc_vec.getPtr(), d_free_dof, d_prolong_rowp, d_prolong_cols,
                d_prolong_vals, d_SU_vals.getPtr());
            // CHECK_CUDA(cudaDeviceSynchronize());

            d_UTU_vals = DeviceVec<T>(ndiag_vals);
            // printf("4.3.4 - smoothprolong: compute UTU vals\n");
            k_orthog_projector_computeUTU<T>
                <<<OP_grid, OP_block>>>(nnodes, block_dim, d_Bc_vec.getPtr(), d_free_dof,
                                        d_prolong_rowp, d_prolong_cols, d_UTU_vals.getPtr());
            // CHECK_CUDA(cudaDeviceSynchronize());

            // now compute the LU factor and inverse matrix UTUinv for each fine node (same size and
            // like Dinv matrix) reuse same pointers and nnzb sizes as Dinv cause same dimensions
            // printf("4.3.5 - smoothprolong: perform ILU0 factorization of UTU\n");
            CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                                 d_UTU_vals.getPtr(), d_diag_rowp, d_diag_cols,
                                                 trans_L, trans_U, policy_L, policy_U, dir);
            // CHECK_CUDA(cudaDeviceSynchronize());

            // now compute UTUinv linear operator like I did for the Dinv
            d_UTUinv_vals = DeviceVec<T>(ndiag_vals);  // inv linear operator of UTU
            // printf("4.3.6 - smoothprolong: compute UTU_inv from 6 LU solves each\n");
            for (int i = 0; i < block_dim; i++) {
                // set d_temp to ei (one of e1 through e6 per block)
                cudaMemset(d_temp, 0.0, N * sizeof(T));
                dim3 block(32);
                dim3 grid((nnodes + 31) / 32);
                k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

                // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L,
                    d_UTU_vals.getPtr(), d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp,
                    d_resid, policy_L,
                    pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes, nnodes,
                                                     &alpha, descr_U, d_UTU_vals.getPtr(),
                                                     d_diag_rowp, d_diag_cols, block_dim, info_U,
                                                     d_resid, d_temp2, policy_U, pBuffer));

                // now copy temp2 into columns of new operator
                dim3 grid2((N + 31) / 32);
                k_setLUinv_operator<T>
                    <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_UTUinv_vals.getPtr());

                // remove rigid-body row-sums
                // printf("4.3.7 - smoothprolong: apply orthog projector with SU and UTUinv\n");
                k_orthog_projector_removeRowSums<T><<<OP_grid, OP_block>>>(
                    nnodes, block_dim, d_Bc_vec.getPtr(), d_free_dof, d_prolong_rowp,
                    d_prolong_cols, d_SU_vals.getPtr(), d_UTUinv_vals.getPtr(), d_prolong_vals);
            }
            // CHECK_CUDA(cudaDeviceSynchronize());
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // add Z into P (the prolongation update)
        dim3 add_block(64);
        T scale = 1.0;
        // printf("4.3.8 - smoothprolong: add colored submat update P += Z\n");
        k_add_colored_submat_PFP<T>
            <<<DP_grid, add_block>>>(P_nnzb, block_dim, scale, 0, d_Z_vals, d_prolong_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("\n\n");
        // printf("CHECK SMOOTH PROLONG VALUES\n");
        // T *h_prolong_vals = DeviceVec<T>(P_nnzb * 36, d_prolong_vals).createHostVec().getPtr();
        // for (int inode = 0; inode < nnodes; inode++) {
        //     for (int jp = h_prolong_rowp[inode]; jp < h_prolong_rowp[inode + 1]; jp++) {
        //         int iagg = h_prolong_cols[jp];
        //         printf("smoothed prolong on (node %d, agg %d): \n", inode, iagg);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_prolong_vals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }
    }

    void compute_matmat_prod_nz_pattern() {
        // get pointers

        nnzb_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add K*P
                                          // fillin (for better prolong)
                    for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
                        int j2 = h_prolong_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    nnzb_prod++;
                }
            }
        }
        // printf("nnzb_prod = %d\n", nnzb_prod);
        // now allocate the block indices of the product
        int *h_PF_blocks = new int[nnzb_prod];
        int *h_K_blocks = new int[nnzb_prod];
        int *h_P_blocks = new int[nnzb_prod];
        memset(h_PF_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_K_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_P_blocks, 0, nnzb_prod * sizeof(int));
        int inz_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;
                    int _jp2 = -1;
                    for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
                        int j2 = h_prolong_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                            _jp2 = jp2;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    h_PF_blocks[inz_prod] = jp;
                    h_K_blocks[inz_prod] = kp;
                    h_P_blocks[inz_prod] = _jp2;
                    inz_prod++;
                }
            }
        }
        // now allocate onto the device
        d_Z_prodBlocks = HostVec<int>(nnzb_prod, h_PF_blocks).createDeviceVec().getPtr();
        d_K_prodBlocks = HostVec<int>(nnzb_prod, h_K_blocks).createDeviceVec().getPtr();
        d_P_prodBlocks = HostVec<int>(nnzb_prod, h_P_blocks).createDeviceVec().getPtr();

        // printf("DEBUG: PF_nnzb = %d, nnzb_prod %d\n", P_nnzb, nnzb_prod);
    }

    void compute_coarse_grid_nz_pattern() {
        // 1) compute P^T nonzero pattern (restriction)
        std::vector<int> prolong_tr_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> prolong_tr_cols;                 // column indices for P

        h_prolong_tr_row_cts = HostVec<int>(num_aggregates).getPtr();
        h_prolong_tr_rowp = HostVec<int>(num_aggregates + 1).getPtr();
        h_prolong_tr_cols = HostVec<int>(P_nnzb).getPtr();

        printf("coarse_grid_nz 1 - get P^T pattern\n");

        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                h_prolong_tr_row_cts[j]++;
            }
        }

        for (int i = 0; i < num_aggregates; i++) {
            h_prolong_tr_rowp[i + 1] = h_prolong_tr_rowp[i] + h_prolong_tr_row_cts[i];
        }

        // reset to zero
        memset(h_prolong_tr_row_cts, 0, num_aggregates * sizeof(int));
        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                int ip = h_prolong_tr_rowp[j] + h_prolong_tr_row_cts[j];
                h_prolong_tr_cols[ip] = i;
                h_prolong_tr_row_cts[j]++;
            }
        }
        // printf("h_prolong_tr_rowp: ");
        // printVec<int>(num_aggregates + 1, h_prolong_tr_rowp);
        // printf("h_prolong_tr_cols: ");
        // printVec<int>(P_nnzb, h_prolong_tr_cols);

        // also compute map between P and P^T block storage since I don't have that storage
        int *h_prolong_tr_map = new int[P_nnzb];  // PT block input => P block output
        for (int iagg = 0; iagg < num_aggregates; iagg++) {
            for (int jp = h_prolong_tr_rowp[iagg]; jp < h_prolong_tr_rowp[iagg + 1]; jp++) {
                int jnode = h_prolong_tr_cols[jp];

                // now find equivalent block ind in prolong
                for (int kp = h_prolong_rowp[jnode]; kp < h_prolong_rowp[jnode + 1]; kp++) {
                    int kagg = h_prolong_cols[kp];
                    if (kagg == iagg) {
                        h_prolong_tr_map[jp] = kp;
                        // printf("h_prolong_tr_map on (node %d, agg %d) with jp %d => kp %d\n",
                        // jnode,
                        //        iagg, jp, kp);
                        // break;
                    }
                }
            }
        }
        // printf("h_prolong_tr_map: ");
        // printVec<int>(P_nnzb, h_prolong_tr_map);

        // 2) compute A*P nonzero pattern
        // printf("coarse_grid_nz 2 - compute A*P pattern\n");
        std::vector<int> AP_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> AP_cols;                 // column indices for P
        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // Add the tentative prolongation pattern of row i (usually the "diagonal" entry).
            for (int kp = h_prolong_rowp[i]; kp < h_prolong_rowp[i + 1]; kp++) {
                uniqueIndices.insert(h_prolong_cols[kp]);
            }
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                int j = h_kmat_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that
                // is j).
                for (int kp = h_prolong_rowp[j]; kp < h_prolong_rowp[j + 1]; kp++) {
                    uniqueIndices.insert(h_prolong_cols[kp]);
                }
            }

            // The number of entries for row i is the size of the set.
            AP_rowp[i + 1] = AP_rowp[i] + uniqueIndices.size();

            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                AP_cols.push_back(col);
            }
        }

        // 3) compute P^T * (AP) nz pattern now
        // printf("coarse_grid_nz 3 - compute P^T * A * P pattern\n");
        // printf("\tnum agg = %d\n", num_aggregates);
        int num_coarse = num_aggregates;
        std::vector<int> PTAP_rowp(num_coarse + 1, 0);
        std::vector<int> PTAP_cols;
        for (int i = 0; i < num_coarse; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that
                // is j).
                for (int kp = AP_rowp[j]; kp < AP_rowp[j + 1]; kp++) {
                    uniqueIndices.insert(AP_cols[kp]);
                }
            }

            // The number of entries for row i is the size of the set.
            PTAP_rowp[i + 1] = PTAP_rowp[i] + uniqueIndices.size();

            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                PTAP_cols.push_back(col);
            }
        }

        PTAP_nnzb = PTAP_rowp[num_coarse] * 1;
        h_PTAP_rowp = HostVec<int>(num_coarse + 1).getPtr();
        h_PTAP_cols = HostVec<int>(PTAP_nnzb).getPtr();
        memcpy(h_PTAP_rowp, PTAP_rowp.data(), (num_coarse + 1) * sizeof(int));
        memcpy(h_PTAP_cols, PTAP_cols.data(), PTAP_nnzb * sizeof(int));

        // now compute LU fillin for direct solve if necessary..
        if (!is_coarse_mg) {
            // printf("MAKING coarse LU pattern for direct solve\n");
            auto c_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, h_PTAP_rowp,
                                      h_PTAP_cols, nullptr, nullptr, true);
            c_bsr_data.compute_full_LU_pattern(10.0, false);
            // now get new nnzb, rowp and cols
            delete[] h_PTAP_rowp;
            delete[] h_PTAP_cols;
            PTAP_nnzb = c_bsr_data.nnzb;
            h_PTAP_rowp = c_bsr_data.rowp;
            h_PTAP_cols = c_bsr_data.cols;
        } else {
            // is coarse MG
            auto c_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, h_PTAP_rowp,
                                      h_PTAP_cols, nullptr, nullptr, true);
            c_bsr_data.compute_nofill_pattern();
            // now get new nnzb, rowp and cols
            delete[] h_PTAP_rowp;
            delete[] h_PTAP_cols;
            PTAP_nnzb = c_bsr_data.nnzb;
            h_PTAP_rowp = c_bsr_data.rowp;
            h_PTAP_cols = c_bsr_data.cols;
        }

        d_PTAP_rowp = HostVec<int>(num_coarse + 1, h_PTAP_rowp).createDeviceVec().getPtr();
        d_PTAP_cols = HostVec<int>(PTAP_nnzb, h_PTAP_cols).createDeviceVec().getPtr();
        // assign Kc or PTAP coarse grid matrix values
        d_PTAP_vec = DeviceVec<T>(block_dim2 * PTAP_nnzb);
        d_PTAP_vals = d_PTAP_vec.getPtr();
        d_PTAP_free_vec = DeviceVec<T>(block_dim2 * PTAP_nnzb);
        d_PTAP_free_vals = d_PTAP_free_vec.getPtr();

        // 4) compute nonzero product block pattern..
        // printf("coarse_grid_nz 4 - compute P^T * A * P 6x6 block triple-mat prod patterns\n");
        PTAP_nnzb_prod = 0;
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k + 1]; lp++) {
                        int l = h_prolong_cols[lp];
                        PTAP_nnzb_prod++;
                    }
                }
            }
        }
        // printf("\tPTAP_nnzb_prod = %d\n", PTAP_nnzb_prod);
        h_PTAP_Kc_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P1_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P2_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();

        int inzb_prod = 0;
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k + 1]; lp++) {
                        int l = h_prolong_cols[lp];

                        // find block entry in PTAP matrix
                        int _mp = -1;
                        for (int mp = h_PTAP_rowp[i]; mp < h_PTAP_rowp[i + 1]; mp++) {
                            int m = h_PTAP_cols[mp];
                            if (m == l) {
                                _mp = mp;
                            }
                        }
                        if (_mp < 0) {
                            printf(
                                "BAD PTAP MAP: coarse row %d missing coarse col %d (j=%d k=%d "
                                "lp=%d kp=%d jp=%d)\n",
                                i, l, j, k, lp, kp, jp);
                            fflush(stdout);
                            abort();
                        }
                        int jp_untr =
                            h_prolong_tr_map[jp];  // P^T to P storage since we don't store P^T
                        h_PTAP_Kc_blocks[inzb_prod] = _mp;      // output Kc
                        h_PTAP_P1_blocks[inzb_prod] = jp_untr;  // transpose P
                        h_PTAP_K_blocks[inzb_prod] = kp;        // K
                        h_PTAP_P2_blocks[inzb_prod] = lp;       // P on right
                        inzb_prod++;
                    }
                }
            }
        }

        // DEBUG block product pattern
        // for (int iblock = 0; iblock < PTAP_nnzb_prod; iblock++) {
        //     if (h_PTAP_Kc_blocks[iblock] == 0) {
        //         // Kc(0,0) block, check which blocks of P1, K, P2 we incur
        //         int P1_block = h_PTAP_P1_blocks[iblock];
        //         int K_block = h_PTAP_K_blocks[iblock];
        //         int P2_block = h_PTAP_P2_blocks[iblock];
        //         printf("block %d in PTAP(0,0) prod\n", iblock);
        //         printf("\tP1_block %d, K_block %d, P2_block %d\n", P1_block, K_block, P2_block);

        //         // now figure out which nodes, agg they correspond to each..
        //         for (int i = 0; i < num_aggregates; i++) {
        //             for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
        //                 int j = h_prolong_tr_cols[jp];
        //                 if (jp == P1_block) {
        //                     printf("\tPT_block (node %d, iagg %d)\n", j, i);
        //                 }
        //             }
        //         }

        //         for (int i = 0; i < nnodes; i++) {
        //             for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
        //                 int j = h_kmat_cols[jp];
        //                 if (jp == K_block) {
        //                     printf("\tK_block (node %d, node %d)\n", i, j);
        //                 }
        //             }
        //         }

        //         for (int i = 0; i < nnodes; i++) {
        //             for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
        //                 int j = h_prolong_cols[jp];
        //                 if (jp == P2_block) {
        //                     printf("\tP_block (node %d, node %d)\n", i, j);
        //                 }
        //             }
        //         }
        //     }
        // }

        // DEBUG temporarily change to just do one block product (for nxe = 2 case)
        // PTAP_nnzb_prod = 1;
        // h_PTAP_Kc_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        // h_PTAP_P1_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        // h_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        // h_PTAP_P2_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        // h_PTAP_Kc_blocks[0] = 0;
        // h_PTAP_P1_blocks[0] = 4;
        // h_PTAP_K_blocks[0] = 24;
        // h_PTAP_P2_blocks[0] = 4;

        // printf("coarse grid product pattern with nnzb_prod %d\n", PTAP_nnzb_prod);
        // printf("\tnote also P_nnzb %d, K_nnzb %d, Kc_nnzb %d\n", P_nnzb, kmat_nnzb, PTAP_nnzb);
        // printf("h_PTAP_Kc_blocks: ");
        // printVec<int>(PTAP_nnzb_prod, h_PTAP_Kc_blocks);
        // printf("h_PTAP_P1_blocks: ");
        // printVec<int>(PTAP_nnzb_prod, h_PTAP_P1_blocks);
        // printf("h_PTAP_K_blocks: ");
        // printVec<int>(PTAP_nnzb_prod, h_PTAP_K_blocks);
        // printf("h_PTAP_P2_blocks: ");
        // printVec<int>(PTAP_nnzb_prod, h_PTAP_P2_blocks);
        // printf("h_PTAP_Kc_blocks: ");
        // printVec<int>(100, h_PTAP_Kc_blocks);
        // printf("h_PTAP_P1_blocks: ");
        // printVec<int>(100, h_PTAP_P1_blocks);
        // printf("h_PTAP_K_blocks: ");
        // printVec<int>(100, h_PTAP_K_blocks);
        // printf("h_PTAP_P2_blocks: ");
        // printVec<int>(100, h_PTAP_P2_blocks);

        // printf("h_Ac_rowp: ");
        // printVec<int>(num_aggregates + 1, h_PTAP_rowp);
        // printf("h_Ac_cols: ");
        // printVec<int>(PTAP_nnzb, h_PTAP_cols);

        // put prod blocks on GPU
        // printf("coarse_grid_nz 5 - allocate block prod patterns on GPU\n");
        d_PTAP_Kc_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_Kc_blocks).createDeviceVec().getPtr();
        d_PTAP_P1_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_P1_blocks).createDeviceVec().getPtr();
        d_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_K_blocks).createDeviceVec().getPtr();
        d_PTAP_P2_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_P2_blocks).createDeviceVec().getPtr();
    }

    // void _debug_device_prodmap() {
    //     // convert all necessary vecs to host..

    //     int *h_PTAP_Kc_blocks2 =
    //         DeviceVec<int>(PTAP_nnzb, h_PTAP_Kc_blocks).createHostVec().getPtr();
    //     int *h_PTAP_P1_blocks2 =
    //         DeviceVec<int>(PTAP_nnzb, d_PTAP_P1_blocks).createHostVec().getPtr();
    //     int *h_PTAP_K_blocks2 = DeviceVec<int>(PTAP_nnzb,
    //     d_PTAP_K_blocks).createHostVec().getPtr(); int *h_PTAP_P2_blocks2 =
    //         DeviceVec<int>(PTAP_nnzb, d_PTAP_P2_blocks).createHostVec().getPtr();
    // }

    void compute_coarse_grid_values() {
        // 1) compute coarse grid Galerkin product Ac = P^T * A * P, and Ac_free = P^T * Afree * Ap
        cudaMemset(d_PTAP_vals, 0.0, PTAP_nnzb * block_dim2 * sizeof(T));
        cudaMemset(d_PTAP_free_vals, 0.0, PTAP_nnzb * block_dim2 * sizeof(T));

        cudaPointerAttributes attr;

        //         auto check_ptr = [&](const char *name, const void *ptr) {
        //             auto err = cudaPointerGetAttributes(&attr, ptr);
        //             printf("%s ptr=%p  err=%s", name, ptr, cudaGetErrorString(err));
        // #if CUDART_VERSION >= 10000
        //             if (err == cudaSuccess)
        //                 printf(" type=%d\n", (int)attr.type);
        //             else
        //                 printf("\n");
        // #else
        //             if (err == cudaSuccess)
        //                 printf(" memoryType=%d\n", (int)attr.memoryType);
        //             else
        //                 printf("\n");
        // #endif
        //             fflush(stdout);
        //         };

        // check_ptr("d_PTAP_Kc_blocks", d_PTAP_Kc_blocks);
        // check_ptr("d_PTAP_P1_blocks", d_PTAP_P1_blocks);
        // check_ptr("d_PTAP_K_blocks", d_PTAP_K_blocks);
        // check_ptr("d_PTAP_P2_blocks", d_PTAP_P2_blocks);
        // check_ptr("d_prolong_vals", d_prolong_vals);
        // check_ptr("d_kmat_vals", d_kmat_vals);
        // check_ptr("d_PTAP_vals", d_PTAP_vals);

        // cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, 0);
        // printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
        // printf("maxGridSize = %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
        //        prop.maxGridSize[2]);
        // printf("sharedMemPerBlock = %zu\n", prop.sharedMemPerBlock);
        // fflush(stdout);

        // _debug_device_prodmap();

        // printf("\tcompute coarse grid Galerkin product with nprod %d\n", PTAP_nnzb_prod);
        // k_compute_PTAP_product6<T><<<PTAP_nnzb_prod, 64>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("compute matmat prod to Ac\n");

        // k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("[AMG L%d] PTAP launch: nnodes=%d aggs=%d PTAP_nnzb=%d PTAP_prod=%d\n", level,
        //        nnodes, num_aggregates, PTAP_nnzb, PTAP_nnzb_prod);
        // fflush(stdout);

        // cudaError_t old_err = cudaGetLastError();  // clears stale error
        // printf("[AMG L%d] cleared pre-launch err = %s\n", level, cudaGetErrorString(old_err));
        // fflush(stdout);

        // k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);

        k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
            PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
            d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);

        // auto err = cudaPeekAtLastError();
        // printf("[AMG L%d] PTAP post-launch err = %s\n", level, cudaGetErrorString(err));
        // fflush(stdout);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("[AMG L%d] PTAP sync done\n", level);
        // fflush(stdout);

        // printf("compute matmat prod to Ac_free\n");

        // temp just
        k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
            PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
            d_PTAP_P2_blocks, d_prolong_vals, d_kmat_free_vals, d_PTAP_free_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
        // CHECK_CUDA(cudaMemcpy(d_PTAP_free_vals, d_PTAP_vals, PTAP_nnzb * block_dim2 * sizeof(T),
        //                       cudaMemcpyDeviceToDevice));
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tdone with coarse grid free Galerkin product\n");

        // printf("\n\n");
        // printf("CHECK fine grid kmat values\n");
        // T *h_Avals = DeviceVec<T>(kmat_nnzb * 36, d_kmat_vals).createHostVec().getPtr();
        // // printf("tentative prolongator: ");
        // // printVec<T>(36 * P_nnzb, h_prolong_vals);
        // for (int inode = 0; inode < nnodes; inode++) {
        //     for (int jp = h_kmat_rowp[inode]; jp < h_kmat_rowp[inode + 1]; jp++) {
        //         int jnode = h_kmat_cols[jp];
        //         printf("A mat on (node %d, node %d): \n", inode, jnode);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_Avals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }

        // printf("\n\n");
        // printf("CHECK Coarse grid Galerkin values\n");
        // T *h_PTAP_vals = DeviceVec<T>(PTAP_nnzb * 36, d_PTAP_vals).createHostVec().getPtr();
        // // printf("tentative prolongator: ");
        // // printVec<T>(36 * P_nnzb, h_prolong_vals);
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     for (int jp = h_PTAP_rowp[iagg]; jp < h_PTAP_rowp[iagg + 1]; jp++) {
        //         int jagg = h_PTAP_cols[jp];
        //         printf("PTAP mat on (agg %d, agg %d): \n", iagg, jagg);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_PTAP_vals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }

        // get the rows for coarse kmat
        h_PTAP_rows = new int[PTAP_nnzb];
        memset(h_PTAP_rows, 0, PTAP_nnzb * sizeof(int));
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_PTAP_rowp[i]; jp < h_PTAP_rowp[i + 1]; jp++) {
                int j = h_PTAP_cols[jp];
                h_PTAP_rows[jp] = i;
            }
        }
        d_PTAP_rows = HostVec<int>(PTAP_nnzb, h_PTAP_rows).createDeviceVec().getPtr();

        // now make a coarse grid galerkin matrix
        coarse_kmat_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, d_PTAP_rowp,
                                       d_PTAP_cols, nullptr, nullptr, false);
        coarse_kmat_bsr_data.rows = d_PTAP_rows;
        coarse_kmat = BsrMat<DeviceVec<T>>(coarse_kmat_bsr_data, d_PTAP_vec);
        coarse_free_kmat = BsrMat<DeviceVec<T>>(coarse_kmat_bsr_data, d_PTAP_free_vec);
    }

    // References to CUDA library handles.
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrKmat = 0;

    // for kmat
    BsrMat<DeviceVec<T>> kmat, kmat_free, coarse_kmat, coarse_free_kmat;
    BsrData coarse_kmat_bsr_data;
    DeviceVec<T> rigid_body_modes;
    T *d_kmat_vals, *d_kmat_free_vals;
    int *d_kmat_rowp, *d_kmat_rows, *d_kmat_cols;
    int *h_kmat_rowp, *h_kmat_cols, kmat_nnzb;
    int *h_kmat_diagp, *d_kmat_diagp;
    int nsmooth;
    int *d_PTAP_rows, *h_PTAP_rows;

    // settings for Smooth aggregation AMG
    int N, block_dim, nnodes;
    int block_dim2;
    int coarse_node_threshold;
    int level;

    // strength matrix (CSR pattern)
    T sparse_threshold;
    T *d_diag_norms;
    int strength_nnz;
    bool *d_strength_indicator, *h_strength_indicator;
    int *h_strength_rowp, *h_strength_cols;

    // aggregation pattern and assignments
    int *h_aggregate_ind, P_nnzb;
    int *h_tentative_rowp, *h_tentative_cols;
    int *h_prolong_rowp, *h_prolong_rows, *h_prolong_cols;
    int *d_prolong_rowp, *d_prolong_rows, *d_prolong_cols;
    int *h_tentative_block_map, *d_tentative_block_map;
    int num_aggregates;
    int *d_aggregate_ind;
    DeviceVec<T> d_Bc_vec;
    T *d_aggregate_norms2, *d_prolong_vals;
    T *d_mode_inner_products;
    int *d_P_prodBlocks, *d_K_prodBlocks, *d_Z_prodBlocks;
    int nnzb_prod;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    BsrData d_diag_bsr_data;
    DeviceVec<T> d_diag_vec;
    T *d_diag_LU_vals;
    DeviceVec<T> d_dinv_vec;
    T *d_dinv_vals;
    bool debug;
    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    // more objects for ilu0 factorization
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M =
        CUSPARSE_SOLVE_POLICY_USE_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;

    // for spectral radius computation
    DeviceVec<T> d_temp_vec, d_rhs_vec, d_inner_soln_vec;
    T *d_temp, *d_temp2, *d_resid;
    T *d_rhs, *d_inner_soln;
    T *d_z;
    /* CG-Lanczos data */
    bool CG_LANCZOS;
    DeviceVec<T> d_lanczos_loads_vec;
    int N_LANCZOS = 10;
    T spectral_radius = 1.0;
    T *alpha_vals, *beta_vals;  // cg coefficients
    T *delta_vals, *eta_vals;   // lanczos coefficients

    // coarse grid galerkin product
    int *h_prolong_tr_row_cts, *h_prolong_tr_rowp, *h_prolong_tr_cols;
    int PTAP_nnzb;
    int *h_PTAP_rowp, *h_PTAP_cols;
    int *d_PTAP_rowp, *d_PTAP_cols;
    int PTAP_nnzb_prod;
    int *h_PTAP_Kc_blocks, *h_PTAP_P1_blocks, *h_PTAP_K_blocks, *h_PTAP_P2_blocks;
    int *d_PTAP_Kc_blocks, *d_PTAP_P1_blocks, *d_PTAP_K_blocks, *d_PTAP_P2_blocks;
    T *d_PTAP_vals, *d_PTAP_free_vals;
    DeviceVec<T> d_PTAP_vec, d_PTAP_free_vec;

    // smoothed prolongation and projectors
    T *d_Z_vals;
    T omegaJac;  // for smoothed prolongation
    DeviceVec<T> d_SU_vals, d_UTU_vals, d_UTUinv_vals;
    bool *d_free_dof;

    // coarse transfer
    bool _done_post_apply_bcs;
    int Nc;
    DeviceVec<T> d_coarse_rhs_vec, d_coarse_soln_vec;
    T *d_coarse_rhs, *d_coarse_soln;

    // dirichlet bcs (really for coarse grid only)
    DeviceVec<int> d_bcs;
};