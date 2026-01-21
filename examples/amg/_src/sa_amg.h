#pragma once

#include <cstdlib>      // For rand()
#include <set>
#include <iterator>     // For std::advance
#include <cstring>      // For memset
#include <vector>

// basic utils
#include "multigrid/solvers/solve_utils.h"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "lapacke.h"

// include from GMG multigrid sections
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/prolongation/_unstructured.cuh" // for transpose mat-vec product

// local sa amg imports
#include "fake_assembler.h"
#include "sa_amg.cuh"
#include "_rigid_modes.cuh"

template <typename T, class Smoother>
class SmoothAggregationAMG : public BaseSolver {
    /* based on python code in _py_demo/_src/bsr_aggregation.py */
public:

    using Assembler = FakeAssembler<T>;
    using CoarseMG = SmoothAggregationAMG<T, Smoother>;
    using CoarseDirect = CusparseMGDirectLU<T, Assembler>;

    SmoothAggregationAMG(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, Smoother *smoother_,
        int nnodes_, BsrMat<DeviceVec<T>> kmat_, DeviceVec<T> rigid_body_modes_, 
        int coarse_dof_threshold_ = 6000, T sparse_threshold_ = 0.25, T omegaJac_ = 0.5) : 
        cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_), 
        smoother(smoother_), kmat(kmat_), nnodes(nnodes_), rigid_body_modes(rigid_body_modes_), 
        coarse_dof_threshold(coarse_dof_threshold_), sparse_threshold(sparse_threshold_) {
        
        // get data out of kmat
        auto d_kmat_bsr_data = kmat.getBsrData();
        d_kmat_vals = kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_rows = d_kmat_bsr_data.rows;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        block_dim = d_kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;
        N = nnodes * block_dim;
        omegaJac = omegaJac_;

        // setup phase (first version)
        initCuda();
        form_node_aggregates();
        compute_prolongation_nz_pattern();
        compute_prolongator_values();
        compute_coarse_grid_nz_pattern();
        compute_coarse_grid_values();

        is_coarse_mg = num_aggregates > coarse_dof_threshold;
    }

    void update_after_assembly(DeviceVec<T> &vars) { 
    // TODO 
    }
    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one

    void build_coarse_system(Assembler coarse_assembler, Smoother *coarse_smoother) {
        // need to build the coarse smoother from coarse_kmat and then pass that in here..

        // pointer for either solver and store bool of which one we use
        if (!is_coarse_mg) {
            // then instead build coarse direct solver
            coarse_direct = new CoarseDirect(cublasHandle, cusparseHandle, coarse_assembler, coarse_kmat);
        } else {
            // then build coarse AMG solver and new coarse smoother
            coarse_mg = new CoarseMG(cublasHandle, cusparseHandle, coarse_smoother, 
                num_aggregates, coarse_kmat, d_Bc_vec, coarse_dof_threshold, sparse_threshold);
        }
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // solve this multigrid level (V-cycle)

        // setup rhs and soln with init guess of 0
        cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_inner_soln, 0.0, N * sizeof(T));  // re-zero the solution

        // pre-smooth defect
        this->smoother->solve(d_rhs_vec, d_inner_soln_vec);

        // restrict
        d_coarse_rhs_vec.zeroValues();  // zero before add new result
        int nprods = PTAP_nnzb_prod * block_dim2;
        dim3 block0(32), grid0((nprods + 31) / 32);
        k_bsrmv_transpose<T><<<grid0, block0>>>(PTAP_nnzb_prod, block_dim, d_prolong_rowp, 
            d_prolong_cols, d_prolong_vals,
            d_rhs_vec.getPtr(), d_coarse_rhs_vec.getPtr());
        
        // coarse solve
        if (!is_coarse_mg) { // direct solve
            this->coarse_direct->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        } else {
            this->coarse_mg->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        }

        // prolongation
        T a = 1.0, b = 0.0;
        int mb = nnodes, nb = num_aggregates;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, P_nnzb, &a, descrKmat,
                                      d_prolong_vals, d_prolong_rowp, d_prolong_cols, block_dim,
                                      d_coarse_soln_vec.getPtr(), &b, d_temp));
        // add to previous inner soln (see bsr_aggregation.py)
        a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_inner_soln, 1));
        
        // post-smooth
        this->smoother->solve(d_rhs_vec, d_inner_soln_vec);

        // copy internal soln to external solution of the solve method
        cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);

        return false;
    }

    BsrData get_coarse_bsr_data() { return coarse_kmat_bsr_data; }
    int get_num_aggregates() { return num_aggregates; }
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
        h_kmat_cols  = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();
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

        // for linear solver / precond use
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();

        Nc = num_aggregates * block_dim;
        d_coarse_rhs_vec = DeviceVec<T>(Nc);
        d_coarse_rhs = d_coarse_rhs_vec.getPtr();
        d_coarse_soln_vec = DeviceVec<T>(Nc);
        d_coarse_soln = d_coarse_soln_vec.getPtr();
    }

    void form_node_aggregates() {
        // 1) compute strength pattern on GPU
        k_get_diag_norms<T><<<nnodes, 32>>>(nnodes, d_kmat_diagp, block_dim, d_kmat_vals, d_diag_norms);
        k_compute_strength_bools<T><<<kmat_nnzb, 32>>>(kmat_nnzb, block_dim, d_diag_norms, d_kmat_rows, 
            d_kmat_cols, d_kmat_vals, sparse_threshold, d_strength_indicator);
        
        // 2) compute strength indices to host
        h_strength_indicator = DeviceVec<bool>(kmat_nnzb, d_strength_indicator).createHostVec().getPtr();
        int strength_nnz = 0;
        for (int iblock = 0; iblock < kmat_nnzb; iblock++) {
            if (h_strength_indicator[iblock]) strength_nnz++;
        }
        h_strength_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_strength_cols = HostVec<int>(strength_nnz).getPtr();
        for (int i = 0; i < nnodes; i++) {
            h_strength_rowp[i+1] = h_strength_rowp[i]; // update from sum of last one..
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i+1]; jp++) {
                int j = h_kmat_cols[jp];
                if (h_strength_indicator[jp]) h_strength_rowp[i+1]++;
            }
        }

        // 3) do greedy serial aggregation pattern on host
        _greedy_serial_aggregation();
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

        // Second phase: assign all remaining nodes to a nearby aggregate
        for (int i = 0; i < nnodes; i++) {
            if (h_aggregate_ind[i] != -1)
                continue; // Node already assigned, skip it

            // Collect aggregates from strong neighbors
            std::set<int> nearby_aggregates;
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                int j = h_strength_cols[jp];
                if (h_aggregate_ind[j] != -1) {
                    // Insert the aggregate index of neighbor j
                    nearby_aggregates.insert(h_aggregate_ind[j]);
                }
            }

            if (!nearby_aggregates.empty()) {
                // Randomly choose one aggregate from the set
                int set_size = nearby_aggregates.size();
                int random_index = rand() % set_size;  // using rand() for simplicity; consider C++11 random generators for production code
                auto it = nearby_aggregates.begin();
                std::advance(it, random_index);
                int chosen_aggregate = *it;
                h_aggregate_ind[i] = chosen_aggregate;
            } else {
                // No nearby aggregate; assign as a new aggregate.
                h_aggregate_ind[i] = num_aggregates++;
            }
        }

        d_aggregate_ind = HostVec<int>(nnodes, d_aggregate_ind).createDeviceVec().getPtr();
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
            h_tentative_rowp[i+1] = i + 1;
        }

        //--------------------------------------------------------------------------
        // Step 2: Compute the smoothed A*P prolongation pattern.
        // We will use a temporary vector to build the final CSR arrays.
        // For each row i in the prolongator, the pattern is given by:
        //     P(i) = tentative(i) ∪ (⋃ for j in row i of kmat) tentative(j)
        // We use an std::set<int> to guarantee uniqueness.
        //--------------------------------------------------------------------------
        std::vector<int> prolong_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> prolong_cols;                   // column indices for P

        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            
            // Add the tentative prolongation pattern of row i (usually the "diagonal" entry).
            for (int kp = h_tentative_rowp[i]; kp < h_tentative_rowp[i+1]; kp++) {
                uniqueIndices.insert(h_tentative_cols[kp]);
            }
            
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i+1]; jp++) {
                int j = h_kmat_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that is j).
                for (int kp = h_tentative_rowp[j]; kp < h_tentative_rowp[j+1]; kp++) {
                    uniqueIndices.insert(h_tentative_cols[kp]);
                }
            }
            
            // The number of entries for row i is the size of the set.
            prolong_rowp[i+1] = prolong_rowp[i] + uniqueIndices.size();
            
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
        h_prolong_cols  = HostVec<int>(P_nnzb).getPtr();
        
        memcpy(h_prolong_rowp, prolong_rowp.data(), (nnodes + 1) * sizeof(int));
        memcpy(h_prolong_cols, prolong_cols.data(), P_nnzb * sizeof(int));
        
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i+1]; jp++) {
                int j = h_prolong_cols[jp];
                h_prolong_rows[jp] = i;
            }
        }

        d_prolong_rowp = HostVec<int>(nnodes + 1, h_prolong_rowp).createDeviceVec().getPtr();
        d_prolong_rows = HostVec<int>(P_nnzb, h_prolong_rows).createDeviceVec().getPtr();
        d_prolong_cols = HostVec<int>(P_nnzb, h_prolong_cols).createDeviceVec().getPtr();
        d_prolong_vals = DeviceVec<T>(P_nnzb * block_dim2).getPtr();
        d_Z_vals = DeviceVec<T>(P_nnzb * block_dim2).getPtr();

        // 4) compute the block locations of each part of tentative prolongator
        h_tentative_block_map = HostVec<int>(nnodes).getPtr();
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i+1]; jp++) {
                int j = h_prolong_cols[jp];
                if (j == h_aggregate_ind[i]) {
                    h_tentative_block_map[i] = jp;
                }
            }
        }
        d_tentative_block_map = HostVec<int>(nnodes, h_tentative_block_map).createDeviceVec().getPtr();
    }

    template <bool startup = true>
    void _compute_diag_vals() {
        // first need to construct rowp and cols for diagonal (fairly easy)

        // startup section
        int ndiag_vals = block_dim * block_dim * nnodes;
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
        }  // end of startup

        // regular jacobi preconditioner
        //  zero previous values (to get new Dinv, in case optimization or nonlinear problem)
        d_diag_vec.zeroValues();  // this is vector for the opinter d_diag_LU_vals (confusing, can
                                   // fix later
        k_copyBlockDiagFromBsrMat<T><<<(ndiag_vals + 31) / 32, 32>>>(
            nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);

        // ilu0 factoriation
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

        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                                         info_M, policy_M, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
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
                                          diag_inv_nnzb, &a, descrKmat, d_dinv_vals,
                                          d_diag_rowp, d_diag_cols, block_dim, d_resid, &b, d_z));
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
                                      diag_inv_nnzb, &a, descrKmat, d_dinv_vals,
                                      d_diag_rowp, d_diag_cols, block_dim, d_resid, &b, d_z));
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
        _graham_schmidt_QR_factorization();

        // 2) compute spectral radius of fine grid matrix
        _compute_diag_vals();
        // _compute_spectral_radius(); // don't do this for now..
 
        // 2) compute smoothed prolongator
        // compute -omega/rho(Dinv*A) * beta_k * A*P into Z first (scaled prolong defect matrix)
        _smooth_prolongator();
    }

    void _graham_schmidt_QR_factorization() {
        // compute tentative prolongator using QR factorization designed explicitly for N x 6 matrix
        // loop over each of 6 columns doing Graham-Schmidt decomp to find P * B = Bc
        // where B is known and the rigid body modes of fine grid

        d_Bc_vec = DeviceVec<T>(block_dim2 * num_aggregates);
        d_aggregate_norms2 = DeviceVec<T>(block_dim * num_aggregates).getPtr();

        // perform block-norm of each 
        for (int imode = 0; imode < block_dim; imode++) {
            // compute orthogonalization against previous modes
            for (int jmode = 0; jmode < imode; jmode++) {
                // compute inner products with previous modes and store in d_Bc_vec (imode, jmode) row and col of that aggregate
                k_compute_GS_inner_product<T><<<(N + 31) / 32, 32>>>(imode, jmode, nnodes, block_dim, 
                    d_aggregate_ind, d_tentative_block_map, d_prolong_vals, d_Bc_vec.getPtr());
            
                // subtract inner product multiple of previous mode in Graham-Schmidt (to orthogonalize against previous modes)
                k_remove_GS_projector_mode<T><<<(N + 31) / 32, 32>>>(imode, jmode, nnodes, block_dim, 
                    d_aggregate_ind, d_tentative_block_map, d_Bc_vec.getPtr(), d_prolong_vals);
            }
            
            // compute norms in each aggregate and then normalize P
            k_compute_aggregate_norms2<T><<<(N + 31) / 32, 32>>>(imode, nnodes, block_dim, 
                d_aggregate_ind, rigid_body_modes.getPtr(), d_aggregate_norms2);
            k_compute_sqrt_norms<T><<<(num_aggregates + 31) / 32, 32>>>(imode,num_aggregates, 
                block_dim, d_aggregate_norms2, d_Bc_vec.getPtr()); // and store norm2 in Bc norm
            k_normalize_tentative_modes<T><<<(N + 31) / 32, 32>>>(imode, nnodes, block_dim, d_aggregate_ind,
                d_tentative_block_map, rigid_body_modes.getPtr(), d_Bc_vec.getPtr(), d_prolong_vals);
        }
    }

    void _smooth_prolongator() {
        // Z_mat->zeroValues();
        cudaMemset(d_Z_vals, 0.0, P_nnzb * block_dim2 * sizeof(T));
        dim3 PKP_block(216), PKP_grid(nnzb_prod);
        T a = -omegaJac;
        // T a = -omega / spectral_radius; // if just jacobi
        k_compute_mat_mat_prod<T><<<PKP_grid, PKP_block>>>(
            nnzb_prod, block_dim, a, d_K_prodBlocks, d_P_prodBlocks, d_Z_prodBlocks,
            d_kmat_vals, d_prolong_vals, d_Z_vals);

        // compute Dinv*Z into Z in-place (equiv to Dinv*scale*A*P => Z)
        dim3 DP_block(216), DP_grid(P_nnzb);
        k_compute_Dinv_P_mmprod<T><<<DP_grid, DP_block>>>(
            P_nnzb, block_dim, d_dinv_vals, d_prolong_rows, d_Z_vals);

        // compute free variables
        auto free_var_vec = DeviceVec<bool>(N);
        free_var_vec.setFullVecToConstValue(true); // set all to default true meaning free var
        d_free_dof = free_var_vec.getPtr();

        // do orthogonal projector on Z (only really needed for coarse-grid galerkin AMG,
        // not smooth GMG) 
        // if constexpr (do_orthog_projector) {
        dim3 OP_block(32), OP_grid(nnodes);
        // d_SU_vals.zeroValues();
        int ndiag_vals = block_dim * block_dim * nnodes;
        d_SU_vals = DeviceVec<T>(ndiag_vals);
        // compute SU matrix
        k_orthog_projector_computeSU<T><<<OP_grid, OP_block>>>(nnodes,
            block_dim, d_Bc_vec.getPtr(), d_free_dof, d_prolong_rowp, d_prolong_cols, d_prolong_vals, d_SU_vals.getPtr());

        d_UTU_vals = DeviceVec<T>(ndiag_vals);
        k_orthog_projector_computeUTU<T><<<OP_grid, OP_block>>>(nnodes, block_dim, d_Bc_vec.getPtr(), 
            d_free_dof, d_prolong_rowp, d_prolong_cols, d_UTU_vals.getPtr());
        
        // now compute the LU factor and inverse matrix UTUinv for each fine node (same size and like Dinv matrix)
        // reuse same pointers and nnzb sizes as Dinv cause same dimensions
        CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                            &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                            d_UTU_vals.getPtr(), d_diag_rowp, d_diag_cols, trans_L,
                                            trans_U, policy_L, policy_U, dir);
        // now compute UTUinv linear operator like I did for the Dinv
        d_UTUinv_vals = DeviceVec<T>(ndiag_vals); // inv linear operator of UTU
        for (int i = 0; i < block_dim; i++) {
            // set d_temp to ei (one of e1 through e6 per block)
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            dim3 block(32);
            dim3 grid((nnodes + 31) / 32);
            k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

            // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_UTU_vals.getPtr(),
                d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
                pBuffer));  // prob only need U^-1 part for block diag.. TBD

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_UTU_vals.getPtr(),
                d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

            // now copy temp2 into columns of new operator
            dim3 grid2((N + 31) / 32);
            k_setLUinv_operator<T>
                <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_UTUinv_vals.getPtr());
        }

        // remove rigid-body row-sums
        k_orthog_projector_removeRowSums<T><<<OP_grid, OP_block>>>(nnodes,
            block_dim, d_Bc_vec.getPtr(), d_free_dof, d_prolong_rowp, d_prolong_cols, d_SU_vals.getPtr(),
            d_UTUinv_vals.getPtr(), d_prolong_vals);
        // }

        // add Z into P (the prolongation update)
        dim3 add_block(64);
        T scale = 1.0;
        k_add_colored_submat_PFP<T>
            <<<DP_grid, add_block>>>(P_nnzb, block_dim, scale, 0, d_Z_vals, d_prolong_vals);
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
        std::vector<int> prolong_tr_cols;                   // column indices for P

        h_prolong_tr_row_cts = HostVec<int>(nnodes).getPtr();
        h_prolong_tr_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_prolong_tr_cols = HostVec<int>(P_nnzb).getPtr();

        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i+1]; jp++) {
                int j = h_prolong_cols[jp];
                h_prolong_tr_row_cts[j]++;
            }
        }

        for (int i = 0; i < nnodes; i++) {
            h_prolong_tr_rowp[i+1] = h_prolong_tr_rowp[i] + h_prolong_tr_row_cts[i];
        }

        // reset to zero
        memset(h_prolong_tr_row_cts, 0, nnodes * sizeof(int));
        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i+1]; jp++) {
                int j = h_prolong_cols[jp];
                int ip = h_prolong_tr_rowp[j] + h_prolong_tr_row_cts[j];
                h_prolong_tr_cols[ip] = i;
                h_prolong_tr_row_cts[j]++;
            }
        }

        // 2) compute A*P nonzero pattern (extra fillin from A*T => P needed for coarse grid Galerkin)
        std::vector<int> AP_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> AP_cols;                   // column indices for P
        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // Add the tentative prolongation pattern of row i (usually the "diagonal" entry).
            for (int kp = h_prolong_rowp[i]; kp < h_prolong_rowp[i+1]; kp++) {
                uniqueIndices.insert(h_tentative_cols[kp]);
            }
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i+1]; jp++) {
                int j = h_kmat_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that is j).
                for (int kp = h_prolong_rowp[j]; kp < h_prolong_rowp[j+1]; kp++) {
                    uniqueIndices.insert(h_prolong_cols[kp]);
                }
            }
            
            // The number of entries for row i is the size of the set.
            AP_rowp[i+1] = AP_rowp[i] + uniqueIndices.size();
            
            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                AP_cols.push_back(col);
            }
        }

        // 3) compute P^T * (AP) nz pattern now
        int num_coarse = num_aggregates;
        std::vector<int> PTAP_rowp(num_coarse + 1, 0);
        std::vector<int> PTAP_cols;
        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i+1]; jp++) {
                int j = h_prolong_tr_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity, that is j).
                for (int kp = AP_rowp[j]; kp < AP_rowp[j+1]; kp++) {
                    uniqueIndices.insert(AP_cols[kp]);
                }
            }
            
            // The number of entries for row i is the size of the set.
            PTAP_rowp[i+1] = PTAP_rowp[i] + uniqueIndices.size();
            
            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                PTAP_cols.push_back(col);
            }
        }

        PTAP_nnzb = PTAP_rowp[num_coarse];
        h_PTAP_rowp = HostVec<int>(num_coarse + 1).getPtr();
        h_PTAP_cols = HostVec<int>(PTAP_nnzb).getPtr();
        memcpy(h_PTAP_rowp, PTAP_rowp.data(), (num_coarse + 1) * sizeof(int));
        memcpy(h_PTAP_cols, PTAP_cols.data(), PTAP_nnzb * sizeof(int));
        d_PTAP_rowp = HostVec<int>(num_coarse + 1, h_PTAP_rowp).createDeviceVec().getPtr();
        d_PTAP_cols = HostVec<int>(PTAP_nnzb, h_PTAP_cols).createDeviceVec().getPtr();
        // assign Kc or PTAP coarse grid matrix values
        d_PTAP_vec = DeviceVec<T>(PTAP_nnzb);
        d_PTAP_vals = d_PTAP_vec.getPtr();

        // 3) compute nonzero product block pattern..
        PTAP_nnzb_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i+1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                // find block in non-transposed pattern
                int _ip = -1;
                for (int ip = h_prolong_rowp[j]; ip < h_prolong_rowp[j+1]; ip++) {
                    int i2 = h_prolong_cols[ip];
                    if (i2 == i) {
                        _ip = ip; break;
                    }
                }

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j+1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k+1]; lp++) {
                        int l = h_prolong_cols[lp];
                        PTAP_nnzb_prod++;
                    }
                }
            }
        }
        h_PTAP_Kc_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P1_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P2_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();

        PTAP_nnzb_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i+1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                // find block in non-transposed pattern
                int _ip = -1;
                for (int ip = h_prolong_rowp[j]; ip < h_prolong_rowp[j+1]; ip++) {
                    int i2 = h_prolong_cols[ip];
                    if (i2 == i) {
                        _ip = ip; break;
                    }
                }

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j+1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k+1]; lp++) {
                        int l = h_prolong_cols[lp];

                        // find block entry in PTAP matrix
                        int _mp = -1;
                        for (int mp = h_PTAP_rowp[i]; mp < h_PTAP_rowp[i+1]; mp++) {
                            int m = h_PTAP_cols[mp];
                            if (m == l) {
                                _mp = mp;
                            }
                        }
                        h_PTAP_Kc_blocks[PTAP_nnzb_prod] = _mp; // output Kc
                        h_PTAP_P1_blocks[PTAP_nnzb_prod] = jp; // transpose P
                        h_PTAP_K_blocks[PTAP_nnzb_prod] = kp; // K
                        h_PTAP_P2_blocks[PTAP_nnzb_prod] = lp; // P on right
                        PTAP_nnzb_prod++;
                    }
                }
            }
        }

        // put prod blocks on GPU
        d_PTAP_Kc_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_Kc_blocks).createDeviceVec().getPtr();
        d_PTAP_P1_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_P1_blocks).createDeviceVec().getPtr();
        d_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_K_blocks).createDeviceVec().getPtr();
        d_PTAP_P2_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_P2_blocks).createDeviceVec().getPtr();
    }

    void compute_coarse_grid_values() {
        // 1) compute coarse grid Galerkin product Ac = P^T * A * P
        k_compute_PTAP_product6<T><<<PTAP_nnzb_prod, 64>>>(PTAP_nnzb_prod, block_dim,
            d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks, d_PTAP_P2_blocks, 
            d_prolong_vals, d_kmat_vals, d_PTAP_vals);

        // now make a coarse grid galerkin matrix
        coarse_kmat_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, 
            d_PTAP_rowp, d_PTAP_cols, nullptr, nullptr, false);
        coarse_kmat = BsrMat<DeviceVec<T>>(coarse_kmat_bsr_data, d_PTAP_vec);
    }

    // References to CUDA library handles.
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrKmat = 0;

    // for kmat
    BsrMat<DeviceVec<T>> kmat, coarse_kmat;
    BsrData coarse_kmat_bsr_data;
    DeviceVec<T> rigid_body_modes;
    T *d_kmat_vals;
    int *d_kmat_rowp, *d_kmat_rows, *d_kmat_cols;
    int *h_kmat_rowp, *h_kmat_cols, kmat_nnzb;
    int *h_kmat_diagp, *d_kmat_diagp;

    // settings for Smooth aggregation AMG
    int N, block_dim, nnodes;
    int block_dim2;
    int coarse_dof_threshold;

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
    T *d_PTAP_vals;
    DeviceVec<T> d_PTAP_vec;

    // smoothed prolongation and projectors
    T *d_Z_vals;
    T omegaJac; // for smoothed prolongation
    DeviceVec<T> d_SU_vals, d_UTU_vals, d_UTUinv_vals;
    bool *d_free_dof;

    // coarse transfer
    int Nc;
    DeviceVec<T> d_coarse_rhs_vec, d_coarse_soln_vec;
    T *d_coarse_rhs, *d_coarse_soln;
};