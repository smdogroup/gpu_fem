#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include "linalg/vec.h"
#include "multigrid/solvers/solve_utils.h"
#include "spai.cuh"

template <typename T>
class SPAI : public BaseSolver {
   public:
    // Constructor: fill specifies how many fill iterations to perform.
    SPAI(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
         BsrMat<DeviceVec<T>> kmat_, int fill = 3, int optim_ = 2)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_), kmat(kmat_) {
        auto _bsr_data = kmat_.getBsrData();
        block_dim = _bsr_data.block_dim;
        N = _bsr_data.mb * block_dim;
        nnodes = N / block_dim;
        temp = DeviceVec<T>(N);

        // get data out of kmat
        auto d_kmat_bsr_data = kmat.getBsrData();
        d_kmat_vals = kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        optim = optim_;

        // kmat_nnzb, d_kmat_rowp, d_kmat_cols, etc. are assumed to be set up already.
        printf("1: compute power fillin SPAI(%d)\n", fill);
        _compute_power_fillin(fill);  // one more in order to
        printf("2: compute initial preconditioner");
        _compute_initial_preconditioner();
        printf("3: compute temporary matrices\n");
        _create_temp_matrices();
    }

    void factor() {
        // then do the SPAI optimization with the new matrix values
        printf("perform SPAI optimization\n");
        _spai_optimization(optim);
    }

    void update_after_assembly(DeviceVec<T> &vars) { factor(); }
    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one
    void set_cycle_type(std::string cycle_) {}

    T precond_complexity() {
        // get [nnzb(precond) + nnzb(A)] / nnzb(A)
        int precond_nnzb = M_nnzb;
        return (precond_nnzb + kmat_nnzb) * 1.0 / kmat_nnzb;
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // compute M*rhs => temp
        T a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, M_nnzb, &a,
                                      descr_M, d_M_vals, d_M_rowp, d_M_cols, block_dim,
                                      rhs.getPtr(), &b, soln.getPtr()));

        // // compute K*temp => soln
        // CHECK_CUSPARSE(cusparseDbsrmv(
        //             cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //             nnodes, nnodes, kmat_nnzb, &a, descr_M, d_kmat_vals,
        //             d_kmat_rowp, d_kmat_cols, block_dim, temp.getPtr(), &b, soln.getPtr()));

        return false;
    }

   private:
    void _compute_power_fillin(int fill) {
        // KEEP THESE EXACTLY AS IS:
        // Get host versions of kmat sparsity
        h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // initial M sparsity is nofill (same as K's pattern)
        int *h_M_rowp0 = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        int *h_M_cols0 = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // ----------------------------------------------------------
        // Allocate host arrays for the current M sparsity pattern.
        // We start with M = no-fill pattern (same as K)
        // ----------------------------------------------------------
        M_rowp = HostVec<int>(nnodes + 1).getPtr();
        int cur_nnz = kmat_nnzb;
        M_cols = HostVec<int>(cur_nnz).getPtr();

        // Copy the no-fill pattern into our working pattern arrays.
        for (int i = 0; i < nnodes + 1; i++) {
            M_rowp[i] = h_M_rowp0[i];
        }
        for (int i = 0; i < cur_nnz; i++) {
            M_cols[i] = h_M_cols0[i];
        }
        printf("initial M with nnzb=%d\n", kmat_nnzb);
        // printf("\trowp: ");
        // printVec<int>(nnodes+1, M_rowp);
        // printf("\tcols: ");
        // printVec<int>(cur_nnz, M_cols);

        int *new_M_rowp, *new_M_cols;

        // ----------------------------------------------------------
        // Iteratively update the sparsity pattern to be that of (M*K*M)
        // ----------------------------------------------------------
        for (int iter = 0; iter < fill; iter++) {
            // We will compute the new pattern and store as new_M_rowp/new_M_cols.
            new_M_rowp = HostVec<int>(nnodes + 1).getPtr();
            new_M_rowp[0] = 0;

            // For each row, we collect the union of column indices.
            // We'll use an auxiliary STL vector for each row to remove duplicates.
            // We first need to know how many nonzeros each row will contain.
            std::vector<std::vector<int>> union_rows(nnodes);

            for (int i = 0; i < nnodes; i++) {
                std::vector<int> union_cols;  // temporary container for unique column indices

                // Loop over nonzeros in row i of current M.
                for (int idx = M_rowp[i]; idx < M_rowp[i + 1]; idx++) {
                    int r = M_cols[idx];

                    // Multiply with K: loop over row r of K.
                    for (int kp = h_kmat_rowp[r]; kp < h_kmat_rowp[r + 1]; kp++) {
                        int colK = h_kmat_cols[kp];

                        // Multiply with M: chain with M's row corresponding to colK.
                        for (int mp = M_rowp[colK]; mp < M_rowp[colK + 1]; mp++) {
                            int colM = M_cols[mp];
                            union_cols.push_back(colM);
                        }
                    }
                }

                // Sort and remove duplicates from union_cols.
                if (!union_cols.empty()) {
                    std::sort(union_cols.begin(), union_cols.end());
                    auto it = std::unique(union_cols.begin(), union_cols.end());
                    union_cols.erase(it, union_cols.end());
                }
                union_rows[i] = union_cols;

                // Update new_M_rowp for row i+1 (will add later).
                new_M_rowp[i + 1] = new_M_rowp[i] + static_cast<int>(union_cols.size());
            }

            // Allocate new_M_cols with the final total nnz.
            int new_nnz = new_M_rowp[nnodes];
            new_M_cols = HostVec<int>(new_nnz).getPtr();

            // Fill in new_M_cols using the union_rows data.
            for (int i = 0; i < nnodes; i++) {
                int start = new_M_rowp[i];
                const std::vector<int> &union_cols = union_rows[i];
                for (size_t j = 0; j < union_cols.size(); j++) {
                    new_M_cols[start + j] = union_cols[j];
                }
            }

            // Delete the old sparsity pattern.
            delete[] M_rowp;
            delete[] M_cols;
            // Update M to the new pattern.
            M_rowp = new_M_rowp;
            M_cols = new_M_cols;
            cur_nnz = new_nnz;

            printf("M fillin step %d with nnzb=%d\n", iter, cur_nnz);
            // printf("\trowp: ");
            // printVec<int>(nnodes+1, M_rowp);
            // printf("\tcols: ");
            // printVec<int>(cur_nnz, M_cols);
        }  // end for fill iterations

        // At this point, M_rowp and M_cols hold the final sparsity pattern.
        M_nnzb = cur_nnz;  // store the nonzero count for M

        // compute the host kmat diagonal pointer
        int *h_M_diagp = new int[nnodes];
        memset(h_M_diagp, 0, nnodes * sizeof(int));
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int jp = M_rowp[block_row]; jp < M_rowp[block_row + 1]; jp++) {
                int block_col = M_cols[jp];
                // printf("row %d, col %d\n", block_row, block_col);
                if (block_row == block_col) {
                    h_M_diagp[block_row] = jp;
                }
            }
        }
        // printf("h_M_diagp: ");
        // printVec<int>(nnodes, h_M_diagp);

        // create M matrix with filled in sparsity on device
        d_M_rowp = HostVec<int>(nnodes + 1, M_rowp).createDeviceVec().getPtr();
        d_M_cols = HostVec<int>(M_nnzb, M_cols).createDeviceVec().getPtr();
        d_M_diagp = HostVec<int>(nnodes, h_M_diagp).createDeviceVec().getPtr();
        d_M_vals = DeviceVec<T>(block_dim * block_dim * M_nnzb).getPtr();

        // #ifdef DEBUG
        //         printf("Final M sparsity: %d nonzeros\n", M_nnzb);
        // #endif

        // Free host memory allocated for the current pattern (if you no longer need them on host).
        // delete[] M_rowp;
        // delete[] M_cols;
    }

    void _compute_initial_preconditioner() {
        /* set M = s * I where s = 1/||K||_1 */
        // compute abs-value row-sums in temp vec
        cudaMemset(temp.getPtr(), 0.0, N * sizeof(T));
        int nkmat_vals = kmat_nnzb * block_dim * block_dim;
        dim3 grid((nkmat_vals + 31) / 32);
        dim3 block(32);
        k_abs_value_col_sums<T>
            <<<grid, block>>>(kmat_nnzb, block_dim, d_kmat_cols, d_kmat_vals, temp.getPtr());

        // compute max across all row-sums
        T *norm1 = DeviceVec<T>(1).getPtr();
        dim3 grid2((N + 31) / 32);
        k_vec_max<T><<<grid2, block>>>(N, temp.getPtr(), norm1);
        T *h_norm1 = DeviceVec<T>(1, norm1).createHostVec().getPtr();
        T inorm1 = 1.0 / h_norm1[0];
        printf("norm1 of matrix: %.4e, inorm scale %.4e\n", h_norm1[0], inorm1);

        // fillin value s on diagonal of entry (setting all entries) with kernel
        // printf("trying to add diag in initial preconditioner\n");
        block_dim2 = block_dim * block_dim;
        cudaMemset(d_M_vals, 0.0, nnodes * block_dim2 * sizeof(T));
        dim3 grid3((nnodes * block_dim + 31) / 32);
        k_add_diag_matrix<T><<<grid3, block>>>(nnodes, block_dim, inorm1, d_M_diagp, d_M_vals);

        // T *h_M_vals = DeviceVec<T>(block_dim2 * nno

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tadded diag in initial preconditioner\n");
    }

    void _create_temp_matrices() {
        printf("computing prod block pattern...\n");
        _compute_block_product_pattern();
        printf("\tdone computing prod block pattern\n");

        // R, Z, Q matrices with M sparsity
        // printf("d_R_vals nnzb = %d\n", M_nnzb);
        d_R_vals = DeviceVec<T>(block_dim * block_dim * M_nnzb).getPtr();
        d_Z_vals = DeviceVec<T>(block_dim * block_dim * M_nnzb).getPtr();
        d_Q_vals = DeviceVec<T>(block_dim * block_dim * M_nnzb).getPtr();

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));
        CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
    }

    void _compute_block_product_pattern() {
        // compute sparsity pattern for block products
        // K uses h_kmat_rowp, h_kmat_cols, kmat_nnzb
        // M uses M_rowp, M_cols, M_nnzb

        // took this code from grid.h which does K * P => PF sparsity
        // want list of blocks and in each case the KM_blocks1, KM_blocks2, KM_blocks3 of mat1 =
        // mat2 * mat3 which gives block indices in each matrix.. also the number of block product
        // terms
        KM_nprod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = M_rowp[i]; jp < M_rowp[i + 1]; jp++) {
                int j = M_cols[jp];  // (M)_{ij} output
                // now inner loop k for K_{ik} * M_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check M_{kj} nz
                    bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add K*P
                                          // fillin (for better prolong)
                    for (int jp2 = M_rowp[k]; jp2 < M_rowp[k + 1]; jp2++) {
                        int j2 = M_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    KM_nprod++;
                }
            }
        }
        // now allocate the block indices of the product
        int *h_PF_blocks = new int[KM_nprod];
        int *h_K_blocks = new int[KM_nprod];
        int *h_P_blocks = new int[KM_nprod];
        memset(h_PF_blocks, 0, KM_nprod * sizeof(int));
        memset(h_K_blocks, 0, KM_nprod * sizeof(int));
        memset(h_P_blocks, 0, KM_nprod * sizeof(int));
        int inz_prod = 0;
        printf("kmat_nnzb %d, M_nnzb %d\n", kmat_nnzb, M_nnzb);
        for (int i = 0; i < nnodes; i++) {
            for (int jp = M_rowp[i]; jp < M_rowp[i + 1]; jp++) {
                int j = M_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;
                    int _jp2 = -1;
                    for (int jp2 = M_rowp[k]; jp2 < M_rowp[k + 1]; jp2++) {
                        int j2 = M_cols[jp2];
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
        printf("KM nprod %d\n", KM_nprod);
        // printf("PF_prod_blocks: ");
        // printVec<int>(KM_nprod, h_PF_blocks);
        // printf("K_prod_blocks: ");
        // printVec<int>(KM_nprod, h_K_blocks);
        // printf("P_prod_blocks: ");
        // printVec<int>(KM_nprod, h_P_blocks);

        // now allocate onto the device
        d_KM_prodBlocks1 = HostVec<int>(KM_nprod, h_PF_blocks).createDeviceVec().getPtr();
        d_KM_prodBlocks2 = HostVec<int>(KM_nprod, h_K_blocks).createDeviceVec().getPtr();
        d_KM_prodBlocks3 = HostVec<int>(KM_nprod, h_P_blocks).createDeviceVec().getPtr();

        /* now the M * M => M sparsity product */
        MM_nprod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = M_rowp[i]; jp < M_rowp[i + 1]; jp++) {
                int j = M_cols[jp];  // (M)_{ij} output
                // now inner loop k for M_{ik} * M_{kj}
                for (int kp = M_rowp[i]; kp < M_rowp[i + 1]; kp++) {
                    int k = M_cols[kp];

                    // check M_{kj} nz
                    bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add K*P
                                          // fillin (for better prolong)
                    for (int jp2 = M_rowp[k]; jp2 < M_rowp[k + 1]; jp2++) {
                        int j2 = M_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    MM_nprod++;
                }
            }
        }
        // now allocate the block indices of the product
        int *h_PF_blocks2 = new int[MM_nprod];
        int *h_K_blocks2 = new int[MM_nprod];
        int *h_P_blocks2 = new int[MM_nprod];
        memset(h_PF_blocks2, 0, MM_nprod * sizeof(int));
        memset(h_K_blocks2, 0, MM_nprod * sizeof(int));
        memset(h_P_blocks2, 0, MM_nprod * sizeof(int));
        inz_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = M_rowp[i]; jp < M_rowp[i + 1]; jp++) {
                int j = M_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for M_{ik} * P_{kj}
                for (int kp = M_rowp[i]; kp < M_rowp[i + 1]; kp++) {
                    int k = M_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;
                    int _jp2 = -1;
                    for (int jp2 = M_rowp[k]; jp2 < M_rowp[k + 1]; jp2++) {
                        int j2 = M_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                            _jp2 = jp2;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    h_PF_blocks2[inz_prod] = jp;
                    h_K_blocks2[inz_prod] = kp;
                    h_P_blocks2[inz_prod] = _jp2;
                    inz_prod++;
                }
            }
        }
        // now allocate onto the device
        d_MM_prodBlocks1 = HostVec<int>(MM_nprod, h_PF_blocks2).createDeviceVec().getPtr();
        d_MM_prodBlocks2 = HostVec<int>(MM_nprod, h_K_blocks2).createDeviceVec().getPtr();
        d_MM_prodBlocks3 = HostVec<int>(MM_nprod, h_P_blocks2).createDeviceVec().getPtr();
    }

    void _spai_optimization(int fill) {
        /* do self-preconditioned minimal residual iteration for SPAI objective ||I - AM||_F^2 */
        // using mat-mat products for the iteration

        // do global MR iteration here.. (one iteration per fill level, no dropping here..)
        for (int iter = 0; iter < fill; iter++) {
            // 1) compute R = -K * M (sparse mat-mat product)
            cudaMemset(d_R_vals, 0.0, block_dim2 * M_nnzb * sizeof(T));
            k_compute_mat_mat_prod2<T>
                <<<KM_nprod, 216>>>(KM_nprod, block_dim, -1.0, d_KM_prodBlocks2, d_KM_prodBlocks3,
                                    d_KM_prodBlocks1, d_kmat_vals, d_M_vals, d_R_vals);
            // CHECK_CUDA(cudaDeviceSynchronize());

            // 2) add I to R so that R = I - K * M
            dim3 grid3((nnodes * block_dim + 31) / 32), block(32);
            T one = 1.0;
            k_add_diag_matrix<T><<<grid3, block>>>(nnodes, block_dim, one, d_M_diagp, d_R_vals);

            // 3) compute Z = M * R
            cudaMemset(d_Z_vals, 0.0, block_dim2 * M_nnzb * sizeof(T));
            k_compute_mat_mat_prod2<T>
                <<<MM_nprod, 216>>>(MM_nprod, block_dim, 1.0, d_MM_prodBlocks2, d_MM_prodBlocks3,
                                    d_MM_prodBlocks1, d_M_vals, d_R_vals, d_Z_vals);

            // 4) compute Q = K * Z
            cudaMemset(d_Q_vals, 0.0, block_dim2 * M_nnzb * sizeof(T));
            k_compute_mat_mat_prod2<T>
                <<<KM_nprod, 216>>>(KM_nprod, block_dim, 1.0, d_KM_prodBlocks2, d_KM_prodBlocks3,
                                    d_KM_prodBlocks1, d_kmat_vals, d_Z_vals, d_Q_vals);

            // 5) compute num = <R, Q>
            auto d_num = DeviceVec<T>(1);
            int num_M_vals = M_nnzb * block_dim2;
            CHECK_CUBLAS(
                cublasDdot(cublasHandle, num_M_vals, d_R_vals, 1, d_Q_vals, 1, d_num.getPtr()));
            T num = d_num.createHostVec().getPtr()[0];
            // printf("compute num = %.4e\n", num);

            // 6) compute den = <Q, Q>
            auto d_den = DeviceVec<T>(1);
            CHECK_CUBLAS(
                cublasDdot(cublasHandle, num_M_vals, d_Q_vals, 1, d_Q_vals, 1, d_den.getPtr()));
            T den = d_den.createHostVec().getPtr()[0];
            // printf("compute den = %.4e\n", den);

            // 7) alpha = num / den
            T alpha = num / den;
            // printf("compute alpha = %.4e\n", alpha);

            // 8) M += alpha * Z
            k_add_colored_submat_PFP2<T>
                <<<M_nnzb, 64>>>(M_nnzb, block_dim, alpha, 0, d_Z_vals, d_M_vals);

            // 9) compute objective as well.. obj = <R, R>
            auto d_obj = DeviceVec<T>(1);
            CHECK_CUBLAS(
                cublasDdot(cublasHandle, num_M_vals, d_R_vals, 1, d_R_vals, 1, d_obj.getPtr()));
            T obj = d_obj.createHostVec().getPtr()[0];
            printf("SPAI opt step %d, obj = %.4e\n", iter, obj);
        }
    }

    // private data
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descr_M = 0;
    int optim;  // num SPAI optimization steps

    // TACS assembler
    int N, block_dim, nnodes, block_dim2;

    // temp vecs
    DeviceVec<T> temp;

    // Matrices
    BsrMat<DeviceVec<T>> kmat;  // original stiffness matrix in block CSR format

    // Temporary matrices for preconditioners (not used in this snippet)
    T *d_R_vals, *d_Z_vals, *d_Q_vals;

    // Stiffness matrix sparsity (no-fill) on device.
    int kmat_nnzb;
    int *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals;

    // Preconditioner sparsity (power series fill-in) on device.
    int M_nnzb;
    int *d_M_rowp, *d_M_cols, *d_M_diagp;
    T *d_M_vals;
    int *M_rowp, *M_cols;

    // for mat-mat products
    int KM_nprod;
    int *d_KM_prodBlocks1, *d_KM_prodBlocks2, *d_KM_prodBlocks3;
    int MM_nprod;
    int *d_MM_prodBlocks1, *d_MM_prodBlocks2, *d_MM_prodBlocks3;

    // some useful host vectors
    int *h_kmat_rowp, *h_kmat_cols, *h_M_diagp;
};