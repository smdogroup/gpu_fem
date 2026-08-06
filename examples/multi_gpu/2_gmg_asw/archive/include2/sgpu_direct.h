#pragma once
#include <chrono>

#include "../solve_utils.h"
#include "linalg/bsr_data.h"
#include "linalg/vec.h"  // for DeviceVec

template <typename T, class MultiGPUPartition, class SingleGPUPartition>
class SingleGPUDirectLU {
    // single GPU coarse direct solver for multi GPU format
   public:
    using MultiGPUVec = GPUvec<T, MultiGPUPartition>;
    using SingleGPUVec = GPUvec<T, SingleGPUPartition>;
    using Mat = GPUbsrmat<T, SingleGPUPartition>;

    SingleGPUDirectLU(MultiGPUContext *ctx_, MultiGPUPartition *mpart_, SingleGPUPartition *spart_,
                      Mat *mat_)
        : ctx(ctx_),
          mpart(mpart_),
          spart(spart_),
          mat(mat_),
          cublasHandles(ctx_->cublasHandles),
          cusparseHandles(ctx_->cusparseHandles),
          streams(ctx_->streams),
    {
        compute_reorderedLU_pattern();
        compute_block_copy_maps();
    }

    void compute_reorderedLU_pattern() {
        // get bsr data of original nofill pattern
        h_nofill_bsr_data = BsrData(spart->num_elements, spart->num_nodes, spart->nodes_per_elem,
                                    spart->block_dim, spart->h_elem_conn);
        num_nodes = h_nofill_bsr_data.mb;
        nofill_nnzb = h_nofill_bsr_data.nnzb;
        h_nofill_rowp = h_nofill_bsr_data.rowp;
        h_nofill_cols = h_nofill_bsr_data.cols;

        h_lu_bsr_data = h_nofill_bsr_data.AMD_reordering();
        h_lu_bsr_data = h_lu_bsr_data.compute_full_LU_pattern();
        h_perm = h_lu_bsr_data.perm, h_iperm = h_lu_bsr_data.iperm;
        h_rowp = h_lu_bsr_data.rowp;
        h_cols = h_lu_bsr_data.cols;

        d_lu_bsr_data = h_lu_bsr_data.createDeviceBsrData();
        d_perm = d_lu_bsr_data.perm, d_iperm = d_lu_bsr_data.iperm;
        nnzb = d_lu_bsr_data.nnzb;
        block_dim = d_lu_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;
        nnz = nnzb * block_dim2;
        nofill_nnz = nofill_nnzb * block_dim2;

        d_mat_lu_vals = DeviceVec<T>(nnz).getPtr();

        N = spart->num_nodes() * block_dim;
        CHECK_CUDA(cudaSetDevice(0));
        s_rhs = new DeviceVec<T>(N);
        s_temp = new DeviceVec<T>(N);
        s_soln = new DeviceVec<T>(N);
    }

    void compute_block_copy_maps() {
        h_dest_blocks = new int[nofill_nnzb];
        memset(h_dest_blocks, 0, nofill_nnzb * sizeof(int));
        for (int i = 0; i < num_nodes; i++) {
            for (int jp = h_nofill_rowp[i]; jp < h_nofill_rowp[i + 1]; jp++) {
                int j = h_nofill_cols[jp];

                // find matching block in other matrix
                int matching_block = -1;
                int i2 = h_iperm[i];
                for (int kp = h_rowp[i2]; kp < h_rowp[i2 + 1]; kp++) {
                    int k2 = h_cols[kp];
                    int k = h_perm[k2];
                    if (k == j) {
                        matching_block = kp;
                    }
                }

                if (matching_block) {
                    h_dest_blocks[jp] = kp;
                } else {
                    // throw error
                }
            }
        }

        // set on root GPU (GPU 0)
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc((void **)&d_dest_blocks, nofill_nnzb * sizeof(int)));
        cudaMemcpy(d_dest_blocks, h_dest_blocks, nofill_nnzb * sizeof(int), cudaMemcpyHostToDevice);
    }

    void setup_LU_solves() {
        // for checking residual with SpMV
        CHECK_CUDA(cudaSetDevice(0));
        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // startup factorization steps
        // -----------------------------------

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
        CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M, d_vals,
                                                    d_rowp, d_cols, block_dim, info_M,
                                                    &pBufferSize_M));
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                  d_vals, d_rowp, d_cols, block_dim, info_L,
                                                  &pBufferSize_L));
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                  d_vals, d_rowp, d_cols, block_dim, info_U,
                                                  &pBufferSize_U));
        pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
        // cudaMalloc((void **)&pBuffer, pBufferSize);
        cudaMalloc((void **)&pBuffer, pBufferSize);

        // perform ILU symbolic factorization on L
        CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_M, d_vals,
                                                  d_rowp, d_cols, block_dim, info_M, policy_M,
                                                  pBuffer));
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
        }

        // analyze sparsity patern of L for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                d_vals, d_rowp, d_cols, block_dim, info_L, policy_L,
                                                pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // analyze sparsity pattern of U for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                d_vals, d_rowp, d_cols, block_dim, info_U, policy_U,
                                                pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void factor() {
        // TODO : kernel call on GPU 0 to copy matrix values using block copy maps from
        // src: mat->getLocalVals(0);
        // dst: d_mat_vals;

        // perform ILU numeric factorization (with M policy)
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, d_vals_ILU0,
                                         d_rowp, d_cols, block_dim, info_M, policy_M, pBuffer));
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void solve(MultiGPUVec *rhs, MultiGPUVec *soln) {
        // copy from multi to single GPU vec
        rhs->copyToSingleGPU(s_rhs->getPtr());

        // permute from baseline to AMD ordering
        s_rhs->permuteData(block_dim, d_iperm);

        // then solve inside here as usual
        CHECK_CUDA(cudaSetDevice(0));
        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals_ILU0, d_rowp, d_cols,
            block_dim, info_L, s_rhs->getPtr(), s_temp->getPtr(), policy_L, pBuffer));

        // triangular solve U*y = z
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_U, mb, nnzb, &alpha, descr_U, d_vals_ILU0, d_rowp, d_cols,
            block_dim, info_U, s_temp->getPtr(), s_soln->getPtr(), policy_U, pBuffer));

        // permute from AMD ordering back to baseline
        s_soln->permuteData(block_dim, d_perm);

        // broadcast to multi-GPU vector
        soln->copyFromSingleGPU(s_soln->getPtr());
    }

   private:
    MultiGPUContext *ctx = nullptr;
    MultiGPUPartition *mpart;
    SingleGPUPartition *spart;
    Mat *mat;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;

    BsrData h_nofill_bsr_data, h_lu_bsr_data, d_lu_bsr_data;
    int num_nodes;
    int *h_perm, *h_iperm;
    int *d_perm, *d_iperm;
    int nofill_nnzb, nofill_nnz;
    int nnzb, nnz, block_dim, block_dim2;
    int *h_dest_blocks, *d_dest_blocks;
    int *h_nofill_rowp, *h_nofill_cols;
    int *h_rowp, *h_cols;

    T *d_mat_lu_vals;
    DeviceVec<T> *s_rhs, s_soln, s_temp;

    // matrix utilities
    // cusparse and cublas data
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseMatDescr_t descrK;

    // factor utilities
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M =
        CUSPARSE_SOLVE_POLICY_USE_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;
};