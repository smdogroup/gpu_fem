#pragma once
#include <algorithm>
#include <chrono>
#include <cstring>

#include "linalg/bsr_data.h"
#include "linalg/vec.h"  // for DeviceVec
#include "matvec/_matvec.cuh"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class MultiGPUPartition, class SingleGPUPartition>
class SingleGPUDirectLU {
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
          streams(ctx_->streams) {
        CHECK_CUDA(cudaSetDevice(0));
        cublasHandle = cublasHandles[0];
        cusparseHandle = cusparseHandles[0];

        CHECK_CUBLAS(cublasSetStream(cublasHandle, streams[0]));
        CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, streams[0]));

        compute_reorderedLU_pattern();
        compute_block_copy_maps();
        setup_LU_solves();
    }

    void free() {
        CHECK_CUDA(cudaSetDevice(0));

        if (pBuffer) cudaFree(pBuffer);
        if (d_dest_blocks) cudaFree(d_dest_blocks);
        if (d_mat_lu_vals) cudaFree(d_mat_lu_vals);

        if (descrK) cusparseDestroyMatDescr(descrK);
        if (descr_M) cusparseDestroyMatDescr(descr_M);
        if (descr_L) cusparseDestroyMatDescr(descr_L);
        if (descr_U) cusparseDestroyMatDescr(descr_U);

        if (info_M) cusparseDestroyBsrilu02Info(info_M);
        if (info_L) cusparseDestroyBsrsv2Info(info_L);
        if (info_U) cusparseDestroyBsrsv2Info(info_U);

        delete[] h_dest_blocks;

        delete s_rhs;
        delete s_temp;
        delete s_soln;
    }

    void compute_reorderedLU_pattern() {
        CHECK_CUDA(cudaSetDevice(0));

        // Original no-fill matrix pattern in the original node ordering
        h_nofill_bsr_data = BsrData(spart->num_elements, spart->num_nodes, spart->nodes_per_elem,
                                    mat->getBlockDim(), spart->h_elem_conn);

        printf("spart num_elements=%d, num_nodes=%d\n", spart->num_elements, spart->num_nodes);

        num_nodes = h_nofill_bsr_data.mb;
        mb = num_nodes;

        nofill_nnzb = h_nofill_bsr_data.nnzb;
        h_nofill_rowp = new int[num_nodes + 1];
        h_nofill_cols = new int[nofill_nnzb];
        memcpy(h_nofill_rowp, h_nofill_bsr_data.rowp, (num_nodes + 1) * sizeof(int));
        memcpy(h_nofill_cols, h_nofill_bsr_data.cols, nofill_nnzb * sizeof(int));

        // Separate LU pattern object
        h_lu_bsr_data = h_nofill_bsr_data;
        h_lu_bsr_data.AMD_reordering();
        h_lu_bsr_data.compute_full_LU_pattern();

        nnzb = h_lu_bsr_data.nnzb;
        h_perm = new int[num_nodes];
        h_iperm = new int[num_nodes];
        h_rowp = new int[num_nodes + 1];
        h_cols = new int[nnzb];
        memcpy(h_perm, h_lu_bsr_data.perm, num_nodes * sizeof(int));
        memcpy(h_iperm, h_lu_bsr_data.iperm, num_nodes * sizeof(int));
        memcpy(h_rowp, h_lu_bsr_data.rowp, (num_nodes + 1) * sizeof(int));
        memcpy(h_cols, h_lu_bsr_data.cols, nnzb * sizeof(int));

        d_lu_bsr_data = h_lu_bsr_data.createDeviceBsrData();

        d_perm = d_lu_bsr_data.perm;
        d_iperm = d_lu_bsr_data.iperm;
        d_rowp = d_lu_bsr_data.rowp;
        d_cols = d_lu_bsr_data.cols;

        block_dim = d_lu_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;

        nnz = nnzb * block_dim2;
        nofill_nnz = nofill_nnzb * block_dim2;

        CHECK_CUDA(cudaMalloc((void **)&d_mat_lu_vals, nnz * sizeof(T)));
        CHECK_CUDA(cudaMemsetAsync(d_mat_lu_vals, 0, nnz * sizeof(T), streams[0]));

        N = spart->num_nodes * block_dim;

        s_rhs = new DeviceVec<T>(N);
        s_temp = new DeviceVec<T>(N);
        s_soln = new DeviceVec<T>(N);

        CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    }

    void compute_block_copy_maps() {
        CHECK_CUDA(cudaSetDevice(0));

        h_dest_blocks = new int[nofill_nnzb];
        std::fill(h_dest_blocks, h_dest_blocks + nofill_nnzb, -1);

        for (int i = 0; i < num_nodes; i++) {
            int i_lu = h_iperm[i];

            for (int jp = h_nofill_rowp[i]; jp < h_nofill_rowp[i + 1]; jp++) {
                int j = h_nofill_cols[jp];
                int j_lu = h_iperm[j];

                int matching_block = -1;

                for (int kp = h_rowp[i_lu]; kp < h_rowp[i_lu + 1]; kp++) {
                    if (h_cols[kp] == j_lu) {
                        matching_block = kp;
                        break;
                    }
                }

                if (matching_block >= 0) {
                    h_dest_blocks[jp] = matching_block;
                } else {
                    printf(
                        "SingleGPUDirectLU error: missing nofill block "
                        "original (%d,%d), reordered (%d,%d)\n",
                        i, j, i_lu, j_lu);
                }
            }
        }

        CHECK_CUDA(cudaMalloc((void **)&d_dest_blocks, nofill_nnzb * sizeof(int)));
        CHECK_CUDA(cudaMemcpyAsync(d_dest_blocks, h_dest_blocks, nofill_nnzb * sizeof(int),
                                   cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    }

    void setup_LU_solves() {
        CHECK_CUDA(cudaSetDevice(0));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
        CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&info_M));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
        CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
        CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));
        CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_L));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_U));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
        CHECK_CUSPARSE(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
        CHECK_CUSPARSE(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));
        CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_U));

        CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M,
                                                    d_mat_lu_vals, d_rowp, d_cols, block_dim,
                                                    info_M, &pBufferSize_M));

        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                  d_mat_lu_vals, d_rowp, d_cols, block_dim, info_L,
                                                  &pBufferSize_L));

        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                  d_mat_lu_vals, d_rowp, d_cols, block_dim, info_U,
                                                  &pBufferSize_U));

        pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));
        CHECK_CUDA(cudaMalloc((void **)&pBuffer, pBufferSize));

        CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_M,
                                                  d_mat_lu_vals, d_rowp, d_cols, block_dim, info_M,
                                                  policy_M, pBuffer));

        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
        if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
            printf("SingleGPUDirectLU structural zero pivot at block %d\n", structural_zero);
        }

        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                d_mat_lu_vals, d_rowp, d_cols, block_dim, info_L,
                                                policy_L, pBuffer));

        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                d_mat_lu_vals, d_rowp, d_cols, block_dim, info_U,
                                                policy_U, pBuffer));

        CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    }

    void factor() {
        CHECK_CUDA(cudaSetDevice(0));

        CHECK_CUDA(cudaMemsetAsync(d_mat_lu_vals, 0, nnz * sizeof(T), streams[0]));

        T *d_mat_nofill_vals = mat->getLocalVals(0);

        dim3 block(128);
        dim3 grid((nofill_nnz + block.x - 1) / block.x);

        k_copy_nofill_vals_to_lu_vals<T><<<grid, block, 0, streams[0]>>>(
            nofill_nnzb, block_dim2, d_dest_blocks, d_mat_nofill_vals, d_mat_lu_vals);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));

        // printMatValues();  // for debug

        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, d_mat_lu_vals,
                                         d_rowp, d_cols, block_dim, info_M, policy_M, pBuffer));

        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
            printf("SingleGPUDirectLU numerical zero pivot at block %d\n", numerical_zero);
        }

        CHECK_CUDA(cudaStreamSynchronize(streams[0]));
    }

    void printMatValues() {  // TODO
    }

    void solve(MultiGPUVec *rhs, MultiGPUVec *soln) {
        rhs->copyToSingleGPU(s_rhs->getPtr());

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));

        s_rhs->permuteData(block_dim, d_iperm);
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));

        const T alpha = 1.0;

        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_mat_lu_vals, d_rowp, d_cols,
            block_dim, info_L, s_rhs->getPtr(), s_temp->getPtr(), policy_L, pBuffer));

        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_U, mb, nnzb, &alpha, descr_U, d_mat_lu_vals, d_rowp, d_cols,
            block_dim, info_U, s_temp->getPtr(), s_soln->getPtr(), policy_U, pBuffer));

        CHECK_CUDA(cudaStreamSynchronize(streams[0]));

        s_soln->permuteData(block_dim, d_perm);
        CHECK_CUDA(cudaStreamSynchronize(streams[0]));

        soln->copyFromSingleGPU(s_soln->getPtr());
        ctx->sync();
    }

   private:
    MultiGPUContext *ctx = nullptr;
    MultiGPUPartition *mpart = nullptr;
    SingleGPUPartition *spart = nullptr;
    Mat *mat = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;

    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;

    BsrData h_nofill_bsr_data, h_lu_bsr_data, d_lu_bsr_data;

    int num_nodes = 0;
    int mb = 0;
    int N = 0;

    int *h_perm = nullptr;
    int *h_iperm = nullptr;
    int *d_perm = nullptr;
    int *d_iperm = nullptr;

    int nofill_nnzb = 0;
    int nofill_nnz = 0;
    int nnzb = 0;
    int nnz = 0;

    int block_dim = 0;
    int block_dim2 = 0;

    int *h_dest_blocks = nullptr;
    int *d_dest_blocks = nullptr;

    int *h_nofill_rowp = nullptr;
    int *h_nofill_cols = nullptr;
    int *h_rowp = nullptr;
    int *h_cols = nullptr;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;

    T *d_mat_lu_vals = nullptr;

    DeviceVec<T> *s_rhs = nullptr;
    DeviceVec<T> *s_soln = nullptr;
    DeviceVec<T> *s_temp = nullptr;

    cusparseMatDescr_t descrK = nullptr;

    cusparseMatDescr_t descr_M = nullptr;
    bsrilu02Info_t info_M = nullptr;

    cusparseMatDescr_t descr_L = nullptr;
    cusparseMatDescr_t descr_U = nullptr;
    bsrsv2Info_t info_L = nullptr;
    bsrsv2Info_t info_U = nullptr;

    void *pBuffer = nullptr;

    int pBufferSize_M = 0;
    int pBufferSize_L = 0;
    int pBufferSize_U = 0;
    int pBufferSize = 0;

    int structural_zero = -1;
    int numerical_zero = -1;

    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseStatus_t status;
};