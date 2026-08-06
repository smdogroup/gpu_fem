#pragma once
#include <cmath>
#include <vector>

#include "_unstruct_utils.h"
#include "_unstructured.cuh"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/multigpu_context.h"

template <typename T, class Assembler, class Partitioner>
class MultiGPUUnstructuredProlongation {
   public:
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;
    using Basis = typename Assembler::Basis;
    static constexpr bool is_bsr = true;

    MultiGPUUnstructuredProlongation(MultiGPUContext *ctx_, Partitioner *fine_part_,
                                     Partitioner *coarse_part_, Assembler *fine_assembler_,
                                     Assembler *crs_assembler_, int block_dim_, Mat *fine_mat_,
                                     Mat *crs_mat_, int ELEM_MAX_ = 10)
        : ctx(ctx_), fine_part(fine_part_), coarse_part(coarse_part_) {
        fine_num_nodes = fine_part->num_nodes;
        coarse_num_nodes = coarse_part->num_nodes;
        fine_num_elements = fine_part->num_elements;
        coarse_num_elements = coarse_part->num_elements;
        ngpus = ctx->ngpus;
        cublasHandles = ctx->cublasHandles;
        cusparseHandles = ctx->cusparseHandles;
        streams = ctx->streams;
        block_dim = block_dim_;
        weights = new Vec(ctx, fine_part, block_dim);
        fine_mat = fine_mat_;
        coarse_mat = crs_mat_;
        fine_assembler = fine_assembler_;
        crs_assembler = crs_assembler_;
        fine_xpts = fine_assembler->getDeviceXpts();
        crs_xpts = crs_assembler->getDeviceXpts();
        ELEM_MAX = ELEM_MAX_;

        descr_P = new cusparseMatDescr_t[ngpus];
        descr_PT = new cusparseMatDescr_t[ngpus];

        for (int g = 0; g < ngpus; g++) {
            descr_P[g] = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_P[g]));
            CHECK_CUSPARSE(cusparseSetMatType(descr_P[g], CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_P[g], CUSPARSE_INDEX_BASE_ZERO));

            descr_PT[g] = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_PT[g]));
            CHECK_CUSPARSE(cusparseSetMatType(descr_PT[g], CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_PT[g], CUSPARSE_INDEX_BASE_ZERO));
        }

        construct_nz_pattern();
        assemble_matrices();

        printf("FINAL constructor check for prolongation object\n");
        debug_check_P("P at end of MultiGPUUnstructuredProlongation constructor");

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());

            int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
            int crs_loc_nnodes = crs_xpts->getExpandedNodes(g);

            checkBsrMat("P final constructor check", g, P_bsr_data[g], d_P_vals[g], fine_loc_nnodes,
                        crs_loc_nnodes);
            checkBsrMat("PT final constructor check", g, PT_bsr_data[g], d_PT_vals[g],
                        crs_loc_nnodes, fine_loc_nnodes);
        }
    }

    void construct_nz_pattern() {
        fine_xpts->expandToLocal();
        crs_xpts->expandToLocal();

        d_n2e_ptr = new int *[ngpus];
        d_n2e_elems = new int *[ngpus];
        d_n2e_xis = new T *[ngpus];
        P_bsr_data = new BsrData[ngpus];
        PT_bsr_data = new BsrData[ngpus];

        d_P_vals = new DeviceVec<T>[ngpus];
        d_PT_vals = new DeviceVec<T>[ngpus];

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int *h_fine_loc_elem_conn = fine_mat->getHostLocalElemConn(g);
            int *h_crs_loc_elem_conn = coarse_mat->getHostLocalElemConn(g);

            int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
            int crs_loc_nnodes = crs_xpts->getExpandedNodes(g);

            T *h_fine_loc_xpts = fine_xpts->getLocalVecOnHost(g);
            T *h_crs_loc_xpts = crs_xpts->getLocalVecOnHost(g);

            int fine_nelems = fine_part->getLocalNumElements(g);
            int coarse_nelems = coarse_part->getLocalNumElements(g);

            int *h_fine_elem_comp = fine_assembler->getLocalElemComponents(g);
            int *h_crs_elem_comp = crs_assembler->getLocalElemComponents(g);

            init_unstructured_grid_maps<T, Basis, true, true>(
                block_dim, h_fine_loc_xpts, h_crs_loc_xpts, fine_loc_nnodes, crs_loc_nnodes,
                h_fine_loc_elem_conn, h_crs_loc_elem_conn, fine_nelems, coarse_nelems,
                h_fine_elem_comp, h_crs_elem_comp, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                P_bsr_data[g], PT_bsr_data[g], d_P_vals[g], d_PT_vals[g], ELEM_MAX);

            checkBsrMat("P after construct", g, P_bsr_data[g], d_P_vals[g], fine_loc_nnodes,
                        crs_loc_nnodes);
            checkBsrMat("PT after construct", g, PT_bsr_data[g], d_PT_vals[g], crs_loc_nnodes,
                        fine_loc_nnodes);
        }
    }

    void assemble_matrices() {
        d_coarse_weights = new DeviceVec<T>[ngpus];

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
            int crs_loc_nnodes = crs_xpts->getExpandedNodes(g);

            int *d_crs_loc_elem_conn = coarse_mat->getLocalElemConn(g);
            int *d_fine_iperm = P_bsr_data[g].iperm;
            int *d_coarse_iperm = PT_bsr_data[g].iperm;

            int N_loc_coarse = crs_loc_nnodes * block_dim;

            d_coarse_weights[g] = DeviceVec<T>(N_loc_coarse);
            d_coarse_weights[g].zeroValues();

            printf("\nassemble_matrices gpu %d\n", g);
            printf("fine_loc_nnodes=%d crs_loc_nnodes=%d block_dim=%d N_loc_coarse=%d\n",
                   fine_loc_nnodes, crs_loc_nnodes, block_dim, N_loc_coarse);

            checkBsrMat("P before asm", g, P_bsr_data[g], d_P_vals[g], fine_loc_nnodes,
                        crs_loc_nnodes);
            checkBsrMat("PT before asm", g, PT_bsr_data[g], d_PT_vals[g], crs_loc_nnodes,
                        fine_loc_nnodes);

            dim3 block(32);
            dim3 grid((fine_loc_nnodes + 31) / 32);

            printf("gpu %d: enter k_prolong_mat_assembly\n", g);

            k_prolong_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                fine_loc_nnodes, d_fine_iperm, P_bsr_data[g].rowp, P_bsr_data[g].cols, block_dim,
                d_P_vals[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            printf("gpu %d: exit k_prolong_mat_assembly\n", g);

            checkBsrMat("P after P asm", g, P_bsr_data[g], d_P_vals[g], fine_loc_nnodes,
                        crs_loc_nnodes);
            checkBsrMat("PT after P asm", g, PT_bsr_data[g], d_PT_vals[g], crs_loc_nnodes,
                        fine_loc_nnodes);

            printf("gpu %d: enter k_restrict_mat_assembly\n", g);

            k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                fine_loc_nnodes, d_fine_iperm, PT_bsr_data[g].rowp, PT_bsr_data[g].cols, block_dim,
                d_PT_vals[g].getPtr(), d_coarse_weights[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            printf("gpu %d: exit k_restrict_mat_assembly\n", g);

            checkBsrMat("P after PT asm", g, P_bsr_data[g], d_P_vals[g], fine_loc_nnodes,
                        crs_loc_nnodes);
            checkBsrMat("PT after PT asm", g, PT_bsr_data[g], d_PT_vals[g], crs_loc_nnodes,
                        fine_loc_nnodes);
            checkVecLocal("coarse weights after PT asm", g, d_coarse_weights[g].getPtr(),
                          N_loc_coarse);
        }
    }

    void prolongate(Vec *coarse_in, Vec *fine_out) {
        printf("prolongate: start\n");

        coarse_in->expandToLocal();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();
        printf("prolongate: expandToLocal done\n");

        fine_out->zeroAll();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();
        printf("prolongate: zeroAll done\n");

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_in->getLocalPtr(g);
            T *loc_fine = fine_out->getLocalPtr(g);

            int mb = P_bsr_data[g].mb;
            int nb = P_bsr_data[g].nb;
            int nnzb = P_bsr_data[g].nnzb;

            int fine_exp_size = fine_out->getLocalSize(g);
            int crs_exp_size = coarse_in->getLocalSize(g);

            printf("prolong gpu %d: mb=%d nb=%d bdim=%d nnzb=%d fine_exp=%d crs_exp=%d\n", g, mb,
                   nb, block_dim, nnzb, fine_exp_size, crs_exp_size);

            if (fine_exp_size != mb * block_dim) {
                printf("BAD prolong gpu %d: fine_exp_size=%d mb*block_dim=%d\n", g, fine_exp_size,
                       mb * block_dim);
            }
            if (crs_exp_size != nb * block_dim) {
                printf("BAD prolong gpu %d: crs_exp_size=%d nb*block_dim=%d\n", g, crs_exp_size,
                       nb * block_dim);
            }

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            checkBsrMat("P before bsrmv", g, P_bsr_data[g], d_P_vals[g], mb, nb);
            checkVecLocal("coarse input before bsrmv", g, loc_coarse, crs_exp_size);
            checkVecLocal("fine output before bsrmv", g, loc_fine, fine_exp_size);

            T a = 1.0;
            T b = 0.0;

            printf("prolongate gpu %d: entering cusparseDbsrmv\n", g);

            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandles[g], CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &a,
                                          descr_P[g], d_P_vals[g].getPtr(), P_bsr_data[g].rowp,
                                          P_bsr_data[g].cols, block_dim, loc_coarse, &b, loc_fine));

            printf("prolongate gpu %d: exited cusparseDbsrmv\n", g);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            checkVecLocal("fine output after bsrmv", g, loc_fine, fine_exp_size);
        }

        printf("prolongate: before global sync\n");
        ctx->sync();
        printf("prolongate: after global sync\n");

        fine_out->reduceFromLocal();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();

        printf("prolongate: reduceFromLocal done\n");
    }

    void debug_check_P(const char *tag) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());
            checkBsrMat(tag, g, P_bsr_data[g], d_P_vals[g], P_bsr_data[g].mb, P_bsr_data[g].nb);
        }
    }

    void restrict_vec(Vec *fine_in, Vec *coarse_out) {
        printf("restrict_vec: start\n");

        fine_in->expandToLocal();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();

        coarse_out->zeroAll();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_out->getLocalPtr(g);
            T *loc_fine = fine_in->getLocalPtr(g);

            int mb = PT_bsr_data[g].mb;
            int nb = PT_bsr_data[g].nb;
            int nnzb = PT_bsr_data[g].nnzb;

            int crs_exp_size = coarse_out->getLocalSize(g);
            int fine_exp_size = fine_in->getLocalSize(g);

            printf("restrict gpu %d: mb=%d nb=%d bdim=%d nnzb=%d crs_exp=%d fine_exp=%d\n", g, mb,
                   nb, block_dim, nnzb, crs_exp_size, fine_exp_size);

            if (crs_exp_size != mb * block_dim) {
                printf("BAD restrict gpu %d: crs_exp_size=%d mb*block_dim=%d\n", g, crs_exp_size,
                       mb * block_dim);
            }
            if (fine_exp_size != nb * block_dim) {
                printf("BAD restrict gpu %d: fine_exp_size=%d nb*block_dim=%d\n", g, fine_exp_size,
                       nb * block_dim);
            }

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            checkBsrMat("PT before bsrmv", g, PT_bsr_data[g], d_PT_vals[g], mb, nb);
            checkVecLocal("fine input before restrict bsrmv", g, loc_fine, fine_exp_size);
            checkVecLocal("coarse output before restrict bsrmv", g, loc_coarse, crs_exp_size);

            T a = 1.0;
            T b = 0.0;

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandles[g], CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                nb, nnzb, &a, descr_PT[g], d_PT_vals[g].getPtr(), PT_bsr_data[g].rowp,
                PT_bsr_data[g].cols, block_dim, loc_fine, &b, loc_coarse));

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            checkVecLocal("coarse output after restrict bsrmv", g, loc_coarse, crs_exp_size);
        }

        ctx->sync();

        coarse_out->reduceFromLocal();
        CHECK_CUDA(cudaGetLastError());
        ctx->sync();

        printf("restrict_vec: done\n");
    }

    void free() {
        if (descr_P) {
            for (int g = 0; g < ngpus; g++) {
                if (descr_P[g]) CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_P[g]));
            }
            delete[] descr_P;
            descr_P = nullptr;
        }
        if (descr_PT) {
            for (int g = 0; g < ngpus; g++) {
                if (descr_PT[g]) CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_PT[g]));
            }
            delete[] descr_PT;
            descr_PT = nullptr;
        }
        delete weights;
        weights = nullptr;
        delete[] d_n2e_ptr;
        delete[] d_n2e_elems;
        delete[] d_n2e_xis;
        delete[] P_bsr_data;
        delete[] PT_bsr_data;
        delete[] d_P_vals;
        delete[] d_PT_vals;
        delete[] d_coarse_weights;
        d_n2e_ptr = nullptr;
        d_n2e_elems = nullptr;
        d_n2e_xis = nullptr;
        P_bsr_data = nullptr;
        PT_bsr_data = nullptr;
        d_P_vals = nullptr;
        d_PT_vals = nullptr;
        d_coarse_weights = nullptr;
    }

   private:
    void checkDevicePtr(const char *name, int g, const void *ptr) {
        if (!ptr) {
            printf("BAD PTR %s gpu %d: nullptr\n", name, g);
            return;
        }

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

        if (err != cudaSuccess) {
            printf("BAD PTR %s gpu %d: ptr=%p cudaPointerGetAttributes failed: %s\n", name, g, ptr,
                   cudaGetErrorString(err));
            cudaGetLastError();
            return;
        }

#if CUDART_VERSION >= 10000
        printf("PTR %s gpu %d: ptr=%p type=%d device=%d\n", name, g, ptr, (int)attr.type,
               attr.device);
#else
        printf("PTR %s gpu %d: ptr=%p memoryType=%d device=%d\n", name, g, ptr,
               (int)attr.memoryType, attr.device);
#endif
    }

    int checkBsrMat(const char *name, int g, BsrData &A, DeviceVec<T> &vals, int expected_mb,
                    int expected_nb) {
        int mb = A.mb;
        int nb = A.nb;
        int nnzb = A.nnzb;

        printf("%s gpu %d: mb=%d nb=%d nnzb=%d block_dim=%d expected_mb=%d expected_nb=%d\n", name,
               g, mb, nb, nnzb, block_dim, expected_mb, expected_nb);

        if (mb != expected_mb || nb != expected_nb) {
            printf("BAD %s gpu %d: mb/nb mismatch\n", name, g);
        }

        checkDevicePtr("rowp", g, A.rowp);
        checkDevicePtr("cols", g, A.cols);
        checkDevicePtr("vals", g, vals.getPtr());

        std::vector<int> h_rowp(mb + 1);
        std::vector<int> h_cols(nnzb);

        CHECK_CUDA(
            cudaMemcpy(h_rowp.data(), A.rowp, (mb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_cols.data(), A.cols, nnzb * sizeof(int), cudaMemcpyDeviceToHost));

        int bad = 0;

        if (h_rowp[0] != 0 || h_rowp[mb] != nnzb) {
            printf("BAD %s gpu %d: rowp[0]=%d rowp[mb]=%d nnzb=%d\n", name, g, h_rowp[0],
                   h_rowp[mb], nnzb);
            bad = 1;
        }

        for (int i = 0; i < mb; i++) {
            if (h_rowp[i] < 0 || h_rowp[i] > h_rowp[i + 1] || h_rowp[i + 1] > nnzb) {
                printf("BAD %s rowp gpu %d: i=%d rowp[i]=%d rowp[i+1]=%d nnzb=%d\n", name, g, i,
                       h_rowp[i], h_rowp[i + 1], nnzb);
                bad = 1;
                break;
            }
        }

        for (int jp = 0; jp < nnzb; jp++) {
            if (h_cols[jp] < 0 || h_cols[jp] >= nb) {
                printf("BAD %s col gpu %d: jp=%d col=%d nb=%d\n", name, g, jp, h_cols[jp], nb);
                bad = 1;
                break;
            }
        }

        size_t nvals = (size_t)nnzb * block_dim * block_dim;
        std::vector<T> h_vals(nvals);

        CHECK_CUDA(
            cudaMemcpy(h_vals.data(), vals.getPtr(), nvals * sizeof(T), cudaMemcpyDeviceToHost));

        int bad_vals = 0;
        T max_abs = 0.0;
        size_t max_i = 0;

        for (size_t i = 0; i < nvals; i++) {
            T v = h_vals[i];
            if (!std::isfinite((double)v)) {
                printf("BAD %s val gpu %d: i=%zu val=%e\n", name, g, i, (double)v);
                bad_vals = 1;
                break;
            }

            T av = fabs(v);
            if (av > max_abs) {
                max_abs = av;
                max_i = i;
            }
        }

        printf("%s gpu %d: structure_bad=%d vals_bad=%d max_abs_val=%e at i=%zu\n", name, g, bad,
               bad_vals, (double)max_abs, max_i);

        return bad || bad_vals;
    }

    int checkVecLocal(const char *name, int g, T *ptr, int n) {
        printf("%s gpu %d: ptr=%p n=%d\n", name, g, (void *)ptr, n);

        checkDevicePtr(name, g, ptr);

        std::vector<T> h(n);
        CHECK_CUDA(cudaMemcpy(h.data(), ptr, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));

        int bad = 0;
        T nrm2 = 0.0;
        T max_abs = 0.0;
        int max_i = 0;

        for (int i = 0; i < n; i++) {
            T v = h[i];

            if (!std::isfinite((double)v)) {
                printf("BAD %s gpu %d: i=%d val=%e\n", name, g, i, (double)v);
                bad = 1;
                break;
            }

            nrm2 += v * v;

            T av = fabs(v);
            if (av > max_abs) {
                max_abs = av;
                max_i = i;
            }
        }

        printf("%s gpu %d: bad=%d nrm=%e max_abs=%e at i=%d\n", name, g, bad, (double)sqrt(nrm2),
               (double)max_abs, max_i);

        return bad;
    }

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;
    MultiGPUContext *ctx = nullptr;
    Partitioner *fine_part = nullptr;
    Partitioner *coarse_part = nullptr;

    Assembler *fine_assembler = nullptr;
    Assembler *crs_assembler = nullptr;

    int ngpus = 0;
    int block_dim = 0;
    int fine_num_nodes = 0;
    int coarse_num_nodes = 0;
    int fine_num_elements = 0;
    int coarse_num_elements = 0;
    int nxe_coarse = 0;
    int nxe_fine = 0;

    Vec *weights = nullptr;
    Mat *fine_mat = nullptr;
    Mat *coarse_mat = nullptr;
    Vec *fine_xpts = nullptr;
    Vec *crs_xpts = nullptr;

    cusparseMatDescr_t *descr_P = nullptr;
    cusparseMatDescr_t *descr_PT = nullptr;

    BsrData *P_bsr_data = nullptr;
    BsrData *PT_bsr_data = nullptr;

    DeviceVec<T> *d_P_vals = nullptr;
    DeviceVec<T> *d_PT_vals = nullptr;
    DeviceVec<T> *d_coarse_weights = nullptr;

    int **d_n2e_ptr = nullptr;
    int **d_n2e_elems = nullptr;
    T **d_n2e_xis = nullptr;

    int ELEM_MAX = 10;
};