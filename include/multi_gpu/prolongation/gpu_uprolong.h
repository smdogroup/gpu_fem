#pragma once

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
    }

    // void construct_nz_pattern() {
    //     fine_xpts->expandToLocal();
    //     crs_xpts->expandToLocal();

    //     d_n2e_ptr = new int *[ngpus];
    //     d_n2e_elems = new int *[ngpus];
    //     d_n2e_xis = new T *[ngpus];
    //     P_bsr_data = new BsrData[ngpus];
    //     PT_bsr_data = new BsrData[ngpus];

    //     d_P_vals = new DeviceVec<T>[ngpus];
    //     d_PT_vals = new DeviceVec<T>[ngpus];

    //     for (int g = 0; g < ngpus; g++) {
    //         CHECK_CUDA(cudaSetDevice(g));

    //         int *h_fine_loc_elem_conn = fine_mat->getHostLocalElemConn(g);
    //         int *h_crs_loc_elem_conn = coarse_mat->getHostLocalElemConn(g);

    //         int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
    //         int crs_loc_nnodes = crs_xpts->getExpandedNodes(g);

    //         T *h_fine_loc_xpts = fine_xpts->getLocalVecOnHost(g);
    //         T *h_crs_loc_xpts = crs_xpts->getLocalVecOnHost(g);

    //         int fine_nelems = fine_part->getLocalNumElements(g);
    //         int coarse_nelems = coarse_part->getLocalNumElements(g);

    //         int *h_fine_elem_comp = fine_assembler->getLocalElemComponents(g);
    //         int *h_crs_elem_comp = crs_assembler->getLocalElemComponents(g);

    //         init_unstructured_grid_maps<T, Basis, true, true>(
    //             block_dim, h_fine_loc_xpts, h_crs_loc_xpts, fine_loc_nnodes, crs_loc_nnodes,
    //             h_fine_loc_elem_conn, h_crs_loc_elem_conn, fine_nelems, coarse_nelems,
    //             h_fine_elem_comp, h_crs_elem_comp, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
    //             P_bsr_data[g], PT_bsr_data[g], d_P_vals[g], d_PT_vals[g], ELEM_MAX);
    //     }
    // }

    void construct_nz_pattern() {
        fine_xpts->expandToLocal();
        crs_xpts->expandToLocal();

        d_n2e_ptr = new int *[ngpus];
        d_n2e_elems = new int *[ngpus];
        d_n2e_xis = new T *[ngpus];
        d_fine_num_attached_elems = new int *[ngpus];

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
        }

        std::vector<int> global_num_attached(fine_part->num_nodes, 0);

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
            int *h_n2e_ptr = new int[fine_loc_nnodes + 1];

            CHECK_CUDA(cudaMemcpy(h_n2e_ptr, d_n2e_ptr[g], (fine_loc_nnodes + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            for (int loc = 0; loc < fine_loc_nnodes; loc++) {
                int global_node = fine_part->h_local_nodes[g][loc];
                int local_count = h_n2e_ptr[loc + 1] - h_n2e_ptr[loc];
                global_num_attached[global_node] += local_count;
            }

            delete[] h_n2e_ptr;
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int fine_loc_nnodes = fine_xpts->getExpandedNodes(g);
            int *h_local_num_attached = new int[fine_loc_nnodes];

            for (int loc = 0; loc < fine_loc_nnodes; loc++) {
                int global_node = fine_part->h_local_nodes[g][loc];
                h_local_num_attached[loc] = global_num_attached[global_node];

                if (h_local_num_attached[loc] <= 0) {
                    h_local_num_attached[loc] = 1;
                }
            }

            CHECK_CUDA(
                cudaMalloc((void **)&d_fine_num_attached_elems[g], fine_loc_nnodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_fine_num_attached_elems[g], h_local_num_attached,
                                  fine_loc_nnodes * sizeof(int), cudaMemcpyHostToDevice));

            delete[] h_local_num_attached;
        }

        ctx->sync();
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

            dim3 block(32);
            dim3 grid((fine_loc_nnodes + 31) / 32);

            k_prolong_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                d_fine_num_attached_elems[g], fine_loc_nnodes, d_fine_iperm, P_bsr_data[g].rowp,
                P_bsr_data[g].cols, block_dim, d_P_vals[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block, 0, streams[g]>>>(
                d_coarse_iperm, d_crs_loc_elem_conn, d_n2e_ptr[g], d_n2e_elems[g], d_n2e_xis[g],
                d_fine_num_attached_elems[g], fine_loc_nnodes, d_fine_iperm, PT_bsr_data[g].rowp,
                PT_bsr_data[g].cols, block_dim, d_PT_vals[g].getPtr(),
                d_coarse_weights[g].getPtr());

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

    void prolongate(Vec *coarse_in, Vec *fine_out) {
        fine_out->zeroAll(false);
        coarse_in->expandToLocal();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_in->getLocalPtr(g);
            T *loc_fine = fine_out->getLocalPtr(g);

            T a = 1.0;
            T b = 0.0;

            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandles[g], CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, P_bsr_data[g].mb,
                                          P_bsr_data[g].nb, P_bsr_data[g].nnzb, &a, descr_P[g],
                                          d_P_vals[g].getPtr(), P_bsr_data[g].rowp,
                                          P_bsr_data[g].cols, block_dim, loc_coarse, &b, loc_fine));
        }

        ctx->sync();
        fine_out->reduceFromLocal();
    }

    void restrict_vec(Vec *fine_in, Vec *coarse_out) {
        coarse_out->zeroAll(false);
        fine_in->expandToLocal();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            T *loc_coarse = coarse_out->getLocalPtr(g);
            T *loc_fine = fine_in->getLocalPtr(g);

            T a = 1.0;
            T b = 0.0;

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandles[g], CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                PT_bsr_data[g].mb, PT_bsr_data[g].nb, PT_bsr_data[g].nnzb, &a, descr_PT[g],
                d_PT_vals[g].getPtr(), PT_bsr_data[g].rowp, PT_bsr_data[g].cols, block_dim,
                loc_fine, &b, loc_coarse));
        }

        ctx->sync();
        coarse_out->reduceFromLocal();
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
    int **d_fine_num_attached_elems = nullptr;
};