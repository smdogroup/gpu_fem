#pragma once
#include "_structured.cuh"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/multigpu_context.h"

template <typename T, class Partitioner, class Basis, ProlongationGeom geom>
class MultiGPUStructuredProlongation {
   public:
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;

    MultiGPUStructuredProlongation(MultiGPUContext *ctx_, Partitioner *fine_part_,
                                   Partitioner *coarse_part_, int nxe_fine_, int nxe_coarse_,
                                   int block_dim_, Mat *fine_mat_, Mat *crs_mat_)
        : ctx(ctx_), fine_part(fine_part_), coarse_part(coarse_part_) {
        fine_num_nodes = fine_part->num_nodes;
        coarse_num_nodes = coarse_part->num_nodes;
        fine_num_elements = fine_part->num_elements;
        coarse_num_elements = coarse_part->num_elements;
        ngpus = ctx->ngpus;
        cublasHandles = ctx->cublasHandles;
        streams = ctx->streams;
        block_dim = block_dim_;
        nxe_fine = nxe_fine_, nxe_coarse = nxe_coarse_;
        weights = new Vec(ctx, fine_part, block_dim);
        // matrices only stored since they contain reduced elem_conn needed for prolong assembly
        fine_mat = fine_mat_, coarse_mat = crs_mat_;

        compute_row_sum_weights();
    }

    void compute_row_sum_weights() {
        weights->zeroAll();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_weights = weights->getLocalPtr(g);
            int start_fine_elem = fine_part->getStartElem(g);
            int start_crs_elem = coarse_part->getStartElem(g);
            int *loc_fine_elem_conn = fine_mat->getLocalElemConn(g);
            int *loc_crs_elem_conn = coarse_mat->getLocalElemConn(g);

            dim3 block(32);
            dim3 grid((loc_fine_nelems + block.x - 1) / block.x);
            k_structured_weights<T, Basis, geom><<<grid, block, 0, streams[g]>>>(
                start_fine_elem, start_crs_elem, loc_fine_elem_conn, loc_crs_elem_conn, block_dim,
                nxe_coarse, nxe_fine, loc_fine_nelems, loc_weights);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        // reduce add then broadcasts from owned to local
        weights->reduceFromLocal();
        weights->expandToLocal();

        // printf("s_prolong weights\n");
        // weights->printValuesOnHost();
    }

    void prolongate(Vec *coarse_in, Vec *fine_out) {
        coarse_in->expandToLocal();
        fine_out->zeroAll();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_coarse = coarse_in->getLocalPtr(g);
            T *loc_fine = fine_out->getLocalPtr(g);
            T *loc_weights = weights->getLocalPtr(g);

            int start_fine_elem = fine_part->getStartElem(g);
            int start_crs_elem = coarse_part->getStartElem(g);
            int *loc_fine_elem_conn = fine_mat->getLocalElemConn(g);
            int *loc_crs_elem_conn = coarse_mat->getLocalElemConn(g);

            dim3 block(32);
            dim3 grid((loc_fine_nelems + block.x - 1) / block.x);
            k_structured_prolongate<T, Basis, geom><<<grid, block, 0, streams[g]>>>(
                start_fine_elem, start_crs_elem, loc_fine_elem_conn, loc_crs_elem_conn, block_dim,
                nxe_coarse, nxe_fine, loc_fine_nelems, loc_weights, loc_coarse, loc_fine);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
        fine_out->reduceFromLocal();
    }

    void restrict_vec(Vec *fine_in, Vec *coarse_out) {
        fine_in->expandToLocal();
        coarse_out->zeroAll();
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            int loc_fine_nelems = fine_part->getLocalNumElements(g);
            T *loc_fine = fine_in->getLocalPtr(g);
            T *loc_coarse = coarse_out->getLocalPtr(g);
            T *loc_weights = weights->getLocalPtr(g);

            int start_fine_elem = fine_part->getStartElem(g);
            int start_crs_elem = coarse_part->getStartElem(g);
            int *loc_fine_elem_conn = fine_mat->getLocalElemConn(g);
            int *loc_crs_elem_conn = coarse_mat->getLocalElemConn(g);

            dim3 block(32);
            dim3 grid((loc_fine_nelems + block.x - 1) / block.x);
            k_structured_restrict<T, Basis, geom><<<grid, block>>>(
                start_fine_elem, start_crs_elem, loc_fine_elem_conn, loc_crs_elem_conn, block_dim,
                nxe_coarse, nxe_fine, loc_fine_nelems, loc_weights, loc_fine, loc_coarse);
            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();

        coarse_out->reduceFromLocal();
    }

    void free() {
        // TBD
    }

   private:
    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;
    MultiGPUContext *ctx = nullptr;
    Partitioner *fine_part = nullptr;
    Partitioner *coarse_part = nullptr;

    int ngpus, block_dim;
    int fine_num_nodes, coarse_num_nodes;
    int fine_num_elements, coarse_num_elements;
    int nxe_coarse, nxe_fine;
    Vec *weights;
    Mat *fine_mat, *coarse_mat;
};