#pragma once
#include "cuda_utils.h"

class MultiGPUContext {
   public:
    MultiGPUContext(bool debug_ = false) : debug(debug_) {
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaGetDeviceCount(&ngpus));

        if (debug) {
            ngpus = 5;  // logical GPUs all mapped to device 0
        }

        cublasHandles = new cublasHandle_t[ngpus];
        cusparseHandles = new cusparseHandle_t[ngpus];
        streams = new cudaStream_t[ngpus];

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            cublasHandles[g] = nullptr;
            cusparseHandles[g] = nullptr;

            CHECK_CUDA(cudaStreamCreate(&streams[g]));

            CHECK_CUBLAS(cublasCreate(&cublasHandles[g]));
            CHECK_CUSPARSE(cusparseCreate(&cusparseHandles[g]));

            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));
        }
    }

    MultiGPUContext(int ngpus_override_) : debug(false) {
        CHECK_CUDA(cudaSetDevice(0));
        // CHECK_CUDA(cudaGetDeviceCount(&ngpus));
        ngpus = ngpus_override_;

        cublasHandles = new cublasHandle_t[ngpus];
        cusparseHandles = new cusparseHandle_t[ngpus];
        streams = new cudaStream_t[ngpus];

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            cublasHandles[g] = nullptr;
            cusparseHandles[g] = nullptr;

            CHECK_CUDA(cudaStreamCreate(&streams[g]));

            CHECK_CUBLAS(cublasCreate(&cublasHandles[g]));
            CHECK_CUSPARSE(cusparseCreate(&cusparseHandles[g]));

            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));
        }
    }

    void free() {
        sync();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            if (cublasHandles[g]) CHECK_CUBLAS(cublasDestroy(cublasHandles[g]));
            if (cusparseHandles[g]) CHECK_CUSPARSE(cusparseDestroy(cusparseHandles[g]));

            CHECK_CUDA(cudaStreamDestroy(streams[g]));
        }

        delete[] cublasHandles;
        delete[] cusparseHandles;
        delete[] streams;
    }

    void sync() const {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

    int device(int g) const { return debug ? 0 : g; }

    int ngpus = 0;
    bool debug = false;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;
};