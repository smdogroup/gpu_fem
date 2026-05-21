#include "cuda_utils.h"
#include "utils.h"

class LightPartitioner {
    // for coarse problem of BDDC, not exposing elements directly, just node subdivisions
    LightPartitioner(MultiGPUContext *ctx_, const int num_nodes_, const int **h_local_nodes_) {
        ctx = ctx_;
        num_nodes = num_nodes;
        h_local_nodes = h_local_nodes_;
    }

    int num_nodes;
    MultiGPUContext *ctx;
    int **h_local_nodes;
};