#pragma once

#include "linalg/vec.h"
#include "linalg/bsr_mat.h"


template <typename T_>
class FakeAssembler {
    /* need fake assembler for AMG since coarse grid problems don't use mesh */
public:
    using T = T_;
    FakeAssembler() = default;  // for pointers
    FakeAssembler(BsrData bsr_data_, int nnodes_) {
        bsr_data = bsr_data_;
        block_dim = bsr_data.block_dim;
        nnodes = nnodes_;
        N = block_dim * nnodes;
    }

    // do nothing calls
    void set_variables(DeviceVec<T> vars) {}
    void add_jacobian_fast(BsrMat<DeviceVec<T>> kmat) {}
    void apply_bcs(BsrMat<DeviceVec<T>> kmat) {}
    void free() {}
    int get_num_vars() { return N; }
    BsrData getBsrData() { return bsr_data; }

private:
    int block_dim, nnodes, N;
    BsrData bsr_data;
};