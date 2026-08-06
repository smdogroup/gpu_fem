#pragma once

#include "linalg/bsr_mat.h"
#include "linalg/vec.h"

template <typename T_, class Assembler_>
class FakeAssembler : public Assembler_ {
   public:
    using Director = typename Assembler_::Director;
    using Basis = typename Assembler_::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename Assembler_::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    static constexpr int32_t vars_per_node = Phys::vars_per_node;

    /* need fake assembler for AMG since coarse grid problems don't use mesh */
   public:
    using T = T_;
    FakeAssembler() = default;  // for pointers
    FakeAssembler(BsrData bsr_data_, int nnodes_, int num_elements_ = 1, int num_components_ = 1) {
        bsr_data = bsr_data_;
        block_dim = bsr_data.block_dim;
        nnodes = nnodes_;
        num_elements = num_elements_;
        num_components = num_components_;
        N = block_dim * nnodes;

        // create fake data for d_xpts, d_vars, comp_data, elem_components
        // blank for now (so hope that isn't needed, definitely don't need xpts, vars for coarse
        // problem assembly in most cases for AMG and DomDec)
        elem_components = DeviceVec<int>(num_elements);
        comp_data = DeviceVec<Data>(num_components);
        d_xpts = DeviceVec<T>(3 * nnodes);
        d_vars = DeviceVec<T>(vars_per_node * nnodes);
        d_bcs = DeviceVec<int>(0);  // should be unused
    }

    // do nothing calls
    void set_variables(DeviceVec<T> vars) {}
    void add_jacobian_fast(BsrMat<DeviceVec<T>> kmat) {}
    void apply_bcs(BsrMat<DeviceVec<T>> kmat) {}
    void free() {}
    int get_num_vars() { return N; }
    BsrData getBsrData() { return bsr_data; }
    int get_num_elements() { return num_elements; }
    int get_num_nodes() { return nnodes; }
    int get_num_components() { return num_components; }
    DeviceVec<int> getConn() { return elem_components; }
    DeviceVec<T> getXpts() { return d_xpts; }
    DeviceVec<T> getVars() { return d_vars; }
    DeviceVec<int> getElemComponents() { return elem_components; }
    DeviceVec<Data> getCompData() { return comp_data; }
    DeviceVec<int> getBCs() { return d_bcs; }
    DeviceVec<T> createVarsVec() {
        auto vec = DeviceVec<T>(N);
        return vec;
    }

   private:
    int block_dim, nnodes, N, num_elements, num_components;
    BsrData bsr_data;
    DeviceVec<int> elem_components, d_bcs;
    DeviceVec<Data> comp_data;
    DeviceVec<T> d_xpts, d_vars;
};