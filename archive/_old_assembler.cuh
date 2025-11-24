template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void k_compute_sectional_loads(const int32_t num_elements,
                                               const Vec<int32_t> elem_comp,
                                               const Vec<int32_t> geo_conn,
                                               const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                               const Vec<T> vars, Vec<Data> physData, const int32_t *iperm, 
                                               Vec<T> loads, Vec<int> load_cts) {
    // compute sectional resultants like in plane loads, moments
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;
    const int nodes_per_elem = Basis::num_nodes;
    const int num_quad_pts = Quadrature::num_quad_pts;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_loads[elems_per_block][vars_per_node];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                            Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyValuesToShared(active_thread, threadIdx.y, blockDim.y, Phys::vars_per_node,
                            Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        memset(&block_loads[local_elem][0], 0.0, vars_per_node * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.x * blockIdx.x;
            int component_ind = elem_comp[global_elem_thread];
            block_data[local_thread] = _phys_data[component_ind];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.y;

    T quadpt_loads[vars_per_node];
    memset(quadpt_loads, 0.0, sizeof(T) * vars_per_node);

    ElemGroup::template compute_element_quadpt_sectional_loads<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], quadpt_loads);

    // use nodal averaging of the stresses here, alternative is max among quadpts
    // this should make stress field smoother (can be slightly unconservative sometimes)
    // but prevents preliminary design optimization from responding to non-physical stress
    // concentrations from kinks in a structure (artifact of shell elements)

    for (int idof = 0; idof < vars_per_node; idof++) {
        atomicAdd(&block_loads[local_elem][0], 1.0 / num_quad_pts * quadpt_loads[idof]);
    }

    // now add from shared to global (same averaged vars_per_node x 1 stresses added to each node in
    // element) stress cts later used to complete the averaged at nodal level, that is nodal stress
    // = sum stresses / #element stresses added to node don't use perm here since this goes to
    // visualization and not to solve

    // parallelize across quadpt dimension of block here
    for (int i = threadIdx.y; i < vars_per_elem; i += blockIdx.y) {
        int local_inode = i / vars_per_node;
        int idof = i % vars_per_node;
        int global_inode = vars_elem_conn[local_inode];
        // note we don't use perm here since this goes to visualization not solve

        atomicAdd(&loads[vars_per_node * global_inode + idof], block_loads[local_elem][idof]);
    }

    // also add up load counts
    for (int inode = threadIdx.y; inode < nodes_per_elem; inode += blockIdx.y) {
        int global_inode = vars_elem_conn[inode];
        atomicAdd(&load_cts[global_inode], 1.0);
    }

}  // end of compute_sectional_loads_kernel


template <typename T, class ElemGroup, template <typename> class Vec>
__GLOBAL__ void k_normalize_states(Vec<T> states, Vec<int> state_cts) {
    // normalize states : stresses or strains averaged to nodal level
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;

    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    const int vars_per_node = Phys::vars_per_node;

    int global_node = blockIdx.x;
    T *loc_state = states[vars_per_node * global_node];
    int nodal_state_ct = state_cts[global_node];

    for (int idof = threadIdx.x; idof < vars_per_node; idof += blockDim.x) {
        loc_state[idof] /= nodal_state_ct;
    }

}  // end of normalize_stresses_kernel