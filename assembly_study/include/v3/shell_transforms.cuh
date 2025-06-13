template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_shell_nodal_transforms(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                               const Vec<T> xpts, Vec<Data> physData, 
                                               Vec<T> Tmatn, Vec<T> XdinvTn) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = blockDim.x * blockIdx.x + local_elem;
    bool active_thread = global_elem < num_elements;

    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][Basis::num_nodes * 3];
    __SHARED__ Data block_data[elems_per_block];
    // don't need shared block detXdq, Tmatq, XdinvTq since no reduction, just straight to global

    // load from global to shared
    T *shared_xpts = &block_xpts[local_elem][0];
    Data &shared_data = block_data[local_elem];

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                                Basis::num_nodes, geo_elem_conn, shared_xpts);

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T loc_Tmatn[36], loc_XdinvTn[36];
    memset(loc_Tmatn, 0.0, sizeof(T) * 36);
    memset(loc_XdinvTn, 0.0, sizeof(T) * 36);

    if (active_thread) {
        block_data[local_elem] = _phys_data[global_elem];
    }
    __syncthreads();

    int inode = threadIdx.y;
    ElemGroup::template compute_nodal_transforms<Data>(
        active_thread, inode, shared_xpts, shared_data,
        &loc_Tmatn[9 * inode], &loc_XdinvTn[9 * inode]
    );

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    // add from local to global, no atomics or warp reduction needed
    if (active_thread) {
        for (int i = threadIdx.y; i < 36; i += blockDim.y) {
            Tmatn[36 * global_elem + i] = loc_Tmatn[i];
            XdinvTn[36 * global_elem + i] = loc_XdinvTn[i];
        }
    }
}  // end of compute_shell_quadpt_transforms kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void compute_shell_quadpt_transforms(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                               const Vec<T> xpts, Vec<Data> physData, 
                                               Vec<T> detXdq) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.x;
    int global_elem = blockDim.x * blockIdx.x + local_elem;
    bool active_thread = global_elem < num_elements;

    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][Basis::num_nodes * 3];
    __SHARED__ Data block_data[elems_per_block];
    // don't need shared block detXdq, Tmatq, XdinvTq since no reduction, just straight to global

    // load from global to shared
    T *shared_xpts = &block_xpts[local_elem][0];
    Data &shared_data = block_data[local_elem];

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.y, blockDim.y, Geo::spatial_dim,
                                Basis::num_nodes, geo_elem_conn, shared_xpts);

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T loc_detXdq[4];
    memset(loc_detXdq, 0.0, sizeof(T) * 4);

    if (active_thread) {
        block_data[local_elem] = _phys_data[global_elem];
    }
    __syncthreads();

    int iquad = threadIdx.y;
    ElemGroup::template compute_quadpt_transforms<Data>(
        active_thread, iquad, shared_xpts, 
        &loc_detXdq[iquad] // TODO : add Tmatq, XdinvTq maybe
    );

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    // add from local to global, no atomics or warp reduction needed
    if (active_thread) {
        for (int i = threadIdx.y; i < 4; i += blockDim.y) {
            detXdq[4 * global_elem + i] = loc_detXdq[i];
        }
    }
}  // end of compute_shell_quadpt_transforms kernel