template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void fast_drill_strain_add_residual(const int32_t num_elements,
                                               const Vec<int32_t> geo_conn,
                                               const Vec<int32_t> vars_conn, const Vec<T> xpts,
                                               const Vec<T> vars, Vec<Data> physData, Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    // if you want to precompute some things?
    // __SHARED__ T geo_data[elems_per_block][Geo::geo_data_size];
    // __SHARED__ T basis_data[elems_per_block][Geo::geo_data_size];

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    int iquad = threadIdx.x;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // store Tmat, XdinvT for each node and element
    const int nmat = 9 * Basis::num_nodes;  // 36 for quad elements
    __SHARED__ T block_Tmat[elems_per_block][nmat];
    __SHARED__ T block_XdinvT[elems_per_block][nmat];
    __SHARED__ T block_XdinvzT[elems_per_block][nmat];

    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];

    T *shared_xpts = &block_xpts[local_elem][0];
    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];
    T *shared_res = &block_res[local_elem][0];

    // try extra block work array
    // __SHARED__ T block_work_arrays[elems_per_block * 4][21];

    // load data into block shared mem using some subset of threads
    constexpr bool unrolled = true;
    // inconsequential change here..
    if constexpr (unrolled) {
        int local_node = iquad;

        // load xpts unrolled
        const T *global_xpts = xpts.getPtr();
        const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
        int global_node = geo_elem_conn[local_node];
        shared_xpts[3 * local_node] = global_xpts[3 * global_node];
        shared_xpts[3 * local_node + 1] = global_xpts[3 * global_node];
        shared_xpts[3 * local_node + 2] = global_xpts[3 * global_node];

        // load vars unrolled
        const T *global_vars = vars.getPtr();
        const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
        int global_vnode = vars_elem_conn[local_node];
        shared_vars[6 * local_node] = global_vars[6 * global_vnode];
        shared_vars[6 * local_node + 1] = global_vars[6 * global_vnode + 1];
        shared_vars[6 * local_node + 2] = global_vars[6 * global_vnode + 2];
        shared_vars[6 * local_node + 3] = global_vars[6 * global_vnode + 3];
        shared_vars[6 * local_node + 4] = global_vars[6 * global_vnode + 4];
        shared_vars[6 * local_node + 5] = global_vars[6 * global_vnode + 5];

    } else {
        xpts.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Geo::spatial_dim,
                                    Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

        vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                    Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);
    }

    T local_res[vars_per_elem];
#pragma unroll
    for (int i = iquad; i < 24; i += 4) {
        shared_res[i] = 0.0;
        // local_res[i] = 0.0;
    }

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason)
    // apparently memset uses vectorized instructions.. and doesn't create extra registers for
    // setting this here..
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    if (active_thread) {
        // if (threadIdx.y == 0) memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    // can toggle on different strain energy terms (could also potentially launch in separate
    // kernels if need be)
    constexpr bool fast_drill_strain = false, include_drill_strain = true;

    if constexpr (fast_drill_strain) {
        ElemGroup::template _add_drill_strain_quadpt_residual_fast<Data>(
            active_thread, iquad, shared_xpts, shared_vars, shared_data, shared_vars, shared_vars,
            local_res);
    }

    // drill strain terms.. using nodal shell transforms
    if constexpr (include_drill_strain) {
        int inode = iquad;  // for now
        ElemGroup::template compute_nodal_shell_transforms<Data>(
            active_thread, inode, block_xpts[local_elem], block_data[local_elem],
            &block_Tmat[local_elem][9 * inode], &block_XdinvT[local_elem][9 * inode]);
        __syncthreads();  // each thread needs to then use the nodal transforms

        ElemGroup::template _add_drill_strain_quadpt_residual_fast<Data>(
            active_thread, iquad, shared_xpts, shared_vars, shared_data, block_Tmat[local_elem],
            block_XdinvT[local_elem], local_res);
    }

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // __shfl_down_sync(mask, val, offset)

            // Reduce within quadpt group of 4
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                shared_res[i] = val;
            }
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  &block_res[local_elem][0]);
        __syncthreads();
    }

    // will go from 3.4e-4 to 3.2e-4 with atomicAdds (what? weird.., so graph coloring may not
    // help..)
    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, &block_res[local_elem][0]);
}  // end of add_residual_gpu kernel