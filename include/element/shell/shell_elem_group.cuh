
#pragma once

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void shell_elem_add_residual_gpu(const int32_t num_elements, const Vec<int32_t> geo_conn,
                                 const Vec<int32_t> vars_conn, const Vec<T> xpts, const Vec<T> vars,
                                 Vec<Data> physData, Vec<T> res) {
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
    const int nmat = 9 * Basis::num_nodes; // 36 for quad elements
    __SHARED__ T block_Tmat[elems_per_block][nmat];
    __SHARED__ T block_XdinvT[elems_per_block][nmat];
    __SHARED__ T block_XdinvzT[elems_per_block][nmat];

    // try extra block work array
    // __SHARED__ T block_work_arrays[elems_per_block * 4][21];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, &block_vars[local_elem][0]);

    if (active_thread) {
        if (threadIdx.y == 0) memset(&block_res[local_elem][0], 0.0, vars_per_elem * sizeof(T));
        // for (int i = 0; i < vars_per_elem; i++) {
        //     block_res[local_elem][0] = 0.0;
        // }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // printf("<<<res GPU kernel>>>\n");

    int iquad = threadIdx.x;

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // can toggle on different strain energy terms (could also potentially launch in separate kernels if need be)
    constexpr bool fast_drill_strain = false, 
                    include_drill_strain = true, 
                   include_tying_strain = false,
                   include_bending_strain = false;

    if constexpr (fast_drill_strain) {
        ElemGroup::template _add_drill_strain_quadpt_residual_fast<Data>(active_thread, iquad, block_xpts[local_elem],
            block_vars[local_elem], block_data[local_elem], block_vars[local_elem], block_vars[local_elem], local_res);
    }

    // drill strain terms.. using nodal shell transforms
    if constexpr (include_drill_strain) {
        int inode = iquad; // for now
        ElemGroup::template compute_nodal_shell_transforms<Data>(active_thread, inode, 
            block_xpts[local_elem], block_data[local_elem], &block_Tmat[local_elem][9*inode], 
            &block_XdinvT[local_elem][9*inode]);
        __syncthreads(); // each thread needs to then use the nodal transforms

        ElemGroup::template _add_drill_strain_quadpt_residual_fast<Data>(active_thread, iquad, block_xpts[local_elem],
            block_vars[local_elem], block_data[local_elem], &block_Tmat[local_elem][0], 
            &block_XdinvT[local_elem][0], local_res);
    }

    if constexpr (include_tying_strain) {
        // only need quadpt Tmat, XdinvT for tying strains
        ElemGroup::template compute_quadpt_shell_transforms<Data>(active_thread, iquad, 
            block_xpts[local_elem], block_data[local_elem], &block_Tmat[local_elem][9*iquad], 
            &block_XdinvT[local_elem][9*iquad], &block_XdinvzT[local_elem][9*iquad]);

        ElemGroup::template _add_tying_strain_quadpt_residual_fast<Data>(active_thread, iquad, block_xpts[local_elem],
            block_vars[local_elem], block_data[local_elem], 
            &block_XdinvT[local_elem][9*iquad], local_res);
    }

    if constexpr (include_bending_strain) {
        // only need quadpt Tmat, XdinvT for bending strains
        if constexpr (!include_tying_strain) {
            // only need to recompute if not done tying strains
            ElemGroup::template compute_quadpt_shell_transforms<Data>(active_thread, iquad, 
                block_xpts[local_elem], block_data[local_elem], &block_Tmat[local_elem][9*iquad], 
                &block_XdinvT[local_elem][9*iquad], &block_XdinvzT[local_elem][9*iquad]);
        }

        ElemGroup::template _add_bending_strain_quadpt_residual_fast<Data>(active_thread, iquad, block_xpts[local_elem],
            block_vars[local_elem], block_data[local_elem], &block_Tmat[local_elem][9*iquad], 
            &block_XdinvT[local_elem][9*iquad], &block_XdinvzT[local_elem][9*iquad], local_res);
    }

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem kernel\n");

    // warp reduction across quadpts (need all 4 quadpts in the same warp)
    for (int i = 0; i < vars_per_elem; i++) {
        double val = local_res[i];

        // __shfl_down_sync(mask, val, offset)

        // Reduce within quadpt group of 4
        val += __shfl_down_sync(0xffffffff, val, 1);
        val += __shfl_down_sync(0xffffffff, val, 2);

        if (iquad == 0) {
            // Only the first thread in each element writes to shared memory
            block_res[local_elem][i] = val;
        }
    }

    // Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
    //                           &block_res[local_elem][0]);
    // __syncthreads();

    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn,
                                   &block_res[local_elem][0]);
}  // end of add_residual_gpu kernel