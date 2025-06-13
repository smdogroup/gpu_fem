#pragma once

template <typename T, class ElemGroup, class Data, int32_t mat_cols_per_block = 1,
          template <typename> class Vec, class Mat>
__GLOBAL__ void drill_strain_jac(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Mat mat) {
    /* using best of drill strain resid as starting point */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    int num_elem_cols = num_elements * vars_per_elem;
    constexpr int elems_per_block = (mat_cols_per_block + vars_per_elem - 1) / vars_per_elem;

    // compute which elem and elem col we are out of global #
    int local_elem_col = threadIdx.y;
    int local_elem = local_elem_col / vars_per_elem;
    int global_elem_col_start = blockDim.x * blockIdx.x;
    int global_elem_col = global_elem_col_start + local_elem_col;
    int elem_block = (global_elem_col / Phys::vars_per_node) % 4;
    // int global_elem_start = global_elem_col_start / vars_per_elem;
    int global_elem = global_elem_col / vars_per_elem;
    int inner_col = (global_elem_col % Phys::vars_per_node);

    bool active_thread = global_elem_col < num_elem_cols;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code   

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here-1\n");
    // return;

    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("global_elem %d, elems_per_block %d, active_thread %d, num_elements %d\n", global_elem, elems_per_block, active_thread, num_elements);
    // }

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here0\n");

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    // storing 9 vs 6 here and then pulling out cols to rows doesn't affect registers
    // or performance (see old version in local2 kernel)
    T loc_Tmatn[6], loc_XdinvTn[6], loc_detXdq;
    if (active_thread) {
        for (int i = 0; i < 3; i++) {
            // t1, stored row major
            loc_Tmatn[i] = elem_Tmatn[9 * inode + 3*i];
            // t2
            loc_Tmatn[3 + i] = elem_Tmatn[9*inode + 3 * i + 1];

            // col0, stored row major
            loc_XdinvTn[i] = elem_XdinvTn[9 * inode + 3*i];
            // col1
            loc_XdinvTn[3 + i] = elem_XdinvTn[9*inode + 3 * i + 1];
        }   
        loc_detXdq = elem_detXd[iquad];

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    // suppose I only want to compute mat here, not res (for less registers, how fast can it get compared to local_res?)
    T loc_elem_col[vars_per_elem];
    memset(loc_elem_col, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here1\n");

    ElemGroup::template add_drill_strain_quadpt_jac_fast<Data>(
        active_thread, ideriv, iquad, shared_vars, shared_data, 
        loc_Tmatn, loc_XdinvTn, loc_detXdq,
        loc_elem_col);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here2\n");

    // warp reduction is way faster here..
    for (int i = 0; i < vars_per_elem; i++) {
        double val = loc_elem_col[i];

        // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
        // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
        val += __shfl_down_sync(0xffffffff, val, 1);
        val += __shfl_down_sync(0xffffffff, val, 2);

        if (iquad == 0) {
            // Only the first thread in each element writes to shared memory
            loc_elem_col[i] = val;
        }

        // need warp bacast here so I can then use local_res to add directly to global
        int lane = local_thread % 32;
        int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
        loc_elem_col[i] = __shfl_sync(0xffffffff, loc_elem_col[i], group_root);
    }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here3\n");
    // // return;

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x < 14) {
    //     printf("elem_block %d, global_elem %d, blockIdx.x %d, global_elem_col %d\n", elem_block, global_elem, blockIdx.x, global_elem_col);
    // }
    // return;

    mat.addElementMatRow(active_thread, elem_block, inner_col, global_elem, threadIdx.x, blockDim.x, Phys::vars_per_node,
                         Basis::num_nodes, vars_elem_conn, loc_elem_col);  
}

template <typename T, class ElemGroup, class Data, int32_t mat_cols_per_block = 1,
          template <typename> class Vec, class Mat>
__GLOBAL__ void drill_strain_jac_fast(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Mat mat) {
    /* improvement if drill strain is rank 1 because linear, then we  */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    int num_elem_cols = num_elements * vars_per_elem;
    constexpr int elems_per_block = (mat_cols_per_block + vars_per_elem - 1) / vars_per_elem;

    // compute which elem and elem col we are out of global #
    int local_elem_col = threadIdx.y;
    int local_elem = local_elem_col / vars_per_elem;
    int global_elem_col_start = blockDim.x * blockIdx.x;
    int global_elem_col = global_elem_col_start + local_elem_col;
    int elem_block = (global_elem_col / Phys::vars_per_node) % 4;
    // int global_elem_start = global_elem_col_start / vars_per_elem;
    int global_elem = global_elem_col / vars_per_elem;
    int inner_col = (global_elem_col % Phys::vars_per_node);

    bool active_thread = global_elem_col < num_elem_cols;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code   

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here-1\n");
    // return;

    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("global_elem %d, elems_per_block %d, active_thread %d, num_elements %d\n", global_elem, elems_per_block, active_thread, num_elements);
    // }

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here0\n");

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    // storing 9 vs 6 here and then pulling out cols to rows doesn't affect registers
    // or performance (see old version in local2 kernel)
    T loc_Tmatn[6], loc_XdinvTn[6], loc_detXdq;
    if (active_thread) {
        for (int i = 0; i < 3; i++) {
            // t1, stored row major
            loc_Tmatn[i] = elem_Tmatn[9 * inode + 3*i];
            // t2
            loc_Tmatn[3 + i] = elem_Tmatn[9*inode + 3 * i + 1];

            // col0, stored row major
            loc_XdinvTn[i] = elem_XdinvTn[9 * inode + 3*i];
            // col1
            loc_XdinvTn[3 + i] = elem_XdinvTn[9*inode + 3 * i + 1];
        }   
        loc_detXdq = elem_detXd[iquad];

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    // suppose I only want to compute mat here, not res (for less registers, how fast can it get compared to local_res?)
    T loc_elem_col[vars_per_elem];
    memset(loc_elem_col, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here1\n");

    ElemGroup::template add_drill_strain_quadpt_jac_fast<Data>(
        active_thread, ideriv, iquad, shared_vars, shared_data, 
        loc_Tmatn, loc_XdinvTn, loc_detXdq,
        loc_elem_col);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here2\n");

    // warp reduction is way faster here..
    for (int i = 0; i < vars_per_elem; i++) {
        double val = loc_elem_col[i];

        // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
        // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
        val += __shfl_down_sync(0xffffffff, val, 1);
        val += __shfl_down_sync(0xffffffff, val, 2);

        if (iquad == 0) {
            // Only the first thread in each element writes to shared memory
            loc_elem_col[i] = val;
        }

        // need warp bacast here so I can then use local_res to add directly to global
        int lane = local_thread % 32;
        int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
        loc_elem_col[i] = __shfl_sync(0xffffffff, loc_elem_col[i], group_root);
    }

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) printf("here3\n");
    // // return;

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x < 14) {
    //     printf("elem_block %d, global_elem %d, blockIdx.x %d, global_elem_col %d\n", elem_block, global_elem, blockIdx.x, global_elem_col);
    // }
    // return;

    mat.addElementMatRow(active_thread, elem_block, inner_col, global_elem, threadIdx.x, blockDim.x, Phys::vars_per_node,
                         Basis::num_nodes, vars_elem_conn, loc_elem_col);  
}
