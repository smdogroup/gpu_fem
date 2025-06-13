#pragma once

#include "drill_strain_jac.cuh"

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_shared3(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    /* this one starts from a fast baseline kernel with drill_strain_residual_local2 that ran with on the RT 3060 Ti
       GPU at 2.7e-4 sec. The change is that I'm not not putting block_res in shared mem. No reason since I'm doing warp reduction
       and then only one add into shared and then global memory. No reason to take up extra space for that. 
       TODO : is still to change data to be a struct of arrays for faster load for that maybe. Could also try taking Data out of shared memory. 
       But I think that one kind of makes sense. */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];

    __SHARED__ T block_Tmatn[elems_per_block][24];
    __SHARED__ T block_XdinvTn[elems_per_block][24];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    
    T *shared_Tmatn = &block_Tmatn[local_elem][6 * inode];
    T *shared_XdinvTn = &block_XdinvTn[local_elem][6 * inode];

    // storing 9 vs 6 here and then pulling out cols to rows doesn't affect registers
    // or performance (see old version in local2 kernel)
    T loc_Tmatn[6], loc_XdinvTn[6], loc_detXdq;
    if (active_thread) {

        constexpr bool load_global = true;

        if constexpr (load_global) {
            for (int i = 0; i < 3; i++) {
                // t1, stored row major
                shared_Tmatn[i] = elem_Tmatn[9 * inode + 3*i];
                // t2
                shared_Tmatn[3 + i] = elem_Tmatn[9*inode + 3 * i + 1];

                // col0, stored row major
                shared_XdinvTn[i] = elem_XdinvTn[9 * inode + 3*i];
                // col1
                shared_XdinvTn[3 + i] = elem_XdinvTn[9*inode + 3 * i + 1];
            }   
            loc_detXdq = elem_detXd[iquad];
        } else {
            for (int i = 0; i < 6; i++) {
                shared_Tmatn[i] = 0.0;
                shared_XdinvTn[i] = 0.0;
            }
            loc_detXdq = 0.0;
        }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T local_res[24];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here1\n");
    // }

    ElemGroup::template add_drill_strain_quadpt_residual_fast<Data>(
        active_thread, iquad, shared_vars, shared_data, 
        shared_Tmatn, shared_XdinvTn, loc_detXdq,
        local_res);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here2\n");
    // }

    // warp reduction is way faster here..
    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
            // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                local_res[i] = val;
            }

            // need warp bacast here so I can then use local_res to add directly to global
            int lane = local_thread % 32;
            int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
            local_res[i] = __shfl_sync(0xffffffff, local_res[i], group_root);
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  local_res);
        __syncthreads();
    }

    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, local_res);  
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_local3_simple(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    /* this one starts from a fast baseline kernel with drill_strain_residual_local2 that ran with on the RT 3060 Ti
       GPU at 2.7e-4 sec. The change is that I'm not not putting block_res in shared mem. No reason since I'm doing warp reduction
       and then only one add into shared and then global memory. No reason to take up extra space for that. 
       TODO : is still to change data to be a struct of arrays for faster load for that maybe. Could also try taking Data out of shared memory. 
       But I think that one kind of makes sense. */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    // storing 9 vs 6 here and then pulling out cols to rows doesn't affect registers
    // or performance (see old version in local2 kernel)
    T loc_Tmatn[9], loc_XdinvTn[9], loc_detXdq;
    if (active_thread) {

        #pragma unroll
            for (int i = 0; i < 9; i++) {
                loc_Tmatn[i] = elem_Tmatn[9 * inode + i];
                loc_XdinvTn[i] = elem_XdinvTn[9 * inode + i];
            }
            loc_detXdq = elem_detXd[iquad];
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T local_res[24];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here1\n");
    // }

    constexpr bool simple = true; // this one uses simple mat operations, no scalar triple products
    ElemGroup::template add_drill_strain_quadpt_residual_fast<Data, simple>(
        active_thread, iquad, shared_vars, shared_data, 
        loc_Tmatn, loc_XdinvTn, loc_detXdq,
        local_res);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here2\n");
    // }

    // warp reduction is way faster here..
    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
            // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                local_res[i] = val;
            }

            // need warp bacast here so I can then use local_res to add directly to global
            int lane = local_thread % 32;
            int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
            local_res[i] = __shfl_sync(0xffffffff, local_res[i], group_root);
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  local_res);
        __syncthreads();
    }

    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, local_res);  
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_local3(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    /* this one starts from a fast baseline kernel with drill_strain_residual_local2 that ran with on the RT 3060 Ti
       GPU at 2.7e-4 sec. The change is that I'm not not putting block_res in shared mem. No reason since I'm doing warp reduction
       and then only one add into shared and then global memory. No reason to take up extra space for that. 
       TODO : is still to change data to be a struct of arrays for faster load for that maybe. Could also try taking Data out of shared memory. 
       But I think that one kind of makes sense. */

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

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

        constexpr bool load_global = true;

        if constexpr (load_global) {
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
        } else {
            for (int i = 0; i < 6; i++) {
                loc_Tmatn[i] = 0.0;
                loc_XdinvTn[i] = 0.0;
            }
            loc_detXdq = 0.0;
        }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T local_res[24];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here1\n");
    // }

    ElemGroup::template add_drill_strain_quadpt_residual_fast<Data>(
        active_thread, iquad, shared_vars, shared_data, 
        loc_Tmatn, loc_XdinvTn, loc_detXdq,
        local_res);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here2\n");
    // }

    // warp reduction is way faster here..
    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
            // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                local_res[i] = val;
            }

            // need warp bacast here so I can then use local_res to add directly to global
            int lane = local_thread % 32;
            int group_root = lane & ~0x3; // finds starting line e.g. 0,4,8, etc.
            local_res[i] = __shfl_sync(0xffffffff, local_res[i], group_root);
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  local_res);
        __syncthreads();
    }

    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, local_res);  
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_local2(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    // newest kernel that uses pre-computed Tmat, XdinvT at nodal levels into local not shared memory (for better occupancy hopefully, depends
    // on whether you are shared mem or register memory bound
    // launches with (4,32,1) so 4 threads per element used to compute nodal data and interp to quadpt with warp operations
    // then reverse warp sync and bcast to hopefully get 4x less compute (separate from even the fact we are not computing Tmat, XdinvT which were expensive computes before)

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad = threadIdx.x;
    int inode = iquad; // do iquad and inode parallelism at different parts of the code

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    // TODO : switch from array of structs (AOS) to struct of arrays (SOA) for faster and more coalesced memory transfer of physData

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];
    T *shared_res = &block_res[local_elem][0];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    T loc_Tmatn[9], loc_XdinvTn[9], loc_detXdq;
    if (active_thread) {

        constexpr bool load_global = true;

        if constexpr (load_global) {
            for (int i = 0; i < 9; i++) {
                loc_Tmatn[i] = elem_Tmatn[9 * inode + i];
                loc_XdinvTn[i] = elem_XdinvTn[9 * inode + i];
            }   
            loc_detXdq = elem_detXd[iquad];
        } else {
            for (int i = 0; i < 9; i++) {
                loc_Tmatn[i] = 0.0;
                loc_XdinvTn[i] = 0.0;
            }
            loc_detXdq = 0.0;
        }

    #pragma unroll
        for (int i = iquad; i < 24; i += 4) {
            shared_res[i] = 0.0;
            // local_res[i] = 0.0;
        }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T local_res[24];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here1\n");
    // }

    ElemGroup::template add_drill_strain_quadpt_residual_fast<Data>(
        active_thread, iquad, shared_vars, shared_data, 
        loc_Tmatn, loc_XdinvTn, loc_detXdq,
        local_res);

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("here2\n");
    // }

    // warp reduction is way faster here..
    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt group of 4 in case of QuadBasis and QuadElement
            // shuffles down by 2, 1 to reduce every 4 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);

            if (iquad == 0 && inode == 0) {
                // Only the first thread in each element writes to shared memory
                shared_res[i] = val;
            }
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  shared_res);
        __syncthreads();
    }

    // don't have to add from root of every 4th thread here since was added into shared memory..
    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, shared_res);  
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_shared(const int32_t num_elements,
                                               const Vec<int32_t> geo_conn,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad_node = threadIdx.x;
    // inode is inner most dimension of block
    int inode = iquad_node % Basis::num_nodes;
    int iquad = iquad_node / Basis::num_nodes;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];
    
    // TODO : these guys are pre-computed once for each design (not on re-assembly necessarily)
    // and stored in the assembler (the global vecs)
    const int nmat = 9 * Basis::num_nodes;  // 36 for quad elements
    __SHARED__ T block_Tmatn[elems_per_block][nmat];
    __SHARED__ T block_XdinvTn[elems_per_block][nmat];
    __SHARED__ T block_detXd[elems_per_block][Basis::num_nodes];

    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];
    T *shared_Tmatn = &block_Tmatn[local_elem][0];
    T *shared_XdinvTn = &block_XdinvTn[local_elem][0];
    T *shared_detXdq = &block_detXd[local_elem][0];
    T *shared_res = &block_res[local_elem][0];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);
    Tmatn.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, 9,
                                Basis::num_nodes, geo_elem_conn, shared_Tmatn);
    XdinvTn.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, 9,
                                Basis::num_nodes, geo_elem_conn, shared_XdinvTn);
    // load detXdq from global elem*4 into local_elem*4
    const T *_detXd = detXdq.getPtr();
    const T *elem_detXd = &_detXd[global_elem * Quadrature::num_quad_pts];
    for (int i = threadIdx.x; i < Quadrature::num_quad_pts; i += blockDim.x) {
        shared_detXdq[i] = elem_detXd[i];
    }

    T local_res[vars_per_elem];
#pragma unroll
    for (int i = iquad; i < 24; i += 4) {
        shared_res[i] = 0.0;
        // local_res[i] = 0.0;
    }

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    if (active_thread) {
        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }
    __syncthreads();

    // TODO : would need to write one to handle inode and iquad parallelism,
    // but my new kernel above handles etn => etq with only 4 threads per elem, not like 16 here
    // so as mem access so slow in this kernel, not bothering to make this compute step here

    // ElemGroup::template add_drill_strain_quadpt_residual_fast<Data>(
    //     active_thread, inode, iquad, shared_vars, shared_data, 
    //     shared_Tmatn, shared_XdinvTn, shared_detXdq[iquad],
    //     local_res);

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt & node group of 16 in case of QuadBasis and QuadElement
            // shuffles down by 8, 4, 2, 1 to reduce every 16 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 8);

            if (iquad == 0) {
                // Only the first thread in each element writes to shared memory
                shared_res[i] = val;
            }
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  shared_res);
        __syncthreads();
    }

    // TODO : graph coloring to reduce cost of atomicAdd? graph coloring ensures no two elements share a node in the same thread block
    // will go from 3.4e-4 to 3.2e-4 with atomicAdds (what? weird.., so graph coloring may not
    // help..)
    res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                   Basis::num_nodes, vars_elem_conn, shared_res);
}  // end of add_residual_gpu kernel

template <typename T, class ElemGroup, class Data, int32_t elems_per_block = 1,
          template <typename> class Vec>
__GLOBAL__ void drill_strain_residual_local(const int32_t num_elements,
                                               const Vec<int32_t> vars_conn,
                                               const Vec<T> vars, Vec<Data> physData, 
                                               const Vec<T> Tmatn, const Vec<T> XdinvTn, 
                                               const Vec<T> detXdq,
                                               Vec<T> res) {
    // note in the above : CPU code passes Vec<> objects by reference
    // GPU kernel code cannot do so for complex objects otherwise weird behavior
    // occurs

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename ElemGroup::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + blockDim.x * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    int local_thread =
        (blockDim.x * blockDim.y) * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    
    int iquad_node = threadIdx.x;
    // inode is inner most dimension of block
    int inode = iquad_node % Basis::num_nodes;
    int iquad = iquad_node / Basis::num_nodes;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;

    const int32_t *_vars_conn = vars_conn.getPtr();
    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    const Data *_phys_data = physData.getPtr();

    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_res[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    T *shared_vars = &block_vars[local_elem][0];
    Data &shared_data = block_data[local_elem];
    T *shared_res = &block_res[local_elem][0];

    vars.copyElemValuesToShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, shared_vars);

    // load Tmatn, XdinvTn, detXdq into local memory (diff for nodal and quadpt, so technically some overlap)
    const T *_Tmatn = Tmatn.getPtr();
    const T *elem_Tmatn = &_Tmatn[36 * global_elem];
    const T *_XdinvTn = XdinvTn.getPtr();
    const T *elem_XdinvTn = &_XdinvTn[36 * global_elem];
    const T *_detXdq = detXdq.getPtr();
    const T *elem_detXd = &_detXdq[global_elem * Quadrature::num_quad_pts];
    T loc_Tmatn[9], loc_XdinvTn[9], loc_detXdq;
    if (active_thread) {

        constexpr bool load_global = false;

        if constexpr (load_global) {
            for (int i = 0; i < 9; i++) {
                loc_Tmatn[i] = elem_Tmatn[9 * inode + i];
                loc_XdinvTn[i] = elem_XdinvTn[9 * inode + i];
            }   
            loc_detXdq = elem_detXd[iquad];
        } else {
            for (int i = 0; i < 9; i++) {
                loc_Tmatn[i] = 0.0;
                loc_XdinvTn[i] = 0.0;
            }
            loc_detXdq = 0.0;
        }

    #pragma unroll
        for (int i = iquad; i < 24; i += 4) {
            shared_res[i] = 0.0;
            // local_res[i] = 0.0;
        }

        if (local_thread < elems_per_block) {
            int global_elem_thread = local_thread + blockDim.y * blockIdx.y;
            block_data[local_thread] = _phys_data[global_elem_thread];
        }
    }

    // try removing syncthreads
    __syncthreads();

    // much faster to do memset locally.. (shaves off 2e-4 sec for some reason, uses vectorized instructions)
    T local_res[24];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // TODO : would need to write one to handle inode and iquad parallelism,
    // but my new kernel above handles etn => etq with only 4 threads per elem, not like 16 here
    // so as mem access so slow in this kernel, not bothering to make this compute step here

    // ElemGroup::template add_drill_strain_quadpt_residual_fast<Data>(
    //     active_thread, inode, iquad, shared_vars, shared_data, 
    //     loc_Tmatn, loc_XdinvTn, loc_detXdq,
    //     local_res);

    // if (global_elem == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("ran custom shell elem
    // kernel\n");

    // warp reduction is way faster here..
    constexpr bool use_warp_reduction = true;

    // using warp reduction reduces runtime from 8e-4 to 3e-4 sec!
    if constexpr (use_warp_reduction) {
        // warp reduction across quadpts (need all 4 quadpts in the same warp)
        // #pragma unroll // this does nothing
        for (int i = 0; i < vars_per_elem; i++) {
            double val = local_res[i];

            // Reduce within quadpt & node group of 16 in case of QuadBasis and QuadElement
            // shuffles down by 8, 4, 2, 1 to reduce every 16 threads, // __shfl_down_sync(mask, val, offset)
            val += __shfl_down_sync(0xffffffff, val, 1);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 8);

            if (iquad == 0 && inode == 0) {
                // Only the first thread in each element writes to shared memory
                shared_res[i] = val;
            }
        }
    } else {
        // slower by 3e-4 to 8e-4
        Vec<T>::copyLocalToShared(active_thread, 1.0, vars_per_elem, &local_res[0],
                                  shared_res);
        __syncthreads();
    }

    // it appears about half the time is in local to global write, that takes more time with more threads??

    constexpr int to_global_option = 1;

    T *res_data = res.getPtr();
    int dof_per_node = 6;
    if constexpr (to_global_option == 1) { // atomics
        // atomics on all threads for same element, about 5e-4 sec
        if (active_thread) {
            for (int idof = threadIdx.x; idof < 24; idof += blockDim.x) {
                int local_inode = idof / dof_per_node;
                int _global_inode = vars_elem_conn[local_inode];
                int iglobal = _global_inode * dof_per_node + (idof % dof_per_node);
                atomicAdd(&res_data[iglobal], shared_res[idof]);
            }
        }
        // res.addElementValuesFromShared(active_thread, threadIdx.x, blockDim.x, Phys::vars_per_node,
        //                            Basis::num_nodes, vars_elem_conn, shared_res);
    } else if (to_global_option == 2) {
        // unsafe adds for all threads in element, could use graph coloring here
        // about 5e-4 sec, hmm why same speed as atomicAdd? (just like 1e-6 faster than atomics, so not really change)
        if (active_thread) {
            for (int idof = threadIdx.x; idof < 24; idof += blockDim.x) {
                int local_inode = idof / dof_per_node;
                int _global_inode = vars_elem_conn[local_inode];
                int iglobal = _global_inode * dof_per_node + (idof % dof_per_node);

                res_data[iglobal] += shared_res[idof];
            }
        }
    } else if (to_global_option == 3) {
        // use fewer active threads for the atomics/unsafe, see if that helps recover 4x speedup
        // unsuccessful about 5e-4 sec here still
        if (active_thread && threadIdx.x < 4) {
            for (int idof = threadIdx.x; idof < 24; idof += 4) {
                int local_inode = idof / dof_per_node;
                int _global_inode = vars_elem_conn[local_inode];
                int iglobal = _global_inode * dof_per_node + (idof % dof_per_node);

                res_data[iglobal] += shared_res[idof];
            }
        }
    } else if (to_global_option == 4) {
        // total time reduces from 9.8e-4 to 5.2e-4
        if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0) {
            printf("res:");
            printVec<T>(24, local_res);
        }
    } else if (to_global_option == 5) {
        // coalesced memory writes (all 32 threads write to consecutive place in memory) 
        // would be ideal.. however this doesn't work well in FEM when each node corresponds to different global nodes in the same element
        // so the 24 dof per element are scattered all over the place
        // pretend here I have a memory buffer and that somehow I can write from the memory buffer to the global res in a second kernel with better
        // memory coalescing.. (maybe not though.. lol)
        // unsuccessful, this takes about the same amount of time 5e-4 sec also.. damn it
        if (active_thread) {
            int all_dof = 24 * elems_per_block;
            int stride = blockDim.x * blockDim.y;

            // pretend just one atomic add per thread?
            res_data[local_thread] += shared_res[threadIdx.x];
            // atomicAdd(&res_data[local_thread], shared_res[threadIdx.x]);
        }
    }
    
    
}  // end of add_residual_gpu kernel