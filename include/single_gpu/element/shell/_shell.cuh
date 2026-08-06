#pragma once
#include "cuda_utils.h"
#include "strains/basic_utils.h"
#include "strains/strain_types.h"

// add_jacobian fast kernel
// -------------------
template <typename T, int elems_per_block, class ElemGroup, class Data, template <typename> class Vec, class Mat>
__GLOBAL__ static void k_add_jacobian_fast(int32_t vars_num_nodes, int32_t num_elements, int cols_per_elem, Vec<int32_t> elem_comp, 
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> compData, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    int local_elem_col = threadIdx.y + cols_per_elem * threadIdx.z;
    int cols_per_block = elems_per_block * cols_per_elem;
    int global_elem_col = local_elem_col + blockIdx.x * cols_per_block;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    int n_elem_cols = num_elements * vars_per_elem;
    bool active_thread = global_elem_col < n_elem_cols;
    if (!active_thread) return;
    int global_elem = global_elem_col / vars_per_elem;
    int tid_xy = blockDim.x * threadIdx.y + threadIdx.x;
    int nthreads_xy = blockDim.x * blockDim.y;
    int tid = nthreads_xy * threadIdx.z + tid_xy;
    int local_elem = threadIdx.z;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    if (tid_xy == 0) {
        memset(&block_xpts[0][0], 0.0, sizeof(T) * nxpts_per_elem);
    }
    __syncthreads();

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);
    __syncthreads();

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, block_vars[local_elem]);
    __syncthreads();
    

    if (tid_xy == 0) {
        int icomp = elem_comp[global_elem];
        block_data[local_elem] = _comp_data[icomp];
    }
    __syncthreads();

    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    constexpr bool fast_method = true;
    // constexpr bool fast_method = false;

    // OLD slower calls
    if constexpr (!fast_method) {
        T local_res[vars_per_elem];
        memset(local_res, 0.0, sizeof(T) * vars_per_elem);

        // used here for verification + 
        constexpr STRAIN strain = ALL;
        // constexpr STRAIN strain = DRILL;

        ElemGroup::template add_element_quadpt_jacobian_col<Data, strain>(
            active_thread, iquad, ideriv, block_xpts[local_elem], block_vars[local_elem],
            block_data[local_elem], local_res, local_mat_col);
        __syncthreads();
    }

    // NEW faster call
    if constexpr (fast_method) {
        // some prelim computations with xpts and quadrature point
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T fn[nxpts_per_elem];
        ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);
        T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
        T scale = detXd * weight; // scale for energy derivatives
        __syncthreads();
        T XdinvT[9], Tmat[9], XdinvzT[9];
        computeShellRotations<T, Basis, Data>(pt, block_data[local_elem].refAxis, block_xpts[local_elem], fn, Tmat, XdinvT, XdinvzT);
        __syncthreads();

        T p_vars[vars_per_elem];
        memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
        p_vars[ideriv] = 1.0;

        // // compute drill strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, DRILL>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], p_vars, local_mat_col);
        __syncthreads();

        // // compute bending strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, BENDING>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], p_vars, local_mat_col);
        __syncthreads();

        // // // compute tying strains
        ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, TYING>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], p_vars, local_mat_col);
        __syncthreads();
    }

    // if (threadIdx.x == 0 && blockIdx.x == 4) {
    //     printf("local_mat_col of elem %d, elemCol %d: ");
    //     printVec<T>(vars_per_elem, local_mat_col);
    // }


    if constexpr (Quadrature::num_quad_pts == 4) {
        /* warp shuffle here.. */
        // only works for 4 quadpts or linear quadrature
        int lane = tid % 32;
        int group_start = (lane / 4) * 4;
        for (int idof = 0; idof < vars_per_elem; idof++) {
            T lane_val = local_mat_col[idof];
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

            // warp broadcast
            lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
            local_mat_col[idof] = lane_val;
        }
        __syncthreads();

        int elem_block_row = ideriv / Phys::vars_per_node;
        int elem_inner_row = ideriv % Phys::vars_per_node;
        mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, iquad, Quadrature::num_quad_pts,
            Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);
    } else if (Quadrature::num_quad_pts != 4) {
        // 9 quadpts not sure how to warp reduce that yet (some groups of 9 lie on different warps)
        // so just atomicAdd directly into global mem (slower for now, TBD)

        int elem_block_row = ideriv / Phys::vars_per_node;
        int elem_inner_row = ideriv % Phys::vars_per_node;
        // 0, 1 mean start=0, stride=1 here.. slower too
        mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, 0, 1,
            Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_col);
    }

}  // end of add_jacobian_fast


template <typename T, int elems_per_block, class ElemGroup, class Data, template <typename> class Vec>
__GLOBAL__ static void k_add_residual_fast(int32_t vars_num_nodes, int32_t num_elements, Vec<int32_t> elem_comp,
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> compData, Vec<T> res) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + elems_per_block * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    if (!active_thread) return;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    // int nthreads = blockDim.x * blockDim.y;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_elem2 = vars_per_elem * vars_per_elem;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.x, Quadrature::num_quad_pts, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, block_xpts[local_elem]);
    __syncthreads();

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, threadIdx.x, Quadrature::num_quad_pts, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, block_vars[local_elem]);
     __syncthreads();

    if (threadIdx.x == 0) {
        int icomp = elem_comp[global_elem];
        block_data[local_elem] = _comp_data[icomp];
    }
    __syncthreads();

    int iquad = threadIdx.x;

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    constexpr bool fast_method = true;
    // constexpr bool fast_method = false;

    // OLD slower call
    if constexpr (!fast_method) {
        // // used here for verification + 
        // constexpr STRAIN strain = ALL;
        // constexpr STRAIN strain = DRILL;

        ElemGroup::template add_element_quadpt_residual<Data>(
        active_thread, iquad, block_xpts[local_elem], block_vars[local_elem],
        block_data[local_elem], local_res);
    }

    // NEW faster call
    if constexpr (fast_method) {
        // some prelim computations with xpts and quadrature point
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T fn[nxpts_per_elem];
        ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);
        T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
        T scale = detXd * weight; // scale for energy derivatives
        __syncthreads();
        T XdinvT[9], Tmat[9], XdinvzT[9];
        computeShellRotations<T, Basis, Data>(pt, block_data[local_elem].refAxis, 
            block_xpts[local_elem], fn, Tmat, XdinvT, XdinvzT);
        __syncthreads();

        // // compute drill strains
        ElemGroup::template add_element_quadpt_residual_fast<Data, DRILL>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], local_res);
        __syncthreads();

        // // compute bending strains
        ElemGroup::template add_element_quadpt_residual_fast<Data, BENDING>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], local_res);
        __syncthreads();

        // // // compute tying strains
        ElemGroup::template add_element_quadpt_residual_fast<Data, TYING>(
            pt, scale, block_xpts[local_elem], fn, 
            XdinvT, Tmat, XdinvzT, block_data[local_elem],
            block_vars[local_elem], local_res);
        __syncthreads();
    }

    if constexpr (Quadrature::num_quad_pts == 4) {
        /* warp shuffle here.. */
        // only works for 4 quadpts, linear quadrature
        int lane = tid % 32;
        int group_start = (lane / 4) * 4;
        for (int idof = 0; idof < vars_per_elem; idof++) {
            T lane_val = local_res[idof];
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

            // warp broadcast
            lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
            local_res[idof] = lane_val;
        }
        __syncthreads();

        res.addElementValuesFromShared(true, threadIdx.x, blockDim.x, Phys::vars_per_node,
                                    Basis::num_nodes, vars_elem_conn,
                                    local_res);
    } else if (Quadrature::num_quad_pts != 4) {
        // 9 quadpts not sure how to warp reduce that yet (some groups of 9 lie on different warps)
        // so just atomicAdd directly into global mem (slower for now, TBD)
        res.addElementValuesFromShared(true, 0, 1, Phys::vars_per_node,
                                    Basis::num_nodes, vars_elem_conn,
                                    local_res);
    }
    
}  // end of add_residual_fast

template <typename T, int elems_per_block, class ElemGroup, class Data,
          class LoadMagnitude, template <typename> class Vec>
__GLOBAL__ static void k_add_fext_fast(int32_t num_elements, LoadMagnitude mag,
                                       Vec<int32_t> elem_comp, Vec<int32_t> geo_conn,
                                       Vec<int32_t> vars_conn, Vec<T> xpts,
                                       Vec<Data> compData, T load_mag, Vec<T> fext, T xfrac = 0.0) {

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    constexpr int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    constexpr int xpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    int local_elem = threadIdx.y;
    int global_elem = local_elem + elems_per_block * blockIdx.x;
    bool active_thread = global_elem < num_elements;
    if (!active_thread) return;

    int iquad = threadIdx.x;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][xpts_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, threadIdx.x, Quadrature::num_quad_pts,
                                Geo::spatial_dim, Geo::num_nodes, geo_elem_conn,
                                block_xpts[local_elem]);
    __syncthreads();

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];

    if (threadIdx.x == 0) {
        int icomp = elem_comp[global_elem];
        block_data[local_elem] = _comp_data[icomp];
    }
    __syncthreads();

    T local_fext[vars_per_elem];
    for (int i = 0; i < vars_per_elem; i++) local_fext[i] = T(0);

    T pt[2];
    T wt = Quadrature::getQuadraturePoint(iquad, pt);

    // nodal normals
    T fn[xpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);
    __syncthreads();

    T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
    T scale = fabs(detXd) * wt;

    // basis values
    T N[Basis::num_nodes];
    Basis::getBasis(pt, N);

    // interpolate geometry at quadrature point
    T xq[3] = {0};
    Basis::template interpFields<3, 3>(pt, block_xpts[local_elem], xq);

    // evaluate load magnitude
    T q = mag(xq[0], xq[1], xq[2]);

    // interpolate shell normal
    T n0[3] = {0};
    Basis::template interpFields<3, 3>(pt, fn, n0);

    T e3[3] = {0.0, 0.0, 1.0};
    T mult = A2D::VecDotCore<T, 3>(n0, e3);

    T coeff = scale * q * load_mag;
    coeff *= mult; // optional

    // assemble element force vector
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        local_fext[Phys::vars_per_node * inode + 0] += (n0[0] + xfrac) * coeff * N[inode];
        local_fext[Phys::vars_per_node * inode + 1] += n0[1] * coeff * N[inode];
        local_fext[Phys::vars_per_node * inode + 2] += n0[2] * coeff * N[inode];
    }

    // reduction across quadrature points (same warp trick as your residual kernel)
    if constexpr (Quadrature::num_quad_pts == 4) {
        int tid = blockDim.x * threadIdx.y + threadIdx.x;
        int lane = tid % 32;
        int group_start = (lane / 4) * 4;

        for (int idof = 0; idof < vars_per_elem; idof++) {
            T lane_val = local_fext[idof];
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);
            lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
            local_fext[idof] = lane_val;
        }
        __syncthreads();

        fext.addElementValuesFromShared(true, threadIdx.x, blockDim.x,
                                        Phys::vars_per_node, Basis::num_nodes,
                                        vars_elem_conn, local_fext);
    } else {
        fext.addElementValuesFromShared(true, 0, 1,
                                        Phys::vars_per_node, Basis::num_nodes,
                                        vars_elem_conn, local_fext);
    }
}

template <typename T, int elems_per_block, class ElemGroup, class Data, template <typename> class Vec, class Mat>
__GLOBAL__ static void k_add_lockstrain_jacobian_fast(int32_t vars_num_nodes, int32_t num_elements, int cols_per_elem, Vec<int32_t> elem_comp, 
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> compData, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    int local_elem_col = threadIdx.y + cols_per_elem * threadIdx.z;
    int cols_per_block = elems_per_block * cols_per_elem;
    int global_elem_col = local_elem_col + blockIdx.x * cols_per_block;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    int n_elem_cols = num_elements * vars_per_elem;
    bool active_thread = global_elem_col < n_elem_cols;
    if (!active_thread) return;
    int global_elem = global_elem_col / vars_per_elem;
    int tid_xy = blockDim.x * threadIdx.y + threadIdx.x;
    int nthreads_xy = blockDim.x * blockDim.y;
    int tid = nthreads_xy * threadIdx.z + tid_xy;
    int local_elem = threadIdx.z;

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    if (tid_xy == 0) {
        memset(&block_xpts[0][0], 0.0, sizeof(T) * nxpts_per_elem);
    }
    __syncthreads();

    // load data into block shared mem using some subset of threads
    const int32_t *geo_elem_conn = &_geo_conn[global_elem * Geo::num_nodes];
    xpts.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Geo::spatial_dim,
                                Geo::num_nodes, geo_elem_conn, &block_xpts[local_elem][0]);
    __syncthreads();

    const int32_t *vars_elem_conn = &_vars_conn[global_elem * Basis::num_nodes];
    vars.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Phys::vars_per_node,
                                Basis::num_nodes, vars_elem_conn, block_vars[local_elem]);
     __syncthreads();
    

    if (tid_xy == 0) {
        int icomp = elem_comp[global_elem];
        block_data[local_elem] = _comp_data[icomp];
    }
    __syncthreads();

    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;

    T local_mat_row[vars_per_elem];
    memset(local_mat_row, 0.0, sizeof(T) * vars_per_elem);

    
    // some prelim computations with xpts and quadrature point
    // NOTE : it actually doesn't matter what quadpt we are at.. except for quadrature weights.. for this
    // the lockstrain G_f^T G_f product uses the tying strains at their MITC points which is before tying to quadpt interp
    T pt[2] = {0.0, 0.0}; // just choose centroid of element in [-1,1]^2 comp domain for anything quadpt related like detXd, etc.
    T weight = 1.0;
    // T pt[2]; // actually do need quadpt in order to get physical coords
    // T weight = Quadrature::getQuadraturePoint(iquad, pt);
    T fn[nxpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);
    T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
    T scale = detXd * weight; // scale for energy derivatives
    __syncthreads();
    T XdinvT[9], Tmat[9], XdinvzT[9];
    computeShellRotations<T, Basis, Data>(pt, block_data[local_elem].refAxis, block_xpts[local_elem], fn, Tmat, XdinvT, XdinvzT);
    __syncthreads();

    T p_vars[vars_per_elem];
    memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
    p_vars[ideriv] = 1.0;

    // const bool method 

    // // compute drill strains
    // NOTE : need drill strains in the lock-strain prod? prob not for now.. Check this later..
    // ElemGroup::template add_element_lockstrain_jacobian_col_fast<Data, DRILL>(
    //     pt, scale, block_xpts[local_elem], fn, 
    //     XdinvT, Tmat, XdinvzT, block_data[local_elem],
    //     block_vars[local_elem], p_vars, local_mat_col);
    // __syncthreads();

    // // // compute tying strains
    // ElemGroup::template add_element_lockstrain_jacobian_col_fast_v1
    ElemGroup::template add_element_lockstrain_jacobian_col_fast_v2<Data, TYING>(
        pt, scale, block_xpts[local_elem], fn, 
        XdinvT, Tmat, XdinvzT, block_data[local_elem],
        block_vars[local_elem], p_vars, local_mat_row);
    __syncthreads();

    // no quadrature point sum here (uses tying points instead)
    int elem_block_row = ideriv / Phys::vars_per_node;
    int elem_inner_row = ideriv % Phys::vars_per_node;
    mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, 0, 1,
        Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_row);
    
    // /* warp shuffle here.. */
    // // only works for 4 quadpts or linear quadrature
    // int lane = tid % 32;
    // int group_start = (lane / 4) * 4;
    // for (int idof = 0; idof < vars_per_elem; idof++) {
    //     T lane_val = local_mat_row[idof];
    //     lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
    //     lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

    //     // warp broadcast
    //     lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
    //     local_mat_row[idof] = lane_val;
    // }
    // __syncthreads();

    // int elem_block_row = ideriv / Phys::vars_per_node;
    // int elem_inner_row = ideriv % Phys::vars_per_node;
    // mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, iquad, Quadrature::num_quad_pts,
    //     Phys::vars_per_node, Basis::num_nodes, vars_elem_conn, local_mat_row);

}  // end of add_lockstrain_jacobian_fast



template <typename T, int elems_per_block, class ElemGroup, class Data, template <typename> class Vec, class Mat>
__GLOBAL__ static void k_add_lockstrain_fc_jacobian_fast(
    int32_t fine_num_nodes, int coarse_num_nodes, 
    int fine_num_elements, int coarse_num_elements,
    int *fc_elem_map,
    int cols_per_elem, Vec<int32_t> elem_comp, 
    Vec<int32_t> fine_geo_conn, Vec<int32_t> fine_vars_conn, 
    Vec<int32_t> coarse_geo_conn, Vec<int32_t> coarse_vars_conn,
    Vec<T> fine_xpts, Vec<T> fine_vars, 
    Vec<T> coarse_xpts, Vec<T> coarse_vars, 
    Vec<Data> compData, Mat mat) {

    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    int local_elem_col = threadIdx.y + cols_per_elem * threadIdx.z;
    int cols_per_block = elems_per_block * cols_per_elem;
    int global_elem_col = local_elem_col + blockIdx.x * cols_per_block;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    int n_elem_cols = fine_num_elements * vars_per_elem;
    bool active_thread = global_elem_col < n_elem_cols;
    if (!active_thread) return;
    int global_elem = global_elem_col / vars_per_elem;
    int tid_xy = blockDim.x * threadIdx.y + threadIdx.x;
    int nthreads_xy = blockDim.x * blockDim.y;
    int tid = nthreads_xy * threadIdx.z + tid_xy;
    int local_elem = threadIdx.z;

    int fine_global_elem = global_elem;
    int coarse_global_elem = fc_elem_map[fine_global_elem];

    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    const int32_t *_fine_geo_conn = fine_geo_conn.getPtr();
    const int32_t *_fine_vars_conn = fine_vars_conn.getPtr();
    const int32_t *_coarse_geo_conn = coarse_geo_conn.getPtr();
    const int32_t *_coarse_vars_conn = coarse_vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_fine_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_fine_vars[elems_per_block][vars_per_elem];
    __SHARED__ T block_coarse_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_coarse_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    if (tid_xy == 0) {
        memset(&block_fine_xpts[0][0], 0.0, sizeof(T) * nxpts_per_elem);
        memset(&block_coarse_xpts[0][0], 0.0, sizeof(T) * nxpts_per_elem);
    }
    __syncthreads();

    // load data into block shared mem using some subset of threads
    const int32_t *fine_geo_elem_conn = &_fine_geo_conn[fine_global_elem * Geo::num_nodes];
    fine_xpts.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Geo::spatial_dim,
                                Geo::num_nodes, fine_geo_elem_conn, &block_fine_xpts[local_elem][0]);
    __syncthreads();

    const int32_t *coarse_geo_elem_conn = &_coarse_geo_conn[coarse_global_elem * Geo::num_nodes];
    coarse_xpts.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Geo::spatial_dim,
                                Geo::num_nodes, coarse_geo_elem_conn, &block_coarse_xpts[local_elem][0]);
    __syncthreads();

    const int32_t *fine_vars_elem_conn = &_fine_vars_conn[fine_global_elem * Basis::num_nodes];
    fine_vars.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Phys::vars_per_node,
                                Basis::num_nodes, fine_vars_elem_conn, block_fine_vars[local_elem]);
     __syncthreads();

    const int32_t *coarse_vars_elem_conn = &_coarse_vars_conn[coarse_global_elem * Basis::num_nodes];
    coarse_vars.copyElemValuesToShared(active_thread, tid_xy, nthreads_xy, Phys::vars_per_node,
                                Basis::num_nodes, coarse_vars_elem_conn, block_coarse_vars[local_elem]);
    __syncthreads();
    

    if (tid_xy == 0) {
        int icomp = elem_comp[global_elem];
        block_data[local_elem] = _comp_data[icomp];
    }
    __syncthreads();

    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;

    T local_mat_row[vars_per_elem];
    memset(local_mat_row, 0.0, sizeof(T) * vars_per_elem);
    
    // some prelim computations with xpts and quadrature point
    // NOTE : it actually doesn't matter what quadpt we are at.. except for quadrature weights.. for this
    // the lockstrain G_f^T G_f product uses the tying strains at their MITC points which is before tying to quadpt interp
    T pt[2] = {0.0, 0.0}; // just choose centroid of element in [-1,1]^2 comp domain for anything quadpt related like detXd, etc.
    T weight = 1.0;
    // T pt[2]; // actually do need quadpt in order to get physical coords
    // T weight = Quadrature::getQuadraturePoint(iquad, pt);
    T fine_fn[nxpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_fine_xpts[local_elem], fine_fn);
    T coarse_fn[nxpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_coarse_xpts[local_elem], coarse_fn); // got here when failed last time
    T detXd = getDetXd<T, Basis>(pt, block_coarse_xpts[local_elem], fine_fn);
    T scale = detXd * weight; // scale for energy derivatives
    __syncthreads();
    T fine_XdinvT[9], fine_Tmat[9], fine_XdinvzT[9];
    computeShellRotations<T, Basis, Data>(pt, block_data[local_elem].refAxis, block_fine_xpts[local_elem], 
        fine_fn, fine_Tmat, fine_XdinvT, fine_XdinvzT);
    T coarse_XdinvT[9], coarse_Tmat[9], coarse_XdinvzT[9];
    computeShellRotations<T, Basis, Data>(pt, block_data[local_elem].refAxis, block_coarse_xpts[local_elem], 
        coarse_fn, coarse_Tmat, coarse_XdinvT, coarse_XdinvzT);
    __syncthreads();

    T p_vars[vars_per_elem];
    memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
    p_vars[ideriv] = 1.0;

    // compute tying strains (fine-coarse version)
    // since we are doing matRow not matCol (because more efficient on GPU), p_vars now represents fine 
    // and output mat_row is like coarse dim
    // ElemGroup::template add_element_lockstrain_fc_jacobian_row_fast_v1
    ElemGroup::template add_element_lockstrain_fc_jacobian_row_fast_v2<Data, TYING>(
        pt, scale, 
        block_fine_xpts[local_elem], block_coarse_xpts[local_elem], 
        fine_fn, coarse_fn,
        fine_XdinvT, fine_Tmat, fine_XdinvzT, 
        coarse_XdinvT, coarse_Tmat, coarse_XdinvzT, 
        block_data[local_elem],
        block_fine_vars[local_elem], block_coarse_vars[local_elem],
        p_vars, local_mat_row);
    __syncthreads();

    // no quadrature point sum here (uses tying points instead)
    // need to add properly as elemMatCol instead of elemMatRow (transpose trick) into prolong matrix
    // P = G_f^T * P_gam * G_c + lam * P_0   is not symmetric [F,C] dimensions
    int elem_block_row = ideriv / Phys::vars_per_node;
    int elem_inner_row = ideriv % Phys::vars_per_node;
    mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, 0, 1,
        Phys::vars_per_node, Basis::num_nodes, fine_vars_elem_conn, local_mat_row);
    
    // /* warp shuffle here.. */
    // // only works for 4 quadpts or linear quadrature
    // int lane = tid % 32;
    // int group_start = (lane / 4) * 4;
    // for (int idof = 0; idof < vars_per_elem; idof++) {
    //     T lane_val = local_mat_row[idof];
    //     lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
    //     lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

    //     // warp broadcast
    //     lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
    //     local_mat_row[idof] = lane_val;
    // }
    // __syncthreads();

    // int elem_block_row = ideriv / Phys::vars_per_node;
    // int elem_inner_row = ideriv % Phys::vars_per_node;
    // mat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, iquad, Quadrature::num_quad_pts,
    //     Phys::vars_per_node, Basis::num_nodes, _fine_vars_conn, local_mat_row);

}  // end of k_add_lockstrain_fc_jacobian_fast