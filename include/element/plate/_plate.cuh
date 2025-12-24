#pragma once
#include "cuda_utils.h"
#include "physics/_plate_utils.h"

// copied from / based on "shell/_shell.cuh"

// add_jacobian fast kernel
// -------------------
template <typename T, int elems_per_block, class ElemGroup, class Data, template <typename> class Vec, class Mat>
__GLOBAL__ static void k_add_aigplate_jacobian_fast(int32_t vars_num_nodes, int32_t num_elements, int cols_per_elem, Vec<int32_t> elem_comp, 
                                        Vec<int32_t> geo_conn, Vec<int32_t> vars_conn, Vec<T> xpts,
                                        Vec<T> vars, Vec<Data> compData, Mat mat) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Quadrature = typename Basis::Quadrature;

    int local_elem_col = threadIdx.y + cols_per_elem * threadIdx.z;
    int cols_per_block = elems_per_block * cols_per_elem;
    int global_elem_col = local_elem_col + blockIdx.x * cols_per_block;
    const int vars_per_node = Phys::vars_per_node;
    const int vars_per_elem = Basis::num_nodes * vars_per_node;
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
    __SHARED__ bool block_bndry[elems_per_block][4];

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

        // compute left, right, top, bottom bndry indicator array
        // which modifies slightly the edge IGA basis functions
        int nxe = sqrt((float)num_elements); // assumes square grid (equal dimension in x and y)
        int ix_elem = global_elem % nxe;
        int iy_elem = global_elem / nxe;
        bool left = ix_elem == 0, right = ix_elem == (nxe - 1);
        bool bottom = iy_elem == 0, top = iy_elem == (nxe - 1);

        block_bndry[local_elem][0] = left;
        block_bndry[local_elem][1] = right;
        block_bndry[local_elem][2] = bottom;
        block_bndry[local_elem][3] = top;
    }
    __syncthreads();

    int ideriv = global_elem_col % vars_per_elem;
    int iquad = threadIdx.x;

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    T p_vars[vars_per_elem];
    memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
    p_vars[ideriv] = 1.0;

    // get original thickness 
    T orig_thick = block_data[local_elem].thick;

    // asymptotic transformation for stiffness matrix
    if (tid_xy == 0) {
        // rescale xyz coords by thickness and replace thickness with 1.0
        ElemGroup::template _asymptotic_lhs_transform<Data>(block_xpts[local_elem], block_data[local_elem]);

        // rescale rotations by original thickness
        asymptoticRotationTransform<T, vars_per_node, Basis>(orig_thick, block_vars[local_elem]);
    }
    __syncthreads();

    // if (blockIdx.x == 0) {
    //     printf("new thick = %.4e from orig thick %.4e\n", block_data[local_elem].thick, orig_thick);
    // }

    // rescale rotations by original thickness
    asymptoticRotationTransform<T, vars_per_node, Basis>(orig_thick, p_vars);

    // compute drill strains
    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, DRILL>(
        iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
        block_vars[local_elem], p_vars, local_mat_col);
    __syncthreads();

    // compute bending strains
    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, BENDING>(
        iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
        block_vars[local_elem], p_vars, local_mat_col);
    __syncthreads();

    // compute tying strains
    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, TYING>(
        iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
        block_vars[local_elem], p_vars, local_mat_col);
    __syncthreads();

    // rescale rotations by original thickness
    asymptoticRotationTransform<T, vars_per_node, Basis>(orig_thick, local_mat_col);


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
__GLOBAL__ static void k_add_aigplate_residual_fast(int32_t vars_num_nodes, int32_t num_elements, Vec<int32_t> elem_comp,
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

    const int vars_per_node = Phys::vars_per_node;
    const int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;
    const int vars_per_elem = Basis::num_nodes * Phys::vars_per_node;
    // const int vars_per_elem2 = vars_per_elem * vars_per_elem;

    const int32_t *_geo_conn = geo_conn.getPtr();
    const int32_t *_vars_conn = vars_conn.getPtr();
    const Data *_comp_data = compData.getPtr();

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];
    __SHARED__ bool block_bndry[elems_per_block][4];

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

        // compute left, right, top, bottom bndry indicator array
        // which modifies slightly the edge IGA basis functions
        int nxe = sqrt((float)num_elements); // assumes square grid (equal dimension in x and y)
        int ix_elem = global_elem % nxe;
        int iy_elem = global_elem / nxe;
        bool left = ix_elem == 0, right = ix_elem == (nxe - 1);
        bool bottom = iy_elem == 0, top = iy_elem == (nxe - 1);

        block_bndry[local_elem][0] = left;
        block_bndry[local_elem][1] = right;
        block_bndry[local_elem][2] = bottom;
        block_bndry[local_elem][3] = top;
    }
    __syncthreads();

    T orig_thick = block_data[local_elem].thick;

    int iquad = threadIdx.x;

    T local_res[vars_per_elem];
    memset(local_res, 0.0, sizeof(T) * vars_per_elem);

    // asymptotic transformation for stiffness matrix
    if (tid == 0) {
        // rescale xyz coords by thickness and replace thickness with 1.0
        ElemGroup::template _asymptotic_lhs_transform<Data>(block_xpts[local_elem], block_data[local_elem]);

        // rescale rotations by original thickness
        asymptoticRotationTransform<T, vars_per_node, Basis>(orig_thick, block_vars[local_elem]);
    }
    __syncthreads();

    

    // compute drill strains
    ElemGroup::template add_element_quadpt_residual_fast<Data, DRILL>(
        iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
        block_vars[local_elem], local_res);
    __syncthreads();

    // compute bending strains
    ElemGroup::template add_element_quadpt_residual_fast<Data, BENDING>(
    iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
    block_vars[local_elem], local_res);
    __syncthreads();

    // compute tying strains
    ElemGroup::template add_element_quadpt_residual_fast<Data, TYING>(
    iquad, block_bndry[local_elem], block_xpts[local_elem], block_data[local_elem],
    block_vars[local_elem], local_res);
    __syncthreads();

    // rescale rotations by original thickness
    asymptoticRotationTransform<T, vars_per_node, Basis>(orig_thick, local_res);

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
    
}  // end of add_jacobian_fast


template <typename T, class ElemGroup, class Data, template <typename> class Vec>
__GLOBAL__ static void k_asymptotic_rhs_transform(int32_t num_nodes, int32_t num_elements, Vec<int32_t> elem_comp,
                                        Vec<Data> compData, Vec<T> fext) {

    using Phys = typename ElemGroup::Phys;
    int fine_node = threadIdx.x + blockDim.x * blockIdx.x;
    const int vars_per_node = Phys::vars_per_node;
    if (fine_node >= num_nodes) return;
    const Data *_comp_data = compData.getPtr();

    // TODO / WARNING : for now I'm just choosing / assuming one component for thickness rescale (right now)
    //  would it still work asymptotically if during optimization different thicknesses?
    //  maybe you'd choose one thickness rescale for whole structure and there might be a rotation force between components .. TBD
    
    int num_comp = elem_comp.getSize();
    if (num_comp > 0) {
        printf("num comp %d > 0 => asymptotic RHS transform not written for multiple components yet.. TBD and think of theory for this then fix\n");
        printf("FATAL not adjusting RHS\n");
        return;
    }

    __SHARED__ Data block_data[1];
    if (threadIdx.x == 0) {
        int global_elem = 0; // NOTE : not fully general yet (assumes one component)
        int icomp = elem_comp[global_elem];
        block_data[0] = _comp_data[icomp];
    }
    __syncthreads();

    // now rescale right hand side for this node here..
    T *fext_ptr = fext.getPtr();
    T *fext_node = &fext_ptr[vars_per_node * fine_node];
    ElemGroup::template _asymptotic_rhs_transform<Data>(fext_node, block_data[0]);
    
}