#pragma once
#include "cuda_utils.h"
#include "element/shell/strains/basic_utils.h"
#include "element/shell/strains/strain_types.h"

template <typename T, int elems_per_block, class ElemGroup>
__GLOBAL__ static void k_add_multigpu_jacobian_fast(
    int32_t loc_num_nodes,
    int32_t loc_num_elements,
    int cols_per_elem,
    const int *loc_elem_comps,
    const int32_t *loc_elem_conn,
    const T *loc_xpts,
    const T *loc_vars,
    const typename ElemGroup::Data *loc_comp_data,
    const int *loc_elem_ind_map,
    T *loc_mat_vals
) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    static constexpr int vars_per_node = Phys::vars_per_node;
    static constexpr int nodes_per_elem = Basis::num_nodes;
    static constexpr int vars_per_elem = nodes_per_elem * vars_per_node;
    static constexpr int blocks_per_elem = nodes_per_elem * nodes_per_elem;
    static constexpr int block_dim = vars_per_node;
    static constexpr int block_dim2 = block_dim * block_dim;

    static constexpr int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;


    static_assert(Quadrature::num_quad_pts == 4,
                  "This fast residual kernel assumes exactly 4 quadrature points.");

    int local_elem_col = threadIdx.y + cols_per_elem * threadIdx.z;
    int cols_per_block = elems_per_block * cols_per_elem;
    int global_elem_col = local_elem_col + blockIdx.x * cols_per_block;

    int n_elem_cols = loc_num_elements * vars_per_elem;
    bool active_thread = global_elem_col < n_elem_cols;
    if (!active_thread) return;

    int elem = global_elem_col / vars_per_elem;
    int ideriv = global_elem_col % vars_per_elem;

    int tid_xy = blockDim.x * threadIdx.y + threadIdx.x;
    int nthreads_xy = blockDim.x * blockDim.y;
    int tid = nthreads_xy * threadIdx.z + tid_xy;
    int local_elem = threadIdx.z;

    int iquad = threadIdx.x;

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    const int32_t *elem_conn = &loc_elem_conn[elem * nodes_per_elem];

    for (int i = tid_xy; i < nxpts_per_elem; i += nthreads_xy) {
        int inode = i / Geo::spatial_dim;
        int idim = i % Geo::spatial_dim;
        int loc_node = elem_conn[inode];

        block_xpts[local_elem][i] =
            loc_xpts[loc_node * Geo::spatial_dim + idim];
    }

    for (int i = tid_xy; i < vars_per_elem; i += nthreads_xy) {
        int inode = i / vars_per_node;
        int idof = i % vars_per_node;
        int loc_node = elem_conn[inode];

        block_vars[local_elem][i] =
            loc_vars[loc_node * vars_per_node + idof];
    }

    if (tid_xy == 0) {
        int icomp = static_cast<int>(loc_elem_comps[elem]);
        block_data[local_elem] = loc_comp_data[icomp];
    }

    __syncthreads();

    T local_mat_col[vars_per_elem];
    memset(local_mat_col, 0.0, sizeof(T) * vars_per_elem);

    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    T fn[nxpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);

    T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
    T scale = detXd * weight;

    T XdinvT[9], Tmat[9], XdinvzT[9];

    computeShellRotations<T, Basis, Data>(
        pt,
        block_data[local_elem].refAxis,
        block_xpts[local_elem],
        fn,
        Tmat,
        XdinvT,
        XdinvzT
    );

    T p_vars[vars_per_elem];
    memset(p_vars, 0.0, sizeof(T) * vars_per_elem);
    p_vars[ideriv] = 1.0;

    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, DRILL>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        p_vars,
        local_mat_col
    );

    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, BENDING>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        p_vars,
        local_mat_col
    );

    ElemGroup::template add_element_quadpt_jacobian_col_fast<Data, TYING>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        p_vars,
        local_mat_col
    );

    // warp reduction
    int lane = tid % 32;
    int group_start = (lane / 4) * 4;

    for (int idof = 0; idof < vars_per_elem; idof++) {
        T lane_val = local_mat_col[idof];

        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 2);
        lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, 1);

        lane_val = __shfl_sync(0xFFFFFFFF, lane_val, group_start);
        local_mat_col[idof] = lane_val;
    }

    __syncthreads();

    if (iquad == 0) {
        int elem_block_col = ideriv / block_dim;
        int elem_inner_col = ideriv % block_dim;

        for (int erow = 0; erow < vars_per_elem; erow++) {
            int elem_block_row = erow / block_dim;
            int elem_inner_row = erow % block_dim;

            int elem_block = elem_block_row * nodes_per_elem + elem_block_col;
            int glob_block_ind =
                loc_elem_ind_map[elem * blocks_per_elem + elem_block];

            if (glob_block_ind < 0) continue;

            int inz = elem_inner_row * block_dim + elem_inner_col;
            // int inz = elem_inner_col * block_dim + elem_inner_row;

            atomicAdd(
                &loc_mat_vals[glob_block_ind * block_dim2 + inz],
                local_mat_col[erow]
            );
        }
    }
}


template <typename T, int elems_per_block, class ElemGroup>
__GLOBAL__ static void k_add_multigpu_residual_fast(
    int32_t loc_num_nodes,
    int32_t loc_num_elements,
    const int *loc_elem_comps,
    const int32_t *loc_row_elem_conn,
    const int32_t *loc_col_elem_conn,
    const T *loc_xpts,
    const T *loc_vars,
    const typename ElemGroup::Data *loc_comp_data,
    T *loc_res
) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    static constexpr int vars_per_node = Phys::vars_per_node;
    static constexpr int nodes_per_elem = Basis::num_nodes;
    static constexpr int vars_per_elem = nodes_per_elem * vars_per_node;
    static constexpr int nxpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    static_assert(Quadrature::num_quad_pts == 4,
                  "This fast residual kernel assumes exactly 4 quadrature points.");

    int local_elem = threadIdx.y;
    int elem = local_elem + elems_per_block * blockIdx.x;
    bool active_elem = elem < loc_num_elements;

    int iquad = threadIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    __SHARED__ T block_xpts[elems_per_block][nxpts_per_elem];
    __SHARED__ T block_vars[elems_per_block][vars_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    if (active_elem) {
        const int32_t *col_elem_conn =
            &loc_col_elem_conn[elem * nodes_per_elem];

        for (int i = threadIdx.x; i < nxpts_per_elem; i += blockDim.x) {
            int inode = i / Geo::spatial_dim;
            int idim = i % Geo::spatial_dim;
            int loc_node = col_elem_conn[inode];

            block_xpts[local_elem][i] =
                loc_xpts[loc_node * Geo::spatial_dim + idim];
        }

        for (int i = threadIdx.x; i < vars_per_elem; i += blockDim.x) {
            int inode = i / vars_per_node;
            int idof = i % vars_per_node;
            int loc_node = col_elem_conn[inode];

            block_vars[local_elem][i] =
                loc_vars[loc_node * vars_per_node + idof];
        }

        if (threadIdx.x == 0) {
            int icomp = static_cast<int>(loc_elem_comps[elem]);
            block_data[local_elem] = loc_comp_data[icomp];
        }
    }

    __syncthreads();

    if (!active_elem) {
        return;
    }

    T local_res[vars_per_elem];
    memset(local_res, 0, sizeof(T) * vars_per_elem);

    T pt[2];
    T weight = Quadrature::getQuadraturePoint(iquad, pt);

    T fn[nxpts_per_elem];
    ShellComputeNodeNormals<T, Basis>(block_xpts[local_elem], fn);

    T detXd = getDetXd<T, Basis>(pt, block_xpts[local_elem], fn);
    T scale = detXd * weight;

    T XdinvT[9], Tmat[9], XdinvzT[9];

    computeShellRotations<T, Basis, Data>(
        pt,
        block_data[local_elem].refAxis,
        block_xpts[local_elem],
        fn,
        Tmat,
        XdinvT,
        XdinvzT
    );

    ElemGroup::template add_element_quadpt_residual_fast<Data, DRILL>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        local_res
    );

    ElemGroup::template add_element_quadpt_residual_fast<Data, BENDING>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        local_res
    );

    ElemGroup::template add_element_quadpt_residual_fast<Data, TYING>(
        pt,
        scale,
        block_xpts[local_elem],
        fn,
        XdinvT,
        Tmat,
        XdinvzT,
        block_data[local_elem],
        block_vars[local_elem],
        local_res
    );

    // For quad_pts = 4, threadIdx.x = 0,1,2,3 are the 4 quadrature threads
    // for each local_elem. Reduce those 4 threads to threadIdx.x == 0.
    unsigned mask = __activemask();

    for (int idof = 0; idof < vars_per_elem; idof++) {
        T val = local_res[idof];

        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);

        local_res[idof] = val;
    }

    if (iquad == 0) {
        const int32_t *row_elem_conn =
            &loc_row_elem_conn[elem * nodes_per_elem];

        for (int i = 0; i < vars_per_elem; i++) {
            int inode = i / vars_per_node;
            int idof = i % vars_per_node;

            int owned_node = row_elem_conn[inode];
            if (owned_node < 0) continue;

            atomicAdd(
                &loc_res[owned_node * vars_per_node + idof],
                local_res[i]
            );
        }
    }
}


template <typename T, int elems_per_block, class ElemGroup, class LoadMagnitude>
__GLOBAL__ static void k_add_multigpu_fext_fast(
    int32_t loc_num_elements,
    LoadMagnitude mag,
    const int *loc_elem_comps,
    const int32_t *loc_row_elem_conn,
    const int32_t *loc_col_elem_conn,
    const T *loc_xpts,
    const typename ElemGroup::Data *loc_comp_data,
    T load_mag,
    T *loc_fext,
    T xfrac = 0.0
) {
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;

    static constexpr int vars_per_node = Phys::vars_per_node;
    static constexpr int nodes_per_elem = Basis::num_nodes;
    static constexpr int vars_per_elem = nodes_per_elem * vars_per_node;
    static constexpr int xpts_per_elem = Geo::num_nodes * Geo::spatial_dim;

    static_assert(Quadrature::num_quad_pts == 4,
                  "This fast residual kernel assumes exactly 4 quadrature points.");

    int local_elem = threadIdx.y;
    int elem = local_elem + elems_per_block * blockIdx.x;
    bool active_elem = elem < loc_num_elements;

    int iquad = threadIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    __SHARED__ T block_xpts[elems_per_block][xpts_per_elem];
    __SHARED__ Data block_data[elems_per_block];

    if (active_elem) {
        const int32_t *col_elem_conn =
            &loc_col_elem_conn[elem * nodes_per_elem];

        for (int i = threadIdx.x; i < xpts_per_elem; i += blockDim.x) {
            int inode = i / Geo::spatial_dim;
            int idim = i % Geo::spatial_dim;
            int loc_node = col_elem_conn[inode];

            block_xpts[local_elem][i] =
                loc_xpts[loc_node * Geo::spatial_dim + idim];
        }

        if (threadIdx.x == 0) {
            int icomp = static_cast<int>(loc_elem_comps[elem]);
            block_data[local_elem] = loc_comp_data[icomp];
        }
    }

    __syncthreads();

    if (!active_elem) {
        return;
    }

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

    // For quad_pts = 4, threadIdx.x = 0,1,2,3 are the 4 quadrature threads
    // for each local_elem. Reduce those 4 threads to threadIdx.x == 0.
    unsigned mask = __activemask();

    for (int idof = 0; idof < vars_per_elem; idof++) {
        T val = local_fext[idof];

        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);

        local_fext[idof] = val;
    }

    if (iquad == 0) {
        const int32_t *row_elem_conn =
            &loc_row_elem_conn[elem * nodes_per_elem];

        for (int i = 0; i < vars_per_elem; i++) {
            int inode = i / vars_per_node;
            int idof = i % vars_per_node;

            int owned_node = row_elem_conn[inode];
            if (owned_node < 0) continue;

            atomicAdd(
                &loc_fext[owned_node * vars_per_node + idof],
                local_fext[i]
            );
        }
    }
}