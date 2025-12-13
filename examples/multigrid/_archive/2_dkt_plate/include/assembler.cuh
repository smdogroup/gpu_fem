#pragma once
#include "elem.cuh"
#include "linalg/bsr_mat.h"

/* device helper functions */
template <typename T>
__device__ T d_getPlateLoads(const T x, const T y, const T load_mag) {
    double PI = 3.1415926535897;
    T r = sqrt(x * x + y * y);
    T th = atan2(y, x);
    T nodal_load = load_mag * sin(5.0 * PI * r) * cos(4.0 * th);
    nodal_load *= std::pow(sin(2 * th), 0.25); // damps out some near edges (so more stable and smooth loading)
    // otherwise crazy loading field does result in weird rots near edges
    nodal_load *= std::pow(sin(PI / sqrt(2.0) * r), 0.25); // also damp out at diag ends too.. to stabilize loading
    // 0.25 root makes it not over-damp the solution near the edges.. like super-ellipse effect
    return nodal_load;
}

__device__ int d_getPlateBC(const int nx, const int full_idof) {
    int inode = full_idof / 3, ivar = full_idof % 3;
    int ix = inode % nx, iy = inode / nx;

    // helper bools first
    int x_bndry = ix == 0 || ix == (nx - 1);
    int y_bndry = iy == 0 || iy == (nx - 1);

    // three cases for each DOF
    int w_bc = ivar == 0 && (x_bndry || y_bndry);
    int thx_bc = ivar == 1 && x_bndry;
    int thy_bc = ivar == 2 && y_bndry;

    return w_bc || thx_bc || thy_bc;
}

/* helper kernel functions for the assembler using the elem.cuh device functions.. */
template <typename T>
__global__ void k_assemble_stiffness_matrix(const int nxe, const T E, const T thick, const T nu, BsrMat<DeviceVec<T>> Kmat) {
    // just using the simple unit square mesh here..
    // need to have already computed the sparsity in Kmat

    // get problem size here..
    int nx = nxe + 1;
    // int N = nx * nx;
    // int ndof = 3 * N;
    int nelems = nxe * nxe * 2; // because two triangle elems per quad
    // total of 3 quadpts and 9 nodes => 27 quadpt_kelem_col computations, one per thread..
    // can we actually do warp reduction though? not quite since quadpt reductions multiple of 3? TBD..
    int num_quadpt_cols = nelems * 27;
 
    // get the col and quadpt numbers for each thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 1D thread id..
    if (tid >= num_quadpt_cols) return;
    int ielem = tid / 27;
    int icol_elem = tid % 27;
    int inode = icol_elem / 3;
    int iquadpt = icol_elem % 3;
    T h = 1.0 / nxe;

    // compute the xpts for this element
    int iquad_elem = ielem / 2, itri = ielem % 2;
    int ixe = iquad_elem % nxe, iye = iquad_elem / nxe;
    T x1 = h * ixe, x2 = h * (ixe + 1), y1 = h * iye, y2 = h * (iye + 1);
    
    // use 0 or 1 bools to set the xpts coords.. for one of two tris in each quad spot
    T elem_xpts[6]; // itri == 0 is first tri in quad, itri == 1 is second tri in 
    // first set x1, x2, x3
    elem_xpts[0] = x1 * (1 - itri) + x2 * itri;
    elem_xpts[2] = x2 * (1 - itri) + x1 * itri;
    elem_xpts[4] = x1 * (1 - itri) + x2 * itri;
    // then set y1, y2, y3
    elem_xpts[1] = y1 * (1 - itri) + y2 * itri;
    elem_xpts[3] = y1 * (1 - itri) + y2 * itri;
    elem_xpts[5] = y2 * (1 - itri) + y1 * itri;
    // no z coords included here in xpts.. planar but does bend out of plane..

    // now compute local nodes..
    int n1 = nx * iye + ixe;
    int n2 = n1 + 1, n3 = n1 + nx, n4 = n1 + nx + 1;
    int loc_nodes[3];
    loc_nodes[0] = n1 * (1 - itri) + n4 * itri;
    loc_nodes[1] = n2 * (1 - itri) + n3 * itri;
    loc_nodes[2] = n3 * (1 - itri) + n2 * itri;

    // maybe do syncthreads here to split registers?, so doing same style comp on each thread
    __syncthreads();
    
    // set the quadpts and weights
    int is_zero = iquadpt == 0, is_one = iquadpt == 1;
    T xi = 1.0 / 6.0 + 1.0 / 2.0 * is_zero; // 1/6 if not zero, 2/3 if zero
    T eta = 1.0 / 6.0 + 1.0 / 2.0 * is_one; // 1/6 if not one, 2/3 if zero (iquadpt)
    T quadpt_wt = 1.0 / 3.0;

    // if (tid == 27) {
    //     printf("pre kelem col, thread %d : itri %d, xi %.2e, eta %.2e, weight %.2e, E %.2e, thick %.2e, nu %.2e, and xpts\n", 
    //         tid, itri, xi, eta, quadpt_wt, E, thick, nu);
    //     printVec<T>(6, elem_xpts);
    // }
    // // return;

    // compute kelem col..
    T kelem_col[9];
    memset(kelem_col, 0.0, 9 * sizeof(T));
    DKTElement<T>::get_quadpt_kelem_col(inode, elem_xpts, xi, eta, quadpt_wt, E, thick, nu, kelem_col);
    __syncthreads(); // divide registers between compute and the next part of adding into final data structure

    // if (tid == 0) {
    //     printf("ielem %d, icol %d, Kelem_col: ", ielem, icol_elem);
    //     printVec<T>(9, kelem_col);
    // }
    // // return;

    // add into global Kelem.. with permute?
    int ideriv = inode;
    int elem_block_row = ideriv / 3;
    int elem_inner_row = ideriv % 3;
    int global_elem = ielem;
    int start = 0;
    int stride = 1;
    int dof_per_node = 3;
    int nodes_per_elem = 3;

    Kmat.addElementMatRow(true, elem_block_row, elem_inner_row, global_elem, start, stride,
        dof_per_node, nodes_per_elem, kelem_col);
}

template <typename T>
__global__ void k_applyBCsLHS(int nxe, int nnzb, int *d_rows, int *d_cols, int *d_perm, T *d_vals) {
    // loops through all nnz right now (maybe not most efficient)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 9 * nnzb) return;

    int nx = nxe + 1;
    // int nnodes = nx * nx;
    // int ndof = nnodes * 3;

    // only zeroing out bcs in rows, not cols here (makes stiffness matrix not sym)
    // otherwise need rows for each nnz as well..
    int csr_ind = tid / 9; // like which node here..
    int block_ind = tid % 9;
    int perm_brow = d_rows[csr_ind];
    int perm_bcol = d_cols[csr_ind];

    int brow = d_perm[perm_brow], bcol = d_perm[perm_bcol];
    int row = 3 * brow + block_ind / 3;
    int col = 3 * bcol + block_ind % 3;
    __syncthreads();

    // check bcs..
    int row_bc = d_getPlateBC(nx, row);
    int col_bc = d_getPlateBC(nx, col);
    int bndry = row_bc || col_bc;
    int diag_bndry = (row == col) && bndry;

    // apply bcs to values
    T val = d_vals[tid];
    val = !bndry ? val : 0.0;
    val = !diag_bndry ? val : 1.0;

    __syncthreads();
    // split registers here for write..

    d_vals[tid] = val;
}

template <typename T>
__global__ void k_assembleRHS(const T load_mag, int nxe, int *d_iperm, T *d_rhs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nx = nxe + 1;
    int nnodes = nx * nx;
    int ndof = nnodes * 3;
    
    if (tid >= ndof) return;
    // includes bcs here also
    int inode = tid / 3;
    int ix = inode % nx, iy = inode / nx;
    T dx = 1.0 / nxe;
    T x = ix * dx, y = iy * dx;

    T load = tid % 3 == 0 ? d_getPlateLoads<T>(x, y, load_mag) : 0.0;
    int is_bc = d_getPlateBC(nx, tid);
    load *= (1 - is_bc);

    int perm_dof = 3 * d_iperm[inode] + tid % 3; 
    d_rhs[perm_dof] = load;
}

