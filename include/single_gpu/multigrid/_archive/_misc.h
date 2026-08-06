    auto h_soln = grid.d_defect.createPermuteVec(grid.block_dim, grid.Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_0.vtk");
    damping the colored GS (that's like damped jacobi)
    // T omega = 0.7;
    T omega = 1.0;
    DEBUG
    int color_order[] = {0,3,1,2};
    for (int iter = 0; iter < n_iters; iter++) {
        for (int _icolor = 0; _icolor < num_colors; _icolor++) {
            // don't go 0,1,2,3 colors go 0,3,2,1 order.. can also do damping or reverse order after that..
            // int icolor = color_order[_icolor];
            int icolor = _icolor;

            // get active rows / cols for this color
            int start = color_rowp[icolor], end = color_rowp[icolor + 1];
            int nblock_rows_color = end - start;
            // int block_dim2 = grid.block_dim * grid.block_dim;  // 36

            printf("iter %d, color %d : block rows [%d,%d)\n", iter, icolor, start, end);

            // 1) compute Dinv_c * defect_c => dx_c  (c indicates color subset)
            // int color_Dinv_nnzb = nblock_rows_color;
            // can use same rowp, cols here (0,1,...,nrows)
            // int *d_color_diag_cols = &grid.d_diag_cols[grid.block_dim * start];
            // T *d_Dinv_vals_color = &d_diag_inv_vals[block_dim2 * start];
            T *d_defect_color = &grid.d_defect.getPtr()[grid.block_dim * start];
            cudaMemset(grid.d_temp, 0.0, N * sizeof(T));  // holds dx_color
            T *d_temp_color = &grid.d_temp[grid.block_dim * start];
            T *d_temp_color2 = &grid.d_temp2[grid.block_dim * start];
            cudaMemset(grid.d_temp2, 0.0, N * sizeof(T)); // DEBUG
            cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * grid.block_dim * sizeof(T), cudaMemcpyDeviceToDevice);

            T a = 1.0, b = 0.0;

            const double alpha = 1.0;
            // try normal triang solve first.. DEBUG
            // printf("try normal triang solve first (DEBUG), get rid of this\n");
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(grid.cusparseHandle, grid.dir, grid.trans_L, grid.nnodes,
                                                    grid.diag_inv_nnzb, &alpha, grid.descr_L,
                                                    grid.d_diag_LU_vals, grid.d_diag_rowp, grid.d_diag_cols,
                                                    grid.block_dim, grid.info_L, grid.d_temp2,
                                                    grid.d_resid, grid.policy_L, grid.pBuffer)); // prob only need U^-1 part for block diag.. TBD
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(grid.cusparseHandle, grid.dir, grid.trans_U, grid.nnodes,
                                                    grid.diag_inv_nnzb, &alpha, grid.descr_U,
                                                    grid.d_diag_LU_vals, grid.d_diag_rowp, grid.d_diag_cols,
                                                    grid.block_dim, grid.info_U, grid.d_resid,
                                                    grid.d_temp, grid.policy_U, grid.pBuffer));

            // 2) update soln x_color += dx_color
            int nrows_color = nblock_rows_color * grid.block_dim;
            T *d_soln_color = &grid.d_soln.getPtr()[grid.block_dim * start];
            // a = 1.0;
            a = omega;
            CHECK_CUBLAS(
                cublasDaxpy(grid.cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));
            // printf("here3\n");

            // 3) update defect, defect -= K[color,:]^T * dx_color, with KT_color = N x
            // nrows_color matrix
            // int kmat_bnz_start = grid.h_color_bnz_ptr[icolor];
            // int kmat_bnz_end = grid.h_color_bnz_ptr[icolor + 1];
            // int color_kmat_nnzb = kmat_bnz_end - kmat_bnz_start;
            // T *d_color_kmat_vals = &grid.d_kmat_vals[36 * kmat_bnz_start];
            // int local_color_rowp_start = grid.h_color_local_rowp_ptr[icolor];
            // printf("kmat bnz %d to %d, local rowp start %d\n", kmat_bnz_start, kmat_bnz_end,
            //         local_color_rowp_start);
            // int *d_kmat_color_local_rowp = &grid.d_color_local_rowps[local_color_rowp_start];
            // int *d_kmat_color_cols = &grid.d_kmat_cols[kmat_bnz_start];
            // a = -1.0,
            a = -omega,
            b = 1.0;  // so that defect := defect - mat*vec

            CHECK_CUSPARSE(cusparseDbsrmv(
                grid.cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                grid.nnodes, grid.nnodes, grid.kmat_nnzb, &a, grid.descrKmat, grid.d_kmat_vals, grid.d_kmat_rowp, grid.d_kmat_cols,
                grid.block_dim, grid.d_temp, &b, grid.d_defect.getPtr()));
            // printf("here4\n");

            // auto h_soln = grid.d_soln.createHostVec(); // DEBUG
            auto h_soln = grid.d_defect.createPermuteVec(grid.block_dim, grid.Kmat.getPerm()).createHostVec();
             printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_" + std::to_string(num_colors * iter + icolor + 1) + ".vtk");

        }  // next color iteration

        printf("iter %d, done with color iterations\n", iter);

        /* report progress of defect nrm if printing.. */
        T defect_nrm;
        CHECK_CUBLAS(cublasDnrm2(grid.cublasHandle, N, grid.d_defect.getPtr(), 1, &defect_nrm));
        if (print && iter % print_freq == 0)
            printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

    }  // next block-GS iteration




// TODO : would like to just rewrite the full assembler later, let's do less intrusive wrappers for
// assembler from BDF, plate, cylinder or other geometries like what I did in the python for mitc
// shell..

// #include <cusparse_v2.h>
// #include <thrust/device_vector.h>

// #include "cublas_v2.h"
// #include "cuda_utils.h"
// #include "linalg/vec.h"
// #include "element.cuh"

// #include "element/shell/shell_elem_group.h"
// #include "element/shell/shell_elem_group.cuh"

// class ShellMultigridAssembler {
//     // lighter weight assembler for geometric multigrid (single grid here)
//     // TODO : would like to rewrite this from scratch again.. so more parallel, NZ pattern on GPU
//   public:

//     using T = double;
//     using Quad = QuadLinearQuadrature<T>;
//     using Basis = ShellQuadBasis<T, Quad, 2>;
//     using Geo = Basis::Geo;
//     using Data = ShellIsotropicData<T, false>;
//     using Physics = IsotropicShell<T, Data, false>;
//     using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;

//     static constexpr int xpts_per_node = 3;
//     static constexpr int vars_per_node = 6;

//     ShellMultigridAssembler(int num_nodes_, int num_elements_, DeviceVec<int> d_elem_conn,
//         DeviceVec<int> d_bcs, DeviceVec<T> d_xpts, DeviceVec<Data> d_data) :
//         num_nodes(num_nodes_), num_elements(num_elements_) {

//         elem_conn = d_elem_conn;
//         bcs = d_bcs;
//         xpts = d_xpts;
//         data = d_data;

//         num_dof = num_nodes * vars_per_node;

//         // zero vars
//         vars = DeviceVec<T>(num_dof);

//         // construct initial bsr data
//         h_bsr_data = BsrData(num_elements, num_nodes, 4, 6, elem_conn.getPtr());
//         d_bsr_data = h_bsr_data.createDeviceBsrData();
//     }

//     void compute_nofill_pattern() {
//         // compute nofill pattern (with desired ordering as well..) for coloring

//     }

//     void assembleJacobian() {

//         // assemble the Kmat
//         dim3 block(1, 24, 4); // (1 elems_per_block, 24 DOF per elem, 4 quadpts per elem)
//         int nblocks = (num_elements + block.x - 1) / block.x;
//         dim3 grid(nblocks);

//         add_jacobian_gpu<T, ElemGroup, Data, 1, DeviceVec, BsrMat><<<grid, block>>>(
//             num_nodes, num_elements, elem_conn, xpts, vars, physData, res, K_mat);

//         CHECK_CUDA(cudaDeviceSynchronize());
//     }

//     // public data
//     int num_nodes, num_dof, num_elements;
//     DeviceVec<int> bcs, elem_conn;
//     DeviceVec<T> xpts, vars;
//     DeviceVec<Data> data;
//     BsrData bsr_data;
//     BsrMat K_mat, Dinv_mat;
// };



template <class Basis>
class UnstructuredProlongationBsr {
   public:
    using T = double;
    static constexpr bool structured = false;
    static constexpr bool assembly = false;
    static constexpr int is_bsr = true;

    static void assemble_matrices(int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems,
                                  T *d_n2e_xis, BsrMat<DeviceVec<T>> &P_mat,
                                  BsrMat<DeviceVec<T>> &PT_mat) {
        // get some data out..
        auto P_bsr_data = P_mat.getBsrData();
        auto PT_bsr_data = PT_mat.getBsrData();
        int nnodes_fine = P_bsr_data.nnodes;
        // int nnodes_coarse = PT_bsr_data.nnodes;
        int *d_coarse_iperm = PT_bsr_data.iperm;
        int *d_fine_iperm = P_bsr_data.iperm;
        int block_dim = P_bsr_data.block_dim;

        // assemble P mat
        dim3 block(32);
        dim3 grid((nnodes_fine + 31) / 32);
        int *d_P_rowp = P_bsr_data.rowp, *d_P_cols = P_bsr_data.cols;
        T *d_P_vals = P_mat.getPtr();
        k_prolong_mat_assembly<T, Basis, is_bsr>
            <<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                              nnodes_fine, d_fine_iperm, d_P_rowp, d_P_cols, block_dim, d_P_vals);

        // assemble PT mat
        int *d_PT_rowp = PT_bsr_data.rowp, *d_PT_cols = PT_bsr_data.cols;
        T *d_PT_vals = PT_mat.getPtr();
        k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block>>>(
            d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, nnodes_fine,
            d_fine_iperm, d_PT_rowp, d_PT_cols, block_dim, d_PT_vals);
    }

    static void prolongate(int nnodes_fine, int *d_coarse_conn, int *d_n2e_ptr, int *d_n2e_elems,
                           T *d_n2e_xis, int *d_coarse_iperm, int *d_fine_iperm,
                           DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
        // zero anything out again
        int N_coarse = coarse_soln_in.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = dx_fine.getSize();

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_prolongate<T, Basis>
            <<<grid, block>>>(coarse_soln_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr,
                              d_n2e_elems, d_n2e_xis, nnodes_fine, d_fine_iperm, dx_fine.getPtr());
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    static void restrict_defect(int nnodes_fine, const int *d_coarse_conn, int *d_n2e_ptr,
                                int *d_n2e_elems, T *d_n2e_xis, int *d_coarse_iperm,
                                int *d_fine_iperm, DeviceVec<T> fine_defect_in,
                                DeviceVec<T> coarse_defect_out, T *d_weights) {
        // zero anything out again
        int N_coarse =
            coarse_defect_out.getSize();  // this includes dof per node (not num nodes here)
        int N_fine = fine_defect_in.getSize();
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        dim3 block(32);
        int nblocks = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks);

        k_unstruct_restrict<T, Basis><<<grid, block>>>(
            fine_defect_in.getPtr(), d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems,
            d_n2e_xis, nnodes_fine, d_fine_iperm, coarse_defect_out.getPtr(), d_weights);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // TRY not normalizing restriction (that kind of breaks P and P^T relationship..) to
        // comment, seems to be fine either way.. this out.. normalize
        // int nblock2 = (N_coarse + 31) / 32;
        // dim3 grid2(nblock2);
        // k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_defect_out.getPtr(), d_weights);
    }
};

template <typename T, class Basis>
__global__ static void k_unstruct_prolongate(const T *coarse_soln, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *fine_soln) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];
    
    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get local coarse elem disps.. (with permutations here?)
        T c_elem_disps[24];
        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            c_elem_disps[i] = coarse_soln[6 * perm_coarse_node + loc_dof];
        }

        // interp the disps from coarse disps
        T fine_disp_add[6];
        Basis::template interpFields<6, 6>(pt, c_elem_disps, fine_disp_add);

        // now add into the fine solution and fine weights
        T scale = 1.0 / (double) num_attached_elems;
        for (int idof = 0; idof < 6; idof++) {
            atomicAdd(&fine_soln[6 * perm_fine_node + idof], fine_disp_add[idof] * scale);
        }
    }
}

template <typename T, class Basis>
__global__ static void k_unstruct_restrict(const T *fine_defect_in, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *coarse_soln_out, T *coarse_wts) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln
    int num_attached_elems = node2elem_ptr[fine_node + 1] - node2elem_ptr[fine_node];

    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get fine defect disps..
        const T *fine_nodal_defect = &fine_defect_in[6 * perm_fine_node];

        // now do interpFieldsTranspose to coarse defect on each node in element
        T coarse_elem_defect[24];
        memset(coarse_elem_defect, 0.0, 24 * sizeof(T));
        Basis::template interpFieldsTranspose<6, 6>(pt, fine_nodal_defect, coarse_elem_defect);

        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            T scale = 1.0 / (double) num_attached_elems;
            atomicAdd(&coarse_soln_out[6 * perm_coarse_node + loc_dof], 
                coarse_elem_defect[i] * scale);
            atomicAdd(&coarse_wts[6 * perm_coarse_node + loc_dof], scale);
        }
    }
}

