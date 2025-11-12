/* demo smoothed aggregation GMG on a plate */

// based on the paper, \href{https://link.springer.com/article/10.1007/s006070050022}{Energy Optimization of Algebraic Multigrid Bases}
// goal here is to do small plate first, explicitly forming and modifying the prolong matrix
// the smoother Dinv and mat-mat products are computed slow version here using incomplete LU (not fast one like the fast + optimized GSMC smoother in multigrid/smoothers/mc_smooth1.h)
// will performance optimize later if the process is right

// some temp source code will come from _src.h

#include <vector>
// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>


int main() {

    /* 1) type definitions for this shell element */
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Basis = LagrangeQuadBasis<T, Quad, 2>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // some problem specific user inputs
    int nxe = 64; // two-grid problem with second grid the coarser one
    double SR = 10.0;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    /* 2) create the fine grid assembler, multicolor reordering  */
    
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nxe/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *fine_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q);
    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    int N = res.getSize();
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(kmat);

    auto f_smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, 1.0);
    auto f_prolongation = new Prolongation(assembler);
    auto f_grid = GRID(assembler, f_prolongation, f_smoother, kmat, loads, cublasHandle, cusparseHandle);


    /* 3) create the coarse grid assembler, multicolor reordering  */
    
    nxe_per_comp = nxe / 8, nye_per_comp = nxe / 8; // for now (should have 25 grids)
    auto c_assembler = createPlateAssembler<Assembler>(nxe / 2, nxe / 2, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    T *coarse_loads = getPlateLoads<T, Physics>(nxe / 2, nxe / 2, Lx, Ly, Q);
    auto &c_bsr_data = c_assembler.getBsrData();
    int c_num_colors, *_c_color_rowp;
    c_bsr_data.multicolor_reordering(c_num_colors, _c_color_rowp);
    c_bsr_data.compute_nofill_pattern();
    auto c_h_color_rowp = HostVec<int>(c_num_colors + 1, _c_color_rowp);
    c_assembler.moveBsrDataToDevice();
    auto c_loads = c_assembler.createVarsVec(coarse_loads);
    assembler.apply_bcs(c_loads);
    auto c_kmat = createBsrMat<Assembler, VecType<T>>(c_assembler);
    auto c_res = c_assembler.createVarsVec();
    int c_N = c_res.getSize();
    c_assembler.add_jacobian(c_res, c_kmat);
    c_assembler.apply_bcs(c_kmat);

    auto c_smoother = new Smoother(cublasHandle, cusparseHandle, c_assembler, c_kmat, c_h_color_rowp, 1.0);
    auto c_prolongation = new Prolongation(c_assembler);
    auto c_grid = GRID(c_assembler, c_prolongation, c_smoother, c_kmat, coarse_loads, cublasHandle, cusparseHandle);


    /* 4) assemble initial prolongation matrix from coarse to fine (for structured prolong) */

    
    // BSR nodal nonzero pattern first (in multicolored order)
    int nnodes_coarse = c_assembler.get_num_nodes();
    int nnodes_fine = asesmbler.get_num_nodes();
    int *h_f_iperm = DeviceVec<int>(nnodes_fine, f_grid->d_iperm).createHostVec().getPtr();
    int *h_c_iperm = DeviceVec<int>(nnodes_coarse, c_grid->d_iperm).createHostVec().getPtr();    
    int *h_f_perm = DeviceVec<int>(nnodes_fine, f_grid->d_perm).createHostVec().getPtr();
    int *h_c_perm = DeviceVec<int>(nnodes_coarse, c_grid->d_perm).createHostVec().getPtr();    
    int *P_rowp = new int[nnodes_fine + 1];
    memset(rowp, 0.0, (nnodes_fine + 1) * sizeof(int));
    int nx_f = nxe + 1, nx_c = nxe/2 + 1;
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f / nx_f, iy = inode_f % nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        for (int ixc = ix_c0 - 1; ixc < ix_c0 + 1; ixc++) {
            for (int iyc = iy_c0 - 1; iyc < iy_c0 + 1; iyc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int adj = case1 || case2 || case3;
                if (adj) {
                    P_rowp[perm_inodef]++;
                }
            }
        }
        // update rowp
        P_rowp[perm_inodef + 1] = P_rowp[perm_inodef];
    }
    // then compute the cols sparsity now
    int P_nnzb = P_rowp[nnodes_fine];
    int inz = 0;
    int *P_cols = new int[P_nnzb];
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f / nx_f, iy = inode_f % nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        std::vector<int> c_cols;
        for (int iyc = iy_c0 - 1; iyc < iy_c0 + 1; iyc++) {
            for (int ixc = ix_c0 - 1; ixc < ix_c0 + 1; ixc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int adj = case1 || case2 || case3;
                if (adj) {
                    int icoarse = nx_c * iy_c + ix_c;
                    int perm_icoarse = h_c_iperm[icoarse];
                    c_cols.push_back(perm_icoarse);
                }
            }
        }
        std::sort(&c_cols[0], &c_cols[c_cols.size()]);
        for (int ic = 0; ic < c_cols.size(); ic++) {
            P_cols[perm_inodef] = c_cols[ic];
            inz++;
        }
    }
    // now compute values of P on host (full BSR form)
    T *P_vals = new T[36 * P_nnzb];
    memset(P_vals, 0.0, 36 * P_nnzb);
    inz = 0; // reset nz counter again
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f / nx_f, iy = inode_f % nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        for (int iyc = iy_c0 - 1; iyc < iy_c0 + 1; iyc++) {
            for (int ixc = ix_c0 - 1; ixc < ix_c0 + 1; ixc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int adj = case1 || case2 || case3;
                if (adj) {
                    int icoarse = nx_c * iy_c + ix_c;
                    int perm_icoarse = h_c_iperm[icoarse];
                    T scale = 1.0;
                    if (case2) scale = 0.5;
                    if (case3) scale = 0.25;
                    for (int ib = 0; ib < 6; ib++) {
                        // set diag of 6x6 block matrix here
                        P_vals[36 * inz + 6 * ib + ib] = scale;
                        inz++;
                    }
                }
            }
        }
    }
    // TODO : apply bcs to P_mat also?
    // partition of unity normalize the P_vals
    for (int brow = 0; brow < nnodes_fine; brow++) {
        T total_scale = 0.0;
        for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
            total_scale += P_vals[36 * jp]; // (0,0) entry of a block
        }
        // normalize by this now
        for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
            for (int ib = 0; ib < 6; ib++) {
                P_vals[36 * jp + 6 * ib + 6] /= total_scale;
            }
        }
    }
    // copy the Pmat onto the device now
    int *d_P_rowp = HostVec<int>(nnodes_fine + 1, P_rowp).createDeviceVec().getPtr();
    int *d_P_cols = HostVec<int>(P_nnzb, P_cols).createDeviceVec().getPtr();
    T *d_P_vals = HostVec<T>(36 * P_nnzb, P_vals).createDeviceVec().getPtr();
    int P_mb = nnodes_fine, P_nb = nnodes_coarse;

    // TODO : test out on a color-ordered vector to check see if P_mat is reasonable..
    


    /* 5) build Dinv matrix for GS smoother using ILU factorization (copied from MC-smooth code) */

    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    DeviceVec<T> d_diag_vals;
    T *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;
    bool build_lu_inv_operator;
    int *d_kmat_diagp;
    BsrData d_diag_bsr_data;
    DeviceVec<T> d_dinv_vals;
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M =
        CUSPARSE_SOLVE_POLICY_USE_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;
    int *h_diag_rowp = new int[nnodes + 1];
    diag_inv_nnzb = nnodes;
    int *h_diag_cols = new int[nnodes];
    h_diag_rowp[0] = 0;
    for (int i = 0; i < nnodes; i++) {
        h_diag_rowp[i + 1] = i + 1;
        h_diag_cols[i] = i;
    }
    // on host, get the pointer locations in Kmat of the block diag entries..
    int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
    int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();
    // now copy to device
    d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
    d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();
    // create the bsr data object on device
    d_diag_bsr_data = BsrData(nnodes, 6, diag_inv_nnzb, d_diag_rowp, d_diag_cols, nullptr,
                                nullptr, false);
    delete[] h_diag_rowp;
    delete[] h_diag_cols;
    // now allocate DeviceVec for the values
    int ndiag_vals = block_dim * block_dim * nnodes;
    d_diag_vals = DeviceVec<T>(ndiag_vals);
    d_diag_LU_vals = d_diag_vals.getPtr();  // just copy these pointers..
    int *h_kmat_diagp = new int[nnodes];
    for (int block_row = 0; block_row < nnodes; block_row++) {
        for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
            int block_col = h_kmat_cols[jp];
            // printf("row %d, col %d\n", block_row, block_col);
            if (block_row == block_col) {
                h_kmat_diagp[block_row] = jp;
            }
        }
    }
    d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();
    delete[] h_kmat_rowp;
    delete[] h_kmat_cols;
    // call the kernel to copy out diag vals first
    int ndiag_vals = block_dim * block_dim * nnodes;
    dim3 block(32);
    int nblocks = (ndiag_vals + 31) / 32;
    dim3 grid(nblocks);
    k_copyBlockDiagFromBsrMat<T>
        <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                         &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // build Dinv diag matrix so I can just do multiplies (with my own kernels)
    cusparseMatDescr_t descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
    CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

    auto d_dinv_vals = DeviceVec<T>(ndiag_vals);

    // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
    // to get effective matrix.. need six temp vectors..
    for (int i = 0; i < block_dim; i++) {
        // set d_temp to ei (one of e1 through e6 per block)
        cudaMemset(d_temp, 0.0, N * sizeof(T));
        dim3 block(32);
        dim3 grid((nnodes + 31) / 32);
        k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

        // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_diag_LU_vals,
            d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
            pBuffer));  // prob only need U^-1 part for block diag.. TBD

        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_diag_LU_vals,
            d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

        // now copy temp2 into columns of new operator
        dim3 grid2((N + 31) / 32);
        k_setLUinv_operator<T>
            <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vals.getPtr());
    }

    if constexpr (startup) {
        D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_dinv_vals);
    }

    /* 6) compute the coarse mesh rigid body/nullspace modes of the plate (from its xpts),
         should be easy I did it in python */
    

    /* 7) prolong smoothing loop */
    int nsmooth = 1;
    // int nsmooth = 4;
    for (int ismooth = 0; ismooth < nsmooth; ismooth++) {

        /* 7.1) compute Kmat * P to get defect matrix (no fillin first) */
        // could maybe do it in place, but not worried about extra mem or inefficiencies rn



        /* 7.2) apply smoother using custom submat transpose product kernels */

        for (int icolor = 0; icolor < num_colors; icolor++) {
            // bounding nodes for this color set
            int start_node = h_color_rowp[icolor], end_node = h_color_rowp[icolor+1];
            int ncolor_nodes = end_node - start_node;
            block(32);
            grid((ncolor_nodes + 31) / 32);
            // uses d_P_defect_vals to update the d_P_vals
            k_applyDinv_to_prolong<<<grid, block>>>(start_node, end_node, d_dinv_vals, d_P_defect_vals, d_P_vals);

            

        }


        /* 7.3) apply orthogonal projector with the rigid body modes matrix */


    }

    


    /* 8) verification, compare smoothed prolong matrix to original prolong matrix on a vec (with standard smoothing) */



    return 0;
}