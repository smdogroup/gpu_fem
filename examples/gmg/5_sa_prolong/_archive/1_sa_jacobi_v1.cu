/* demo smoothed aggregation GMG on a plate */

// based on the paper, \href{https://link.springer.com/article/10.1007/s006070050022}{Energy Optimization of Algebraic Multigrid Bases}
// goal here is to do small plate first, explicitly forming and modifying the prolong matrix
// the smoother Dinv and mat-mat products are computed slow version here using incomplete LU (not fast one like the fast + optimized GSMC smoother in multigrid/smoothers/mc_smooth1.h)
// will performance optimize later if the process is right

// some temp source code will come from _src.h

#include <vector>
#include <unordered_set>
#include <set>

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

// helper kernels
#include "_sa.cuh"

template <typename T>
T get_max_disp(DeviceVec<T> &d_soln, int idof = 2) {
    T *h_soln = d_soln.createHostVec().getPtr();
    int nvars = d_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}


int main() {

    /* 1) type definitions for this shell element */
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // some problem specific user inputs
    // int nxe = 64; // two-grid problem with second grid the coarser one
    // int nxe = 32;
    int nxe = 16;
    // int nxe = 8;
    // int nxe = 4;

    // double SR = 1.0;
    double SR = 10.0;

    printf("1) create handles\n");
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    /* 2) create the fine grid assembler, multicolor reordering  */
    
    printf("2.1) create fine assembler\n");
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 2, nye_per_comp = nxe/2; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *fine_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q);
    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();
    auto loads = assembler.createVarsVec(fine_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    int N = res.getSize();
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    printf("2.2) create fine smoother, prolong + grid\n");
    auto f_smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, 1.0);
    auto f_prolongation = new Prolongation(assembler);
    auto f_grid = new GRID(assembler, f_prolongation, f_smoother, kmat, loads, cublasHandle, cusparseHandle);


    /* 3) create the coarse grid assembler, full LU + AMD reordering  */
    
    printf("3.1) create coarse assembler\n");
    nxe_per_comp = nxe / 2, nye_per_comp = nxe / 2; // for now (should have 25 grids)
    auto c_assembler = createPlateAssembler<Assembler>(nxe / 2, nxe / 2, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    T *coarse_loads = getPlateLoads<T, Physics>(nxe / 2, nxe / 2, Lx, Ly, Q);
    auto &c_bsr_data = c_assembler.getBsrData();
    int c_num_colors = 1, *_c_color_rowp = new int[2];
    _c_color_rowp[0] = 0, _c_color_rowp[1] = c_assembler.get_num_nodes() + 1;
    c_bsr_data.AMD_reordering();
    c_bsr_data.compute_full_LU_pattern(10.0, false);
    auto c_h_color_rowp = HostVec<int>(c_num_colors + 1, _c_color_rowp);
    c_assembler.moveBsrDataToDevice();
    auto c_loads = c_assembler.createVarsVec(coarse_loads);
    c_assembler.apply_bcs(c_loads);
    auto c_kmat = createBsrMat<Assembler, VecType<T>>(c_assembler);
    auto c_res = c_assembler.createVarsVec();
    auto c_soln = c_assembler.createVarsVec();
    int c_N = c_res.getSize();
    c_assembler.add_jacobian_fast(c_kmat);
    c_assembler.apply_bcs(c_kmat);
    printf("3.2) create coarse smoother, prolong + grid\n");
    auto c_smoother = new Smoother(cublasHandle, cusparseHandle, c_assembler, c_kmat, c_h_color_rowp, 1.0);
    auto c_prolongation = new Prolongation(c_assembler);
    auto c_grid = new GRID(c_assembler, c_prolongation, c_smoother, c_kmat, c_loads, cublasHandle, cusparseHandle);


    /* 4) assemble initial prolongation matrix from coarse to fine (for structured prolong) */

    printf("4.1) fine prolong init coarse data\n");
    f_prolongation->init_coarse_data(c_assembler); // init baseline unsmooth prolongator (just for comparison)
    // the smoothed one we will just build here locally
    
    // BSR nodal nonzero pattern first (in multicolored order)
    printf("4.2) create fine prolong mat to be smoothed on host\n");
    int nnodes_coarse = c_assembler.get_num_nodes();
    int nnodes_fine = assembler.get_num_nodes();
    int *h_f_iperm = DeviceVec<int>(nnodes_fine, f_grid->d_iperm).createHostVec().getPtr();
    int *h_c_iperm = DeviceVec<int>(nnodes_coarse, c_grid->d_iperm).createHostVec().getPtr();    
    int *h_f_perm = DeviceVec<int>(nnodes_fine, f_grid->d_perm).createHostVec().getPtr();
    int *h_c_perm = DeviceVec<int>(nnodes_coarse, c_grid->d_perm).createHostVec().getPtr();    
    int *P_rowp = new int[nnodes_fine + 1];
    memset(P_rowp, 0.0, (nnodes_fine + 1) * sizeof(int));
    int nx_f = nxe + 1, nx_c = nxe/2 + 1;
    printf("4.2.1) create P_rowp\n");
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f % nx_f, iy = inode_f / nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        // printf("fine node (%d,%d) => start coarse node (%d,%d)\n", ix, iy, ix_c0, iy_c0);
        P_rowp[perm_inodef + 1] = P_rowp[perm_inodef];
        for (int ixc = ix_c0 - 1; ixc < ix_c0 + 2; ixc++) {
            for (int iyc = iy_c0 - 1; iyc < iy_c0 + 2; iyc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int in_bounds = (0 <= ixc) && (ixc < nx_c) && (0 <= iyc) && (iyc < nx_c);
                int adj = (case1 || case2 || case3) && in_bounds;
                if (adj) {
                    // printf("fine node (%d,%d) adj to coarse node (%d,%d)\n", ix, iy, ixc, iyc);
                    P_rowp[perm_inodef+1]++;
                } else {
                    // printf("fine node (%d,%d) is NOT adj to coarse node (%d,%d)\n", ix, iy, ixc, iyc);
                }
            }
        }
    }
    // then compute the cols sparsity now
    int P_nnzb = P_rowp[nnodes_fine];
    int inz = 0;
    int *P_cols = new int[P_nnzb];
    int *P_rows = new int[P_nnzb];
    printf("4.2.2) create P_cols\n");
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f % nx_f, iy = inode_f / nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        std::vector<int> c_cols;
        for (int iyc = iy_c0 - 1; iyc < iy_c0 + 2; iyc++) {
            for (int ixc = ix_c0 - 1; ixc < ix_c0 + 2; ixc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int in_bounds = (0 <= ixc) && (ixc < nx_c) && (0 <= iyc) && (iyc < nx_c);
                int adj = (case1 || case2 || case3) && in_bounds;
                if (adj) {
                    int icoarse = nx_c * iyc + ixc;
                    int perm_icoarse = h_c_iperm[icoarse];
                    c_cols.push_back(perm_icoarse);
                }
            }
        }
        std::sort(&c_cols[0], &c_cols[c_cols.size()]);
        for (int ic = 0; ic < c_cols.size(); ic++) {
            P_cols[inz] = c_cols[ic];
            P_rows[inz] = perm_inodef;
            inz++;
        }
    }
    printf("here\n");
    printf("P_rowp: ");
    printVec<int>(min(50, nnodes_fine + 1), P_rowp);
    printf("P_cols: ");
    printVec<int>(min(50, P_nnzb), P_cols);
    // now compute values of P on host (full BSR form)
    T *P_vals = new T[36 * P_nnzb];
    memset(P_vals, 0.0, 36 * P_nnzb);
    inz = 0; // reset nz counter again
    printf("4.2.3) create P_vals on host\n");
    for (int perm_inodef = 0; perm_inodef < nnodes_fine; perm_inodef++) {
        int inode_f = h_f_perm[perm_inodef]; // convert out of colored perm order to vis order
        int ix = inode_f % nx_f, iy = inode_f / nx_f;
        int ix_c0 = ix / 2, iy_c0 = iy / 2; // loop over nearby coarse nodes +-1 each side
        for (int iyc = iy_c0 - 1; iyc < iy_c0 + 2; iyc++) {
            for (int ixc = ix_c0 - 1; ixc < ix_c0 + 2; ixc++) {
                // compoute equiv fine node of each coarse node
                int ix2 = 2 * ixc, iy2 = 2 * iyc;
                // check adjacency with the orig fine node
                int dx = abs(ix2 - ix), dy = abs(iy2 - iy);
                int case1 = dx == 0 && dy == 0; // fine node matches coarse
                int case2 = (dx == 1 && dy == 0) || (dx == 0 && dy == 1); // fine node on edge
                int case3 = dx == 1 && dy == 1; // fine node in center of coarse elem
                int in_bounds = (0 <=ixc) && (ixc < nx_c) && (0 <= iyc) && (iyc < nx_c);
                int adj = (case1 || case2 || case3) && in_bounds;
                if (adj) {
                    int icoarse = nx_c * iyc + ixc;
                    int perm_icoarse = h_c_iperm[icoarse];
                    T scale = 1.0;
                    if (case2) scale = 0.5;
                    if (case3) scale = 0.25;
                    for (int ib = 0; ib < 6; ib++) {
                        // set diag of 6x6 block matrix here
                        P_vals[36 * inz + 6 * ib + ib] += scale;
                    }
                    inz++;
                }
            }
        }
    }
    // TODO : apply bcs to P_mat also?
    // partition of unity normalize the P_vals
    printf("4.2.3) partition of unity operation on P_vals host\n");
    for (int brow = 0; brow < nnodes_fine; brow++) {
        T total_scale = 0.0;
        for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
            total_scale += P_vals[36 * jp]; // (0,0) entry of a block
        }
        // normalize by this now
        for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
            for (int ib = 0; ib < 6; ib++) {
                P_vals[36 * jp + 6 * ib + ib] /= total_scale;
            }
        }
    }
    // copy the Pmat onto the device now
    printf("4.2.4) copy P_vals onto device\n");
    int *d_P_rowp = HostVec<int>(nnodes_fine + 1, P_rowp).createDeviceVec().getPtr();
    int *d_P_cols = HostVec<int>(P_nnzb, P_cols).createDeviceVec().getPtr();
    int *d_P_rows = HostVec<int>(P_nnzb, P_rows).createDeviceVec().getPtr();
    T *d_P_vals0 = HostVec<T>(36 * P_nnzb, P_vals).createDeviceVec().getPtr(); // nofill sparsity
    int P_mb = nnodes_fine, P_nb = nnodes_coarse;
    // set up for mat-mult
    cusparseMatDescr_t descr_Pmat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_Pmat));
    CHECK_CUSPARSE(cusparseSetMatType(descr_Pmat, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_Pmat, CUSPARSE_INDEX_BASE_ZERO));


    // 4.1 - : test out on a color-ordered vector to check see if P_mat is reasonable..
    // first run a coarse grid direct solve with loads permuted to solve order (Not vis order)
    printf("4.3) compare local Pmat to orig structured prolong\n");
    bool permute_inout = false; // so that I have to do one less perm step for c_soln to be in solve order
    CUSPARSE::direct_LU_solve(c_kmat, c_grid->d_rhs, c_soln, true, permute_inout);
    c_soln.permuteData(6, c_grid->d_perm); // permute solve to vis order
    auto h_c_soln = c_soln.createHostVec(); 
    c_soln.permuteData(6, c_grid->d_iperm); // permute back to solve order
    printToVTK<Assembler,HostVec<T>>(c_assembler, h_c_soln, "out/_plate_coarse.vtk");
    // compare to coarse-fine from the structured plate prolongator
    soln.zeroValues();
    f_prolongation->prolongate(c_soln, soln);
    soln.permuteData(6, f_grid->d_perm);
    auto h_f_soln2 = soln.createHostVec(); // permute solve to vis order
    printToVTK<Assembler,HostVec<T>>(assembler, h_f_soln2, "out/_plate_cf_orig.vtk");
    // try running permute on the coarse solution to fine with P_mat built here
    T a = 1.0, b = 0.0;
    int block_dim = 6;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        P_mb, P_nb, P_nnzb, &a, descr_Pmat, d_P_vals0, d_P_rowp, d_P_cols, block_dim, 
        c_soln.getPtr(), &b, soln.getPtr()));
    soln.permuteData(6, f_grid->d_perm);
    auto h_f_soln = soln.createHostVec(); // permute solve to vis order
    printToVTK<Assembler,HostVec<T>>(assembler, h_f_soln, "out/_plate_cf_new.vtk");
    

    // printf("stop here after P_mat verification for step 1 debug\n");
    // return 0;


    /* 5) build Dinv matrix for GS smoother using ILU factorization (copied from MC-smooth code) */
    printf("5.1) build Dinv matrix - allocate sparsity\n");
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    DeviceVec<T> d_diag_vals;
    T *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;
    bool build_lu_inv_operator;
    int *d_kmat_diagp;
    BsrData d_diag_bsr_data;
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
    int nnodes = nnodes_fine;
    int *h_diag_rowp = new int[nnodes + 1];
    diag_inv_nnzb = nnodes;
    int *h_diag_cols = new int[nnodes];
    h_diag_rowp[0] = 0;
    for (int i = 0; i < nnodes; i++) {
        h_diag_rowp[i + 1] = i + 1;
        h_diag_cols[i] = i;
    }
    // on host, get the pointer locations in Kmat of the block diag entries..
    int *d_kmat_rowp = bsr_data.rowp;
    int *d_kmat_cols = bsr_data.cols;
    int kmat_nnzb = bsr_data.nnzb;
    T *d_kmat_vals = kmat.getPtr();
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
    // delete[] h_kmat_rowp;
    // delete[] h_kmat_cols;
    // call the kernel to copy out diag vals first
    printf("5.2) build Dinv matrix - copy values\n");
    dim3 block(32);
    int nblocks = (ndiag_vals + 31) / 32;
    dim3 grid(nblocks);
    k_copyBlockDiagFromBsrMat<T>
        <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);
    printf("5.3) build Dinv matrix - LU factorization of Dinv diag\n");
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                         &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // build Dinv diag matrix so I can just do multiplies (with my own kernels)
    cusparseMatDescr_t descrDinvMat = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
    CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));

    auto d_dinv_vals = DeviceVec<T>(ndiag_vals);
    T *d_temp = DeviceVec<T>(N).getPtr();
    T *d_temp2 = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();

    // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
    // to get effective matrix.. need six temp vectors..
    printf("5.4) build Dinv matrix - compute Dinv linear operator of LU factor\n");
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

    auto D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_dinv_vals);

    /* 6) compute the coarse mesh rigid body/nullspace modes of the plate (from its xpts),
         should be easy I did it in python */

    // compute on the host first
    printf("6) compute the coarse mesh rigid body modes\n");
    T *xpts_coarse = c_assembler.getXpts().createHostVec().getPtr();
    T *Bc = new T[36 * nnodes_coarse]; // get it in solve order so need h_c_perm
    memset(Bc, 0.0, 36 * nnodes_coarse);
    for (int i = 0; i < 6 * nnodes_coarse; i++) {
        int ic = i / 6, ii = i % 6;
        int pic = h_c_iperm[ic];
        T x = xpts_coarse[3 * ic], y = xpts_coarse[3 * ic + 1], z = xpts_coarse[3 * ic + 2];
        Bc[36 * pic + 6 * 0 + 0] = 1.0; // u translation
        Bc[36 * pic + 6 * 1 + 1] = 1.0; // v translation
        Bc[36 * pic + 6 * 2 + 2] = 1.0; // w translation
        // thx rotation
        Bc[36 * pic + 6 * 1 + 3] = -z;
        Bc[36 * pic + 6 * 2 + 3] = y;
        Bc[36 * pic + 6 * 3 + 3] = 1.0;        
        // thy rotatoin
        Bc[36 * pic + 6 * 0 + 4] = z;
        Bc[36 * pic + 6 * 2 + 4] = -x;
        Bc[36 * pic + 6 * 4 + 4] = 1.0;
        // thz rotation
        Bc[36 * pic + 6 * 0 + 5] = -y;
        Bc[36 * pic + 6 * 1 + 5] = x;
        Bc[36 * pic + 6 * 5 + 5] = 1.0;
    }
    // move to device
    T *d_Bc = HostVec<T>(36 * nnodes_coarse, Bc).createDeviceVec().getPtr();
    // then compute the fine node BC indicator matrix Fi later?


    /* 7) prolong smoothing loop */
    printf("7.1.1) prolong smoothing - PKP mat-mat product nz pattern\n");
    bool compute_PF_fillin = true;
    // bool compute_PF_fillin = false;
    int PF_nnzb, *PF_rowp, *d_PF_rowp, *PF_rows, *d_PF_rows, *PF_cols, *d_PF_cols;
    if (compute_PF_fillin) {
        // compute the nz pattern of PF = -K*P sparsity
        //   first get the rowp of PF
        //   note there may be more efficient ways of getting fillin, not worried about that rn (just demo script)
        PF_rowp = new int[nnodes_fine + 1];
        memset(PF_rowp, 0, (nnodes_fine + 1) * sizeof(int));
        for (int i = 0; i < nnodes_fine; i++) {
            PF_rowp[i+1] = PF_rowp[i];
            std::unordered_set<int> unique_cols;
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];
                for (int jp = P_rowp[k]; jp < P_rowp[k+1]; jp++) {
                    int j = P_cols[jp];
                    unique_cols.insert(j);
                }
            }
            PF_rowp[i+1] += unique_cols.size();
        }
        printf("PF_rowp: ");
        printVec<int>(50, PF_rowp);
        PF_nnzb = PF_rowp[nnodes_fine];
        PF_cols = new int[PF_nnzb];
        PF_rows = new int[PF_nnzb];
        int iinz = 0;
        for (int i = 0; i < nnodes_fine; i++) {
            std::set<int> ordered_unique_cols;
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];
                for (int jp = P_rowp[k]; jp < P_rowp[k+1]; jp++) {
                    int j = P_cols[jp];
                    ordered_unique_cols.insert(j);
                }
            }
            for (int col : ordered_unique_cols) {
                PF_cols[iinz] = col;
                PF_rows[iinz] = i;
                iinz++;
            }
        }
        printf("PF_cols: ");
        printVec<int>(50, PF_cols);
        d_PF_rowp = HostVec<int>(nnodes_fine + 1, PF_rowp).createDeviceVec().getPtr();
        d_PF_rows = HostVec<int>(PF_nnzb, PF_rows).createDeviceVec().getPtr();
        d_PF_cols = HostVec<int>(PF_nnzb, PF_cols).createDeviceVec().getPtr();
    } else {
        // not doing PF fillin as one option
        PF_nnzb = P_nnzb;
        PF_rowp = P_rowp, PF_cols = P_cols;
        PF_rows = P_rows;
        d_PF_rowp = d_P_rowp, d_PF_cols = d_P_cols;
        d_PF_rows = d_P_rows;
    }
    // now move rowp, cols to the device
    T *d_PF_vals = DeviceVec<T>(36 * PF_nnzb).getPtr();
    // make P filled in too, so now copy it's data into new sparsity
    // by computing copy block locations first on the host
    T *d_P_vals = DeviceVec<T>(36 * PF_nnzb).getPtr();
    int *h_P_fill_map = new int[P_nnzb]; // maps of matching copy block locations
    for (int i = 0; i < nnodes_fine; i++) {
        for (int jp = P_rowp[i]; jp < P_rowp[i+1]; jp++) {
            int j = P_cols[jp];
            for (int jp2 = PF_rowp[i]; jp2 < PF_rowp[i+1]; jp2++) {
                int j2 = PF_cols[jp2];
                if (j == j2) {
                    h_P_fill_map[jp] = jp2;
                }
            }
        }
    }
    // move it to the device
    int *d_P_fill_map = HostVec<int>(P_nnzb, h_P_fill_map).createDeviceVec().getPtr();
    dim3 block0(64);
    dim3 grid0(P_nnzb);
    k_copy_P_to_fillP<T><<<grid0, block0>>>(P_nnzb, block_dim, d_P_fill_map, d_P_vals0, d_P_vals);
    // DEBUG, check the cf with filled in P still works
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        P_mb, P_nb, PF_nnzb, &a, descr_Pmat, d_P_vals, d_PF_rowp, d_PF_cols, block_dim, 
        c_soln.getPtr(), &b, soln.getPtr()));
    soln.permuteData(6, f_grid->d_perm);
    auto h_f_soln3 = soln.createHostVec(); // permute solve to vis order
    printToVTK<Assembler,HostVec<T>>(assembler, h_f_soln3, "out/_plate_cf_new_fill.vtk");
    // compute difference to the original
    T *_hv1 = h_f_soln2.getPtr();
    T *_hv2 = h_f_soln3.getPtr();
    T *_dh_cf = new T[h_f_soln2.getSize()];
    memset(_dh_cf, 0.0, N * sizeof(T));
    T abs_err = 0.0;
    T abs_disp = 0.0;
    int err_dof = 0;
    for (int i = 0; i < N; i++) {
        _dh_cf[i] = _hv2[i] - _hv1[i];
        if (abs(_dh_cf[i]) > abs_err) {
            abs_err = abs(_dh_cf[i]);
            err_dof = i;
        }
        abs_disp = max(abs_disp, abs(_hv1[i]));
    }
    printf("abs err of new CF %.8e / %.8e at DOF %d or node %d\n", abs_err, abs_disp, err_dof, err_dof/6);
    // return 0;


    // allocate some data first
    printf("PF_rowp (DEBUG) with PF_nnzb = %d: ", PF_nnzb);
    printVec<int>(50, PF_rowp);
    // compute the initial P_F = -K*P matrix as defect matrix (no fillin)
    // compute the block indices in each matrix of each block-product, using 3 arrays where in P_F, K and P
    // first get how many nz block-products (no fillin)
    int nnzb_prod = 0;
    for (int i = 0; i < nnodes_fine; i++) {
        for (int jp = PF_rowp[i]; jp < PF_rowp[i+1]; jp++) {
            int j = PF_cols[jp]; // (P_F)_{ij} output
            // now inner loop k for K_{ik} * P_{kj}
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];

                // check P_{kj} nz
                bool nz_Pkj = false; // now also use PF = -K*P sparsity for P cause we add K*P fillin (for better prolong)
                for (int jp2 = PF_rowp[k]; jp2 < PF_rowp[k+1]; jp2++) {
                    int j2 = PF_cols[jp2];
                    if (j2 == j) {
                        nz_Pkj = true;
                    }
                }
                if (!nz_Pkj) continue;
                // otherwise, we do have a valid nz product here
                nnzb_prod++;
            }
        }
    }
    // now allocate the block indices of the product
    int *h_PF_blocks = new int[nnzb_prod];
    int *h_K_blocks = new int[nnzb_prod];
    int *h_P_blocks = new int[nnzb_prod];
    memset(h_PF_blocks, 0, nnzb_prod * sizeof(int));
    memset(h_K_blocks, 0, nnzb_prod * sizeof(int));
    memset(h_P_blocks, 0, nnzb_prod * sizeof(int));
    int inz_prod = 0;
    printf("7.1.2) prolong smoothing - PKP mat-mat product cols\n");
    for (int i = 0; i < nnodes_fine; i++) {
        for (int jp = PF_rowp[i]; jp < PF_rowp[i+1]; jp++) {
            int j = PF_cols[jp]; // (P_F)_{ij} output
            // now inner loop k for K_{ik} * P_{kj}
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];

                // check P_{kj} nz
                bool nz_Pkj = false;
                int _jp2 = -1;
                for (int jp2 = PF_rowp[k]; jp2 < PF_rowp[k+1]; jp2++) {
                    int j2 = PF_cols[jp2];
                    if (j2 == j) {
                        nz_Pkj = true;
                        _jp2 = jp2;
                    }
                }
                if (!nz_Pkj) continue;
                // otherwise, we do have a valid nz product here
                h_PF_blocks[inz_prod] = jp;
                h_K_blocks[inz_prod] = kp;
                h_P_blocks[inz_prod] = _jp2;
                inz_prod++;
            }
        }
    }
    printf("KP => PF mat-mat product has %d nnzb_prod\n", nnzb_prod);
    printf("h_P_blocks: ");
    printVec<int>(50, h_P_blocks);
    printf("h_K_blocks: ");
    printVec<int>(50, h_K_blocks);
    printf("h_PF_blocks: ");
    printVec<int>(50, h_PF_blocks);
    // now allocate onto the device
    int *d_PF_blocks = HostVec<int>(nnzb_prod, h_PF_blocks).createDeviceVec().getPtr();
    int *d_K_blocks = HostVec<int>(nnzb_prod, h_K_blocks).createDeviceVec().getPtr();
    int *d_P_blocks = HostVec<int>(nnzb_prod, h_P_blocks).createDeviceVec().getPtr();
    // define Kmat for Kmat SpMV stuff for products
    cusparseMatDescr_t descr_Kmat = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_Kmat));
    CHECK_CUSPARSE(cusparseSetMatType(descr_Kmat, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_Kmat, CUSPARSE_INDEX_BASE_ZERO));

    // verify the PKP product first (DEBUG) against K * P of the actual solution..
    // has no fillin, but hopefully should still be somewhat close
    a = -1.0; // compute PF = -K * P matrix-matrix
    printf("7.1.3) prolong smoothing - PKP mat-mat product, try demo here\n");
    dim3 PKP_block(216);
    dim3 PKP_grid(nnzb_prod);
    cudaMemset(d_PF_vals, 0.0, PF_nnzb * 36 * sizeof(T)); // zero d_PF_vals
    k_compute_P_K_P_mmprod<T><<<PKP_grid, PKP_block>>>(nnzb_prod, block_dim, a, d_K_blocks, 
        d_P_blocks, d_PF_blocks, d_kmat_vals, d_P_vals, d_PF_vals);
    //      now compare -K*P*v of orig smoother to PF * v w/ PF matrix here
    //      should be slightly different cause PF has no fillin.. but hopefully not way off?
    // take coarse solution c_soln and compute w = P*v again
    printf("7.1.4) prolong smoothing - PKP mat-mat; orig -P*K*v\n");
    soln.zeroValues();
    f_prolongation->prolongate(c_soln, soln);
    // then compute tmp = 0 * tmp + -1 * K * w  (aka gives us -K*w = -K*P*c_soln)
    a = -1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        nnodes_fine, nnodes_fine, kmat_nnzb, &a, descr_Kmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols, block_dim, 
        soln.getPtr(), &b, d_temp));
    auto d_temp_vec = DeviceVec<T>(N, d_temp);
    // get initial defect disp
    T def_nrm0 = get_max_disp<T>(d_temp_vec);
    d_temp_vec.permuteData(6, f_grid->d_perm); // permute solve to vis order
    auto h_KPv1 = d_temp_vec.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_KPv1, "out/_plate_nKPv_orig.vtk");
    // now compute -KPv through PF*v matrix operator now
    printf("7.1.5) prolong smoothing - PKP mat-mat; new -P*K*v = PF*v prod\n");
    a = 1.0, b = 0.0; // v = 0*v + PF*v
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        nnodes_fine, nnodes_coarse, PF_nnzb, &a, descr_Pmat, d_PF_vals, d_PF_rowp, d_PF_cols, block_dim, 
        c_soln.getPtr(), &b, d_temp));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tdone with PF*v prod\n");
    auto d_temp_vec2 = DeviceVec<T>(N, d_temp);
    d_temp_vec2.permuteData(6, f_grid->d_perm);
    auto h_KPv2 = d_temp_vec2.createHostVec(); 
    printToVTK<Assembler,HostVec<T>>(assembler, h_KPv2, "out/_plate_nKPv_new.vtk");
    // compare vectors in the -K*P*v step, where the error is.. (VERIFICATION / DEBUG)
    _hv1 = h_KPv1.getPtr();
    _hv2 = h_KPv2.getPtr();
    memset(_dh_cf, 0.0, N * sizeof(T));
    abs_err = 0.0;
    abs_disp = 0.0;
    err_dof = 0;
    for (int i = 0; i < N; i++) {
        _dh_cf[i] = _hv2[i] - _hv1[i];
        if (abs(_dh_cf[i]) > abs_err) {
            abs_err = abs(_dh_cf[i]);
            err_dof = i;
        }
        abs_disp = max(abs_disp, abs(_hv1[i]));
    }
    printf("abs err of -K*P*v step %.8e / %.8e at DOF %d or node %d\n", abs_err, abs_disp, err_dof, err_dof/6);
    // return 0;

    /* begin smoothing phase of the P matrix! */
    printf("h_color_rowp: ");
    printVec<int>(num_colors + 1, h_color_rowp.getPtr());
    printf("\twith nnodes_fine = %d\n", nnodes_fine);
    // printf("PF_rowp[4224-on]: ");
    // prinVec<int>(2, PF_rowp)
    auto d_temp2_vec = DeviceVec<T>(N, d_temp2);

    // do pre partition of unity normalization.. (so initially valid, is initially valid if using |cdot|_1 norms or row abs sums)
    // dim3 norm_block(32);
    // dim3 norm_grid(nnodes);
    // // printf("\tnormalize rows\n");
    // // int max_inner_row = 3; // just normalize u,v,w rows
    // int max_inner_row = 6;
    // k_normalize_rows<T><<<norm_grid, norm_block>>>(nnodes, block_dim, max_inner_row, d_PF_rowp, d_P_vals);

    // int nsmooth = 1;
    // int nsmooth = 2;
    // int nsmooth = 4;
    // int nsmooth = 10;
    // int nsmooth = 20;
    // int nsmooth = 40;
    int nsmooth = 100;
    // int nsmooth = 300;

    // omegaMC < 2/rho(Dinv*A) is what it needs to be (may need to estimate from Arnoldi hessenberg iteration, the max eigenvalue)
    // T omegaMC = 5e-3;
    T omegaMC = 0.01;
    // T omegaMC = 0.1;
    // T omegaMC = 0.4;
    // T omegaMC = 0.7; // smoother constant
    // T omegaMC = 1.5;

    // for debugging
    // bool write_vtk = true;
    bool write_vtk = false;

    // do orthog projection of near kernel modes
    

    // do normalization of rows
    bool norm_rows = true;
    // bool norm_rows = false;

    printf("Jacobi inner-prod space energy min: init_defect %.8e\n", def_nrm0);

    int itest = 0;
    for (int ismooth = 0; ismooth < nsmooth; ismooth++) {
    
        /* 7.2) compute Kmat * P to get defect matrix (no fillin first) */
        // could maybe do it in place, but not worried about extra mem or inefficiencies rn
        
        // compute Kmat * P => P_defect (could do up front and just update as normal, 
        // but not gonna do the more efficient way yet)
        cudaMemset(d_PF_vals, 0.0, PF_nnzb * 36 * sizeof(T));
        a = -1.0; // compute PF = -K * P matrix-matrix
        k_compute_P_K_P_mmprod<T><<<PKP_grid, PKP_block>>>(nnzb_prod, block_dim, a, d_K_blocks, 
            d_P_blocks, d_PF_blocks, d_kmat_vals, d_P_vals, d_PF_vals);


        // NOW just doing block-jacobi here Dinv (no colors)
        /* 7.3) apply smoother using custom submat transpose product kernels */
        //     7.3.1) compute Dc^{-1} * PF => PF in place (applies Dinv to the rows of this color)
        // get num nnzb in PF of this color part of submat
        // printf("\tcolor %d with nodes %d to %d and nnzb = %d\n", icolor, start_node, end_node, PF_color_nnzb);
        dim3 DP_block(216);
        dim3 DP_grid(PF_nnzb);
        k_compute_Dinv_P_mmprod<T><<<DP_grid, DP_block>>>(PF_nnzb, block_dim, 
            d_dinv_vals.getPtr(), d_PF_rows, d_PF_vals);

        /* 7.4) now add colored modified rows from PF into P as the colored update from the smoother */
        // (considering that PF has some fillin to P (which we'll drop?))
        dim3 add_block(64);
        k_add_colored_submat_PFP<T><<<DP_grid, add_block>>>(PF_nnzb, block_dim, omegaMC, 0,
            d_PF_vals, d_P_vals);

        /* 7.5) TBD: apply orthogonal projector with the rigid body modes matrix */
        // without this, lots of nullspace rigid body rotations appear in the element (as not penalized away by optimization,
        // i.e. they don't affect the gradient of P and can just be added in). You still drop the objective of the P^T*A*P
        // but it does so by dropping magnitude of disp using rigid body modes (and get non-smooth disp and defect still)
        


        /* 7.6) partition of unity projection also ? */
        // either P^T P = I orthonorm or row-sum P constraints? Can't be row-sum with multiple DOF per node
        // other paper does include P^T * P = I orthonorm constraint (energy-min does not but just 3x3 anisotropic poisson very benign and uncoupled)
        // just doing diag(P^T P) = diag(I) or each row is unit vec constraint in ||\cdot||_2 norm
        if (norm_rows) {
            dim3 norm_block(32);
            dim3 norm_grid(nnodes);
            // printf("\tnormalize rows\n");
            // int max_inner_row = 3; // just normalize u,v,w rows
            int max_inner_row = 6; // normalize all 6 dof in each block
            k_normalize_rows<T><<<norm_grid, norm_block>>>(nnodes, block_dim, max_inner_row, d_PF_rowp, d_P_vals);
            // CHECK_CUDA(cudaDeviceSynchronize());
            // printf("\tnormalize rows done\n");
        }
        


        /* 7.7) compute the defect norms to check progress */
        // compute new PF -K*P defect matrix one more time,
        cudaMemset(d_PF_vals, 0.0, PF_nnzb * 36 * sizeof(T));
        a = -1.0; // compute PF = -K * P matrix-matrix
        k_compute_P_K_P_mmprod<T><<<PKP_grid, PKP_block>>>(nnzb_prod, block_dim, a, d_K_blocks, 
            d_P_blocks, d_PF_blocks, d_kmat_vals, d_P_vals, d_PF_vals);
        // compute both P*v and -K*P*v and PF*v, three things to check..
        //   first d_temp = P*u_c
        a = 1.0, b = 0.0; // v = 0*v + P*v
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nnodes_fine, nnodes_coarse, PF_nnzb, &a, descr_Pmat, d_P_vals, d_PF_rowp, d_PF_cols, block_dim, 
            c_soln.getPtr(), &b, d_temp));
        if (write_vtk || ismooth == nsmooth - 1) {
            d_temp_vec.permuteData(6, f_grid->d_perm); // permute solve to vis order
            auto h_soln_smooth = d_temp_vec.createHostVec();
            d_temp_vec.permuteData(6, f_grid->d_iperm); // permute back to solve order for next step
            std::stringstream outputFile1;
            outputFile1 << "out/plate_cf_smooth_" << ismooth << ".vtk";
            printToVTK<Assembler,HostVec<T>>(assembler, h_soln_smooth, outputFile1.str());
        }
        T max_disp = get_max_disp<T>(d_temp_vec);
        
        // then compute -K*temp = -K*P*u_c => d_temp2
        a = -1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nnodes_fine, nnodes_fine, kmat_nnzb, &a, descr_Kmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols, block_dim, 
            d_temp, &b, d_temp2));
        T def_nrm = get_max_disp<T>(d_temp2_vec);
        if (write_vtk || ismooth == nsmooth - 1) {
            d_temp2_vec.permuteData(6, f_grid->d_perm); // permute solve to vis order
            auto h_def0 = d_temp2_vec.createHostVec();
            std::stringstream outputFile2;
            outputFile2 << "out/plate_nKPv_" << ismooth << ".vtk";
            printToVTK<Assembler,HostVec<T>>(assembler, h_def0, outputFile2.str());
        }
        
        // then compute PF*u_c equiv to -K*P*u_c (using PF matrix here)
        a = 1.0, b = 0.0; // v = 0*v + PF*v
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nnodes_fine, nnodes_coarse, PF_nnzb, &a, descr_Pmat, d_PF_vals, d_PF_rowp, d_PF_cols, block_dim, 
            c_soln.getPtr(), &b, d_temp));
        T PF_def_nrm = get_max_disp<T>(d_temp_vec);
        if (write_vtk || ismooth == nsmooth - 1) {
            d_temp_vec.permuteData(6, f_grid->d_perm); // permute solve to vis order
            auto h_def1 = d_temp_vec.createHostVec();
            std::stringstream outputFile3;
            outputFile3 << "out/plate_PFv_" << ismooth << ".vtk";
            printToVTK<Assembler,HostVec<T>>(assembler, h_def1, outputFile3.str());
        }
        
        
        printf("matrix smoothing step %d / %d: max disp %.4e, act defect %.4e, and PF defect %.4e\n", ismooth + 1, nsmooth, max_disp, def_nrm, PF_def_nrm);

    } // end of smoothing loop

    f_grid->free();
    c_grid->free();
    assembler.free();
    c_assembler.free();
    cudaFree(Bc);
    cudaFree(d_temp);
    cudaFree(d_temp2);
    cudaFree(d_resid);
    soln.free();
    c_soln.free();
    loads.free();
    c_loads.free();
    kmat.free();
    c_kmat.free();
    cudaFree(d_P_vals);
    cudaFree(d_PF_vals);
    // cudaMemfree


    /* 9) verification, compare smoothed prolong matrix to original prolong matrix on a vec (with standard smoothing) */
    // TBD


    return 0;
}