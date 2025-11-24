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
    // double SR = 10.0;
    double SR = 100.0;

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
    // T *fine_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q); // comlicated load case
    int m = 1, n = 2; // (1,2) sine-sine load
    T *fine_loads = getPlateSineLoads<T, Physics>(nxe, nxe, Lx, Ly, m, n, Q); // sine-sine load case
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
    // T *coarse_loads = getPlateLoads<T, Physics>(nxe / 2, nxe / 2, Lx, Ly, Q);
    T *coarse_loads = getPlateSineLoads<T, Physics>(nxe / 2, nxe / 2, Lx, Ly, m, n, Q); // sine-sine load case
    auto &c_bsr_data = c_assembler.getBsrData();
    int c_num_colors = 1, *_c_color_rowp = new int[2];
    _c_color_rowp[0] = 0, _c_color_rowp[1] = c_assembler.get_num_nodes() + 1;
    c_bsr_data.AMD_reordering();
    c_bsr_data.compute_full_LU_pattern(10.0, false);
    auto c_h_color_rowp = HostVec<int>(c_num_colors + 1, _c_color_rowp);
    c_assembler.moveBsrDataToDevice();
    T *xpts_coarse = c_assembler.getXpts().createHostVec().getPtr();
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
        for (int i = 0; i < 6; i++) { // each of 6 rows
            T total_scale = 0.0;
            for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
                total_scale += P_vals[36 * jp + 6 * i + i]; // (i,i) entry of a block
            }
            // now you have total scale
            for (int jp = P_rowp[brow]; jp < P_rowp[brow+1]; jp++) {
                P_vals[36 * jp + 6 * i + i] /= total_scale;
                // P_vals[36 * jp + 6 * i + 6] /= total_scale; // old was mistake here
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
    // OPTIONAL do sparsity increase again
    // because prelim prolong doesn't have enough sparsity increase I don't think with AP_0
    bool compute_PF_fillin2 = true;
    // bool compute_PF_fillin2 = false;
    int PF2_nnzb, *PF2_rowp, *d_PF2_rowp, *PF2_rows, *d_PF2_rows, *PF2_cols, *d_PF2_cols;
    if (compute_PF_fillin2) {
        //   note there may be more efficient ways of getting fillin, not worried about that rn (just demo script)
        PF2_rowp = new int[nnodes_fine + 1];
        memset(PF2_rowp, 0, (nnodes_fine + 1) * sizeof(int));
        for (int i = 0; i < nnodes_fine; i++) {
            PF2_rowp[i+1] = PF2_rowp[i];
            std::unordered_set<int> unique_cols;
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];
                for (int jp = PF_rowp[k]; jp < PF_rowp[k+1]; jp++) {
                    int j = PF_cols[jp];
                    unique_cols.insert(j);
                }
            }
            PF2_rowp[i+1] += unique_cols.size();
        }
        printf("PF2_rowp: ");
        printVec<int>(50, PF2_rowp);
        PF2_nnzb = PF2_rowp[nnodes_fine];
        PF2_cols = new int[PF2_nnzb];
        PF2_rows = new int[PF2_nnzb];
        int iinz = 0;
        for (int i = 0; i < nnodes_fine; i++) {
            std::set<int> ordered_unique_cols;
            for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i+1]; kp++) {
                int k = h_kmat_cols[kp];
                for (int jp = PF_rowp[k]; jp < PF_rowp[k+1]; jp++) {
                    int j = PF_cols[jp];
                    ordered_unique_cols.insert(j);
                }
            }
            for (int col : ordered_unique_cols) {
                PF2_cols[iinz] = col;
                PF2_rows[iinz] = i;
                iinz++;
            }
        }
        printf("PF2_cols: ");
        printVec<int>(50, PF2_cols);
        d_PF2_rowp = HostVec<int>(nnodes_fine + 1, PF2_rowp).createDeviceVec().getPtr();
        d_PF2_rows = HostVec<int>(PF2_nnzb, PF2_rows).createDeviceVec().getPtr();
        d_PF2_cols = HostVec<int>(PF2_nnzb, PF2_cols).createDeviceVec().getPtr();

        // hten replace into old sparsity
        PF_rowp = PF2_rowp, PF_rows = PF2_rows, PF_cols = PF2_cols;
        PF_nnzb = PF2_nnzb;
        d_PF_rowp = d_PF2_rowp, d_PF_rows = d_PF2_rows, d_PF_cols = d_PF2_cols;
    } else {
        // do nothing, no additional sparsity increase
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

    // 7.1) compute on the host first
    bool proper_rot_bcs = true;
    // bool proper_rot_bcs = false; // just does row sums and R or Bc = I

    printf("7.1) compute the coarse mesh rigid body modes\n");
    T *Bc = new T[36 * nnodes_coarse]; // get it in solve order so need h_c_perm
    memset(Bc, 0.0, 36 * nnodes_coarse);
    for (int pic = 0; pic < nnodes_coarse; pic++) {
        // pic is permuted or solve order coarse nodes, while ic is vis or natural order
        int ic = h_c_perm[pic];
        T x = xpts_coarse[3 * ic], y = xpts_coarse[3 * ic + 1], z = xpts_coarse[3 * ic + 2];
        Bc[36 * pic + 6 * 0 + 0] = 1.0; // u translation
        Bc[36 * pic + 6 * 1 + 1] = 1.0; // v translation
        Bc[36 * pic + 6 * 2 + 2] = 1.0; // w translation
        if (proper_rot_bcs) {
            // baseline (what I think is) correct
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
            // I have a test for rigid body modes later on, these are the correct ones..
            // changing signs messes them up (this is correct at least for a plate)

        } else {
            // temp debug, if we do this to just enforce row-sums
            // not exactly correct, but just try it for a sec..
            Bc[36 * pic + 6 * 3 + 3] = 1.0;        
            Bc[36 * pic + 6 * 4 + 4] = 1.0;
            Bc[36 * pic + 6 * 5 + 5] = 1.0;
        }

        // print out set values, DEBUG
        // printf("Bc(%d) = \n", pic);
        // for (int i = 0; i < 6; i++) {
        //     printf("\t");
        //     printVec<T>(6, &Bc[36 * pic + 6 * i]);
        // }
    }
    // move to device
    T *d_Bc = HostVec<T>(36 * nnodes_coarse, Bc).createDeviceVec().getPtr();
    // then compute the fine node BC indicator matrix Fi later?
    // printf("nnodes_coarse %d\n", nnodes_coarse);

    // =============================================
    // DEBUG: 
    // bool test_rigid_body_modes = true;
    bool test_rigid_body_modes = false;
    // test that the coarse rigid body modes have zero apparent energy b^T K b on the coarse mesh for each b in Bc
    if (test_rigid_body_modes) {
        printf("test coarse rigid body modes are rigid\n");
        // construct the b vectors on the host for coarse mesh size
        int Nc = c_assembler.get_num_vars();
        T *h_b = new T[Nc];
        T *h_force = new T[Nc];
        auto d_b = DeviceVec<T>(Nc);
        auto d_ctemp = DeviceVec<T>(Nc);
        T *d_cresid = DeviceVec<T>(Nc).getPtr();
        // get rowp, cols, values of the coarse kmat
        int *d_ckmat_rowp = c_bsr_data.rowp;
        int *d_ckmat_cols = c_bsr_data.cols;
        int ckmat_nnzb = c_bsr_data.nnzb;
        T *d_ckmat_vals = c_kmat.getPtr();
        int *h_ckmat_rowp = DeviceVec<int>(nnodes_coarse + 1, d_ckmat_rowp).createHostVec().getPtr();
        int *h_ckmat_cols = DeviceVec<int>(ckmat_nnzb, d_ckmat_cols).createHostVec().getPtr();
        // compute a reference disp mode energy, <c_soln, K * c_soln> which should be nonzero
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
            nnodes_coarse, nnodes_coarse, ckmat_nnzb, &a, descr_Kmat, d_ckmat_vals, d_ckmat_rowp, d_ckmat_cols, block_dim, 
            c_soln.getPtr(), &b, d_ctemp.getPtr()));
        // now comput dot product <d_ctemp, d_b>
        T vkv_dot_ref;
        CHECK_CUBLAS(cublasDdot(cublasHandle, Nc, c_soln.getPtr(), 1, d_ctemp.getPtr(), 1, &vkv_dot_ref));
        // divide by the vec or disp norm
        T v_normsq;
        CHECK_CUBLAS(cublasDdot(cublasHandle, Nc, c_soln.getPtr(), 1, c_soln.getPtr(), 1, &v_normsq));
        vkv_dot_ref /= v_normsq;
        // I'd basically need to have the coarse kmat with no essential BCs (so let's reassemble it with no bcs)
        c_assembler.add_jacobian_fast(c_kmat);
        // c_assembler.apply_bcs(c_kmat); // don't do this cause we want to measure no BC rigid body mdoes (verification)
        // now begin the loop over each of hte 6 rigid body modes
        for (int irbm = 0; irbm < 6; irbm++) {
            // copy the irbm column from Bc into the b vector of coarse size (in solve order), on host first
            memset(h_b, 0.0, Nc * sizeof(T));
            for (int i = 0; i < Nc; i++) {
                h_b[i] = Bc[36 * (i / 6) + 6 * (i % 6) + irbm]; // other disps were roughly 1e-9 or 1e-8 so similar mag
                // h_b[i] *= 1e-9; // dividing by vec norms later (so no need for this)
            }
            // print the values of this rigid body mode
            // printf("rigid body mode %d has vals: ", irbm);
            // printVec<T>(50, h_b);
            // then copy to device
            cudaMemcpy(d_b.getPtr(), h_b, Nc * sizeof(T), cudaMemcpyHostToDevice);
            // now compute K * d_b => d_ctemp
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                nnodes_coarse, nnodes_coarse, ckmat_nnzb, &a, descr_Kmat, d_ckmat_vals, d_ckmat_rowp, d_ckmat_cols, block_dim, 
                d_b.getPtr(), &b, d_ctemp.getPtr()));
            // // and apply bcs on residuals too? (no cause I didn't do on this kmat now)
            // // c_assembler.apply_bcs(d_ctemp); // so don't penalize boundary displacements..
            // copy the right hand sides to host and printout..
            // cudaMemcpy(h_force, d_ctemp.getPtr(), Nc * sizeof(T), cudaMemcpyDeviceToHost);
            // printf("K*h_b mode: ");
            // printVec<T>(50, h_force);
            // now comput dot product <d_ctemp, d_b>
            T bkb_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, Nc, d_b.getPtr(), 1, d_ctemp.getPtr(), 1, &bkb_dot));
            T b_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, Nc, d_b.getPtr(), 1, d_b.getPtr(), 1, &b_dot));
            // divide the <b,b> dot product to normalize it
            bkb_dot /= b_dot;
            T rel_energy = bkb_dot / vkv_dot_ref;
            printf("\trigid body  mode %d : bkb dot = %.4e / vkv_dot_ref = %.4e => rel energy %.4e\n", irbm, bkb_dot, vkv_dot_ref, rel_energy);
        }
        printf("done testing coarse rigid body modes are actually rbmodes\n");
        return 0;
    }
    // =============================================
    // so yes those are the rigid body modes, it does work

    /* 7.2) compute free var unknowns */
    // get the bcs and compute a free DOF map on the device
    auto free_var_vec = DeviceVec<bool>(N);
    free_var_vec.setFullVecToConstValue(true); // set all to default true meaning free var
    bool *d_free_dof_ptr = free_var_vec.getPtr();
    auto bcs_vec = assembler.getBCs();
    int n_bcs = bcs_vec.getSize();
    dim3 bcs_block(32), bcs_grid((n_bcs + 31) / 32);
    // computes false or true bool pointer d_free_dof_ptr in solve node order (not vis order)
    k_get_free_dof<<<bcs_grid, bcs_block>>>(n_bcs, block_dim, bcs_vec.getPtr(), f_grid->d_iperm, d_free_dof_ptr);

    /* 7.3) construct (UTU+eps*I)^-1 matrix as linear operator just like Dinv */
    // for the orthogonal projector
    auto d_UTU_vals = DeviceVec<T>(ndiag_vals);
    // compute UTU + eps*I values in a kernel
    dim3 OP_block0(32), OP_grid0(nnodes_fine);
    k_orthog_projector_computeUTU<T><<<OP_grid0, OP_block0>>>(nnodes_fine, block_dim, d_Bc, 
        d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_UTU_vals.getPtr());
    CHECK_CUDA(cudaDeviceSynchronize());
    // DEBUG, printout the computed UTU values on host
    // if (nxe <= 8) {
    //     auto h_UTU_vals = d_UTU_vals.createHostVec();
    //     printf("h_UTU_vals: ");
    //     T *h_UTU_ptr = h_UTU_vals.getPtr();
    //     for (int inode = 0; inode < nnodes_fine; inode++) {
    //         printf("  UTU-node(%d) = \n", inode);
    //         T *h_UTU_node = &h_UTU_ptr[36 * inode];
    //         for (int i = 0; i < 6; i++) {
    //             printf("\t");
    //             printVec<T>(6, &h_UTU_node[6 * i]);
    //         }
    //     }
    //     return 0;
    // }
    
    // now compute the LU factor and inverse matrix UTUinv for each fine node (same size and like Dinv matrix)
    // reuse same pointers and nnzb sizes as Dinv cause same dimensions
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                         &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                         d_UTU_vals.getPtr(), d_diag_rowp, d_diag_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);
    // now compute UTUinv linear operator like I did for the Dinv
    auto d_UTUinv_vals = DeviceVec<T>(ndiag_vals); // inv linear operator of UTU
    for (int i = 0; i < block_dim; i++) {
        // set d_temp to ei (one of e1 through e6 per block)
        cudaMemset(d_temp, 0.0, N * sizeof(T));
        dim3 block(32);
        dim3 grid((nnodes + 31) / 32);
        k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

        // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_UTU_vals.getPtr(),
            d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
            pBuffer));  // prob only need U^-1 part for block diag.. TBD

        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
            cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_UTU_vals.getPtr(),
            d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

        // now copy temp2 into columns of new operator
        dim3 grid2((N + 31) / 32);
        k_setLUinv_operator<T>
            <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_UTUinv_vals.getPtr());
    }
    // auto UTUinv_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_UTUinv_vals);
    // DEBUG the UTUinv vals now
    // if (nxe <= 8) {
    //     auto h_UTU_vals = d_UTU_vals.createHostVec();
    //     auto h_UTUinv_vals = d_UTUinv_vals.createHostVec();
    //     printf("h_UTU vs UTUinv vals: \n");
    //     T *h_UTU_ptr = h_UTU_vals.getPtr();
    //     T *h_UTUinv_ptr = h_UTUinv_vals.getPtr();
    //     for (int pif = 0; pif < nnodes_fine; pif++) {
    //         int inode = h_f_perm[pif];
    //         int ix = inode % nx_f, iy = inode / nx_f;
    //         int is_interior = (0 < ix) && (ix < nx_f-1) && (0 < iy) && (iy < nx_f-1);
    //         int nc = PF_rowp[pif+1] - PF_rowp[pif]; // num corase nodes attached
    //         printf("  UTU-node(%d) with %d coarse neighbors (is_interior %d)\n", pif, nc, is_interior);
    //         T *h_UTU_node = &h_UTU_ptr[36 * pif];
    //         for (int i = 0; i < 6; i++) {
    //             printf("\t");
    //             printVec<T>(6, &h_UTU_node[6 * i]);
    //         }

    //         printf("  UTUinv-node(%d) (is_interior %d) = \n", pif, is_interior);
    //         T *h_UTUinv_node = &h_UTUinv_ptr[36 * pif];
    //         for (int i = 0; i < 6; i++) {
    //             printf("\t");
    //             printVec<T>(6, &h_UTUinv_node[6 * i]);
    //         }
    //     }
    //     return 0;
    // }
    // also set storage for SU matrix (aggregated for each fine node, like row sums of prolong, same size as D and UTU matrix)
    auto d_SU_vals = DeviceVec<T>(ndiag_vals);

    /* begin smoothing phase of the P matrix! */
    printf("h_color_rowp: ");
    printVec<int>(num_colors + 1, h_color_rowp.getPtr());
    printf("\twith nnodes_fine = %d\n", nnodes_fine);
    // printf("PF_rowp[4224-on]: ");
    // prinVec<int>(2, PF_rowp)
    auto d_temp2_vec = DeviceVec<T>(N, d_temp2);

    // int nsmooth = 1;
    // int nsmooth = 2;
    // int nsmooth = 4; // good num here, shouldn't need too many steps
    int nsmooth = 5;
    // int nsmooth = 10;
    // int nsmooth = 20;
    // int nsmooth = 40;
    // int nsmooth = 100; 
    // int nsmooth = 300;

    // omegaMC < 2/rho(Dinv*A) is what it needs to be (may need to estimate from Arnoldi hessenberg iteration, the max eigenvalue)
    // T omegaMC = 1e-5;
    // T omegaMC = -1e-5;
    // T omegaMC = 1e-4;
    // T omegaMC = -1e-4; // wrong direction?
    // T omegaMC = 4e-4;
    // T omegaMC = -4e-4;
    // T omegaMC = -5e-4;
    // T omegaMC = 5e-4;
    // T omegaMC = -1e-3;
    // T omegaMC = 1e-3;
    // T omegaMC = 2e-3;
    // T omegaMC = 5e-3;
    // T omegaMC = -5e-3; // wrong direction?
    // T omegaMC = 0.01;
    // T omegaMC = 0.04;
    // T omegaMC = 0.05;
    // T omegaMC = -0.05;
    // T omegaMC = 0.1;
    // T omegaMC = -0.1;
    // T omegaMC = 0.2;
    T omegaMC = 0.4;
    // T omegaMC = 0.7; // smoother constant
    // T omegaMC = 1.5;

    // for debugging
    // bool write_vtk = true;
    bool write_vtk = false;

    // use multicolor vs jacobi smoother
    bool use_multicolor = true;
    // bool use_multicolor = false;
    int nloop_colors = 1;
    if (use_multicolor) {
        // // TEMP DEBUG
        // nsmooth = 1;
        // nloop_colors = 1; 

        nloop_colors = num_colors;
        // not hte most efficient right now for the MC case bc of 
    }

    // do orthog projection of near kernel modes
    

    // do orthog projection (should want to do this)
    // bool do_orthog_proj = true;
    bool do_orthog_proj = false;

    printf("\n\n=========================================");
    printf("=========================================\n");
    printf("inner-prod space energy min: init_defect %.8e and using omega = %.4e\n", def_nrm0, omegaMC);
    if (use_multicolor) {
        printf("\tusing multicolor smoother with %d colors\n", num_colors);
    } else {
        printf("\tusing jacobi smoother so only 1 color\n");
    }
    if (proper_rot_bcs && do_orthog_proj) {
        printf("\tusing true [I3,Omega;0,I3] rigid body modes for linear shells\n");
    } else if (do_orthog_proj) {
        printf("\tusing I6 row-sums instead of true rigid body modes (debug/test)\n");
    } else {
        printf("\tnot doing orthogonal projector for row-sum + rigid body mode constraint\n");
    }

    int itest = 0;
    for (int ismooth = 0; ismooth < nsmooth; ismooth++) {

        for (int icolor = 0; icolor < nloop_colors; icolor++) { // is just 1 color if jacobi
    
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

        /* 7.4) apply orthogonal projector to the P matrix update dP = Dinv * PF, Q(dP) => dP' */
        // adds in terms that are lost as we don't fillin P matrix each time
        // this is important cause scalar equivalent is row-sum constraint (normalizes matrix in a way)
        // while vector cause does that + accounts for rigid body mdoes
        if (do_orthog_proj) {

            dim3 OP_block(32), OP_grid(nnodes_fine);
            // old call 
            // k_orthog_projector_old<T><<<OP_grid, OP_block>>>(nnodes_fine, block_dim, d_Bc, 
            //     d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_PF_vals);
            // new calls have UTU precomputed out front and data allocated for SU result for each fine node (block row-sums)
            //   thus the new calls use much less shared mem in each kernel call
            //   so first just compute the SU resultant for each fine node
            // zero out SU since last smooth step (since we add into it again)
            d_SU_vals.zeroValues();
            k_orthog_projector_computeSU<T><<<OP_grid, OP_block>>>(nnodes_fine, block_dim, d_Bc, 
                d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_PF_vals, d_SU_vals.getPtr());
            // DEBUG the SU vals
            // if (nxe <= 8) {
            //     auto h_S_vals = DeviceVec<T>(36 * PF_nnzb, d_PF_vals).createHostVec();
            //     auto h_SU_vals = d_SU_vals.createHostVec();
            //     printf("host S and SU resultant values: \n");
            //     T *h_S_ptr = h_S_vals.getPtr();
            //     T *h_SU_ptr = h_SU_vals.getPtr();
            //     for (int inode = 0; inode < nnodes_fine; inode++) {
            //         // printout the S or dP values in each row
            //         int pif = h_f_iperm[inode]; // permuted fine node
            //         int nc = PF_rowp[pif + 1] - PF_rowp[pif]; // num coarse nodes in this fine node block row
            //         printf("S = dP values for fine node %d, with %d coarse nodes\n", inode, nc);
            //         for (int jp = PF_rowp[pif]; jp < PF_rowp[pif + 1]; jp++) {
            //             int pic = PF_cols[jp]; // permuted coarse node
            //             int ic = h_c_perm[pic];
            //             printf("  S or dP of fine node %d, coarse node %d\n", inode, ic);
            //             T *h_S_node = &h_S_ptr[36 * jp];
            //             for (int i = 0; i < 6; i++) {
            //                 printf("\t");
            //                 printVec<T>(6, &h_S_node[6 * i]);
            //             }
            //         }

            //         printf("  SU-node(%d) = \n", inode);
            //         T *h_SU_node = &h_SU_ptr[36 * pif];
            //         for (int i = 0; i < 6; i++) {
            //             printf("\t");
            //             printVec<T>(6, &h_SU_node[6 * i]);
            //         }
            //     }
            //     return 0;
            // }
            // then compute S = S - (SU) * (UTUinv) * U^T mat-mat products where SU and UTUinv are fixed resultants for each fine node block row
            //   but U^T is the U matrix for that particular block of the S == dP matrix or P update matrix
            //   it thus eliminates any rigid body row-sums from dP update (preventing state drift in P matrix update, normalizing it)
            k_orthog_projector_removeRowSums<T><<<OP_grid, OP_block>>>(nnodes_fine, block_dim, d_Bc, 
                d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_SU_vals.getPtr(), d_UTUinv_vals.getPtr(), d_PF_vals);
            // DEBUG the new S = dP vals
            // if (nxe <= 8) {
            //     auto h_S_vals = DeviceVec<T>(36 * PF_nnzb, d_PF_vals).createHostVec();
            //     printf("host dP values after orthog projector (for updating prolongation P += dP where S = dP): \n");
            //     T *h_S_ptr = h_S_vals.getPtr();
            //     for (int inode = 0; inode < nnodes_fine; inode++) {
            //         // printout the S or dP values in each row
            //         int pif = h_f_iperm[inode]; // permuted fine node
            //         int nc = PF_rowp[pif + 1] - PF_rowp[pif]; // num coarse nodes in this fine node block row
            //         printf("S = dP values for fine node %d, with %d coarse nodes\n", inode, nc);
            //         for (int jp = PF_rowp[pif]; jp < PF_rowp[pif + 1]; jp++) {
            //             int pic = PF_cols[jp]; // permuted coarse node
            //             int ic = h_c_perm[pic];
            //             printf("  S or dP of fine node %d, coarse node %d\n", inode, ic);
            //             T *h_S_node = &h_S_ptr[36 * jp];
            //             for (int i = 0; i < 6; i++) {
            //                 printf("\t");
            //                 printVec<T>(6, &h_S_node[6 * i]);
            //             }
            //         }
            //     }
            //     return 0;
            // } 
            // DEBUG check row col sums in each S row (if R = I set for debug), row sum of S should be zero in each DOF row
            // if (nxe <= 8) {
            //     auto h_S_vals = DeviceVec<T>(36 * PF_nnzb, d_PF_vals).createHostVec();
            //     printf("\n===========================\n");
            //     printf("host dP values after orthog projector (for updating prolongation P += dP where S = dP): \n");
            //     printf("\n===========================\n");
            //     T *h_S_ptr = h_S_vals.getPtr();
            //     for (int inode = 0; inode < nnodes_fine; inode++) {
            //         // printout the S or dP values in each row
            //         int pif = h_f_iperm[inode]; // permuted fine node
            //         int ix = inode % nx_f, iy = inode / nx_f;
            //         int is_interior = (0 < ix) && (ix < nx_f-1) && (0 < iy) && (iy < nx_f-1);
                    
            //         int nc = PF_rowp[pif + 1] - PF_rowp[pif];
            //         printf("S = dP values for fine node %d and pif %d, with %d coarse nodes (is_interior %d)\n", inode, pif, nc, is_interior);
            //         for (int idof = 0; idof < 6; idof++) {
            //             // compute row sum first
            //             T rc_sums[6] = { };
            //             for (int jp = PF_rowp[pif]; jp < PF_rowp[pif+1]; jp++) {
            //                 for (int jj = 0; jj < 6; jj++) {
            //                     rc_sums[jj] += h_S_ptr[36 * jp + 6 * idof + jj]; 
            //                 }
            //             }
            //             printf("  dof row %d has rc sums: ", idof);
            //             printVec<T>(6, rc_sums);

            //             for (int jp = PF_rowp[pif]; jp < PF_rowp[pif+1]; jp++) {
            //                 int j = PF_cols[jp];
            //                 printf("\t");
            //                 printVec<T>(6, &h_S_ptr[36 * jp + 6 * idof]);
            //             }
            //             printf("\n");
            //         }
            //     }
            //     return 0;
            // } 

        } // end of do_orthog_proj if statement
        

        /* 7.5) now add modified P update into the P matrix (after orthog projection) for projected steepest descent here */
        if (use_multicolor) {
            int start_node = h_color_rowp[icolor], end_node = h_color_rowp[icolor+1];
            // int start_block = PF_rowp[start_node], end_block = PF_rowp[end_node + 1];
            int start_block = PF_rowp[start_node], end_block = PF_rowp[end_node]; // should just be end_node?
            int PF_color_nnzb = end_block - start_block;
            dim3 add_block(64);
            dim3 add_grid(PF_color_nnzb);
            // printf("\tadd dP color %d update\n", icolor);
            k_add_colored_submat_PFP<T><<<add_grid, add_block>>>(PF_nnzb, block_dim, omegaMC, start_block,
                d_PF_vals, d_P_vals);
        } else {
            // add whole dP update in
            dim3 add_block(64);
            // printf("\tadd dP update\n");
            k_add_colored_submat_PFP<T><<<DP_grid, add_block>>>(PF_nnzb, block_dim, omegaMC, 0,
                d_PF_vals, d_P_vals);
        }
        
        // printf("\t\tdone with add colored submat PFP\n");


        // } // end of color loop (optional SPOT 1)


        /* 7.6) compute the defect norms to check progress */

        // compute new PF -K*P defect matrix one more time,
        // printf("compute defect norms\n");
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
        
        
        printf("matrix smoothing step %d / %d, icolor %d: max disp %.4e, act defect %.4e, and PF defect %.4e\n", ismooth + 1, nsmooth, icolor, max_disp, def_nrm, PF_def_nrm);

        } // end of color loop (OPTIONAL spot #2 - if want to check defect after every color)

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