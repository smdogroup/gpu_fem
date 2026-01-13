/* demo smoothed aggregation GMG on a plate */
// 2_sa_plate_gen.cu uses smoothed aggregation on plate with general header code (not hard-coded locally for this case)

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
// #include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/smoothers/cheb4_poly.h"
// #include "multigrid/prolongation/structured.h"
#include "multigrid/prolongation/unstruct_smooth.h"
#include "multigrid/prolongation/unstructured.h"
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
    // using Smoother = MulticolorGSSmoother_V1<Assembler>;
    // using Prolongation = StructuredProlongation<Assembler, PLATE>;

    const bool KMAT_FILLIN = true; // means that K*P sparsity used in optimized prolongator
    // const bool KMAT_FILLIN = false; // DEBUG
    
    using Prolongation = UnstructuredSmoothProlongation<Assembler, Basis, KMAT_FILLIN>;
    // using Prolongation = UnstructuredProlongation<Assembler, Basis, false>;
    
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
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

    // smoother settings
    double omega = 0.4;
    // double omega = 0.4*3;
    // int ORDER = 1; // like jacobi (as 1_sa_plate_hc.cu does)
    int ORDER = 4;
    // int smooth_matrix_iters = 5;
    int smooth_matrix_iters = 1;

    printf("1) create handles\n");
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    /* 2) create the fine grid assembler, multicolor reordering  */
    
    printf("2.1) create fine assembler\n");
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nxe; // for now (should have 25 grids)
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
    // auto f_smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, 1.0);
    auto f_smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER);
    f_smoother->setup_cg_lanczos(loads);
    int ELEM_MAX = 10;
    auto f_prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
    T omegaLS_min = 0.5, omegaLS_max = 2.0;
    int startup_smooth_matrix_iters = 0; // so that we have to call smooth matrix after doing unsmooth first (normally smooth_matrix_iters here should call on startup)
    auto f_grid = new GRID(assembler, f_prolongation, f_smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max, startup_smooth_matrix_iters);


    /* 3) create the coarse grid assembler, full LU + AMD reordering  */
    printf("3.1) create coarse assembler\n");
    nxe_per_comp = nxe/2, nye_per_comp = nxe / 2; // for now (should have 25 grids)
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
    // auto c_smoother = new Smoother(cublasHandle, cusparseHandle, c_assembler, c_kmat, c_h_color_rowp, 1.0);
    auto c_smoother = new Smoother(cublasHandle, cusparseHandle, c_assembler, c_kmat, omega, ORDER);
    // c_smoother->setup_cg_lanczos(c_loads);
    auto c_prolongation = new Prolongation(cusparseHandle, c_assembler, ELEM_MAX);
    auto c_grid = new GRID(c_assembler, c_prolongation, c_smoother, c_kmat, c_loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max, startup_smooth_matrix_iters);

    // initialize coarse data of prolongator
    printf("3.3) init coarse data\n");
    f_prolongation->init_coarse_data(c_grid->assembler);

    /* 4) test initial prolongation */

    // 4.1 - : test out on a color-ordered vector to check see if P_mat is reasonable..
    // first run a coarse grid direct solve with loads permuted to solve order (Not vis order)
    printf("4.1) test initial prolongator\n");
    bool permute_inout = false; // so that I have to do one less perm step for c_soln to be in solve order
    CUSPARSE::direct_LU_solve(c_kmat, c_grid->d_rhs, c_soln, true, permute_inout);
    c_soln.permuteData(6, c_grid->d_perm); // permute solve to vis order
    auto h_c_soln = c_soln.createHostVec(); 
    c_soln.permuteData(6, c_grid->d_iperm); // permute back to solve order
    printToVTK<Assembler,HostVec<T>>(c_assembler, h_c_soln, "out/_plate_coarse.vtk");
    // compare to coarse-fine from the structured plate prolongator
    soln.zeroValues();
    f_prolongation->prolongate(c_soln, soln); // to solve order
    soln.permuteData(6, f_grid->d_perm); // from solve to vis order
    auto h_f_soln2 = soln.createHostVec(); // permute solve to vis order
    printToVTK<Assembler,HostVec<T>>(assembler, h_f_soln2, "out/_plate_cf_orig2.vtk");

    /* 5) perform matrix smoothing and test smoothed prolongation */
    if constexpr (KMAT_FILLIN) {
        printf("5.1) smooth prolongation matrix\n");
        f_grid->smoothMatrix(smooth_matrix_iters); // call smooth matrix ourselves
        // then try coarse-fine prolong again
        printf("5.2) test smooth prolongator\n");
        soln.zeroValues();
        f_prolongation->prolongate(c_soln, soln);
        soln.permuteData(6, f_grid->d_perm);
        auto h_f_soln3 = soln.createHostVec(); // permute solve to vis order
        printToVTK<Assembler,HostVec<T>>(assembler, h_f_soln3, "out/_plate_cf_smooth2.vtk");
    } else {
        printf("DEBUG: warning, kmat fillin skipped so no matrix smoothing done\n");
    }

    f_grid->free();
    c_grid->free();
    assembler.free();
    c_assembler.free();
    soln.free();
    c_soln.free();
    loads.free();
    c_loads.free();
    kmat.free();
    c_kmat.free();
    // cudaMemfree


    /* 9) verification, compare smoothed prolong matrix to original prolong matrix on a vec (with standard smoothing) */
    // TBD


    return 0;
}