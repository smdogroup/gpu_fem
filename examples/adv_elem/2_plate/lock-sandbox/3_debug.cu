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

// chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/asw_unstruct.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

// local utils
#include "../include/lock_prolongation.h"
#include "../include/lock_smoother.h"

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Assembler>
void multigrid_solve(int nxe, double SR, int nsmooth, int ninnercyc, int nsmooth_mat, T omega, std::string cycle_type) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    // using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    using Smoother = UnstructuredQuadElementAdditiveSchwarzSmoother<T, Assembler>;
    // using Prolongation = StructuredProlongation<Assembler, PLATE>;
    using Prolongation = LockingAwareUnstructuredProlongation<Assembler, Basis>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;
    using LockingSmoother = LockingChebyshevSmoother<Assembler>;
    
    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();

    MG *mg;
    KMG *kmg;

    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    T omegaLS_min = 1e-2, omegaLS_max = 4.0;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    bool is_kcycle = cycle_type == "K";
    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // int pre_nxe_min = 8;
    int pre_nxe_min = 2; // so 2^2 -> 4^2 elem levels here
    // int pre_nxe_min = nxe > 32 ? 32 : 8;
    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe, nye_per_comp = c_nye; // for now (should have 25 grids)
        bool theta_ss_bc = false;
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, 
            rho, ys, nxe_per_comp, nye_per_comp, theta_ss_bc);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Basis, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d\n", c_nxe);

        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        // make the grid
        bool full_LU = c_nxe == nxe_min;
        if (full_LU) {
            printf("WARNING - turned AMD ordering on coarse grid off for DEBUG\n");
            // bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
        } else {
            bsr_data.compute_nofill_pattern();
        }

        assembler.moveBsrDataToDevice();
        auto loads = assembler.createVarsVec(my_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto res = assembler.createVarsVec();
        int N = res.getSize();

        // assemble the kmat
        auto start0 = std::chrono::high_resolution_clock::now();
        // assembler.add_jacobian(res, kmat);
        assembler.add_jacobian_fast(kmat);
        // assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = end0 - start0;
        printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations..
        printf("nsmooth %d, omega = %.4e\n", nsmooth, omega);
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
            omega, nsmooth);
        auto prolongation = new Prolongation(cusparseHandle, assembler);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max);
        
        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (full_LU) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
        }
    }

    assert(is_kcycle); // for first try..
    auto &grids = kmg->grids;

    // =====================================================
    // locking-aware prolongation
    // =====================================================

    int nlevels = grids.size();
    printf("before locking-aware prolongation\n");

    for (int i = 0; i < nlevels - 1; i++) {

        printf("Step 0 : get fine grid data on grid %d\n", i);
        // register the coarse assemblers to the prolongations..
        double lam = 1e-12; // see python locking script
        // if (is_kcycle) {
        // clear and re-assemble kmat to hold G_f^T G_f + lam * I
        auto &f_assembler = grids[i].assembler;
        auto &f_kmat = grids[i].Kmat;
        auto d_fine_bcs = f_assembler.getBCs();

        printf("Step 1 - compute G_f^T G_f LHS locking matrix\n");
        f_assembler.add_lockstrain_jacobian_fast(f_kmat);
        CHECK_CUDA(cudaDeviceSynchronize()); // slower but for debugging
        f_assembler.apply_bcs(f_kmat);
        f_kmat.add_diag_nugget(lam);

        // DEBUG : printout LHS
        auto d_kmat_vec = f_kmat.getVec();
        int size = d_kmat_vec.getSize();
        T *h_kmat_vals = d_kmat_vec.createHostVec().getPtr();
        auto d_kmat_bsr_data = f_kmat.getBsrData();
        int nnodes_fine = f_assembler.get_num_nodes();
        int kmat_nnzb = d_kmat_bsr_data.nnzb;
        int *h_kmat_rowp = DeviceVec<int>(nnodes_fine + 1, d_kmat_bsr_data.rowp).createHostVec().getPtr();
        int *h_kmat_rows = DeviceVec<int>(kmat_nnzb, d_kmat_bsr_data.rows).createHostVec().getPtr();
        int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_bsr_data.cols).createHostVec().getPtr();
        // printf("locking-kmat with nnzb %d\n", kmat_nnzb);
        // printf("h_kmat_rowp (same as locking): ");
        // printVec<int>(nnodes_fine + 1, h_kmat_rowp);
        // printf("h_kmat_cols (same as locking): ");
        // printVec<int>(kmat_nnzb, h_kmat_cols);

        int keep_dof[3] = {2, 3, 4}; // {w, thx, thy}
        // for (int iblock = 0; iblock < kmat_nnzb; iblock++) {
        //     T *block_vals = &h_kmat_vals[36 * iblock];
        //     int node_row = h_kmat_rows[iblock], node_col = h_kmat_cols[iblock];
        //     printf("\n\nlocking-kmat block node (%d,%d):\n", node_row, node_col);
        //     // for (int i = 0; i < 6; i++) {
        //     //     printf("\t");
        //     //     printVec<T>(6, &block_vals[6 * i]);
        //     // }
        //     for (int i = 0; i < 3; i++) {
        //         int j = keep_dof[i];
        //         printf("\t");
        //         for (int i2 = 0; i2 < 3; i2++) {
        //             int j2 = keep_dof[i2];
        //             // *64.0 = 8^2 (*8 missing in each strain side) cause we keep in comp coords for plate..
        //             T val = block_vals[6 * j + j2];
        //             bool is_one = abs(val - 1.0) < 1e-10;
        //             val = is_one ? 1.0 : 64.0 * val;
        //             printf("%.6e  ", val);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // printf("\ndone with h_kmat (G_f^T G_f + lam*I) matrix vals\n");

        // make device fine-coarse elem map (here just use structured pattern)
        printf("Step 2 - compute device fc elem map\n");
        int num_fine_elements = f_assembler.get_num_elements();
        int *h_fc_elem_map = new int[num_fine_elements];
        int c_nxe = (int)sqrt(num_fine_elements);
        int c_nxec = c_nxe / 2;
        for (int ielem = 0; ielem < num_fine_elements; ielem++) {
            int ixe = ielem % c_nxe, iye = ielem / c_nxe;
            int ixe_c = ixe / 2, iye_c = iye / 2;
            int ielem_c = c_nxec * iye_c + ixe_c;
            h_fc_elem_map[ielem] = ielem_c;
        }
        int *d_fc_elem_map = HostVec<int>(num_fine_elements, h_fc_elem_map).createDeviceVec().getPtr();

        // get initial prolongator
        printf("Step 3 - get initial prolongator\n");
        auto &c_assembler = grids[i+1].assembler;
        auto &c_kmat = grids[i+1].Kmat;
        auto d_coarse_bcs = c_assembler.getBCs();
        auto &prolong = grids[i].prolongation;
        prolong->init_coarse_data_manual(c_assembler, h_fc_elem_map);
        // get P_0 matrix .. standard prolongation
        auto &P_mat = prolong->prolong_mat;
        auto &RHS_mat = prolong->RHS_mat;

        // apply fine and coarse bcs to P0_mat
        printf("Step 4 - apply bcs on initial prolongator\n");
        const bool ones_on_diag = false; // just zero out completely for prolong matrix
        P_mat->template apply_bc_rows<ones_on_diag>(d_fine_bcs);
        P_mat->template apply_bc_cols<ones_on_diag>(d_coarse_bcs);

        // DEBUG print initial prolongation matrix
        auto P_bsr_data = P_mat->getBsrData();
        int nnodes_coarse = P_bsr_data.nb;
        int P_nnzb = P_bsr_data.nnzb;
        int *h_P0_rowp = DeviceVec<int>(nnodes_fine + 1, P_bsr_data.rowp).createHostVec().getPtr();
        int *h_P0_rows = DeviceVec<int>(P_nnzb, P_bsr_data.rows).createHostVec().getPtr();
        int *h_P0_cols = DeviceVec<int>(P_nnzb, P_bsr_data.cols).createHostVec().getPtr();
        T *h_P0_vals = P_mat->getVec().createHostVec().getPtr();
        // printf("init-prolongMat with nnzb %d\n", P_nnzb);
        // printf("h_P0_rowp: ");
        // printVec<int>(nnodes_fine + 1, h_P0_rowp);
        // printf("h_P0_cols: ");
        // printVec<int>(P_nnzb, h_P0_cols);
        // for (int iblock = 0; iblock < P_nnzb; iblock++) {
        //     T *block_vals = &h_P0_vals[36 * iblock];
        //     int node_row = h_P0_rows[iblock], node_col = h_P0_cols[iblock];
        //     printf("\n\ninit-prolongMat block node (%d,%d):\n", node_row, node_col);
        //     // for (int i = 0; i < 6; i++) {
        //     //     printf("\t");
        //     //     printVec<T>(6, &block_vals[6 * i]);
        //     // }
        //     for (int i = 0; i < 3; i++) {
        //         int j = keep_dof[i];
        //         printf("\t");
        //         for (int i2 = 0; i2 < 3; i2++) {
        //             int j2 = keep_dof[i2];
        //             // all vals here should be either {0, 0.25, 0.5, 1} so let's round it so I can do quick compare
        //             T val = block_vals[6 * j + j2]; // I checked and there are sometimes like 1e-7 errors because I do optimization of (xi,eta) pairs
        //             val = std::round(val * 100) / 100; // rounds to two decimal places (so txt file comparison will work better)
        //             printf("%.6e  ", val);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // printf("\ndone with init prolong P0 mat\n");

        // now assemble G_f^T * P_gam * G_c + lam * P_0  RHS prolong matrix with K*P0 sparsity
        printf("Step 5 - compute locking RHS fine-coarse matrix\n");
        f_assembler.add_lockstrain_fc_jacobian_fast(c_assembler, d_fc_elem_map, *RHS_mat);
        CHECK_CUDA(cudaDeviceSynchronize()); // slower but for debugging
        printf("\tdone with assembly FC matrix from step 5\n");
        // apply bcs to P_rhs matrix
        RHS_mat->template apply_bc_rows<ones_on_diag>(d_fine_bcs);
        RHS_mat->template apply_bc_cols<ones_on_diag>(d_coarse_bcs);

        // auto rhs_bsr_data = RHS_mat->getBsrData();
        // int rhs_nnzb = rhs_bsr_data.nnzb;
        // int *h_rhs_rowp = DeviceVec<int>(nnodes_fine + 1, rhs_bsr_data.rowp).createHostVec().getPtr();
        // int *h_rhs_rows = DeviceVec<int>(rhs_nnzb, rhs_bsr_data.rows).createHostVec().getPtr();
        // int *h_rhs_cols = DeviceVec<int>(rhs_nnzb, rhs_bsr_data.cols).createHostVec().getPtr();
        // T *h_rhs_vals = RHS_mat->getVec().createHostVec().getPtr();
        // printf("locking-p-rhs mat with nnzb %d\n", rhs_nnzb);
        // printf("h_rhs_rowp: ");
        // printVec<int>(nnodes_fine + 1, h_rhs_rowp);
        // printf("h_rhs_cols: ");
        // printVec<int>(rhs_nnzb, h_rhs_cols);
        // for (int iblock = 0; iblock < rhs_nnzb; iblock++) {
        //     T *block_vals = &h_rhs_vals[36 * iblock];
        //     int node_row = h_rhs_rows[iblock], node_col = h_rhs_cols[iblock];
        //     printf("\n\nlocking-p-rhs block node (%d,%d):\n", node_row, node_col);
        //     // for (int i = 0; i < 6; i++) {
        //     //     printf("\t");
        //     //     printVec<T>(6, &block_vals[6 * i]);
        //     // }
        //     for (int i = 0; i < 3; i++) {
        //         int j = keep_dof[i];
        //         printf("\t");
        //         for (int i2 = 0; i2 < 3; i2++) {
        //             int j2 = keep_dof[i2];
        //             // all vals here should be either {0, 0.25, 0.5, 1} so let's round it so I can do quick compare
        //             T val = block_vals[6 * j + j2]; // I checked and there are sometimes like 1e-7 errors because I do optimization of (xi,eta) pairs
        //             val *= 32; // 1/2 the 64 of fine-fine matrix (because the coarse dimension has strain derivs one degree lower and I still have strains in param (xi,eta) not physical space (x,y))
        //             // val = std::round(val * 100) / 100; // rounds to two decimal places (so txt file comparison will work better)
        //             printf("%.6e  ", val);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // printf("\ndone with locking-p-rhs mat\n");
        
        // apply bcs to standard prolongator then add it into P_rhs
        // RHS_mat.add(lam, P0_mat); // make new add method here for P_rhs += lam * P_0
        printf("Step 6 - compute full RHS including lam*P_0 term\n");
        auto bsr_data = P_mat->getBsrData();
        // int P_nnzb = bsr_data.nnzb;
        int block_dim = bsr_data.block_dim;
        T *d_P_vals = P_mat->getPtr(), *d_RHS_vals = RHS_mat->getPtr();
        k_add_colored_submat_PFP<T>
            <<<P_nnzb, 64>>>(P_nnzb, block_dim, lam, 0, d_P_vals, d_RHS_vals);

        // do jacobi smoothing of P_0 => P matrix using kmat and rhs
        printf("Step 7 - perform block-Jacobi smoothing using locking energy for the prolongator\n");
        T omega_p = 0.5; // omega for prolongation
        // T omega_p = 0.3;
        // T omega_p = 0.9; // if spectral radius defined below
        auto lock_smoother = new LockingSmoother(cublasHandle, cusparseHandle, f_assembler, f_kmat, omega_p);
        // do CG-Lanczos for spectral radius
        // lock_smoother->setup_cg_lanczos(grids[i].d_defect, 10);

        // TBD : See unstruct prolongation class
        // use new ./include/lock_prolongation.h class here
        // int n_smooth_prolong = 6;
        // int n_smooth_prolong = 0;
        int n_smooth_prolong = nsmooth_mat;
        lock_smoother->smoothMatrix(n_smooth_prolong, prolong->prolong_mat, prolong->Z_mat,
                                prolong->RHS_mat, prolong->nnzb_prod,
                                prolong->d_K_prodBlocks, prolong->d_P_prodBlocks,
                                prolong->d_Z_prodBlocks);
        prolong->update_after_smooth(); // update coarse weights for nonlinear problems by row-sums of P^T

        T *h_P_vals = P_mat->getVec().createHostVec().getPtr();
        printf("final-prolongMat with nnzb %d\n", P_nnzb);
        printf("h_P_rowp: ");
        printVec<int>(nnodes_fine + 1, h_P0_rowp);
        printf("h_P_cols: ");
        printVec<int>(P_nnzb, h_P0_cols);
        for (int iblock = 0; iblock < P_nnzb; iblock++) {
            T *block_vals = &h_P_vals[36 * iblock];
            int node_row = h_P0_rows[iblock], node_col = h_P0_cols[iblock];
            printf("\n\nfinal-prolongMat block node (%d,%d):\n", node_row, node_col);
            // for (int i = 0; i < 6; i++) {
            //     printf("\t");
            //     printVec<T>(6, &block_vals[6 * i]);
            // }
            for (int i = 0; i < 3; i++) {
                int j = keep_dof[i];
                printf("\t");
                for (int i2 = 0; i2 < 3; i2++) {
                    int j2 = keep_dof[i2];
                    // all vals here should be either {0, 0.25, 0.5, 1} so let's round it so I can do quick compare
                    T val = block_vals[6 * j + j2]; // I checked and there are sometimes like 1e-7 errors because I do optimization of (xi,eta) pairs
                    // val = std::round(val * 100) / 100; // rounds to two decimal places (so txt file comparison will work better)
                    printf("%.6e  ", val);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\ndone with final prolong P0 mat\n");

        // re-assemble usual kmat (proceed with multigrid solve after that..)
        printf("Step 8 - reassemble kmat on grid %d\n", i);
        f_assembler.add_jacobian_fast(f_kmat);
        f_assembler.apply_bcs(f_kmat);
    }

    printf("DONE with lock-aware prolongation\n");


    // I do that explicitly right now..
    // NOTE : as of right now for this new locking prolongation it calls a dummy init_coarse_data method in here
    // and does zero smoothing matrix iterations (cause it doesn't have the auxillary smoother)
    // it uses main MG smoother which is ASW (additive schwarz) and that doesn't do matrix-smoothing
    // the only thing this method does is copy the prolong class to coarse grid as restrictor
    // which is needed call..
    kmg->template init_prolongations<Basis>();

    // ===========================================
    // end of locking aware prolongation
    // ===========================================

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    auto start1 = std::chrono::high_resolution_clock::now();
    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    bool print = true;
    // bool print = false;
    T atol = 1e-10, rtol = 1e-6;
    bool double_smooth = false;
    // bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)

    int n_cycles = 500; // max # cycles
    int print_freq = 3;

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    // fastest is K-cycle usually
    if (cycle_type == "V") {
        mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq); //(good option)
    } else if (cycle_type == "W") {
        mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
    } else if (cycle_type == "F") {
        mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq); // also decent
    } else if (cycle_type == "K") {
        kmg->solve(); // best
    }

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("plate GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    if (is_kcycle) {
        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_mg.vtk");
    } else {
        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/plate_mg.vtk");
    }
}

template <typename T, class Assembler>
void direct_solve(int nxe, double SR) {
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    int c_nxe = nxe;
    int c_nye = c_nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Basis, Physics>(c_nxe, c_nye, Lx, Ly, Q);
    printf("making grid with nxe %d\n", c_nxe);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian_fast(kmat);
    assembler.add_residual_fast(res);
    // assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    // solve the linear system
    CUSPARSE::direct_LU_solve(kmat, loads, soln);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;

    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    int ndof = assembler.get_num_vars();
    printf("plate direct solve, ndof %d : solve time %.2e, with mem (MB) %.2e\n", ndof, solve_time.count(), mem_mb);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, int nxe, double SR, int nsmooth, int ninnercyc, int nsmooth_mat, T omega, std::string cycle_type) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(nxe, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);
    } else {
        direct_solve<T, Assembler>(nxe, SR);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = true;
    // int nxe = 256; // default value
    int nxe = 4; // for comparison with python GMG
    double SR = 1e3; // default
    double omega = 0.2; // smaller omega for ASW

    int nsmooth = 2; // typically faster right now
    int ninnercyc = 1;
    int nsmooth_mat = 1; // more iterations not converging yet
    // int ninnercyc = 2; // inner V-cycles to precond K-cycle (ends up being a bit faster here..)
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    // std::string cycle_type = "V"; // "V", "F", "W", "K"

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--cycle") == 0) {
            if (i + 1 < argc) {
                cycle_type = argv[++i];
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth_mat") == 0) {
            if (i + 1 < argc) {
                nsmooth_mat = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth_mat\n";
                return 1;
            }
        } else if (strcmp(arg, "--ninnercyc") == 0) {
            if (i + 1 < argc) {
                ninnercyc = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    printf("plate mesh with MITC4-LP elements, nxe %d and SR %.2e\n------------\n", nxe, SR);
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);
    

    return 0;

    
}