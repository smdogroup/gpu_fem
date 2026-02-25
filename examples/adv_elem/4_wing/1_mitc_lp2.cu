// """
// wing geometric multigrid with MITC4 shells
// * uses MITC-EP (locking-smooth prolongation)
// * 2x2 node Element-ASW (additive schwarz subdomain smoother)
// * default is K-cycle with V-cycle precond (PCG solver), other options V-cycle, K-cycle and F-cycle solver
// """

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
#include "../2_plate/include/lock_prolongation.h"
#include "../2_plate/include/lock_smoother.h"

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
void multigrid_solve(MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, int nsmooth_mat, T omega, std::string cycle_type) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
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

    // for some reason better not with LS constraints on wing.. ?
    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    // T omegaLS_min = 1e-2, omegaLS_max = 4.0;

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

    // make each grid
    for (int i = level; i >= 0; i--) {
        
        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "../../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        // TODO : run optimized design from AOB case
        printf("making assembler+GMG for mesh '%s' => ", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // do multicolor junction reordering
        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        bool coarsest_grid = i == 0;
        if (!coarsest_grid) {
            // don't do coloring for additive schwarz smoother..
            // WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
            bsr_data.compute_nofill_pattern();
        } else {
            // full LU pattern for coarsest grid
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
            num_colors = 0;
            _color_rowp = new int[2];
            _color_rowp[0] = 0, _color_rowp[1] = nnodes;
        }
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
        assembler.moveBsrDataToDevice();


        // now compute loads, bcs and assemble kmat
        auto loads = assembler.createVarsVec(my_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto vars = assembler.createVarsVec();
        assembler.set_variables(vars);
        auto res = assembler.createVarsVec();
        auto starta = std::chrono::high_resolution_clock::now();
        // assembler.add_jacobian(res, kmat);
        const int elems_per_blockk = 1; // 1 versus 2 elements => similar runtime (1 slightly better)
        // const int elems_per_blockk = 2;
        assembler.template add_jacobian_fast<elems_per_blockk>(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto enda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = enda - starta;
        printf("assemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations..
        // printf("nsmooth %d, omega = %.4e\n", nsmooth, omega);
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
            omega, nsmooth);
        int ELEM_MAX = 10;
        auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max);
        
        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (coarsest_grid) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
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
        // double lam = 1e-12; // see python locking script
        // double lam = 1e-6;
        // if (is_kcycle) {
        // clear and re-assemble kmat to hold G_f^T G_f + lam * I
        auto &f_assembler = grids[i].assembler;
        auto &f_kmat = grids[i].Kmat;
        auto d_fine_bcs = f_assembler.getBCs();

        // printf("Step 1 - compute G_f^T G_f LHS locking matrix\n");
        // f_assembler.add_lockstrain_jacobian_fast(f_kmat);
        // CHECK_CUDA(cudaDeviceSynchronize()); // slower but for debugging
        // f_assembler.apply_bcs(f_kmat);
        // f_kmat.add_diag_nugget(lam);

        // make device fine-coarse elem map (here just use structured pattern)
        printf("Step 2 - compute device fc elem map\n");
        int num_fine_elements = f_assembler.get_num_elements();
        int *h_fc_elem_map = new int[num_fine_elements];
        memset(h_fc_elem_map, 0, num_fine_elements * sizeof(int));
        // int c_nxe = (int)sqrt(num_fine_elements);
        // int c_nxec = c_nxe / 2;
        // for (int ielem = 0; ielem < num_fine_elements; ielem++) {
        //     int ixe = ielem % c_nxe, iye = ielem / c_nxe;
        //     int ixe_c = ixe / 2, iye_c = iye / 2;
        //     int ielem_c = c_nxec * iye_c + ixe_c;
        //     h_fc_elem_map[ielem] = ielem_c;
        // }
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

        // // now assemble G_f^T * P_gam * G_c + lam * P_0  RHS prolong matrix with K*P0 sparsity
        // printf("Step 5 - compute locking RHS fine-coarse matrix\n");
        // f_assembler.add_lockstrain_fc_jacobian_fast(c_assembler, d_fc_elem_map, *RHS_mat);
        // CHECK_CUDA(cudaDeviceSynchronize()); // slower but for debugging
        // printf("\tdone with assembly FC matrix from step 5\n");
        // // apply bcs to P_rhs matrix
        // RHS_mat->template apply_bc_rows<ones_on_diag>(d_fine_bcs);
        // RHS_mat->template apply_bc_cols<ones_on_diag>(d_coarse_bcs);
        
        // // apply bcs to standard prolongator then add it into P_rhs
        // // RHS_mat.add(lam, P0_mat); // make new add method here for P_rhs += lam * P_0
        // printf("Step 6 - compute full RHS including lam*P_0 term\n");
        // auto bsr_data = P_mat->getBsrData();
        // int P_nnzb = bsr_data.nnzb, block_dim = bsr_data.block_dim;
        // T *d_P_vals = P_mat->getPtr(), *d_RHS_vals = RHS_mat->getPtr();
        // k_add_colored_submat_PFP<T>
        //     <<<P_nnzb, 64>>>(P_nnzb, block_dim, lam, 0, d_P_vals, d_RHS_vals);

        // do jacobi smoothing of P_0 => P matrix using kmat and rhs
        printf("Step 7 - perform block-Jacobi smoothing using locking energy for the prolongator\n");
        // T omega_p = 0.5;
        // T omega_p = 0.3; // omega for prolongation
        // T omega_p = 0.1;

        T omega_p = 0.9;
        auto lock_smoother = new LockingSmoother(cublasHandle, cusparseHandle, f_assembler, f_kmat, omega_p);
        // do CG-Lanczos for spectral radius
        lock_smoother->setup_cg_lanczos(grids[i].d_defect, 10);

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

        // // re-assemble usual kmat (proceed with multigrid solve after that..)
        // printf("Step 8 - reassemble kmat on grid %d\n", i);
        // f_assembler.add_jacobian_fast(f_kmat);
        // f_assembler.apply_bcs(f_kmat);
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

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();
    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    bool print = true;
    // bool print = false;
    T atol = 1e-10, rtol = 1e-6;
    int print_freq = 3;
    int n_cycles = 500;

    bool double_smooth = false;
    // bool double_smooth = true; // true tends to be slightly faster sometimes

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, 
            n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    if (is_kcycle) {
        int nlevels = kmg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            kmg->grids[i].smoother->factor();
        } 
    } else {
        int nlevels = mg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            mg->grids[i].smoother->factor();
        } 
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> asw_factor_time = end_factor - start1;
    printf("ASW factor time %.4e\n", asw_factor_time.count());

    // fastest is K-cycle usually
    if (cycle_type == "V") {
        mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); //(good option)
    } else if (cycle_type == "W") {
        mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
    } else if (cycle_type == "F") {
        mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); // also decent
    } else if (cycle_type == "K") {
        kmg->solve();
    }
    

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    if (is_kcycle) {
        // double check with true resid nrm
        T resid_nrm = kmg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    } else {
        // double check with true resid nrm
        T resid_nrm = mg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    }
}


template <typename T, class Assembler>
void solve_linear_direct(MPI_Comm &comm, int level, double SR) {
  
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "../../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
  mesh_loader.scanBDFFile(fname.c_str());

  //   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
  double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // TODO : set this in from optimized design from AOB case

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();

//   // TRY INSTEAD Mc REORDERING
//   int num_colors, *_color_rowp, *nodal_num_comps, *node_geom_ind;
//   GRID::get_nodal_geom_indices(assembler, nodal_num_comps, node_geom_ind);
//   bsr_data.multicolor_junction_reordering_v2(node_geom_ind, num_colors, _color_rowp);

  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 10.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  auto starta = std::chrono::high_resolution_clock::now();
//   assembler.add_jacobian(res, kmat);
  assembler.add_jacobian_fast(kmat);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto enda = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> assemb_time = enda - starta;

  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> startup_time = start1 - start0;

  // solve the linear system
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end1 - start1;
  std::chrono::duration<double> total_time = end1 - start0;

  size_t bytes_per_double = sizeof(double);
  double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
  printf("direct LU solve on #dof %d, uses memory(MB) %.2e\n", nvars, mem_mb);
  printf("\tassembly %.2e and ovr startup %.2e, solve time %.2e and total time %.2e (sec)\n", assemb_time.count(), startup_time.count(), solve_time.count(), total_time.count());

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/aob_direct_L" + std::to_string(level) + ".vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, 
    int nsmooth_mat, T omega, std::string cycle_type) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);
    } else {
        solve_linear_direct<T, Assembler>(comm, level, SR);
    }
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // input ----------
    bool is_multigrid = true;
    int level = 2;
    double SR = 1e3; // default
    double omega = 0.2; // smaller omega for ASW

    int nsmooth = 4; // typically faster right now
    int ninnercyc = 1;
    int nsmooth_mat = 2; // more iterations not converging yet
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
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
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

    printf("AOB wing mesh with MITC4-LP elements, level %d and SR %.2e\n------------\n", level, SR);
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);    

    
    MPI_Finalize();
    return 0;    
}