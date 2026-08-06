// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"
#include "multigrid/utils/fea.h"
#include <string>
#include <chrono>
#include "multigrid/solvers/direct/cusp_directLU.h"

#include "domdec/bddc_assembler.h"
#include "multigrid/grid.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"
#include "multigrid/solvers/krylov/bsr_gmres.h"
#include "multigrid/solvers/krylov/bsr_gmres_matfree.h"

// AMG multigrid
#include "multigrid/amg/sa_amg.h"
#include "multigrid/amg/rn_amg.h"
#include "multigrid/amg/_rigid_modes.cuh"
#include "multigrid/solvers/krylov/bsr_pcg.h"

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

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

template <typename T>
T get_max_disp(HostVec<T> h_soln, int idof = 2) {
    T *h_soln_ptr = h_soln.getPtr();
    int nvars = h_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln_ptr[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}


// declare a couple different types of loads..
template <typename T>
struct UniformPressure {

    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        return 1.0;
    }
};

template <typename T>
struct ObliqueCylinderLoad {

    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        T x_hat = x / 1.0; // assumes L = 1.0
        T th = atan2(y, z);
        T th_hat = th / 2 / M_PI;
        T mag = 1.0e2 *
                (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                    0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);
        return mag;
    }
};



template <typename T>
struct ObliqueShearSineLoad {
    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        const T pi = T(3.14159265358979323846);

        T r = sqrt(x * x + y * y);
        T theta = atan2(y, x);

        return sin(T(5.0) * pi * r) * cos(T(4.0) * theta);
    }
};

int main(int argc, char **argv) {
    // NOTE : this version uses inner direct solvers

    using T = double;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC4
    using Quad = QuadLinearQuadrature<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;

    // MITC9, doesn't help the global solution recovery be more accurate
    // using Quad = QuadQuadraticQuadrature<T>;
    // using Basis = LagrangeQuadBasis<T, Quad, 2>;

    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    using BDDC = BddcSolver<T, Assembler, VecType, BsrMat>;
    using InnerSolver = CusparseMGDirectLU<T, Assembler>;
    using InnerSolver_JUSTLU = CusparseMGDirectLU<T, Assembler, true>;
    using DUMMY = InnerSolver;
    using GRID = SingleGrid<Assembler, DUMMY, DUMMY, NONE>; // GRID class largely unused
    using KIPCG = PCGSolver<T, GRID>;
    // using GamPCG = MatrixFreePCGSolver<T, BDDC>; // BDDC is the operator and preconditioner
    using GamGMRES = MatrixFreeGMRESSolver<T, BDDC, GRID, 100>;

    // AMG
    using FAssembler = FakeAssembler<T, Assembler>;
    using Smoother = ChebyshevPolynomialSmoother<FAssembler>; // uses fake assembler for smoother so can also build on coarser grids
    const bool ORTHOG_PROJECTOR = true;
    // using AMG = SmoothAggregationAMG<T, FAssembler, Smoother, ORTHOG_PROJECTOR>;
    using AMG = RootNodeAMG<T, FAssembler, Smoother, ORTHOG_PROJECTOR>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID2 = SingleGrid<FAssembler, Prolongation, Smoother, LINE_SEARCH>;
    using PCG = PCGSolver<T, GRID2>;

    // can't run this small a problem (1 vertex with S_VV coarse solver for some reason)
    // int nxe = 4, nxe_subdomain_size = 2;
    // int nxe = 6, nxe_subdomain_size = 2;
    // int nxe = 128, nxe_subdomain_size = 4; // this problem has optimal runtime for 4x4 subdomains
    // int nxe = 128, nxe_subdomain_size = 8;
    int nxe = 256, nxe_subdomain_size = 4;
    // int nxe = 256, nxe_subdomain_size = 8; // 8 subdomains slightly faster (cause shrinks coarse problem) for local + HPC
    // NOTE : full fillin with fill_level = -1, but lower fill results in less ILU(k) factor time
    // for the coarse problem..
    T omega;
    int nsmooth, fill_level;
    T thick = 1e-3;
    // bool print_mem = false;
    bool print_mem = true;
    T mag = 1.0e2;
    
    // optional smoothing
    // 1) if ILU(k) here, ability to do multiple smoothing steps (Richardson)
    // omega = 0.5, nsmooth = 2, fill_level = 2;
    // or single-level (no multiple smoothing) and ILU(k)
    // omega = 1.0, nsmooth = 1, fill_level = 2;
    // 2) or if full LU fillin
    // think I actually need to run with full direct fillin (cause otherwise solution recovery is wrong)
    omega = 1.0, nsmooth = 1, fill_level = -1; // -1 indicates full LU fillin

    // if (!MULTI_SMOOTH) {
    //     printf("NOTE: MULTI_SMOOTH is false, so omega, nsmooth inputs are ignored.\n");
    // }

    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--thick") == 0) {
            if (i + 1 < argc) {
                thick = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --thick\n";
                return 1;
            }
        } else if (strcmp(arg, "--mag") == 0) {
            if (i + 1 < argc) {
                mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --mag\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--subdomain") == 0) {
            if (i + 1 < argc) {
                nxe_subdomain_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --subdomain\n";
                return 1;
            }
        } else if (strcmp(arg, "--fill") == 0) {
            if (i + 1 < argc) {
                fill_level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --fill\n";
                return 1;
            }
        } else if (strcmp(arg, "--print_mem") == 0) {
            if (i + 1 < argc) {
                print_mem = (bool)std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --print_mem\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/krylov] [--nxe value] [--SR value] [--nsmooth int]" << std::endl;
            return 1;
        }
    }


    // =================

    int nye = nxe;
    int nxs = nxe / nxe_subdomain_size;
    int nys = nxe / nxe_subdomain_size;
    
    double SR = 1e3;
    double Lx = 1.0;
    double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye;
    double R = 0.5;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto assembler = createCylinderAssembler<Assembler>(nxe, nye, Lx, R, E, nu, thick, 
        imperfection, imp_x, imp_hoop);

    // auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto soln2 = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto loads = assembler.createVarsVec();

    // kmat assembly (not actually needed here, but is needed for the nonlinear problem))
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_assembly = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_assembly = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end_assembly - start_assembly;

    cublasHandle_t cublasHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // bool print_timing = true; // profiling
    bool print_timing = false;
    auto bddc = new BDDC(cublasHandle, cusparseHandle, assembler, kmat, print_timing);

    bool close_hoop = true; // true for cylinder case (not cylindrical panel)
    bddc->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);
    // bddc->setup_wing_subdomains(nxe_subdomain_size, nxe_subdomain_size); // debug this method (for wing case)

    // perform LU fillin and reordering (optional)
    auto &I_bsr_data = bddc->I_bsr_data;
    auto &IE_bsr_data = bddc->IE_bsr_data;
    I_bsr_data.AMD_reordering(); 
    I_bsr_data.compute_full_LU_pattern(10.0);
    IE_bsr_data.AMD_reordering(); 
    IE_bsr_data.compute_full_LU_pattern(10.0);

    // now compute matrix sparsity, copy maps
    // printf("setup matrix sparsity\n");
    bddc->setup_matrix_sparsity();
    // printf("\tdone with setup matrix sparsity\n");

    // then perform coarse matrix fillin and compute sparsity
    auto &Svv_bsr_data = bddc->Svv_bsr_data;
    // since using AMG
    Svv_bsr_data.compute_nofill_pattern();
    // Svv_bsr_data.AMD_reordering();
    // Svv_bsr_data.compute_full_LU_pattern(10.0);

    bddc->setup_coarse_matrix_sparsity();

    // assemble local FETI-DP blocks
    bddc->assemble_subdomains();

    // external load (can add internally)
    ObliqueCylinderLoad<T> load;
    // ObliqueShearSineLoad<T> load;
    bddc->add_subdomain_fext(load, mag);

    bddc->set_IEV_residual(1.0, 0.0, vars);

    // ----------------------------------------
    // you still need actual solver objects here
    // ----------------------------------------
    //
    // Example sketch only; replace with your actual solver classes:
    
    // setup direct solver
    auto *ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE, omega, nsmooth);
    auto *i_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->kmat_I, omega, nsmooth);
    // auto *v_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV, omega, nsmooth);

    // factor each solver
    ie_solver->factor();
    i_solver->factor();

    // also build the new K_II Krylov solver (subdomain parallel + needed for set rhs and soln recovery)
    SolverOptions amg_krylov_opts;
    amg_krylov_opts.ncycles = 100;
    // opts.ncycles = 500;
    amg_krylov_opts.print = true;
    // amg_krylov_opts.print = false;
    amg_krylov_opts.print_freq = 10;
    amg_krylov_opts.debug = true;
    // amg_krylov_opts.rtol = 1e-6;
    // amg_krylov_opts.rtol = 1e-2;
    amg_krylov_opts.rtol = 1e-3;
    amg_krylov_opts.atol = 1e-30;

    auto temp_v = nullptr;

    bddc->set_inner_solvers(ie_solver, i_solver, temp_v);
    // v_solver->factor();    

    // this builds Schur complement matrix
    bddc->assemble_coarse_problem();

    // NOW right here built the AMG problem
    // ===========================================

    T omegas = 0.2;
    int nsmooth_amg = 1;
    int ORDER = 4;

    auto Svv_mat = bddc->getCoarseSVVmat();
    int nnodes_coarse = Svv_bsr_data.nnodes;
    auto fake_coarse_assembler = FAssembler(Svv_bsr_data, nnodes_coarse);
    Smoother *coarse_smoother = new Smoother(cublasHandle, cusparseHandle, fake_coarse_assembler, 
        *Svv_mat, omegas, ORDER, nsmooth_amg);

    // get rigid body modes from coarse xpts
    auto d_coarse_xpts = bddc->getCoarseXpts();
    auto coarse_rbm = DeviceVec<T>(36 * nnodes_coarse); // each of 6 rigid body modes
    int block_dim = 6;
    k_compute_linear_rigid_body_modes<T><<<(nnodes_coarse + 31) / 32, 32>>>(nnodes_coarse, block_dim, 
        d_coarse_xpts.getPtr(), coarse_rbm.getPtr());
    
    int coarse_node_th = 100; // this value is problem dependent
    // T sparse_th = 1e-2;
    T sparse_th = 1e-3;
    // T sparse_th = 1e-4;
    auto d_bcs = DeviceVec<int>(0); // no bcs on interior vertex nodes
    int rbm_nsmooth = 2;
    T omegap = 0.25;
    // int nmat_smooth = 1;
    int nmat_smooth = 1;
    std::string coarsening_type = "standard"; // inactive for SA-AMG

    // top-level AMG solver (for solving coarse BDDC problem)
    // Svv_mat used for both kmat and kmat_free (cause no BCs, and only one instance of matrix, 
    //  this is fine, though matrix is duplicated unnecessarily)
    AMG *top_amg = new AMG(cublasHandle, cusparseHandle, coarse_smoother, nnodes_coarse, 
        *Svv_mat, *Svv_mat, coarse_rbm, d_bcs, 
        coarse_node_th, sparse_th, omegap, nsmooth, 0, 
        rbm_nsmooth, nmat_smooth, coarsening_type);

    // assist in making smoothers at coarser levels
    // printf("MAIN: build fine AMG solver\n");
    AMG *c_amg = top_amg;
    bool built_direct = false;
    while (c_amg != nullptr && (c_amg->is_coarse_mg || !built_direct)) {
        // build smoother for coarser problem (but it can't use assembler though..)
        auto c_bsr_data = c_amg->get_coarse_bsr_data();
        auto coarse_kmat = c_amg->get_coarse_kmat();
        int c_nnodes = c_amg->get_num_aggregates();
        printf("MAIN: build coarse system with %d aggregates\n", c_nnodes);
        auto fake_c_assembler = FAssembler(c_bsr_data, c_nnodes);
        Smoother *c_smoother = new Smoother(cublasHandle, cusparseHandle, fake_c_assembler, 
            coarse_kmat, omegas, ORDER, nsmooth);

        // build coarser system
        // printf("MAIN: build coarse system\n");
        c_amg->build_coarse_system(fake_c_assembler, c_smoother);
        // printf("\tMAIN: done building coarse system\n");

        if (!c_amg->get_coarse_mg()) {
            // factor coarse direct problem
            printf("\tfactoring coarse direct solver\n");
            c_amg->coarse_direct->factor();
            built_direct = true;
            break;
        } 

        // then set current amg (c_amg) to coarser problem
        c_amg = c_amg->coarse_mg;
        // if (c_amg != nullptr) c_amg->set_matrix_nsmooth(nmat_smooth);
    }

    auto coarse_loads = fake_coarse_assembler.createVarsVec();

    // build prolongation and fine grid also (unnecessary but required arg of PCG solver right now for some reason)
    auto prolongation = new Prolongation(fake_coarse_assembler);
    auto grid2 = new GRID2(fake_coarse_assembler, prolongation, coarse_smoother, *Svv_mat, 
        coarse_loads, cublasHandle, cusparseHandle);
    auto pc = top_amg; // topamg level used as preconditioner for AMG krylov
    int level = 0;
    auto v_amg_krylov = new PCG(cublasHandle, cusparseHandle, grid2, pc, amg_krylov_opts, level);
    

    // now set the last solver in (v_amg_krylov)
    bddc->set_inner_solvers(ie_solver, i_solver, v_amg_krylov);

    // =============================================


    // lambda rhs
    VecType<T> gam_rhs(bddc->getLambdaSize());
    VecType<T> gam(bddc->getLambdaSize());
    bddc->get_lam_rhs(gam_rhs);

    // matrix-free PCG for FETI-DP interface problem
    SolverOptions opts;
    // opts.ncycles = 2;
    // opts.ncycles = 50;
    opts.ncycles = 150;
    // opts.ncycles = 500;
    opts.print = true;
    opts.print_freq = 5;
    opts.debug = true;
    opts.rtol = 1e-6;
    opts.atol = 1e-30;


    // auto fine_prolongation = new Prolongation(assembler);
    // Smoother *fine_smoother = new Smoother(cublasHandle, cusparseHandle, assembler, 
    //     *kmat, omegas, ORDER, nsmooth_amg);
    auto grid = new GRID(assembler, nullptr, nullptr, kmat, 
        loads, cublasHandle, cusparseHandle);

    auto *gam_solver =
        new GamGMRES(cublasHandle, cusparseHandle, grid, bddc, bddc, opts, bddc->getLambdaSize());

    // optional: true initial residual before solve
    gam.zeroValues();
    T init_gam_resid = gam_solver->getResidualNorm(gam_rhs, gam);
    printf("initial gamma residual = %.8e\n", init_gam_resid);

    // ==============================================
    // SOLVE (TIMING)
    // ==============================================

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // run the assembly and factor (call from krylov solver, that also calls for preconditioner)
    gam_solver->update_after_assembly(vars);
        
    // end timing of the factor + assembly part
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> IEV_factor_time = start1 - start0;


    bool gam_fail = gam_solver->solve(gam_rhs, gam, true);
    bddc->get_global_soln(gam, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end - start1;
    std::chrono::duration<double> total_time = end - start0;

    T final_gam_resid = gam_solver->getResidualNorm(gam_rhs, gam);
    // printf("final lambda residual = %.8e in %.4e sec\n", final_gam_resid, solve_time.count());
    // printf("\nassembly in %.4e sec, IEV-assembly+factor in %.4e sec, PCG-solve in %.4e sec\n", assembly_time.count(), IEV_factor_time.count(), solve_time.count());
    // printf("\ttotal IEV-setup and PCG in %.4e sec\n", total_time.count());

    printf("\nBDDC solve summary:\n");
    printf("  final gamma residual : %.8e\n\n", final_gam_resid);

    printf("Timing breakdown:\n");
    printf("  assembly              : %.4e s\n", assembly_time.count());
    printf("  IEV assembly+factor   : %.4e s\n", IEV_factor_time.count());
    printf("  PCG solve             : %.4e s\n", solve_time.count());
    printf("  --------------------------------\n");
    printf("  total setup + solve   : %.4e s\n\n", total_time.count());

    int kmat_nnzb;

    if (print_mem) {
        // get memory usage
        size_t bytes_per_double = sizeof(double);
        double bytes_per_block = static_cast<double>(bytes_per_double) * 36.0;

        // nnzb counts
        kmat_nnzb   = kmat.getBsrData().nnzb;
        int IEV_nnzb    = bddc->kmat_IEV->getBsrData().nnzb;
        int IE_nnzb     = IE_bsr_data.nnzb;
        int I_nnzb      = I_bsr_data.nnzb;
        // int coarse_nnzb = Svv_bsr_data.nnzb;
        int coarse_nnzb = top_amg->get_total_nnzb();

        // no-fill counts (must already exist in your code)
        int IE_nofill_nnzb     = bddc->IE_nofill_nnzb;
        int I_nofill_nnzb      = bddc->I_nofill_nnzb;
        int coarse_nofill_nnzb = bddc->Svv_nofill_nnzb;   // or whatever your variable is called

        // memory in MB
        double kmat_mem_mb   = bytes_per_block * static_cast<double>(kmat_nnzb)   / 1024.0 / 1024.0;
        double IEV_mem_mb    = bytes_per_block * static_cast<double>(IEV_nnzb)    / 1024.0 / 1024.0;
        double IE_mem_mb     = bytes_per_block * static_cast<double>(IE_nnzb)     / 1024.0 / 1024.0;
        double I_mem_mb      = bytes_per_block * static_cast<double>(I_nnzb)      / 1024.0 / 1024.0;
        double coarse_mem_mb = bytes_per_block * static_cast<double>(coarse_nnzb) / 1024.0 / 1024.0;

        // total stored blocks:
        //   kmat * 1
        //   IEV  * 1
        //   IE   * 2
        //   I    * 1
        //   coarse * 1
        long long total_stored_nnzb =
            static_cast<long long>(kmat_nnzb) +
            static_cast<long long>(IEV_nnzb) +
            2LL * static_cast<long long>(IE_nnzb) +
            static_cast<long long>(I_nnzb) +
            static_cast<long long>(coarse_nnzb);

        double total_mem_mb =
            bytes_per_block * static_cast<double>(total_stored_nnzb) / 1024.0 / 1024.0;

        // fill ratios
        double IE_fill_ratio =
            (IE_nofill_nnzb > 0) ? static_cast<double>(IE_nnzb) / static_cast<double>(IE_nofill_nnzb) : 0.0;
        double I_fill_ratio =
            (I_nofill_nnzb > 0) ? static_cast<double>(I_nnzb) / static_cast<double>(I_nofill_nnzb) : 0.0;
        double coarse_fill_ratio =
            (coarse_nofill_nnzb > 0) ? static_cast<double>(coarse_nnzb) / static_cast<double>(coarse_nofill_nnzb) : 0.0;

        // optional: added fill blocks
        int IE_fill_added     = IE_nnzb - IE_nofill_nnzb;
        int I_fill_added      = I_nnzb - I_nofill_nnzb;
        int coarse_fill_added = coarse_nnzb - coarse_nofill_nnzb;
        T overall_fill_ratio = total_stored_nnzb * 1.0 / kmat_nnzb;

        printf("\nBDDC mem: total %.2f MB (nnzb=%lld, fill=%.2f)\n",
            total_mem_mb, total_stored_nnzb, overall_fill_ratio);

        printf("  K:%d | IEV:%d | IE:%d(%.2f) | I:%d(%.2f) | SVV:%d(%.2f)\n",
            kmat_nnzb, IEV_nnzb, IE_nnzb, IE_fill_ratio,
            I_nnzb, I_fill_ratio, coarse_nnzb, coarse_fill_ratio);

    }


    if (gam_fail) {
        printf("BDDC lambda PCG failed\n");
    }

    auto h_soln = soln.createHostVec();
    printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/plate_bddc.vtk");

    // compare to direct solver (the solution)
    bool compare_direct = true;
    // bool compare_direct = false;
    if (compare_direct) {
        // and need it globally too..
        auto loads = assembler.createVarsVec();
        assembler.add_fext_fast(load, mag, loads);
        assembler.apply_bcs(loads);

        auto assembler2 = createCylinderAssembler<Assembler>(nxe, nye, Lx, R, E, nu, thick, 
            imperfection, imp_x, imp_hoop);

        // BSR factorization (need to change it to )
        auto& bsr_data = assembler2.getBsrData();
        double fillin = 10.0;  // 10.0
        bool print = true;
        bsr_data.AMD_reordering();
        bsr_data.compute_full_LU_pattern(fillin, print);
        assembler2.moveBsrDataToDevice();


        auto kmat2 = createBsrMat<Assembler, VecType<T>>(assembler2);
        assembler2.add_jacobian_fast(kmat2);
        assembler2.apply_bcs(kmat2);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto start3 = std::chrono::high_resolution_clock::now();

        CUSPARSE::direct_LU_solve(kmat2, loads, soln2);

        CHECK_CUDA(cudaDeviceSynchronize());
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solve_time_dir = end3 - start3;

        size_t bytes_per_double = sizeof(double);
        double bytes_per_block = static_cast<double>(bytes_per_double) * 36.0;
        int LU_nnzb = kmat2.getBsrData().nnzb;
        double LU_mat_mem   = bytes_per_block * static_cast<double>(LU_nnzb)   / 1024.0 / 1024.0;
        double LU_fill = LU_nnzb * 1.0 / kmat_nnzb;
        int nvars = assembler2.get_num_vars();

        printf("BDDC solve time in %.4e sec, direct in %.4e sec\n", total_time.count(), solve_time_dir.count());
        printf("\tLU mat mem %.2f MB (nnzb=%d, fill=%.2f)\n", LU_mat_mem, LU_nnzb, LU_fill);
        printf("\t#DOF = %d\n", nvars);

        // T lin_max_disp = get_max_disp(soln2);
        auto h_soln3 = soln2.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler2, h_soln3, "out/plate_lin.vtk");

        // now also compute solution error on host and print to VTK
        auto h_err = HostVec<T>(h_soln3.getSize());
        for (int i = 0; i < h_soln3.getSize(); i++) {
            h_err[i] = h_soln3[i] - h_soln[i];
        }
        printToVTK<Assembler,HostVec<T>>(assembler2, h_err, "out/plate_err.vtk");

        // for (int idof = 0; idof < 6; idof++) {
        //     T orig_nrm = get_max_disp(h_soln, idof);
        //     T err_nrm = get_max_disp(h_err, idof);
        //     T rel_nrm = err_nrm / (orig_nrm + 1e-30);
        //     printf("\tidof %d, orig |u|=%.4e, err nrm %.4e, rel err nrm %.4e to direct solve\n", idof, orig_nrm, err_nrm, rel_nrm);
        // }
        
        // now compute the residuals of each..
        assembler2.set_variables(soln2);
        assembler2.add_residual_fast(res);
        T a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler2.apply_bcs(res);
        T load_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, loads.getPtr(), 1, &load_nrm));
        T res_norm_direct;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, res.getPtr(), 1, &res_norm_direct));
        T res_rel_nrm_direct = res_norm_direct / load_nrm;
        printf("\tres_nrm_direct %.4e, rel_nrm %.4e\n", res_norm_direct, res_rel_nrm_direct);

        assembler2.set_variables(soln);
        res.zeroValues();
        assembler2.add_residual_fast(res);
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler2.apply_bcs(res);
        T res_norm_feti;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, res.getPtr(), 1, &res_norm_feti));
        T res_rel_nrm_feti = res_norm_feti / load_nrm;
        printf("\tres_nrm_feti %.4e, rel_nrm %.4e\n", res_norm_feti, res_rel_nrm_feti);
        
    }

    delete gam_solver;
    // delete ie_solver;
    // delete i_solver;
    // delete v_solver;
    delete bddc;


    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}
