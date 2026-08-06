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
#include "multigrid/smoothers/asw_unstruct.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/krylov/bsr_pcg_matfree.h"

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
    T q0;

    __HOST_DEVICE__
    UniformPressure(T q0_) : q0(q0_) {}

    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        return q0;
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
    using GamPCG = MatrixFreePCGSolver<T, BDDC>; // BDDC is the operator and preconditioner

    // AMG
    using FAssembler = FakeAssembler<T, Assembler>;
    using ASW = UnstructuredQuadElementAdditiveSchwarzSmoother<T, Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID2 = SingleGrid<FAssembler, Prolongation, ASW, LINE_SEARCH>;
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
    T mag = 1.0;
    
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
    double Lx = 1.0, Ly = 1.0;
    double E = 70e9, nu = 0.3, rho = 2500, ys = 350e6;
    // int nxe_per_comp = nxe, nye_per_comp = nye;
    int nxe_per_comp = nxe, nye_per_comp = nye;

    auto assembler = createPlateClampedAssembler<Assembler>(
        nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

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

    bool close_hoop = false; // true for cylinder case (not cylindrical panel)
    // bddc->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);
    // // bddc->setup_wing_subdomains(nxe_subdomain_size, nxe_subdomain_size); // debug this method (for wing case)

    // 2x2 subdomains on coarse BDDC problem
    int nxs2 = nxs / 2; // num subdomains in x-direction (2x fewer on coarse problem)
    int nys2 = nys / 2;
    int coarse_num_elements, coarse_num_nodes, coarse_elem_nnz;
    int *coarse_elem_ptr, *coarse_elem_conn, *coarse_elem_sd_ind;
    printf("setup coarse structured subdomains\n");
    bddc->setup_coarse_structured_subdomains(nxe, nye, nxs, nys, nxs2, nys2, close_hoop, 
        coarse_num_elements, coarse_num_nodes, coarse_elem_nnz, coarse_elem_ptr, 
        coarse_elem_conn, coarse_elem_sd_ind);
    printf("\tdone with setup coarse structured subdomains\n");

    // then setup structured subdomains again?
    // bool close_hoop = false; // true for cylinder case (not cylindrical panel)
    printf("setup structured subdomains\n");
    bddc->setup_structured_subdomains(nxe, nye, nxs, nys, close_hoop);
    printf("\tdone with setup structured subdomains\n");

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
    ObliqueShearSineLoad<T> load;
    bddc->add_subdomain_fext(load, mag);

    bddc->set_IEV_residual(1.0, 0.0, vars);

    // ----------------------------------------
    // you still need actual solver objects here
    // ----------------------------------------
    //
    // Example sketch only; replace with your actual solver classes:

    // get only elements in coarse_elem_conn with 4 nodes_per_elem
    std::vector<int> coarse_elem_conn2_vec;
    int nec = 0;
    for (int e = 0; e < coarse_num_elements; e++) {
        int start = coarse_elem_ptr[e];
        int loc_nodes_per_elem = coarse_elem_ptr[e+1] - start;
        if (loc_nodes_per_elem == 4) {
            nec++;
            for (int l = 0; l < 4; l++) {
                int n = coarse_elem_conn[start+l];
                coarse_elem_conn2_vec.push_back(n);
            }
        }
    }
    int *coarse_elem_conn2 = new int[4 * nec];
    for (int v = 0; v < 4 * nec; v++) {
        coarse_elem_conn2[v] = coarse_elem_conn2_vec[v];
    }
    // printf("coarse elem conn2: ");
    // printVec<int>(4 * nec, coarse_elem_conn2);
    
    // setup direct solver
    auto *ie_solver = new InnerSolver(cublasHandle, cusparseHandle, assembler, *bddc->kmat_IE, omega, nsmooth);
    auto *i_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->kmat_I, omega, nsmooth);
    // auto *v_solver  = new InnerSolver_JUSTLU(cublasHandle, cusparseHandle, assembler, *bddc->S_VV, omega, nsmooth);

    // factor each solver
    ie_solver->factor();
    i_solver->factor();

    auto temp_v = nullptr;

    bddc->set_inner_solvers(ie_solver, i_solver, temp_v);
    // v_solver->factor();    

    // this builds Schur complement matrix
    bddc->assemble_coarse_problem();

    // NOW right here built the AMG problem
    // ===========================================

    T omega_asw = 0.15;
    int nsmooth_asw = 2;

    auto Svv_mat = bddc->getCoarseSVVmat();
    int nnodes_coarse = Svv_bsr_data.nnodes;
    printf("nnodes_coarse = %d vs coarse_num_nodes = %d, nec = %d\n", nnodes_coarse, coarse_num_nodes, nec);

    auto d_coarse_elem_conn = HostVec<int>(4 * nec, coarse_elem_conn2).createDeviceVec();

    int block_dim = 6;
    printf("make ASW\n");
    auto asw = new ASW(cublasHandle, cusparseHandle, nec, d_coarse_elem_conn, 
        coarse_num_nodes * block_dim, block_dim, *Svv_mat, omega_asw, nsmooth_asw);
    printf("\tdone with make ASW\n");

    auto fake_coarse_assembler = FAssembler(Svv_bsr_data, nnodes_coarse);
    auto coarse_loads = fake_coarse_assembler.createVarsVec();

    SolverOptions asw_krylov_opts;
    asw_krylov_opts.ncycles = 100;
    // opts.ncycles = 500;
    asw_krylov_opts.print = true;
    // amg_krylov_opts.print = false;
    asw_krylov_opts.print_freq = 10;
    asw_krylov_opts.debug = true;
    // amg_krylov_opts.rtol = 1e-6;
    // amg_krylov_opts.rtol = 1e-2;
    asw_krylov_opts.rtol = 1e-4;
    asw_krylov_opts.atol = 1e-30;

    // build prolongation and fine grid also (unnecessary but required arg of PCG solver right now for some reason)
    auto prolongation = new Prolongation(fake_coarse_assembler);
    auto grid2 = new GRID2(fake_coarse_assembler, prolongation, asw, *Svv_mat, 
        coarse_loads, cublasHandle, cusparseHandle);
    auto pc = asw; // topamg level used as preconditioner for AMG krylov
    int level = 0;
    auto v_amg_krylov = new PCG(cublasHandle, cusparseHandle, grid2, pc, asw_krylov_opts, level);
    

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

    auto *gam_solver =
        new GamPCG(cublasHandle, bddc, bddc, opts, bddc->getLambdaSize(), 0);

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
        int coarse_nnzb = Svv_bsr_data.nnzb;

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

        auto assembler2 = createPlateClampedAssembler<Assembler>(
            nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

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
