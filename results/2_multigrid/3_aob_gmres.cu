// Standard library
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

// General gpu_fem imports
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// Shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// Lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// Chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// Multigrid imports
#include "multigrid/grid.h"
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/smoothers/asw_unstruct.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/solvers/gmg.h"
#include "multigrid/utils/fea.h"

// K-cycle and Krylov imports
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_gmres.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"
#include "multigrid/solvers/solve_utils.h"

/*
Supported smoothers:
    asw
    gsmc
    chebyshev
    jacobi
    direct

Supported multigrid cycles:
    V
    W
    F
    VK
    WK
    FK
    FSK

Examples:
    ./aob_wing.out --smoother asw --cycle VK --level 3 --sr 300 \
        --omega 0.15 --nsmooth 4

    ./aob_wing.out --smoother asw --cycle F --level 3 --sr 300 \
        --omega 0.15 --nsmooth 4

    ./aob_wing.out --smoother gsmc --cycle FK --level 3 --sr 300 \
        --omega 1.0 --nsmooth 2

    ./aob_wing.out --smoother direct --level 3 --sr 300
*/

std::string lowercase_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

    return value;
}

std::string uppercase_copy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

    return value;
}

bool is_plain_multigrid_cycle(const std::string &cycle_type) {
    return cycle_type == "V" ||
           cycle_type == "W" ||
           cycle_type == "F";
}

bool is_krylov_multigrid_cycle(const std::string &cycle_type) {
    return cycle_type == "VK" ||
           cycle_type == "WK" ||
           cycle_type == "FK" ||
           cycle_type == "FSK";
}

bool is_supported_cycle(const std::string &cycle_type) {
    return is_plain_multigrid_cycle(cycle_type) ||
           is_krylov_multigrid_cycle(cycle_type);
}

bool is_supported_smoother(const std::string &smoother_type) {
    return smoother_type == "asw" ||
           smoother_type == "gsmc" ||
           smoother_type == "chebyshev" ||
           smoother_type == "jacobi" ||
           smoother_type == "direct";
}

void print_usage(const char *program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name
        << " [--level int]"
        << " [--sr double]"
        << " [--omega double]"
        << " [--smoother type]"
        << " [--cycle type]"
        << " [--nsmooth int]"
        << " [--ninnercyc int]"
        << " [--order int]\n\n"
        << "Smoothers:\n"
        << "  asw, gsmc, chebyshev, jacobi, direct\n\n"
        << "Cycles:\n"
        << "  V, W, F, VK, WK, FK, FSK\n";
}

template <typename T, class Smoother, class Assembler>
void multigrid_solve(
    MPI_Comm &comm,
    int level,
    const std::string &smoother_type,
    double SR,
    int nsmooth,
    int ninnercyc,
    T omega,
    int chebyshev_order,
    const std::string &cycle_type) {

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

    constexpr SCALER scaler = LINE_SEARCH;
    constexpr bool is_bsr = false;

    using Prolongation =
        UnstructuredProlongation<Assembler, Basis, is_bsr>;

    using GRID =
        SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    using CoarseSolver =
        CusparseMGDirectLU<T, Assembler>;

    using MG =
        GeometricMultigridSolver<GRID, CoarseSolver>;

    using ASW =
        UnstructuredQuadElementAdditiveSchwarzSmoother<
            T,
            Assembler,
            false>;

    constexpr int N_SUBSPACE = 100;

    using KrylovSolve =
        GMRESSolver<T, GRID, N_SUBSPACE>;

    using TwoLevelSolve =
        MultigridTwoLevelSolver<GRID>;

    using KMG =
        MultilevelKcycleSolver<
            GRID,
            CoarseSolver,
            TwoLevelSolve,
            KrylovSolve>;

    const bool use_kmg = is_krylov_multigrid_cycle(cycle_type);

    MG *mg = nullptr;
    KMG *kmg = nullptr;

    if (use_kmg) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto startup_begin =
        std::chrono::high_resolution_clock::now();

    /*
    Build hierarchy from the finest mesh to the coarsest mesh.

    The grids are therefore stored as:

        grids[0]       = finest
        grids[level]   = coarsest
    */
    for (int mesh_level = level; mesh_level >= 0; --mesh_level) {
        TACSMeshLoader mesh_loader{comm};

        const std::string mesh_filename =
            "../../examples/gmg/3_aob_wing/meshes/"
            "aob_wing_L" +
            std::to_string(mesh_level) +
            ".bdf";

        mesh_loader.scanBDFFile(mesh_filename.c_str());

        const double E = 70.0e9;
        const double nu = 0.3;
        const double thickness = 2.0 / SR;

        printf(
            "making assembler + GMG grid for mesh '%s'\n",
            mesh_filename.c_str());

        auto assembler =
            Assembler::createFromBDF(
                mesh_loader,
                Data(E, nu, thickness));

        const int nvars = assembler.get_num_vars();
        const int nnodes = assembler.get_num_nodes();

        /*
        Create the load vector in the assembler's original nodal ordering.

        createVarsVec() handles conversion into the active solver ordering
        after the BSR reordering has been established.
        */
        HostVec<T> h_loads(nvars);
        T *h_loads_ptr = h_loads.getPtr();

        const T load_magnitude = static_cast<T>(10.0);

        for (int inode = 0; inode < nnodes; ++inode) {
            h_loads_ptr[6 * inode + 2] = load_magnitude;
        }

        auto &bsr_data = assembler.getBsrData();

        int num_colors = 0;
        int *color_rowp = nullptr;

        const bool coarsest_grid = mesh_level == 0;

        if (!coarsest_grid) {
            /*
            The wingbox coloring changes the nodal solver ordering.

            Apply it only for the multicolor Gauss-Seidel smoother.

            In particular, do not apply this reordering for ASW. The
            element-based ASW setup worked previously with the original
            assembler ordering and no multicolor permutation.
            */
            if constexpr (
                std::is_same_v<
                    Smoother,
                    MulticolorGSSmoother_V1<Assembler>>) {

                printf(
                    "\tapplying wingbox multicolor ordering "
                    "on level %d\n",
                    mesh_level);

                WingboxMultiColoring<Assembler>::apply_coloring(
                    assembler,
                    bsr_data,
                    num_colors,
                    color_rowp);
            } else {
                printf(
                    "\tpreserving original ordering "
                    "on level %d\n",
                    mesh_level);

                num_colors = 0;
                color_rowp = new int[2];

                color_rowp[0] = 0;
                color_rowp[1] = nnodes;
            }

            bsr_data.compute_nofill_pattern();
        } else {
            /*
            The coarsest level uses a direct LU solve, so use AMD and
            construct the full LU sparsity pattern.
            */
            printf(
                "\tapplying AMD/full-LU ordering "
                "on coarse level %d\n",
                mesh_level);

            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);

            num_colors = 0;
            color_rowp = new int[2];

            color_rowp[0] = 0;
            color_rowp[1] = nnodes;
        }

        auto h_color_rowp =
            HostVec<int>(num_colors + 1, color_rowp);

        assembler.moveBsrDataToDevice();

        auto loads = assembler.createVarsVec(h_loads_ptr);
        assembler.apply_bcs(loads);

        auto kmat =
            createBsrMat<Assembler, VecType<T>>(assembler);

        auto vars = assembler.createVarsVec();
        assembler.set_variables(vars);

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto assembly_begin =
            std::chrono::high_resolution_clock::now();

        constexpr int elems_per_block = 1;

        assembler
            .template add_jacobian_fast<elems_per_block>(kmat);

        assembler.apply_bcs(kmat);

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto assembly_end =
            std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double> assembly_time =
            assembly_end - assembly_begin;

        printf(
            "\tassembled K in %.4e sec\n",
            assembly_time.count());

        /*
        Construct the selected smoother.

        Do not factor the smoother here. Factor every smoother exactly
        once after the complete hierarchy and prolongation relationships
        have been initialized.
        */
        Smoother *smoother = nullptr;

        if constexpr (
            std::is_same_v<
                Smoother,
                ChebyshevPolynomialSmoother<
                    Assembler,
                    false>>) {

            const int polynomial_order =
                smoother_type == "jacobi"
                    ? 1
                    : chebyshev_order;

            smoother = new Smoother(
                cublas_handle,
                cusparse_handle,
                assembler,
                kmat,
                omega,
                polynomial_order,
                nsmooth);
        } else if constexpr (
            std::is_same_v<
                Smoother,
                MulticolorGSSmoother_V1<Assembler>>) {

            constexpr bool symmetric = true;

            smoother = new Smoother(
                cublas_handle,
                cusparse_handle,
                assembler,
                kmat,
                h_color_rowp,
                omega,
                symmetric,
                nsmooth);
        } else if constexpr (
            std::is_same_v<Smoother, ASW>) {

            printf(
                "\tconstructing ASW smoother on level %d\n",
                mesh_level);

            smoother = new Smoother(
                cublas_handle,
                cusparse_handle,
                assembler,
                kmat,
                omega,
                nsmooth);
        } else {
            static_assert(
                !std::is_same_v<Smoother, Smoother>,
                "Unsupported smoother type");
        }

        constexpr int ELEM_MAX = 10;

        auto prolongation = new Prolongation(
            cusparse_handle,
            assembler,
            ELEM_MAX);

        auto grid = GRID(
            assembler,
            prolongation,
            smoother,
            kmat,
            loads,
            cublas_handle,
            cusparse_handle);

        if (use_kmg) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);

            if (coarsest_grid) {
                mg->coarse_solver = new CoarseSolver(
                    cublas_handle,
                    cusparse_handle,
                    assembler,
                    kmat);
            }
        }
    }

    /*
    Register adjacent coarse assemblers with the corresponding
    prolongation operators.
    */
    if (use_kmg) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }

    const int pre_smooth = nsmooth;
    const int post_smooth = nsmooth;

    constexpr bool print_iterations = true;
    constexpr bool double_smooth = false;
    constexpr bool print_cycle_times = false;

    const T atol = static_cast<T>(1.0e-20);
    const T rtol = static_cast<T>(1.0e-6);

    constexpr int max_cycles = 500;
    constexpr int print_frequency = 3;
    constexpr int max_krylov_iterations = 500;

    if (use_kmg) {
        kmg->init_outer_solver(
            cublas_handle,
            cusparse_handle,
            nsmooth,
            ninnercyc,
            max_krylov_iterations,
            omega,
            atol,
            rtol,
            print_frequency,
            print_iterations,
            double_smooth);

        if (cycle_type == "VK") {
            kmg->set_cycle_type("V");
        } else if (cycle_type == "WK") {
            kmg->set_cycle_type("W");
        } else if (cycle_type == "FK") {
            kmg->set_cycle_type("F");
        } else if (cycle_type == "FSK") {
            kmg->set_cycle_type("Fsym");
        }
    }

    /*
    Factor all smoothers once.

    This is especially important for ASW: factorization should occur
    after the hierarchy has been completely constructed, rather than
    once during construction and potentially again before the solve.
    */
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto factor_begin =
        std::chrono::high_resolution_clock::now();

    if (use_kmg) {
        for (auto &grid : kmg->grids) {
            grid.smoother->factor();
        }

        kmg->coarse_solver->factor();
    } else {
        for (auto &grid : mg->grids) {
            grid.smoother->factor();
        }

        mg->coarse_solver->factor();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto factor_end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> factor_time =
        factor_end - factor_begin;

    printf(
        "smoother + coarse factor time %.8e sec\n",
        factor_time.count());

    const auto startup_end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> startup_time =
        startup_end - startup_begin;

    /*
    K-cycle/Krylov path.

    The external rhs and solution vectors are only needed here.
    */
    if (use_kmg) {
        const int fine_size = kmg->grids[0].N;

        DeviceVec<T> rhs(fine_size);
        DeviceVec<T> soln(fine_size);

        /*
        Explicitly initialize the solution because the residual
        calculation and Krylov solve assume a defined initial guess.
        */
        soln.zeroValues();

        kmg->grids[0].d_defect.copyValuesTo(rhs);

        auto *outer_solver =
            static_cast<KrylovSolve *>(kmg->outer_solver);

        const T initial_residual =
            outer_solver->getResidualNorm(rhs, soln);

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto solve_begin =
            std::chrono::high_resolution_clock::now();

        kmg->solve(rhs, soln);

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto solve_end =
            std::chrono::high_resolution_clock::now();

        const T final_residual =
            outer_solver->getResidualNorm(rhs, soln);

        const std::chrono::duration<double> solve_time =
            solve_end - solve_begin;

        const int ndof = kmg->grids[0].N;

        const double total_time =
            startup_time.count() + solve_time.count();

        const double memory_mb =
            kmg->get_memory_usage_mb();

        printf(
            "AOB-wing %s solve, ndof %d: "
            "startup %.4e, solve %.4e, total %.4e sec, "
            "memory %.4e MB\n",
            cycle_type.c_str(),
            ndof,
            startup_time.count(),
            solve_time.count(),
            total_time,
            memory_mb);

        double log_reduction_rate = 0.0;

        if (initial_residual > static_cast<T>(0.0) &&
            final_residual > static_cast<T>(0.0) &&
            solve_time.count() > 0.0) {

            log_reduction_rate =
                (std::log(initial_residual) -
                 std::log(final_residual)) /
                std::log(10.0) /
                solve_time.count();
        }

        printf(
            "\nGMRES-%s on AOB wing, level %d, SR %.4e\n",
            cycle_type.c_str(),
            level,
            SR);

        printf(
            "\tinitial residual %.6e => final residual %.6e\n",
            static_cast<double>(initial_residual),
            static_cast<double>(final_residual));

        printf(
            "\tsolve time %.6e sec, "
            "log10(reduction)/sec %.6e\n",
            solve_time.count(),
            log_reduction_rate);

        int *d_perm = kmg->grids[0].d_perm;

        auto h_soln =
            soln
                .createPermuteVec(6, d_perm)
                .createHostVec();

        printToVTK<Assembler, HostVec<T>>(
            kmg->grids[0].assembler,
            h_soln,
            "out/wing_mg_lin.vtk");
    } else {
        /*
        Standalone V/W/F multigrid path.

        These routines use the solution and defect vectors stored in the
        finest GRID, so do not access any KMG data here.
        */
        const T initial_residual =
            mg->grids[0].getResidNorm();

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto solve_begin =
            std::chrono::high_resolution_clock::now();

        if (cycle_type == "V") {
            mg->vcycle_solve(
                0,
                pre_smooth,
                post_smooth,
                max_cycles,
                print_iterations,
                atol,
                rtol,
                double_smooth,
                print_frequency,
                print_cycle_times);
        } else if (cycle_type == "W") {
            mg->wcycle_solve(
                0,
                pre_smooth,
                post_smooth,
                max_cycles,
                print_iterations,
                atol,
                rtol);
        } else if (cycle_type == "F") {
            mg->fcycle_solve(
                0,
                pre_smooth,
                post_smooth,
                max_cycles,
                print_iterations,
                atol,
                rtol,
                double_smooth,
                print_frequency,
                print_cycle_times);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        const auto solve_end =
            std::chrono::high_resolution_clock::now();

        const T final_residual =
            mg->grids[0].getResidNorm();

        const std::chrono::duration<double> solve_time =
            solve_end - solve_begin;

        const int ndof = mg->grids[0].N;

        const double total_time =
            startup_time.count() + solve_time.count();

        const double memory_mb =
            mg->get_memory_usage_mb();

        printf(
            "AOB-wing %s-cycle solve, ndof %d: "
            "startup %.4e, solve %.4e, total %.4e sec, "
            "memory %.4e MB\n",
            cycle_type.c_str(),
            ndof,
            startup_time.count(),
            solve_time.count(),
            total_time,
            memory_mb);

        double log_reduction_rate = 0.0;

        if (initial_residual > static_cast<T>(0.0) &&
            final_residual > static_cast<T>(0.0) &&
            solve_time.count() > 0.0) {

            log_reduction_rate =
                (std::log(initial_residual) -
                 std::log(final_residual)) /
                std::log(10.0) /
                solve_time.count();
        }

        printf(
            "\n%s-cycle GMG on AOB wing, level %d, SR %.4e\n",
            cycle_type.c_str(),
            level,
            SR);

        printf(
            "\tinitial residual %.6e => final residual %.6e\n",
            static_cast<double>(initial_residual),
            static_cast<double>(final_residual));

        printf(
            "\tsolve time %.6e sec, "
            "log10(reduction)/sec %.6e\n",
            solve_time.count(),
            log_reduction_rate);

        int *d_perm = mg->grids[0].d_perm;

        auto h_soln =
            mg->grids[0]
                .d_soln
                .createPermuteVec(6, d_perm)
                .createHostVec();

        printToVTK<Assembler, HostVec<T>>(
            mg->grids[0].assembler,
            h_soln,
            "out/wing_mg_lin.vtk");
    }

    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

template <typename T, class Assembler>
void solve_direct(
    MPI_Comm &comm,
    int level,
    double SR) {

    using Basis = typename Assembler::Basis;

    using Smoother =
        MulticolorGSSmoother_V1<Assembler>;

    constexpr SCALER scaler = LINE_SEARCH;
    constexpr bool is_bsr = false;

    using Prolongation =
        UnstructuredProlongation<
            Assembler,
            Basis,
            is_bsr>;

    using GRID =
        SingleGrid<
            Assembler,
            Prolongation,
            Smoother,
            scaler>;

    using Preconditioner =
        CusparseMGDirectLU<T, Assembler>;

    using LinearSolver =
        PCGSolver<T, GRID>;

    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto startup_begin =
        std::chrono::high_resolution_clock::now();

    TACSMeshLoader mesh_loader{comm};

    const std::string mesh_filename =
        "../../examples/gmg/3_aob_wing/meshes/"
        "aob_wing_L" +
        std::to_string(level) +
        ".bdf";

    mesh_loader.scanBDFFile(mesh_filename.c_str());

    const double E = 70.0e9;
    const double nu = 0.3;
    const double thickness = 2.0 / SR;

    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

    auto assembler =
        Assembler::createFromBDF(
            mesh_loader,
            Data(E, nu, thickness));

    const int nvars = assembler.get_num_vars();
    const int nnodes = assembler.get_num_nodes();

    HostVec<T> h_loads(nvars);
    T *h_loads_ptr = h_loads.getPtr();

    const T load_magnitude = static_cast<T>(10.0);

    for (int inode = 0; inode < nnodes; ++inode) {
        h_loads_ptr[6 * inode + 2] = load_magnitude;
    }

    auto &bsr_data = assembler.getBsrData();

    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(10.0, false);

    int coarse_rowp[2] = {0, nnodes};
    auto h_color_rowp =
        HostVec<int>(2, coarse_rowp);

    assembler.moveBsrDataToDevice();

    auto loads =
        assembler.createVarsVec(h_loads_ptr);

    assembler.apply_bcs(loads);

    auto kmat =
        createBsrMat<Assembler, VecType<T>>(assembler);

    auto linear_solution =
        assembler.createVarsVec();

    auto variables =
        assembler.createVarsVec();

    assembler.set_variables(variables);

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto assembly_begin =
        std::chrono::high_resolution_clock::now();

    constexpr int elems_per_block = 1;

    assembler
        .template add_jacobian_fast<elems_per_block>(kmat);

    assembler.apply_bcs(kmat);

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto assembly_end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> assembly_time =
        assembly_end - assembly_begin;

    printf(
        "\tassembled K in %.4e sec\n",
        assembly_time.count());

    const T dummy_omega = static_cast<T>(1.0);
    constexpr bool dummy_symmetric = false;
    constexpr int dummy_nsmooth = 1;

    auto smoother = new Smoother(
        cublas_handle,
        cusparse_handle,
        assembler,
        kmat,
        h_color_rowp,
        dummy_omega,
        dummy_symmetric,
        dummy_nsmooth);

    constexpr int ELEM_MAX = 10;

    auto prolongation = new Prolongation(
        cusparse_handle,
        assembler,
        ELEM_MAX);

    auto grid = new GRID(
        assembler,
        prolongation,
        smoother,
        kmat,
        loads,
        cublas_handle,
        cusparse_handle);

    auto preconditioner = new Preconditioner(
        cublas_handle,
        cusparse_handle,
        assembler,
        kmat);

    SolverOptions options;
    options.ncycles = 800;
    options.print_freq = 10;

    auto linear_solver = new LinearSolver(
        cublas_handle,
        cusparse_handle,
        grid,
        preconditioner,
        options);

    linear_solver->set_rel_tol(1.0e-6);
    linear_solver->set_abs_tol(1.0e-20);
    linear_solver->set_print(true);

    const auto startup_end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> startup_time =
        startup_end - startup_begin;

    linear_solution.zeroValues();

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto solve_begin =
        std::chrono::high_resolution_clock::now();

    preconditioner->factor();

    const T initial_residual =
        linear_solver->getResidualNorm(
            grid->d_defect,
            linear_solution);

    const bool failed =
        linear_solver->solve(
            grid->d_defect,
            linear_solution,
            true);

    const T final_residual =
        linear_solver->getResidualNorm(
            grid->d_defect,
            linear_solution);

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto solve_end =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> solve_time =
        solve_end - solve_begin;

    double log_reduction_rate = 0.0;

    if (initial_residual > static_cast<T>(0.0) &&
        final_residual > static_cast<T>(0.0) &&
        solve_time.count() > 0.0) {

        log_reduction_rate =
            (std::log(initial_residual) -
             std::log(final_residual)) /
            std::log(10.0) /
            solve_time.count();
    }

    printf(
        "\nDirectLU-PCG on AOB wing, level %d, SR %.4e\n",
        level,
        SR);

    printf(
        "\tinitial residual %.6e => final residual %.6e\n",
        static_cast<double>(initial_residual),
        static_cast<double>(final_residual));

    printf(
        "\tsolve time %.6e sec, "
        "log10(reduction)/sec %.6e\n",
        solve_time.count(),
        log_reduction_rate);

    const double total_time =
        startup_time.count() + solve_time.count();

    const double memory_mb =
        static_cast<double>(sizeof(T)) *
        static_cast<double>(bsr_data.nnzb) *
        36.0 /
        1024.0 /
        1024.0;

    printf(
        "direct-LU solve, ndof %d: "
        "assembly %.4e, startup %.4e, solve %.4e, "
        "total %.4e sec, memory %.4e MB\n",
        nvars,
        assembly_time.count(),
        startup_time.count(),
        solve_time.count(),
        total_time,
        memory_mb);

    int *d_perm = linear_solver->grid->d_perm;

    auto h_solution =
        linear_solution
            .createPermuteVec(6, d_perm)
            .createHostVec();

    printToVTK<Assembler, HostVec<T>>(
        linear_solver->grid->assembler,
        h_solution,
        "out/wing_direct_lin.vtk");

    if (failed) {
        printf("DirectLU-PCG linear solver failed\n");
    }

    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

template <typename T, class Smoother, class Assembler>
void dispatch_solver(
    const std::string &smoother_type,
    MPI_Comm &comm,
    int level,
    double SR,
    int nsmooth,
    int ninnercyc,
    T omega,
    int chebyshev_order,
    const std::string &cycle_type) {

    if (smoother_type == "direct") {
        solve_direct<T, Assembler>(
            comm,
            level,
            SR);

        return;
    }

    multigrid_solve<T, Smoother, Assembler>(
        comm,
        level,
        smoother_type,
        SR,
        nsmooth,
        ninnercyc,
        omega,
        chebyshev_order,
        cycle_type);
}

int main(int argc, char **argv) {
    /*
    Defaults chosen to match the previously working ASW wing case more
    closely than the newer level-1/SR-50 configuration.
    */
    int level = 3;
    double SR = 300.0;
    double omega = 0.15;

    int chebyshev_order = 8;
    int nsmooth = 4;
    int ninnercyc = 1;

    std::string smoother_type = "asw";
    std::string cycle_type = "VK";

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    bool argument_error = false;

    for (int iarg = 1; iarg < argc; ++iarg) {
        const std::string option =
            lowercase_copy(argv[iarg]);

        auto require_value =
            [&](const char *option_name) -> bool {
                if (iarg + 1 >= argc) {
                    std::cerr
                        << "Missing value for "
                        << option_name
                        << "\n";

                    argument_error = true;
                    return false;
                }

                return true;
            };

        if (option == "--level") {
            if (!require_value("--level")) {
                break;
            }

            level = std::atoi(argv[++iarg]);
        } else if (option == "--sr") {
            if (!require_value("--sr")) {
                break;
            }

            SR = std::atof(argv[++iarg]);
        } else if (option == "--omega") {
            if (!require_value("--omega")) {
                break;
            }

            omega = std::atof(argv[++iarg]);
        } else if (option == "--smoother") {
            if (!require_value("--smoother")) {
                break;
            }

            smoother_type =
                lowercase_copy(argv[++iarg]);
        } else if (option == "--cycle") {
            if (!require_value("--cycle")) {
                break;
            }

            cycle_type =
                uppercase_copy(argv[++iarg]);
        } else if (option == "--nsmooth") {
            if (!require_value("--nsmooth")) {
                break;
            }

            nsmooth = std::atoi(argv[++iarg]);
        } else if (option == "--ninnercyc") {
            if (!require_value("--ninnercyc")) {
                break;
            }

            ninnercyc = std::atoi(argv[++iarg]);
        } else if (option == "--order") {
            if (!require_value("--order")) {
                break;
            }

            chebyshev_order =
                std::atoi(argv[++iarg]);
        } else if (option == "--help" ||
                   option == "-h") {
            print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            std::cerr
                << "Unknown option: "
                << argv[iarg]
                << "\n";

            argument_error = true;
            break;
        }
    }

    if (argument_error) {
        print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (level < 0) {
        std::cerr
            << "--level must be nonnegative\n";

        MPI_Finalize();
        return 1;
    }

    if (SR <= 0.0) {
        std::cerr
            << "--sr must be positive\n";

        MPI_Finalize();
        return 1;
    }

    if (nsmooth <= 0) {
        std::cerr
            << "--nsmooth must be positive\n";

        MPI_Finalize();
        return 1;
    }

    if (ninnercyc <= 0) {
        std::cerr
            << "--ninnercyc must be positive\n";

        MPI_Finalize();
        return 1;
    }

    if (chebyshev_order <= 0) {
        std::cerr
            << "--order must be positive\n";

        MPI_Finalize();
        return 1;
    }

    if (!is_supported_smoother(smoother_type)) {
        std::cerr
            << "Unsupported smoother: "
            << smoother_type
            << "\n";

        print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (smoother_type != "direct" &&
        !is_supported_cycle(cycle_type)) {

        std::cerr
            << "Unsupported cycle: "
            << cycle_type
            << "\n";

        print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    using T = double;

    using Quadrature =
        QuadLinearQuadrature<T>;

    using Director =
        LinearizedRotation<T>;

    constexpr bool has_reference_axis = false;
    constexpr bool is_nonlinear = false;

    using ShellData =
        ShellIsotropicData<
            T,
            has_reference_axis>;

    using Physics =
        IsotropicShell<
            T,
            ShellData,
            is_nonlinear>;

    using Basis =
        LagrangeQuadBasis<
            T,
            Quadrature,
            1>;

    using Assembler =
        MITCShellAssembler<
            T,
            Director,
            Basis,
            Physics,
            VecType,
            BsrMat>;

    printf(
        "AOB-wing MITC4 solve\n"
        "--------------------\n"
        "level       : %d\n"
        "SR          : %.6e\n"
        "smoother    : %s\n"
        "cycle       : %s\n"
        "omega       : %.6e\n"
        "nsmooth     : %d\n"
        "ninnercyc   : %d\n"
        "Cheb order  : %d\n"
        "--------------------\n",
        level,
        SR,
        smoother_type.c_str(),
        cycle_type.c_str(),
        omega,
        nsmooth,
        ninnercyc,
        chebyshev_order);

    if (smoother_type == "asw") {
        using Smoother =
            UnstructuredQuadElementAdditiveSchwarzSmoother<
                T,
                Assembler,
                false>;

        dispatch_solver<T, Smoother, Assembler>(
            smoother_type,
            comm,
            level,
            SR,
            nsmooth,
            ninnercyc,
            omega,
            chebyshev_order,
            cycle_type);
    } else if (
        smoother_type == "chebyshev" ||
        smoother_type == "jacobi") {

        using Smoother =
            ChebyshevPolynomialSmoother<
                Assembler,
                false>;

        dispatch_solver<T, Smoother, Assembler>(
            smoother_type,
            comm,
            level,
            SR,
            nsmooth,
            ninnercyc,
            omega,
            chebyshev_order,
            cycle_type);
    } else if (
        smoother_type == "gsmc" ||
        smoother_type == "direct") {

        using Smoother =
            MulticolorGSSmoother_V1<Assembler>;

        dispatch_solver<T, Smoother, Assembler>(
            smoother_type,
            comm,
            level,
            SR,
            nsmooth,
            ninnercyc,
            omega,
            chebyshev_order,
            cycle_type);
    }

    MPI_Finalize();
    return 0;
}