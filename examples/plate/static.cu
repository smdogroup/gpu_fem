#include "_src/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

/* command line args:
    [linear|nonlinear] [--iterative] [--nxe int]
    * currently nxe must be multiple of 5

    examples:
    ./static.out linear --iterative --nxe 10    to run linear, GMRES solve with nxe=10
    ./static.out nonlinear   to run nonlinear
    if --iterative is not used, full_LU direct solve instead
*/

void solve_linear(bool full_LU, int nxe) {
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 5, nye_per_comp = nye/5; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    if (full_LU) {
        bsr_data.AMD_reordering();
        bsr_data.compute_full_LU_pattern(fillin, print);
    } else {
        /*
        RCM and reorderings actually hurt GMRES performance on the plate case
        because the matrix already has a nice banded structure => RCM increases bandwidth (which means it just doesn't work well for this problem
        as it's whole point is to decrease matrix bandwidth)
        */

        // bsr_data.AMD_reordering();
        // bsr_data.RCM_reordering();
        // bsr_data.qorder_reordering(1.0);
        bsr_data.qorder_reordering(0.25);
        
        // bsr_data.compute_ILUk_pattern(5, fillin);
        bsr_data.compute_nofill_pattern();
        // bsr_data.compute_ILUk_pattern(0, fillin);
        // bsr_data.compute_full_LU_pattern(fillin, print); // reordered full LU here for debug
    }
    // printf("perm:");
    // printVec<int>(bsr_data.nnodes, bsr_data.perm);
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    if (full_LU) {
        CUSPARSE::direct_LU_solve(kmat, loads, soln);
    } else {
        int n_iter = 200, max_iter = 400;
        T abs_tol = 1e-11, rel_tol = 1e-14;
        bool print = true;
        CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);

        // CUSPARSE::GMRES_DR_solve<T, false>(kmat, loads, soln, 50, 10, 65, abs_tol, rel_tol, true, false, 5);
    }

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    printf("resid_norm = %.4e\n", resid_norm);
}

void solve_nonlinear(int nxe) {
    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR factorization
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get the loads
    // double Q = 1e6; // load magnitude
    double Q = 1.0;
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // newton solve
    int num_load_factors = 100, num_newton = 30;
    T min_load_factor = 0.05, max_load_factor = 1.0, abs_tol = 1e-6,
        rel_tol = 1e-4;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/uCRM_";
    constexpr bool fast_assembly = true;
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
        solve_func, kmat, loads, soln, assembler, res, rhs, vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, print);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate_nl.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    bool run_linear = false;
    bool full_LU = true;
    int nxe = 10;  // default value

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "linear") == 0) {
            run_linear = true;
        } else if (strcmp(arg, "nonlinear") == 0) {
            run_linear = false;
        } else if (strcmp(arg, "--iterative") == 0) {
            full_LU = false;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [linear|nonlinear] [--iterative] [--nxe value]" << std::endl;
            return 1;
        }
    }

    if (run_linear) {
        solve_linear(full_LU, nxe);
    } else {
        solve_nonlinear(nxe);
    }
    return 0;
};