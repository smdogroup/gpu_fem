#include "_src/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

/* command line args:
    [linear|nonlinear] [--nxe int] 
    * currently nxe must be multiple of 5

    // old arg, // [--iterative]

    examples:
    ./static.out linear --iterative --nxe 10    to run linear, GMRES solve with nxe=10
    ./static.out nonlinear   to run nonlinear
    if --iterative is not used, full_LU direct solve instead
*/

template <typename T>
void vec_scale(int N, T scale, T *myvec) {
    for (int i = 0; i < N; i++) myvec[i] *= scale;
}

void solve_unsteady_linear(int nxe) {
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 350e6;
    // int nxe_per_comp = nxe / 5, nye_per_comp = nye/5; // for now (should have 25 grids)
    int nxe_per_comp = 1, nye_per_comp = 1;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get some static loads
    double Q = 1.0; // load magnitude
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    int ndof = assembler.get_num_vars();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto mass_mat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();

    // assemble mass matrix
    assembler.add_mass_jacobian(res, mass_mat, true);
    assembler.apply_bcs(mass_mat);

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // time settings
    int num_timesteps = 1000;
    double dt = 0.01;

    // compute the forces on the structure
    T *h_forces = new T[ndof * num_timesteps];
    memset(h_forces, 0.0, ndof * num_timesteps * sizeof(T));
    for (int itime = 0; itime < num_timesteps; itime++) {
        // copy from static loads to unsteady loads
        memcpy(&h_forces[itime * ndof], my_loads, ndof * sizeof(T));
        T time = dt * itime;
        T omega = 1.431;
        // T omega = 4.0;
        T scale = 10.0 * std::sin(3.14159 * omega * time);
        // cblas_dscal(ndof, scale, &h_forces[itime * ndof], 1);
        vec_scale(ndof, scale, &h_forces[itime * ndof]);
    }
    auto forces = HostVec<T>(ndof * num_timesteps, h_forces).createDeviceVec();

    // printf("h_forces");
    // printVec<T>(ndof, h_forces);
    // return;

    // create the linear gen alpha integrator
    auto integrator = LGAIntegrator(mass_mat, kmat, forces, ndof, num_timesteps, dt);

    // now solve and write to vtk
    print = true;
    integrator.solve(print);
    int stride = 2;
    integrator.writeToVTK<Assembler>(assembler, "out/plate_dyn", stride);

    integrator.free();
}

void solve_unsteady_nonlinear(int nxe) {
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 350e6;
    // int nxe_per_comp = nxe / 5, nye_per_comp = nye/5; // for now (should have 25 grids)
    int nxe_per_comp = 1, nye_per_comp = 1;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get some static loads
    double Q = 1.0; // load magnitude
    T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    int ndof = assembler.get_num_vars();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto mass_mat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();

    // time settings
    int num_timesteps = 100;
    // double dt = 0.1;
    double dt = 0.01;

    // compute the forces on the structure
    T *h_forces = new T[ndof * num_timesteps];
    memset(h_forces, 0.0, ndof * num_timesteps * sizeof(T));
    for (int itime = 0; itime < num_timesteps; itime++) {
        // copy from static loads to unsteady loads
        memcpy(&h_forces[itime * ndof], my_loads, ndof * sizeof(T));
        T time = dt * itime;
        T omega = 1.431;
        // T omega = 4.0;
        T scale = 10.0 * std::sin(3.14159 * omega * time);
        // cblas_dscal(ndof, scale, &h_forces[itime * ndof], 1);
        vec_scale(ndof, scale, &h_forces[itime * ndof]);
    }
    auto forces = HostVec<T>(ndof * num_timesteps, h_forces).createDeviceVec();

    int print_freq = 1, max_newton_steps = 30;
    T rel_tol = 1e-15, abs_tol = 1e-8;
    bool lin_print = false;

    // create the linear gen alpha integrator
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    auto integrator = NLGAIntegrator<Assembler>(
        solve_func, assembler, mass_mat, kmat, 
        forces, ndof, num_timesteps, dt, print_freq, 
        rel_tol, abs_tol, max_newton_steps, lin_print);

    // now solve and write to vtk
    print = true;
    integrator.solve(print);
    int stride = 2;
    integrator.writeToVTK(assembler, "out/plate_dyn", stride);
    integrator.free();
}

int main(int argc, char **argv) {
    // input ----------
    bool run_linear = false;
    // bool full_LU = true;
    int nxe = 10;  // default value

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "linear") == 0) {
            run_linear = true;
        } else if (strcmp(arg, "nonlinear") == 0) {
            run_linear = false;
        // } else if (strcmp(arg, "--iterative") == 0) {
        //     full_LU = false;
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
        solve_unsteady_linear(nxe);
    } else {
        solve_unsteady_nonlinear(nxe);
    }
    return 0;
};
