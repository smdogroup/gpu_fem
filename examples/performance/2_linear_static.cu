#include "../plate/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include <chrono>


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

void time_linear_static(int nxe, std::string ordering, std::string fill_type, bool LU_solve = true, int ILU_k = 5,
    double p_factor = 1.0, bool print = true, bool write_vtk = false, bool debug = false) {
    // run the plate problem to time the linear static
    using T = double;   
    int rcm_iters = 5;

    auto start = std::chrono::high_resolution_clock::now();

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
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0

    if (debug) printf("before reordering\n");

    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, debug);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (debug) printf("before fillin\n");
    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(ILU_k, fillin, debug);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    if (debug) printf("before assembly\n");
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    if (debug) printf("before solve\n");
    if (LU_solve) {
        CUSPARSE::direct_LU_solve(kmat, loads, soln);
    } else {
        int n_iter = 200, max_iter = 400;
        T abs_tol = 1e-11, rel_tol = 1e-14;
        bool print = debug;
        CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
    }
    if (debug) printf("done with solve\n");

    // print some of the data of host residual
    if (write_vtk) {
        auto h_soln = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");
    }

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    if (debug) printf("resid_norm = %.4e\n", resid_norm);

    // report runtime to csv printout
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tot_time = end - start;
    std::string fillin_str = "";
    if (fill_type == "ILUk") {
        fillin_str += " ILU(" + std::to_string(ILU_k) + ")";
    } else if (fill_type == "nofill") {
        fillin_str += " nofill";
    } else if (fill_type == "LU") {
        fillin_str += " LU";
    }
    int nnodes = (nxe+1) * (nxe+1);
    std::string solve_str = LU_solve ? "LU" : "GMRES";

    printf("%d, %d, %s, %s, %s, %.4e\n", nxe, nnodes, ordering.c_str(), fillin_str.c_str(), solve_str.c_str(), tot_time.count());
}

int main() {
    printf("nxe, nnodes, ordering, fillin, solve, tot_time (s)\n");

    // LU solves
    bool run_lu = false;
    if (run_lu) {
        // time_linear_static(160, "RCM", "LU", true);
        // time_linear_static(160, "qorder", "LU", true);
        // time_linear_static(300, "AMD", "LU", true);
        for (int i = 0, nxe = 10; i < 20 && nxe < 500; i++, nxe *= 2) {
            // true means using LU solve not GMRES
            if (nxe < 300) time_linear_static(nxe, "none", "LU", true);
            time_linear_static(nxe, "AMD", "LU", true);
            if (nxe < 150) {
                // printf("here\n");
                time_linear_static(nxe, "RCM", "LU", true);
                time_linear_static(nxe, "qorder", "LU", true);
            }
        }   
    }

    // GMRES solves
    bool run_gmres = true;
    if (run_gmres) {
        int ILU_k = 3; // 5
        // time_linear_static(300, "AMD", "ILUk", false);
        for (int i = 0, nxe = 10; i < 20 && nxe < 300; i++, nxe *= 2) {
            // false means using GMRES not LU solve
            time_linear_static(nxe, "none", "ILUk", false, ILU_k);
            time_linear_static(nxe, "AMD", "ILUk", false, ILU_k);
            time_linear_static(nxe, "RCM", "ILUk", false, ILU_k);
            time_linear_static(nxe, "qorder", "ILUk", false, ILU_k);
        }
    }

    // try to run some large cases with GMRES
    bool run_large_gmres = false;
    if (run_large_gmres) {
        int ILU_k = 3;
        double p_order = 1.0;
        bool print = true, write_vtk = false, debug = true;
        time_linear_static(640, "AMD", "ILUk", false, ILU_k, p_order, print, write_vtk, debug);
        // time_linear_static(1280, "AMD", "ILUk", false, ILU_k);
    }
};