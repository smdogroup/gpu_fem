#include "_cylinder_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"
#include <chrono>

void run_linear_static(std::string ordering, std::string fill_type, bool LU_solve,
    bool print = false, int ILUk = 3, double p_factor = 1.0, bool qprint = false) {

    // input ----------
    
    int rcm_iters = 10;
    double fillin = 10.0;

    // ----------------

    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // geometry of cylinder
    double L = 1.0;
    double Lr = 3.0;
    double rt = 100.0; // r/t the cylinder slenderness
    double R = L / Lr;
    double thick = R / rt;

    // mesh settings
    int nxe_0 = 10, nhe_0 = 10;
    int refinement = 10;
    // int refinement = 30;
    int nxe = nxe_0 * refinement, nhe = nhe_0 * refinement;

    // material properties (aluminum)
    double E = 70e9, nu = 0.3;

    // make the cylinder
    auto assembler = createCylinderAssembler<Assembler>(nxe, nhe, L, R, E, nu, thick);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();

    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, print);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(ILUk, fillin, print);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 100.0; // load magnitude
    constexpr int compressive = false; // compressive or transverse style loads
    T *my_loads = getCylinderLoads<T, Physics, compressive>(nxe, nhe, L, R, Q);
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
    auto start = std::chrono::high_resolution_clock::now();
    if (LU_solve) {
        CUSPARSE::direct_LU_solve(kmat, loads, soln, print);
    } else {
        int n_iter = 200, max_iter = 200;
        T abs_tol = 1e-11, rel_tol = 1e-14;
        bool debug = false;
        int print_freq = 20;
        CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, 
            print, debug, print_freq);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tot_time = end - start;

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "cylinder.vtk");

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);

    // print out resid norm
    std::string testName = "Cylinder linear static, ";

    if (LU_solve) {
        testName += "LU-solve";
    } else {
        testName += "GMRES-solve";
    }

    testName += ", " + ordering;  // always include the ordering name

    if (fill_type == "ILUk") {
        testName += " ILU(" + std::to_string(ILUk) + ")";
    } else if (fill_type == "nofill") {
        testName += " nofill";
    } else if (fill_type == "LU") {
        testName += " LU";
    }

    if (print) printf("%s, resid_norm = %.4e\n", testName.c_str(), resid_norm);

    // write to csv here
    std::string fillin_str = "";
    if (fill_type == "ILUk") {
        fillin_str += " ILU(" + std::to_string(ILUk) + ")";
    } else if (fill_type == "nofill") {
        fillin_str += " nofill";
    } else if (fill_type == "LU") {
        fillin_str += " LU";
    }
    std::string solve_str = LU_solve ? "LU" : "GMRES";

    if (qprint) {
        // custom printout for qordering study
        printf("%s, %s,%s,  %.4f, %.4e, %.4e\n", ordering.c_str(), fillin_str.c_str(), solve_str.c_str(), p_factor, tot_time.count(), resid_norm);
    } else {
        // regular printout for not the qordererd study
        printf("%s, %s, %s, %.4e, %.4e\n", ordering.c_str(), fillin_str.c_str(), solve_str.c_str(), tot_time.count(), resid_norm);
    }
}

void run_for_csv() {
    // LU solves
    bool print = false;
    bool run_gmres = false;
    bool run_qorder = true;
    bool run_lu = false;
    
    if (run_qorder && !(run_gmres || run_lu)) {
        printf("ordering,fill_type,solver,p_factor,runtime(s),resid_norm\n");
    } else {
        printf("ordering,fill_type,solver,runtime(s),resid_norm\n");
    }

    // GMRES with different ILUk
    
    bool LU_solve = false; // GMRES solve
    double p_factor = 0.5;

    if (run_gmres) {
        int klist[5] = {2, 3, 4, 5, 10};
        for (int k = 0; k < 5; k++) {
            int ILUk = klist[k];
            run_linear_static("none", "ILUk", LU_solve, print, ILUk);
            run_linear_static("AMD", "ILUk", LU_solve, print, ILUk);
            run_linear_static("RCM", "ILUk", LU_solve, print, ILUk);
            run_linear_static("qorder", "ILUk", LU_solve, print, ILUk, p_factor);
        }
    }

    // eval qordering ILU(k) and p_factor optimal cases
    if (run_qorder) {
        int klist[5] = {2, 3, 4, 5, 10};
        double plist[6] = {0.1, 0.25, 0.5, 1.0, 2.0, 4.0};
        for (int k = 0; k < 5; k++) {
            int ILUk = klist[k];
            for (int ip = 0; ip < 6; ip++) {
                p_factor = plist[ip];
                run_linear_static("qorder", "ILUk", LU_solve, print, ILUk, p_factor, true); // true for custom qprint
            }
        }
    }

    // now LU solves
    if (run_lu) {
        LU_solve = true;
        // run_linear_static("none", "LU", LU_solve, print);
        run_linear_static("AMD", "LU", LU_solve, print);
        run_linear_static("RCM", "LU", LU_solve, print);
        run_linear_static("qorder", "LU", LU_solve, print, 1, p_factor);
    }
}

void test_individual_runs() {
    bool print = true;
    bool LU_solve = true;
    run_linear_static("AMD", "LU", LU_solve, print); 

    LU_solve = false; // GMRES solve
    int ILUk = 10;
    double p_factor = 0.5;
    run_linear_static("none", "ILUk", LU_solve, print, ILUk);
    run_linear_static("AMD", "ILUk", LU_solve, print, ILUk);
    run_linear_static("RCM", "ILUk", LU_solve, print, ILUk);
    run_linear_static("qorder", "ILUk", LU_solve, print, ILUk, p_factor);
}

int main() {
    
    // test_individual_runs();
    // run_for_csv();

    // run individual run
    bool print = true;
    run_linear_static("AMD", "LU", true, print); 
    
};