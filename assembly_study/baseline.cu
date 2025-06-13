#include "../examples/plate/_plate_utils.h"
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
    // using T = double;  
    using T = float; 
    int rcm_iters = 5;

    auto start = std::chrono::high_resolution_clock::now();

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

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    if (debug) printf("before assembly\n");
    assembler.apply_bcs(res); // warmup call
    assembler.add_residual(res, print);
    assembler.add_jacobian(res, kmat, print);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);
}

int main() {
    printf("nxe, nnodes, ordering, fillin, solve, tot_time (s)\n");

    time_linear_static(100, "AMD", "ILUk", true, 0);
};