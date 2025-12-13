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

void time_linear_static(int nxe, bool print = true, bool write_vtk = false, bool debug = false) {
    // run the plate problem to time the linear static
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
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0

    if (debug) printf("before reordering\n");
    auto start0 = std::chrono::high_resolution_clock::now();
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin);
    assembler.moveBsrDataToDevice();
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nz_time = end0 - start0;

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
    auto start1 = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end1 - start1;

    // solve the linear system
    if (debug) printf("before solve\n");
    auto start2 = std::chrono::high_resolution_clock::now();
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end2 - start2;
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
    int nnodes = (nxe+1) * (nxe+1);
    int dof = nnodes * 6;
    double tot_Time = nz_time.count() + assembly_time.count() + solve_time.count();
    printf("%d, %d, %.4e, %.4e, %.4e, %.4e\n", nxe, dof, nz_time.count(), assembly_time.count(), solve_time.count(), tot_Time);
}

int main() {
    
    printf("nxe, ndof, nz_time(s), assembly_time(s), solve_time(s), tot_time(s)\n");
    for (int i = 0, nxe = 10; i < 20 && nxe < 500; i++, nxe *= 2) {
        time_linear_static(nxe);
    }
};