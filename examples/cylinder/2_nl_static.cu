#include "_cylinder_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"


// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

int main() {
    using T = double;   

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
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

    // imperfection
    bool imperfection = true;

    // mesh settings
    int nxe_0 = 10, nhe_0 = 10;
    int refinement = 10;
    // int refinement = 30;
    int nxe = nxe_0 * refinement, nhe = nhe_0 * refinement;

    // material properties (aluminum)
    double E = 70e9, nu = 0.3;

    // make the cylinder
    auto assembler = createCylinderAssembler<Assembler>(nxe, nhe, L, R, E, nu, thick, imperfection);

    // BSR factorization
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = imperfection ? 3e4 : 5e4; // load magnitude
    constexpr int compressive = true; // compressive or transverse style loads
    T *my_loads = getCylinderLoads<T, Physics, compressive>(nxe, nhe, L, R, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // newton solve
    int num_load_factors = 10, num_newton = 30;
    T min_load_factor = 0.05, max_load_factor = 1.0, abs_tol = 1e-6,
        rel_tol = 1e-4;
    bool write_vtk = true;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/cylinder_";
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler>(
        solve_func, kmat, loads, soln, assembler, res, rhs, vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, print, write_vtk);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "cylinder_nl.vtk");
};