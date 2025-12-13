#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "../../../tests/test_commons.h"
#include "_plate_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class StructSolver, class Function>
void function_FD_test(StructSolver &struct_solver, DeviceVec<T> &d_loads, Function &func, T &h = 1e-6) {
    /* full struct solver path FD test (testing total derivs) */

    // design vars pert
    int ndvs = struct_solver.get_num_dvs();
    auto h_pert_dvs = HostVec<T>(ndvs);
    for (int i = 0; i < ndvs; i++) {
      h_pert_dvs[i] = ((T)rand()) / RAND_MAX;
    }
    auto d_pert = h_pert_dvs.createDeviceVec();
    HostVec<T> h_dvs(ndvs, 1e-2);
    auto dvs = h_dvs.createDeviceVec();
    struct_solver.set_design_variables(dvs);

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    
    // gradient at initial design and loads
    struct_solver.solve(d_loads);
    T f0 = struct_solver.evalFunction(func);
    // struct_solver.set_design_variables(dvs); // debug (reset here bc directLU?)
    struct_solver.solve_adjoint(func);
    T HC_deriv;
    cublasDdot(cublasHandle, ndvs, func.dv_sens.getPtr(), 1, d_pert.getPtr(), 1, &HC_deriv);

    // FD gradient
    T a = 1.0 * h;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, dvs.getPtr(), 1));
    struct_solver.set_design_variables(dvs);
    if (func.has_adjoint) struct_solver.solve(d_loads);
    T f1 = struct_solver.evalFunction(func);

    // backwards pert
    a = -2.0 * h;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, dvs.getPtr(), 1));
    struct_solver.set_design_variables(dvs);
    if (func.has_adjoint) struct_solver.solve(d_loads);
    T fn1 = struct_solver.evalFunction(func);

    T FD_deriv = (f1 - fn1) * 0.5 / h;

    T rel_err = abs((HC_deriv - FD_deriv) / FD_deriv);
    bool passed = rel_err < 1e-4;
    std::string testName = "total deriv " + func.name + " FD test";
    printTestReport(testName, passed, rel_err);
    printf("\ttotal deriv test: FD %.8e, HC %.8e\n", FD_deriv, HC_deriv);
}

template <typename T, class StructSolver, class Assembler>
void test_run(StructSolver &struct_solver, Assembler &assembler, DeviceVec<T> &d_loads) {
  // init design variables
  int ndvs = assembler.get_num_dvs();
  HostVec<T> h_dvs(ndvs, 1e-2);
  auto dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(dvs);
  auto dfdu = assembler.createVarsVec();

  // functions
  T rhoKS = 100.0;
  auto mass = Mass<T, DeviceVec>();
  auto ksfail = KSFailure<T, DeviceVec>(rhoKS);

  bool mass_has_adjoint = mass.has_adjoint;
  printf("mass has adjoint %d\n", (int)mass_has_adjoint);

  // compute gradient
  struct_solver.solve(d_loads);
  T mass0 = struct_solver.evalFunction(mass);
  T ksfail0 = struct_solver.evalFunction(ksfail);
  printf("init mass = %.4e, ksfail %.4e\n", mass0, ksfail0);
  struct_solver.solve_adjoint(mass);
  struct_solver.solve_adjoint(ksfail);

  // print dfdx
  auto h_dmdx = mass.dv_sens.createHostVec();
  auto h_dkdx = ksfail.dv_sens.createHostVec();
  printf("dm/dx:");
  printVec<T>(10, h_dmdx.getPtr());
  printf("dk/dx:");
  printVec<T>(10, h_dkdx.getPtr());

  struct_solver.writeSoln("out/uCRM_coupled_us.vtk");
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(int argc, char **argv) {
  using T = double;

  // important user settings
  // -----------------------
  constexpr bool nonlinear_strain = false; // true

  // type definitions
  // ----------------

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, nonlinear_strain>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;
  using StructSolver = TacsLinearStatic<T, Assembler>;

  // build the Tacs prelim objects
  // -----------------------------

  // int nxe = 100, nye = 100;
  // int nx_comp = 5, ny_comp = 5;

  int nxe = 2, nye = 2;
  int nx_comp = 2, ny_comp = 2;

    double load_mag = 30.0;
    assert(nxe % nx_comp == 0); // evenly divisible by number of elems_per_comp
    assert(nye % ny_comp == 0);
    int nxe_per_comp = nxe / nx_comp, nye_per_comp = nye / ny_comp;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 250e6;
    
    Assembler assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    // factor & move to GPU
    {
        auto &bsr = assembler.getBsrData();
        bsr.AMD_reordering();
        bsr.compute_full_LU_pattern(10.0, false);
    }
    assembler.moveBsrDataToDevice();

    // 2) Build loads
    int nvars = assembler.get_num_vars();
    int nn = assembler.get_num_nodes();
    using Phys = typename Assembler::Phys;
    T *my_loads = getPlateLoads<T, Phys>(nxe, nye, Lx, Ly, load_mag);
    auto d_loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(d_loads);

    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto linear_solve = CUSPARSE::direct_LU_solve<T>;

  // optimization
  // ------------

  bool struct_print = false;
  auto struct_solver = StructSolver(
        assembler, kmat, linear_solve, struct_print);

  // prelim debugging
  // test_run<T, StructSolver, Assembler>(struct_solver, assembler, d_loads);

  using MyFunction1 = Mass<T, DeviceVec>;
  auto mass = MyFunction1();
  T h = 1e-6;
  function_FD_test<T, StructSolver, MyFunction1>(struct_solver, d_loads, mass, h);

  // ksfail deriv test
  T rhoKS = 100.0;
  using MyFunction2 = KSFailure<T, DeviceVec>;
  auto ksfail = KSFailure<T, DeviceVec>(rhoKS);
  h = 1e-6;
  function_FD_test<T, StructSolver, MyFunction2>(struct_solver, d_loads, ksfail, h);

  // free
  struct_solver.free();
  assembler.free();
  d_loads.free();

  return 0;
};