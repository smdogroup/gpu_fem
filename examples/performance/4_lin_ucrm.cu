
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
<<<<<<< HEAD
#include "element/shell/shell_elem_group_v2.h" // new one for unittesting

int main() {
=======

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
<<<<<<< HEAD
  bool mesh_print = false;
  TACSMeshLoader<T> mesh_loader{mesh_print};
=======
  // bool mesh_print = false;
  TACSMeshLoader mesh_loader{comm};
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  // mesh_loader.scanBDFFile("../uCRM/CRM_box_2nd.bdf");
  mesh_loader.scanBDFFile("uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
<<<<<<< HEAD
  constexpr bool is_nonlinear = true;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  // try using new ElemGroup to speedup assembly
  // using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using ElemGroup = ShellElementGroupV2<T, Director, Basis, Physics>;

=======
  constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto start1 = std::chrono::high_resolution_clock::now();
  auto& bsr_data = assembler.getBsrData();
<<<<<<< HEAD
  // double fillin = 10.0;  // 10.0
  // bsr_data.AMD_reordering();
  // bsr_data.compute_full_LU_pattern(fillin, print);

  // bsr_data.AMD_reordering();
  // bsr_data.qorder_reordering(0.2);
  // bsr_data.compute_ILUk_pattern(5, fillin, print);
=======
  double fillin = 10.0;  // 10.0
  // bsr_data.AMD_reordering();
  // bsr_data.compute_full_LU_pattern(fillin, print);

  bsr_data.AMD_reordering();
  // bsr_data.qorder_reordering(0.2);
  bsr_data.compute_ILUk_pattern(10, fillin, print);
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  // bsr_data.compute_full_LU_pattern(fillin, print);

  assembler.moveBsrDataToDevice();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_nz_time = end1 - start1;

<<<<<<< HEAD
  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();

  // debug
  int nvars = assembler.get_num_vars();
  auto h_soln1 = soln.createHostVec();
  for (int i = 0; i < nvars; i++) {
      h_soln1[i] = 1.1343 + 2.3142 * i + 4.132 * i * i;
      h_soln1[i] *= 1e-6;
  }
  auto soln2 = h_soln1.createDeviceVec();
  assembler.set_variables(soln2);

  assembler.add_residual(res, print); // warmup call
  assembler.add_residual(res, print);

  // // check residual not zero
  // printf("check resid\n");
  // auto h_res = res.createHostVec();
  // printVec<T>(100, h_res.getPtr());

  // return 0;

  assembler.add_jacobian(res, kmat, print);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // get the loads
  //int nvars = assembler.get_num_vars();
=======
  // get the loads
  int nvars = assembler.get_num_vars();
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 3.0 * 23.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

<<<<<<< HEAD
=======
  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();
  // assembler.add_residual(res, print); // warmup call
  assembler.add_residual(res, print);
  assembler.add_jacobian(res, kmat, print);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // return 0;

>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
  auto start2 = std::chrono::high_resolution_clock::now();

  // newton solve => go to 10x the 1m up disp from initial loads
  bool LU_solve = false;
  if (LU_solve) {
    CUSPARSE::direct_LU_solve(kmat, loads, soln, print);
  } else {
    int n_iter = 300, max_iter = 300;
    constexpr bool right = false;
    T abs_tol = 1e-6, rel_tol = 1e-6; // for left preconditioning
    CUSPARSE::GMRES_solve<T, true, right>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
  }

  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> lin_solve_time = end2 - start2;

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/uCRM_lin.vtk");

  printf("uCRM LIN case on GPU\n");
  printf("\tcompute nz time = %.4f\n", compute_nz_time.count());
  printf("\tlinear solve time = %.4f\n", lin_solve_time.count());
  

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  h_soln.free();
<<<<<<< HEAD
};
=======
};
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
