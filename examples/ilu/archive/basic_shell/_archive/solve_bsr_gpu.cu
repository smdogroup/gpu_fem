#include "assembler.h"
#include "chrono"
#include "linalg/bsr_mat.h"
#include "linalg/cusparse_solve.h"
#include "linalg/vec.h"
#include "shell/shell.h"

/*
@Author : Sean Engelstad
@Description: this example uses ILU(0) or no fill pattern to solve
the linear elastostatic BVP of a random-generated mesh. The solver is cusparse
which solves from data stored on the GPU (mat and vecs). The ILU(0)
fill pattern can result in high error so next steps are to solve with ILU(k)
higher-order fill patterns.
*/

int main(void) {
  using T = double;

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, DeviceVec, BsrMat>;

  // printf("running!\n");

  // int num_geo_nodes = 10;
  // int num_vars_nodes = 10;
  // int num_elements = 20;

  int num_geo_nodes = 100;
  int num_vars_nodes = num_geo_nodes;
  int num_elements = 1e3;

  // make fake bcs
  HostVec<int> bcs(10);
  bcs.randomize(num_vars_nodes);

  // make fake element connectivity for testing
  int N = Geo::num_nodes * num_elements;
  HostVec<int32_t> geo_conn(N);
  geo_conn.randomize(num_geo_nodes);
  make_unique_conn(num_elements, Geo::num_nodes, num_geo_nodes,
                   geo_conn.getPtr());

  // randomly generate the connectivity for the variables / basis
  int N2 = Basis::num_nodes * num_elements;
  HostVec<int32_t> vars_conn(N2);
  vars_conn.randomize(num_vars_nodes);
  make_unique_conn(num_elements, Basis::num_nodes, num_vars_nodes,
                   vars_conn.getPtr());

  // set the xpts randomly for this example
  int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
  HostVec<T> xpts(num_xpts);
  xpts.randomize();

  // initialize ElemData
  double E = 70e9, nu = 0.3, t = 0.005;  // aluminum plate
  // initialize same data object for each element
  HostVec<Data> physData(num_elements, Data(E, nu, t));

  // make the assembler
  // NOTE : even for GPU case, HostVec inputs to Assembler in order to get BSR
  // structure
  Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn,
                      vars_conn, xpts, bcs, physData);

  // init variables u
  int32_t num_vars = assembler.get_num_vars();
  HostVec<T> h_vars(num_vars);
  h_vars.randomize();
  auto d_vars = h_vars.createDeviceVec();

  // init res, kmat for add_jacobian
  DeviceVec<T> res(num_vars);
  auto kmat = createBsrMat<Assembler, DeviceVec<T>>(assembler);

  // init soln vec
  DeviceVec<T> d_soln(num_vars);

  // time add jacobian method and cusparse solve
  auto start = std::chrono::high_resolution_clock::now();

  // add or assemble the jacobian on the GPU
  assembler.set_variables(d_vars);
  assembler.add_jacobian(res, kmat);  // TODO : need to scale res by -1?

  // this will launch kernels inside the res, mat classes if on device
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // copy soln back to host
  auto kmat_dvals = kmat.getVec();
  auto kmat_hvals = kmat_dvals.createHostVec();

  // print kmat before solve because it changes kmat values in place
  printf("kmat pre-solve: ");
  printVec<double>(24, kmat_hvals.getPtr());
  printf("\n");

  // do the cusparse solve
  cusparse_solve<T>(kmat, res, d_soln);

  // stop timing
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // copy soln back to host
  kmat_dvals = kmat.getVec();
  kmat_hvals = kmat_dvals.createHostVec();
  auto h_res = res.createHostVec();
  auto h_soln = d_soln.createHostVec();

  // print cusparse results
  printf("Cusparse solve kmat * soln = rhs\n");
  printf("--------------------------------\n");

  printf("kmat: ");
  printVec<double>(24, kmat_hvals.getPtr());
  printf("\n");

  printf("soln: ");
  printVec<double>(24, h_soln.getPtr());
  printf("\n");

  printf("rhs: ");
  printVec<double>(24, h_res.getPtr());
  printf("\n");

  printf("took %d microseconds to solve linear system\n",
         (int)duration.count());

  return 0;
};
