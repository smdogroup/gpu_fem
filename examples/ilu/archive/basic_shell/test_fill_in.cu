#include "_local_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

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
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  // make a specific mesh where I know the correct fill-in
  /*
  
    2 elements, 6 nodes, 36 dof
    28*36 = 1008 nnz before fill-in
    4*36  =  144 nnz added during fill-in
    32*36 = 1152 nnz after fill-in

    fill-in applied before cusparse ILU solve
    so applied to both lower and upper parts of matrix

    -> x1
    (up is x2, off page is x3)

    0  -  1  -  4
    |  e1 |  e2 |
    2  -  3  -  5

    panel is 2 x 1 (m) with 1e-3 m thick

    BCs : 11 total (simply supported and some axial restraints on half edges)
    node 0 : dof 13       [0,2]
    node 2 : dof 123456   [12-17]
    node 4 : dof 3        [26]
    node 5 : dof 23       [31,32]

  */
  
  int mybcs[] = {0, 2, 12, 13, 14, 15, 16, 17, 26, 31, 32};
  HostVec<int> bcs(11, mybcs);

  int num_elements = 2;
  int N = Geo::num_nodes * num_elements;
  int32_t my_geo_conn[] = {0, 1, 2, 3, 1, 3, 4, 5};
  HostVec<int32_t> geo_conn(N, my_geo_conn);

  int N2 = Basis::num_nodes * num_elements;
  int32_t my_vars_conn[] = {0, 1, 2, 3, 1, 3, 4, 5};
  HostVec<int32_t> vars_conn(N2, my_vars_conn);

  int num_geo_nodes = 6;
  int num_vars_nodes = 6;
  int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
  T my_xpts[] = {
    0, 1, 0, // node 0
    1, 1, 0, // node 1
    0, 0, 0, // node 2
    1, 0, 0, // node 3
    2, 1, 0, // node 4
    2, 0, 0 // node 5
  };
  HostVec<T> xpts(num_xpts, my_xpts);

  double E = 70e9, nu = 0.3, t = 0.001; // aluminum plate
    HostVec<Data> physData(num_elements, Data(E, nu, t));

  Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn,
                        vars_conn, xpts, bcs, physData);

  // init variables u
  bool randomize = false; // just zero it for now (linear static analysis)
  auto vars = assembler.createVarsVec(randomize);
  assembler.set_variables(vars);

  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto bsrData = kmat.getBsrData();

  // original fillin
  printf("rowPtr:\n");
  printVec<int32_t>(num_vars_nodes, bsrData.rowPtr);
  printf("colPtr:\n");
  printVec<int32_t>(bsrData.nnzb, bsrData.colPtr);

  // now do cusparse solve with this

  return 0;
};