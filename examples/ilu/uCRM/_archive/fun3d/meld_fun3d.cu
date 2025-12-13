
#include "coupled/meld.h"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include "_crm_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T>
void verify_conservation(HostVec<T> h_us, HostVec<T> h_ua, HostVec<T> h_fs, HostVec<T> h_fa) {
  // compute total aero forces
  T fA_tot[3], fS_tot[3];
  int na = h_fa.getSize() / 3;
  int ns = h_fs.getSize() / 3;
  memset(fA_tot, 0.0, 3 * sizeof(T));
  memset(fS_tot, 0.0, 3 * sizeof(T));

  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          fA_tot[idim] += h_fa[3 * ia + idim];
      }
  }

  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          fS_tot[idim] += h_fs[3 * is + idim];
      }
  }

  printf("fA_tot:");
  printVec<double>(3, &fA_tot[0]);

  printf("fS_tot:");
  printVec<double>(3, &fS_tot[0]);

  // compute total work done
  T W_A = 0.0, W_S = 0.0;
  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          W_A += h_fa[3 * ia + idim] * h_ua[3 * ia + idim];
      }
  }
  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          W_S += h_fs[3 * is + idim] * h_us[3 * is + idim];
      }
  }

  // compute the relative error
  T W_rel_err = abs(W_A - W_S) / abs(W_S);

  printf("W_A %.6e, W_S %.6e, rel err %.6e\n", W_A, W_S, W_rel_err);
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

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

  double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  auto xs0 = assembler.getXpts();

  // make struct disps
  auto h_us = HostVec<T>(3 * ns);
  double disp_mag = 1.0;
  // double disp_mag = 0.0;
  auto h_xs0 = xs0.createHostVec();
  for (int i = 0; i < ns; i++) {
    // h_us[3 * i + 2] = disp_mag;
    // h_us[3*i+2] = h_xs0[3*i]/50 * disp_mag;
    h_us[3 * i + 2] = disp_mag * (1.0 - std::sin(h_xs0[3*i]/50 * 3.1415 * 1.0));
  }
  auto us = h_us.createDeviceVec();

  // load the coarse mesh for the aero surf
  // --------------------------------------

  double *xyz_forces;
  auto _assembler_aero = makeFun3dAeroSurfMeshFromDat<Assembler>("fun3d_forces.dat", &xyz_forces);
  auto xa0 = _assembler_aero.getXpts();
  int na = _assembler_aero.get_num_nodes();
  auto h_xa0 = xa0.createHostVec();

  // compute aero loads
  auto h_loads = HostVec<T>(6 * na);
  auto h_fa = HostVec<T>(3*na);
  for (int i = 0; i < na; i++) {
    h_loads[6*i] = xyz_forces[3*i];
    h_loads[6*i+1] = xyz_forces[3*i+1];
    h_loads[6*i+2] = xyz_forces[3*i+2];
    for (int idim = 0; idim < 3; idim++) {
      h_fa[3*i+idim] = h_loads[6*i + idim];
    }
  }
  auto d_fa = h_loads.createDeviceVec();

  // make the MELD transfer scheme
  // -----------------------------

  // T beta = 1e-3, Hreg = 1e-8;
  T beta = 0.1, Hreg = 1e-4;
  int sym = -1, nn = 128;
  static constexpr int NN_PER_BLOCK = 32;
  bool meld_print = true;
  using TransferScheme = MELD<T, NN_PER_BLOCK>;
  auto meld = TransferScheme(xs0, xa0, beta, nn, sym, Hreg, meld_print);
  meld.initialize();

  // disp transfer
  // -------------

  auto ua = meld.transferDisps(us);

  // visualization
  auto h_us_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_us);
  printToVTK<Assembler, HostVec<T>>(assembler, h_us_ext, "uCRM_us.vtk");

  // now extend to full 6-length size (MELD only handles length 3)
  auto h_ua = ua.createHostVec();
  auto h_ua_ext = MELD<T>::expandVarsVec<3, 6>(na, h_ua);

  printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_ua_ext, "uCRM_ua.vtk");

  // load transfer
  // -------------

  auto fs = meld.transferLoads(d_fa);
  // DeviceVec<T> fs = DeviceVec<T>(3 * na);
  // for (int i = 0; i < 10; i++) {
  //     fs = meld.transferLoads(fa);
  // }

  // extend both fs, fa to length 6 vars_per_node not 3
  auto h_fs = fs.createHostVec();
  auto h_fa_ext = MELD<T>::expandVarsVec<3, 6>(na, h_fa);
  auto h_fs_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_fs);

  // visualization of fa and fs
  printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_fa_ext, "uCRM_fa.vtk");
  printToVTK<Assembler, HostVec<T>>(assembler, h_fs_ext, "uCRM_fs.vtk");

  // verify conservation of force and work
  verify_conservation<T>(h_us, h_ua, h_fs, h_fa);

  // free data
  assembler.free();
  meld.free();
  h_xs0.free();
  h_us.free();
  h_xa0.free();
  h_fa.free();
  h_us_ext.free();
  h_fa_ext.free();
  h_fs_ext.free();
};
