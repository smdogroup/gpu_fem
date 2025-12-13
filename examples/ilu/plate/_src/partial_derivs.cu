
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include "_plate_utils.h"
#include "../../../tests/test_commons.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

template <typename T, class Assembler, class Data>
Assembler makeAssembler(MPI_Comm &comm, bool solve = true) {
  // 1) Build mesh & assembler
    int nxe = 100, nye = 100;
    int nx_comp = 5, ny_comp = 5;

    // int nxe = 2, nye = 2;
    // int nx_comp = 2, ny_comp = 2;

    // int nxe = 1, nye = 1;
    // int nx_comp = 1, ny_comp = 1;

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
    auto vars = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();

    // assemble the kmat
    bool print = false;
    assembler.set_variables(vars);
    assembler.add_jacobian(res, kmat, print);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // solve the linear system
    CUSPARSE::direct_LU_solve<T>(kmat, d_loads, soln, false);

    // set variables in for later ksfailure evaluations
    assembler.set_variables(soln);

    return assembler;
}

template <typename T, class Assembler>
void mass_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 2e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // call mass on init design
  T mass = assembler._compute_mass();
  // should match 1.8826e4
  // printf("mass init %.8e\n", mass);

  // mass gradient
  DeviceVec<T> dmdx(ndvs);
  assembler._compute_mass_DVsens(dmdx);
  auto h_dmdx = dmdx.createHostVec();

  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dmass_HC = 0.0;
  cublasDdot(cublasHandle, ndvs, dmdx.getPtr(), 1, d_pert.getPtr(), 1, &dmass_HC);

  /* now FD test */
  T massn1, mass1;

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  mass1 = assembler._compute_mass();

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  massn1 = assembler._compute_mass();

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));

  // report unittest result
  T dmass_FD = (mass1 - massn1) / 2 / h;
  T rel_err = abs((dmass_FD - dmass_HC) / dmass_FD);
  bool pass = rel_err < 1e-4;
  printTestReport("assembler mass FD test", pass, rel_err);
  printf("\tdmass_FD %.8e dmass_HC %.8e\n", dmass_FD, dmass_HC);
}

template <typename T, class Assembler>
void ksfailure_DV_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 2e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // call mass on init design
  T rhoKS = 100.0;
  // T SF = 1.5;
  T SF = 1.0;
  // T rhoKS = 100000.0;
  T ksfail0 = assembler._compute_ks_failure(rhoKS, SF);
  // should match 1.1460e0
  // printf("ksfailure init %.8e\n", ksfail0);

  // mass gradient
  DeviceVec<T> dksdx(ndvs);
  assembler._compute_ks_failure_DVsens(rhoKS, SF, dksdx);
  auto h_dksdx = dksdx.createHostVec();
  // printf("h_dkxdx:");
  // printVec<T>(ndvs, h_dksdx.getPtr());

  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dks_HC = 0.0;
  cublasDdot(cublasHandle, ndvs, dksdx.getPtr(), 1, d_pert.getPtr(), 1, &dks_HC);

  /* now FD test */
  T ksn1, ks1;

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  ks1 = assembler._compute_ks_failure(rhoKS, SF);

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  ksn1 = assembler._compute_ks_failure(rhoKS, SF);

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));

  // report unittest result
  T dks_FD = (ks1 - ksn1) / 2 / h;
  T rel_err = abs((dks_FD - dks_HC) / dks_FD);
  bool passed = rel_err < 1e-4;
  printTestReport("assembler ks DV (designvar) FD test", passed, rel_err);
  // printf("ksfail DV sens\n");
  printf("\tdks_FD %.8e dks_HC %.8e\n", dks_FD, dks_HC);
}

template <typename T, class Data>
void ksfail_data_strain_FD_test(T &h = 1e-6) {
  T rhoKS = 100.0;
  T E[9], pE[9];
  for (int i = 0; i < 9; i++) {
    E[i] = ((double) rand()) / RAND_MAX;
    pE[i] = ((double) rand()) / RAND_MAX;
  }
  double SF = 1.5;
  double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  Data physData0(E_, nu, thick, rho, ys);
  T fail_index = physData0.evalFailure(rhoKS, SF, E);
  // printf("fail_index %.8e\n", fail_index);

  // FD vs HC strain sens
  T dE[9];
  memset(dE, 0.0, 9 * sizeof(T));
  physData0.evalFailureStrainSens(1.0, rhoKS, SF, E, dE);
  T dks_HC = A2D::VecDotCore<T,9>(dE, pE);

  // FD 
  T newE[9];
  memset(newE, 0.0, 9 * sizeof(T));
  A2D::VecAddCore<T,9>(1.0, E, newE);
  A2D::VecAddCore<T,9>(h, pE, newE);
  T fail1 = physData0.evalFailure(rhoKS, SF, newE);
  A2D::VecAddCore<T,9>(-2.0 * h, pE, newE);
  T failn1 = physData0.evalFailure(rhoKS, SF, newE);
  T dks_FD = (fail1 - failn1) / 2.0 / h;

  T rel_err = abs((dks_FD - dks_HC) / dks_FD);
  bool passed = rel_err < 1e-4;
  printTestReport("IsotropicData strainSens FD test", passed, rel_err);
  printf("\tdks_strain FD %.8e HC %.8e\n", dks_FD, dks_HC);
}


template <typename T, class ElemGroup, class Data>
void ksfail_elem_SV_FD_test(T &h = 1e-6) {
  /* check shell elemgroup state var sens */
  T rhoKS = 100.0;
  double SF = 1.5;

  double E_ = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 250e6;

  // double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  Data data(E_, nu, thick, rho, ys);
  // T xpts[12], vars[24];
  T p_vars[24], new_vars[24];
  // for (int i = 0; i < 12; i++) {
  //   xpts[i] = ((double)rand()) / RAND_MAX;
  // }
  for (int i = 0; i < 24; i++) {
    // vars[i] = ((double)rand()) / RAND_MAX;
    p_vars[i] = ((double)rand()) / RAND_MAX;
    // vars[i] *= 1e-4;
    p_vars[i] *= 1e-4;
  }

  // test for xpts and vars on failing plate case
  T xpts1[12] = {0.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,5.00000e-01,0.00000e+00,1.00000e+00,5.00000e-01,0.00000e+00};
  T vars1[24] = {1.56679e-05,4.00944e-05,1.29790e-05,1.08809e-05,9.98925e-05,2.18257e-05,5.12932e-05,8.39112e-05,6.12640e-05,2.96032e-05,6.37552e-05,5.24287e-05,4.00229e-05,8.91529e-05,2.83315e-05,3.52458e-05,8.07725e-05,9.19026e-05,6.97553e-06,9.49327e-05,5.25995e-05,8.60558e-06,1.92214e-05,6.63227e-05};

  // element 2
  T xpts2[12] = {1.00000e+00,0.00000e+00,0.00000e+00,2.00000e+00,0.00000e+00,0.00000e+00,1.00000e+00,5.00000e-01,0.00000e+00,2.00000e+00,5.00000e-01,0.00000e+00};
  T vars2[24] = {3.35223e-05,7.68230e-05,2.77775e-05,5.53970e-05,4.77397e-05,6.28871e-05,3.64784e-05,5.13401e-05,9.52230e-05,9.16195e-05,6.35712e-05,7.17297e-05,1.56679e-05,4.00944e-05,1.29790e-05,1.08809e-05,9.98925e-05,2.18257e-05,5.12932e-05,8.39112e-05,6.12640e-05,2.96032e-05,6.37552e-05,5.24287e-05};

  // element 3
  T xpts3[12] = {0.00000e+00,5.00000e-01,0.00000e+00,1.00000e+00,5.00000e-01,0.00000e+00,0.00000e+00,1.00000e+00,0.00000e+00,1.00000e+00,1.00000e+00,0.00000e+00};
  T vars3[24] = {1.41603e-05,6.06969e-05,1.63006e-06,2.42887e-05,1.37232e-05,8.04177e-05,1.56679e-05,4.00944e-05,1.29790e-05,1.08809e-05,9.98925e-05,2.18257e-05,4.93583e-05,9.72775e-05,2.92517e-05,7.71358e-05,5.26745e-05,7.69914e-05,4.00229e-05,8.91529e-05,2.83315e-05,3.52458e-05,8.07725e-05,9.19026e-05};

  // element 4
  T xpts4[12] = {1.00000e+00,5.00000e-01,0.00000e+00,2.00000e+00,5.00000e-01,0.00000e+00,1.00000e+00,1.00000e+00,0.00000e+00,2.00000e+00,1.00000e+00,0.00000e+00};
  T vars4[24] = {1.56679e-05,4.00944e-05,1.29790e-05,1.08809e-05,9.98925e-05,2.18257e-05,5.12932e-05,8.39112e-05,6.12640e-05,2.96032e-05,6.37552e-05,5.24287e-05,4.00229e-05,8.91529e-05,2.83315e-05,3.52458e-05,8.07725e-05,9.19026e-05,6.97553e-06,9.49327e-05,5.25995e-05,8.60558e-06,1.92214e-05,6.63227e-05};

  T xpts[12], vars[24];

  for (int ielem = 0; ielem < 4; ielem++) {
    if (ielem == 0) {
      memcpy(xpts, xpts1, 12 * sizeof(T));
      memcpy(vars, vars1, 24 * sizeof(T));
    } else if (ielem == 1) {
      memcpy(xpts, xpts2, 12 * sizeof(T));
      memcpy(vars, vars2, 24 * sizeof(T));
    } else if (ielem == 2) {
      memcpy(xpts, xpts3, 12 * sizeof(T));
      memcpy(vars, vars3, 24 * sizeof(T));
    } else if (ielem == 3) {
      memcpy(xpts, xpts4, 12 * sizeof(T));
      memcpy(vars, vars4, 24 * sizeof(T));
    }

    for (int iquad = 0; iquad < 4; iquad++) {
      T fail0;
      ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, vars, data, rhoKS, SF, fail0);
      printf("fail0 = %.8e\n", fail0);

      // HC sens
      T fail_du_sens[24];
      memset(fail_du_sens, 0.0, 24 * sizeof(T));
      ElemGroup::template compute_element_quadpt_failure_sv_sens<Data>(true, iquad, xpts, vars, data, rhoKS, SF, 1.0, fail_du_sens);
      T dks_HC = A2D::VecDotCore<T,24>(p_vars, fail_du_sens);

      if (ielem == 0 && iquad == 0) {
        printf("elem_vars:");
        printVec<T>(24, vars);
        printf("elem: fail_du_sens\n");
        printVec<T>(24, fail_du_sens);
      }

      // FD sens
      memset(new_vars, 0.0, 24 * sizeof(T));
      A2D::VecAddCore<T,24>(1.0, vars, new_vars);
      A2D::VecAddCore<T,24>(h, p_vars, new_vars);
      T fail1, failn1;
      ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, new_vars, data, rhoKS, SF, fail1);
      A2D::VecAddCore<T,24>(-2.0 * h, p_vars, new_vars);
      ElemGroup::template get_element_quadpt_failure_index<Data>(true, iquad, xpts, new_vars, data, rhoKS, SF, failn1);
      T dks_FD = (fail1 - failn1) / 2.0 / h;

      T rel_err = abs((dks_FD - dks_HC) / dks_FD);
      bool passed = rel_err < 1e-4;
      printTestReport("ShellElement ks SV (statevar) FD test", passed, rel_err);
      printf("\tdks_strain_elem FD %.8e HC %.8e\n", dks_FD, dks_HC);
    }
  }
}

template <typename T, class Assembler>
void ksfailure_SV_FD_test(Assembler &assembler, T &h2 = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  T h = 1e-4;

  // prelim, set initial vars in
  int nvars = assembler.get_num_vars();
  // perturbation
  HostVec<T> h_vars(nvars);
  for (int i = 0; i < nvars; i++) {
    h_vars[i] = ((double) rand()) / RAND_MAX;
    h_vars[i] *= 1e-4;
  }
  auto d_vars = h_vars.createDeviceVec();
  assembler.set_variables(d_vars);

  // temp debug, what if we zero d_vars? nonzero derivs still?
  // d_vars.zeroValues();

  // perturbation
  HostVec<T> h_pertvars(nvars);
  for (int i = 0; i < nvars; i++) {
    h_pertvars[i] = ((double) rand()) / RAND_MAX;
    h_pertvars[i] *= 1e-4;
  }
  // h_pertvars[0] = 1.0; // first state var
  auto d_pertvars = h_pertvars.createDeviceVec();

  // call KS on initial vars
  T rhoKS = 100.0;
  // T rhoKS = 1.0;
  T SF = 1.5;
  // T rhoKS = 1e-2;
  T ksfail0 = assembler._compute_ks_failure(rhoKS, SF);

  // ksfailure gradient
  DeviceVec<T> dksdu(nvars);
  assembler._compute_ks_failure_SVsens(rhoKS, SF, dksdu);
  // auto h_dksdu = dksdu.createHostVec();
  // printf("h_dksdu:");
  // printVec<T>(nvars, h_dksdu.getPtr());

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  /* directional deriv (pert with gradient) */
  T dks_HC = 0.0;
  // nvars = 10; // debug change how many vars tested in deriv at once
  cublasDdot(cublasHandle, nvars, dksdu.getPtr(), 1, d_pertvars.getPtr(), 1, &dks_HC);

  /* now FD test */
  T ksn1, ks1;

  // int ind = 0;
  // int ind = 2;
  // T *h_vars1 = d_vars.createHostVec().getPtr();
  // printf("h_vars0 = %.8e\n", h_vars1[ind]);

  // forward pert
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));
  assembler.set_variables(d_vars);
  ks1 = assembler._compute_ks_failure(rhoKS, SF);
  
  // T *h_vars2 = d_vars.createHostVec().getPtr();
  // printf("h_vars0+p_vars*h = %.8e\n", h_vars2[ind]);

  // backwards pert
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));
  assembler.set_variables(d_vars);
  ksn1 = assembler._compute_ks_failure(rhoKS, SF);

  // T *h_vars3 = d_vars.createHostVec().getPtr();
  // printf("h_vars0-p_vars*h = %.8e\n", h_vars3[ind]);

  // reset DVs
  a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, d_pertvars.getPtr(), 1, d_vars.getPtr(), 1));

  // printf("ksfailn1 %.8e, ksfail1 %.8e\n", ksn1, ks1);

  // report unittest result
  T dks_FD = (ks1 - ksn1) / 2 / h;
  T rel_err = abs((dks_FD - dks_HC) / dks_FD);
  bool passed = rel_err < 1e-4;
  printTestReport("assembler ks SV (statevar) FD test", passed, rel_err);
  // printf("ksfail SV sens test:\n");
  printf("\tks_FD %.8e dks_HC %.8e\n", dks_FD, dks_HC);
}

template <typename T, class ElemGroup, class Data>
void ksfail_elem_adjres_FD_test(T &h = 1e-6) {
  /* check shell elemgroup state var sens */
  h = 1e-4;
  double E_ = 70e9, nu = 0.3, thick = 0.02, rho = 2500.0, ys = 350e6;  // material & thick properties
  // debug
  thick = 1.0;
  Data data(E_, nu, thick, rho, ys);
  T xpts[12], vars[24], psi[24];
  for (int i = 0; i < 12; i++) {
    xpts[i] = ((double)rand()) / RAND_MAX;
  }
  for (int i = 0; i < 24; i++) {
    vars[i] = ((double)rand()) / RAND_MAX;
    vars[i] *= 1e-4;
    psi[i] = ((double)rand()) / RAND_MAX;
  }

  // HC sens
  int iquad = 0;
  T adjres_HC;
  ElemGroup::template compute_element_quadpt_adj_res_product<Data>(true, iquad, xpts, vars, data, psi, &adjres_HC);
  // ElemGroup::template compute_element_quadpt_adj_res_product2<Data>(true, iquad, xpts, vars, data, psi, &adjres_HC);

  // FD sens
  T res[24], res_FD[24];
  memset(res_FD, 0.0, 24 * sizeof(T));
  data.thick += h;
  memset(res, 0.0, 24 * sizeof(T));
  ElemGroup::template add_element_quadpt_residual<Data>(true, iquad, xpts, vars, data, res);
  A2D::VecAddCore<T,24>(0.5 / h, res, res_FD);
  data.thick -= 2 * h;
  memset(res, 0.0, 24 * sizeof(T));
  ElemGroup::template add_element_quadpt_residual<Data>(true, iquad, xpts, vars, data, res);
  A2D::VecAddCore<T,24>(-0.5 / h, res, res_FD);
  T adjres_FD = A2D::VecDotCore<T,24>(res_FD, psi);
  // printf("res_FD:");
  // printVec<T>(6, res_FD);

  T rel_err = abs((adjres_FD - adjres_HC) / adjres_FD);
  bool passed = rel_err < 1e-4;
  printTestReport("ShellElement ks adjres FD test", passed, rel_err);
  printf("adjres elem test: FD %.8e HC %.8e\n", adjres_FD, adjres_HC);
}

template <typename T, class Assembler>
void ksfailure_adj_res_prod_FD_test(Assembler &assembler, T &h = 1e-6) {
  /* finite diff directional deriv test of the mass function for isotropic shells */

  // prelim, set initial vars in
  int ndvs = assembler.get_num_dvs();
  T thick = 1e-2;
  HostVec<T> h_dvs(ndvs, thick);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // set vars pert and adjoint
  int nvars = assembler.get_num_vars();
  HostVec<T> h_psi(nvars);
  for (int i = 0; i < nvars; i++) {
    h_psi[i] = ((double) rand()) / RAND_MAX;
  }
  auto psi = h_psi.createDeviceVec();
  // perturbation
  HostVec<T> hpert(ndvs);
  for (int i = 0; i < ndvs; i++) {
    hpert[i] = ((double) rand()) / RAND_MAX;
  }
  auto d_pert = hpert.createDeviceVec();

  // call mass on init design
  auto f_int = assembler.createVarsVec();
  auto dfint_FD = assembler.createVarsVec();

  // compute the adjoint res product HC
  DeviceVec<T> dRdx(ndvs);
  assembler._compute_adjResProduct(psi, dRdx);

  /* Create CUBLAS context */
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  // compute the dot product of dRdx with x pert
  T adjres_HC;
  cublasDdot(cublasHandle, ndvs, dRdx.getPtr(), 1, d_pert.getPtr(), 1, &adjres_HC);

  // compute adjoint res product by FD
  // ---------------------------------

  // compute fint(x + p * h, u)
  T a = 1.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  assembler.add_residual(f_int);
  // add f_int / 2 / h into fint_FD
  a = 0.5 / h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, f_int.getPtr(), 1, dfint_FD.getPtr(), 1));

  // compute fint(x - p * h, u)
  a = -2.0 * h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndvs, &a, d_pert.getPtr(), 1, global_dvs.getPtr(), 1));
  assembler.set_design_variables(global_dvs);
  assembler.add_residual(f_int);
  // subtract f_int / 2 / h into fint_FD
  a = -0.5 / h;
  CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, f_int.getPtr(), 1, dfint_FD.getPtr(), 1));

  // compute the dot product of fint_FD with psi
  T adjres_FD;
  cublasDdot(cublasHandle, nvars, dfint_FD.getPtr(), 1, psi.getPtr(), 1, &adjres_FD);

  T rel_err = abs((adjres_FD - adjres_HC) / adjres_FD);
  bool passed = rel_err < 1e-4;
  printTestReport("Assembler ks adjres FD test", passed, rel_err);
  // printf("adjoint res product\n");
  printf("\tadjres_FD %.8e adjres_HC %.8e\n", adjres_FD, adjres_HC);
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

  // type declarations
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

  // temp test
  T h = 1e-6;
  ksfail_data_strain_FD_test<T, Data>(h);
  ksfail_elem_SV_FD_test<T, ElemGroup, Data>(h);
  ksfail_elem_adjres_FD_test<T, ElemGroup, Data>(h);
  // return 0;

  bool solve = true;
  auto assembler = makeAssembler<T, Assembler, Data>(comm, solve);

  // unittests
  // ---------
  
  mass_FD_test<T, Assembler>(assembler, h);
  ksfailure_DV_FD_test<T, Assembler>(assembler, h);
  ksfailure_SV_FD_test<T, Assembler>(assembler, h);
  ksfailure_adj_res_prod_FD_test<T, Assembler>(assembler, h);
};