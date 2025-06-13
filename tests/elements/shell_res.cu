#include "../test_commons.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "utils/local_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/shell_elem_group_v2.h"

template <bool is_nonlinear>
void test_elemres_GPU() {
  using T = double;
  bool print = false;

  // clang-format off
    T cpu_NL_res[] = {5.76705909465822e+15,6.00566313548771e+15,6.24201552862050e+15,
        -6.39142354442196e+13,-1.56745622359615e+12,6.08050133696314e+13,
        -6.47856025812250e+15,-6.85394554456327e+15,-7.22707918330733e+15,
        -1.68776368878385e+14,-3.75694741528583e+12,1.61339545165627e+14,
        -6.11549220955595e+15,-5.57983748464603e+15,-5.05506081441068e+15,
        -1.36488359395817e+14,-2.70769511543578e+12,1.31150040282758e+14,
        6.82699337302024e+15,6.42811989372159e+15,6.04012446909751e+15,
        -3.16260015571479e+13,-5.18196637802530e+11,3.06152986541469e+13};
    // clang-format off
    T cpu_LIN_res[] = {1.57656535218330e+09,1.78363791458271e+09,1.99465797439516e+09,
        -4.48442873729651e+05,1.89786490155125e+05,8.53706226644056e+05,
        -1.95663989406560e+09,-2.18564719983879e+09,-2.41860200302502e+09,
        -4.37136740698410e+05,4.63283650028749e+05,1.44077515856866e+06,
        -2.01710391245073e+09,-2.18230231174991e+09,-2.34973649239328e+09,
        -5.43219852602934e+05,3.28840326976585e+05,1.27797162436941e+06,
        2.39717845433303e+09,2.58431159700599e+09,2.77368052102314e+09,
        -5.54519222196700e+05,5.53504530467806e+04,6.90910500894843e+05};

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using ElemGroup = ShellElementGroupV2<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

    // printf("running!\n");
    int num_bcs = 2;
    auto assembler = createOneElementAssembler<Assembler>(num_bcs);

    // init variables u
    auto res = assembler.createVarsVec();
    int num_vars = assembler.get_num_vars();
    auto h_vars = HostVec<T>(num_vars);
    auto p_vars = HostVec<T>(num_vars);

    // fixed perturbations of the host and pert vars
    for (int ivar = 0; ivar < 24; ivar++) {
        h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
        if (is_nonlinear) h_vars[ivar] *= 1e6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    }

    auto vars = h_vars.createDeviceVec();
    assembler.set_variables(vars);

    // warmup kernel
    assembler.apply_bcs(res);
    

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_residual(res);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;

    // compute total direc derivative of analytic residual
    auto h_res = res.createHostVec();
    T res_TD = A2D::VecDotCore<T, 24>(p_vars.getPtr(), h_res.getPtr());
    if (print) printf("Analytic residual\n");
    if (print) printf("res TD = %.8e\n", res_TD);

    // print data of host residual
    if (print) printf("res: ");
    if (print) printVec<double>(24, h_res.getPtr());

    if constexpr (is_nonlinear) {
        double max_ref = max(24, cpu_NL_res);
        double max_abs_err = abs_err(h_res, cpu_NL_res);
        double rel_err = max_abs_err / max_ref;
        bool passed = rel_err < 1e-10;
        printTestReport("shell elem-res geom nonlinear", passed, rel_err);
        printf("\tabs err %.4e, max ref %.4e, norm err %.4e\n", max_abs_err, max_ref, rel_err);
    } else {
        double max_ref = max(24, cpu_LIN_res);
        double max_abs_err = abs_err(h_res, cpu_LIN_res);
        double rel_err = max_abs_err / max_ref;
        bool passed = rel_err < 1e-10;
        printTestReport("shell elem-res linear", passed, rel_err);
        printf("\tabs err %.4e, max ref %.4e, norm err %.4e\n", max_abs_err, max_ref, rel_err);
    }
    

    // old way with rel err check
    // double max_rel_err = rel_err(h_res, cpu_LIN_res, 1e-9);
    // bool passed = max_rel_err < 1e-5;
    // printTestReport("shell elem-res linear", passed, max_rel_err);

    printKernelTiming(duration.count());
}

int main() {
    
    test_elemres_GPU<false>(); // linear
    test_elemres_GPU<true>(); // nonlinear
    return 0;
};