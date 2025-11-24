#include "../test_commons.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "utils/local_utils.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

template <bool is_nonlinear>
void test_elemres_GPU() {
  using T = double;
//   bool print = false;
    // bool print = is_nonlinear;
    bool print = true;

    T cpu_NL_res[24], cpu_LIN_res[24];

    // const STRAIN my_strain = DRILL;
    // const STRAIN my_strain = TYING;
    // const STRAIN my_strain = BENDING;
    const STRAIN my_strain = ALL;

    if constexpr (my_strain == ALL) {
        T cpu_NL_res_all[24] = {5.76705909465822e+15,6.00566313548771e+15,6.24201552862050e+15,-6.39142354442196e+13,-1.56745622359615e+12,6.08050133696314e+13,-6.47856025812250e+15,-6.85394554456327e+15,-7.22707918330733e+15,-1.68776368878385e+14,-3.75694741528583e+12,1.61339545165627e+14,-6.11549220955595e+15,-5.57983748464603e+15,-5.05506081441068e+15,-1.36488359395817e+14,-2.70769511543578e+12,1.31150040282758e+14,6.82699337302024e+15,6.42811989372159e+15,6.04012446909751e+15,-3.16260015571479e+13,-5.18196637802530e+11,3.06152986541469e+13};
        T cpu_LIN_res_all[24] = {1.57656535218330e+15,1.78363791458271e+15,1.99465797439516e+15,-4.48442873729601e+11,1.89786490155055e+11,8.53706226644081e+11,-1.95663989406560e+15,-2.18564719983879e+15,-2.41860200302502e+15,-4.37136740698321e+11,4.63283650028628e+11,1.44077515856870e+12,-2.01710391245074e+15,-2.18230231174991e+15,-2.34973649239328e+15,-5.43219852602942e+11,3.28840326976651e+11,1.27797162436936e+12,2.39717845433303e+15,2.58431159700600e+15,2.77368052102314e+15,-5.54519222196723e+11,5.53504530468530e+10,6.90910500894797e+11};
        memcpy(cpu_LIN_res, cpu_LIN_res_all, 24 * sizeof(T));
        memcpy(cpu_NL_res, cpu_NL_res_all, 24 * sizeof(T));

    } else if constexpr (my_strain == DRILL) {
        T cpu_NL_res_drill[24] = {-1.39247108650785e+10,-1.53976137778393e+10,-1.68705166906001e+10,4.28172876739558e+09,-8.56345753479117e+09,4.28172876739560e+09,9.50600212679576e+09,1.09789050395565e+10,1.24518079523173e+10,1.28451863021869e+10,-2.56903726043738e+10,1.28451863021869e+10,1.39247108650792e+10,1.53976137778401e+10,1.68705166906009e+10,1.28451863021872e+10,-2.56903726043744e+10,1.28451863021873e+10,-9.50600212679639e+09,-1.09789050395572e+10,-1.24518079523181e+10,4.28172876739587e+09,-8.56345753479176e+09,4.28172876739591e+09};
        T cpu_LIN_res_drill[24] = {-1.39247108650785e+10,-1.53976137778393e+10,-1.68705166906001e+10,4.28172876739558e+09,-8.56345753479117e+09,4.28172876739560e+09,9.50600212679576e+09,1.09789050395565e+10,1.24518079523173e+10,1.28451863021869e+10,-2.56903726043738e+10,1.28451863021869e+10,1.39247108650792e+10,1.53976137778401e+10,1.68705166906009e+10,1.28451863021872e+10,-2.56903726043744e+10,1.28451863021873e+10,-9.50600212679639e+09,-1.09789050395572e+10,-1.24518079523181e+10,4.28172876739587e+09,-8.56345753479176e+09,4.28172876739591e+09};
        memcpy(cpu_LIN_res, cpu_LIN_res_drill, 24 * sizeof(T));
        memcpy(cpu_NL_res, cpu_NL_res_drill, 24 * sizeof(T));
    } else if constexpr (my_strain == TYING) {
        T cpu_NL_res_tying[24] = {5.76707262514010e+15,6.00567853310149e+15,6.24203279336618e+15,-6.39184042144594e+13,-1.55888909755306e+12,6.08006260193530e+13,-6.47857034986614e+15,-6.85395652346831e+15,-7.22709104937377e+15,-1.68789155212112e+14,-3.73125513310799e+12,1.61326644945896e+14,-6.11550422216717e+15,-5.57985288225981e+15,-5.05507959702702e+15,-1.36501205338395e+14,-2.68200476836781e+12,1.31137195801659e+14,6.82700194689322e+15,6.42813087262663e+15,6.04013785303462e+15,-3.16304543407423e+13,-5.09638732813092e+11,3.06111768751158e+13};
        T cpu_LIN_res_tying[24] = {1.57657927689417e+15,1.78365331219648e+15,1.99467484491185e+15,-4.52721227309598e+11,1.98353616198145e+11,8.49428459705885e+11,-1.95664940006773e+15,-2.18565817874383e+15,-2.41861445483297e+15,-4.49980115985846e+11,4.88975932206472e+11,1.42793198039879e+12,-2.01711783716160e+15,-2.18231770936369e+15,-2.34975336290997e+15,-5.56065032373778e+11,3.54530674044614e+11,1.26512638046300e+12,2.39718796033516e+15,2.58432257591104e+15,2.77369297283110e+15,-5.58806143697530e+11,6.39083580362870e+10,6.86622859770097e+11};
        memcpy(cpu_LIN_res, cpu_LIN_res_tying, 24 * sizeof(T));
        memcpy(cpu_NL_res, cpu_NL_res_tying, 24 * sizeof(T));
    } else if constexpr (my_strain == BENDING) {
        T cpu_NL_res_bending[24] = {3.94228987067582e+08,1.67017158224579e-07,-3.94228987067582e+08,-1.12958527644497e+08,-3.66850829920208e+06,1.05621511046093e+08,5.85741512128690e+08,-7.72605816379057e-06,-5.85741512128684e+08,-5.88525755023333e+07,-1.90957347047303e+06,5.50334285613872e+07,-1.91209965045887e+09,1.31966730564951e-05,1.91209965045886e+09,7.56276065478253e+05,2.55364120773841e+04,-7.05203241323480e+05,9.32129151262594e+08,-5.63763205092907e-06,-9.32129151262590e+08,1.71054827081353e+08,5.55254535759647e+06,-1.59949736366158e+08};
        T cpu_LIN_res_bending[24] = {-3.28921893605745e-09,-1.14587075879992e-10,3.06004478429742e-09,-3.37518739863449e+06,-3.66850829919994e+06,-3.96182919976538e+06,2.78006509099332e-08,1.09016376283077e-09,-2.56203233842714e-08,-1.81101466288637e+06,-1.90957347046510e+06,-2.00813227804383e+06,-4.05108492696416e-08,-1.61489705912207e-09,3.72810551513971e-08,-6.53134972312753e+03,2.55364120795369e+04,5.76041738822011e+04,1.59994172957659e-08,6.39320372171297e-10,-1.47207765514231e-08,5.19273341124401e+06,5.55254535758556e+06,5.91235730392705e+06};
        memcpy(cpu_LIN_res, cpu_LIN_res_bending, 24 * sizeof(T));
        memcpy(cpu_NL_res, cpu_NL_res_bending, 24 * sizeof(T));
    }

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, DenseMat>;

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
        // if (is_nonlinear) h_vars[ivar] *= 1e6;
        h_vars[ivar] *= 1e6;
        p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    }

    auto vars = h_vars.createDeviceVec();
    assembler.set_variables(vars);
    

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();
    // assembler.add_residual(res);
    assembler.add_residual_fast(res);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // compute total direc derivative of analytic residual
    auto h_res = res.createHostVec();
    T res_TD = A2D::VecDotCore<T, 24>(p_vars.getPtr(), h_res.getPtr());
    if (print) printf("Analytic residual\n");
    if (print) printf("res TD = %.8e\n", res_TD);

    // print data of host residual
    if (print) printf("res: ");
    if (print) printVecLong<double>(24, h_res.getPtr());

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