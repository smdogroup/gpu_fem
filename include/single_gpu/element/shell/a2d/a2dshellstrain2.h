#ifndef A2D_SHELL_STRAIN_H
#define A2D_SHELL_STRAIN_H

#include <type_traits>

#include "a2ddefs.h"
// #include "../core/a2dgreenstraincore.h"

namespace A2D {

enum class ShellStrainType { LINEAR, NONLINEAR };

/*
V2 just does bending here.. since everything else is linear
*/

template <typename T> //, bool tying, bool bending, bool drill>
A2D_FUNCTION void BendingStrainCore(const T u0x[9], const T u1x[9], T k[3]) {
    k[0] = u1x[0];           // k11
    k[1] = u1x[4];           // k22
    k[2] = u1x[1] + u1x[3];  // k12
}

template <typename T>
A2D_FUNCTION void BendingStrainForwardCore(const T u0xF[9], const T u1xF[9], T kF[3]) {
    BendingStrainCore<T, bending, tying, drill>(u0xF, u1xF, kF);
}

template <typename T>
A2D_FUNCTION void BendingStrainReverseCore(const T kb[3], T u0xb[9], T u1xb[9]) {
    u1xb[0] += kb[0];  // k11
    u1xb[4] += kb[1];  // k22
    u1xb[1] += kb[2];  // k12
    u1xb[3] += kb[2];  // k12
}

template <typename T>
A2D_FUNCTION void NonlinearBendingStrainCore(const T u0x[9], const T u1x[9], T k[3]) {

    // first get the linear part
    BendingStrainCore<T>(u0x, u1x, k);

    // only nonlinear part comes from bending
    if constexpr (bending) {
        k[0] += u0x[0] * u1x[0] + u0x[3] * u1x[3] + u0x[6] * u1x[6];  // k11
        k[1] += u0x[1] * u1x[1] + u0x[4] * u1x[4] + u0x[7] * u1x[7];  // k22
        k[2] = u0x[0] * u1x[1] + u0x[3] * u1x[4] + u0x[6] * u1x[7] + u1x[0] * u0x[1] +
                u1x[3] * u0x[4] + u1x[6] * u0x[7];  // k12
    }
}

template <typename T, bool bending, bool tying, bool drill>
A2D_FUNCTION void NonlinearShellStrainForwardCore(
    const T u0x[9], const T u1x[9], 
    const T u0xF[9], const T u1xF[9], const T e0tyF[6], const T etF[1],
    T EmF[3], T EbF[3], T EsF[2], T EtF[1]) {

    // linear part
    LinearShellStrainForwardCore<T, bending, tying, drill>(u0xF, u1xF, e0tyF, etF, EmF, EbF, EsF, EtF);

    // nonlinear part
    if constexpr (bending) {
        EbF[0] = u0xF[0] * u1x[0] + u0x[0] * u1xF[0] + +u0xF[3] * u1x[3] + u0x[3] * u1xF[3] +
                       u0xF[6] * u1x[6] + u0x[6] * u1xF[6];  // k11
        EbF[1] = u0xF[1] * u1x[1] + u0x[1] * u1xF[1] + u0xF[4] * u1x[4] + u0x[4] * u1xF[4] +
                        u0xF[7] * u1x[7] + u0x[7] * u1xF[7];  // k22
        EbF[2] = u0xF[0] * u1x[1] + u0x[0] * u1xF[1] + u0xF[3] * u1x[4] + u0x[3] * u1xF[4] +
                u0xF[6] * u1x[7] + u0x[6] * u1xF[7] + u1xF[0] * u0x[1] + u1x[0] * u0xF[1] +
                u1xF[3] * u0x[4] + u1x[3] * u0xF[4] + u1xF[6] * u0x[7] + u1x[6] * u0xF[7];  // k12
    }
}

template <typename T, bool bending, bool tying, bool drill>
A2D_FUNCTION void NonlinearShellStrainReverseCore(
    const T Emb[3], const T Ebb[3], const T Esb[2], const T Etb[1], 
    const T u0x[9], const T u1x[9],
    T u0xb[9], T u1xb[9], T e0tyb[6], T etb[1]) {

    // linear part
    LinearShellStrainReverseCore<T, bending, tying, drill>(Emb, Ebb, Esb, Etb, u0xb, u1xb, e0tyb, etb);

    if constexpr (bending) {
        // only nonlinear part is bending strains
        // k11 computation
        u0xb[0] += u1x[0] * Ebb[0];
        u1xb[0] += u0x[0] * Ebb[0];
        u0xb[3] += u1x[3] * Ebb[0];
        u1xb[3] += u0x[3] * Ebb[0];
        u0xb[6] += u1x[6] * Ebb[0];
        u1xb[6] += u0x[6] * Ebb[0];
        // k22 computation
        u0xb[1] += u1x[1] * Ebb[1];
        u1xb[1] += u0x[1] * Ebb[1];
        u0xb[4] += u1x[4] * Ebb[1];
        u1xb[4] += u0x[4] * Ebb[1];
        u0xb[7] += u1x[7] * Ebb[1];
        u1xb[7] += u0x[7] * Ebb[1];
        // k12 computation
        u0xb[0] += u1x[1] * Ebb[2];
        u1xb[0] += u0x[1] * Ebb[2];
        u0xb[1] += u1x[0] * Ebb[2];
        u1xb[1] += u0x[0] * Ebb[2];
        u0xb[3] += u1x[4] * Ebb[2];
        u1xb[3] += u0x[4] * Ebb[2];
        u0xb[4] += u1x[3] * Ebb[2];
        u1xb[4] += u0x[3] * Ebb[2];
        u0xb[6] += u1x[7] * Ebb[2];
        u1xb[6] += u0x[7] * Ebb[2];
        u0xb[7] += u1x[6] * Ebb[2];
        u1xb[7] += u0x[6] * Ebb[2];
    }
}

template <typename T, bool bending, bool tying, bool drill>
A2D_FUNCTION void NonlinearShellStrainHessianReverseCore(
    const T Emh[3], const T Ebh[3], const T Esh[2], const T Eth[1], 
    const T Emb[3], const T Ebb[3], const T Esb[2], const T Etb[1],
    const T u0x[9], const T u1x[9], 
    const T u0xp[9], const T u1xp[9],
    T u0xh[9], T u1xh[9], T e0tyh[6], T eth[1]) {

    // nonlinear backprop grad style strains_h => disp grads_h
    NonlinearShellStrainReverseCore<T, bending, tying, drill>(
        Emh, Ebh, Esh, Eth, u0x, u1x, u0xh, u1xh, e0th, eth
    );

    // extra nonlinear term from proj hessian 2nd order
    // strain_bar * d^2strains/d(disp_grad)^2 * disp_grad_p => proj hessian of strains
    // (p are forward derivs)
    if constexpr (bending) {
        // only nonlinear part is bending strains
        // k11 computation
        u0xh[0] += u1xp[0] * Ebb[0];
        u1xh[0] += u0xp[0] * Ebb[0];
        u0xh[3] += u1xp[3] * Ebb[0];
        u1xh[3] += u0xp[3] * Ebb[0];
        u0xh[6] += u1xp[6] * Ebb[0];
        u1xh[6] += u0xp[6] * Ebb[0];
        // k22 computatio
        u0xh[1] += u1xp[1] * Ebb[1];
        u1xh[1] += u0xp[1] * Ebb[1];
        u0xh[4] += u1xp[4] * Ebb[1];
        u1xh[4] += u0xp[4] * Ebb[1];
        u0xh[7] += u1xp[7] * Ebb[1];
        u1xh[7] += u0xp[7] * Ebb[1];
        // k12 computatio
        u0xh[0] += u1xp[1] * Ebb[2];
        u1xh[0] += u0xp[1] * Ebb[2];
        u0xh[1] += u1xp[0] * Ebb[2];
        u1xh[1] += u0xp[0] * Ebb[2];
        u0xh[3] += u1xp[4] * Ebb[2];
        u1xh[3] += u0xp[4] * Ebb[2];
        u0xh[4] += u1xp[3] * Ebb[2];
        u1xh[4] += u0xp[3] * Ebb[2];
        u0xh[6] += u1xp[7] * Ebb[2];
        u1xh[6] += u0xp[7] * Ebb[2];
        u0xh[7] += u1xp[6] * Ebb[2];
        u1xh[7] += u0xp[6] * Ebb[2];
    }
}

template <ShellStrainType straintype, typename T, bool bending, bool tying, bool drill>
A2D_FUNCTION void ShellStrain(const Mat<T, 3, 3>& u0x, const Mat<T, 3, 3>& u1x,
                              const SymMat<T, 3>& e0ty, const Vec<T, 1> et, 
                              Vec<T, 3> &Em, Vec<T, 3> &Eb, Vec<T, 2> &Es, Vec<T, 1> &Et) {
    if constexpr (straintype == ShellStrainType::LINEAR) {
        LinearShellStrainCore<T, bending, tying, drill>(
            get_data(u0x), get_data(u1x), get_data(e0ty), get_data(et),
            get_data(Em), get_data(Eb), get_data(Es), get_data(Et));
    } else {
        NonlinearShellStrainCore<T, bending, tying, drill>(
            get_data(u0x), get_data(u1x), get_data(e0ty), get_data(et),
            get_data(Em), get_data(Eb), get_data(Es), get_data(Et));
    }
}

template <ShellStrainType straintype, class u0xtype, class u1xtype, class e0tytype, class ettype,
          class etype>
class ShellStrainExpr {
   public:
    // Extract the numeric type to use
    typedef typename get_object_numeric_type<etype>::type T;

    // Extract the dimensions of the underlying matrix
    static constexpr int u0x_rows = get_matrix_rows<u0xtype>::size;
    static constexpr int u0x_cols = get_matrix_columns<u0xtype>::size;
    static constexpr int u1x_rows = get_matrix_rows<u1xtype>::size;
    static constexpr int u1x_cols = get_matrix_columns<u1xtype>::size;
    static constexpr int e0ty_size = get_symmatrix_size<e0tytype>::size;
    static constexpr int et_size = get_vec_size<ettype>::size;
    // static constexpr int e_size = get_vec_size<etype>::size;

    // make sure the correct sizes
    static_assert((u0x_rows == 3) && (u0x_cols == 3) && (u1x_rows == 3) && (u1x_cols == 3) &&
                      (e0ty_size == 3) && (et_size == 1)); //&& (e_size == 9),
                  "Shell Strain Expression does not have right size..");

    static constexpr ADiffType adu0x = get_diff_type<u0xtype>::diff_type;
    static constexpr ADiffType adu1x = get_diff_type<u1xtype>::diff_type;
    static constexpr ADiffType ade0ty = get_diff_type<e0tytype>::diff_type;
    static constexpr ADiffType adet = get_diff_type<ettype>::diff_type;

    // Get the differentiation order from the output
    static constexpr ADorder order = get_diff_order<etype>::order;

    // Make sure that the order matches
    static_assert(get_diff_order<u0xtype>::order == order, "ADorder does not match");

    A2D_FUNCTION ShellStrainExpr(u0xtype& u0x, u1xtype& u1x, e0tytype& e0ty, ettype& et, etype& e)
        : u0x(u0x), u1x(u1x), e0ty(e0ty), et(et), e(e) {}

    A2D_FUNCTION void eval() {
        if constexpr (straintype == ShellStrainType::LINEAR) {
            LinearShellStrainCore<T>(get_data(u0x), get_data(u1x), get_data(e0ty), get_data(et),
                                     get_data(e));
        } else {
            NonlinearShellStrainCore<T>(get_data(u0x), get_data(u1x), get_data(e0ty), get_data(et),
                                        get_data(e));
        }
    }

    A2D_FUNCTION void bzero() { e.bzero(); }

    template <ADorder forder>
    A2D_FUNCTION void forward() {
        static_assert(!(order == ADorder::FIRST and forder == ADorder::SECOND),
                      "Can't perform second order forward with first order objects");
        constexpr ADseed seed =
            conditional_value<ADseed, forder == ADorder::FIRST, ADseed::b, ADseed::p>::value;

        // need more statements here? (maybe some with only some pvalues transferred
        // forward at a time? see matSum expr)
        if constexpr (straintype == ShellStrainType::LINEAR) {
            LinearShellStrainForwardCore<T>(
                GetSeed<seed>::get_data(u0x), GetSeed<seed>::get_data(u1x),
                GetSeed<seed>::get_data(e0ty), GetSeed<seed>::get_data(et),
                GetSeed<seed>::get_data(e));
        } else {
            NonlinearShellStrainForwardCore<T>(
                get_data(u0x), get_data(u1x), get_data(e0ty), get_data(et),
                GetSeed<seed>::get_data(u0x), GetSeed<seed>::get_data(u1x),
                GetSeed<seed>::get_data(e0ty), GetSeed<seed>::get_data(et),
                GetSeed<seed>::get_data(e));
        }
    }

    A2D_FUNCTION void reverse() {
        constexpr ADseed seed = ADseed::b;
        // need more conditions on which ADseeds are active here
        if constexpr (straintype == ShellStrainType::LINEAR) {
            LinearShellStrainReverseCore<T>(
                GetSeed<seed>::get_data(e), GetSeed<seed>::get_data(u0x),
                GetSeed<seed>::get_data(u1x), GetSeed<seed>::get_data(e0ty),
                GetSeed<seed>::get_data(et));
        } else {
            NonlinearShellStrainReverseCore<T>(
                GetSeed<seed>::get_data(e), get_data(u0x), get_data(u1x), get_data(e0ty),
                get_data(et), GetSeed<seed>::get_data(u0x), GetSeed<seed>::get_data(u1x),
                GetSeed<seed>::get_data(e0ty), GetSeed<seed>::get_data(et));
        }
    }

    A2D_FUNCTION void hzero() { e.hzero(); }

    A2D_FUNCTION void hreverse() {
        // need more conditions on which ADseeds are active here
        constexpr ADseed seed = ADseed::h;
        if constexpr (straintype == ShellStrainType::LINEAR) {
            LinearShellStrainReverseCore<T>(
                GetSeed<seed>::get_data(e), GetSeed<seed>::get_data(u0x),
                GetSeed<seed>::get_data(u1x), GetSeed<seed>::get_data(e0ty),
                GetSeed<seed>::get_data(et));
        } else {
            NonlinearShellStrainHessianReverseCore<T>(
                GetSeed<ADseed::h>::get_data(e), GetSeed<ADseed::b>::get_data(e), get_data(u0x),
                get_data(u1x), get_data(e0ty), get_data(et), GetSeed<ADseed::p>::get_data(u0x),
                GetSeed<ADseed::p>::get_data(u1x), GetSeed<ADseed::p>::get_data(e0ty),
                GetSeed<ADseed::p>::get_data(et), GetSeed<ADseed::h>::get_data(u0x),
                GetSeed<ADseed::h>::get_data(u1x), GetSeed<ADseed::h>::get_data(e0ty),
                GetSeed<ADseed::h>::get_data(et));
        }
    }

    u0xtype& u0x;
    u1xtype& u1x;
    e0tytype& e0ty;
    ettype& et;
    etype& e;
};

// template <ShellStrainType straintype, typename T>
// A2D_FUNCTION void ShellStrain(const Mat<T,3,3> &u0x, const Mat<T,3,3> &u1x,
//                                 const SymMat<T,3> &e0ty, const T &et,
//                                 Vec<T,9> &e) {

template <ShellStrainType straintype, class u0xtype, class u1xtype, class e0tytype, class ettype,
          class etype>
A2D_FUNCTION auto ShellStrain(ADObj<u0xtype>& u0x, ADObj<u1xtype>& u1x, ADObj<e0tytype>& e0ty,
                              ADObj<ettype>& et, ADObj<etype>& e) {
    return ShellStrainExpr<straintype, ADObj<u0xtype>, ADObj<u1xtype>, ADObj<e0tytype>,
                           ADObj<ettype>, ADObj<etype>>(u0x, u1x, e0ty, et, e);
}

template <ShellStrainType straintype, class u0xtype, class u1xtype, class e0tytype, class ettype,
          class etype>
A2D_FUNCTION auto ShellStrain(A2DObj<u0xtype>& u0x, A2DObj<u1xtype>& u1x, A2DObj<e0tytype>& e0ty,
                              A2DObj<ettype>& et, A2DObj<etype>& e) {
    return ShellStrainExpr<straintype, A2DObj<u0xtype>, A2DObj<u1xtype>, A2DObj<e0tytype>,
                           A2DObj<ettype>, A2DObj<etype>>(u0x, u1x, e0ty, et, e);
}

namespace Test {

// template <ShellStrainType straintype, typename T>
// A2D_FUNCTION void ShellStrain(const Mat<T,3,3> &u0x, const Mat<T,3,3> &u1x,
//                                 const SymMat<T,3> &e0ty, const T &et,
//                                 Vec<T,9> &e)

template <ShellStrainType straintype, typename T>
class ShellStrainTest
    : public A2DTest<T, Vec<T, 9>, Mat<T, 3, 3>, Mat<T, 3, 3>, SymMat<T, 3>, Vec<T, 1>> {
   public:
    using Input = VarTuple<T, Mat<T, 3, 3>, Mat<T, 3, 3>, SymMat<T, 3>, Vec<T, 1>>;
    using Output = VarTuple<T, Vec<T, 9>>;

    // Assemble a string to describe the test
    std::string name() {
        std::stringstream s;
        s << "ShellStrain<";
        if (straintype == ShellStrainType::LINEAR) {
            s << "LINEAR>";
        } else {
            s << "NONLINEAR>";
        }

        return s.str();
    }

    // Evaluate the matrix-matrix product
    Output eval(const Input& x) {
        Mat<T, 3, 3> u0x, u1x;
        SymMat<T, 3> e0ty;
        Vec<T, 1> et;
        Vec<T, 9> e;
        x.get_values(u0x, u1x, e0ty, et);
        ShellStrain<straintype>(u0x, u1x, e0ty, et, e);
        return MakeVarTuple<T>(e);
    }

    // Compute the derivative
    void deriv(const Output& seed, const Input& x, Input& g) {
        ADObj<Mat<T, 3, 3>> u0x, u1x;
        ADObj<SymMat<T, 3>> e0ty;
        ADObj<Vec<T, 1>> et;
        ADObj<Vec<T, 9>> e;
        x.get_values(u0x.value(), u1x.value(), e0ty.value(), et.value());
        auto stack = MakeStack(ShellStrain<straintype>(u0x, u1x, e0ty, et, e));
        seed.get_values(e.bvalue());
        stack.reverse();
        g.set_values(u0x.bvalue(), u1x.bvalue(), e0ty.bvalue(), et.bvalue());
    }

    // Compute the second-derivative
    void hprod(const Output& seed, const Output& hval, const Input& x, const Input& p, Input& h) {
        A2DObj<Mat<T, 3, 3>> u0x, u1x;
        A2DObj<SymMat<T, 3>> e0ty;
        A2DObj<Vec<T, 1>> et;
        A2DObj<Vec<T, 9>> e;
        x.get_values(u0x.value(), u1x.value(), e0ty.value(), et.value());
        p.get_values(u0x.pvalue(), u1x.pvalue(), e0ty.pvalue(), et.pvalue());
        auto stack = MakeStack(ShellStrain<straintype>(u0x, u1x, e0ty, et, e));
        seed.get_values(e.bvalue());
        hval.get_values(e.hvalue());
        stack.hproduct();
        h.set_values(u0x.hvalue(), u1x.hvalue(), e0ty.hvalue(), et.hvalue());
    }
};

bool ShellStrainTestAll(bool component = false, bool write_output = true) {
    using Tc = A2D_complex_t<double>;

    ShellStrainTest<ShellStrainType::LINEAR, Tc> test1;
    bool passed = Run(test1, component, write_output);

    ShellStrainTest<ShellStrainType::NONLINEAR, Tc> test2;
    passed = passed && Run(test2, component, write_output);

    return passed;
}

}  // namespace Test

}  // namespace A2D

#endif  // A2D_SHELL_STRAIN_H
