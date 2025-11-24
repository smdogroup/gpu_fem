#pragma once
#include "../../../cuda_utils.h"
#include "a2dcore.h"

struct ShellConstants {
    static constexpr double transverseShearCorrectionFactor = 5.0 / 6.0;
    static constexpr double drillingRegularization = 10.0;
    // static constexpr double drillingRegularization = 50.0;
};

// could be an internal class for each type of physics
// but I would like physics to be linear vs nonlinear and this class has methods which get ABD, etc.
template <typename T, bool has_ref_axis_ = true>
class ShellIsotropicData {
   public:
    // static constexpr double transverseShearCorrectionFactor = 5.0 / 6.0;
    // static constexpr double drillingRegularization = 10.0;  // default drilling regularization

    ShellIsotropicData() = default;
    static constexpr bool has_ref_axis = has_ref_axis_;
    static constexpr int ndvs_per_comp = 1;  // just thickness for metals

    // constructor with ref Axis
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T refAxis_[], T rho_ = 1.0,
                                       T ys_ = 1.0, T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {
        // deep copy the ref axis
        for (int i = 0; i < 3; i++) {
            refAxis[i] = refAxis_[i];
        }
    }

    // constructor without refAxis input
    template <bool U = has_ref_axis, typename std::enable_if<!U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T rho_ = 1.0, T ys_ = 1.0,
                                       T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {}

    __HOST_DEVICE__ T getPanelIzz() const { return thick * thick * thick / 12.0; }
    __HOST_DEVICE__ T getPanelIzzSens() const { return thick * thick / 4.0; }

    // constitutive methods
    __HOST_DEVICE__ static void evalTangentStiffness2D(const T E_, const T nu_, T C[]) {
        // isotropic C matrix for stress = C * strain in-plane with no twist
        T D = E_ / (1.0 - nu_ * nu_);
        C[0] = D;
        C[1] = nu_ * D;
        C[2] = 0.0;
        C[3] = D;
        C[4] = 0.0;
        T G = E_ / 2.0 / (1.0 + nu_);
        C[5] = G;
    }

    __HOST_DEVICE__ void evalStrainDVSensProduct(const T &scale, const T strain[],
                                                 const T psi_strain[], T loc_dv_sens[]) const {
        /* compute psi[E]^T * d^2Pi/dE/dx product at strain level (equiv to back at disp level) */
        T C[6];
        evalTangentStiffness2D(E, nu, C);

        // dPi/dE = stress[9] vector, here we compute dstress[9] the thickness derivs
        T dstress[9];
        // Nij = A * Eij, thick deriv is dNij = C * Eij
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strain, dstress);
        // Nij += B * kij; deriv is dNij += C * -2 * thick * tOffset
        // A2D::SymMatVecCoreScale3x3<T, true>(-2.0 * thick * tOffset, C, &strain[3], dstress);
        // Mij = B * Eij, deriv is dMij = C * 2 * -thick * tOffset
        // A2D::SymMatVecCoreScale3x3<T, false>(-2.0 * thick * tOffset, C, strain, &dstress[3]);
        // Mij += D * kij; deriv is dD/dx = (t^2/4 + tOffset^2 * 3 * thick^2) * C
        T dI = getPanelIzzSens();
        A2D::SymMatVecCoreScale3x3<T, true>(dI, C, &strain[3], &dstress[3]);
        // A2D::SymMatVecCoreScale3x3<T, true>(3.0 * thick * thick * tOffset * tOffset, C,
        // &strain[3],
        //                                     &dstress[3]);

        // compute transverse shear components
        T dAs = getTransShearCorrFactor() * C[5];
        T ddrill = getDrillingRegularization() * dAs;
        dstress[6] = dAs * strain[6];
        dstress[7] = dAs * strain[7];
        dstress[8] = ddrill * strain[8];

        // now compute <dstress, psi_strain> as adjoint resid product, only one dv
        loc_dv_sens[0] = scale * A2D::VecDotCore<T, 9>(psi_strain, dstress);
    }

    __HOST_DEVICE__ static T vonMisesFailure2D(const T s[], const T &ys, const T &sf) {
        return sqrt(s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3.0 * s[2] * s[2]) / ys *
               sf;  // sf is safety factor
    }

    __HOST_DEVICE__ static T vonMisesFailure2DSens(const T s[], const T ds[], const T &ys,
                                                   const T &sf, const T &vm) {
        return (s[0] * ds[0] + s[1] * ds[1] - 0.5 * (s[0] * ds[1] + s[1] * ds[0]) +
                3 * s[2] * ds[2]) *
               sf * sf / (ys * ys) / (vm + 1e-12);  // divide by ys twice so equiv to 1/sqrt()
    }

    __HOST_DEVICE__ static void vonMises2DFailureRevSens(const T &scale, const T &ys, const T &sf,
                                                         const T &vm, const T s[], T sb[]) {
        T jac = scale * sf * sf / ys / ys / 2.0 / (vm + 1e-12);
        sb[0] = jac * (2 * s[0] - s[1]);
        sb[1] = jac * (2 * s[1] - s[0]);
        sb[2] = jac * 6 * s[2];
    }

    __HOST_DEVICE__ T evalFailure(const T &rhoKS, const T &safetyFactor, const T e[9]) const {
        // von Mises failure index, use ks max for to pand bottom stresses
        T et[3], eb[3];
        T ht = (0.5 - tOffset) * thick;
        T hb = -(0.5 + tOffset) * thick;

        et[0] = e[0] + ht * e[3];
        et[1] = e[1] + ht * e[4];
        et[2] = e[2] + ht * e[5];

        eb[0] = e[0] + hb * e[3];
        eb[1] = e[1] + hb * e[4];
        eb[2] = e[2] + hb * e[5];

        T C[6];
        evalTangentStiffness2D(E, nu, C);

        T st[3], sb[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, et, st);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, eb, sb);

        T top = vonMisesFailure2D(st, ys, safetyFactor);
        T bot = vonMisesFailure2D(sb, ys, safetyFactor);

        // get max first
        T max = (top > bot) ? top : bot;

        // then KS max using max offset to prevent overflow
        T ksSum = exp(rhoKS * (top - max)) + exp(rhoKS * (bot - max));
        return max + log(ksSum) / rhoKS;
    }

    __HOST_DEVICE__ void evalFailureDVSens(const T &rhoKS, const T &safetyFactor, const T e[9],
                                           const T &scale, T dv_sens[]) const {
        /* compute dsigma_KS/dthick */

        // forward analysis part
        // ---------------------
        // von Mises failure index, use ks max for to pand bottom stresses
        T et[3], eb[3];
        T ht = (0.5 - tOffset) * thick;
        T hb = -(0.5 + tOffset) * thick;

        et[0] = e[0] + ht * e[3];
        et[1] = e[1] + ht * e[4];
        et[2] = e[2] + ht * e[5];

        eb[0] = e[0] + hb * e[3];
        eb[1] = e[1] + hb * e[4];
        eb[2] = e[2] + hb * e[5];

        T C[6];
        evalTangentStiffness2D(E, nu, C);

        T st[3], sb[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, et, st);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, eb, sb);

        T top = vonMisesFailure2D(st, ys, safetyFactor);
        T bot = vonMisesFailure2D(sb, ys, safetyFactor);

        // get max first
        T max = (top > bot) ? top : bot;

        // then KS max using max offset to prevent overflow
        T eTop = exp(rhoKS * (top - max));
        T eBot = exp(rhoKS * (bot - max));
        T ksSum = eTop + eBot;

        // derivatives (forward AD style)
        // ------------------------------

        T dht = (0.5 - tOffset);
        T dhb = -(0.5 + tOffset);
        T det[3], deb[3];
        det[0] = dht * e[3];
        det[1] = dht * e[4];
        det[2] = dht * e[5];
        deb[0] = dhb * e[3];
        deb[1] = dhb * e[4];
        deb[2] = dhb * e[5];
        T dst[3], dsb[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, det, dst);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, deb, dsb);

        T dtop = vonMisesFailure2DSens(st, dst, ys, safetyFactor, top);
        T dbot = vonMisesFailure2DSens(sb, dsb, ys, safetyFactor, bot);

        dv_sens[0] = scale / ksSum * (eTop * dtop + eBot * dbot);
    }

    __HOST_DEVICE__ void evalFailureStrainSens(const T &scale, const T &rhoKS,
                                               const T &safetyFactor, const T e[9], T er[9]) const {
        /* compute dsigma_KS/dstrain */

        // forward analysis part
        // ---------------------
        // von Mises failure index, use ks max for to pand bottom stresses
        T et[3], eb[3];
        T ht = (0.5 - tOffset) * thick;
        T hb = -(0.5 + tOffset) * thick;

        et[0] = e[0] + ht * e[3];
        et[1] = e[1] + ht * e[4];
        et[2] = e[2] + ht * e[5];

        eb[0] = e[0] + hb * e[3];
        eb[1] = e[1] + hb * e[4];
        eb[2] = e[2] + hb * e[5];

        T C[6];
        evalTangentStiffness2D(E, nu, C);

        T st[3], sb[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, et, st);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, eb, sb);

        T top = vonMisesFailure2D(st, ys, safetyFactor);
        T bot = vonMisesFailure2D(sb, ys, safetyFactor);

        // get max first
        T max = (top > bot) ? top : bot;

        // then KS max using max offset to prevent overflow
        T eTop = exp(rhoKS * (top - max));
        T eBot = exp(rhoKS * (bot - max));
        T ksSum = eTop + eBot;

        // derivatives (reverse AD style)
        // ------------------------------

        T dtop = eTop * scale / ksSum;
        T dbot = eBot * scale / ksSum;

        T dst[3], dsb[3];
        vonMises2DFailureRevSens(dtop, ys, safetyFactor, top, st, dst);
        vonMises2DFailureRevSens(dbot, ys, safetyFactor, bot, sb, dsb);

        T det[3], deb[3];
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, dst, det);
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, dsb, deb);

#pragma unroll
        for (int i = 0; i < 3; i++) {
            er[i] = det[i] + deb[i];
            er[3 + i] = ht * det[i] + hb * deb[i];
        }

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        //     printf("ksSum %.4e\n", ksSum);
        //     printf("scale %.4e\n", scale);
        //     printf("dtop %.4e, dbot %.4e\n", dtop, dbot);
        //     printf("dst:");
        //     printVec<T>(3, dst);
        //     printf("dsb:");
        //     printVec<T>(3, dsb);
        // }
    }

    __HOST_DEVICE__ static T getTransShearCorrFactor() { return T(5.0 / 6.0); }
    __HOST_DEVICE__ static T getDrillingRegularization() { return T(10.0); }

    __HOST_DEVICE__ void set_design_variables(T loc_dvs[]) { thick = loc_dvs[0]; }

    // TODO : add thickness offset (rarely used but to be complete.. can allow B matrix to be
    // nonzero)
    T E, nu, thick, tOffset, ys, rho;
    T refAxis[3];
};