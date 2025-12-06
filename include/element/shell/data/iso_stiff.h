#pragma once
#include "../../../cuda_utils.h"
#include "a2dcore.h"
#include "isotropic.h"

// maybe should be internal class of physics
template <typename T, bool has_ref_axis_ = true>
class StiffenedIsotropicShellData : public ShellIsotropicData<T, has_ref_axis_> {
   public:
    // TODO : can add flange fraction later if needed

    StiffenedIsotropicShellData() = default;
    static constexpr bool has_ref_axis = has_ref_axis_;
    static constexpr int ndvs_per_comp = 4;

    // constructor with ref Axis
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
    __HOST_DEVICE__ StiffenedIsotropicShellData(T E_, T nu_, T thick_, T stiffHeight_,
                                                T stiffThick_, T stiffPitch_, T refAxis_[],
                                                T rho_ = 1.0, T ys_ = 1.0, T tOffset_ = 0.0,
                                                T panelLength_ = 1.0)
        : stiffHeight(stiffHeight_),
          stiffThick(stiffThick_),
          stiffPitch(stiffPitch_),
          panelLength(panelLength_),
          ShellIsotropicData<T, has_ref_axis_>(E_, nu_, thick_, refAxis_, rho_, ys_, tOffset_) {}

    // constructor without refAxis input
    template <bool U = has_ref_axis, typename std::enable_if<!U, int>::type = 0>
    __HOST_DEVICE__ StiffenedIsotropicShellData(T E_, T nu_, T thick_, T stiffHeight_,
                                                T stiffThick_, T stiffPitch_, T rho_ = 1.0,
                                                T ys_ = 1.0, T tOffset_ = 0.0, T panelLength_ = 1.0)
        : stiffHeight(stiffHeight_),
          stiffThick(stiffThick_),
          stiffPitch(stiffPitch_),
          panelLength(panelLength_),
          ShellIsotropicData<T, has_ref_axis_>(E_, nu_, thick_, rho_, ys_, tOffset_) {}

    __HOST_DEVICE__ void set_design_variables(T loc_dvs[]) {
        this->thick = loc_dvs[0];
        stiffHeight = loc_dvs[1];
        stiffThick = loc_dvs[2];
        stiffPitch = loc_dvs[3];
    }

    // override panel MOI method so unstiffened superclass modifies it's stress
    __HOST_DEVICE__ T getOverallCentroid() const {
        // compute bending inertias about overall centroid so that B = 0 and avoid bending-axial
        // coupling uses weighted areas z1 to z2, where panel at z1=0 and stiffener at z2= -h/2 -
        // hs/2
        T panel_area = this->thick;                            // is area per unit width
        T stiff_area = stiffHeight * stiffThick / stiffPitch;  // per unit width
        // T z_panel = 0.0; the panel center (original)
        T z_stiff =
            this->thick / 2.0 +
            stiffHeight /
                2.0;  // stiffener above panel (doesn't really matter, cause gets squared later)
        return z_stiff * stiff_area / (panel_area + stiff_area);
    }
    template <int DV>
    __HOST_DEVICE__ T getOverallCentroidSens() const {
        T panel_area = this->thick;                            // is area per unit width
        T stiff_area = stiffHeight * stiffThick / stiffPitch;  // per unit width
        T z_stiff =
            this->thick / 2.0 +
            stiffHeight /
                2.0;  // stiffener above panel (doesn't really matter, cause gets squared later)
        T total_area = panel_area + stiff_area;

        if constexpr (DV == 0) {
            // panel thick deriv
            return 0.5 * stiff_area / total_area - z_stiff * stiff_area / total_area / total_area;
        } else if (DV == 1) {
            // stiffener height deriv
            T dstiff_area = stiff_area / stiffHeight;
            return (0.5 * stiff_area + z_stiff * dstiff_area) / total_area -
                   z_stiff * stiff_area / total_area / total_area * dstiff_area;
        } else if (DV == 2) {
            // stiffener thick
            T dstiff_area = stiff_area / stiffThick;
            return z_stiff * dstiff_area / total_area -
                   z_stiff * stiff_area / total_area / total_area * dstiff_area;
        } else if (DV == 3) {
            // stiffener pitch
            T dstiff_area = stiff_area * -1.0 / stiffPitch;
            return z_stiff * dstiff_area / total_area -
                   z_stiff * stiff_area / total_area / total_area * dstiff_area;
        }
    }
    // had override here before (but )
    __HOST_DEVICE__ T getPanelIzz() const {
        // use modified overall centroid here, int z^2 * dz = int (z+zc)^2 dz = int z^2 dz + zc^2 *
        // int dz
        T zc = getOverallCentroid();
        T tp = this->thick;
        T tp3 = tp * tp * tp;
        return tp3 / 12.0 + tp * zc * zc;
    }
    template <int DV>
    __HOST_DEVICE__ T getPanelIzzSens() const {
        // get Izz panel thick sens here to unstiffened panel subclass
        T zc = getOverallCentroid();
        T dzc = getOverallCentroidSens<DV>();  // get pthick sens
        T tp = this->thick;
        T zc_bar = 2.0 * tp * zc * dzc;
        if constexpr (DV == 0) {
            return tp * tp / 4.0 + zc * zc + zc_bar;
        } else {
            return zc_bar;
        }
    }
    __HOST_DEVICE__ void addPanelIzzSens(const T &scale, T dv_sens[]) const {
        // add + backprop from panel izz output back to dvs
        dv_sens[0] += scale * getPanelIzzSens<0>();
        dv_sens[1] += scale * getPanelIzzSens<1>();
        dv_sens[2] += scale * getPanelIzzSens<2>();
        dv_sens[3] += scale * getPanelIzzSens<3>();
    }
    __HOST_DEVICE__ T getStiffenerArea() const { return stiffHeight * stiffThick; }
    __HOST_DEVICE__ T getStiffenerI11() const {
        // TODO : add flange fraction
        T hs = stiffHeight, ts = stiffThick, sp = stiffPitch, zc = getOverallCentroid();
        return hs * ts / sp * (hs * hs / 12.0 + zc * zc);
    }
    template <int DV>
    __HOST_DEVICE__ T getStiffenerI11Sens() const {
        T hs = stiffHeight, ts = stiffThick, sp = stiffPitch, zc = getOverallCentroid();
        T dzc = getOverallCentroidSens<DV>();
        T zc_bar = hs * ts * 2.0 * zc * dzc;
        T eff_thick = hs * ts / sp;
        T term1 = eff_thick * hs * hs / 12.0, term2 = eff_thick * zc * zc;
        if constexpr (DV == 0) {
            return zc_bar;
        } else if (DV == 1) {
            return 3.0 * term1 / hs + term2 / hs + zc_bar;
        } else if (DV == 2) {
            return (term1 + term2) / ts + zc_bar;
        } else if (DV == 3) {
            return (term1 + term2) * -1.0 / sp + zc_bar;
        }
    }
    __HOST_DEVICE__ void addStiffenerI11Sens(const T &scale, T dv_sens[]) const {
        // add + backprop from panel izz output back to dvs
        dv_sens[0] += scale * getStiffenerI11Sens<0>();
        dv_sens[1] += scale * getStiffenerI11Sens<1>();
        dv_sens[2] += scale * getStiffenerI11Sens<2>();
        dv_sens[3] += scale * getStiffenerI11Sens<3>();
    }

    /* ------------------------------------------------ */
    /* failure and strain evaluations and sensitivities */
    /* ------------------------------------------------ */

    __HOST_DEVICE__ void evalStrainDVSensProduct(const T &scale, const T strain[],
                                                 const T psi_strain[], T loc_dv_sens[]) const {
        /* compute psi[E]^T * d^2Pi/dE/dx product at strain level (equiv to back at disp level) */
        // = psi[E]^T * dstress/dx = inner_prod(psi[E], stress_dot) for each DV

        // don't think I can just call the panel backprop here (cause multiple terms in the stress)
        // TBD:: need panel and stiffener backprop parts separately maybe
        // maybe I can also just call the panel subclass part here TBD

        /* 1) panel stress-panel thick derivative */
        T C[6];
        evalTangentStiffness2D(this->E, this->nu, C);
        // dPi/dE = stress[9] vector, here we compute dstress[9] the thickness derivs
        T dstress[9];
        // Nij = A * Eij, thick deriv is dNij = C * Eij
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strain, dstress);
        T dI = getPanelIzzSens<0>();  // dI/dpanel_thick
        A2D::SymMatVecCoreScale3x3<T, true>(dI, C, &strain[3], &dstress[3]);
        // compute transverse shear components
        T dAs = this->getTransShearCorrFactor() * C[5];
        T ddrill = this->getDrillingRegularization() * dAs;
        dstress[6] = dAs * strain[6];
        dstress[7] = dAs * strain[7];
        dstress[8] = ddrill * strain[8];
        // now compute <dstress, psi_strain> as adjoint resid product, only one dv
        loc_dv_sens[0] = scale * A2D::VecDotCore<T, 9>(psi_strain, dstress);

        /* 2) panel stress- other derivatives (only through bending or M = D * k overall centroid)
         */
        T dI1 = getPanelIzzSens<1>();  // dI/dstiffHeight
        // false for not additive
        A2D::SymMatVecCoreScale3x3<T, false>(dI1, C, &strain[3], &dstress[3]);
        // now only take dot product through bending stress + strain
        loc_dv_sens[1] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);
        // then repeat for derivs 2 and 3
        T dI2 = getPanelIzzSens<2>();  // dI/dstiffThick
        A2D::SymMatVecCoreScale3x3<T, false>(dI2, C, &strain[3], &dstress[3]);
        loc_dv_sens[2] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);
        T dI3 = getPanelIzzSens<3>();  // dI/dstiffPitch
        A2D::SymMatVecCoreScale3x3<T, false>(dI3, C, &strain[3], &dstress[3]);
        loc_dv_sens[3] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);

        /* 3) take derivatives now through smeared stiffener stress (all four derivs) */
        // 3.1) first derivs through s11 += EA/sp * e11, A11 smeared stiffness
        T A11_stiff = getStiffenerArea() / stiffPitch;
        T A11_energy = scale * A11_stiff * psi_strain[0] * strain[0];
        loc_dv_sens[1] += A11_energy / stiffHeight;
        loc_dv_sens[2] += A11_energy / stiffThick;
        loc_dv_sens[3] += A11_energy * -1.0 / stiffPitch;

        // 3.2) take derivatives through trv shear stiffnesses
        T A44_stiff = A11_stiff * this->getTransShearCorrFactor();
        T A44_energy = scale * A44_stiff * (psi_strain[6] * strain[6] + psi_strain[7] * strain[7]);
        loc_dv_sens[1] += A44_energy / stiffHeight;
        loc_dv_sens[2] += A44_energy / stiffThick;
        loc_dv_sens[3] += A44_energy * -1.0 / stiffPitch;

        // 3.3) take derivatives through stiffener bend stiffness D11
        // stiffener I11 = E * I / sp (so already sp norm or technically I11 per unit width)
        T D11_energy = scale * psi_strain[3] * strain[3];
        loc_dv_sens[0] += D11_energy * getStiffenerI11Sens<0>();
        loc_dv_sens[1] += D11_energy * getStiffenerI11Sens<1>();
        loc_dv_sens[2] += D11_energy * getStiffenerI11Sens<2>();
        loc_dv_sens[3] += D11_energy * getStiffenerI11Sens<3>();
    }

    __HOST_DEVICE__ T evalFailure(const T &rhoKS, const T &safetyFactor, const T e[9]) const {
        // von Mises failure index, use ks max for to pand bottom stresses
        T fails[3];  // 1) panel strength, 2) local buckling, 3) global buckling
        T max = _getFailModes(rhoKS, safetyFactor, e, fails);

        /* 4) compute combined failure criterion among all three fail modes */
        T ks_sum = exp(rhoKS * (fails[0] - max)) + exp(rhoKS * (fails[1] - max)) +
                   exp(rhoKS * (fails[2] - max));
        T ks_fail = log(ks_sum) / rhoKS;
        return ks_fail;
    }

    __HOST_DEVICE__ void evalFailureDVSens(const T &rhoKS, const T &safetyFactor, const T e[9],
                                           const T &scale, T dv_sens[]) const {
        /* compute dsigma_KS/dthick */

        /* 0) backprop from ks_fail output to individual fail mode sens */
        //   first need individual fail modes and the standard max for ks fail computation
        T fails[3];  // 1) panel strength, 2) local buckling, 3) global buckling
        T max = _getFailModes(rhoKS, safetyFactor, e, fails);
        T exp_fails[3] = {exp(rhoKS * (fails[0] - max)), exp(rhoKS * (fails[1] - max)),
                          exp(rhoKS * (fails[2] - max))};
        // backprop leads to partition of unity on worst failure mode here
        T dfails[3];  // failure sens
        T exp_sum = exp_fails[0] + exp_fails[1] + exp_fails[2];
        dfails[0] = scale * exp_fails[0] / exp_sum;
        dfails[1] = scale * exp_fails[1] / exp_sum;
        dfails[2] = scale * exp_fails[2] / exp_sum;

        /* 1) backprop through panel strength failure */
        // call unstiff panel backprop (only to panel thick), since this eval is unchanged by
        // stiffening, only difference is scale changed to dfails[0] sens
        ShellIsotropicData<T, has_ref_axis>::evalFailureDVSens(rhoKS, safetyFactor, e, dfails[0],
                                                               dv_sens);

        /* 2) backprop through local panel buckling */
        // some needed coeffs
        T E = this->E, nu = this->nu;
        T A11 = E * this->thick / (1 - nu * nu), A11s = E * stiffHeight * stiffThick / stiffPitch;
        T A66 = A11 * (1 - nu) / 2.0;
        // e[0] = e11, e[3] = e22, e[1] = e12
        T N11 = A11 * (e[0] + nu * e[3]) + A11s * e[0], N12 = A66 * e[1];
        const T pi2 = 9.869604401089358;  // pi^2
        T sp2 = stiffPitch * stiffPitch, D_loc = E * getPanelIzz();
        T Ncr = D_loc * pi2 / sp2;  // generic Ncr without consts
        // fails[1] = -c1*thick/Ip - c2 * h2 * ts / sp / Ip + c3 * thick^2 / Ip^2
        //   use power series rules on each term to deriv efficiently
        T Lterm1 = -A11 * (e[0] + nu * e[3]) / Ncr / 4.0;
        T Lterm2 = -A11s * e[0] / Ncr / 4.0;
        T Lterm3 = A66 * A66 * e[1] * e[1] / Ncr / Ncr / 5.374 / 5.374;
        // first all non Ncr derivs
        dv_sens[0] += dfails[1] * Lterm1 / this->thick + 2.0 * Lterm3 / this->thick;
        dv_sens[1] += dfails[1] * Lterm2 / stiffHeight;
        dv_sens[2] += dfails[1] * Lterm2 / stiffThick;
        dv_sens[3] -= dfails[1] * Lterm2 / stiffPitch;  // exponent -1.0
        // then all Ncr derivs (aka Ip panel bending stiff derivs)
        T Ip_Lsens = -1.0 * dfails[1] * (Lterm1 + Lterm2 + 2.0 * Lterm3) / getPanelIzz();
        addPanelIzzSens(Ip_Lsens, dv_sens);

        /* 3) backprop through global panel buckling aka fails[2] sens */
        // similarly define separate additive terms and use power series rules for efficient
        // derivatives term1 = -c1*thick/D_axial, term2 = -c2 * hs*ts/sp / D_axial, term3 =
        // c3*thick^2/D_shear^2
        T a2 = panelLength * panelLength;
        T D_axial = D_loc + E * getStiffenerI11();
        T D_shear = D_loc + 0.5 / 4.7 * E * getStiffenerI11();  // shear is less affected by stiff
        T Ncr_axial = D_axial * pi2 / a2, Ncr_shear = D_shear * pi2 / a2 * 4.7;
        T Gterm1 = -A11 * (e[0] + nu * e[3]) / Ncr_axial;
        T Gterm2 = -A11s * e[0] / Ncr_axial;
        T Gterm3 = A66 * A66 * e[1] * e[1] / Ncr_shear;
        // now backprop through standard in-plane loads again (same as local but diff start sens)
        dv_sens[0] += dfails[2] * Gterm1 / this->thick + 2.0 * Gterm3 / this->thick;
        dv_sens[1] += dfails[2] * Gterm2 / stiffHeight;
        dv_sens[2] += dfails[2] * Gterm2 / stiffThick;
        dv_sens[3] -= dfails[2] * Gterm2 / stiffPitch;  // exponent -1.0
        // then backprop to the bending stiffnesses D_axial, D_shear first
        T Da_bar = -1.0 * dfails[2] * (Gterm1 + Gterm2) / D_axial;
        T Ds_bar = -2.0 * dfails[2] * Gterm3 / D_axial / D_axial;
        // backprop from D_axial and D_shear sens to panel Ip and stiff inertia Is
        T Ip_Gsens = (Da_bar + Ds_bar) / E, Is_Gsens = (Da_bar + 0.5 / 4.7 * Ds_bar) / E;
        addPanelIzzSens(Ip_Gsens, dv_sens);
        addStiffenerI11Sens(Is_Gsens, dv_sens);
    }

    __HOST_DEVICE__ void evalFailureStrainSens(const T &scale, const T &rhoKS,
                                               const T &safetyFactor, const T e[9], T er[9]) const {
        /* compute dsigma_KS/dstrain */

        /* 0) backprop from ks_fail output to individual fail mode sens */
        //   first need individual fail modes and the standard max for ks fail computation
        T fails[3];  // 1) panel strength, 2) local buckling, 3) global buckling
        T max = _getFailModes(rhoKS, safetyFactor, e, fails);
        T exp_fails[3] = {exp(rhoKS * (fails[0] - max)), exp(rhoKS * (fails[1] - max)),
                          exp(rhoKS * (fails[2] - max))};
        // backprop leads to partition of unity on worst failure mode here
        T dfails[3];  // failure sens
        T exp_sum = exp_fails[0] + exp_fails[1] + exp_fails[2];
        dfails[0] = scale * exp_fails[0] / exp_sum;
        dfails[1] = scale * exp_fails[1] / exp_sum;
        dfails[2] = scale * exp_fails[2] / exp_sum;

        /* 1) backprop through panel strength failure */
        ShellIsotropicData<T, has_ref_axis>::evalFailureStrainSens(dfails[0], rhoKS, safetyFactor, e,
                                                                   er);

        /* 2) backprop through local panel buckling */
        // some needed coeffs,  e[0] = e11, e[3] = e22, e[1] = e12
        T E = this->E, nu = this->nu;
        T A11 = E * this->thick / (1 - nu * nu), A11s = E * stiffHeight * stiffThick / stiffPitch;
        T A66 = A11 * (1 - nu) / 2.0;
        const T pi2 = 9.869604401089358;  // pi^2
        T sp2 = stiffPitch * stiffPitch, D_loc = E * getPanelIzz();
        T Ncr = D_loc * pi2 / sp2;  // generic Ncr without consts
        // only three terms, c1 * e[0] + c2 * e[3] + c3 * e[1]^2, easy to differentiate
        T Lc1 = -(A11 + A11s) / Ncr / 4.0;
        T Lc2 = -(A11 * nu) / Ncr / 4.0;
        T Lc3 = A66 * A66 / Ncr / Ncr / 5.374 / 5.374;
        // backprop to strains now
        er[0] += Lc1 * dfails[1];
        er[3] += Lc2 * dfails[1];
        er[1] += 2.0 * Lc3 * dfails[1];

        /* 3) backprop through global panel buckling aka fails[2] sens */
        // similarly define separate additive terms and use power series rules for efficient
        // derivatives term1 = -c1*thick/D_axial, term2 = -c2 * hs*ts/sp / D_axial, term3 =
        // c3*thick^2/D_shear^2
        T a2 = panelLength * panelLength;
        T D_axial = D_loc + E * getStiffenerI11();
        T D_shear = D_loc + 0.5 / 4.7 * E * getStiffenerI11();  // shear is less affected by stiff
        T Ncr_axial = D_axial * pi2 / a2, Ncr_shear = D_shear * pi2 / a2 * 4.7;
        // only three terms, c1 * e[0] + c2 * e[3] + c3 * e[1]^2, easy to differentiate
        T Gc1 = -(A11 + A11s) / Ncr_axial;
        T Gc2 = -(A11 * nu) / Ncr_axial;
        T Gc3 = A66 * A66 / Ncr_shear / Ncr_shear;
        // backprop to strains now
        er[0] += Gc1 * dfails[2];
        er[3] += Gc2 * dfails[2];
        er[1] += 2.0 * Gc3 * dfails[2];
    }

    __HOST_DEVICE__ T _getFailModes(const T &rhoKS, const T &safetyFactor, const T e[9],
                                    T fails[3]) const {
        /* 1) panel strength failure */
        // smeared stiffener doesn't affect top+bottom panel stress (only function of panel props)
        //    so unchanged from unstiffened panel subclass
        T panel_strength_fail =
            ShellIsotropicData<T, has_ref_axis>::evalFailure(rhoKS, safetyFactor, e);
        fails[0] = panel_strength_fail;

        /* 2 and 3 pre) compute in-plane loads */
        T E = this->E, nu = this->nu;
        T A11 = E * this->thick / (1 - nu * nu), A11s = E * stiffHeight * stiffThick / stiffPitch;
        T A66 = A11 * (1 - nu) / 2.0;
        // e[0] = e11, e[3] = e22, e[1] = e12
        T N11 = A11 * (e[0] + nu * e[3]) + A11s * e[0], N12 = A66 * e[1];

        /* 2) local buckling failure */
        const T pi2 = 9.869604401089358;  // pi^2
        T sp2 = stiffPitch * stiffPitch, D_loc = E * getPanelIzz();
        T Ncr = D_loc * pi2 / sp2;  // generic Ncr without consts
        T loc_axial = -N11 / Ncr / 4.0;
        T loc_shear = N12 / Ncr / 5.374;
        fails[1] = loc_axial + loc_shear * loc_shear;
        fails[1] *= safetyFactor;

        /* 3) global buckling failure */
        T a2 = panelLength * panelLength;
        T D_axial = D_loc + E * getStiffenerI11();
        T D_shear = D_loc + 0.5 / 4.7 * E * getStiffenerI11();  // shear is less affected by stiff
        T Ncr_axial = D_axial * pi2 / a2;
        T Ncr_shear = D_shear * pi2 / a2 * 4.7;
        T glob_axial = -N11 / Ncr_axial, glob_shear = N12 / Ncr_shear;
        fails[2] = glob_axial + glob_shear * glob_shear;
        fails[2] *= safetyFactor;

        // then get standard max here also
        T max12 = (fails[0] > fails[1]) ? fails[0] : fails[1];
        T max = (max12 > fails[2]) ? max12 : fails[2];
        return max;
    }

    // private:
    T stiffHeight, stiffThick, stiffPitch;
    T panelLength;
    // T panelWidth;
};
