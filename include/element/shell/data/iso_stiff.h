#pragma once
#include "../../../cuda_utils.h"
#include "a2dcore.h"
#include "isotropic.h"

// maybe should be internal class of physics
template <typename T, bool has_ref_axis_ = true>
class StiffenedIsotropicShellData : public ShellIsotropicData<T, ref_axis> {
   public:
    // TODO : can add flange fraction later if needed

    StiffenedIsotropicShellData() = default;
    static constexpr bool has_ref_axis = has_ref_axis_;
    static constexpr int ndvs_per_comp = 4; 

    // constructor with ref Axis
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T stiffHeight_, T stiffThick_, 
                                       T stiffPitch_, T refAxis_[], T rho_ = 1.0,
                                       T ys_ = 1.0, T tOffset_ = 0.0) :
    stiffHeight(stiffHeight_), stiffThick(stiffThick_),
    stiffPitch(stiffPitch_), panelLength(1.0), ShellIsotropicData<T, ref_axis>(E_, nu_, thick_, refAxis_, rho_, ys_, tOffset_) {}

    // constructor without refAxis input
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicData(T E_, T nu_, T thick_, T stiffHeight_, T stiffThick_, 
                                       T stiffPitch_, T rho_ = 1.0,
                                       T ys_ = 1.0, T tOffset_ = 0.0) :
    stiffHeight(stiffHeight_), stiffThick(stiffThick_),
    stiffPitch(stiffPitch_), panelLength(1.0), ShellIsotropicData<T, ref_axis>(E_, nu_, thick_, rho_, ys_, tOffset_) {}

    __HOST_DEVICE__ void set_design_variables(T loc_dvs[]) { 
        thick = loc_dvs[0];
        stiffHeight = loc_dvs[1];
        stiffThick = loc_dvs[2];
        stiffPitch = loc_dvs[3];
    }

    // override panel MOI method so unstiffened superclass modifies it's stress 
    __HOST_DEVICE__ T getOverallCentroid() {
        // compute bending inertias about overall centroid so that B = 0 and avoid bending-axial coupling
        // uses weighted areas z1 to z2, where panel at z1=0 and stiffener at z2= -h/2 - hs/2
        T panel_area = thick; // is area per unit width
        T stiff_area = stiffHeight * stiffThick / stiffPitch; // per unit width
        // T z_panel = 0.0; the panel center (original)
        T z_stiff = thick/2.0 + stiffHeight/2.0; // stiffener above panel (doesn't really matter, cause gets squared later)
        return z_stiff * stiff_area / (panel_area + stiff_area);
    }
    template <int DV>
    __HOST_DEVICE__ T getOverallCentroidSens() {
        T panel_area = thick; // is area per unit width
        T stiff_area = stiffHeight * stiffThick / stiffPitch; // per unit width
        T z_stiff = thick/2.0 + stiffHeight/2.0; // stiffener above panel (doesn't really matter, cause gets squared later)
        T total_area = panel_area + stiff_area;

        if constexpr (DV == 0) {
            // panel thick deriv
            return 0.5 * stiff_area / total_area - z_stiff * stiff_area / total_area / total_area;
        } else if (DV == 1) {
            // stiffener height deriv
            T dstiff_area = stiff_area / stiffHeight;
            return (0.5 * stiff_area + z_stiff * dstiff_area) / total_area - z_stiff * stiff_area / total_area / total_area * dstiff_area;
        } else if (DV == 2) {
            // stiffener thick
            T dstiff_area = stiff_area / stiffThick;
            return z_stiff * dstiff_area / total_area - z_stiff * stiff_area / total_area / total_area * dstiff_area;
        } else if (DV == 3) {
            // stiffener pitch
            T dstiff_area = stiff_area * -1.0 / stiffPitch;
            return z_stiff * dstiff_area / total_area - z_stiff * stiff_area / total_area / total_area * dstiff_area;
        }
    }
    __HOST_DEVICE__ T getPanelIzz() override {
        // use modified overall centroid here, int z^2 * dz = int (z+zc)^2 dz = int z^2 dz + zc^2 * (int dz) Parallel axis thm
        T zc = getOverallCentroid();
        return thick * thick * thick / 12.0 + thick * zc * zc;
    }

    template <int DV>
    __HOST_DEVICE__ T getPanelIzzSens() {
        // get Izz panel thick sens here to unstiffened panel subclass
        T dzc = getOverallCentroidSens<DV>(); // get pthick sens 
	T zc_bar = 2.0 * thick * zc * dzc;
        if constexpr (DV == 0) {
		return thick * thick / 4.0 + zc * zc + zc_bar;
	} else {
		return zc_bar;
	}
    }
    __HOST_DEVICE__ T getStiffenerArea() { return stiffHeight * stiffThick; }
    __HOST_DEVICE__ T getStiffenerI11() { 
        // TODO : add flange fraction
	T hs = stiffHeight, ts = stiffThick, sp = stiffPitch, zc = getOverallCentroid();
	return hs * ts / sp * (hs * hs / 12.0 + zc * zc);
    }
    template <int DV>
    __HOST_DEVICE__ T getStiffenerI11Sens() {
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
        evalTangentStiffness2D(E, nu, C);
        // dPi/dE = stress[9] vector, here we compute dstress[9] the thickness derivs
        T dstress[9];
        // Nij = A * Eij, thick deriv is dNij = C * Eij
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, strain, dstress);
        T dI = getPanelIzzSens<0>(); // dI/dpanel_thick
        A2D::SymMatVecCoreScale3x3<T, true>(dI, C, &strain[3], &dstress[3]);
        // compute transverse shear components
        T dAs = getTransShearCorrFactor() * C[5];
        T ddrill = getDrillingRegularization() * dAs;
        dstress[6] = dAs * strain[6];
        dstress[7] = dAs * strain[7];
        dstress[8] = ddrill * strain[8];
        // now compute <dstress, psi_strain> as adjoint resid product, only one dv
        loc_dv_sens[0] = scale * A2D::VecDotCore<T, 9>(psi_strain, dstress);

        /* 2) panel stress- other derivatives (only through bending or M = D * k overall centroid) */
        T dI1 = getPanelIzzSens<1>(); // dI/dstiffHeight
        // false for not additive
        A2D::SymMatVecCoreScale3x3<T, false>(dI1, C, &strain[3], &dstress[3]);
        // now only take dot product through bending stress + strain
        loc_dv_sens[1] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);
        // then repeat for derivs 2 and 3
        T dI2 = getPanelIzzSens<2>(); // dI/dstiffThick
        A2D::SymMatVecCoreScale3x3<T, false>(dI2, C, &strain[3], &dstress[3]);
        loc_dv_sens[2] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);
        T dI3 = getPanelIzzSens<3>(); // dI/dstiffPitch
        A2D::SymMatVecCoreScale3x3<T, false>(dI3, C, &strain[3], &dstress[3]);
        loc_dv_sens[3] = scale * A2D::VecDotCore<T, 3>(&psi_strain[3], &dstress[3]);

        /* 3) take derivatives now through smeared stiffener stress (all four derivs) */
        // 3.1) first derivs through s11 += EA/sp * e11, A11 smeared stiffness
        T A11_stiff = data.getStiffenerArea() / data.stiffPitch;
        T A11_energy = scale * A11_stiff * psi_strain[0] * strain[0];
        loc_dv_sens[1] += A11_energy / stiffHeight;
        loc_dv_sens[2] += A11_energy / stiffThick;
        loc_dv_sens[3] += A11_energy * -1.0 / stiffPitch;

        // 3.2) take derivatives through trv shear stiffnesses
        T A44_stiff = A11_stiff * Data::getTransShearCorrFactor();
        T A44_energy = scale * A44_stiff * (psi_strain[6] * strain[6] + psi_strain[7] * strain[7]);
        loc_dv_sens[1] += A44_energy / stiffHeight;
        loc_dv_sens[2] += A44_energy / stiffThick;
        loc_dv_sens[3] += A44_energy * -1.0 / stiffPitch;

	// 3.3) take derivatives through stiffener bend stiffness D11
	// stiffener I11 = E * I / sp (so already sp norm or technically I11 per unit width)
	T D11_energy = scale * psi_strain[3] * strain[3];
	loc_dv_sens[0] += D11_energy * data.getStiffenerI11Sens<0>();
        loc_dv_sens[1] += D11_energy * data.getStiffenerI11Sens<1>();
        loc_dv_sens[2] += D11_energy * data.getStiffenerI11Sens<2>();
        loc_dv_sens[3] += D11_energy * data.getStiffenerI11Sens<3>();
    }

    __HOST_DEVICE__ T evalFailure(const T &rhoKS, const T &safetyFactor, const T e[9]) const {
        // von Mises failure index, use ks max for to pand bottom stresses
        T fails[3]; // 1) panel strength, 2) local buckling, 3) global buckling

	/* 1) panel strength failure */
	// smeared stiffener doesn't affect top+bottom panel stress (only function of panel props)
        //    so unchanged from unstiffened panel subclass
	T panel_strength_fail = ShellIsotropicData<T, ref_axis>::evalFailure(rhoKS, safetyFactor, e);
        fails[0] = panel_strength_fail;

        
        /* 2 and 3 pre) compute in-plane loads */
	T A11 = E * thick / (1 - nu * nu); 
        T A66 = A11 * (1 - nu) / 2.0;
	A11 += E * stiffHeight * stiffThick / stiffPitch; // add smeared stiffener stiffness
	// e[0] = e11, e[3] = e22, e[1] = e12
        T N11 = A11 * (e[0] + nu * e[3]), N12 = A66 * e[1];
        
        /* 2) local buckling failure */
	const T pi = 3.14159265358979323846;
        T N11_hat = N11 / pi / pi, N12_hat = N12 / pi / pi;
	T sp2 = stiffPitch * stiffPitch, D_loc = E * getPanelIzz();
        T loc_axial = -N11_hat * sp2 / D_loc / 4.0;
	T loc_shear = N12_hat * sp2 / D_loc / 5.374;
	fails[1] = loc_axial + loc_shear * loc_shear;
	fails[1] *= safetyFactor;

	/* 3) global buckling failure */
	T a2 = panelLength * panelLength;
	T D_axial = D_loc + E * getStiffenerIzz();
	T D_shear = D_loc + 0.5 / 4.7 * E * getStiffenerIzz();
	T glob_axial = -N11_hat * a2 / D_axial / 1.0; // no 2+2*xi (rho0^{-2} term dominates
        T glob_shear = N12_hat * a2 / D_shear / 4.7; // less stiffening effect (but still some)
	fails[2] = glob_axial + glob_shear * glob_shear;
        fails[2] *= safetyFactor;

        /* 4) compute combined failure criterion among all three fail modes */
	T max12 = (fails[0] > fails[1]) ? fails[0] : fails[1];
        T max = (max12 > fails[2]) ? max12 : fails[2];
        T ks_sum = exp(rhoKS * (fails[0] - max)) + exp(rhoKS * (fails[1] - max)) + exp(rhoKS * (fails[2] - max));
        T ks_fail = log(ks_sum) / rhoKS;
        return ks_fail;	
    }

    __HOST_DEVICE__ void evalFailureDVSens(const T &rhoKS, const T &safetyFactor, const T e[9],
                                           const T &scale, T dv_sens[]) const {
        /* compute dsigma_KS/dthick */

        // panel strength failure only depends on panel props I think
	//    smeared stiff just reduced the strains, so can call unstiff panel superclass
	ShellIsotropicData<T, ref_axis>::evalFailureDVSens(rhoKS, safetyFactor, e, scale, dv_sens);

	// TODO : add panel buckling constraint + then overall failure
    }

    __HOST_DEVICE__ void evalFailureStrainSens(const T &scale, const T &rhoKS, const T &safetyFactor, 
		                               const T e[9], T er[9]) const {
        /* compute dsigma_KS/dstrain */
      
        // panel strength only depends on panel props, so can call unstiff panel method here
	//   smearing only goes through strain sens
	ShellIsotropicData<T, ref_axis>::evalFailureStrainSens(scale, rhoKS, safetyFactor, e, er);

	// TODO : add panel buckling constraint + then overall failure
    }


    // private:
    T stiffHeight, stiffThick, stiffPitch;
    T panelLength;
    // T panelWidth;
};
