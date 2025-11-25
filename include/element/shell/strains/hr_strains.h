#pragma once
#include "../../../cuda_utils.h"

template <typename T, class Basis>
__DEVICE__ void computeHR_tyingStrain(const T pt[], const T hr_vars[], 
    const T XdinvT[], A2D::SymMat<T, 3> &e0ty) {
    // computes all 5 tying strains for Hellinger-Reissner (the problematic strains)
    T us_xi[5], ux_eta[5];
    Basis::template interpFieldsGrad<5, 5>(hr_vars, us_xi, us_eta);

    // convert comp coords to shell frame derivs (1,2) with XdinvT
    // all but the e12 = e0ty[1] strain in now
    // 1) g11, or e11 membrane strain
    e0ty[0] = us_xi[0] * XdinvT[0] + us_eta[0] * XdinvT[1];
    // 2) g22, or e22 membrane strain
    e0ty[1] = us_eta[2] * XdinvT[3] + us_eta[2] * XdinvT[4];
    // 3) g13, or gam13 trv shear strain
    e0ty[2] = us_xi[3] * XdinvT[0] + us_xi[3] * XdinvT[1];
    // 4) g23 or gam23 trv shear strain
    e0ty[4] = us_eta[4] * XdinvT[3] + us_eta[2] * XdinvT[4];

    // TODO : get mixed derivative of 2nd entry
    // including frame rotations?
    T us12_12 = ?;
}