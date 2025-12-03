#pragma once
#include "../../../cuda_utils.h"

template <typename T, class Basis>
__DEVICE__ void computeHR_tyingStrain(const T pt[], const T hr_vars[], const T XdinvT[],
                                      A2D::SymMat<T, 3>& e0ty) {
    T us_xi[5], us_eta[5];
    Basis::template interpFieldsGrad<5, 5>(pt, hr_vars, us_xi, us_eta);
    // strain-gap disp are [v11, v12, v22, v13, v23]
    // e0ty strain order is [e11, e12, e13, e22, e23] (slightly different order, e13 and e22 swap)

    e0ty[0] = us_xi[0] * XdinvT[0] + us_eta[0] * XdinvT[3];  // e11 membrane
    e0ty[3] = us_xi[2] * XdinvT[1] + us_eta[2] * XdinvT[4];  // e22 membrane
    e0ty[2] = us_xi[3] * XdinvT[0] + us_eta[3] * XdinvT[3];  // gam13 transverse shear
    e0ty[4] = us_xi[4] * XdinvT[1] + us_eta[4] * XdinvT[4];  // gam23 transverse shear
    e0ty[5] = 0.0;

    if constexpr (Basis::order > 1) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
            printf("FATAL ERROR : HR element only supports 1st order basis\n");
        }
        __trap();
    }

    T d2mixed[5];
    Basis::template interpFieldsMixedGrad<5, 5>(pt, hr_vars, d2mixed);
    e0ty[1] = (XdinvT[0] * XdinvT[4] + XdinvT[1] * XdinvT[3]) *
              d2mixed[1];  // e12 mixed derivative rotated to shell frame
}

template <typename T>
__DEVICE__ void negateHR_tyingStrain(A2D::SymMat<T, 3>& e0ty) {
    // negate HR tying strains for the -He term with strain-gap disps to itself in output
    e0ty[0] *= -1.0;
    e0ty[1] *= -1.0;
    e0ty[2] *= -1.0;
    e0ty[3] *= -1.0;
    e0ty[4] *= -1.0;
    e0ty[5] = 0.0;
}

template <typename T, class Basis>
__DEVICE__ void computeHR_tyingStrainTranspose(const T pt[], const A2D::SymMat<T, 3>& e0tyb,
                                               const T XdinvT[], T hr_res[]) {
    if constexpr (Basis::order > 1) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
            printf("FATAL ERROR : HR element only supports 1st order basis\n");
        }
        __trap();
    }

    // note e22 and e13 are swapped in spots 2,3 for this case
    T us_xi_b[5] = {0}, us_eta_b[5] = {0};
    us_xi_b[0] += e0tyb[0] * XdinvT[0];
    us_eta_b[0] += e0tyb[0] * XdinvT[3];  // e11 back to xi,eta
    us_xi_b[2] += e0tyb[3] * XdinvT[1];
    us_eta_b[2] += e0tyb[3] * XdinvT[4];  // e22
    us_xi_b[3] += e0tyb[2] * XdinvT[0];
    us_eta_b[3] += e0tyb[2] * XdinvT[3];  // gam13
    us_xi_b[4] += e0tyb[4] * XdinvT[1];
    us_eta_b[4] += e0tyb[4] * XdinvT[4];  // gam23
    us_eta_b[1] +=
        e0tyb[1] * (XdinvT[0] * XdinvT[4] + XdinvT[1] * XdinvT[3]);  // e12 mixed derivative

    Basis::template interpFieldsGradTranspose<5, 5>(pt, us_xi_b, us_eta_b, hr_res);
    // zero other entries (so only us_eta_b[1] NZ for this part)
    us_eta_b[0] = 0.0, us_eta_b[2] = 0.0, us_eta_b[3] = 0.0, us_eta_b[4] = 0.0;
    Basis::template interpFieldsMixedGradTranspose<5, 5>(pt, us_eta_b,
                                                         hr_res);  // push back e12 contribution
    // temp debug: add small neg-diag entry into e12 part?
    // hr_res[1] -= 1e-2;
}

template <typename T>
__HOST_DEVICE__ void blockPermuteHRVarsPerNode(const T* full_vars, T* hr_vars, T* vars,
                                               int num_nodes, int hr_dof = 5, int std_dof = 6) {
    // HR DOFs: per-node block
    for (int n = 0; n < num_nodes; n++) {
        for (int f = 0; f < hr_dof; f++) {
            hr_vars[n * hr_dof + f] = full_vars[n * (hr_dof + std_dof) + f];
        }
    }

    // standard 6 DOFs: per-node block
    for (int n = 0; n < num_nodes; n++) {
        for (int f = 0; f < std_dof; f++) {
            vars[n * std_dof + f] = full_vars[n * (hr_dof + std_dof) + hr_dof + f];
        }
    }
}

// reverse permutation: put back into interleaved
template <typename T>
__HOST_DEVICE__ void blockAddUnpermuteHRVarsPerNode(const T* hr_res, const T* res, T* full_res,
                                                    int num_nodes, int hr_dof = 5,
                                                    int std_dof = 6) {
    for (int n = 0; n < num_nodes; n++) {
        for (int f = 0; f < hr_dof; f++)
            full_res[n * (hr_dof + std_dof) + f] += hr_res[n * hr_dof + f];
        for (int f = 0; f < std_dof; f++)
            full_res[n * (hr_dof + std_dof) + hr_dof + f] += res[n * std_dof + f];
    }
}
