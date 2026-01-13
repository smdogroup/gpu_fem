#pragma once

#include "a2dcore.h"
#include "helper.cuh"

// helper routines for unstruct prolongation demo on two grids

template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeCenterNormal(const T Xpts[], T fn[]) {
    // the nodal normal vectors are used for director methods
    // fn is 3*num_nodes each node 
    T pt[2] = {0.0, 0.0}; // center of element

    // compute the computational coord gradients of Xpts for xi, eta
    T dXdxi[3], dXdeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, Xpts, dXdxi, dXdeta);

    // compute the normal vector fn at each node
    T tmp[3];
    A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
    T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
    A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, fn);
}

template <typename T>
void get_elem_xpts(int ielem, const int *h_elem_conn, const T *h_xpts, T xpts_elem[12]) {
    // get coarse xpts_elem
    for (int iloc = 0; iloc < 4; iloc++) {
        int inode = h_elem_conn[4 * ielem + iloc];

        for (int idim = 0; idim < 3; idim++) {
            xpts_elem[3 * iloc + idim] = h_xpts[3 * inode + idim];
        }
    }
}


template <typename T, class Basis>
void get_comp_coords(const T coarse_xpts[], const T fine_xpt[], T xis[3], bool print = false) {
    // from coarse element, compute (xi,eta,zeta) triple of the fine node

    memset(xis, 0.0, 3 * sizeof(T));

    // get the shell node normal
    T fn[3];
    ShellComputeCenterNormal<T, Basis>(coarse_xpts, fn);

    // need to actually use the basis functions to get xi, eta..
    // can't just do planar calcs (cause still need to converge quadratic xyz(xi,eta) function even if elim zeta)
    for (int ct = 0; ct < 3; ct++) {

        T xyz[3], dxi[3], deta[3];

        Basis::template interpFields<3, 3>(xis, coarse_xpts, xyz);
        Basis::template interpFieldsGrad<3, 3>(xis, coarse_xpts, dxi, deta);
        // dzeta = fn the normal vec

        // printf("-----\nstart of comp coords iter %d\n", ct);
        // printf("xis: ");
        // printVec<T>(3, xis);
        // printf("xyz: ");
        // printVec<T>(3, xyz);
        // printf("dxi: ");
        // printVec<T>(3, dxi);
        // printf("deta: ");
        // printVec<T>(3, deta);

        // get error in interp xyz point
        T d_xyz[3];
        for (int idim = 0; idim < 3; idim++) d_xyz[idim] = xyz[idim] - fine_xpt[idim];

        // printf("d_xyz: ");
        // printVec<T>(3, d_xyz);

        // second order grad..
        T dxi_deta[3];
        memset(dxi_deta, 0.0, 3 * sizeof(T));
        for (int i = 0; i < 12; i++) {
            int inode = i / 3, idim = i % 3;
            // see xyz_dxi_deta func in gen_fc_map.py
            int sign = inode % 2 == 0 ? 1 : -1;
            int coeff = 0.25 * sign;

            dxi_deta[idim] += coarse_xpts[i] * coeff;
        }

        // printf("dxi_deta: ");
        // printVec<T>(3, dxi_deta);

        // compute the grad of xyz error objective
        T grad[3];
        grad[0] = 2.0 * A2D::VecDotCore<T,3>(d_xyz, dxi);
        grad[1] = 2.0 * A2D::VecDotCore<T,3>(d_xyz, deta);
        grad[2] = 2.0 * A2D::VecDotCore<T,3>(d_xyz, fn); // fn = dzeta

        // printf("grad: ");
        // printVec<T>(3, grad);

        // now compute hessian entries [only need 3 entries and (3,3) entry is just (dzeta,dzeta) = (fn, fn) = 1 ]
        T hess[3];
        hess[0] = 2.0 * A2D::VecDotCore<T,3>(dxi, dxi);
        hess[1] = 2.0 * A2D::VecDotCore<T,3>(dxi, deta) + 2.0 * A2D::VecDotCore<T,3>(d_xyz, dxi_deta);
        hess[2] = 2.0 * A2D::VecDotCore<T,3>(deta, deta);
        // hess[3] = 1 (not stored)

        // printf("hess: ");
        // printVec<T>(3, hess);

        // now solve xis += -hess^-1 * grad
        T discrim = hess[0] * hess[2] - hess[1] * hess[1];
        xis[0] -= (hess[2] * grad[0] - hess[1] * grad[1]) / discrim;
        xis[1] -= (hess[0] * grad[1] - hess[1] * grad[0]) / discrim;
        xis[2] -= grad[2];

        if (print) {
            // check resid for debug
            Basis::template interpFields<3, 3>(xis, coarse_xpts, xyz);
            for (int idim = 0; idim < 3; idim++) d_xyz[idim] = xyz[idim] - fine_xpt[idim];
            T resid = sqrt(A2D::VecDotCore<T, 3>(d_xyz, d_xyz));
            printf("ct = %d, |dxyz| = %.2e\n", ct, resid);
        }
    }

}   

template <typename T>
bool xis_in_elem(const T xis[3]) {
    T tol = 1e-3;
    T lb = -1.0 - tol, ub = 1.0 + tol;
    bool valid_xi = lb <= xis[0] && xis[0] <= ub;
    bool valid_eta = lb <= xis[1] && xis[1] <= ub;
    // int valid_zeta

    return valid_xi && valid_eta;
}