#pragma once
#include "utils.h"

template <typename T>
class DKTElement {
    // triangle plate bending element from https://web.mit.edu/kjb/www/Publications_Prior_to_1998/A_Study_of_Three-Node_Triangular_Plate_Bending_Elements.pdf
  public:

    /* helper device functions */
    __device__ static void d_get_triang_basis(const T xi, const T eta, T shape_funcs[6]) {
        shape_funcs[0] = 2.0 * (1 - xi - eta) * (0.5 - xi - eta);
        shape_funcs[1] = xi * (2 * xi - 1.0);
        shape_funcs[2] = eta * (2.0 * eta - 1.0);
        shape_funcs[3] = 4.0 * xi * eta;
        shape_funcs[4] = 4.0 * eta * (1.0 - xi - eta);
        shape_funcs[5] = 4.0 * xi * (1.0 - xi - eta);
    }

    __device__ static void d_get_xpts_diffs(const T xpts[6], T dx[3], T dy[3]) {
        dx[0] = xpts[2] - xpts[4]; // x1 - x2
        dx[1] = xpts[4] - xpts[0]; // x2 - x0
        dx[2] = xpts[0] - xpts[2]; // x0 - x1

        dy[0] = xpts[3] - xpts[5]; // y1 - y2
        dy[1] = xpts[5] - xpts[1]; // y2 - y0
        dy[2] = xpts[1] - xpts[3]; // y0 - y1
    }

    __device__ static void d_get_w_shape_funcs(const T xpts[], const T xi, const T eta, T Hw[]) {
        // get Hw(xi,eta) shape funcs using xpts basis (for prolongation from john fish paper)

        // compute xij aka dx, yij (x and y nodal differences)
        T dx[3], dy[3];
        d_get_xpts_diffs(xpts, dx, dy);

        Hw[0] = 1 - xi - eta;
        Hw[1] = 0.5 * (1 - xi - eta) * (-dy[2] * xi + dy[1] * eta);
        Hw[2] = 0.5 * (1 - xi - eta) * (dx[2] * xi - dx[1] * eta);
        Hw[3] = xi;
        Hw[4] = 0.5 * xi * (-dy[0] * eta + dy[2] * (1 - xi - eta));
        Hw[5] = 0.5 * xi * (dx[0] * eta - dx[2] * (1 - xi - eta));
        Hw[6] = eta;
        Hw[7] = 0.5 * eta * (-dy[1] * (1 - xi - eta) + dy[0] * xi);
        Hw[8] = 0.5 * eta * (dx[1] * (1 - xi - eta) - dx[0] * xi);
    }

    __device__ static void d_get_H_shape_funcs(const T xpts[], const T xi, const T eta, T Hx[], T Hy[]) {
        // compute xij aka dx, yij (x and y nodal differences)
        T dx[3], dy[3];
        d_get_xpts_diffs(xpts, dx, dy);

        // get the triangle basis
        T N[6];
        d_get_triang_basis(xi, eta, N);


        // make three calls to the helper function.. so that way we don't do if statement style like in python..
        // this is explicit..
        int i = 4, j = 5, m = 0; // minus 1 shifted from python version
        _d_get_H_shape_funcs_inner(dx, dy, i, j, m, N, &Hx[0], &Hy[0]);
        i = 5, j = 3, m = 1;
        _d_get_H_shape_funcs_inner(dx, dy, i, j, m, N, &Hx[3], &Hy[3]);
        i = 3, j = 4, m = 2;
        _d_get_H_shape_funcs_inner(dx, dy, i, j, m, N, &Hx[6], &Hy[6]);
    }

    __device__ static void _d_get_H_shape_funcs_inner(const T dx[], const T dy[], const int i, const int j, const int m, const T N[], T *Hxn, T *Hyn) {
        // helper function for each of the three Hx or Hy vals in d_get_H_shape_funcs method for single node..

        // i : a,b,c,d,e and L
        T dx_i = dx[i-4], dy_i = dy[i-4];
        T L_i = sqrt(dx_i * dx_i + dy_i * dy_i);
        T a_i = dx_i / L_i / L_i;
        T b_i = 0.75 * dx_i * dy_i / L_i / L_i;
        T c_i = (0.25 * dx_i * dx_i - 0.5 * dy_i * dy_i) / L_i / L_i;
        
        // j : a,b,c,d,e and L
        T dx_j = dx[j-4], dy_j = dy[j-4];
        T L_j = sqrt(dx_j * dx_j + dy_j * dy_j);
        T a_j = dx_j / L_j / L_j;
        T b_j = 0.75 * dx_j * dy_j / L_j / L_j;
        T c_j = (0.25 * dx_j * dx_j - 0.5 * dy_j * dy_j) / L_j / L_j;

        // here Hx, Hy are shifted to each of three values in inner loop
        Hxn[0] = 1.5 * (a_j * N[j] - a_i * N[i]);
        Hxn[1] = b_i * N[i] + b_j * N[j];
        Hxn[2] = N[m] - c_i * N[i] - c_j * N[j];

        T d_i = -dy_i / L_i / L_i;
        T e_i = (0.25 * dy_i * dy_i - 0.5 * dx_i * dx_i) / L_i / L_i;
        T d_j = -dy_j / L_j / L_j;
        T e_j = (0.25 * dy_j * dy_j - 0.5 * dx_j * dx_j) / L_j / L_j;

        Hyn[0] = 1.5 * (d_j * N[j] - d_i * N[i]);
        Hyn[1] = -N[m] + e_i * N[i] + e_j * N[j];
        Hyn[2] = -(b_i * N[i] + b_j * N[j]);
    }

    __device__ static void d_get_H_shape_func_grads(const T dx[3], const T dy[3], const T xi, const T eta, T Hx_xi[], T Hx_eta[], T Hy_xi[], T Hy_eta[]) {
        // compute shape func grads for strain-disp
        T L[3], P[3], q[3], t[3], r[3];

        for (int i = 0; i < 3; i++) {
            L[i] = sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
            T L2 = L[i] * L[i];
            P[i] = 6 * -dx[i] / L2;
            q[i] = 4 * 0.75 * dx[i] * dy[i] / L2;
            t[i] = 6 * -dy[i] / L2;
            r[i] = 3 * dy[i] * dy[i] / L2;
        }

        Hx_xi[0] = P[2] * (1 - 2 * xi) + (P[1] - P[2]) * eta;
        Hx_xi[1] = q[2] * (1 - 2*xi) - (q[1] + q[2]) * eta;
        Hx_xi[2] = -4 + 6 * (xi + eta) + r[2] * (1.0 - 2 * xi) - eta * (r[1] + r[2]);
        Hx_xi[3] = -P[2] * (1 - 2 * xi) + eta * (P[0] + P[2]);
        Hx_xi[4] = q[2] * (1 - 2 * xi) - eta * (q[2] - q[0]);
        Hx_xi[5] = -2 + 6 * xi + r[2] * (1 - 2 * xi) + eta * (r[0] - r[2]);
        Hx_xi[6] = -eta * (P[1] + P[0]);
        Hx_xi[7] = eta * (q[0] - q[1]);
        Hx_xi[8] = -eta * (r[1] - r[0]);

        Hy_xi[0] = t[2] * (1 - 2 * xi) + (t[1] - t[2]) * eta;
        Hy_xi[1] = 1.0 + r[2] * (1 - 2*xi) - (r[1] + r[2]) * eta;
        Hy_xi[2] = -q[2] * (1 - 2 * xi) + eta * (q[1] + q[2]);
        Hy_xi[3] = -t[2] * (1 - 2 * xi) + eta * (t[0] + t[2]);
        Hy_xi[4] = -1 + r[2] * (1-2*xi) + eta * (r[0] - r[2]);
        Hy_xi[5] = -q[2] * (1 - 2 * xi) - eta * (q[0] - q[2]);
        Hy_xi[6] = -eta * (t[0] + t[1]);
        Hy_xi[7] = eta * (r[0] - r[1]);
        Hy_xi[8] = -eta * (q[0] - q[1]);

        Hx_eta[0] = -P[1] * (1 - 2 * eta) - xi * (P[2] - P[1]);
        Hx_eta[1] = q[1] * (1 - 2 * eta) - xi * (q[1] + q[2]);
        Hx_eta[2] = -4 + 6 * (xi + eta) + r[1] * (1 - 2 * eta) - xi * (r[1] + r[2]);
        Hx_eta[3] = xi * (P[0] + P[2]);
        Hx_eta[4] = xi * (q[0] - q[2]);
        Hx_eta[5] = -xi * (r[2] - r[0]);
        Hx_eta[6] = P[1] * (1 - 2 * eta) - xi * (P[0] + P[1]);
        Hx_eta[7] = q[1] * (1 - 2 * eta) + xi * (q[0] - q[1]);
        Hx_eta[8] = -2 + 6 * eta + r[1] * (1 - 2 * eta) + xi * (r[0] - r[1]);

        Hy_eta[0] = -t[1] * (1 - 2 * eta) - xi * (t[2] - t[1]);
        Hy_eta[1] = 1 + r[1] * (1 - 2 * eta) - xi * (r[1] + r[2]);
        Hy_eta[2] = -q[1] * (1 - 2 * eta) + xi * (q[1] + q[2]);
        Hy_eta[3] = xi * (t[0] + t[2]);
        Hy_eta[4] = xi * (r[0] - r[2]);
        Hy_eta[5] = -xi * (q[0] - q[2]);
        Hy_eta[6] = t[1] * (1 - 2 * eta) - xi * (t[0] + t[1]);
        Hy_eta[7] = -1 + r[1] * (1 - 2 * eta) + xi * (r[0] - r[1]);
        Hy_eta[8] = -q[1] * (1 - 2 * eta) - xi * (q[0] - q[1]);
    }

    __device__ static T d_get_element_area(const T dx[3], const T dy[3]) {
        // compute area
        return 0.5 * (dx[1] * dx[2] - dx[2] * dy[1]);
    }

    __device__ static void d_apply_strain_mat(const T xpts[], const T xi, const T eta, const T elem_disps[9], T strains[3]) {
        /* compute strains = B @ elem_disps */
        memset(strains, 0.0, 3 * sizeof(T));

        // get dx, dy nodal x and y coord diffs
        T dx[3], dy[3];
        d_get_xpts_diffs(xpts, dx, dy);

        // get element area
        T area = d_get_element_area(dx, dy);

        // get shape funcs grads
        T Hx_xi[9], Hx_eta[9], Hy_xi[9], Hy_eta[9];
        d_get_H_shape_func_grads(dx, dy, xi, eta, Hx_xi, Hx_eta, Hy_xi, Hy_eta);

        // DEBUG
        // int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // if (tid == 0) {
        //     printf("thread %d : in apply strain mat\n", tid);
        //     printf("dx[3]: ");
        //     printVec<T>(3, dx);
        //     printf("dy[3]: ");
        //     printVec<T>(3, dy);
        //     printf("area %.2e\n", area);
        //     printf("elem_disps[9]: ");
        //     printVec<T>(9, elem_disps);
        //     printf("Hx_xi[9]: ");
        //     printVec<T>(9, Hx_xi);
        // }
        
        // apply them to compute the strains
        for (int i = 0; i < 9; i++) {
            strains[0] += 0.5 / area * (dy[1] * Hx_xi[i] + dy[2] * Hx_eta[i]) * elem_disps[i];
            strains[1] += 0.5 / area * (-dx[1] * Hy_xi[i] -dx[2] * Hy_eta[i]) * elem_disps[i];
            strains[2] += 0.5 / area * (-dx[1] * Hx_xi[i] - dx[2] * Hx_eta[i] + dy[1] * Hy_xi[i] + dy[2] * Hy_eta[i]) * elem_disps[i];
        }
    }

    __device__ static  void d_get_quadpt_stresses(const T E, const T thick, const T nu, const T strains[], T stress[3]) {
        // compute strains => stresses using stress = Db @ strain (from material relation / Hooke's law in bending)
        T D = E * thick * thick * thick / 12.0 / (1.0 - nu * nu); // flexural modulus
        stress[0] = D * (strains[0] + nu * strains[1]);
        stress[1] = D * (strains[1] + nu * strains[0]);
        stress[2] = D * 0.5 * (1.0 - nu) * strains[2];
    }

    __device__ static void d_apply_strain_mat_transpose(const T xpts[], const T xi, const T eta, const T dstrains[3], T delem_disps[9]) {
        /* compute delem_disps = B^T @ dstrains */

        // get dx, dy nodal x and y coord diffs
        T dx[3], dy[3];
        d_get_xpts_diffs(xpts, dx, dy);

        // get element area
        T area = d_get_element_area(dx, dy);

        // get shape funcs grads
        T Hx_xi[9], Hx_eta[9], Hy_xi[9], Hy_eta[9];
        d_get_H_shape_func_grads(dx, dy, xi, eta, Hx_xi, Hx_eta, Hy_xi, Hy_eta);
        
        // apply them to compute the strains
        for (int i = 0; i < 9; i++) {
            // first one sets val, next ones add to it..
            delem_disps[i] = 0.5 / area * (dy[1] * Hx_xi[i] + dy[2] * Hx_eta[i]) * dstrains[0];
            delem_disps[i] += 0.5 / area * (-dx[1] * Hy_xi[i] -dx[2] * Hy_eta[i]) * dstrains[1];
            delem_disps[i] += 0.5 / area * (-dx[1] * Hx_xi[i] - dx[2] * Hx_eta[i] + dy[1] * Hy_xi[i] + dy[2] * Hy_eta[i]) * dstrains[2];
        }
    }

    __device__ static void get_quadpt_kelem_col(const int i_column, const T xpts[], const T xi, const T eta,  const T weight,
            const T E, const T thick, const T nu, T Kelem_col[]) {
        // compute single column of kelem..

        // set input elem_disps as p[elem_disps] = e_i for each col i (Kelem is 9x9 so 9 columns)
        T p_disp[9];
        memset(p_disp, 0.0, 9 * sizeof(T));
        p_disp[i_column] = 1.0;

        // DEBUG
        // int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // if (tid == 0) {
        //     printf("thread %d : xi %.2e, eta %.2e, weight %.2e, E %.2e, thick %.2e, nu %.2e, and xpts\n", tid, xi, eta, weight, E, thick, nu);
        //     printVec<T>(6, xpts);
        // }

        // strain-disp relation, B * p_disp => p_strains
        T p_strains[3];
        d_apply_strain_mat(xpts, xi, eta, p_disp, p_strains);

        // // DEBUG
        // // int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // if (tid == 0) {
        //     printf("p_strains on thread %d: %.2e, %.2e, %.2e\n", tid, p_strains[0], p_strains[1], p_strains[2]);
        // }

        // strains => stresses, where stresses equiv to d_strains
        T d_strains[3];
        d_get_quadpt_stresses(E, thick, nu, p_strains, d_strains);

        // stresses = d_strains => d_disps (in projection of e_i disps col)
        d_apply_strain_mat_transpose(xpts, xi, eta, d_strains, Kelem_col);

        // scale Kelem_col by weight
        for (int i = 0; i < 9; i++) {
            Kelem_col[i] *= weight;
        }
    }
};

