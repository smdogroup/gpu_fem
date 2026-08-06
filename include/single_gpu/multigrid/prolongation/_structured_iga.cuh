// helper GPU kernels for geometry-specific prolongation operators
// NOTE : I will write more general one when it comes to wingbox later..
#pragma once

template <typename T, class Basis>
__global__ static void k_plate_iga_prolongate(const int order, const int vars_per_node, const int nxe_coarse, const int nxe_fine, 
    int nnodes_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *coarse_soln_in, T *dx_fine, T *d_fine_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    // write it first, then we'll add some device helper methods?
    int nx_c = nxe_coarse + order, nx_f = nxe_fine + order;
    int perm_fine_node = d_fine_iperm[fine_node];
    int ix_f = fine_node % nx_f, iy_f = fine_node / nx_f;

    // four coarse nodes for every fine node (for 2nd order IGA element)
    //   a little different than 9 coarse for each fine in lagrange-lobatto
    for (int local_coarse = 0; local_coarse < 4; local_coarse++) {
        int ix_c = ix_f / 2 + local_coarse % 2;
        int iy_c = iy_f / 2 + local_coarse / 2;

        // compute x prolongation coefficient
        bool x_diag = (ix_f % 2) == (local_coarse % 2);
        T P_x = x_diag ? 0.75 : 0.25;
        // adjust for boundary conditions
        P_x += ((ix_f / 2) == 0) * ((local_coarse % 2 == 0) ? 0.25 : -0.25);
        P_x += ((ix_f / 2) == (nx_c - 2)) * ((local_coarse % 2 == 0) ? -0.25 : 0.25);

        // compute y prolongation coefficient
        bool y_diag = (iy_f % 2) == (local_coarse / 2);
        T P_y = y_diag ? 0.75 : 0.25;
        // adjust for boundary conditions
        P_y += ((iy_f / 2) == 0) * ((local_coarse / 2 == 0) ? 0.25 : -0.25);
        P_y += ((iy_f / 2) == (nx_c - 2)) * ((local_coarse / 2 == 0) ? -0.25 : 0.25);

        // compute full prolongation coefficient and perform on permuted vecs
        int coarse_node = nx_c * iy_c + ix_c;
        int perm_coarse_node = d_coarse_iperm[coarse_node];
        T scale = P_x * P_y; // kronecker product

        // printf("P_x = %.4e from ix_c %d to ix_f %d\n", P_x, ix_c, ix_f);
        // printf("P_y = %.4e from iy_c %d to iy_f %d\n", P_y, iy_c, iy_f);
        // printf("P_scale = %.4e for (f_node, c_node)=(%d,%d)\n", scale, fine_node, coarse_node);

        // now loop over each DOF
        for (int idof = 0; idof < vars_per_node; idof++) {
            int coarse_dof = vars_per_node * perm_coarse_node + idof;
            int fine_dof = vars_per_node * perm_fine_node + idof;
            T val = coarse_soln_in[coarse_dof];
            val *= scale;

            atomicAdd(&dx_fine[fine_dof], val);
            atomicAdd(&d_fine_wts[fine_dof], scale);
        }
    } // end of local coarse node loop
}

template <typename T, class Basis>
__global__ static void k_plate_iga_restrict(const int order, const int vars_per_node, const int nxe_coarse, const int nxe_fine, 
    int nnodes_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *defect_fine_in, T *defect_coarse_out, T *d_coarse_wts) {
    // restriction for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    // write it first, then we'll add some device helper methods?
    int nx_c = nxe_coarse + order, nx_f = nxe_fine + order;
    int perm_fine_node = d_fine_iperm[fine_node];
    int ix_f = fine_node % nx_f, iy_f = fine_node / nx_f;

    // four coarse nodes for every fine node (for 2nd order IGA element)
    //   a little different than 9 coarse for each fine in lagrange-lobatto
    for (int local_coarse = 0; local_coarse < 4; local_coarse++) {
        int ix_c = ix_f / 2 + local_coarse % 2;
        int iy_c = iy_f / 2 + local_coarse / 2;

        // compute x prolongation coefficient
        bool x_diag = (ix_f % 2) == (local_coarse % 2);
        T P_x = x_diag ? 0.75 : 0.25;
        // adjust for boundary conditions
        P_x += ((ix_f / 2) == 0) * ((local_coarse % 2 == 0) ? 0.25 : -0.25);
        P_x += ((ix_f / 2) == (nx_c - 2)) * ((local_coarse % 2 == 0) ? -0.25 : 0.25);

        // compute y prolongation coefficient
        bool y_diag = (iy_f % 2) == (local_coarse / 2);
        T P_y = y_diag ? 0.75 : 0.25;
        // adjust for boundary conditions
        P_y += ((iy_f / 2) == 0) * ((local_coarse / 2 == 0) ? 0.25 : -0.25);
        P_y += ((iy_f / 2) == (nx_c - 2)) * ((local_coarse / 2 == 0) ? -0.25 : 0.25);
        
        // compute full prolongation coefficient and perform on permuted vecs
        int coarse_node = nx_c * iy_c + ix_c;
        int perm_coarse_node = d_coarse_iperm[coarse_node];
        T scale = P_x * P_y; // kronecker product

        // now loop over each DOF
        for (int idof = 0; idof < vars_per_node; idof++) {
            int coarse_dof = vars_per_node * perm_coarse_node + idof;
            int fine_dof = vars_per_node * perm_fine_node + idof;
            T val = defect_fine_in[fine_dof];
            val *= scale;

            atomicAdd(&defect_coarse_out[coarse_dof], val);
            atomicAdd(&d_coarse_wts[coarse_dof], scale);
        }
    } // end of local coarse node loop
}

template <typename T>
__global__ static void k_vec_normalize(int N, T *vec_in, T *weights) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        vec_in[tid] /= (weights[tid] + 1e-12);
    }
}