#pragma once

enum ProlongationGeom : int {
    PLATE,
    CYLINDER
};

template <typename T, class Basis, ProlongationGeom geom>
__global__ static void k_structured_weights(
    const int start_fine_elem, const int start_crs_elem, 
    const int *loc_fine_elem_conn, const int *loc_crs_elem_conn,
    const int block_dim, const int nxe_coarse, const int nxe_fine, 
    const int loc_nelems_fine, T *fine_weights) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int loc_fine_elem = tid;
    int fine_elem = loc_fine_elem + start_fine_elem;
    if (loc_fine_elem >= loc_nelems_fine) return;

    const int order = Basis::order;
    const int nx = Basis::nx;
    int nx_c = order * nxe_coarse + 1, nx_f = order * nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    int nx2 = nx * nx; // num nodes in each element
    int crs_elem = nxe_coarse * iye_c + ixe_c;
    int loc_crs_elem = crs_elem - start_crs_elem;

    for (int local_fine = 0; local_fine < nx2; local_fine++) {
        int ix_f = order * ixe_f + local_fine % nx, iy_f = order * iye_f + local_fine / nx;      

        // compute the basis functions at the fine node
        // starting corner node
        int ix_f0 = order * ixe_c * 2, iy_f0 = order * iye_c * 2;
        T pt[2];
        pt[0] = -1.0 + 1.0 * (ix_f - ix_f0) / order;
        pt[1] = -1.0 + 1.0 * (iy_f - iy_f0) / order;
        T N[Basis::num_nodes];
        Basis::getBasis(pt, N);

        for (int local_coarse = 0; local_coarse < nx2; local_coarse++) {
            int ix_c = order * ixe_c + local_coarse % nx, iy_c = order * iye_c + local_coarse / nx;

            T scale = N[local_coarse];     
            if (geom == CYLINDER) {
                // loops back on itself in hoop direction
                iy_f = iy_f % nxe_fine;
                iy_c = iy_c % nxe_coarse;
            }

            // get fine and coarse indices now..
            // int coarse_node = nx_c * iy_c + ix_c;
            // int fine_node = iy_f * nx_f + ix_f;
            int fine_loc_node = loc_fine_elem_conn[nx2 * loc_fine_elem + local_fine];
            int crs_loc_node = loc_crs_elem_conn[nx2 * loc_crs_elem + local_coarse];

            // now loop over each DOF..
            for (int idof = 0; idof < block_dim; idof++) {
                int coarse_dof = block_dim * crs_loc_node + idof;
                int fine_dof = block_dim * fine_loc_node + idof;

                atomicAdd(&fine_weights[fine_dof], scale);
            }
        }
    }
}

template <typename T, class Basis, ProlongationGeom geom>
__global__ static void k_structured_prolongate(
    const int start_fine_elem, const int start_crs_elem, 
    const int *loc_fine_elem_conn, const int *loc_crs_elem_conn,
    const int block_dim, const int nxe_coarse, const int nxe_fine, 
    const int loc_nelems_fine, const T *fine_weights, const T *coarse_soln_in, T *dx_fine) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int loc_fine_elem = tid;
    int fine_elem = loc_fine_elem + start_fine_elem;
    if (loc_fine_elem >= loc_nelems_fine) return;

    const int order = Basis::order;
    const int nx = Basis::nx;
    int nx_c = order * nxe_coarse + 1, nx_f = order * nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    int nx2 = nx * nx; // num nodes in each element
    int crs_elem = nxe_coarse * iye_c + ixe_c;
    int loc_crs_elem = crs_elem - start_crs_elem;

    for (int local_fine = 0; local_fine < nx2; local_fine++) {
        int ix_f = order * ixe_f + local_fine % nx, iy_f = order * iye_f + local_fine / nx;      

        // compute the basis functions at the fine node
        // starting corner node
        int ix_f0 = order * ixe_c * 2, iy_f0 = order * iye_c * 2;
        T pt[2];
        pt[0] = -1.0 + 1.0 * (ix_f - ix_f0) / order;
        pt[1] = -1.0 + 1.0 * (iy_f - iy_f0) / order;
        T N[Basis::num_nodes];
        Basis::getBasis(pt, N);

        for (int local_coarse = 0; local_coarse < nx2; local_coarse++) {
            int ix_c = order * ixe_c + local_coarse % nx, iy_c = order * iye_c + local_coarse / nx;

            T scale = N[local_coarse];     
            if (geom == CYLINDER) {
                // loops back on itself in hoop direction
                iy_f = iy_f % nxe_fine;
                iy_c = iy_c % nxe_coarse;
            }

            // get fine and coarse indices now..
            // int coarse_node = nx_c * iy_c + ix_c;
            // int fine_node = iy_f * nx_f + ix_f;
            int fine_loc_node = loc_fine_elem_conn[nx2 * loc_fine_elem + local_fine];
            int crs_loc_node = loc_crs_elem_conn[nx2 * loc_crs_elem + local_coarse];

            // now loop over each DOF..
            for (int idof = 0; idof < block_dim; idof++) {
                int coarse_dof = block_dim * crs_loc_node + idof;
                int fine_dof = block_dim * fine_loc_node + idof;
                T val = coarse_soln_in[coarse_dof];
                val *= scale / (1e-12 + fine_weights[fine_dof]);

                atomicAdd(&dx_fine[fine_dof], val);
            }
        }
    }
}

template <typename T, class Basis, ProlongationGeom geom>
__global__ static void k_structured_restrict(
    const int start_fine_elem, const int start_crs_elem, 
    const int *loc_fine_elem_conn, const int *loc_crs_elem_conn,
    const int block_dim, const int nxe_coarse, const int nxe_fine, 
    const int loc_nelems_fine, const T *fine_weights, const T *defect_fine_in, T *defect_coarse_out) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int loc_fine_elem = tid;
    int fine_elem = loc_fine_elem + start_fine_elem;
    if (loc_fine_elem >= loc_nelems_fine) return;

    const int order = Basis::order;
    const int nx = Basis::nx;
    int nx_c = order * nxe_coarse + 1, nx_f = order * nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    int nx2 = nx * nx; // num nodes in each element
    int crs_elem = nxe_coarse * iye_c + ixe_c;
    int loc_crs_elem = crs_elem - start_crs_elem;

    for (int local_fine = 0; local_fine < nx2; local_fine++) {
        int ix_f = order * ixe_f + local_fine % nx, iy_f = order * iye_f + local_fine / nx;      

        // compute the basis functions at the fine node
        // starting corner node
        int ix_f0 = order * ixe_c * 2, iy_f0 = order * iye_c * 2;
        T pt[2];
        pt[0] = -1.0 + 1.0 * (ix_f - ix_f0) / order;
        pt[1] = -1.0 + 1.0 * (iy_f - iy_f0) / order;
        T N[Basis::num_nodes];
        Basis::getBasis(pt, N);

        for (int local_coarse = 0; local_coarse < nx2; local_coarse++) {
            int ix_c = order * ixe_c + local_coarse % nx, iy_c = order * iye_c + local_coarse / nx;

            T scale = N[local_coarse];     
            if (geom == CYLINDER) {
                // loops back on itself in hoop direction
                iy_f = iy_f % nxe_fine;
                iy_c = iy_c % nxe_coarse;
            }    

            // get fine and coarse indices now..
            // int coarse_node = nx_c * iy_c + ix_c;
            // int fine_node = iy_f * nx_f + ix_f;
            int fine_loc_node = loc_fine_elem_conn[nx2 * loc_fine_elem + local_fine];
            int crs_loc_node = loc_crs_elem_conn[nx2 * loc_crs_elem + local_coarse];

            // now loop over each DOF..
            for (int idof = 0; idof < block_dim; idof++) {
                int coarse_dof = block_dim * crs_loc_node + idof;
                int fine_dof = block_dim * fine_loc_node + idof;
                T val = defect_fine_in[fine_dof];
                val *= scale;
                // val *= scale / fine_weights[fine_dof];

                atomicAdd(&defect_coarse_out[coarse_dof], val);
            }
        }
    }
}

template <typename T>
__global__ static void k_vec_normalize(int N, T *vec_in, T *weights) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        vec_in[tid] /= (weights[tid] + 1e-12);
    }
}