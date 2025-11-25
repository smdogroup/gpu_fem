// helper GPU kernels for geometry-specific prolongation operators
// NOTE : I will write more general one when it comes to wingbox later..
#pragma once

enum ProlongationGeom : int {
    PLATE,
    CYLINDER
};

template <typename T, class Basis, ProlongationGeom geom>
__global__ static void k_plate_prolongate(const int order, const int vars_per_node, const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *coarse_soln_in, T *dx_fine, T *d_fine_wts) {
    // prolongation for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid; // int odd_ielem = _ielem % 2 == 1; // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int fine_elem = tid;
    if (fine_elem >= nelems_fine) return;
    // TODO : on other meshes, use elem conn and some graph maps (not plate method here which is simpler)

    // write it first, then we'll add some device helper methods?
    int nx_c = order * nxe_coarse + 1, nx_f = order * nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    // int coarse_elem = nxe_coarse * iye_c + ixe_c;
    int nx = order + 1;
    int nx2 = nx * nx; // num nodes in each element

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
            int coarse_node = nx_c * iy_c + ix_c;
            int perm_coarse_node = d_coarse_iperm[coarse_node];
            int fine_node = iy_f * nx_f + ix_f;
            int perm_fine_node = d_fine_iperm[fine_node];

            // now loop over each DOF..
            for (int idof = 0; idof < vars_per_node; idof++) {
                int coarse_dof = vars_per_node * perm_coarse_node + idof;
                int fine_dof = vars_per_node * perm_fine_node + idof;
                T val = coarse_soln_in[coarse_dof];
                val *= scale;

                atomicAdd(&dx_fine[fine_dof], val);
                atomicAdd(&d_fine_wts[fine_dof], scale);
            }
        }
    }

}

template <typename T, class Basis, ProlongationGeom geom>
__global__ static void k_plate_restrict(const int order, const int vars_per_node, const int nxe_coarse, const int nxe_fine, 
    const int nelems_fine, const int *d_coarse_iperm, const int *d_fine_iperm,
    const T *defect_fine_in, T *defect_coarse_out, T *d_coarse_wts) {
    // restriction for linear cquad4 shells in plate geometry (corrects for arbitrary reordering here..)

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // TODO : could reorder elems to reduce bank conflicts later (NVIDIA cuda profiling)
    // int _ielem = tid; // int odd_ielem = _ielem % 2 == 1; // int ielem = _ielem / 2 + nelems_fine / 2 * odd_ielem;
    int fine_elem = tid;
    if (fine_elem >= nelems_fine) return;
    // TODO : on other meshes, use elem conn and some graph maps (not plate method here which is simpler)
    
    // write it first, then we'll add some device helper methods?
    int nx_c = order * nxe_coarse + 1, nx_f = order * nxe_fine + 1;
    int ixe_f = fine_elem % nxe_fine, iye_f = fine_elem / nxe_fine;
    int ixe_c = ixe_f / 2, iye_c = iye_f / 2;
    // int coarse_elem = nxe_coarse * iye_c + ixe_c;
    int nx = order + 1;
    int nx2 = nx * nx;

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
            int coarse_node = nx_c * iy_c + ix_c;
            int perm_coarse_node = d_coarse_iperm[coarse_node];
            int fine_node = iy_f * nx_f + ix_f;
            int perm_fine_node = d_fine_iperm[fine_node];

            // now loop over each DOF..
            for (int idof = 0; idof < vars_per_node; idof++) {
                int coarse_dof = vars_per_node * perm_coarse_node + idof;
                int fine_dof = vars_per_node * perm_fine_node + idof;
                T val = defect_fine_in[fine_dof];
                val *= scale;

                atomicAdd(&defect_coarse_out[coarse_dof], val);
                atomicAdd(&d_coarse_wts[coarse_dof], scale);
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

/* helper device functions */
// template <typename T>
// __device__ void d_get_
