/* kernel functions now.. */

template <typename T, class Basis>
__global__ static void k_unstruct_prolongate(const T *coarse_soln, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *fine_soln, T *fine_weights) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln

    // loop over nearby coarse elements and their nodes..
    int n2e_start = node2elem_ptr[fine_node], n2e_end = node2elem_ptr[fine_node + 1];
    int ne = n2e_end - n2e_start;
    
    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get local coarse elem disps.. (with permutations here?)
        T c_elem_disps[24];
        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            c_elem_disps[i] = coarse_soln[6 * perm_coarse_node + loc_dof];
        }

        // interp the disps from coarse disps
        T fine_disp_add[6];
        memset(fine_disp_add, 0.0, 6 * sizeof(T));
        Basis::template interpFields<6, 6>(pt, c_elem_disps, fine_disp_add);

        // if (fine_node == 0) {
        //     printf("ielem_c = %d\n", ielem_c);
        //     printf("c_elem_nodes");
        //     printVec<int>(4, c_elem_nodes);
        //     printf("c_elem_disps:");
        // }

        // now add into the fine solution and fine weights
        for (int idof = 0; idof < 6; idof++) {
            atomicAdd(&fine_soln[6 * perm_fine_node + idof], fine_disp_add[idof]);
            atomicAdd(&fine_weights[6 * perm_fine_node + idof], 1.0);
        }
    }
}

template <typename T, class Basis>
__global__ static void k_unstruct_restrict(const T *fine_defect_in, const int *d_coarse_iperm, const int *coarse_elem_conn, const int *node2elem_ptr, const int *node2elem_elems, 
    const T *node2elem_xis, const int nnodes_fine, const int *d_fine_iperm, T *coarse_soln_out, T *coarse_weights) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int fine_node = tid;
    if (fine_node >= nnodes_fine) return;

    int perm_fine_node = d_fine_iperm[fine_node]; // for writing into perm fine soln

    // loop over nearby coarse elements and their nodes..
    int n2e_start = node2elem_ptr[fine_node], n2e_end = node2elem_ptr[fine_node + 1];
    int ne = n2e_end - n2e_start;
    
    for (int jp = node2elem_ptr[fine_node]; jp < node2elem_ptr[fine_node + 1]; jp++) {
        int ielem_c = node2elem_elems[jp];
        const int *c_elem_nodes = &coarse_elem_conn[4 * ielem_c];

        // get comp coords for interp of coarse-fine
        T pt[2];
        pt[0] = node2elem_xis[2 * jp];
        pt[1] = node2elem_xis[2 * jp + 1];

        // get fine defect disps..
        T *fine_nodal_defect = &fine_defect_in[6 * perm_fine_node];

        // now do interpFieldsTranspose to coarse defect on each node in element
        T coarse_elem_defect[24];
        Basis::template interpFieldsTranspose<6, 6>(pt, fine_nodal_defect, coarse_elem_defect);

        for (int i = 0; i < 24; i++) {
            int loc_node = i / 6, loc_dof = i % 6;
            int coarse_node = c_elem_nodes[loc_node];
            int perm_coarse_node = d_coarse_iperm[coarse_node];

            atomicAdd(&coarse_soln[6 * perm_coarse_node + idof], coarse_elem_defect[i]);
            atomicAdd(&coarse_weights[6 * perm_coarse_node + idof], 0.0);
        }
    }
}

// template <typename T>
// __global__ static void k_vec_normalize(int N, T *vec_in, T *weights) {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     if (tid < N) {
//         vec_in[tid] /= (weights[tid] + 1e-12);
//     }
// }