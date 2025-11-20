#pragma once
#include "coupled/locate_point.h"

// helper host functions for the structured prolongation
// TBD some of these should be moved onto the GPU

template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeCenterNormal(const T Xpts[], T fn[]) {
    // the nodal normal vectors are used for director methods
    // fn is 3*num_nodes each node
    T pt[2] = {0.0, 0.0};  // center of element

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
__HOST_DEVICE__ void get_elem_xpts(int ielem, const int *h_elem_conn, const T *h_xpts,
                                   T xpts_elem[12]) {
    // get coarse xpts_elem
    for (int iloc = 0; iloc < 4; iloc++) {
        int inode = h_elem_conn[4 * ielem + iloc];

        for (int idim = 0; idim < 3; idim++) {
            xpts_elem[3 * iloc + idim] = h_xpts[3 * inode + idim];
        }
    }
}

template <typename T>
__HOST_DEVICE__ void get_nodal_xpts(int n_local_nodes, int local_nodes[], const T *h_xpts,
                                    T local_xpts[]) {
    // get nearest neighbor coarse node xpts
    for (int iloc = 0; iloc < n_local_nodes; iloc++) {
        int inode = local_nodes[iloc];

        for (int idim = 0; idim < 3; idim++) {
            local_xpts[3 * iloc + idim] = h_xpts[3 * inode + idim];
        }
    }
}

// template <typename T, class Basis>
// __HOST_DEVICE__ T get_dzeta_estimate(int nn, const T coarse_elem_xpts[], const T
// coarse_nn_xpts[]) {
//     // compute element centroid (for estimate here)
//     T pt[2] = {0.0, 0.0};
//     T centroid[3];
//     Basis::template interpFields<3, 3>(pt, coarse_elem_xpts, centroid);

//     // get the shell node normal at centroid
//     T fn[3];
//     ShellComputeCenterNormal<T, Basis>(coarse_elem_xpts, fn);

//     // compute approx zeta or normal distance from element of each nn node
//     T zetas[nn];
//     for (int i = 0; i < nn; i++) {
//         const T *nn_xpt = &coarse_nn_xpts[3 * i];

//         T norm_comp = 0.0;
//         for (int j = 0; j < 3; j++) norm_comp += fn[j] * (nn_xpt[j] - centroid[j]);
//         zetas[i] = abs(norm_comp);
//     }

//     // now compute mean(absolute zeta) for estimate of dzeta among nearest neighbors
//     T mean_zeta = 0.0;
//     for (int i = 0; i < nn; i++) mean_zeta += zetas[i] / (1.0 * nn);
//     return mean_zeta;
// }

template <typename T, class Basis>
__HOST_DEVICE__ void get_comp_coords(const T coarse_xpts[], const T fine_xpt[], T xis[3],
                                     bool print = false, int n_iters = 8) {
    // from coarse element, compute (xi,eta,zeta) triple of the fine node

    memset(xis, 0.0, 3 * sizeof(T));

    // get the shell node normal
    T fn[3];
    ShellComputeCenterNormal<T, Basis>(coarse_xpts, fn);

    // need to actually use the basis functions to get xi, eta..
    // can't just do planar calcs (cause still need to converge quadratic xyz(xi,eta) function even
    // if elim zeta)
    // NOTE : can change the num iterations here if need be..
    for (int ct = 0; ct < n_iters; ct++) {  // 4 iterations is usually enough
        T xyz[3], dxi[3], deta[3];

        Basis::template interpFields<3, 3>(xis, coarse_xpts, xyz);
        for (int i = 0; i < 3; i++) xyz[i] += xis[2] * fn[i];

        Basis::template interpFieldsGrad<3, 3>(xis, coarse_xpts, dxi, deta);
        // dzeta = fn the normal vec

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
        grad[0] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, dxi);
        grad[1] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, deta);
        grad[2] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, fn);  // fn = dzeta

        // printf("grad: ");
        // printVec<T>(3, grad);

        // now compute hessian entries [only need 3 entries and (3,3) entry is just (dzeta,dzeta) =
        // (fn, fn) = 1 ]
        T hess[3];
        hess[0] = 2.0 * A2D::VecDotCore<T, 3>(dxi, dxi);
        hess[1] =
            2.0 * A2D::VecDotCore<T, 3>(dxi, deta) + 2.0 * A2D::VecDotCore<T, 3>(d_xyz, dxi_deta);
        hess[2] = 2.0 * A2D::VecDotCore<T, 3>(deta, deta);
        // hess[3] = 1 (not stored)

        // printf("hess: ");
        // printVec<T>(3, hess);

        // now solve xis += -hess^-1 * grad
        T discrim = hess[0] * hess[2] - hess[1] * hess[1];
        T alpha = 0.8;  // somewhat stabilizes a bit of oscillation..
        xis[0] -= alpha * (hess[2] * grad[0] - hess[1] * grad[1]) / discrim;
        xis[1] -= alpha * (hess[0] * grad[1] - hess[1] * grad[0]) / discrim;
        xis[2] -= alpha * grad[2];

        if (print) {
            // check resid for debug
            Basis::template interpFields<3, 3>(xis, coarse_xpts, xyz);
            for (int i = 0; i < 3; i++) xyz[i] += xis[2] * fn[i];
            for (int idim = 0; idim < 3; idim++) d_xyz[idim] = xyz[idim] - fine_xpt[idim];
            T resid = sqrt(A2D::VecDotCore<T, 3>(d_xyz, d_xyz));
            printf("ct = %d, |dxyz| = %.2e\n", ct, resid);
            printf("\td_xyz: ");
            printVec<T>(3, d_xyz);
        }
    }
}

template <typename T>
__HOST_DEVICE__ bool xis_in_elem(const T xis[3], const bool match_comp) {
    // bool xis_in_elem(const T xis[3], const T dzeta) {
    // T tol = 1e-3;
    T tol = 1e-2;  // don't want to be too strict here (but not too relaxed either)
    // cause elements are normally
    T lb = -1.0 - tol, ub = 1.0 + tol;
    bool valid_xi = lb <= xis[0] && xis[0] <= ub;
    bool valid_eta = lb <= xis[1] && xis[1] <= ub;
    // bool valid_zeta = abs(xis[2]) < dzeta * 0.45;  // approx dzeta criterion?
    // bool valid_zeta = abs(xis[2]) < dzeta * 0.8;  // approx dzeta criterion?
    // bool valid_zeta = abs(xis[2]) < 0.8 * dzeta || abs(xis[2]) < 1e-3;  // in-plane
    // bool valid_zeta = abs(xis[2]) < 0.5 * dzeta || abs(xis[2]) < 1e-5;  // in-plane, stricter

    // // general criterion, really for spar to OML and OML to spar/rib case need dzeta limit and
    // low
    // // zeta planar case too // need 0.5 not 0.8 dzeta to get better UV defects..
    // bool valid_zeta_case1 = abs(xis[2]) < 0.5 * dzeta || abs(xis[2]) < 1e-5;  // in-plane,
    // stricter

    // // center of elem curvilinear surface case (like fine node @ center of coarse element and )
    // // no zeta criterion needed cause it's coarse node neighbors have the coarse element
    // bool center_elem = abs(xis[0]) < 2e-1 && abs(xis[1]) < 2e-1;
    // bool valid_zeta_case2 = center_elem;  // & abs(xis[2]) < 4 * dzeta;
    // bool valid_zeta = valid_zeta_case1 || valid_zeta_case2;
    // // TBD : would like more robust algorithm here (less heuristic.. but we'll see about that
    // later)

    // return valid_xi && valid_eta && valid_zeta;
    return valid_xi && valid_eta && match_comp;
}

template <typename T, class Basis>
void _compute_prolong_xi_coords(const int nnodes_fine, const int nn, int *h_coarse_conn,
                                const int ELEM_MAX, int &ntot_elems, T *h_xpts_fine,
                                T *h_xpts_coarse, int *nn_conn, int *cnode_elem_ptr,
                                int *cnode_elems, int *h_coarse_elem_comps, int *f_ecomps_ptr,
                                int *f_ecomps_comp, int *fine_nodes_celem_cts,
                                int *fine_nodes_celems, T *fine_node_xis) {
    /* compute the computational xi coords of fine nodes and their nearby coarse elems */

    // outputs are: ntot_elems, fine_nodes_celem_cts, fine_nodes_celems, fine_node_xis

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T *fine_node_xpts = &h_xpts_fine[3 * inode_f];

        // should be ~24 elems examined per coarse node (can parallelize this on GPU if needed)
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];

            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                int ielem_c = cnode_elems[jp];
                // get coarse element component
                int c_comp = h_coarse_elem_comps[ielem_c];

                // check among comp ptr to see if fine node also belongs to this component
                bool match_comp = false;
                for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                    int f_comp = f_ecomps_comp[jjp];
                    match_comp += f_comp == c_comp;
                    // if (print_node) printf("%d ", f_comp);
                }
                // if (print_node) printf(", match_comp %d\n", (int)match_comp);

                T coarse_elem_xpts[12];
                get_elem_xpts<T>(ielem_c, h_coarse_conn, h_xpts_coarse, coarse_elem_xpts);

                // bool print_debug = inode_f == 1098;
                bool print_debug = false;

                T xi[3];
                get_comp_coords<T, Basis>(coarse_elem_xpts, fine_node_xpts, xi, print_debug);

                // determine whether xi[3] is in bounds or not; xi & eta in [-1,1] and zeta in
                // [-2,2] for max thick (or can ignore zeta..)
                // bool node_in_elem = xis_in_elem<T>(xi, dzeta);
                bool node_in_elem = xis_in_elem<T>(xi, match_comp);

                if (node_in_elem) {
                    // check if element already in old elems of this node
                    int nelems_prev = fine_nodes_celem_cts[inode_f];
                    bool new_elem = true;
                    for (int i = 0; i < nelems_prev; i++) {
                        int prev_elem = fine_nodes_celems[ELEM_MAX * inode_f + i];
                        new_elem = new_elem && (prev_elem != ielem_c);
                    }

                    if (new_elem) {
                        fine_nodes_celem_cts[inode_f]++;
                        ntot_elems++;
                        fine_nodes_celems[ELEM_MAX * inode_f + nelems_prev] = ielem_c;
                        fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev] = xi[0];
                        fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev + 1] = xi[1];
                    }
                }
            }
        }
    }
}

template <typename T, class Basis>
__GLOBAL__ void k_compute_prolong_xis(const int n2e_nnz, const int *d_coarse_conn,
                                      const T *d_xpts_fine, const T *d_xpts_coarse,
                                      const int *d_n2e_fnodes, const int *d_n2e_celems,
                                      bool *d_n2e_in_elem, T *d_n2e_xis) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n2e_nnz) return;

    // prelim : get node + elem indices and xpts
    int inode_f = d_n2e_fnodes[tid];
    int ielem_c = d_n2e_celems[tid];
    T fine_xpt[3];
    for (int i = 0; i < 3; i++) fine_xpt[i] = d_xpts_fine[3 * inode_f + i];
    __syncthreads();

    T coarse_elem_xpts[12];
    for (int i = 0; i < 12; i++) {
        int lnode_c = i / 3, idof_c = i % 3;
        int inode_c = d_coarse_conn[4 * ielem_c + lnode_c];
        coarse_elem_xpts[i] = d_xpts_coarse[3 * inode_c + idof_c];
    }
    __syncthreads();

    // compute the comp coords (xi,eta,zeta) for each node 2 elem pair
    T xi[3];
    get_comp_coords<T, Basis>(coarse_elem_xpts, fine_xpt, xi, false);
    bool node_in_elem = xis_in_elem<T>(xi, true);
    __syncthreads();

    // now write to output arrays
    d_n2e_in_elem[tid] = node_in_elem;
    d_n2e_xis[2 * tid] = xi[0];
    d_n2e_xis[2 * tid + 1] = xi[1];
}

template <typename T, class Basis>
void _compute_prolong_xi_coords_fast(const int nnodes_fine, const int nnodes_coarse,
                                     const int nelems_coarse, const int nn, int *h_coarse_conn,
                                     const int ELEM_MAX, int &ntot_elems, T *h_xpts_fine,
                                     T *h_xpts_coarse, int *nn_conn, int *cnode_elem_ptr,
                                     int *cnode_elems, int *h_coarse_elem_comps, int *f_ecomps_ptr,
                                     int *f_ecomps_comp, int *fine_nodes_celem_cts,
                                     int *fine_nodes_celems, T *fine_node_xis) {
    /* compute the computational xi coords of fine nodes and their nearby coarse elems */
    // faster version than the one above thanks to GPU kernel

    // 1) compute all candidate coarse elements with matching DV comp first,
    // then, compute the nonzero patterns for efficient GPU implementation:
    // len(nnz) : fine node, e.g. 0,0,0,0,1,1,1,1,1,1,2,2,2,2,2, etc. (like rows CSR array)
    // len(nnz) : coarse elem, etc..
    // --------------------------------------------------------------------------

    // get a new nonzero pattern that ensures only candidate elements with matching DV components
    // are kept
    int n2e_nnz = 0;
    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T *fine_node_xpts = &h_xpts_fine[3 * inode_f];
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];
            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                int ielem_c = cnode_elems[jp];
                int c_comp = h_coarse_elem_comps[ielem_c];

                // check among comp ptr to see if fine node also belongs to this component
                bool match_comp = false;
                for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                    int f_comp = f_ecomps_comp[jjp];
                    match_comp += f_comp == c_comp;
                    // if (print_node) printf("%d ", f_comp);
                }
                if (match_comp) n2e_nnz++;
            }
        }
    }

    // loop back through and store the fine node + coarse elems for each match DV comp
    int *n2e_fnodes = new int[n2e_nnz];
    int *n2e_celems = new int[n2e_nnz];
    int n2e_inz = 0;

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T *fine_node_xpts = &h_xpts_fine[3 * inode_f];
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];
            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                int ielem_c = cnode_elems[jp];
                int c_comp = h_coarse_elem_comps[ielem_c];

                // check among comp ptr to see if fine node also belongs to this component
                bool match_comp = false;
                for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                    int f_comp = f_ecomps_comp[jjp];
                    match_comp += f_comp == c_comp;
                    // if (print_node) printf("%d ", f_comp);
                }
                if (match_comp) {
                    n2e_fnodes[n2e_inz] = inode_f;
                    n2e_celems[n2e_inz] = ielem_c;
                    n2e_inz++;
                }
            }
        }
    }

    // 2) move + allocate output arrays from GPU
    // len(nnz) : is_in_elem bools (whether fine node is in elem)
    // len(2 *  nnz) : (xi,eta) of each coarse elem => fine node pair
    // --------------------------------------------------------------------

    // put these nz arrays on the GPU now
    int *d_n2e_fnodes = HostVec<int>(n2e_nnz, n2e_fnodes).createDeviceVec().getPtr();
    int *d_n2e_celems = HostVec<int>(n2e_nnz, n2e_celems).createDeviceVec().getPtr();

    // create new arrays for the GPU outputs
    bool *d_n2e_in_elem = DeviceVec<bool>(n2e_nnz).getPtr();
    T *d_n2e_xis = DeviceVec<T>(2 * n2e_nnz).getPtr();

    // recreate some things on GPU (NOTE : this is redundant and should be removed)
    T *d_xpts_fine = HostVec<T>(3 * nnodes_fine, h_xpts_fine).createDeviceVec().getPtr();
    T *d_xpts_coarse = HostVec<T>(3 * nnodes_coarse, h_xpts_coarse).createDeviceVec().getPtr();
    int *d_coarse_conn =
        HostVec<int>(Basis::num_nodes * nelems_coarse, h_coarse_conn).createDeviceVec().getPtr();

    // 3) call GPU kernel to compute (xi,eta) and bool of match
    // --------------------------------------------------------------

    dim3 block(32);
    dim3 grid((n2e_nnz + 31) / 32);
    // printf("before compute prolong xis GPU kernel\n");

    k_compute_prolong_xis<T, Basis><<<grid, block>>>(n2e_nnz, d_coarse_conn, d_xpts_fine,
                                                     d_xpts_coarse, d_n2e_fnodes, d_n2e_celems,
                                                     d_n2e_in_elem, d_n2e_xis);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("after compute prolong xis GPU kernel\n");

    // 4) get data out and reformat for host outputs
    // --------------------------------------------------------------
    // outputs are: ntot_elems, fine_nodes_celem_cts, fine_nodes_celems, fine_node_xis

    bool *h_n2e_in_elem = DeviceVec<bool>(n2e_nnz, d_n2e_in_elem).createHostVec().getPtr();
    T *h_n2e_xis = DeviceVec<T>(2 * n2e_nnz, d_n2e_xis).createHostVec().getPtr();

    // now insert and construct the final map, removing repeat elements (on host)
    for (int inz = 0; inz < n2e_nnz; inz++) {
        int inode_f = n2e_fnodes[inz];
        int ielem_c = n2e_celems[inz];
        bool node_in_elem = h_n2e_in_elem[inz];
        T *xi = &h_n2e_xis[2 * inz];

        if (node_in_elem) {
            // check if element already in old elems of this node
            int nelems_prev = fine_nodes_celem_cts[inode_f];
            bool new_elem = true;
            for (int i = 0; i < nelems_prev; i++) {
                int prev_elem = fine_nodes_celems[ELEM_MAX * inode_f + i];
                new_elem = new_elem && (prev_elem != ielem_c);
            }

            if (new_elem) {
                fine_nodes_celem_cts[inode_f]++;
                ntot_elems++;
                fine_nodes_celems[ELEM_MAX * inode_f + nelems_prev] = ielem_c;
                fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev] = xi[0];
                fine_node_xis[2 * ELEM_MAX * inode_f + 2 * nelems_prev + 1] = xi[1];
            }
        }
    }
}

template <typename T, class Assembler, class Basis, bool is_bsr, bool include_restrict = true>
void init_unstructured_grid_maps(Assembler &fine_assembler, Assembler &coarse_assembler,
                                 BsrMat<DeviceVec<T>> *&prolong_mat,
                                 BsrMat<DeviceVec<T>> *&restrict_mat, int *&d_coarse_conn,
                                 int *&d_n2e_ptr, int *&d_n2e_elems, T *&d_n2e_xis,
                                 const int ELEM_MAX = 10) {
    /* initialize the unstructured mesh prolongation map */
    // TBD, want to get the coarse nodes

    auto start0 = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------
    // 1) prelim prolongation maps and data here..

    CHECK_CUDA(cudaDeviceSynchronize());
    auto time_01 = std::chrono::high_resolution_clock::now();

    T *h_xpts_fine = fine_assembler.getXpts().createHostVec().getPtr();
    T *h_xpts_coarse = coarse_assembler.getXpts().createHostVec().getPtr();
    int nnodes_fine = fine_assembler.get_num_nodes();
    int nnodes_coarse = coarse_assembler.get_num_nodes();

    // printf("nnodes_fine %d, nnodes_coarse %d\n", nnodes_fine, nnodes_coarse);
    // printf("h_xpts_fine: ");
    // printVec<T>(20, h_xpts_fine);
    // printf("h_xpts_coarse: ");
    // printVec<T>(20, h_xpts_coarse);
    // return;

    int min_bin_size = 10;
    auto *locator = new LocatePoint<T>(h_xpts_coarse, nnodes_coarse, min_bin_size);

    int nn = 6;  // number of coarse node nearest neighbors
    int *nn_conn = new int[nn * nnodes_fine];
    // temp work arrays for each point
    int *indx = new int[nn];
    T *dist = new T[nn];

    // also get fine nearest neighbors for fine dzeta estimates later..

    // fine to coarse node nearest neighbors
    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T loc_xfine[3];
        memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));  // is this part necessary?

        locator->locateKClosest(nn, indx, dist, loc_xfine);

        for (int k = 0; k < nn; k++) {
            nn_conn[nn * inode_f + k] = indx[k];
        }
    }
    delete[] indx;
    delete[] dist;
    // delete[] locator;

    // fine to fine node nearest neighbors
    // need more nearest neighbors for fine node dzeta estimate
    int nn_f = 20;  // number of coarse node nearest neighbors
    auto *locator_fine = new LocatePoint<T>(h_xpts_fine, nnodes_fine, min_bin_size);
    int *nn_conn_fine = new int[nn_f * nnodes_fine];
    indx = new int[nn_f];
    dist = new T[nn_f];

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T loc_xfine[3];
        memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));  // is this part necessary?
        locator_fine->locateKClosest(nn_f, indx, dist, loc_xfine);

        for (int k = 0; k < nn_f; k++) {
            nn_conn_fine[nn_f * inode_f + k] = indx[k];
        }
    }

    // printf("done with locate point\n");
    // return;

    // printf("nn conn: ");
    // printVec<int>(30, nn_conn);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_02 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> nn_time = time_02 - time_01;
    // printf("\t\t\tnearest neighbor time %.2e\n", nn_time.count());

    // -------------------------------------------------------
    // 2) get coarse elements for each coarse node
    auto d_coarse_conn_vec = coarse_assembler.getConn();
    int *h_coarse_conn = d_coarse_conn_vec.createHostVec().getPtr();
    int *cnode_elem_cts = new int[nnodes_coarse];
    memset(cnode_elem_cts, 0.0, nnodes_coarse * sizeof(int));
    int *cnode_elem_ptr = new int[nnodes_coarse + 1];
    int num_coarse_elems = coarse_assembler.get_num_elements();
    // int ncoarse_elems = num_coarse_elems;
    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < 4; iloc++) {
            int cnode = h_coarse_conn[4 * ielem + iloc];
            cnode_elem_cts[cnode] += 1;
        }
    }

    // printf("cnode elem cts: ");
    // printVec<int>(nnodes_coarse, cnode_elem_cts);

    // like rowp here (says where and how many elems this node is a part of)
    cnode_elem_ptr[0] = 0;
    for (int inode = 0; inode < nnodes_coarse; inode++) {
        cnode_elem_ptr[inode + 1] = cnode_elem_ptr[inode] + cnode_elem_cts[inode];
    }
    int n_coarse_node_elems = cnode_elem_ptr[nnodes_coarse];

    // now we put which elems each coarse node is connected to (like cols array here)
    // reset row cts to 0, so you can trick which local elem you're writing in..
    int *_cnode_next = new int[nnodes_coarse];
    memcpy(_cnode_next, cnode_elem_ptr, nnodes_coarse * sizeof(int));
    int *cnode_elems = new int[n_coarse_node_elems];
    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < 4; iloc++) {
            int cnode = h_coarse_conn[4 * ielem + iloc];
            cnode_elems[_cnode_next[cnode]++] = ielem;
        }
    }

    // printf("cnode elem ptr: ");
    // printVec<int>(nnodes_coarse + 1, cnode_elem_ptr);
    // printf("cnode elems (cols): ");
    // printVec<int>(n_coarse_node_elems, cnode_elems);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_03 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> coarse_n2e_time = time_03 - time_02;
    // printf("\t\t\tcoarse n2e time %.2e\n", coarse_n2e_time.count());

    // -----------------------------------------------------------
    // 2.5 ) get the components for each coarse and fine node
    int num_fine_elems = fine_assembler.get_num_elements();
    int *h_fine_elem_comps = fine_assembler.getElemComponents().createHostVec().getPtr();
    int *h_coarse_elem_comps = coarse_assembler.getElemComponents().createHostVec().getPtr();
    int *h_fine_conn = fine_assembler.getConn().createHostVec().getPtr();

    // printf("h_fine_elem_comps: ");
    // printVec<int>(100, h_fine_elem_comps);

    // printf("h_fine_conn: ");
    // printVec<int>(30, h_fine_conn);

    int *f_ecomps_cts = new int[nnodes_fine];
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
    int *f_ecomps_ptr0 = new int[nnodes_fine + 1];
    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        // int fine_comp = h_fine_elem_comps[ielem];
        for (int iloc = 0; iloc < 4; iloc++) {
            int fnode = h_fine_conn[4 * ielem + iloc];
            // printf("fine elem %d, node %d\n", ielem, fnode);
            f_ecomps_cts[fnode]++;
        }
    }

    f_ecomps_ptr0[0] = 0;
    for (int inode = 0; inode < nnodes_fine; inode++) {
        f_ecomps_ptr0[inode + 1] = f_ecomps_ptr0[inode] + f_ecomps_cts[inode];
    }
    int f_ecomps_nnz0 = f_ecomps_ptr0[nnodes_fine];
    int *f_ecomps_comp0 = new int[f_ecomps_nnz0];
    // reset counts to fill comp
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
    memset(f_ecomps_comp0, -1, f_ecomps_nnz0 * sizeof(int));
    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        int fine_comp = h_fine_elem_comps[ielem];
        for (int iloc = 0; iloc < 4; iloc++) {
            int fnode = h_fine_conn[4 * ielem + iloc];
            int start = f_ecomps_ptr0[fnode];
            int write = f_ecomps_cts[fnode] + start;
            // check if this comp already written in or not..
            bool new_comp = true;
            for (int jp = start; jp < write; jp++) {
                int _comp = f_ecomps_comp0[jp];
                if (_comp == fine_comp) new_comp = false;
            }

            if (new_comp) {
                f_ecomps_comp0[write] = fine_comp;
                f_ecomps_cts[fnode]++;
            }
        }
    }

    // now that we've prevented repeats, adjust ptr size and write new comp and new nnz
    int *f_ecomps_ptr = new int[nnodes_fine + 1];
    f_ecomps_ptr[0] = 0;
    for (int inode = 0; inode < nnodes_fine; inode++) {
        f_ecomps_ptr[inode + 1] = f_ecomps_ptr[inode] + f_ecomps_cts[inode];
    }
    int f_ecomps_nnz = f_ecomps_ptr[nnodes_fine];
    int *f_ecomps_comp = new int[f_ecomps_nnz];
    // reset counts to fill comp
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
    memset(f_ecomps_comp0, -1, f_ecomps_nnz0 * sizeof(int));
    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        int fine_comp = h_fine_elem_comps[ielem];
        for (int iloc = 0; iloc < 4; iloc++) {
            int fnode = h_fine_conn[4 * ielem + iloc];
            int start = f_ecomps_ptr[fnode];
            int write = f_ecomps_cts[fnode] + start;
            // check if this comp already written in or not..
            bool new_comp = true;
            for (int jp = start; jp < write; jp++) {
                int _comp = f_ecomps_comp[jp];
                if (_comp == fine_comp) new_comp = false;
            }

            if (new_comp) {
                f_ecomps_comp[write] = fine_comp;
                f_ecomps_cts[fnode]++;
            }
        }
    }

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_04 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> e2c_time = time_04 - time_03;
    // printf("\t\t\tcoarse + fine elem2comp time %.2e\n", e2c_time.count());

    // ----------------------------------------------------------------
    // 3) get the coarse element(s) for each fine node (that it's contained in)

    int *fine_nodes_celem_cts = new int[nnodes_fine];
    memset(fine_nodes_celem_cts, 0, nnodes_fine * sizeof(int));
    int *fine_nodes_celems = new int[ELEM_MAX * nnodes_fine];
    memset(fine_nodes_celems, -1, ELEM_MAX * nnodes_fine * sizeof(int));
    T *fine_node_xis = new T[2 * ELEM_MAX * nnodes_fine];
    memset(fine_node_xis, 0.0, 2 * ELEM_MAX * nnodes_fine * sizeof(T));
    int ntot_elems = 0;

    // V1 cpu version
    // _compute_prolong_xi_coords<T, Basis>(nnodes_fine, nn, h_coarse_conn, ELEM_MAX,
    //     ntot_elems, h_xpts_fine, h_xpts_coarse, nn_conn, cnode_elem_ptr,
    //     cnode_elems, h_coarse_elem_comps, f_ecomps_ptr, f_ecomps_comp,
    //     fine_nodes_celem_cts, fine_nodes_celems, fine_node_xis);

    // V2 GPU version TODO
    _compute_prolong_xi_coords_fast<T, Basis>(
        nnodes_fine, nnodes_coarse, num_coarse_elems, nn, h_coarse_conn, ELEM_MAX, ntot_elems,
        h_xpts_fine, h_xpts_coarse, nn_conn, cnode_elem_ptr, cnode_elems, h_coarse_elem_comps,
        f_ecomps_ptr, f_ecomps_comp, fine_nodes_celem_cts, fine_nodes_celems, fine_node_xis);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_05 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> basis_xi_time = time_05 - time_04;
    // printf("\t\t\tcontained elems + xi comp time %.2e\n", basis_xi_time.count());

    // now convert from cts to rowp, cols style as diff # elems per fine node
    int *fine_node2elem_ptr = new int[nnodes_fine + 1];
    fine_node2elem_ptr[0] = 0;
    int *fine_node2elem_elems = new int[ntot_elems];
    T *fine_node2elem_xis = new T[2 * ntot_elems];

    for (int inode = 0; inode < nnodes_fine; inode++) {
        int ct = fine_nodes_celem_cts[inode];
        fine_node2elem_ptr[inode + 1] = fine_node2elem_ptr[inode] + ct;
        int start = fine_node2elem_ptr[inode];

        for (int i = 0; i < ct; i++) {
            int src_block = ELEM_MAX * inode + i, dest_block = start + i;
            fine_node2elem_elems[dest_block] = fine_nodes_celems[src_block];
            fine_node2elem_xis[2 * dest_block] = fine_node_xis[2 * src_block];
            fine_node2elem_xis[2 * dest_block + 1] = fine_node_xis[2 * src_block + 1];
        }
    }

    // printf("h_n2e_ptr: ");
    // printVec<int>(3, fine_node2elem_ptr);
    // printf("h_n2e_elems: ");
    // printVec<int>(20, fine_node2elem_elems);
    // printf("h_n2e_xis: ");
    // printVec<T>(40, fine_node2elem_xis);

    // put these maps on the device now
    d_n2e_ptr = HostVec<int>(nnodes_fine + 1, fine_node2elem_ptr).createDeviceVec().getPtr();
    d_n2e_elems = HostVec<int>(ntot_elems, fine_node2elem_elems).createDeviceVec().getPtr();
    d_n2e_xis = HostVec<T>(2 * ntot_elems, fine_node2elem_xis).createDeviceVec().getPtr();
    d_coarse_conn = d_coarse_conn_vec.getPtr();

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_06 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> misc_maps_time = time_06 - time_05;
    // printf("\t\t\tmisc final n2e maps time %.2e\n", misc_maps_time.count());

    // create P and PT matrices (for coming assembly)
    // --------------------------------------------------------

    // int mb = nnodes_fine, nb = nnodes_coarse;
    int *d_perm = fine_assembler.getBsrData().perm;
    int *h_perm = DeviceVec<int>(nnodes_fine, d_perm).createHostVec().getPtr();
    int *d_iperm = fine_assembler.getBsrData().iperm;
    int *h_iperm = DeviceVec<int>(nnodes_fine, d_iperm).createHostVec().getPtr();
    int *d_coarse_iperm = coarse_assembler.getBsrData().iperm;
    int *h_coarse_iperm = DeviceVec<int>(nnodes_coarse, d_coarse_iperm).createHostVec().getPtr();
    int *d_coarse_perm = coarse_assembler.getBsrData().perm;

    // compute the pattern here first, TODO is to clean this up and use less mem if possible
    // at least free extra stuff
    int *h_prol_row_cts = new int[nnodes_fine];
    int *h_prolT_row_cts = new int[nnodes_coarse];
    memset(h_prol_row_cts, 0.0, nnodes_fine * sizeof(int));
    memset(h_prolT_row_cts, 0.0, nnodes_coarse * sizeof(int));
    // TODO : clean much of this up with sparse symbolic later.. though sparse symbolic uses
    // connectivity, not generating non-square matrices usually uses elem_conn which is like
    // all of [1,2,7,8] connected, not the same as here where I have pairs [1] x [4,5,7,8]
    // elements fine to coarse maybe need to make some new methods in sparse symbolic.. (but
    // let's get it running first)

    for (int inf = 0; inf < nnodes_fine; inf++) {
        int perm_inf = h_iperm[inf];
        // int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
        std::set<int> conn_c_nodes;

        for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
            int ielem_c = fine_node2elem_elems[jp];
            const int *c_elem_nodes = &h_coarse_conn[4 * ielem_c];

            // get xi to interp the fine node (so we can check if some local nodes in element aren't
            // really used) like when the fine node is on the edge node. Can just compute transpose
            // interp map, BUT ONLY DO THIS AFTER TRY SMOOTH PROLONG WITHOUT PRUNING MEM HERE FIRST

            for (int loc_node = 0; loc_node < 4; loc_node++) {
                int inc = c_elem_nodes[loc_node];
                int perm_inc = h_coarse_iperm[inc];

                // unique col cts?
                conn_c_nodes.insert(perm_inc);
            }
        }

        h_prol_row_cts[perm_inf] += conn_c_nodes.size();
        for (int perm_inc : conn_c_nodes) {
            h_prolT_row_cts[perm_inc]++;
        }
    }

    // printf("h_prol_row_cts:");
    // printVec<int>(100, h_prol_row_cts);

    // now construct rowp for P and PT
    int *h_prol_rowp = new int[nnodes_fine + 1];
    int *h_prolT_rowp = new int[nnodes_coarse + 1];
    h_prol_rowp[0] = 0;
    h_prolT_rowp[0] = 0;

    for (int inf = 0; inf < nnodes_fine; inf++) {
        h_prol_rowp[inf + 1] = h_prol_rowp[inf] + h_prol_row_cts[inf];
    }
    for (int inc = 0; inc < nnodes_coarse; inc++) {
        h_prolT_rowp[inc + 1] = h_prolT_rowp[inc] + h_prolT_row_cts[inc];
    }

    // now construct cols
    int P_nnzb = h_prol_rowp[nnodes_fine];
    int PT_nnzb = h_prolT_rowp[nnodes_coarse];
    int *h_prol_cols = new int[P_nnzb];
    int *h_prolT_cols = new int[PT_nnzb];
    int *P_next = new int[nnodes_fine];  // helper arrays to insert values into sparsity
    memcpy(P_next, h_prol_rowp, nnodes_fine * sizeof(int));
    int *PT_next = new int[nnodes_coarse];
    memcpy(PT_next, h_prolT_rowp, nnodes_coarse * sizeof(int));
    // loop back through previous maps to get the sparsity
    // use perm_inf order here to help with PT sparsity
    for (int perm_inf = 0; perm_inf < nnodes_fine; perm_inf++) {
        int inf = h_perm[perm_inf];
        // int n_celems = fine_node2elem_ptr[inf + 1] - fine_node2elem_ptr[inf];
        std::set<int> conn_c_nodes;
        for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
            int ielem_c = fine_node2elem_elems[jp];
            const int *c_elem_nodes = &h_coarse_conn[4 * ielem_c];
            for (int loc_node = 0; loc_node < 4; loc_node++) {
                int inc = c_elem_nodes[loc_node];
                int perm_inc = h_coarse_iperm[inc];

                // unique col cts?
                conn_c_nodes.insert(perm_inc);
            }
        }

        for (int perm_inc : conn_c_nodes) {
            h_prol_cols[P_next[perm_inf]++] = perm_inc;    // P cols
            h_prolT_cols[PT_next[perm_inc]++] = perm_inf;  // PT cols
        }
    }

    // double check all values filled?
    // for (int i = 0; i < nnodes_fine; i++) {
    //     printf("P[i=%d,:] : ", i);
    //     for (int jp = h_prol_rowp[i]; jp < h_prol_rowp[i+1]; jp++) {
    //         int j = h_prol_cols[jp];
    //         printf("%d, ", j);
    //     }
    //     printf("\n");
    // }

    int *h_P_rows = new int[P_nnzb];
    int *h_PT_rows = new int[PT_nnzb];
    for (int inode = 0; inode < nnodes_fine; inode++) {
        for (int jp = h_prol_rowp[inode]; jp < h_prol_rowp[inode + 1]; jp++) {
            h_P_rows[jp] = inode;
        }
    }
    for (int inode = 0; inode < nnodes_coarse; inode++) {
        for (int jp = h_prolT_rowp[inode]; jp < h_prolT_rowp[inode + 1]; jp++) {
            h_PT_rows[jp] = inode;
        }
    }
    int *d_P_rows = HostVec<int>(P_nnzb, h_P_rows).createDeviceVec().getPtr();
    int *d_PT_rows = HostVec<int>(PT_nnzb, h_PT_rows).createDeviceVec().getPtr();

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_07 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> mat_nz_patterm_time = time_07 - time_06;
    // printf("\t\t\tmat nz pattern time %.2e\n", mat_nz_patterm_time.count());

    // now put these on the device with BsrData objects for P and PT
    // TODO : later is to just assemble P and then write my own transpose Bsrmv method in
    // CUDA
    int block_dim = fine_assembler.getBsrData().block_dim;
    int P_block_dim =
        is_bsr ? block_dim : 1;  // if !is_bsr then it does same prolong on each dof_per_node
    int block_dim2 = P_block_dim * P_block_dim;  // should be 36
    auto d_P_rowp = HostVec<int>(nnodes_fine + 1, h_prol_rowp).createDeviceVec().getPtr();
    auto d_P_cols = HostVec<int>(P_nnzb, h_prol_cols).createDeviceVec().getPtr();
    auto d_P_vals = DeviceVec<T>(block_dim2 * P_nnzb);
    auto P_bsr_data =
        BsrData(nnodes_fine, P_block_dim, P_nnzb, d_P_rowp, d_P_cols, d_perm, d_iperm, false);
    P_bsr_data.mb = nnodes_fine, P_bsr_data.nb = nnodes_coarse;
    P_bsr_data.rows = d_P_rows;
    prolong_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals);

    if constexpr (include_restrict) {
        auto d_PT_rowp = HostVec<int>(nnodes_coarse + 1, h_prolT_rowp).createDeviceVec().getPtr();
        auto d_PT_cols = HostVec<int>(PT_nnzb, h_prolT_cols).createDeviceVec().getPtr();
        auto d_PT_vals = DeviceVec<T>(block_dim2 * PT_nnzb);
        auto PT_bsr_data = BsrData(nnodes_coarse, P_block_dim, PT_nnzb, d_PT_rowp, d_PT_cols,
                                d_coarse_perm, d_coarse_iperm, false);
        PT_bsr_data.mb = nnodes_coarse, PT_bsr_data.nb = nnodes_fine;
        PT_bsr_data.rows = d_PT_rows;  // for each nnz, which row is it (not same as rowp),
                                    // helps for efficient mat-prods
        restrict_mat = new BsrMat<DeviceVec<T>>(
            PT_bsr_data, d_PT_vals);  // only store this matrix on the coarse grid
    }
    

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto time_08 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> create_mat_time = time_08 - time_07;
    // printf("\t\t\tcreate mat time time %.2e\n", create_mat_time.count());

    // }  // end of Prolongation::assembly case..

    // printf("done with init unstructured grid maps\n");
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> P_PT_time = end0 - start0;
    printf("\tunstructured grid P,PT assembly in %.2e sec\n", P_PT_time.count());

    // TBD: free up temp arrays
}