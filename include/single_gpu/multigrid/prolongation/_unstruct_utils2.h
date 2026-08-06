#pragma once
#include "coupled/locate_point.h"

// ============================================================
// V2: Fully generic (supports Q1, Q2, etc. via Basis::num_nodes)
// ============================================================

// ------------------------------------------------------------
// Normal computation
// ------------------------------------------------------------
template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeCenterNormal_v2(const T Xpts[], T fn[]) {
    T pt[2] = {0.0, 0.0};

    T dXdxi[3], dXdeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, Xpts, dXdxi, dXdeta);

    T tmp[3];
    A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
    T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
    A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, fn);
}

// ------------------------------------------------------------
// Generic element xpts (Q1, Q2, etc.)
// ------------------------------------------------------------
template <typename T, int nodes_per_elem>
__HOST_DEVICE__ void get_elem_xpts_v2(int ielem, const int *conn, const T *xpts,
                                      T elem_xpts[3 * nodes_per_elem]) {
    for (int iloc = 0; iloc < nodes_per_elem; iloc++) {
        int inode = conn[nodes_per_elem * ielem + iloc];

        for (int idim = 0; idim < 3; idim++) {
            elem_xpts[3 * iloc + idim] = xpts[3 * inode + idim];
        }
    }
}

// ------------------------------------------------------------
// FD-based mixed derivative (works for Q2!)
// ------------------------------------------------------------
template <typename T, class Basis>
__HOST_DEVICE__ void compute_dxi_deta_fd_v2(const T pt[2], const T Xpts[], T dxi_deta[3]) {
    const T h = 1e-6;

    T pt_p[2] = {pt[0], pt[1] + h};
    T pt_m[2] = {pt[0], pt[1] - h};

    T dxi_p[3], dummy[3];
    T dxi_m[3];

    Basis::template interpFieldsGrad<3, 3>(pt_p, Xpts, dxi_p, dummy);
    Basis::template interpFieldsGrad<3, 3>(pt_m, Xpts, dxi_m, dummy);

    for (int i = 0; i < 3; i++) {
        dxi_deta[i] = (dxi_p[i] - dxi_m[i]) / (2.0 * h);
    }
}

// ------------------------------------------------------------
// Compute (xi,eta,zeta)
// ------------------------------------------------------------
template <typename T, class Basis>
__HOST_DEVICE__ void get_comp_coords_v2(const T coarse_xpts[], const T fine_xpt[], T xis[3],
                                        bool print = false, int n_iters = 20) {
    memset(xis, 0.0, 3 * sizeof(T));

    if (print) {
        printf("coarse_xpt: ");
        printVec<T>(27, coarse_xpts);
        printf("fine xpt: ");
        printVec<T>(3, fine_xpt);
    }

    T fn[3];
    ShellComputeCenterNormal_v2<T, Basis>(coarse_xpts, fn);

    for (int ct = 0; ct < n_iters; ct++) {
        T xyz[3], dxi[3], deta[3];
        Basis::template interpFields<3, 3>(xis, coarse_xpts, xyz);

        for (int i = 0; i < 3; i++) {
            xyz[i] += xis[2] * fn[i];
        }

        Basis::template interpFieldsGrad<3, 3>(xis, coarse_xpts, dxi, deta);

        T d_xyz[3];
        for (int i = 0; i < 3; i++) {
            d_xyz[i] = xyz[i] - fine_xpt[i];
        }

        // 🔥 NEW: works for Q2
        T pt[2] = {xis[0], xis[1]};
        T dxi_deta[3];
        compute_dxi_deta_fd_v2<T, Basis>(pt, coarse_xpts, dxi_deta);

        // gradient
        T grad[3];
        grad[0] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, dxi);
        grad[1] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, deta);
        grad[2] = 2.0 * A2D::VecDotCore<T, 3>(d_xyz, fn);

        // hessian (partial)
        T hess[3];
        hess[0] = 2.0 * A2D::VecDotCore<T, 3>(dxi, dxi);
        hess[1] =
            2.0 * A2D::VecDotCore<T, 3>(dxi, deta) + 2.0 * A2D::VecDotCore<T, 3>(d_xyz, dxi_deta);
        hess[2] = 2.0 * A2D::VecDotCore<T, 3>(deta, deta);

        T discrim = hess[0] * hess[2] - hess[1] * hess[1];

        T alpha = 0.4;

        xis[0] -= alpha * (hess[2] * grad[0] - hess[1] * grad[1]) / discrim;
        xis[1] -= alpha * (hess[0] * grad[1] - hess[1] * grad[0]) / discrim;
        xis[2] -= alpha * grad[2];
        // xis[0] -= alpha * grad[0];
        // xis[1] -= alpha * grad[1];
        // xis[2] -= alpha * grad[2];

        if (print) {
            printf("iter %d\n", ct);
            printVec<T>(3, xis);
        }
    }
}

// ------------------------------------------------------------
// Element inclusion check
// ------------------------------------------------------------
template <typename T>
__HOST_DEVICE__ bool xis_in_elem_v2(const T xis[3], bool match_comp) {
    T tol = 1e-2;

    bool valid_xi = (-1.0 - tol <= xis[0]) && (xis[0] <= 1.0 + tol);
    bool valid_eta = (-1.0 - tol <= xis[1]) && (xis[1] <= 1.0 + tol);

    return valid_xi && valid_eta && match_comp;
}

// ------------------------------------------------------------
// GPU kernel (generic num_nodes)
// ------------------------------------------------------------
template <typename T, class Basis>
__GLOBAL__ void k_compute_prolong_xis_v2(int n2e_nnz, const int *conn, const T *xpts_fine,
                                         const T *xpts_coarse, const int *fnodes, const int *celems,
                                         bool *is_in_elem, T *xis_out) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n2e_nnz) return;

    int inode_f = fnodes[tid];
    int ielem_c = celems[tid];

    T fine_xpt[3];
    for (int i = 0; i < 3; i++) {
        fine_xpt[i] = xpts_fine[3 * inode_f + i];
    }

    constexpr int nn = Basis::num_nodes;
    T coarse_elem_xpts[3 * nn];

    for (int i = 0; i < 3 * nn; i++) {
        int lnode = i / 3;
        int idim = i % 3;
        int inode_c = conn[nn * ielem_c + lnode];

        coarse_elem_xpts[i] = xpts_coarse[3 * inode_c + idim];
    }

    T xi[3];
    // bool print = inode_f == 0 && threadIdx.x == 0;
    bool print = false;
    get_comp_coords_v2<T, Basis>(coarse_elem_xpts, fine_xpt, xi, print);

    // if (inode_f == 0 && threadIdx.x == 0) {
    //     printf("xi in fine node %d, coarse elem %d: \n", inode_f, ielem_c);
    //     printVec<T>(3, xi);
    // }

    bool inside = xis_in_elem_v2<T>(xi, true);

    is_in_elem[tid] = inside;
    xis_out[2 * tid] = xi[0];
    xis_out[2 * tid + 1] = xi[1];
}

template <typename T, class Basis>
void _compute_prolong_xi_coords_fast_v2(const int nnodes_fine, const int nnodes_coarse,
                                        const int nelems_coarse, const int nn, int *h_coarse_conn,
                                        const int ELEM_MAX, int &ntot_elems, T *h_xpts_fine,
                                        T *h_xpts_coarse, int *nn_conn, int *cnode_elem_ptr,
                                        int *cnode_elems, int *h_coarse_elem_comps,
                                        int *f_ecomps_ptr, int *f_ecomps_comp,
                                        int *fine_nodes_celem_cts, int *fine_nodes_celems,
                                        T *fine_node_xis) {
    /* V2: compute the computational xi coords of fine nodes and their nearby coarse elems
       Generic in Basis::num_nodes, works for Q1/Q2/etc. */

    constexpr int nnodes_per_elem = Basis::num_nodes;

    // ----------------------------------------------------------
    // 1) build candidate fine-node -> coarse-element pairs
    //    filtered by matching component
    // ----------------------------------------------------------
    int n2e_nnz = 0;

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];

            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                int ielem_c = cnode_elems[jp];
                int c_comp = h_coarse_elem_comps[ielem_c];

                bool match_comp = false;
                for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                    int f_comp = f_ecomps_comp[jjp];
                    match_comp = match_comp || (f_comp == c_comp);
                }

                if (match_comp) {
                    n2e_nnz++;
                }
            }
        }
    }

    int *n2e_fnodes = new int[n2e_nnz];
    int *n2e_celems = new int[n2e_nnz];
    int n2e_inz = 0;

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        for (int i_nn = 0; i_nn < nn; i_nn++) {
            int inode_c = nn_conn[nn * inode_f + i_nn];

            for (int jp = cnode_elem_ptr[inode_c]; jp < cnode_elem_ptr[inode_c + 1]; jp++) {
                int ielem_c = cnode_elems[jp];
                int c_comp = h_coarse_elem_comps[ielem_c];

                bool match_comp = false;
                for (int jjp = f_ecomps_ptr[inode_f]; jjp < f_ecomps_ptr[inode_f + 1]; jjp++) {
                    int f_comp = f_ecomps_comp[jjp];
                    match_comp = match_comp || (f_comp == c_comp);
                }

                if (match_comp) {
                    n2e_fnodes[n2e_inz] = inode_f;
                    n2e_celems[n2e_inz] = ielem_c;
                    n2e_inz++;
                }
            }
        }
    }

    // printf("n2e fnodes: ");
    // printVec<int>(n2e_nnz, n2e_fnodes);

    // ----------------------------------------------------------
    // 2) move candidate pairs to device
    // ----------------------------------------------------------
    int *d_n2e_fnodes = HostVec<int>(n2e_nnz, n2e_fnodes).createDeviceVec().getPtr();
    int *d_n2e_celems = HostVec<int>(n2e_nnz, n2e_celems).createDeviceVec().getPtr();

    bool *d_n2e_in_elem = DeviceVec<bool>(n2e_nnz).getPtr();
    T *d_n2e_xis = DeviceVec<T>(2 * n2e_nnz).getPtr();

    T *d_xpts_fine = HostVec<T>(3 * nnodes_fine, h_xpts_fine).createDeviceVec().getPtr();
    T *d_xpts_coarse = HostVec<T>(3 * nnodes_coarse, h_xpts_coarse).createDeviceVec().getPtr();
    int *d_coarse_conn =
        HostVec<int>(nnodes_per_elem * nelems_coarse, h_coarse_conn).createDeviceVec().getPtr();

    // ----------------------------------------------------------
    // 3) GPU kernel for xi/eta evaluation
    // ----------------------------------------------------------
    dim3 block(32);
    dim3 grid((n2e_nnz + block.x - 1) / block.x);

    k_compute_prolong_xis_v2<T, Basis><<<grid, block>>>(n2e_nnz, d_coarse_conn, d_xpts_fine,
                                                        d_xpts_coarse, d_n2e_fnodes, d_n2e_celems,
                                                        d_n2e_in_elem, d_n2e_xis);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ----------------------------------------------------------
    // 4) bring results back to host
    // ----------------------------------------------------------
    bool *h_n2e_in_elem = DeviceVec<bool>(n2e_nnz, d_n2e_in_elem).createHostVec().getPtr();
    T *h_n2e_xis = DeviceVec<T>(2 * n2e_nnz, d_n2e_xis).createHostVec().getPtr();

    // ----------------------------------------------------------
    // 5) reformat into fine-node -> coarse-elems map
    //    and remove duplicate coarse elements
    // ----------------------------------------------------------
    for (int inz = 0; inz < n2e_nnz; inz++) {
        int inode_f = n2e_fnodes[inz];
        int ielem_c = n2e_celems[inz];
        bool node_in_elem = h_n2e_in_elem[inz];
        T *xi = &h_n2e_xis[2 * inz];

        if (node_in_elem) {
            int nelems_prev = fine_nodes_celem_cts[inode_f];
            bool new_elem = true;

            for (int i = 0; i < nelems_prev; i++) {
                int prev_elem = fine_nodes_celems[ELEM_MAX * inode_f + i];
                if (prev_elem == ielem_c) {
                    new_elem = false;
                    break;
                }
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

    // ----------------------------------------------------------
    // 6) cleanup
    // ----------------------------------------------------------
    delete[] n2e_fnodes;
    delete[] n2e_celems;
}

template <typename T, class Assembler, class Basis, bool is_bsr, bool include_restrict = true>
void init_unstructured_grid_maps_v2(Assembler &fine_assembler, Assembler &coarse_assembler,
                                    BsrMat<DeviceVec<T>> *&prolong_mat,
                                    BsrMat<DeviceVec<T>> *&restrict_mat, int *&d_coarse_conn,
                                    int *&d_n2e_ptr, int *&d_n2e_elems, T *&d_n2e_xis,
                                    const int ELEM_MAX = 10) {
    auto start0 = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto time_01 = std::chrono::high_resolution_clock::now();

    constexpr int nnodes_per_elem = Basis::num_nodes;

    T *h_xpts_fine = fine_assembler.getXpts().createHostVec().getPtr();
    T *h_xpts_coarse = coarse_assembler.getXpts().createHostVec().getPtr();
    int nnodes_fine = fine_assembler.get_num_nodes();
    int nnodes_coarse = coarse_assembler.get_num_nodes();

    // -------------------------------------------------------
    // 1) nearest coarse-node neighbors for each fine node
    // -------------------------------------------------------
    int min_bin_size = 10;
    auto *locator = new LocatePoint<T>(h_xpts_coarse, nnodes_coarse, min_bin_size);

    int nn = 6;
    int *nn_conn = new int[nn * nnodes_fine];
    int *indx = new int[nn];
    T *dist = new T[nn];

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T loc_xfine[3];
        memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));
        locator->locateKClosest(nn, indx, dist, loc_xfine);

        for (int k = 0; k < nn; k++) {
            nn_conn[nn * inode_f + k] = indx[k];
        }
    }

    delete[] indx;
    delete[] dist;
    // delete locator if you want:
    // delete locator;

    // optional fine-fine neighbors, preserved from your original flow
    int nn_f = 20;
    auto *locator_fine = new LocatePoint<T>(h_xpts_fine, nnodes_fine, min_bin_size);
    int *nn_conn_fine = new int[nn_f * nnodes_fine];
    indx = new int[nn_f];
    dist = new T[nn_f];

    for (int inode_f = 0; inode_f < nnodes_fine; inode_f++) {
        T loc_xfine[3];
        memcpy(loc_xfine, &h_xpts_fine[3 * inode_f], 3 * sizeof(T));
        locator_fine->locateKClosest(nn_f, indx, dist, loc_xfine);

        for (int k = 0; k < nn_f; k++) {
            nn_conn_fine[nn_f * inode_f + k] = indx[k];
        }
    }

    // -------------------------------------------------------
    // 2) get coarse elements attached to each coarse node
    // -------------------------------------------------------
    auto d_coarse_conn_vec = coarse_assembler.getConn();
    int *h_coarse_conn = d_coarse_conn_vec.createHostVec().getPtr();

    int *cnode_elem_cts = new int[nnodes_coarse];
    memset(cnode_elem_cts, 0, nnodes_coarse * sizeof(int));

    int *cnode_elem_ptr = new int[nnodes_coarse + 1];
    int num_coarse_elems = coarse_assembler.get_num_elements();

    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < nnodes_per_elem; iloc++) {
            int cnode = h_coarse_conn[nnodes_per_elem * ielem + iloc];
            cnode_elem_cts[cnode] += 1;
        }
    }

    cnode_elem_ptr[0] = 0;
    for (int inode = 0; inode < nnodes_coarse; inode++) {
        cnode_elem_ptr[inode + 1] = cnode_elem_ptr[inode] + cnode_elem_cts[inode];
    }

    int n_coarse_node_elems = cnode_elem_ptr[nnodes_coarse];
    int *_cnode_next = new int[nnodes_coarse];
    memcpy(_cnode_next, cnode_elem_ptr, nnodes_coarse * sizeof(int));

    int *cnode_elems = new int[n_coarse_node_elems];
    for (int ielem = 0; ielem < num_coarse_elems; ielem++) {
        for (int iloc = 0; iloc < nnodes_per_elem; iloc++) {
            int cnode = h_coarse_conn[nnodes_per_elem * ielem + iloc];
            cnode_elems[_cnode_next[cnode]++] = ielem;
        }
    }

    // printf("cnode_elems: ");
    // printVec<int>(n_coarse_node_elems, cnode_elems);

    // -------------------------------------------------------
    // 2.5) fine-node component membership
    // -------------------------------------------------------
    int num_fine_elems = fine_assembler.get_num_elements();
    int *h_fine_elem_comps = fine_assembler.getElemComponents().createHostVec().getPtr();
    int *h_coarse_elem_comps = coarse_assembler.getElemComponents().createHostVec().getPtr();
    int *h_fine_conn = fine_assembler.getConn().createHostVec().getPtr();

    int *f_ecomps_cts = new int[nnodes_fine];
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));

    int *f_ecomps_ptr0 = new int[nnodes_fine + 1];
    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        for (int iloc = 0; iloc < nnodes_per_elem; iloc++) {
            int fnode = h_fine_conn[nnodes_per_elem * ielem + iloc];
            f_ecomps_cts[fnode]++;
        }
    }

    f_ecomps_ptr0[0] = 0;
    for (int inode = 0; inode < nnodes_fine; inode++) {
        f_ecomps_ptr0[inode + 1] = f_ecomps_ptr0[inode] + f_ecomps_cts[inode];
    }

    int f_ecomps_nnz0 = f_ecomps_ptr0[nnodes_fine];
    int *f_ecomps_comp0 = new int[f_ecomps_nnz0];
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
    memset(f_ecomps_comp0, -1, f_ecomps_nnz0 * sizeof(int));

    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        int fine_comp = h_fine_elem_comps[ielem];

        for (int iloc = 0; iloc < nnodes_per_elem; iloc++) {
            int fnode = h_fine_conn[nnodes_per_elem * ielem + iloc];
            int start = f_ecomps_ptr0[fnode];
            int write = f_ecomps_cts[fnode] + start;

            bool new_comp = true;
            for (int jp = start; jp < write; jp++) {
                if (f_ecomps_comp0[jp] == fine_comp) {
                    new_comp = false;
                }
            }

            if (new_comp) {
                f_ecomps_comp0[write] = fine_comp;
                f_ecomps_cts[fnode]++;
            }
        }
    }

    int *f_ecomps_ptr = new int[nnodes_fine + 1];
    f_ecomps_ptr[0] = 0;
    for (int inode = 0; inode < nnodes_fine; inode++) {
        f_ecomps_ptr[inode + 1] = f_ecomps_ptr[inode] + f_ecomps_cts[inode];
    }

    int f_ecomps_nnz = f_ecomps_ptr[nnodes_fine];
    int *f_ecomps_comp = new int[f_ecomps_nnz];
    memset(f_ecomps_cts, 0, nnodes_fine * sizeof(int));
    memset(f_ecomps_comp, -1, f_ecomps_nnz * sizeof(int));

    for (int ielem = 0; ielem < num_fine_elems; ielem++) {
        int fine_comp = h_fine_elem_comps[ielem];

        for (int iloc = 0; iloc < nnodes_per_elem; iloc++) {
            int fnode = h_fine_conn[nnodes_per_elem * ielem + iloc];
            int start = f_ecomps_ptr[fnode];
            int write = f_ecomps_cts[fnode] + start;

            bool new_comp = true;
            for (int jp = start; jp < write; jp++) {
                if (f_ecomps_comp[jp] == fine_comp) {
                    new_comp = false;
                }
            }

            if (new_comp) {
                f_ecomps_comp[write] = fine_comp;
                f_ecomps_cts[fnode]++;
            }
        }
    }

    // -------------------------------------------------------
    // 3) compute fine-node -> containing coarse-element maps
    // -------------------------------------------------------
    int *fine_nodes_celem_cts = new int[nnodes_fine];
    memset(fine_nodes_celem_cts, 0, nnodes_fine * sizeof(int));

    int *fine_nodes_celems = new int[ELEM_MAX * nnodes_fine];
    memset(fine_nodes_celems, -1, ELEM_MAX * nnodes_fine * sizeof(int));

    T *fine_node_xis = new T[2 * ELEM_MAX * nnodes_fine];
    memset(fine_node_xis, 0, 2 * ELEM_MAX * nnodes_fine * sizeof(T));

    int ntot_elems = 0;

    _compute_prolong_xi_coords_fast_v2<T, Basis>(
        nnodes_fine, nnodes_coarse, num_coarse_elems, nn, h_coarse_conn, ELEM_MAX, ntot_elems,
        h_xpts_fine, h_xpts_coarse, nn_conn, cnode_elem_ptr, cnode_elems, h_coarse_elem_comps,
        f_ecomps_ptr, f_ecomps_comp, fine_nodes_celem_cts, fine_nodes_celems, fine_node_xis);

    int *fine_node2elem_ptr = new int[nnodes_fine + 1];
    fine_node2elem_ptr[0] = 0;

    int *fine_node2elem_elems = new int[ntot_elems];
    T *fine_node2elem_xis = new T[2 * ntot_elems];

    for (int inode = 0; inode < nnodes_fine; inode++) {
        int ct = fine_nodes_celem_cts[inode];
        fine_node2elem_ptr[inode + 1] = fine_node2elem_ptr[inode] + ct;
        int start = fine_node2elem_ptr[inode];

        for (int i = 0; i < ct; i++) {
            int src_block = ELEM_MAX * inode + i;
            int dest_block = start + i;

            fine_node2elem_elems[dest_block] = fine_nodes_celems[src_block];
            fine_node2elem_xis[2 * dest_block] = fine_node_xis[2 * src_block];
            fine_node2elem_xis[2 * dest_block + 1] = fine_node_xis[2 * src_block + 1];
        }
    }

    printf("fine n2e ptr: ");
    printVec<int>(ntot_elems, fine_node2elem_ptr);
    printf("fine n2e elems: ");
    printVec<int>(ntot_elems, fine_node2elem_elems);

    d_n2e_ptr = HostVec<int>(nnodes_fine + 1, fine_node2elem_ptr).createDeviceVec().getPtr();
    d_n2e_elems = HostVec<int>(ntot_elems, fine_node2elem_elems).createDeviceVec().getPtr();
    d_n2e_xis = HostVec<T>(2 * ntot_elems, fine_node2elem_xis).createDeviceVec().getPtr();
    d_coarse_conn = d_coarse_conn_vec.getPtr();

    // -------------------------------------------------------
    // 4) build sparsity pattern of P and PT
    // -------------------------------------------------------
    int *d_perm = fine_assembler.getBsrData().perm;
    int *h_perm = DeviceVec<int>(nnodes_fine, d_perm).createHostVec().getPtr();

    int *d_iperm = fine_assembler.getBsrData().iperm;
    int *h_iperm = DeviceVec<int>(nnodes_fine, d_iperm).createHostVec().getPtr();

    int *d_coarse_iperm = coarse_assembler.getBsrData().iperm;
    int *h_coarse_iperm = DeviceVec<int>(nnodes_coarse, d_coarse_iperm).createHostVec().getPtr();

    int *d_coarse_perm = coarse_assembler.getBsrData().perm;

    int *h_prol_row_cts = new int[nnodes_fine];
    int *h_prolT_row_cts = new int[nnodes_coarse];
    memset(h_prol_row_cts, 0, nnodes_fine * sizeof(int));
    memset(h_prolT_row_cts, 0, nnodes_coarse * sizeof(int));

    for (int inf = 0; inf < nnodes_fine; inf++) {
        int perm_inf = h_iperm[inf];
        std::set<int> conn_c_nodes;

        for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
            int ielem_c = fine_node2elem_elems[jp];
            const int *c_elem_nodes = &h_coarse_conn[nnodes_per_elem * ielem_c];

            for (int loc_node = 0; loc_node < nnodes_per_elem; loc_node++) {
                int inc = c_elem_nodes[loc_node];
                int perm_inc = h_coarse_iperm[inc];
                conn_c_nodes.insert(perm_inc);
            }
        }

        h_prol_row_cts[perm_inf] += conn_c_nodes.size();
        for (int perm_inc : conn_c_nodes) {
            h_prolT_row_cts[perm_inc]++;
        }
    }

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

    int P_nnzb = h_prol_rowp[nnodes_fine];
    int PT_nnzb = h_prolT_rowp[nnodes_coarse];

    int *h_prol_cols = new int[P_nnzb];
    int *h_prolT_cols = new int[PT_nnzb];

    int *P_next = new int[nnodes_fine];
    int *PT_next = new int[nnodes_coarse];
    memcpy(P_next, h_prol_rowp, nnodes_fine * sizeof(int));
    memcpy(PT_next, h_prolT_rowp, nnodes_coarse * sizeof(int));

    for (int perm_inf = 0; perm_inf < nnodes_fine; perm_inf++) {
        int inf = h_perm[perm_inf];
        std::set<int> conn_c_nodes;

        for (int jp = fine_node2elem_ptr[inf]; jp < fine_node2elem_ptr[inf + 1]; jp++) {
            int ielem_c = fine_node2elem_elems[jp];
            const int *c_elem_nodes = &h_coarse_conn[nnodes_per_elem * ielem_c];

            for (int loc_node = 0; loc_node < nnodes_per_elem; loc_node++) {
                int inc = c_elem_nodes[loc_node];
                int perm_inc = h_coarse_iperm[inc];
                conn_c_nodes.insert(perm_inc);
            }
        }

        for (int perm_inc : conn_c_nodes) {
            h_prol_cols[P_next[perm_inf]++] = perm_inc;
            h_prolT_cols[PT_next[perm_inc]++] = perm_inf;
        }
    }

    // printf("h_prol_rowp: ");
    // printVec<int>(nnodes_fine + 1, h_prol_rowp);
    // printf("h_prol_cols: ");
    // printVec<int>(P_nnzb, h_prol_cols);

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

    // -------------------------------------------------------
    // 5) allocate P and PT matrices
    // -------------------------------------------------------
    int block_dim = fine_assembler.getBsrData().block_dim;
    int P_block_dim = is_bsr ? block_dim : 1;
    int block_dim2 = P_block_dim * P_block_dim;

    for (int i = 0; i < nnodes_fine; i++) {
        printf("h_prol row[%d]: ", i);
        for (int jp = h_prol_rowp[i]; jp < h_prol_rowp[i + 1]; jp++) {
            int j = h_prol_cols[jp];
            printf("%d, ", j);
        }
        printf("\n");
    }

    auto d_P_rowp = HostVec<int>(nnodes_fine + 1, h_prol_rowp).createDeviceVec().getPtr();
    auto d_P_cols = HostVec<int>(P_nnzb, h_prol_cols).createDeviceVec().getPtr();
    auto d_P_vals = DeviceVec<T>(block_dim2 * P_nnzb);

    auto P_bsr_data =
        BsrData(nnodes_fine, P_block_dim, P_nnzb, d_P_rowp, d_P_cols, d_perm, d_iperm, false);
    P_bsr_data.mb = nnodes_fine;
    P_bsr_data.nb = nnodes_coarse;
    P_bsr_data.rows = d_P_rows;

    // int *h_rowp = DeviceVec<int>(P_bsr_data.mb + 1, P_bsr_data.rowp).createHostVec().getPtr();
    // int *h_cols = DeviceVec<int>(P_bsr_data.nnzb, P_bsr_data.cols).createHostVec().getPtr();

    // for (int i = 0; i < P_bsr_data.mb; i++) {
    //     printf("unstruct util2: h_prol row[%d]: ", i);
    //     for (int jp = h_rowp[i]; jp < h_rowp[i + 1]; jp++) {
    //         int j = h_cols[jp];
    //         printf("%d, ", j);
    //     }
    //     printf("\n");
    // }

    // check bcs exist on prolongator..

    prolong_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals);

    if constexpr (include_restrict) {
        auto d_PT_rowp = HostVec<int>(nnodes_coarse + 1, h_prolT_rowp).createDeviceVec().getPtr();
        auto d_PT_cols = HostVec<int>(PT_nnzb, h_prolT_cols).createDeviceVec().getPtr();
        auto d_PT_vals = DeviceVec<T>(block_dim2 * PT_nnzb);

        auto PT_bsr_data = BsrData(nnodes_coarse, P_block_dim, PT_nnzb, d_PT_rowp, d_PT_cols,
                                   d_coarse_perm, d_coarse_iperm, false);
        PT_bsr_data.mb = nnodes_coarse;
        PT_bsr_data.nb = nnodes_fine;
        PT_bsr_data.rows = d_PT_rows;

        restrict_mat = new BsrMat<DeviceVec<T>>(PT_bsr_data, d_PT_vals);
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> P_PT_time = end0 - start0;
    printf("init_unstructured_grid_maps_v2 done in %.2e sec\n", P_PT_time.count());
}
