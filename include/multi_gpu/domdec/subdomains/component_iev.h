#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "domdec/subdomains/_iev.h"

template <typename T>
class TacsComponentIEVSplitting {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 6;  // update this?

    const int *elem_conn = nullptr;        // size num_elements * nodes_per_elem
    const T *xpts = nullptr;               // size num_nodes * 3
    const int *elem_components = nullptr;  // size num_elements

    int num_components = 0;
    int num_subdomains = 0;

    // element -> subdomain
    std::vector<int> elem_sd_ind;

    // node -> subdomain incidence, with repeats
    int node_elem_nnz = 0;
    std::vector<int> node_elem_rowp;
    std::vector<int> node_elem_ct;
    std::vector<int> node_sd_cols;

    // global node classification
    std::vector<int> node_class_ind;
    std::vector<int> node_nsd;

    int I_nnodes = 0;
    int IE_nnodes = 0;
    int IEV_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;

    // duplicated IEV layout
    std::vector<int> IEV_sd_ptr;     // size num_subdomains + 1
    std::vector<int> IEV_sd_ind;     // size IEV_nnodes
    std::vector<int> IEV_nodes;      // size IEV_nnodes, maps IEV node -> global node
    std::vector<int> IEV_elem_conn;  // size num_elements * nodes_per_elem

    TacsComponentIEVSplitting() = default;

    TacsComponentIEVSplitting(int num_elements_, int num_nodes_, int nodes_per_elem_,
                              const int *elem_conn_, const T *xpts_, const int *elem_components_,
                              int nxse_, int nyse_, int MOD_WRAPAROUND = -1, T wrap_frac = 1.0,
                              bool track_dirichlet = fals)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          elem_conn(elem_conn_),
          xpts(xpts_),
          elem_components(elem_components_) {
        setup_tacs_component_subdomains(nxse_, nyse_, MOD_WRAPAROUND, wrap_frac, track_dirichlet);
    }

    void free() {
        num_components = 0;
        num_subdomains = 0;

        elem_sd_ind.clear();

        node_elem_nnz = 0;
        node_elem_rowp.clear();
        node_elem_ct.clear();
        node_sd_cols.clear();

        node_class_ind.clear();
        node_nsd.clear();

        I_nnodes = 0;
        IE_nnodes = 0;
        IEV_nnodes = 0;
        Vc_nnodes = 0;
        V_nnodes = 0;
        lam_nnodes = 0;

        IEV_sd_ptr.clear();
        IEV_sd_ind.clear();
        IEV_nodes.clear();
        IEV_elem_conn.clear();
    }

   private:
    void setup_tacs_component_subdomains(int nxse_, int nyse_, int MOD_WRAPAROUND = -1,
                                         T wrap_frac = 1.0, bool track_dirichlet = false) {
        clear();

        compute_num_components();

        elem_sd_ind.assign(num_elements, -1);

        std::vector<T> elem_centroids(3 * num_elements, 0.0);
        compute_element_centroids(elem_centroids);

        int running_subdomain = 0;

        for (int icomp = 0; icomp < num_components; icomp++) {
            std::vector<int> comp_elems;
            get_component_elements(icomp, comp_elems);

            if (comp_elems.empty()) {
                continue;
            }

            std::vector<int> e2e_rowp;
            std::vector<int> e2e_cols;
            build_component_elem_adjacency(comp_elems, e2e_rowp, e2e_cols);

            T ref_axis1[3], ref_axis2[3];
            choose_component_axes(comp_elems, elem_centroids, ref_axis1, ref_axis2);

            std::vector<int> ixe_indices(comp_elems.size(), 0);
            std::vector<int> iye_indices(comp_elems.size(), 0);
            assign_component_logical_indices(comp_elems, e2e_rowp, e2e_cols, elem_centroids,
                                             ref_axis1, ref_axis2, ixe_indices, iye_indices);

            int ixe_min = ixe_indices[0];
            int ixe_max = ixe_indices[0];
            int iye_min = iye_indices[0];
            int iye_max = iye_indices[0];

            for (int i = 1; i < (int)comp_elems.size(); i++) {
                ixe_min = std::min(ixe_min, ixe_indices[i]);
                ixe_max = std::max(ixe_max, ixe_indices[i]);
                iye_min = std::min(iye_min, iye_indices[i]);
                iye_max = std::max(iye_max, iye_indices[i]);
            }

            for (int i = 0; i < (int)comp_elems.size(); i++) {
                ixe_indices[i] -= ixe_min;
                iye_indices[i] -= iye_min;
            }

            int nxs_comp = (ixe_max - ixe_min + nxse_ - 1) / nxse_;
            int nys_comp = (iye_max - iye_min + nyse_ - 1) / nyse_;

            if (MOD_WRAPAROUND != -1) {
                for (int i = 0; i < (int)comp_elems.size(); i++) {
                    ixe_indices[i] += MOD_WRAPAROUND;
                    iye_indices[i] += MOD_WRAPAROUND;
                }

                nxs_comp = (ixe_max - ixe_min + MOD_WRAPAROUND + nxse_ - 1) / nxse_;
                nys_comp = (iye_max - iye_min + MOD_WRAPAROUND + nyse_ - 1) / nyse_;
            }

            for (int i = 0; i < (int)comp_elems.size(); i++) {
                int ielem = comp_elems[i];

                int ixs = ixe_indices[i] / nxse_;
                int iys = iye_indices[i] / nyse_;

                elem_sd_ind[ielem] = running_subdomain + ixs + iys * nxs_comp;
            }

            running_subdomain += nxs_comp * nys_comp;
        }

        num_subdomains = running_subdomain;

        check_all_elements_assigned();

        build_node_subdomain_incidence();
        classify_nodes();
        build_IEV_nodes();
        build_IEV_elem_conn();

        printf("TacsComponentIEVSplitting complete:\n");
        printf("  num_components  = %d\n", num_components);
        printf("  num_subdomains  = %d\n", num_subdomains);
        printf("  I_nnodes        = %d\n", I_nnodes);
        printf("  IE_nnodes       = %d\n", IE_nnodes);
        printf("  IEV_nnodes      = %d\n", IEV_nnodes);
        printf("  Vc_nnodes       = %d\n", Vc_nnodes);
        printf("  V_nnodes        = %d\n", V_nnodes);
        printf("  lam_nnodes      = %d\n", lam_nnodes);
    }

    static T dot3(const T *a, const T *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

    void compute_num_components() {
        int max_comp = -1;
        for (int ielem = 0; ielem < num_elements; ielem++) {
            max_comp = std::max(max_comp, elem_components[ielem]);
        }
        num_components = max_comp + 1;
    }

    void compute_element_centroids(std::vector<T> &elem_centroids) {
        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];

            T c[3] = {0.0, 0.0, 0.0};

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];

                c[0] += xpts[3 * gnode + 0];
                c[1] += xpts[3 * gnode + 1];
                c[2] += xpts[3 * gnode + 2];
            }

            T inv = 1.0 / (T)nodes_per_elem;

            elem_centroids[3 * ielem + 0] = inv * c[0];
            elem_centroids[3 * ielem + 1] = inv * c[1];
            elem_centroids[3 * ielem + 2] = inv * c[2];
        }
    }

    void get_component_elements(int icomp, std::vector<int> &comp_elems) {
        comp_elems.clear();

        for (int ielem = 0; ielem < num_elements; ielem++) {
            if (elem_components[ielem] == icomp) {
                comp_elems.push_back(ielem);
            }
        }
    }

    bool elements_share_edge(int ielem, int jelem) const {
        const int *ci = &elem_conn[nodes_per_elem * ielem];
        const int *cj = &elem_conn[nodes_per_elem * jelem];

        int num_match = 0;

        for (int ii = 0; ii < nodes_per_elem; ii++) {
            for (int jj = 0; jj < nodes_per_elem; jj++) {
                if (ci[ii] == cj[jj]) {
                    num_match++;
                }
            }
        }

        return num_match > 1;
    }

    void build_component_elem_adjacency(const std::vector<int> &comp_elems,
                                        std::vector<int> &e2e_rowp, std::vector<int> &e2e_cols) {
        int ne = (int)comp_elems.size();

        e2e_rowp.assign(ne + 1, 0);

        for (int i = 0; i < ne; i++) {
            int ielem = comp_elems[i];

            int count = 0;
            for (int j = 0; j < ne; j++) {
                if (i == j) continue;

                int jelem = comp_elems[j];

                if (elements_share_edge(ielem, jelem)) {
                    count++;
                }
            }

            e2e_rowp[i + 1] = e2e_rowp[i] + count;
        }

        e2e_cols.assign(e2e_rowp[ne], -1);

        for (int i = 0; i < ne; i++) {
            int ielem = comp_elems[i];
            int offset = e2e_rowp[i];

            for (int j = 0; j < ne; j++) {
                if (i == j) continue;

                int jelem = comp_elems[j];

                if (elements_share_edge(ielem, jelem)) {
                    e2e_cols[offset++] = j;
                }
            }
        }
    }

    void choose_component_axes(const std::vector<int> &comp_elems,
                               const std::vector<T> &elem_centroids, T ref_axis1[3],
                               T ref_axis2[3]) {
        T xmin[3] = {elem_centroids[3 * comp_elems[0] + 0], elem_centroids[3 * comp_elems[0] + 1],
                     elem_centroids[3 * comp_elems[0] + 2]};

        T xmax[3] = {xmin[0], xmin[1], xmin[2]};

        for (int ielem : comp_elems) {
            for (int k = 0; k < 3; k++) {
                T x = elem_centroids[3 * ielem + k];
                xmin[k] = std::min(xmin[k], x);
                xmax[k] = std::max(xmax[k], x);
            }
        }

        T range[3] = {xmax[0] - xmin[0], xmax[1] - xmin[1], xmax[2] - xmin[2]};

        int a = 0;
        if (range[1] > range[a]) a = 1;
        if (range[2] > range[a]) a = 2;

        int b = (a == 0) ? 1 : 0;
        for (int k = 0; k < 3; k++) {
            if (k != a && range[k] > range[b]) {
                b = k;
            }
        }

        ref_axis1[0] = ref_axis1[1] = ref_axis1[2] = 0.0;
        ref_axis2[0] = ref_axis2[1] = ref_axis2[2] = 0.0;

        ref_axis1[a] = 1.0;
        ref_axis2[b] = 1.0;
    }

    void assign_component_logical_indices(const std::vector<int> &comp_elems,
                                          const std::vector<int> &e2e_rowp,
                                          const std::vector<int> &e2e_cols,
                                          const std::vector<T> &elem_centroids,
                                          const T ref_axis1[3], const T ref_axis2[3],
                                          std::vector<int> &ixe_indices,
                                          std::vector<int> &iye_indices) {
        int ne = (int)comp_elems.size();

        int min_lelem = -1;
        T min_xi = 1e30;
        T min_eta = 1e30;

        for (int i = 0; i < ne; i++) {
            int ielem = comp_elems[i];
            const T *xc = &elem_centroids[3 * ielem];

            T xi = dot3(xc, ref_axis1);
            T eta = dot3(xc, ref_axis2);

            if (xi + eta < min_xi + min_eta) {
                min_lelem = i;
                min_xi = xi;
                min_eta = eta;
            }
        }

        if (min_lelem < 0) {
            printf("ERROR: failed to find seed element for component\n");
            std::exit(1);
        }

        std::vector<char> assigned(ne, 0);

        ixe_indices[min_lelem] = 0;
        iye_indices[min_lelem] = 0;
        assigned[min_lelem] = 1;

        int n_assigned = 1;
        int i = min_lelem;

        while (n_assigned < ne) {
            bool assigned_new = false;

            for (int jp = e2e_rowp[i]; jp < e2e_rowp[i + 1]; jp++) {
                int j = e2e_cols[jp];

                if (assigned[j]) continue;

                int ielem = comp_elems[i];
                int jelem = comp_elems[j];

                const T *xci = &elem_centroids[3 * ielem];
                const T *xcj = &elem_centroids[3 * jelem];

                T dx[3] = {xcj[0] - xci[0], xcj[1] - xci[1], xcj[2] - xci[2]};

                T xi_dist = dot3(dx, ref_axis1);
                T eta_dist = dot3(dx, ref_axis2);

                if (std::fabs(xi_dist) > std::fabs(eta_dist)) {
                    int sign = (xi_dist > 0.0) ? 1 : -1;
                    ixe_indices[j] = ixe_indices[i] + sign;
                    iye_indices[j] = iye_indices[i];
                } else {
                    int sign = (eta_dist > 0.0) ? 1 : -1;
                    ixe_indices[j] = ixe_indices[i];
                    iye_indices[j] = iye_indices[i] + sign;
                }

                assigned[j] = 1;
                n_assigned++;
                i = j;
                assigned_new = true;
                break;
            }

            if (!assigned_new) {
                bool found_seed = false;

                for (int j = 0; j < ne; j++) {
                    if (!assigned[j]) continue;

                    bool has_unassigned_adj = false;
                    for (int kp = e2e_rowp[j]; kp < e2e_rowp[j + 1]; kp++) {
                        int k = e2e_cols[kp];
                        if (!assigned[k]) {
                            has_unassigned_adj = true;
                            break;
                        }
                    }

                    if (has_unassigned_adj) {
                        i = j;
                        found_seed = true;
                        break;
                    }
                }

                if (!found_seed) {
                    printf(
                        "WARNING: component logical-index assignment disconnected; assigned "
                        "%d/%d\n",
                        n_assigned, ne);
                    break;
                }
            }
        }
    }

    void build_node_subdomain_incidence() {
        node_elem_nnz = 0;

        node_elem_rowp.assign(num_nodes + 1, 0);
        node_elem_ct.assign(num_nodes, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];

                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        std::vector<int> temp(num_nodes, 0);
        node_sd_cols.assign(node_elem_nnz, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];

                int offset = node_elem_rowp[gnode] + temp[gnode];
                node_sd_cols[offset] = isd;
                temp[gnode]++;
            }
        }
    }

    void classify_nodes() {
        node_class_ind.assign(num_nodes, INTERIOR);
        node_nsd.assign(num_nodes, 0);

        I_nnodes = 0;
        IE_nnodes = 0;
        IEV_nnodes = 0;
        Vc_nnodes = 0;
        V_nnodes = 0;
        lam_nnodes = 0;

        for (int inode = 0; inode < num_nodes; inode++) {
            std::unordered_set<int> node_sds;

            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(node_sd_cols[jp]);
            }

            int nsd = (int)node_sds.size();
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = INTERIOR;

                I_nnodes += 1;
                IE_nnodes += 1;
                IEV_nnodes += 1;
            } else if (nsd == 2) {
                node_class_ind[inode] = EDGE;

                lam_nnodes += 1;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = VERTEX;

                Vc_nnodes += 1;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }
    }

    void build_IEV_nodes() {
        IEV_sd_ptr.assign(num_subdomains + 1, 0);
        IEV_sd_ind.assign(IEV_nnodes, -1);
        IEV_nodes.assign(IEV_nnodes, -1);

        int IEV_ind = 0;
        std::vector<int> temp_completion(num_nodes, 0);

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::fill(temp_completion.begin(), temp_completion.end(), 0);

            IEV_sd_ptr[isd + 1] = IEV_sd_ptr[isd];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != isd) continue;

                const int *conn = &elem_conn[nodes_per_elem * ielem];

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = conn[lnode];

                    if (temp_completion[gnode]) continue;

                    if (IEV_ind >= IEV_nnodes) {
                        printf("ERROR: IEV_ind overflow %d >= %d\n", IEV_ind, IEV_nnodes);
                        std::exit(1);
                    }

                    IEV_nodes[IEV_ind] = gnode;
                    IEV_sd_ind[IEV_ind] = isd;
                    IEV_sd_ptr[isd + 1]++;
                    IEV_ind++;

                    temp_completion[gnode] = 1;
                }
            }
        }

        if (IEV_ind != IEV_nnodes) {
            printf("ERROR: IEV node count mismatch: built %d expected %d\n", IEV_ind, IEV_nnodes);
            std::exit(1);
        }
    }

    void build_IEV_elem_conn() {
        IEV_elem_conn.assign(num_elements * nodes_per_elem, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];
                int iev = -1;

                for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        iev = jp;
                        break;
                    }
                }

                if (iev < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d, gnode %d\n",
                           ielem, gnode);
                    std::exit(1);
                }

                IEV_elem_conn[nodes_per_elem * ielem + lnode] = iev;
            }
        }
    }

    void check_all_elements_assigned() {
        for (int ielem = 0; ielem < num_elements; ielem++) {
            if (elem_sd_ind[ielem] < 0) {
                printf("ERROR: elem %d was not assigned to a subdomain\n", ielem);
                std::exit(1);
            }
        }
    }
};