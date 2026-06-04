#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "domdec/subdomains/_iev.h"

class UnstructuredIEVSplitting {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 16;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 20;  // update this?

    const int *elem_conn = nullptr;

    int num_subdomains = 0;

    // element -> subdomain
    std::vector<int> elem_sd_ind;

    // node -> subdomain incidence
    int node_elem_nnz = 0;
    std::vector<int> node_elem_rowp;
    std::vector<int> node_elem_ct;
    std::vector<int> node_sd_cols;

    // node classes
    std::vector<int> node_class_ind;
    std::vector<int> node_nsd;

    int I_nnodes = 0;
    int IE_nnodes = 0;
    int IEV_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;

    // duplicated IEV layout
    std::vector<int> IEV_sd_ptr;
    std::vector<int> IEV_sd_ind;
    std::vector<int> IEV_nodes;
    std::vector<int> IEV_elem_conn;

    UnstructuredIEVSplitting(int num_elements_, int num_nodes_, int nodes_per_elem_,
                             const int *elem_conn_, int target_sd_size_ = 16)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_) {
        setup_unstructured_subdomains();
    }

    void clear() {
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
    void setup_unstructured_subdomains() {
        clear();

        std::vector<int> ne_ptr;
        std::vector<int> ne_cols;
        build_node_element_adjacency(ne_ptr, ne_cols);

        std::vector<int> ee_ptr;
        std::vector<int> ee_cols;
        build_element_element_adjacency(ne_ptr, ne_cols, ee_ptr, ee_cols);

        assign_initial_subdomains(ne_ptr, ne_cols, ee_ptr, ee_cols);

        merge_small_subdomains(ee_ptr, ee_cols);

        compact_subdomain_ids();

        build_node_subdomain_incidence();

        classify_nodes();

        build_IEV_nodes();

        build_IEV_elem_conn();

        printf("UnstructuredIEVSplitting complete:\n");
        printf("  num_subdomains = %d\n", num_subdomains);
        printf("  I_nnodes       = %d\n", I_nnodes);
        printf("  IE_nnodes      = %d\n", IE_nnodes);
        printf("  IEV_nnodes     = %d\n", IEV_nnodes);
        printf("  Vc_nnodes      = %d\n", Vc_nnodes);
        printf("  V_nnodes       = %d\n", V_nnodes);
        printf("  lam_nnodes     = %d\n", lam_nnodes);
    }

    void build_node_element_adjacency(std::vector<int> &ne_ptr, std::vector<int> &ne_cols) {
        std::vector<int> ne_cts(num_nodes, 0);
        int ne_nnz = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];
                ne_cts[gnode]++;
                ne_nnz++;
            }
        }

        ne_ptr.assign(num_nodes + 1, 0);
        for (int inode = 0; inode < num_nodes; inode++) {
            ne_ptr[inode + 1] = ne_ptr[inode] + ne_cts[inode];
        }

        std::fill(ne_cts.begin(), ne_cts.end(), 0);
        ne_cols.assign(ne_nnz, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];
                int offset = ne_ptr[gnode] + ne_cts[gnode];
                ne_cols[offset] = ielem;
                ne_cts[gnode]++;
            }
        }
    }

    void build_element_element_adjacency(const std::vector<int> &ne_ptr,
                                         const std::vector<int> &ne_cols, std::vector<int> &ee_ptr,
                                         std::vector<int> &ee_cols) {
        std::vector<std::vector<int>> adj(num_elements);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            std::unordered_set<int> nbrs;

            const int *conn = &elem_conn[nodes_per_elem * ielem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = conn[lnode];

                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;
                    nbrs.insert(jelem);
                }
            }

            adj[ielem].assign(nbrs.begin(), nbrs.end());
            std::sort(adj[ielem].begin(), adj[ielem].end());
        }

        ee_ptr.assign(num_elements + 1, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            ee_ptr[ielem + 1] = ee_ptr[ielem] + (int)adj[ielem].size();
        }

        ee_cols.assign(ee_ptr[num_elements], -1);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int offset = ee_ptr[ielem];
            for (int j = 0; j < (int)adj[ielem].size(); j++) {
                ee_cols[offset + j] = adj[ielem][j];
            }
        }
    }

    void assign_initial_subdomains(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                   const std::vector<int> &ee_ptr,
                                   const std::vector<int> &ee_cols) {
        elem_sd_ind.assign(num_elements, 0);
        std::vector<char> visited(num_elements, 0);

        int isd = 0;

        while (true) {
            int seed_elem = -1;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (!visited[ielem]) {
                    seed_elem = ielem;
                    break;
                }
            }

            if (seed_elem < 0) break;

            std::vector<int> sd_elems;
            sd_elems.push_back(seed_elem);
            elem_sd_ind[seed_elem] = isd;
            visited[seed_elem] = 1;

            while ((int)sd_elems.size() < target_sd_size) {
                std::unordered_set<int> frontier_set;

                for (int elem : sd_elems) {
                    for (int jp = ee_ptr[elem]; jp < ee_ptr[elem + 1]; jp++) {
                        int nbr = ee_cols[jp];
                        if (!visited[nbr]) {
                            frontier_set.insert(nbr);
                        }
                    }
                }

                if (frontier_set.empty()) break;

                std::vector<int> frontier(frontier_set.begin(), frontier_set.end());

                std::vector<std::tuple<int, int, int>> candidates;
                candidates.reserve(frontier.size());

                for (int frontier_elem : frontier) {
                    int total_corners = 0;
                    int added_corners = 0;

                    score_candidate_corner_count(frontier_elem, sd_elems, ne_ptr, ne_cols,
                                                 total_corners, added_corners);

                    candidates.emplace_back(frontier_elem, total_corners, added_corners);
                }

                std::sort(
                    candidates.begin(), candidates.end(),
                    [](const std::tuple<int, int, int> &a, const std::tuple<int, int, int> &b) {
                        if (std::get<1>(a) != std::get<1>(b)) {
                            return std::get<1>(a) < std::get<1>(b);
                        }
                        return std::get<2>(a) < std::get<2>(b);
                    });

                bool added_any = false;

                for (const auto &cand : candidates) {
                    int elem = std::get<0>(cand);
                    int added_corners = std::get<2>(cand);

                    if ((int)sd_elems.size() >= target_sd_size) break;
                    if (visited[elem]) continue;

                    if (added_corners > 2) continue;

                    sd_elems.push_back(elem);
                    elem_sd_ind[elem] = isd;
                    visited[elem] = 1;
                    added_any = true;
                }

                if (!added_any) break;
            }

            isd++;
        }

        num_subdomains = isd;
    }

    void score_candidate_corner_count(int candidate_elem, const std::vector<int> &sd_elems,
                                      const std::vector<int> &ne_ptr,
                                      const std::vector<int> &ne_cols, int &total_corners,
                                      int &added_corners) const {
        std::vector<int> proposed = sd_elems;
        proposed.push_back(candidate_elem);

        std::unordered_set<int> proposed_nodes;
        for (int elem : proposed) {
            const int *conn = &elem_conn[nodes_per_elem * elem];
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                proposed_nodes.insert(conn[lnode]);
            }
        }

        total_corners = 0;
        added_corners = 0;

        for (int gnode : proposed_nodes) {
            int nelems_in_subdomain = 0;
            bool candidate_contains_node = false;

            for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                int jelem = ne_cols[jp];

                if (jelem == candidate_elem) {
                    candidate_contains_node = true;
                }

                for (int elem : proposed) {
                    if (jelem == elem) {
                        nelems_in_subdomain++;
                        break;
                    }
                }
            }

            if (nelems_in_subdomain == 1) {
                total_corners++;
                if (candidate_contains_node) {
                    added_corners++;
                }
            }
        }
    }

    void merge_small_subdomains(const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols) {
        int old_num_subdomains = num_subdomains;

        std::vector<int> sd_cts(old_num_subdomains, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            sd_cts[elem_sd_ind[ielem]]++;
        }

        std::vector<int> sd_ptr(old_num_subdomains + 1, 0);
        for (int isd = 0; isd < old_num_subdomains; isd++) {
            sd_ptr[isd + 1] = sd_ptr[isd] + sd_cts[isd];
        }

        std::vector<int> temp_cts(old_num_subdomains, 0);
        std::vector<int> sd_cols(num_elements, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            int offset = sd_ptr[isd] + temp_cts[isd];
            sd_cols[offset] = ielem;
            temp_cts[isd]++;
        }

        for (int isd = 0; isd < old_num_subdomains; isd++) {
            if (sd_cts[isd] >= target_sd_size) continue;

            std::unordered_set<int> adj_sds;

            for (int jp = sd_ptr[isd]; jp < sd_ptr[isd + 1]; jp++) {
                int ielem = sd_cols[jp];

                for (int kp = ee_ptr[ielem]; kp < ee_ptr[ielem + 1]; kp++) {
                    int jelem = ee_cols[kp];
                    int jsd = elem_sd_ind[jelem];

                    if (jsd != isd) {
                        adj_sds.insert(jsd);
                    }
                }
            }

            if (adj_sds.empty()) continue;

            int best_jsd = *adj_sds.begin();

            for (int jp = sd_ptr[isd]; jp < sd_ptr[isd + 1]; jp++) {
                int ielem = sd_cols[jp];
                elem_sd_ind[ielem] = best_jsd;
            }
        }
    }

    void compact_subdomain_ids() {
        std::unordered_set<int> sd_set;
        for (int ielem = 0; ielem < num_elements; ielem++) {
            sd_set.insert(elem_sd_ind[ielem]);
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        int old_max = sd_vec.empty() ? -1 : sd_vec.back();
        std::vector<int> sd_imap(old_max + 1, -1);

        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];
            sd_imap[old_isd] = new_isd;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            elem_sd_ind[ielem] = sd_imap[elem_sd_ind[ielem]];
        }

        num_subdomains = (int)sd_vec.size();
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
                node_class_ind[inode] = IEV_INTERIOR;

                I_nnodes += 1;
                IE_nnodes += 1;
                IEV_nnodes += 1;
            } else if (nsd == 2) {
                node_class_ind[inode] = IEV_EDGE;

                lam_nnodes += 1;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = IEV_VERTEX;

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
};