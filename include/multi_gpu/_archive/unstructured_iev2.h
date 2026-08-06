#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "domdec/subdomains/_iev.h"

class UnstructuredIEVSplitting {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 0;
    // int MAX_NUM_VERTEX_PER_SUBDOMAIN = 4;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 12;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;

    // element -> subdomain
    std::vector<int> elem_sd_ind;

    // node -> unique incident subdomains
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
    void die(const char *msg) const {
        printf("ERROR[UnstructuredIEVSplitting]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf(
                "ERROR[UnstructuredIEVSplitting]: invalid gnode %d at elem %d lnode %d "
                "(num_nodes=%d)\n",
                gnode, ielem, lnode, num_nodes);
            std::exit(1);
        }
        return gnode;
    }

    static long long key_isd_node(int isd, int node) {
        return (static_cast<long long>(isd) << 32) ^ static_cast<unsigned int>(node);
    }

    void setup_unstructured_subdomains() {
        clear();

        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) {
            die("bad sizes");
        }
        if (num_elements > 0 && elem_conn == nullptr) {
            die("elem_conn is null");
        }
        if (target_sd_size <= 0) {
            target_sd_size = 1;
        }

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

        printf("elem_sd_ind: ");
        printVec<int>(num_elements, elem_sd_ind.data());
        printf("node_class_ind: ");
        printVec<int>(num_nodes, node_class_ind.data());
        printf("node_nsd: ");
        printVec<int>(num_nodes, node_nsd.data());
        printf("IEV_sd_ptr: ");
        printVec<int>(num_subdomains + 1, IEV_sd_ptr.data());
        printf("IEV_sd_ind: ");
        printVec<int>(IEV_nnodes, IEV_sd_ind.data());
        printf("IEV_nodes: ");
        printVec<int>(IEV_nnodes, IEV_nodes.data());
        printf("IEV_elem_conn: ");
        printVec<int>(4 * num_elements, IEV_elem_conn.data());
    }

    void build_node_element_adjacency(std::vector<int> &ne_ptr, std::vector<int> &ne_cols) {
        std::vector<int> ne_cts(num_nodes, 0);
        int ne_nnz = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
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
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
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

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);

                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem != ielem) {
                        nbrs.insert(jelem);
                    }
                }
            }

            adj[ielem].assign(nbrs.begin(), nbrs.end());
            std::sort(adj[ielem].begin(), adj[ielem].end());
        }

        ee_ptr.assign(num_elements + 1, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            ee_ptr[ielem + 1] = ee_ptr[ielem] + static_cast<int>(adj[ielem].size());
        }

        ee_cols.assign(ee_ptr[num_elements], -1);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int offset = ee_ptr[ielem];
            for (int j = 0; j < static_cast<int>(adj[ielem].size()); j++) {
                ee_cols[offset + j] = adj[ielem][j];
            }
        }
    }

    void assign_initial_subdomains(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                   const std::vector<int> &ee_ptr,
                                   const std::vector<int> &ee_cols) {
        elem_sd_ind.assign(num_elements, -1);
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

            while (static_cast<int>(sd_elems.size()) < target_sd_size) {
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

                std::vector<std::tuple<int, int, int>> candidates;
                candidates.reserve(frontier_set.size());

                for (int frontier_elem : frontier_set) {
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
                        if (std::get<2>(a) != std::get<2>(b)) {
                            return std::get<2>(a) < std::get<2>(b);
                        }
                        return std::get<0>(a) < std::get<0>(b);
                    });

                bool added_any = false;

                for (const auto &cand : candidates) {
                    int elem = std::get<0>(cand);
                    int added_corners = std::get<2>(cand);

                    if (static_cast<int>(sd_elems.size()) >= target_sd_size) break;
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

        std::unordered_set<int> proposed_elem_set;
        proposed_elem_set.reserve(proposed.size() * 2 + 1);
        for (int elem : proposed) {
            proposed_elem_set.insert(elem);
        }

        std::unordered_set<int> proposed_nodes;
        proposed_nodes.reserve(proposed.size() * nodes_per_elem * 2 + 1);

        for (int elem : proposed) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                proposed_nodes.insert(checked_node(elem, lnode));
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

                if (proposed_elem_set.find(jelem) != proposed_elem_set.end()) {
                    nelems_in_subdomain++;
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
        if (num_subdomains <= 1) return;

        bool changed = true;
        int max_passes = 4;

        for (int pass = 0; pass < max_passes && changed; pass++) {
            changed = false;

            std::vector<int> sd_cts(num_subdomains, 0);
            for (int ielem = 0; ielem < num_elements; ielem++) {
                sd_cts[elem_sd_ind[ielem]]++;
            }

            for (int isd = 0; isd < num_subdomains; isd++) {
                if (sd_cts[isd] == 0 || sd_cts[isd] >= target_sd_size) continue;

                std::vector<int> adj_cts(num_subdomains, 0);

                for (int ielem = 0; ielem < num_elements; ielem++) {
                    if (elem_sd_ind[ielem] != isd) continue;

                    for (int kp = ee_ptr[ielem]; kp < ee_ptr[ielem + 1]; kp++) {
                        int jelem = ee_cols[kp];
                        int jsd = elem_sd_ind[jelem];

                        if (jsd != isd) {
                            adj_cts[jsd]++;
                        }
                    }
                }

                int best_jsd = -1;
                int best_score = -1;

                for (int jsd = 0; jsd < num_subdomains; jsd++) {
                    if (jsd == isd) continue;
                    if (adj_cts[jsd] > best_score) {
                        best_score = adj_cts[jsd];
                        best_jsd = jsd;
                    }
                }

                if (best_jsd < 0) continue;

                for (int ielem = 0; ielem < num_elements; ielem++) {
                    if (elem_sd_ind[ielem] == isd) {
                        elem_sd_ind[ielem] = best_jsd;
                    }
                }

                changed = true;
            }

            compact_subdomain_ids();
        }
    }

    void compact_subdomain_ids() {
        std::unordered_set<int> sd_set;
        for (int ielem = 0; ielem < num_elements; ielem++) {
            if (elem_sd_ind[ielem] >= 0) {
                sd_set.insert(elem_sd_ind[ielem]);
            }
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        int old_max = sd_vec.empty() ? -1 : sd_vec.back();
        std::vector<int> sd_imap(old_max + 1, -1);

        for (int new_isd = 0; new_isd < static_cast<int>(sd_vec.size()); new_isd++) {
            sd_imap[sd_vec[new_isd]] = new_isd;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int old_isd = elem_sd_ind[ielem];
            if (old_isd < 0 || old_isd > old_max || sd_imap[old_isd] < 0) {
                die("bad subdomain id during compaction");
            }
            elem_sd_ind[ielem] = sd_imap[old_isd];
        }

        num_subdomains = static_cast<int>(sd_vec.size());
    }

    void build_node_subdomain_incidence() {
        std::vector<std::vector<int>> node_sds(num_nodes);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                node_sds[gnode].push_back(isd);
            }
        }

        node_elem_nnz = 0;
        node_elem_rowp.assign(num_nodes + 1, 0);
        node_elem_ct.assign(num_nodes, 0);

        for (int inode = 0; inode < num_nodes; inode++) {
            auto &v = node_sds[inode];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());

            node_elem_ct[inode] = static_cast<int>(v.size());
            node_elem_nnz += node_elem_ct[inode];
            node_elem_rowp[inode + 1] = node_elem_nnz;
        }

        node_sd_cols.assign(node_elem_nnz, -1);

        for (int inode = 0; inode < num_nodes; inode++) {
            int offset = node_elem_rowp[inode];
            for (int j = 0; j < static_cast<int>(node_sds[inode].size()); j++) {
                node_sd_cols[offset + j] = node_sds[inode][j];
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
            int nsd = node_elem_rowp[inode + 1] - node_elem_rowp[inode];
            node_nsd[inode] = nsd;

            if (nsd == 0) {
                // Unused global node. Do not count it in I/IE/IEV.
                node_class_ind[inode] = INTERIOR;
            } else if (nsd == 1) {
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

        for (int inode = 0; inode < num_nodes; inode++) {
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                int isd = node_sd_cols[jp];
                IEV_sd_ptr[isd + 1]++;
            }
        }

        for (int isd = 0; isd < num_subdomains; isd++) {
            IEV_sd_ptr[isd + 1] += IEV_sd_ptr[isd];
        }

        if (IEV_sd_ptr[num_subdomains] != IEV_nnodes) {
            printf("ERROR[UnstructuredIEVSplitting]: IEV count mismatch from ptrs %d expected %d\n",
                   IEV_sd_ptr[num_subdomains], IEV_nnodes);
            std::exit(1);
        }

        IEV_sd_ind.assign(IEV_nnodes, -1);
        IEV_nodes.assign(IEV_nnodes, -1);

        std::vector<int> temp = IEV_sd_ptr;

        for (int inode = 0; inode < num_nodes; inode++) {
            for (int jp = node_elem_rowp[inode]; jp < node_elem_rowp[inode + 1]; jp++) {
                int isd = node_sd_cols[jp];
                int iev = temp[isd]++;

                IEV_nodes[iev] = inode;
                IEV_sd_ind[iev] = isd;
            }
        }
    }

    void build_IEV_elem_conn() {
        IEV_elem_conn.assign(num_elements * nodes_per_elem, -1);

        std::unordered_map<long long, int> iev_map;
        iev_map.reserve(static_cast<size_t>(IEV_nnodes) * 2 + 1);

        for (int iev = 0; iev < IEV_nnodes; iev++) {
            int isd = IEV_sd_ind[iev];
            int gnode = IEV_nodes[iev];
            iev_map[key_isd_node(isd, gnode)] = iev;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                auto it = iev_map.find(key_isd_node(isd, gnode));

                if (it == iev_map.end()) {
                    printf(
                        "ERROR[UnstructuredIEVSplitting]: failed to find duplicated IEV node "
                        "for elem %d, isd %d, gnode %d\n",
                        ielem, isd, gnode);
                    std::exit(1);
                }

                IEV_elem_conn[nodes_per_elem * ielem + lnode] = it->second;
            }
        }
    }
};