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
    int target_sd_size = 0;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 12;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;

    std::vector<int> elem_sd_ind;

    int node_elem_nnz = 0;
    std::vector<int> node_elem_rowp;
    std::vector<int> node_elem_ct;
    std::vector<int> node_sd_cols;

    std::vector<int> node_class_ind;
    std::vector<int> node_nsd;

    int I_nnodes = 0;
    int IE_nnodes = 0;
    int IEV_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;

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

    void setup_unstructured_subdomains() {
        clear();

        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) die("bad sizes");
        if (num_elements > 0 && elem_conn == nullptr) die("elem_conn is null");
        if (target_sd_size <= 0) target_sd_size = 1;

        std::vector<int> ne_ptr, ne_cols;
        build_node_element_adjacency_legacy(ne_ptr, ne_cols);

        std::vector<int> ee_ptr, ee_cols;
        build_element_element_adjacency_legacy(ne_ptr, ne_cols, ee_ptr, ee_cols);

        assign_initial_subdomains_legacy(ne_ptr, ne_cols, ee_ptr, ee_cols);
        merge_small_subdomains_legacy(ee_ptr, ee_cols);
        compact_subdomain_ids_legacy();

        build_node_subdomain_incidence_legacy();
        classify_nodes_legacy();
        compute_max_vertices_per_subdomain_legacy();

        build_IEV_nodes_legacy();
        build_IEV_elem_conn_legacy();

        printf("UnstructuredIEVSplitting complete:\n");
        printf("  num_subdomains = %d\n", num_subdomains);
        // printf("  I_nnodes       = %d\n", I_nnodes);
        // printf("  IE_nnodes      = %d\n", IE_nnodes);
        printf("  IEV_nnodes     = %d\n", IEV_nnodes);
        printf("  Vc_nnodes      = %d\n", Vc_nnodes);
        // printf("  V_nnodes       = %d\n", V_nnodes);
        // printf("  lam_nnodes     = %d\n", lam_nnodes);
        printf("  MAX_NUM_VERTEX_PER_SUBDOMAIN = %d\n", MAX_NUM_VERTEX_PER_SUBDOMAIN);

        // printf("elem_sd_ind: ");
        // printVec<int>(elem_sd_ind.size(), elem_sd_ind.data());
        // printf("node_class_ind: ");
        // printVec<int>(node_class_ind.size(), node_class_ind.data());
        // printf("node_nsd: ");
        // printVec<int>(node_nsd.size(), node_nsd.data());
        // printf("IEV_sd_ptr: ");
        // printVec<int>(IEV_sd_ptr.size(), IEV_sd_ptr.data());
        // printf("IEV_sd_ind: ");
        // printVec<int>(IEV_sd_ind.size(), IEV_sd_ind.data());
        // printf("IEV_nodes: ");
        // printVec<int>(IEV_nodes.size(), IEV_nodes.data());
        // printf("IEV_elem_conn: ");
        // printVec<int>(IEV_elem_conn.size(), IEV_elem_conn.data());
    }

    void build_node_element_adjacency_legacy(std::vector<int> &ne_ptr, std::vector<int> &ne_cols) {
        int ne_nnz = 0;
        std::vector<int> ne_cts(num_nodes, 0);

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
        ne_cols.assign(ne_nnz, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int offset = ne_ptr[gnode] + ne_cts[gnode];
                ne_cols[offset] = ielem;
                ne_cts[gnode]++;
            }
        }
    }

    void build_element_element_adjacency_legacy(const std::vector<int> &ne_ptr,
                                                const std::vector<int> &ne_cols,
                                                std::vector<int> &ee_ptr,
                                                std::vector<int> &ee_cols) {
        std::vector<int> ee_cts(num_elements, 0);
        int ee_nnz = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;
                    ee_cts[ielem]++;
                    ee_nnz++;
                }
            }
        }

        ee_ptr.assign(num_elements + 1, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            ee_ptr[ielem + 1] = ee_ptr[ielem] + ee_cts[ielem];
        }

        std::fill(ee_cts.begin(), ee_cts.end(), 0);
        ee_cols.assign(ee_nnz, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;

                    int offset = ee_ptr[ielem] + ee_cts[ielem];
                    ee_cols[offset] = jelem;
                    ee_cts[ielem]++;
                }
            }
        }
    }

    void assign_initial_subdomains_legacy(const std::vector<int> &ne_ptr,
                                          const std::vector<int> &ne_cols,
                                          const std::vector<int> &ee_ptr,
                                          const std::vector<int> &ee_cols) {
        elem_sd_ind.assign(num_elements, 0);
        std::vector<bool> visited(num_elements, false);

        int subdomain_ind = 0;
        bool all_visited = false;

        while (!all_visited) {
            int elem = -1;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (!visited[ielem]) {
                    elem = ielem;
                    break;
                }
            }

            if (elem == -1) break;

            std::vector<int> sd_elems;
            sd_elems.push_back(elem);
            elem_sd_ind[elem] = subdomain_ind;
            visited[elem] = true;

            while ((int)sd_elems.size() < target_sd_size) {
                std::unordered_set<int> frontier_set;

                for (auto e : sd_elems) {
                    for (int jp = ee_ptr[e]; jp < ee_ptr[e + 1]; jp++) {
                        int nbr = ee_cols[jp];
                        if (visited[nbr]) continue;
                        frontier_set.insert(nbr);
                    }
                }

                std::vector<int> frontier(frontier_set.begin(), frontier_set.end());
                if (frontier.size() == 0) break;

                int nfrontier = (int)frontier.size();

                std::vector<int> candidate_corner_count(nfrontier, 0);
                std::vector<int> candidate_corner_delta(nfrontier, 0);

                int ii = 0;
                for (auto frontier_elem : frontier) {
                    std::vector<int> proposed_sd_elems;
                    proposed_sd_elems.push_back(frontier_elem);
                    for (auto e : sd_elems) proposed_sd_elems.push_back(e);

                    std::unordered_set<int> proposed_sd_nodes;
                    for (auto sd_elem : proposed_sd_elems) {
                        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                            int gnode = checked_node(sd_elem, lnode);
                            proposed_sd_nodes.insert(gnode);
                        }
                    }

                    int total_corners = 0;
                    int added_corners = 0;

                    for (auto gnode : proposed_sd_nodes) {
                        int nelems_in_subdomain = 0;
                        bool candidate_elem_contains_node = false;

                        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                            int jelem = ne_cols[jp];

                            if (jelem == frontier_elem) {
                                candidate_elem_contains_node = true;
                            }

                            for (auto e : proposed_sd_elems) {
                                if (jelem == e) {
                                    nelems_in_subdomain++;
                                    break;
                                }
                            }
                        }

                        if (nelems_in_subdomain == 1) {
                            total_corners++;
                            if (candidate_elem_contains_node) {
                                added_corners++;
                            }
                        }
                    }

                    candidate_corner_count[ii] = total_corners;
                    candidate_corner_delta[ii] = added_corners;
                    ii++;
                }

                std::vector<std::tuple<int, int, int>> frontier_candidates;
                frontier_candidates.reserve(nfrontier);

                for (int i = 0; i < nfrontier; i++) {
                    frontier_candidates.emplace_back(frontier[i], candidate_corner_count[i],
                                                     candidate_corner_delta[i]);
                }

                std::sort(
                    frontier_candidates.begin(), frontier_candidates.end(),
                    [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); });

                for (int i = 0; i < nfrontier; i++) {
                    frontier[i] = std::get<0>(frontier_candidates[i]);
                    candidate_corner_count[i] = std::get<1>(frontier_candidates[i]);
                    candidate_corner_delta[i] = std::get<2>(frontier_candidates[i]);
                }

                bool added_any = false;

                for (int i = 0; i < nfrontier; i++) {
                    int frontier_elem = frontier[i];

                    if ((int)sd_elems.size() >= target_sd_size) break;
                    if (visited[frontier_elem]) continue;
                    if (candidate_corner_delta[i] > 2) continue;

                    sd_elems.push_back(frontier_elem);
                    elem_sd_ind[frontier_elem] = subdomain_ind;
                    visited[frontier_elem] = true;
                    added_any = true;
                }

                if (!added_any) break;
            }

            subdomain_ind++;

            all_visited = true;
            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (!visited[ielem]) {
                    all_visited = false;
                    break;
                }
            }
        }

        num_subdomains = subdomain_ind;
    }

    void merge_small_subdomains_legacy(const std::vector<int> &ee_ptr,
                                       const std::vector<int> &ee_cols) {
        int n_subdomain = num_subdomains;
        if (n_subdomain <= 1) return;

        std::vector<int> subdomain_cts(n_subdomain, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            subdomain_cts[isd]++;
        }

        std::vector<int> subdomain_ptr(n_subdomain + 1, 0);
        for (int isd = 0; isd < n_subdomain; isd++) {
            subdomain_ptr[isd + 1] = subdomain_ptr[isd] + subdomain_cts[isd];
        }

        std::fill(subdomain_cts.begin(), subdomain_cts.end(), 0);
        std::vector<int> subdomain_cols(num_elements, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            int offset = subdomain_ptr[isd] + subdomain_cts[isd];
            subdomain_cols[offset] = ielem;
            subdomain_cts[isd]++;
        }

        for (int isd = 0; isd < n_subdomain; isd++) {
            int nelems_sd = subdomain_cts[isd];

            if (nelems_sd < target_sd_size) {
                std::vector<int> sd_elems;
                for (int jp = subdomain_ptr[isd]; jp < subdomain_ptr[isd + 1]; jp++) {
                    sd_elems.push_back(subdomain_cols[jp]);
                }

                std::unordered_set<int> adj_subdomains_set;

                for (auto e : sd_elems) {
                    for (int jp = ee_ptr[e]; jp < ee_ptr[e + 1]; jp++) {
                        int jelem = ee_cols[jp];
                        int jsd = elem_sd_ind[jelem];
                        if (jsd == isd) continue;
                        adj_subdomains_set.insert(jsd);
                    }
                }

                std::vector<int> adj_subdomains(adj_subdomains_set.begin(),
                                                adj_subdomains_set.end());

                if (adj_subdomains.size() == 0) continue;

                int jsd = adj_subdomains[0];

                for (auto e : sd_elems) {
                    elem_sd_ind[e] = jsd;
                }
            }
        }
    }

    void compact_subdomain_ids_legacy() {
        int old_n_subdomain = num_subdomains;

        std::unordered_set<int> sd_set;
        for (int ielem = 0; ielem < num_elements; ielem++) {
            sd_set.insert(elem_sd_ind[ielem]);
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        std::vector<int> sd_imap(old_n_subdomain, -1);

        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];
            sd_imap[old_isd] = new_isd;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int old_isd = elem_sd_ind[ielem];
            elem_sd_ind[ielem] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }

    void build_node_subdomain_incidence_legacy() {
        node_elem_nnz = 0;
        node_elem_rowp.assign(num_nodes + 1, 0);
        node_elem_ct.assign(num_nodes, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                node_elem_ct[gnode]++;
                node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            node_elem_rowp[inode + 1] = node_elem_rowp[inode] + node_elem_ct[inode];
        }

        std::vector<int> temp_node_elem(num_nodes, 0);
        node_sd_cols.assign(node_elem_nnz, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = isd;
                temp_node_elem[gnode]++;
            }
        }
    }

    void classify_nodes_legacy() {
        node_class_ind.assign(num_nodes, 0);
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
                I_nnodes++;
                IE_nnodes++;
                IEV_nnodes++;
            } else if (nsd == 2) {
                node_class_ind[inode] = IEV_EDGE;
                lam_nnodes++;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = IEV_VERTEX;
                Vc_nnodes++;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }
    }

    void compute_max_vertices_per_subdomain_legacy() {
        std::vector<int> nvertex(num_subdomains, 0);
        MAX_NUM_VERTEX_PER_SUBDOMAIN = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int node_class = node_class_ind[gnode];

                if (node_class == IEV_VERTEX) {
                    nvertex[isd]++;
                }
            }
        }

        for (int isd = 0; isd < num_subdomains; isd++) {
            MAX_NUM_VERTEX_PER_SUBDOMAIN = std::max(MAX_NUM_VERTEX_PER_SUBDOMAIN, nvertex[isd]);
        }
    }

    void build_IEV_nodes_legacy() {
        IEV_sd_ptr.assign(num_subdomains + 1, 0);
        IEV_sd_ind.assign(IEV_nnodes, 0);
        IEV_nodes.assign(IEV_nnodes, 0);

        int IEV_ind = 0;
        std::vector<int> temp_completion(num_nodes, 0);

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::fill(temp_completion.begin(), temp_completion.end(), 0);
            IEV_sd_ptr[isd + 1] = IEV_sd_ptr[isd];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != isd) continue;

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = checked_node(ielem, lnode);

                    if (temp_completion[gnode]) continue;

                    if (IEV_ind >= IEV_nnodes) {
                        die("IEV_ind exceeded IEV_nnodes in build_IEV_nodes_legacy");
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
            printf("ERROR[UnstructuredIEVSplitting]: IEV_ind %d != IEV_nnodes %d\n", IEV_ind,
                   IEV_nnodes);
            std::exit(1);
        }
    }

    void build_IEV_elem_conn_legacy() {
        IEV_elem_conn.assign(num_elements * nodes_per_elem, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf(
                        "ERROR[UnstructuredIEVSplitting]: failed to find duplicated IEV node "
                        "for elem %d, isd %d, gnode %d\n",
                        ielem, isd, gnode);
                    std::exit(1);
                }

                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }
    }
};