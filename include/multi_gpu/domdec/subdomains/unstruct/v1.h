#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <unordered_set>
#include <vector>

class UnstructSplitterV1 {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 0;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    UnstructSplitterV1(int num_elements_, int num_nodes_, int nodes_per_elem_,
                       const int *elem_conn_, int target_sd_size_ = 16)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_) {
        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) die("bad sizes");
        if (num_elements > 0 && elem_conn == nullptr) die("elem_conn is null");
        if (target_sd_size <= 0) target_sd_size = 1;
    }

    void split(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
               const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols) {
        assign_initial_subdomains_legacy(ne_ptr, ne_cols, ee_ptr, ee_cols);
        merge_small_subdomains_legacy(ee_ptr, ee_cols);
        compact_subdomain_ids_legacy();
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[IEVSplitterV1]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];

        if (gnode < 0 || gnode >= num_nodes) {
            printf(
                "ERROR[IEVSplitterV1]: invalid gnode %d at elem %d lnode %d "
                "(num_nodes=%d)\n",
                gnode, ielem, lnode, num_nodes);
            std::exit(1);
        }

        return gnode;
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
};