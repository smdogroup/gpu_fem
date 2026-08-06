#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <vector>

class CornerSeparatingSplitter {
   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    CornerSeparatingSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
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
        std::vector<std::vector<int>> elem_adj;
        build_unique_element_adjacency(ee_ptr, ee_cols, elem_adj);

        assign_initial_subdomains(ne_ptr, ne_cols, elem_adj);
        merge_small_subdomains(elem_adj);
        compact_subdomain_ids();

        int remaining = count_all_corner_violations(ne_ptr, ne_cols);
        if (remaining > 0) {
            printf("WARNING[CornerSeparatingSplitter]: %d corner violations remain after split\n",
                   remaining);
        }
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[CornerSeparatingSplitter]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf("ERROR[CornerSeparatingSplitter]: bad node %d elem %d lnode %d\n", gnode, ielem,
                   lnode);
            std::exit(1);
        }
        return gnode;
    }

    void build_unique_element_adjacency(const std::vector<int> &ee_ptr,
                                        const std::vector<int> &ee_cols,
                                        std::vector<std::vector<int>> &elem_adj) const {
        elem_adj.assign(num_elements, std::vector<int>());

        for (int ielem = 0; ielem < num_elements; ielem++) {
            std::vector<int> nbrs;

            for (int jp = ee_ptr[ielem]; jp < ee_ptr[ielem + 1]; jp++) {
                int jelem = ee_cols[jp];

                if (jelem < 0 || jelem >= num_elements) {
                    printf("ERROR[CornerSeparatingSplitter]: bad ee elem=%d nbr=%d\n", ielem,
                           jelem);
                    std::exit(1);
                }

                if (jelem != ielem) nbrs.push_back(jelem);
            }

            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
            elem_adj[ielem] = std::move(nbrs);
        }
    }

    int count_sd_elems_at_node(int gnode, int isd, const std::vector<int> &ne_ptr,
                               const std::vector<int> &ne_cols) const {
        int count = 0;
        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
            int e = ne_cols[jp];
            if (elem_sd_ind[e] == isd) count++;
        }
        return count;
    }

    bool is_corner_node(int gnode, int isd, const std::vector<int> &ne_ptr,
                        const std::vector<int> &ne_cols) const {
        return count_sd_elems_at_node(gnode, isd, ne_ptr, ne_cols) == 1;
    }

    int count_corner_violations_in_sd(int isd, const std::vector<int> &sd_elems,
                                      const std::vector<int> &ne_ptr,
                                      const std::vector<int> &ne_cols) const {
        int nviol = 0;

        for (int e : sd_elems) {
            if (elem_sd_ind[e] != isd) continue;

            int ncorners_in_elem = 0;

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(e, lnode);

                if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) {
                    ncorners_in_elem++;
                }
            }

            if (ncorners_in_elem >= 2) {
                nviol++;
            }
        }

        return nviol;
    }

    void collect_violation_repair_candidates(int isd, const std::vector<int> &sd_elems,
                                             const std::vector<int> &ne_ptr,
                                             const std::vector<int> &ne_cols,
                                             std::vector<int> &candidates) const {
        candidates.clear();

        for (int e : sd_elems) {
            if (elem_sd_ind[e] != isd) continue;

            std::vector<int> corner_nodes;

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(e, lnode);

                if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) {
                    corner_nodes.push_back(gnode);
                }
            }

            if ((int)corner_nodes.size() < 2) continue;

            for (int gnode : corner_nodes) {
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int eadd = ne_cols[jp];

                    if (elem_sd_ind[eadd] < 0) {
                        candidates.push_back(eadd);
                    }
                }
            }
        }

        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    }

    int choose_corner_repair_elem(int isd, const std::vector<int> &sd_elems,
                                  const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols) {
        std::vector<int> candidates;
        collect_violation_repair_candidates(isd, sd_elems, ne_ptr, ne_cols, candidates);

        if (candidates.empty()) return -1;

        int old_viol = count_corner_violations_in_sd(isd, sd_elems, ne_ptr, ne_cols);

        int best_elem = -1;
        int best_viol = std::numeric_limits<int>::max();

        for (int eadd : candidates) {
            elem_sd_ind[eadd] = isd;

            std::vector<int> trial_elems = sd_elems;
            trial_elems.push_back(eadd);

            int new_viol = count_corner_violations_in_sd(isd, trial_elems, ne_ptr, ne_cols);

            elem_sd_ind[eadd] = -1;

            if (new_viol < best_viol) {
                best_viol = new_viol;
                best_elem = eadd;
            }
        }

        if (best_viol > old_viol) return -1;

        return best_elem;
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int touch = 0;
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) touch++;
        }
        return touch;
    }

    int choose_frontier_elem(const std::vector<int> &frontier, int isd,
                             const std::vector<std::vector<int>> &elem_adj) const {
        int best_elem = -1;
        int best_score = -1;

        for (int e : frontier) {
            if (elem_sd_ind[e] >= 0) continue;

            int score = count_touching_sd_neighbors(e, isd, elem_adj);

            if (score > best_score) {
                best_score = score;
                best_elem = e;
            }
        }

        return best_elem;
    }

    void add_frontier_from_elem(int elem, const std::vector<std::vector<int>> &elem_adj,
                                std::vector<int> &frontier, std::vector<int> &frontier_flag) {
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] >= 0) continue;

            if (!frontier_flag[nbr]) {
                frontier.push_back(nbr);
                frontier_flag[nbr] = 1;
            }
        }
    }

    void assign_initial_subdomains(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                   const std::vector<std::vector<int>> &elem_adj) {
        elem_sd_ind.assign(num_elements, -1);
        num_subdomains = 0;

        int num_unassigned = num_elements;

        while (num_unassigned > 0) {
            int seed = -1;

            for (int e = 0; e < num_elements; e++) {
                if (elem_sd_ind[e] < 0) {
                    seed = e;
                    break;
                }
            }

            if (seed < 0) break;

            int isd = num_subdomains++;
            int sd_size = 0;

            std::vector<int> sd_elems;
            sd_elems.reserve(target_sd_size + nodes_per_elem);

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            elem_sd_ind[seed] = isd;
            sd_elems.push_back(seed);
            sd_size++;
            num_unassigned--;

            add_frontier_from_elem(seed, elem_adj, frontier, frontier_flag);

            while (sd_size < target_sd_size) {
                int eadd = choose_frontier_elem(frontier, isd, elem_adj);
                if (eadd < 0) break;

                elem_sd_ind[eadd] = isd;
                sd_elems.push_back(eadd);
                sd_size++;
                num_unassigned--;

                add_frontier_from_elem(eadd, elem_adj, frontier, frontier_flag);
            }

            while (true) {
                int nviol = count_corner_violations_in_sd(isd, sd_elems, ne_ptr, ne_cols);
                if (nviol == 0) break;

                int eadd = choose_corner_repair_elem(isd, sd_elems, ne_ptr, ne_cols);
                if (eadd < 0) {
                    printf(
                        "WARNING[CornerSeparatingSplitter]: could not repair %d corner "
                        "violations in subdomain %d\n",
                        nviol, isd);
                    break;
                }

                elem_sd_ind[eadd] = isd;
                sd_elems.push_back(eadd);
                num_unassigned--;

                add_frontier_from_elem(eadd, elem_adj, frontier, frontier_flag);
            }
        }
    }

    int count_all_corner_violations(const std::vector<int> &ne_ptr,
                                    const std::vector<int> &ne_cols) const {
        int total = 0;

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::vector<int> sd_elems;

            for (int e = 0; e < num_elements; e++) {
                if (elem_sd_ind[e] == isd) {
                    sd_elems.push_back(e);
                }
            }

            total += count_corner_violations_in_sd(isd, sd_elems, ne_ptr, ne_cols);
        }

        return total;
    }

    void merge_small_subdomains(const std::vector<std::vector<int>> &elem_adj) {
        if (num_subdomains <= 1) return;

        std::vector<int> sd_size(num_subdomains, 0);

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] >= 0) sd_size[elem_sd_ind[e]]++;
        }

        int min_size = std::max(1, target_sd_size / 2);

        for (int isd = 0; isd < num_subdomains; isd++) {
            if (sd_size[isd] >= min_size) continue;

            std::vector<int> adj_count(num_subdomains, 0);

            for (int e = 0; e < num_elements; e++) {
                if (elem_sd_ind[e] != isd) continue;

                for (int nbr : elem_adj[e]) {
                    int jsd = elem_sd_ind[nbr];
                    if (jsd >= 0 && jsd != isd) adj_count[jsd]++;
                }
            }

            int best_jsd = -1;
            int best_touch = -1;

            for (int jsd = 0; jsd < num_subdomains; jsd++) {
                if (jsd == isd) continue;

                if (adj_count[jsd] > best_touch) {
                    best_touch = adj_count[jsd];
                    best_jsd = jsd;
                }
            }

            if (best_jsd < 0) continue;

            for (int e = 0; e < num_elements; e++) {
                if (elem_sd_ind[e] == isd) {
                    elem_sd_ind[e] = best_jsd;
                    sd_size[best_jsd]++;
                    sd_size[isd]--;
                }
            }
        }
    }

    void compact_subdomain_ids() {
        std::unordered_set<int> sd_set;

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] >= 0) sd_set.insert(elem_sd_ind[e]);
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        std::vector<int> sd_imap(num_subdomains, -1);

        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];
            sd_imap[old_isd] = new_isd;
        }

        for (int e = 0; e < num_elements; e++) {
            int old_isd = elem_sd_ind[e];

            if (old_isd < 0 || old_isd >= num_subdomains || sd_imap[old_isd] < 0) {
                printf("ERROR[CornerSeparatingSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};