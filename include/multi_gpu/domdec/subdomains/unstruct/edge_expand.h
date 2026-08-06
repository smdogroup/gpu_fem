#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <vector>

class UnstructEdgeFillSplitter {
   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    UnstructEdgeFillSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
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

        assign_initial_subdomains_edgefill(ne_ptr, ne_cols, elem_adj);
        merge_small_subdomains(elem_adj);
        compact_subdomain_ids();
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[UnstructEdgeFillSplitter]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf("ERROR[UnstructEdgeFillSplitter]: bad node %d elem %d lnode %d\n", gnode, ielem,
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
                    printf("ERROR[UnstructEdgeFillSplitter]: bad ee elem=%d nbr=%d\n", ielem,
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

    void collect_edge_expansion(int gnode, int isd, const std::vector<int> &ne_ptr,
                                const std::vector<int> &ne_cols,
                                std::vector<int> &add_elems) const {
        add_elems.clear();

        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
            int e = ne_cols[jp];

            if (elem_sd_ind[e] < 0) {
                add_elems.push_back(e);
            }
        }

        std::sort(add_elems.begin(), add_elems.end());
        add_elems.erase(std::unique(add_elems.begin(), add_elems.end()), add_elems.end());
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int touch = 0;

        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) touch++;
        }

        return touch;
    }

    int choose_fallback_elem(const std::vector<int> &frontier, int isd,
                             const std::vector<std::vector<int>> &elem_adj) const {
        int best_elem = -1;
        int best_score = -1;

        for (int e : frontier) {
            if (elem_sd_ind[e] >= 0) continue;

            int touch = count_touching_sd_neighbors(e, isd, elem_adj);
            int score = touch;

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

    void assign_initial_subdomains_edgefill(const std::vector<int> &ne_ptr,
                                            const std::vector<int> &ne_cols,
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
            sd_elems.reserve(target_sd_size);

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            elem_sd_ind[seed] = isd;
            sd_elems.push_back(seed);
            sd_size++;
            num_unassigned--;

            add_frontier_from_elem(seed, elem_adj, frontier, frontier_flag);

            while (sd_size < target_sd_size) {
                int best_node = -1;
                int best_nadd = -1;
                std::vector<int> best_add;

                // Edge expansion move:
                // A candidate edge node has exactly two elements already in this subdomain.
                // Boundary nodes with no unassigned incident elements are ignored.
                for (int e : sd_elems) {
                    for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                        int gnode = checked_node(e, lnode);

                        int n_sd_at_node = count_sd_elems_at_node(gnode, isd, ne_ptr, ne_cols);
                        if (n_sd_at_node != 2) continue;

                        std::vector<int> add_elems;
                        collect_edge_expansion(gnode, isd, ne_ptr, ne_cols, add_elems);

                        if (add_elems.empty()) continue;
                        if (sd_size + (int)add_elems.size() > target_sd_size) continue;

                        if ((int)add_elems.size() > best_nadd) {
                            best_nadd = (int)add_elems.size();
                            best_node = gnode;
                            best_add = std::move(add_elems);
                        }
                    }
                }

                if (best_node >= 0) {
                    for (int eadd : best_add) {
                        if (elem_sd_ind[eadd] >= 0) continue;

                        elem_sd_ind[eadd] = isd;
                        sd_elems.push_back(eadd);
                        sd_size++;
                        num_unassigned--;

                        add_frontier_from_elem(eadd, elem_adj, frontier, frontier_flag);

                        if (sd_size >= target_sd_size) break;
                    }

                    continue;
                }

                // Fallback: no valid edge expansion exists, so add one best frontier element.
                int fallback = choose_fallback_elem(frontier, isd, elem_adj);

                if (fallback < 0) break;

                elem_sd_ind[fallback] = isd;
                sd_elems.push_back(fallback);
                sd_size++;
                num_unassigned--;

                add_frontier_from_elem(fallback, elem_adj, frontier, frontier_flag);
            }
        }
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
                printf("ERROR[UnstructEdgeFillSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};