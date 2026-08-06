#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <vector>

class UnstructMacroElementSplitter {
   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    int macro_max_size = 8;

    UnstructMacroElementSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
                                 const int *elem_conn_, int target_sd_size_ = 16,
                                 int macro_max_size_ = 8)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_),
          macro_max_size(macro_max_size_) {
        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) die("bad sizes");
        if (num_elements > 0 && elem_conn == nullptr) die("elem_conn is null");
        if (target_sd_size <= 0) target_sd_size = 1;
        if (macro_max_size <= 0) macro_max_size = 1;
    }

    void split(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
               const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols) {
        std::vector<std::vector<int>> elem_adj;
        build_unique_element_adjacency(ee_ptr, ee_cols, elem_adj);

        std::vector<std::vector<int>> macros;
        build_node_macros(ne_ptr, ne_cols, elem_adj, macros);

        assign_macro_greedy(macros, elem_adj);
        merge_small_subdomains(elem_adj);
        compact_subdomain_ids();
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[UnstructMacroElementSplitter]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf("ERROR[UnstructMacroElementSplitter]: bad node %d elem %d lnode %d\n", gnode,
                   ielem, lnode);
            std::exit(1);
        }
        return gnode;
    }

    void build_unique_element_adjacency(const std::vector<int> &ee_ptr,
                                        const std::vector<int> &ee_cols,
                                        std::vector<std::vector<int>> &elem_adj) const {
        elem_adj.assign(num_elements, std::vector<int>());

        for (int e = 0; e < num_elements; e++) {
            std::vector<int> nbrs;

            for (int jp = ee_ptr[e]; jp < ee_ptr[e + 1]; jp++) {
                int nbr = ee_cols[jp];

                if (nbr < 0 || nbr >= num_elements) {
                    printf("ERROR[UnstructMacroElementSplitter]: bad ee elem=%d nbr=%d\n", e, nbr);
                    std::exit(1);
                }

                if (nbr != e) nbrs.push_back(nbr);
            }

            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
            elem_adj[e] = std::move(nbrs);
        }
    }

    bool are_adjacent(int a, int b, const std::vector<std::vector<int>> &elem_adj) const {
        const std::vector<int> &nbrs = elem_adj[a];
        return std::binary_search(nbrs.begin(), nbrs.end(), b);
    }

    int macro_internal_edges(const std::vector<int> &macro,
                             const std::vector<std::vector<int>> &elem_adj) const {
        int count = 0;

        for (int i = 0; i < (int)macro.size(); i++) {
            for (int j = i + 1; j < (int)macro.size(); j++) {
                if (are_adjacent(macro[i], macro[j], elem_adj)) count++;
            }
        }

        return count;
    }

    void build_node_macros(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                           const std::vector<std::vector<int>> &elem_adj,
                           std::vector<std::vector<int>> &macros) const {
        macros.clear();

        std::unordered_set<std::string> seen;

        for (int node = 0; node < num_nodes; node++) {
            std::vector<int> m;

            for (int jp = ne_ptr[node]; jp < ne_ptr[node + 1]; jp++) {
                int e = ne_cols[jp];
                if (e < 0 || e >= num_elements) die("bad ne element");
                m.push_back(e);
            }

            std::sort(m.begin(), m.end());
            m.erase(std::unique(m.begin(), m.end()), m.end());

            if ((int)m.size() < 2) continue;
            if ((int)m.size() > macro_max_size) continue;

            int possible = ((int)m.size() * ((int)m.size() - 1)) / 2;
            int internal = macro_internal_edges(m, elem_adj);

            // Reject very loose node stars.
            // For quads/shells, this keeps compact vertex/edge patches.
            if (possible > 0 && 2 * internal < possible) continue;

            std::string key;
            for (int e : m) {
                key += std::to_string(e);
                key += ",";
            }

            if (seen.insert(key).second) {
                macros.push_back(std::move(m));
            }
        }
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int touch = 0;
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) touch++;
        }
        return touch;
    }

    int macro_num_new(const std::vector<int> &macro) const {
        int nnew = 0;
        for (int e : macro) {
            if (elem_sd_ind[e] < 0) nnew++;
        }
        return nnew;
    }

    int macro_touch_score(const std::vector<int> &macro, int isd,
                          const std::vector<std::vector<int>> &elem_adj) const {
        int score = 0;

        for (int e : macro) {
            if (elem_sd_ind[e] >= 0) continue;
            score += count_touching_sd_neighbors(e, isd, elem_adj);
        }

        return score;
    }

    int choose_best_macro(const std::vector<std::vector<int>> &macros, int isd, int sd_size,
                          const std::vector<std::vector<int>> &elem_adj) const {
        int best = -1;
        int best_touch = -1;
        int best_nnew = -1;

        for (int im = 0; im < (int)macros.size(); im++) {
            int nnew = macro_num_new(macros[im]);
            if (nnew <= 0) continue;
            if (sd_size + nnew > target_sd_size) continue;

            int touch = macro_touch_score(macros[im], isd, elem_adj);
            if (touch <= 0) continue;

            if (touch > best_touch || (touch == best_touch && nnew > best_nnew)) {
                best = im;
                best_touch = touch;
                best_nnew = nnew;
            }
        }

        return best;
    }

    int choose_best_single(const std::vector<int> &frontier, int isd,
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
                                std::vector<int> &frontier, std::vector<int> &frontier_flag) const {
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] >= 0) continue;

            if (!frontier_flag[nbr]) {
                frontier.push_back(nbr);
                frontier_flag[nbr] = 1;
            }
        }
    }

    void assign_elem(int e, int isd, int &sd_size, int &num_unassigned,
                     const std::vector<std::vector<int>> &elem_adj, std::vector<int> &frontier,
                     std::vector<int> &frontier_flag) {
        if (elem_sd_ind[e] >= 0) return;

        elem_sd_ind[e] = isd;
        sd_size++;
        num_unassigned--;

        add_frontier_from_elem(e, elem_adj, frontier, frontier_flag);
    }

    void assign_macro_greedy(const std::vector<std::vector<int>> &macros,
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

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            assign_elem(seed, isd, sd_size, num_unassigned, elem_adj, frontier, frontier_flag);

            while (sd_size < target_sd_size) {
                int imacro = choose_best_macro(macros, isd, sd_size, elem_adj);

                if (imacro >= 0) {
                    for (int e : macros[imacro]) {
                        if (sd_size >= target_sd_size) break;
                        assign_elem(e, isd, sd_size, num_unassigned, elem_adj, frontier,
                                    frontier_flag);
                    }
                    continue;
                }

                int e = choose_best_single(frontier, isd, elem_adj);
                if (e < 0) break;

                assign_elem(e, isd, sd_size, num_unassigned, elem_adj, frontier, frontier_flag);
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
                printf("ERROR[UnstructMacroElementSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};