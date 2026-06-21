#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

class UnstructCompactPatchSplitter {
   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    int macro_max_size = 8;
    int min_internal_degree = 2;
    int strict_after_size = 4;

    UnstructCompactPatchSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
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

        assign_compact_patch_greedy(macros, elem_adj);
        merge_small_subdomains(elem_adj);
        compact_subdomain_ids();
        audit_stringy_subdomains(elem_adj);
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[UnstructCompactPatchSplitter]: %s\n", msg);
        std::exit(1);
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
                    printf("ERROR[UnstructCompactPatchSplitter]: bad ee elem=%d nbr=%d\n", e, nbr);
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
        return std::binary_search(elem_adj[a].begin(), elem_adj[a].end(), b);
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

            int internal = macro_internal_edges(m, elem_adj);

            // Need at least a connected element patch.
            if (internal < (int)m.size() - 1) continue;

            std::string key;
            for (int e : m) {
                key += std::to_string(e);
                key += ",";
            }

            if (seen.insert(key).second) macros.push_back(std::move(m));
        }
    }

    int internal_degree_in_set(int e, const std::vector<int> &elems,
                               const std::vector<std::vector<int>> &elem_adj) const {
        int deg = 0;

        for (int nbr : elem_adj[e]) {
            if (std::binary_search(elems.begin(), elems.end(), nbr)) deg++;
        }

        return deg;
    }

    bool compact_enough(const std::vector<int> &elems,
                        const std::vector<std::vector<int>> &elem_adj) const {
        if ((int)elems.size() < strict_after_size) return true;

        for (int e : elems) {
            int deg = internal_degree_in_set(e, elems, elem_adj);
            if (deg < min_internal_degree) return false;
        }

        return true;
    }

    std::vector<int> trial_union(const std::vector<int> &sd_elems,
                                 const std::vector<int> &cand) const {
        std::vector<int> trial = sd_elems;

        for (int e : cand) {
            if (elem_sd_ind[e] < 0) trial.push_back(e);
        }

        std::sort(trial.begin(), trial.end());
        trial.erase(std::unique(trial.begin(), trial.end()), trial.end());

        return trial;
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int touch = 0;

        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) touch++;
        }

        return touch;
    }

    int candidate_new_count(const std::vector<int> &cand) const {
        int nnew = 0;

        for (int e : cand) {
            if (elem_sd_ind[e] < 0) nnew++;
        }

        return nnew;
    }

    int candidate_touch_score(const std::vector<int> &cand, int isd,
                              const std::vector<std::vector<int>> &elem_adj) const {
        int score = 0;

        for (int e : cand) {
            if (elem_sd_ind[e] >= 0) continue;
            score += count_touching_sd_neighbors(e, isd, elem_adj);
        }

        return score;
    }

    int choose_best_macro(const std::vector<std::vector<int>> &macros,
                          const std::vector<int> &sd_elems, int isd,
                          const std::vector<std::vector<int>> &elem_adj) const {
        int sd_size = (int)sd_elems.size();

        int best = -1;
        int best_touch = -1;
        int best_nnew = -1;

        for (int im = 0; im < (int)macros.size(); im++) {
            int nnew = candidate_new_count(macros[im]);
            if (nnew <= 0) continue;
            if (sd_size + nnew > target_sd_size) continue;

            int touch = candidate_touch_score(macros[im], isd, elem_adj);
            if (touch <= 0) continue;

            std::vector<int> trial = trial_union(sd_elems, macros[im]);
            if (!compact_enough(trial, elem_adj)) continue;

            if (touch > best_touch || (touch == best_touch && nnew > best_nnew)) {
                best = im;
                best_touch = touch;
                best_nnew = nnew;
            }
        }

        return best;
    }

    void collect_closure_patch(int seed, int isd, const std::vector<int> &sd_elems,
                               const std::vector<std::vector<int>> &elem_adj,
                               std::vector<int> &patch) const {
        patch.clear();
        if (elem_sd_ind[seed] >= 0) return;

        patch.push_back(seed);

        for (int nbr : elem_adj[seed]) {
            if (elem_sd_ind[nbr] >= 0) continue;

            int touch_sd = count_touching_sd_neighbors(nbr, isd, elem_adj);
            if (touch_sd > 0) patch.push_back(nbr);
        }

        std::sort(patch.begin(), patch.end());
        patch.erase(std::unique(patch.begin(), patch.end()), patch.end());

        while ((int)sd_elems.size() + (int)patch.size() > target_sd_size && !patch.empty()) {
            patch.pop_back();
        }
    }

    bool choose_best_closure_patch(const std::vector<int> &frontier,
                                   const std::vector<int> &sd_elems, int isd,
                                   const std::vector<std::vector<int>> &elem_adj,
                                   std::vector<int> &best_patch) const {
        best_patch.clear();

        int best_touch = -1;
        int best_nnew = -1;

        for (int seed : frontier) {
            if (elem_sd_ind[seed] >= 0) continue;

            std::vector<int> patch;
            collect_closure_patch(seed, isd, sd_elems, elem_adj, patch);

            int nnew = candidate_new_count(patch);
            if (nnew <= 0) continue;
            if ((int)sd_elems.size() + nnew > target_sd_size) continue;

            std::vector<int> trial = trial_union(sd_elems, patch);
            if (!compact_enough(trial, elem_adj)) continue;

            int touch = candidate_touch_score(patch, isd, elem_adj);

            if (touch > best_touch || (touch == best_touch && nnew > best_nnew)) {
                best_patch = std::move(patch);
                best_touch = touch;
                best_nnew = nnew;
            }
        }

        return !best_patch.empty();
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

    void assign_elem(int e, int isd, int &num_unassigned,
                     const std::vector<std::vector<int>> &elem_adj, std::vector<int> &sd_elems,
                     std::vector<int> &frontier, std::vector<int> &frontier_flag) {
        if (elem_sd_ind[e] >= 0) return;

        elem_sd_ind[e] = isd;
        sd_elems.push_back(e);
        num_unassigned--;

        std::sort(sd_elems.begin(), sd_elems.end());
        sd_elems.erase(std::unique(sd_elems.begin(), sd_elems.end()), sd_elems.end());

        add_frontier_from_elem(e, elem_adj, frontier, frontier_flag);
    }

    void assign_patch(const std::vector<int> &patch, int isd, int &num_unassigned,
                      const std::vector<std::vector<int>> &elem_adj, std::vector<int> &sd_elems,
                      std::vector<int> &frontier, std::vector<int> &frontier_flag) {
        for (int e : patch) {
            if ((int)sd_elems.size() >= target_sd_size) break;
            assign_elem(e, isd, num_unassigned, elem_adj, sd_elems, frontier, frontier_flag);
        }
    }

    void assign_compact_patch_greedy(const std::vector<std::vector<int>> &macros,
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

            std::vector<int> sd_elems;
            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            assign_elem(seed, isd, num_unassigned, elem_adj, sd_elems, frontier, frontier_flag);

            while ((int)sd_elems.size() < target_sd_size) {
                int imacro = choose_best_macro(macros, sd_elems, isd, elem_adj);

                if (imacro >= 0) {
                    assign_patch(macros[imacro], isd, num_unassigned, elem_adj, sd_elems, frontier,
                                 frontier_flag);
                    continue;
                }

                std::vector<int> patch;
                if (choose_best_closure_patch(frontier, sd_elems, isd, elem_adj, patch)) {
                    assign_patch(patch, isd, num_unassigned, elem_adj, sd_elems, frontier,
                                 frontier_flag);
                    continue;
                }

                // No compact growth possible. Stop this subdomain instead of making a string.
                break;
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

            int best_jsd = -1;
            int best_touch = -1;

            for (int jsd = 0; jsd < num_subdomains; jsd++) {
                if (jsd == isd || sd_size[jsd] <= 0) continue;

                std::vector<int> trial;

                for (int e = 0; e < num_elements; e++) {
                    if (elem_sd_ind[e] == isd || elem_sd_ind[e] == jsd) {
                        trial.push_back(e);
                    }
                }

                std::sort(trial.begin(), trial.end());

                if ((int)trial.size() > target_sd_size + min_size) continue;
                if (!compact_enough(trial, elem_adj)) continue;

                int touch = 0;

                for (int e = 0; e < num_elements; e++) {
                    if (elem_sd_ind[e] != isd) continue;

                    for (int nbr : elem_adj[e]) {
                        if (elem_sd_ind[nbr] == jsd) touch++;
                    }
                }

                if (touch > best_touch) {
                    best_touch = touch;
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
                printf("ERROR[UnstructCompactPatchSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }

    void audit_stringy_subdomains(const std::vector<std::vector<int>> &elem_adj) const {
        int bad_sds = 0;

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::vector<int> elems;

            for (int e = 0; e < num_elements; e++) {
                if (elem_sd_ind[e] == isd) elems.push_back(e);
            }

            std::sort(elems.begin(), elems.end());

            if (!compact_enough(elems, elem_adj)) {
                bad_sds++;
                printf("WARNING[UnstructCompactPatchSplitter]: stringy subdomain isd=%d size=%d\n",
                       isd, (int)elems.size());
            }
        }

        if (bad_sds > 0) {
            printf("WARNING[UnstructCompactPatchSplitter]: %d stringy subdomains remain\n",
                   bad_sds);
        }
    }
};