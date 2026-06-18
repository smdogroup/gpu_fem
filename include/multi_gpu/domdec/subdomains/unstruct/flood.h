#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <vector>

/**
 * @brief Flood-fill based unstructured element partitioner for BDDC.
 *
 * The previous greedy partitioner tended to create long, one-element-wide
 * "snake" regions while minimizing corner creation. Although acceptable for
 * scalar PDEs, these thin subdomains significantly increase interface length
 * and produce poor-quality subdomains for shell problems, where bending modes
 * are particularly sensitive to narrow connections.
 *
 * This partitioner instead grows each subdomain one element at a time using a
 * flood-fill strategy that favors compact growth, discourages one-element-wide
 * protrusions, and attempts to maintain approximately convex, box-like
 * subdomains with shorter interfaces.
 */
class UnstructFloodFillSplitter {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 0;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    UnstructFloodFillSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
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

        assign_initial_subdomains_floodfill(elem_adj);

        merge_small_subdomains_floodfill(elem_adj);

        compact_subdomain_ids();
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[UnstructSplitterV1]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];

        if (gnode < 0 || gnode >= num_nodes) {
            printf(
                "ERROR[UnstructSplitterV1]: invalid gnode %d at elem %d lnode %d "
                "(num_nodes=%d)\n",
                gnode, ielem, lnode, num_nodes);
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
                    printf("ERROR[UnstructSplitterV1]: bad ee adjacency elem=%d nbr=%d\n", ielem,
                           jelem);
                    std::exit(1);
                }

                if (jelem != ielem) {
                    nbrs.push_back(jelem);
                }
            }

            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

            elem_adj[ielem] = std::move(nbrs);
        }
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int count = 0;

        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) {
                count++;
            }
        }

        return count;
    }

    int count_unassigned_neighbors(int elem, const std::vector<std::vector<int>> &elem_adj) const {
        int count = 0;

        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] < 0) {
                count++;
            }
        }

        return count;
    }

    int candidate_score(int elem, int isd, int seed_dist, int sd_size,
                        const std::vector<std::vector<int>> &elem_adj) const {
        int touch = count_touching_sd_neighbors(elem, isd, elem_adj);
        int unassigned = count_unassigned_neighbors(elem, elem_adj);
        int degree = (int)elem_adj[elem].size();

        int perimeter_delta = degree - 2 * touch;

        int thin_penalty = 0;

        // Once the patch has some area, strongly discourage one-neighbor fingers.
        if (sd_size >= 4 && touch <= 1) {
            thin_penalty += 1000;
        }

        // Stronger penalty after the subdomain is half grown.
        if (sd_size >= target_sd_size / 2 && touch <= 1) {
            thin_penalty += 3000;
        }

        // Prefer candidates that still have room to grow outward.
        int trapped_penalty = 0;
        if (unassigned == 0 && sd_size + 1 < target_sd_size) {
            trapped_penalty += 500;
        }

        // Lower is better.
        return 100 * perimeter_delta - 50 * touch + 10 * seed_dist + thin_penalty + trapped_penalty;
    }

    void add_frontier_neighbors(int elem, int dist, const std::vector<std::vector<int>> &elem_adj,
                                std::vector<int> &frontier, std::vector<int> &frontier_flag,
                                std::vector<int> &frontier_dist) {
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] >= 0) continue;

            if (!frontier_flag[nbr]) {
                frontier.push_back(nbr);
                frontier_flag[nbr] = 1;
                frontier_dist[nbr] = dist + 1;
            } else {
                frontier_dist[nbr] = std::min(frontier_dist[nbr], dist + 1);
            }
        }
    }

    void assign_initial_subdomains_floodfill(const std::vector<std::vector<int>> &elem_adj) {
        elem_sd_ind.assign(num_elements, -1);
        num_subdomains = 0;

        int num_unassigned = num_elements;

        while (num_unassigned > 0) {
            int seed = -1;

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] < 0) {
                    seed = ielem;
                    break;
                }
            }

            if (seed < 0) break;

            int isd = num_subdomains++;

            std::vector<int> sd_elems;
            sd_elems.reserve(target_sd_size);

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);
            std::vector<int> frontier_dist(num_elements, std::numeric_limits<int>::max());

            elem_sd_ind[seed] = isd;
            sd_elems.push_back(seed);
            num_unassigned--;

            add_frontier_neighbors(seed, 0, elem_adj, frontier, frontier_flag, frontier_dist);

            while ((int)sd_elems.size() < target_sd_size && !frontier.empty()) {
                int best_pos = -1;
                int best_elem = -1;
                int best_score = std::numeric_limits<int>::max();

                for (int i = 0; i < (int)frontier.size(); i++) {
                    int elem = frontier[i];

                    if (elem_sd_ind[elem] >= 0) continue;

                    int score = candidate_score(elem, isd, frontier_dist[elem],
                                                (int)sd_elems.size(), elem_adj);

                    if (score < best_score) {
                        best_score = score;
                        best_elem = elem;
                        best_pos = i;
                    }
                }

                if (best_elem < 0) break;

                frontier[best_pos] = frontier.back();
                frontier.pop_back();
                frontier_flag[best_elem] = 0;

                elem_sd_ind[best_elem] = isd;
                sd_elems.push_back(best_elem);
                num_unassigned--;

                add_frontier_neighbors(best_elem, frontier_dist[best_elem], elem_adj, frontier,
                                       frontier_flag, frontier_dist);
            }
        }
    }

    void merge_small_subdomains_floodfill(const std::vector<std::vector<int>> &elem_adj) {
        if (num_subdomains <= 1) return;

        std::vector<int> sd_size(num_subdomains, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];
            if (isd >= 0) sd_size[isd]++;
        }

        int min_size = std::max(1, target_sd_size / 2);

        for (int isd = 0; isd < num_subdomains; isd++) {
            if (sd_size[isd] >= min_size) continue;

            std::vector<int> adj_count(num_subdomains, 0);

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != isd) continue;

                for (int nbr : elem_adj[ielem]) {
                    int jsd = elem_sd_ind[nbr];

                    if (jsd >= 0 && jsd != isd) {
                        adj_count[jsd]++;
                    }
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

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] == isd) {
                    elem_sd_ind[ielem] = best_jsd;
                    sd_size[best_jsd]++;
                    sd_size[isd]--;
                }
            }
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

        std::vector<int> sd_imap(num_subdomains, -1);

        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];

            if (old_isd < 0 || old_isd >= num_subdomains) {
                printf("ERROR[UnstructSplitterV1]: bad old subdomain id %d\n", old_isd);
                std::exit(1);
            }

            sd_imap[old_isd] = new_isd;
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int old_isd = elem_sd_ind[ielem];

            if (old_isd < 0 || old_isd >= num_subdomains || sd_imap[old_isd] < 0) {
                printf("ERROR[UnstructSplitterV1]: failed to compact elem %d old_isd %d\n", ielem,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[ielem] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};