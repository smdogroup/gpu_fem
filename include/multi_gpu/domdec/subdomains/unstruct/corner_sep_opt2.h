#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

class GlobalLocalCornerOptSplitterV2 {
    // does global-local opt like V1 GlobalLocalCornerOptSplitter
    // but also does more systemic regional rebuilds if stalls

   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    int max_local_passes = 25;
    int max_regional_passes = 8;
    int max_region_ring = 3;

    int min_sd_size = 0;
    int max_sd_size = 0;

    GlobalLocalCornerOptSplitterV2(int num_elements_, int num_nodes_, int nodes_per_elem_,
                                   const int *elem_conn_, int target_sd_size_ = 16)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_) {
        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) die("bad sizes");
        if (num_elements > 0 && elem_conn == nullptr) die("elem_conn is null");
        if (target_sd_size <= 0) target_sd_size = 1;

        min_sd_size = std::max(1, target_sd_size / 2);
        max_sd_size = target_sd_size + std::max(2, target_sd_size / 4);
    }

    void split(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
               const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols) {
        std::vector<std::vector<int>> elem_adj;
        build_unique_element_adjacency(ee_ptr, ee_cols, elem_adj);

        assign_initial_subdomains(ne_ptr, ne_cols, elem_adj);
        compact_subdomain_ids();

        local_corner_optimization(ne_ptr, ne_cols, elem_adj);
        compact_subdomain_ids();

        regional_rebuild_optimization(ne_ptr, ne_cols, elem_adj);
        compact_subdomain_ids();

        local_corner_optimization(ne_ptr, ne_cols, elem_adj);
        compact_subdomain_ids();

        int remaining = count_all_corner_violations(ne_ptr, ne_cols);
        if (remaining > 0) {
            printf("WARNING[GlobalLocalCornerOptSplitter]: %d corner violations remain\n",
                   remaining);
        }
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[GlobalLocalCornerOptSplitter]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int e, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * e + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf("ERROR[GlobalLocalCornerOptSplitter]: bad node %d elem %d lnode %d\n", gnode, e,
                   lnode);
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
                if (nbr < 0 || nbr >= num_elements) die("bad ee neighbor");
                if (nbr != e) nbrs.push_back(nbr);
            }

            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
            elem_adj[e] = std::move(nbrs);
        }
    }

    void get_sd_elems(int isd, std::vector<int> &elems) const {
        elems.clear();
        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] == isd) elems.push_back(e);
        }
    }

    int count_sd_size(int isd) const {
        int n = 0;
        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] == isd) n++;
        }
        return n;
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

    int count_corner_violations_in_sd(int isd, const std::vector<int> &ne_ptr,
                                      const std::vector<int> &ne_cols) const {
        int nviol = 0;

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] != isd) continue;

            int ncorners = 0;
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(e, lnode);
                if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) ncorners++;
            }

            if (ncorners >= 2) nviol++;
        }

        return nviol;
    }

    int count_all_corner_violations(const std::vector<int> &ne_ptr,
                                    const std::vector<int> &ne_cols) const {
        int total = 0;
        for (int isd = 0; isd < num_subdomains; isd++) {
            total += count_corner_violations_in_sd(isd, ne_ptr, ne_cols);
        }
        return total;
    }

    int affected_objective(int a, int b, const std::vector<int> &ne_ptr,
                           const std::vector<int> &ne_cols) const {
        int obj = 0;
        if (a >= 0) obj += count_corner_violations_in_sd(a, ne_ptr, ne_cols);
        if (b >= 0 && b != a) obj += count_corner_violations_in_sd(b, ne_ptr, ne_cols);
        return obj;
    }

    bool sd_connected(int isd, const std::vector<std::vector<int>> &elem_adj) const {
        std::vector<int> elems;
        get_sd_elems(isd, elems);
        if ((int)elems.size() <= 1) return true;

        std::unordered_set<int> s(elems.begin(), elems.end());
        std::unordered_set<int> visited;

        std::queue<int> q;
        q.push(elems[0]);
        visited.insert(elems[0]);

        while (!q.empty()) {
            int e = q.front();
            q.pop();

            for (int nbr : elem_adj[e]) {
                if (!s.count(nbr)) continue;
                if (visited.insert(nbr).second) q.push(nbr);
            }
        }

        return visited.size() == s.size();
    }

    int count_touching_sd_neighbors(int elem, int isd,
                                    const std::vector<std::vector<int>> &elem_adj) const {
        int touch = 0;
        for (int nbr : elem_adj[elem]) {
            if (elem_sd_ind[nbr] == isd) touch++;
        }
        return touch;
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

    int choose_best_initial_elem(const std::vector<int> &frontier, int isd,
                                 const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                 const std::vector<std::vector<int>> &elem_adj) {
        int best_elem = -1;
        int best_obj = std::numeric_limits<int>::max();
        int best_touch = -1;

        int old_obj = count_corner_violations_in_sd(isd, ne_ptr, ne_cols);

        for (int e : frontier) {
            if (elem_sd_ind[e] >= 0) continue;

            int touch = count_touching_sd_neighbors(e, isd, elem_adj);
            if (touch <= 0) continue;

            elem_sd_ind[e] = isd;
            int new_obj = count_corner_violations_in_sd(isd, ne_ptr, ne_cols);
            elem_sd_ind[e] = -1;

            if (new_obj < best_obj || (new_obj == best_obj && touch > best_touch)) {
                best_obj = new_obj;
                best_touch = touch;
                best_elem = e;
            }
        }

        if (best_elem < 0) return -1;
        if (best_obj > old_obj + 2) return -1;

        return best_elem;
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

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            elem_sd_ind[seed] = isd;
            sd_size++;
            num_unassigned--;

            add_frontier_from_elem(seed, elem_adj, frontier, frontier_flag);

            while (sd_size < target_sd_size) {
                int eadd = choose_best_initial_elem(frontier, isd, ne_ptr, ne_cols, elem_adj);
                if (eadd < 0) break;

                elem_sd_ind[eadd] = isd;
                sd_size++;
                num_unassigned--;

                add_frontier_from_elem(eadd, elem_adj, frontier, frontier_flag);
            }
        }
    }

    bool try_move_elem(int e, int to_sd, const std::vector<int> &ne_ptr,
                       const std::vector<int> &ne_cols,
                       const std::vector<std::vector<int>> &elem_adj) {
        int from_sd = elem_sd_ind[e];
        if (from_sd < 0 || to_sd < 0 || from_sd == to_sd) return false;

        int from_size = count_sd_size(from_sd);
        int to_size = count_sd_size(to_sd);

        if (from_size <= min_sd_size) return false;
        if (to_size >= max_sd_size) return false;

        int old_obj = affected_objective(from_sd, to_sd, ne_ptr, ne_cols);

        elem_sd_ind[e] = to_sd;

        bool ok = sd_connected(from_sd, elem_adj) && sd_connected(to_sd, elem_adj);
        int new_obj = affected_objective(from_sd, to_sd, ne_ptr, ne_cols);

        if (ok && new_obj < old_obj) return true;

        elem_sd_ind[e] = from_sd;
        return false;
    }

    bool try_swap_elems(int ea, int eb, const std::vector<int> &ne_ptr,
                        const std::vector<int> &ne_cols,
                        const std::vector<std::vector<int>> &elem_adj) {
        int sa = elem_sd_ind[ea];
        int sb = elem_sd_ind[eb];

        if (sa < 0 || sb < 0 || sa == sb) return false;

        int old_obj = affected_objective(sa, sb, ne_ptr, ne_cols);

        elem_sd_ind[ea] = sb;
        elem_sd_ind[eb] = sa;

        bool ok = sd_connected(sa, elem_adj) && sd_connected(sb, elem_adj);
        int new_obj = affected_objective(sa, sb, ne_ptr, ne_cols);

        if (ok && new_obj < old_obj) return true;

        elem_sd_ind[ea] = sa;
        elem_sd_ind[eb] = sb;
        return false;
    }

    bool elem_is_near_violation(int e, const std::vector<int> &ne_ptr,
                                const std::vector<int> &ne_cols) const {
        int isd = elem_sd_ind[e];
        if (isd < 0) return false;

        int ncorners = 0;
        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
            int gnode = checked_node(e, lnode);
            if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) ncorners++;
        }

        return ncorners >= 1;
    }

    bool local_move_pass(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                         const std::vector<std::vector<int>> &elem_adj) {
        bool changed = false;

        for (int e = 0; e < num_elements; e++) {
            if (!elem_is_near_violation(e, ne_ptr, ne_cols)) continue;

            std::vector<int> nbr_sds;

            for (int nbr : elem_adj[e]) {
                int jsd = elem_sd_ind[nbr];
                if (jsd >= 0 && jsd != elem_sd_ind[e]) nbr_sds.push_back(jsd);
            }

            std::sort(nbr_sds.begin(), nbr_sds.end());
            nbr_sds.erase(std::unique(nbr_sds.begin(), nbr_sds.end()), nbr_sds.end());

            for (int jsd : nbr_sds) {
                if (try_move_elem(e, jsd, ne_ptr, ne_cols, elem_adj)) {
                    changed = true;
                    break;
                }
            }
        }

        return changed;
    }

    bool local_swap_pass(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                         const std::vector<std::vector<int>> &elem_adj) {
        bool changed = false;

        for (int ea = 0; ea < num_elements; ea++) {
            if (!elem_is_near_violation(ea, ne_ptr, ne_cols)) continue;

            for (int eb : elem_adj[ea]) {
                if (elem_sd_ind[ea] == elem_sd_ind[eb]) continue;

                if (try_swap_elems(ea, eb, ne_ptr, ne_cols, elem_adj)) {
                    changed = true;
                    break;
                }
            }
        }

        return changed;
    }

    void local_corner_optimization(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                   const std::vector<std::vector<int>> &elem_adj) {
        int old_total = count_all_corner_violations(ne_ptr, ne_cols);

        for (int pass = 0; pass < max_local_passes; pass++) {
            bool changed = false;

            changed |= local_move_pass(ne_ptr, ne_cols, elem_adj);
            changed |= local_swap_pass(ne_ptr, ne_cols, elem_adj);

            int new_total = count_all_corner_violations(ne_ptr, ne_cols);

            printf("GlobalLocalCornerOptSplitter local pass %d: corner violations %d -> %d\n", pass,
                   old_total, new_total);

            if (!changed || new_total >= old_total) break;

            old_total = new_total;
            if (old_total == 0) break;
        }
    }

    void collect_bad_subdomains(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                std::vector<int> &bad_sds) const {
        bad_sds.clear();

        for (int isd = 0; isd < num_subdomains; isd++) {
            int nviol = count_corner_violations_in_sd(isd, ne_ptr, ne_cols);
            if (nviol > 0) bad_sds.push_back(isd);
        }
    }

    void build_sd_adjacency(const std::vector<std::vector<int>> &elem_adj,
                            std::vector<std::vector<int>> &sd_adj) const {
        sd_adj.assign(num_subdomains, std::vector<int>());

        for (int e = 0; e < num_elements; e++) {
            int isd = elem_sd_ind[e];
            if (isd < 0) continue;

            for (int nbr : elem_adj[e]) {
                int jsd = elem_sd_ind[nbr];
                if (jsd >= 0 && jsd != isd) sd_adj[isd].push_back(jsd);
            }
        }

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::sort(sd_adj[isd].begin(), sd_adj[isd].end());
            sd_adj[isd].erase(std::unique(sd_adj[isd].begin(), sd_adj[isd].end()),
                              sd_adj[isd].end());
        }
    }

    void collect_sd_ring(int root_sd, int ring_depth, const std::vector<std::vector<int>> &sd_adj,
                         std::vector<int> &region_sds) const {
        region_sds.clear();

        std::vector<int> dist(num_subdomains, -1);
        std::queue<int> q;

        dist[root_sd] = 0;
        q.push(root_sd);

        while (!q.empty()) {
            int isd = q.front();
            q.pop();

            region_sds.push_back(isd);

            if (dist[isd] >= ring_depth) continue;

            for (int jsd : sd_adj[isd]) {
                if (dist[jsd] < 0) {
                    dist[jsd] = dist[isd] + 1;
                    q.push(jsd);
                }
            }
        }

        std::sort(region_sds.begin(), region_sds.end());
        region_sds.erase(std::unique(region_sds.begin(), region_sds.end()), region_sds.end());
    }

    void collect_region_elements(const std::vector<int> &region_sds,
                                 std::vector<int> &region_elems) const {
        region_elems.clear();

        std::unordered_set<int> sds(region_sds.begin(), region_sds.end());

        for (int e = 0; e < num_elements; e++) {
            if (sds.count(elem_sd_ind[e])) region_elems.push_back(e);
        }
    }

    int choose_region_seed(const std::vector<int> &region_elems,
                           const std::vector<int> &region_flag) const {
        for (int e : region_elems) {
            if (region_flag[e] && elem_sd_ind[e] < 0) return e;
        }
        return -1;
    }

    void add_region_frontier_from_elem(int elem, const std::vector<int> &region_flag,
                                       const std::vector<std::vector<int>> &elem_adj,
                                       std::vector<int> &frontier,
                                       std::vector<int> &frontier_flag) const {
        for (int nbr : elem_adj[elem]) {
            if (!region_flag[nbr]) continue;
            if (elem_sd_ind[nbr] >= 0) continue;

            if (!frontier_flag[nbr]) {
                frontier.push_back(nbr);
                frontier_flag[nbr] = 1;
            }
        }
    }

    int choose_region_frontier_elem(const std::vector<int> &frontier,
                                    const std::vector<int> &region_flag, int isd,
                                    const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                                    const std::vector<std::vector<int>> &elem_adj) {
        int best_elem = -1;
        int best_obj = std::numeric_limits<int>::max();
        int best_touch = -1;

        for (int e : frontier) {
            if (!region_flag[e]) continue;
            if (elem_sd_ind[e] >= 0) continue;

            int touch = count_touching_sd_neighbors(e, isd, elem_adj);
            if (touch <= 0) continue;

            elem_sd_ind[e] = isd;
            int obj = count_corner_violations_in_sd(isd, ne_ptr, ne_cols);
            elem_sd_ind[e] = -1;

            if (obj < best_obj || (obj == best_obj && touch > best_touch)) {
                best_obj = obj;
                best_touch = touch;
                best_elem = e;
            }
        }

        return best_elem;
    }

    void rebuild_region(const std::vector<int> &region_elems, const std::vector<int> &ne_ptr,
                        const std::vector<int> &ne_cols,
                        const std::vector<std::vector<int>> &elem_adj) {
        std::vector<int> region_flag(num_elements, 0);
        for (int e : region_elems) region_flag[e] = 1;

        for (int e : region_elems) elem_sd_ind[e] = -1;

        int num_left = (int)region_elems.size();

        while (num_left > 0) {
            int seed = choose_region_seed(region_elems, region_flag);
            if (seed < 0) break;

            int isd = num_subdomains++;
            int sd_size = 0;

            std::vector<int> frontier;
            std::vector<int> frontier_flag(num_elements, 0);

            elem_sd_ind[seed] = isd;
            sd_size++;
            num_left--;

            add_region_frontier_from_elem(seed, region_flag, elem_adj, frontier, frontier_flag);

            while (sd_size < target_sd_size) {
                int eadd = choose_region_frontier_elem(frontier, region_flag, isd, ne_ptr, ne_cols,
                                                       elem_adj);
                if (eadd < 0) break;

                elem_sd_ind[eadd] = isd;
                sd_size++;
                num_left--;

                add_region_frontier_from_elem(eadd, region_flag, elem_adj, frontier, frontier_flag);
            }
        }
    }

    bool try_rebuild_region(const std::vector<int> &region_sds, const std::vector<int> &ne_ptr,
                            const std::vector<int> &ne_cols,
                            const std::vector<std::vector<int>> &elem_adj) {
        std::vector<int> old_assign = elem_sd_ind;
        int old_num_subdomains = num_subdomains;
        int old_global = count_all_corner_violations(ne_ptr, ne_cols);

        std::vector<int> region_elems;
        collect_region_elements(region_sds, region_elems);

        if ((int)region_elems.size() < target_sd_size) return false;
        if ((int)region_elems.size() > 12 * target_sd_size) return false;

        rebuild_region(region_elems, ne_ptr, ne_cols, elem_adj);
        compact_subdomain_ids();

        int new_global = count_all_corner_violations(ne_ptr, ne_cols);

        if (new_global < old_global) {
            printf(
                "GlobalLocalCornerOptSplitter regional rebuild accepted: %d -> %d, "
                "region_elems=%d region_sds=%d\n",
                old_global, new_global, (int)region_elems.size(), (int)region_sds.size());
            return true;
        }

        elem_sd_ind = old_assign;
        num_subdomains = old_num_subdomains;
        return false;
    }

    void regional_rebuild_optimization(const std::vector<int> &ne_ptr,
                                       const std::vector<int> &ne_cols,
                                       const std::vector<std::vector<int>> &elem_adj) {
        for (int pass = 0; pass < max_regional_passes; pass++) {
            std::vector<int> bad_sds;
            collect_bad_subdomains(ne_ptr, ne_cols, bad_sds);

            int old_total = count_all_corner_violations(ne_ptr, ne_cols);

            if (bad_sds.empty()) break;

            std::vector<std::vector<int>> sd_adj;
            build_sd_adjacency(elem_adj, sd_adj);

            bool changed = false;

            for (int bad_sd : bad_sds) {
                if (bad_sd < 0 || bad_sd >= num_subdomains) continue;

                for (int ring = 1; ring <= max_region_ring; ring++) {
                    std::vector<int> region_sds;
                    collect_sd_ring(bad_sd, ring, sd_adj, region_sds);

                    if (try_rebuild_region(region_sds, ne_ptr, ne_cols, elem_adj)) {
                        changed = true;
                        break;
                    }
                }

                if (changed) break;
            }

            int new_total = count_all_corner_violations(ne_ptr, ne_cols);

            printf("GlobalLocalCornerOptSplitter regional pass %d: corner violations %d -> %d\n",
                   pass, old_total, new_total);

            if (!changed || new_total >= old_total) break;
            if (new_total == 0) break;
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
                printf("ERROR[GlobalLocalCornerOptSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};