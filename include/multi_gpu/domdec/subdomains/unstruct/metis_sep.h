#pragma once

#include <metis.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

template <bool OPTIMIZE = true>
class MetisCornerOptSplitter {
   public:
    int num_elements = 0, num_nodes = 0, nodes_per_elem = 0, target_sd_size = 0;
    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    int max_local_passes = 25;

    int min_sd_size = 0;
    int max_sd_size = 0;

    // METIS imbalance tolerance. 1.05 means roughly 5% imbalance allowed.
    double metis_ubfactor = 1.05;

    MetisCornerOptSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
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

        metis_initial_partition(elem_adj);
        compact_subdomain_ids();

        if constexpr (OPTIMIZE) {
            local_corner_optimization(ne_ptr, ne_cols, elem_adj);
            compact_subdomain_ids();

            int remaining = count_all_corner_violations(ne_ptr, ne_cols);
            if (remaining > 0) {
                printf("WARNING[MetisCornerOptSplitter]: %d corner violations remain\n", remaining);
            }
        }
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[MetisCornerOptSplitter]: %s\n", msg);
        std::exit(1);
    }

    int iceil_div(int a, int b) const { return (a + b - 1) / b; }

    int checked_node(int e, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * e + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf("ERROR[MetisCornerOptSplitter]: bad node %d elem %d lnode %d\n", gnode, e,
                   lnode);
            std::exit(1);
        }
        return gnode;
    }

    void build_unique_element_adjacency(const std::vector<int> &ee_ptr,
                                        const std::vector<int> &ee_cols,
                                        std::vector<std::vector<int>> &elem_adj) const {
        elem_adj.assign(num_elements, std::vector<int>());

        if ((int)ee_ptr.size() != num_elements + 1) die("bad ee_ptr size");

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

    void metis_initial_partition(const std::vector<std::vector<int>> &elem_adj) {
        elem_sd_ind.assign(num_elements, 0);

        if (num_elements == 0) {
            num_subdomains = 0;
            return;
        }

        int estimated = iceil_div(num_elements, target_sd_size);
        estimated = std::max(1, estimated);
        estimated = std::min(estimated, num_elements);

        num_subdomains = estimated;

        if (num_subdomains == 1) {
            std::fill(elem_sd_ind.begin(), elem_sd_ind.end(), 0);
            return;
        }

        std::vector<idx_t> xadj(num_elements + 1, 0);
        std::vector<idx_t> adjncy;

        for (int e = 0; e < num_elements; e++) {
            xadj[e] = (idx_t)adjncy.size();
            for (int nbr : elem_adj[e]) {
                adjncy.push_back((idx_t)nbr);
            }
        }
        xadj[num_elements] = (idx_t)adjncy.size();

        idx_t nvtxs = (idx_t)num_elements;
        idx_t ncon = 1;
        idx_t nparts = (idx_t)num_subdomains;
        idx_t objval = 0;

        std::vector<idx_t> part(num_elements, 0);

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);

        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
        options[METIS_OPTION_CONTIG] = 1;
        options[METIS_OPTION_NUMBERING] = 0;

        real_t ubvec[1];
        ubvec[0] = (real_t)metis_ubfactor;

        int status =
            METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr,
                                nullptr, &nparts, nullptr, ubvec, options, &objval, part.data());

        if (status != METIS_OK) {
            printf("WARNING[MetisCornerOptSplitter]: METIS_PartGraphKway failed, status=%d\n",
                   status);
            fallback_greedy_partition(elem_adj);
            return;
        }

        for (int e = 0; e < num_elements; e++) {
            elem_sd_ind[e] = (int)part[e];
            if (elem_sd_ind[e] < 0 || elem_sd_ind[e] >= num_subdomains) {
                die("bad METIS part id");
            }
        }

        printf("MetisCornerOptSplitter: METIS nparts=%d edgecut=%lld target_sd_size=%d\n",
               num_subdomains, (long long)objval, target_sd_size);
    }

    void fallback_greedy_partition(const std::vector<std::vector<int>> &elem_adj) {
        elem_sd_ind.assign(num_elements, -1);
        num_subdomains = 0;

        int num_left = num_elements;

        while (num_left > 0) {
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

            std::queue<int> q;
            q.push(seed);
            elem_sd_ind[seed] = isd;
            sd_size++;
            num_left--;

            while (!q.empty() && sd_size < target_sd_size) {
                int e = q.front();
                q.pop();

                for (int nbr : elem_adj[e]) {
                    if (elem_sd_ind[nbr] >= 0) continue;

                    elem_sd_ind[nbr] = isd;
                    sd_size++;
                    num_left--;
                    q.push(nbr);

                    if (sd_size >= target_sd_size) break;
                }
            }
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

            printf("MetisCornerOptSplitter local pass %d: corner violations %d -> %d\n", pass,
                   old_total, new_total);

            if (!changed || new_total >= old_total) break;

            old_total = new_total;
            if (old_total == 0) break;
        }
    }

    void compact_subdomain_ids() {
        std::unordered_set<int> sd_set;

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] >= 0) sd_set.insert(elem_sd_ind[e]);
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());
        std::sort(sd_vec.begin(), sd_vec.end());

        if (sd_vec.empty()) {
            num_subdomains = 0;
            return;
        }

        int max_old_sd = -1;
        for (int isd : sd_vec) max_old_sd = std::max(max_old_sd, isd);

        std::vector<int> sd_imap(max_old_sd + 1, -1);

        for (int new_isd = 0; new_isd < (int)sd_vec.size(); new_isd++) {
            int old_isd = sd_vec[new_isd];
            sd_imap[old_isd] = new_isd;
        }

        for (int e = 0; e < num_elements; e++) {
            int old_isd = elem_sd_ind[e];

            if (old_isd < 0 || old_isd > max_old_sd || sd_imap[old_isd] < 0) {
                printf("ERROR[MetisCornerOptSplitter]: bad compact elem=%d old_isd=%d\n", e,
                       old_isd);
                std::exit(1);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = (int)sd_vec.size();
    }
};