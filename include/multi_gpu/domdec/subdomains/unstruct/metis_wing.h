#pragma once

#include <metis.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * METIS-based element partitioner for wing meshes.
 *
 * Optimization goals, in priority order:
 *
 *  1. Minimize BDDC vertex nodes lying on wing component junctions.
 *  2. Minimize the existing element corner violations.
 *  3. Minimize the total number of global BDDC vertex nodes.
 *
 * A global BDDC vertex node is defined here as a mesh node incident to
 * elements belonging to three or more distinct subdomains.
 *
 * Junction-aware METIS partitioning is achieved by assigning a large edge
 * weight to graph edges between adjacent elements sharing a junction node.
 */
template <bool OPTIMIZE = true>
class MetisWingOptSplitter {
   public:
    using NodeFlag = unsigned char;

    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 0;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;
    std::vector<int> elem_sd_ind;

    // Number of local move/swap passes.
    int max_local_passes = 25;
    int max_junction_passes = 40;
    int max_combined_passes = 30;

    // Allowed local subdomain-size range.
    int min_sd_size = 0;
    int max_sd_size = 0;

    /**
     * METIS imbalance tolerance.
     *
     * For example:
     *
     *   1.08 means approximately 8% partition imbalance is permitted.
     */
    double metis_ubfactor = 1.08;

    /**
     * Graph-edge weight assigned when adjacent elements share at least one
     * wing junction node.
     *
     * Ordinary graph edges have weight 1.
     */
    int junction_metis_edge_weight = 50;

    struct PartitionObjective {
        int junction_bddc_vertices = 0;
        int corner_violations = 0;
        int total_bddc_vertices = 0;
    };

    MetisWingOptSplitter(int num_elements_, int num_nodes_, int nodes_per_elem_,
                         const int *elem_conn_, int target_sd_size_ = 16,
                         double metis_ubfactor_ = 1.08, int junction_metis_edge_weight_ = 50)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_),
          metis_ubfactor(metis_ubfactor_),
          junction_metis_edge_weight(junction_metis_edge_weight_) {
        if (num_elements < 0) {
            die("num_elements cannot be negative");
        }

        if (num_nodes < 0) {
            die("num_nodes cannot be negative");
        }

        if (nodes_per_elem <= 0) {
            die("nodes_per_elem must be positive");
        }

        if (num_elements > 0 && elem_conn == nullptr) {
            die("elem_conn is null");
        }

        if (target_sd_size <= 0) {
            target_sd_size = 1;
        }

        if (metis_ubfactor < 1.0) {
            die("metis_ubfactor must be at least 1.0");
        }

        if (junction_metis_edge_weight < 1) {
            junction_metis_edge_weight = 1;
        }

        min_sd_size = std::max(1, target_sd_size / 2);
        max_sd_size = target_sd_size + std::max(2, target_sd_size / 4);
    }

    /**
     * Partition and optimize the mesh.
     *
     * ne_ptr/ne_cols:
     *     Node-to-element CSR adjacency.
     *
     * ee_ptr/ee_cols:
     *     Element-to-element CSR adjacency.
     *
     * is_junction_node:
     *     One flag per global node. Nonzero indicates that the node touches
     *     more than one wing component.
     */
    void split(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
               const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols,
               const std::vector<NodeFlag> &is_junction_node) {
        validate_node_element_adjacency(ne_ptr, ne_cols);

        if (static_cast<int>(is_junction_node.size()) != num_nodes) {
            die("is_junction_node has incorrect size");
        }

        std::vector<std::vector<int>> elem_adj;
        build_unique_element_adjacency(ee_ptr, ee_cols, elem_adj);

        metis_initial_partition(elem_adj, is_junction_node);

        compact_subdomain_ids();

        print_partition_diagnostics("after METIS", ne_ptr, ne_cols, is_junction_node);

        if constexpr (OPTIMIZE) {
            local_combined_optimization(ne_ptr, ne_cols, elem_adj, is_junction_node);

            compact_subdomain_ids();

            print_partition_diagnostics("final", ne_ptr, ne_cols, is_junction_node);

            const PartitionObjective final_obj =
                evaluate_objective(ne_ptr, ne_cols, is_junction_node);

            if (final_obj.corner_violations > 0 || final_obj.junction_bddc_vertices > 0) {
                std::printf(
                    "WARNING[MetisWingOptSplitter]: "
                    "final corner violations=%d, "
                    "junction BDDC vertices=%d, "
                    "total BDDC vertices=%d\n",
                    final_obj.corner_violations, final_obj.junction_bddc_vertices,
                    final_obj.total_bddc_vertices);
            }
        }
    }

    /**
     * Convenience overload when no wing-junction information is available.
     */
    void split(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
               const std::vector<int> &ee_ptr, const std::vector<int> &ee_cols) {
        std::vector<NodeFlag> no_junctions(static_cast<std::size_t>(num_nodes), 0);

        split(ne_ptr, ne_cols, ee_ptr, ee_cols, no_junctions);
    }

    /**
     * Construct one junction flag per assembler node.
     *
     * A node is a junction node when the elements incident to that node do
     * not all belong to the same component.
     *
     * The Assembler type must provide:
     *
     *   get_num_nodes()
     *   get_num_elements()
     *   getElemComponents().createHostVec().getPtr()
     *   getConn().createHostVec().getPtr()
     */
    template <class Assembler>
    static std::vector<NodeFlag> get_junction_node_flags(Assembler &assembler,
                                                         int assembler_nodes_per_elem = 4) {
        const int nnodes = assembler.get_num_nodes();
        const int nelems = assembler.get_num_elements();

        if (nnodes < 0) {
            static_die("assembler returned negative node count");
        }

        if (nelems < 0) {
            static_die("assembler returned negative element count");
        }

        if (assembler_nodes_per_elem <= 0) {
            static_die("assembler_nodes_per_elem must be positive");
        }

        /*
         * Keep these host-vector objects alive while their pointers are used.
         */
        auto h_elem_comps_vec = assembler.getElemComponents().createHostVec();

        auto h_elem_conn_vec = assembler.getConn().createHostVec();

        const int *h_elem_comps = h_elem_comps_vec.getPtr();

        const int *h_elem_conn = h_elem_conn_vec.getPtr();

        if (nelems > 0 && (h_elem_comps == nullptr || h_elem_conn == nullptr)) {
            static_die("assembler returned a null host pointer");
        }

        std::vector<int> first_component(static_cast<std::size_t>(nnodes), -1);

        std::vector<NodeFlag> is_junction_node(static_cast<std::size_t>(nnodes), 0);

        for (int ielem = 0; ielem < nelems; ielem++) {
            const int icomp = h_elem_comps[ielem];

            for (int lnode = 0; lnode < assembler_nodes_per_elem; lnode++) {
                const int inode = h_elem_conn[assembler_nodes_per_elem * ielem + lnode];

                if (inode < 0 || inode >= nnodes) {
                    std::printf(
                        "ERROR[MetisWingOptSplitter]: "
                        "bad assembler node %d in element %d "
                        "at local node %d\n",
                        inode, ielem, lnode);

                    std::exit(EXIT_FAILURE);
                }

                if (first_component[inode] < 0) {
                    first_component[inode] = icomp;
                } else if (first_component[inode] != icomp) {
                    is_junction_node[inode] = 1;
                }
            }
        }

        return is_junction_node;
    }

    /**
     * Return all global nodes touching elements from three or more distinct
     * subdomains.
     */
    std::vector<int> compute_global_bddc_vertex_nodes(const std::vector<int> &ne_ptr,
                                                      const std::vector<int> &ne_cols) const {
        validate_node_element_adjacency(ne_ptr, ne_cols);

        std::vector<int> vertices;

        for (int gnode = 0; gnode < num_nodes; gnode++) {
            if (count_distinct_subdomains_at_node(gnode, ne_ptr, ne_cols) >= 3) {
                vertices.push_back(gnode);
            }
        }

        return vertices;
    }

    int count_global_bddc_vertex_nodes(const std::vector<int> &ne_ptr,
                                       const std::vector<int> &ne_cols) const {
        validate_node_element_adjacency(ne_ptr, ne_cols);

        int count = 0;

        for (int gnode = 0; gnode < num_nodes; gnode++) {
            if (count_distinct_subdomains_at_node(gnode, ne_ptr, ne_cols) >= 3) {
                count++;
            }
        }

        return count;
    }

    int count_bddc_vertex_nodes_on_junction(const std::vector<int> &ne_ptr,
                                            const std::vector<int> &ne_cols,
                                            const std::vector<NodeFlag> &is_junction_node) const {
        validate_node_element_adjacency(ne_ptr, ne_cols);

        if (static_cast<int>(is_junction_node.size()) != num_nodes) {
            die("is_junction_node has incorrect size");
        }

        int count = 0;

        for (int gnode = 0; gnode < num_nodes; gnode++) {
            if (!is_junction_node[gnode]) {
                continue;
            }

            if (count_distinct_subdomains_at_node(gnode, ne_ptr, ne_cols) >= 3) {
                count++;
            }
        }

        return count;
    }

    int count_all_corner_violations(const std::vector<int> &ne_ptr,
                                    const std::vector<int> &ne_cols) const {
        int total = 0;

        for (int isd = 0; isd < num_subdomains; isd++) {
            total += count_corner_violations_in_sd(isd, ne_ptr, ne_cols);
        }

        return total;
    }

   private:
    void die(const char *msg) const {
        std::printf("ERROR[MetisWingOptSplitter]: %s\n", msg);

        std::exit(EXIT_FAILURE);
    }

    static void static_die(const char *msg) {
        std::printf("ERROR[MetisWingOptSplitter]: %s\n", msg);

        std::exit(EXIT_FAILURE);
    }

    int iceil_div(int a, int b) const { return (a + b - 1) / b; }

    int checked_node(int e, int lnode) const {
        if (e < 0 || e >= num_elements) {
            die("bad element index");
        }

        if (lnode < 0 || lnode >= nodes_per_elem) {
            die("bad local-node index");
        }

        const int gnode = elem_conn[nodes_per_elem * e + lnode];

        if (gnode < 0 || gnode >= num_nodes) {
            std::printf(
                "ERROR[MetisWingOptSplitter]: "
                "bad node %d in element %d at local node %d\n",
                gnode, e, lnode);

            std::exit(EXIT_FAILURE);
        }

        return gnode;
    }

    void validate_node_element_adjacency(const std::vector<int> &ne_ptr,
                                         const std::vector<int> &ne_cols) const {
        if (static_cast<int>(ne_ptr.size()) != num_nodes + 1) {
            die("ne_ptr has incorrect size");
        }

        if (ne_ptr.empty() || ne_ptr[0] != 0) {
            die("ne_ptr must begin with zero");
        }

        for (int inode = 0; inode < num_nodes; inode++) {
            if (ne_ptr[inode] > ne_ptr[inode + 1]) {
                die("ne_ptr must be nondecreasing");
            }
        }

        if (ne_ptr[num_nodes] != static_cast<int>(ne_cols.size())) {
            die("ne_ptr final entry does not match ne_cols size");
        }

        for (int e : ne_cols) {
            if (e < 0 || e >= num_elements) {
                die("ne_cols contains invalid element");
            }
        }
    }

    void build_unique_element_adjacency(const std::vector<int> &ee_ptr,
                                        const std::vector<int> &ee_cols,
                                        std::vector<std::vector<int>> &elem_adj) const {
        elem_adj.assign(static_cast<std::size_t>(num_elements), std::vector<int>());

        if (static_cast<int>(ee_ptr.size()) != num_elements + 1) {
            die("ee_ptr has incorrect size");
        }

        if (ee_ptr.empty() || ee_ptr[0] != 0) {
            die("ee_ptr must begin with zero");
        }

        if (ee_ptr[num_elements] != static_cast<int>(ee_cols.size())) {
            die("ee_ptr final entry does not match ee_cols size");
        }

        for (int e = 0; e < num_elements; e++) {
            if (ee_ptr[e] > ee_ptr[e + 1]) {
                die("ee_ptr must be nondecreasing");
            }

            std::vector<int> nbrs;

            for (int jp = ee_ptr[e]; jp < ee_ptr[e + 1]; jp++) {
                const int nbr = ee_cols[jp];

                if (nbr < 0 || nbr >= num_elements) {
                    die("bad element neighbor");
                }

                if (nbr != e) {
                    nbrs.push_back(nbr);
                }
            }

            std::sort(nbrs.begin(), nbrs.end());

            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

            elem_adj[e] = std::move(nbrs);
        }
    }

    bool elements_share_junction_node(int ea, int eb,
                                      const std::vector<NodeFlag> &is_junction_node) const {
        for (int ia = 0; ia < nodes_per_elem; ia++) {
            const int na = checked_node(ea, ia);

            if (!is_junction_node[na]) {
                continue;
            }

            for (int ib = 0; ib < nodes_per_elem; ib++) {
                const int nb = checked_node(eb, ib);

                if (na == nb) {
                    return true;
                }
            }
        }

        return false;
    }

    void metis_initial_partition(const std::vector<std::vector<int>> &elem_adj,
                                 const std::vector<NodeFlag> &is_junction_node) {
        elem_sd_ind.assign(static_cast<std::size_t>(num_elements), 0);

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

        std::vector<idx_t> xadj(static_cast<std::size_t>(num_elements + 1), 0);

        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;

        std::size_t adjacency_nnz = 0;

        for (const auto &nbrs : elem_adj) {
            adjacency_nnz += nbrs.size();
        }

        adjncy.reserve(adjacency_nnz);
        adjwgt.reserve(adjacency_nnz);

        for (int e = 0; e < num_elements; e++) {
            xadj[e] = static_cast<idx_t>(adjncy.size());

            for (int nbr : elem_adj[e]) {
                adjncy.push_back(static_cast<idx_t>(nbr));

                const bool junction_edge = elements_share_junction_node(e, nbr, is_junction_node);

                const int weight = junction_edge ? junction_metis_edge_weight : 1;

                adjwgt.push_back(static_cast<idx_t>(weight));
            }
        }

        xadj[num_elements] = static_cast<idx_t>(adjncy.size());

        idx_t nvtxs = static_cast<idx_t>(num_elements);

        idx_t ncon = 1;

        idx_t nparts = static_cast<idx_t>(num_subdomains);

        idx_t objval = 0;

        std::vector<idx_t> part(static_cast<std::size_t>(num_elements), 0);

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);

        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

        options[METIS_OPTION_CONTIG] = 1;
        options[METIS_OPTION_NUMBERING] = 0;

        real_t ubvec[1];

        ubvec[0] = static_cast<real_t>(metis_ubfactor);

        const int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr,
                                               nullptr, adjwgt.data(), &nparts, nullptr, ubvec,
                                               options, &objval, part.data());

        if (status != METIS_OK) {
            std::printf(
                "WARNING[MetisWingOptSplitter]: "
                "METIS_PartGraphKway failed, status=%d\n",
                status);

            fallback_greedy_partition(elem_adj);
            return;
        }

        for (int e = 0; e < num_elements; e++) {
            elem_sd_ind[e] = static_cast<int>(part[e]);

            if (elem_sd_ind[e] < 0 || elem_sd_ind[e] >= num_subdomains) {
                die("bad METIS partition id");
            }
        }

        std::printf(
            "MetisWingOptSplitter: "
            "METIS nparts=%d "
            "weighted_edgecut=%lld "
            "target_sd_size=%d "
            "ubfactor=%.4f "
            "junction_edge_weight=%d\n",
            num_subdomains, static_cast<long long>(objval), target_sd_size, metis_ubfactor,
            junction_metis_edge_weight);
    }

    void fallback_greedy_partition(const std::vector<std::vector<int>> &elem_adj) {
        elem_sd_ind.assign(static_cast<std::size_t>(num_elements), -1);

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

            if (seed < 0) {
                break;
            }

            const int isd = num_subdomains++;

            int sd_size = 0;

            std::queue<int> q;
            q.push(seed);

            elem_sd_ind[seed] = isd;
            sd_size++;
            num_left--;

            while (!q.empty() && sd_size < target_sd_size) {
                const int e = q.front();
                q.pop();

                for (int nbr : elem_adj[e]) {
                    if (elem_sd_ind[nbr] >= 0) {
                        continue;
                    }

                    elem_sd_ind[nbr] = isd;
                    sd_size++;
                    num_left--;
                    q.push(nbr);

                    if (sd_size >= target_sd_size) {
                        break;
                    }
                }
            }
        }
    }

    void get_sd_elems(int isd, std::vector<int> &elems) const {
        elems.clear();

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] == isd) {
                elems.push_back(e);
            }
        }
    }

    int count_sd_size(int isd) const {
        int count = 0;

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] == isd) {
                count++;
            }
        }

        return count;
    }

    int count_sd_elems_at_node(int gnode, int isd, const std::vector<int> &ne_ptr,
                               const std::vector<int> &ne_cols) const {
        int count = 0;

        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
            const int e = ne_cols[jp];

            if (elem_sd_ind[e] == isd) {
                count++;
            }
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
            if (elem_sd_ind[e] != isd) {
                continue;
            }

            int ncorners = 0;

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                const int gnode = checked_node(e, lnode);

                if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) {
                    ncorners++;
                }
            }

            if (ncorners >= 2) {
                nviol++;
            }
        }

        return nviol;
    }

    int count_distinct_subdomains_at_node(int gnode, const std::vector<int> &ne_ptr,
                                          const std::vector<int> &ne_cols) const {
        if (gnode < 0 || gnode >= num_nodes) {
            die("invalid global node");
        }

        std::vector<int> subdomains;

        const int degree = ne_ptr[gnode + 1] - ne_ptr[gnode];

        subdomains.reserve(static_cast<std::size_t>(std::max(0, degree)));

        for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
            const int e = ne_cols[jp];
            const int isd = elem_sd_ind[e];

            if (isd >= 0) {
                subdomains.push_back(isd);
            }
        }

        std::sort(subdomains.begin(), subdomains.end());

        subdomains.erase(std::unique(subdomains.begin(), subdomains.end()), subdomains.end());

        return static_cast<int>(subdomains.size());
    }

    PartitionObjective evaluate_objective(const std::vector<int> &ne_ptr,
                                          const std::vector<int> &ne_cols,
                                          const std::vector<NodeFlag> &is_junction_node) const {
        PartitionObjective obj;

        obj.corner_violations = count_all_corner_violations(ne_ptr, ne_cols);

        for (int gnode = 0; gnode < num_nodes; gnode++) {
            const int nsd = count_distinct_subdomains_at_node(gnode, ne_ptr, ne_cols);

            if (nsd < 3) {
                continue;
            }

            obj.total_bddc_vertices++;

            if (is_junction_node[gnode]) {
                obj.junction_bddc_vertices++;
            }
        }

        return obj;
    }

    bool objective_better(const PartitionObjective &candidate,
                          const PartitionObjective &reference) const {
        /*
         * Lexicographic comparison:
         *
         * Junction BDDC vertices have the highest priority.
         */
        if (candidate.junction_bddc_vertices != reference.junction_bddc_vertices) {
            return candidate.junction_bddc_vertices < reference.junction_bddc_vertices;
        }

        if (candidate.corner_violations != reference.corner_violations) {
            return candidate.corner_violations < reference.corner_violations;
        }

        return candidate.total_bddc_vertices < reference.total_bddc_vertices;
    }

    bool objective_equal(const PartitionObjective &a, const PartitionObjective &b) const {
        return a.junction_bddc_vertices == b.junction_bddc_vertices &&
               a.corner_violations == b.corner_violations &&
               a.total_bddc_vertices == b.total_bddc_vertices;
    }

    bool sd_connected(int isd, const std::vector<std::vector<int>> &elem_adj) const {
        std::vector<int> elems;
        get_sd_elems(isd, elems);

        if (elems.size() <= 1) {
            return true;
        }

        std::vector<NodeFlag> visited(static_cast<std::size_t>(num_elements), 0);

        std::queue<int> q;
        q.push(elems.front());
        visited[elems.front()] = 1;

        int reached = 0;

        while (!q.empty()) {
            const int e = q.front();
            q.pop();

            reached++;

            for (int nbr : elem_adj[e]) {
                if (elem_sd_ind[nbr] != isd) {
                    continue;
                }

                if (visited[nbr]) {
                    continue;
                }

                visited[nbr] = 1;
                q.push(nbr);
            }
        }

        return reached == static_cast<int>(elems.size());
    }

    bool elem_touches_subdomain(int e, int isd,
                                const std::vector<std::vector<int>> &elem_adj) const {
        for (int nbr : elem_adj[e]) {
            if (elem_sd_ind[nbr] == isd) {
                return true;
            }
        }

        return false;
    }

    bool elem_has_corner(int e, const std::vector<int> &ne_ptr,
                         const std::vector<int> &ne_cols) const {
        const int isd = elem_sd_ind[e];

        if (isd < 0) {
            return false;
        }

        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
            const int gnode = checked_node(e, lnode);

            if (is_corner_node(gnode, isd, ne_ptr, ne_cols)) {
                return true;
            }
        }

        return false;
    }

    bool node_is_junction_bddc_vertex(int gnode, const std::vector<int> &ne_ptr,
                                      const std::vector<int> &ne_cols,
                                      const std::vector<NodeFlag> &is_junction_node) const {
        return is_junction_node[gnode] != 0 &&
               count_distinct_subdomains_at_node(gnode, ne_ptr, ne_cols) >= 3;
    }

    bool elem_touches_junction_bddc_vertex(int e, const std::vector<int> &ne_ptr,
                                           const std::vector<int> &ne_cols,
                                           const std::vector<NodeFlag> &is_junction_node) const {
        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
            const int gnode = checked_node(e, lnode);

            if (node_is_junction_bddc_vertex(gnode, ne_ptr, ne_cols, is_junction_node)) {
                return true;
            }
        }

        return false;
    }

    bool elem_is_problem_candidate(int e, const std::vector<int> &ne_ptr,
                                   const std::vector<int> &ne_cols,
                                   const std::vector<NodeFlag> &is_junction_node) const {
        return elem_touches_junction_bddc_vertex(e, ne_ptr, ne_cols, is_junction_node) ||
               elem_has_corner(e, ne_ptr, ne_cols);
    }

    std::vector<int> collect_candidate_destination_sds(
        int e, const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
        const std::vector<std::vector<int>> &elem_adj) const {
        std::vector<int> destination_sds;

        const int current_sd = elem_sd_ind[e];

        /*
         * Edge-adjacent destination subdomains.
         */
        for (int nbr : elem_adj[e]) {
            const int nbr_sd = elem_sd_ind[nbr];

            if (nbr_sd >= 0 && nbr_sd != current_sd) {
                destination_sds.push_back(nbr_sd);
            }
        }

        /*
         * Also include subdomains touching any node of the element.
         *
         * This is useful around multi-patch junctions where elements can
         * share a node without being present in the edge adjacency.
         */
        for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
            const int gnode = checked_node(e, lnode);

            for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                const int nbr_elem = ne_cols[jp];
                const int nbr_sd = elem_sd_ind[nbr_elem];

                if (nbr_sd >= 0 && nbr_sd != current_sd) {
                    destination_sds.push_back(nbr_sd);
                }
            }
        }

        std::sort(destination_sds.begin(), destination_sds.end());

        destination_sds.erase(std::unique(destination_sds.begin(), destination_sds.end()),
                              destination_sds.end());

        return destination_sds;
    }

    bool try_move_elem_combined(int e, int to_sd, const std::vector<int> &ne_ptr,
                                const std::vector<int> &ne_cols,
                                const std::vector<std::vector<int>> &elem_adj,
                                const std::vector<NodeFlag> &is_junction_node) {
        const int from_sd = elem_sd_ind[e];

        if (from_sd < 0 || to_sd < 0 || from_sd == to_sd) {
            return false;
        }

        const int from_size = count_sd_size(from_sd);

        const int to_size = count_sd_size(to_sd);

        if (from_size <= min_sd_size) {
            return false;
        }

        if (to_size >= max_sd_size) {
            return false;
        }

        /*
         * The new element must be edge-adjacent to the destination to retain
         * a contiguous destination partition.
         */
        if (!elem_touches_subdomain(e, to_sd, elem_adj)) {
            return false;
        }

        const PartitionObjective old_obj = evaluate_objective(ne_ptr, ne_cols, is_junction_node);

        elem_sd_ind[e] = to_sd;

        const bool connected = sd_connected(from_sd, elem_adj) && sd_connected(to_sd, elem_adj);

        if (connected) {
            const PartitionObjective new_obj =
                evaluate_objective(ne_ptr, ne_cols, is_junction_node);

            if (objective_better(new_obj, old_obj)) {
                return true;
            }
        }

        elem_sd_ind[e] = from_sd;
        return false;
    }

    bool try_swap_elems_combined(int ea, int eb, const std::vector<int> &ne_ptr,
                                 const std::vector<int> &ne_cols,
                                 const std::vector<std::vector<int>> &elem_adj,
                                 const std::vector<NodeFlag> &is_junction_node) {
        const int sa = elem_sd_ind[ea];
        const int sb = elem_sd_ind[eb];

        if (sa < 0 || sb < 0 || sa == sb) {
            return false;
        }

        /*
         * Each swapped element must connect to its proposed destination
         * through an element other than the other swapped element.
         */
        bool ea_touches_sb = false;
        bool eb_touches_sa = false;

        for (int nbr : elem_adj[ea]) {
            if (nbr != eb && elem_sd_ind[nbr] == sb) {
                ea_touches_sb = true;
                break;
            }
        }

        for (int nbr : elem_adj[eb]) {
            if (nbr != ea && elem_sd_ind[nbr] == sa) {
                eb_touches_sa = true;
                break;
            }
        }

        if (!ea_touches_sb || !eb_touches_sa) {
            return false;
        }

        const PartitionObjective old_obj = evaluate_objective(ne_ptr, ne_cols, is_junction_node);

        elem_sd_ind[ea] = sb;
        elem_sd_ind[eb] = sa;

        const bool connected = sd_connected(sa, elem_adj) && sd_connected(sb, elem_adj);

        if (connected) {
            const PartitionObjective new_obj =
                evaluate_objective(ne_ptr, ne_cols, is_junction_node);

            if (objective_better(new_obj, old_obj)) {
                return true;
            }
        }

        elem_sd_ind[ea] = sa;
        elem_sd_ind[eb] = sb;

        return false;
    }

    bool local_move_pass(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                         const std::vector<std::vector<int>> &elem_adj,
                         const std::vector<NodeFlag> &is_junction_node, bool junction_only) {
        bool changed = false;

        for (int e = 0; e < num_elements; e++) {
            const bool is_candidate =
                junction_only
                    ? elem_touches_junction_bddc_vertex(e, ne_ptr, ne_cols, is_junction_node)
                    : elem_is_problem_candidate(e, ne_ptr, ne_cols, is_junction_node);

            if (!is_candidate) {
                continue;
            }

            const std::vector<int> destination_sds =
                collect_candidate_destination_sds(e, ne_ptr, ne_cols, elem_adj);

            for (int to_sd : destination_sds) {
                if (try_move_elem_combined(e, to_sd, ne_ptr, ne_cols, elem_adj, is_junction_node)) {
                    changed = true;
                    break;
                }
            }
        }

        return changed;
    }

    bool local_swap_pass(const std::vector<int> &ne_ptr, const std::vector<int> &ne_cols,
                         const std::vector<std::vector<int>> &elem_adj,
                         const std::vector<NodeFlag> &is_junction_node, bool junction_only) {
        bool changed = false;

        for (int ea = 0; ea < num_elements; ea++) {
            const bool is_candidate =
                junction_only
                    ? elem_touches_junction_bddc_vertex(ea, ne_ptr, ne_cols, is_junction_node)
                    : elem_is_problem_candidate(ea, ne_ptr, ne_cols, is_junction_node);

            if (!is_candidate) {
                continue;
            }

            for (int eb : elem_adj[ea]) {
                if (elem_sd_ind[ea] == elem_sd_ind[eb]) {
                    continue;
                }

                if (try_swap_elems_combined(ea, eb, ne_ptr, ne_cols, elem_adj, is_junction_node)) {
                    changed = true;
                    break;
                }
            }
        }

        return changed;
    }

    void local_junction_optimization(const std::vector<int> &ne_ptr,
                                     const std::vector<int> &ne_cols,
                                     const std::vector<std::vector<int>> &elem_adj,
                                     const std::vector<NodeFlag> &is_junction_node) {
        PartitionObjective old_obj = evaluate_objective(ne_ptr, ne_cols, is_junction_node);

        for (int pass = 0; pass < max_junction_passes; pass++) {
            bool changed = false;

            changed |= local_move_pass(ne_ptr, ne_cols, elem_adj, is_junction_node, true);

            changed |= local_swap_pass(ne_ptr, ne_cols, elem_adj, is_junction_node, true);

            const PartitionObjective new_obj =
                evaluate_objective(ne_ptr, ne_cols, is_junction_node);

            std::printf(
                "MetisWingOptSplitter junction pass %d: "
                "junction vertices %d -> %d, "
                "corner violations %d -> %d, "
                "total vertices %d -> %d\n",
                pass, old_obj.junction_bddc_vertices, new_obj.junction_bddc_vertices,
                old_obj.corner_violations, new_obj.corner_violations, old_obj.total_bddc_vertices,
                new_obj.total_bddc_vertices);

            if (!changed || objective_equal(new_obj, old_obj)) {
                break;
            }

            old_obj = new_obj;

            if (old_obj.junction_bddc_vertices == 0) {
                break;
            }
        }
    }

    void local_combined_optimization(const std::vector<int> &ne_ptr,
                                     const std::vector<int> &ne_cols,
                                     const std::vector<std::vector<int>> &elem_adj,
                                     const std::vector<NodeFlag> &is_junction_node) {
        /*
         * First attack junction BDDC vertices aggressively.
         */
        local_junction_optimization(ne_ptr, ne_cols, elem_adj, is_junction_node);

        PartitionObjective old_obj = evaluate_objective(ne_ptr, ne_cols, is_junction_node);

        /*
         * Then continue with a coupled procedure that can consider either a
         * junction BDDC vertex or an ordinary corner violation.
         */
        const int max_passes = std::max(max_local_passes, max_combined_passes);

        for (int pass = 0; pass < max_passes; pass++) {
            bool changed = false;

            changed |= local_move_pass(ne_ptr, ne_cols, elem_adj, is_junction_node, false);

            changed |= local_swap_pass(ne_ptr, ne_cols, elem_adj, is_junction_node, false);

            const PartitionObjective new_obj =
                evaluate_objective(ne_ptr, ne_cols, is_junction_node);

            std::printf(
                "MetisWingOptSplitter combined pass %d: "
                "junction vertices %d -> %d, "
                "corner violations %d -> %d, "
                "total vertices %d -> %d\n",
                pass, old_obj.junction_bddc_vertices, new_obj.junction_bddc_vertices,
                old_obj.corner_violations, new_obj.corner_violations, old_obj.total_bddc_vertices,
                new_obj.total_bddc_vertices);

            if (!changed || objective_equal(new_obj, old_obj)) {
                break;
            }

            old_obj = new_obj;

            if (old_obj.junction_bddc_vertices == 0 && old_obj.corner_violations == 0) {
                break;
            }
        }
    }

    void print_partition_diagnostics(const char *label, const std::vector<int> &ne_ptr,
                                     const std::vector<int> &ne_cols,
                                     const std::vector<NodeFlag> &is_junction_node) const {
        const PartitionObjective obj = evaluate_objective(ne_ptr, ne_cols, is_junction_node);

        int smallest = std::numeric_limits<int>::max();

        int largest = 0;

        for (int isd = 0; isd < num_subdomains; isd++) {
            const int size = count_sd_size(isd);

            smallest = std::min(smallest, size);

            largest = std::max(largest, size);
        }

        if (num_subdomains == 0) {
            smallest = 0;
        }

        std::printf(
            "MetisWingOptSplitter [%s]: "
            "subdomains=%d "
            "size_range=[%d,%d] "
            "corner_violations=%d "
            "BDDC_vertices=%d "
            "junction_BDDC_vertices=%d\n",
            label, num_subdomains, smallest, largest, obj.corner_violations,
            obj.total_bddc_vertices, obj.junction_bddc_vertices);
    }

    void compact_subdomain_ids() {
        std::unordered_set<int> sd_set;

        for (int e = 0; e < num_elements; e++) {
            if (elem_sd_ind[e] >= 0) {
                sd_set.insert(elem_sd_ind[e]);
            }
        }

        std::vector<int> sd_vec(sd_set.begin(), sd_set.end());

        std::sort(sd_vec.begin(), sd_vec.end());

        if (sd_vec.empty()) {
            num_subdomains = 0;
            return;
        }

        int max_old_sd = -1;

        for (int isd : sd_vec) {
            max_old_sd = std::max(max_old_sd, isd);
        }

        std::vector<int> sd_imap(static_cast<std::size_t>(max_old_sd + 1), -1);

        for (int new_isd = 0; new_isd < static_cast<int>(sd_vec.size()); new_isd++) {
            const int old_isd = sd_vec[new_isd];

            sd_imap[old_isd] = new_isd;
        }

        for (int e = 0; e < num_elements; e++) {
            const int old_isd = elem_sd_ind[e];

            if (old_isd < 0 || old_isd > max_old_sd || sd_imap[old_isd] < 0) {
                std::printf(
                    "ERROR[MetisWingOptSplitter]: "
                    "bad compact mapping for element %d, "
                    "old subdomain=%d\n",
                    e, old_isd);

                std::exit(EXIT_FAILURE);
            }

            elem_sd_ind[e] = sd_imap[old_isd];
        }

        num_subdomains = static_cast<int>(sd_vec.size());
    }
};