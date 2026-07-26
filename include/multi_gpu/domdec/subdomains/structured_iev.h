#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "domdec/subdomains/_iev.h"

struct CoarseStructuredSubdomains {
    int num_elements = 0;
    int num_nodes = 0;
    int elem_nnz = 0;
    int num_subdomains = 0;

    // Coarse element -> coarse-node connectivity.
    std::vector<int> elem_ptr;
    std::vector<int> elem_conn;

    // Coarse element (fine subdomain) -> next-level subdomain.
    std::vector<int> elem_sd_ind;

    // Coarse node -> original global node.
    std::vector<int> nodes;
};

class StructuredIEVSplitting {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 4;

    int nxe = 0, nye = 0;
    int nxs = 0, nys = 0;
    int nx = 0, ny = 0;
    int order = 1;
    bool close_hoop = false;
    bool track_dirichlet = false;

    int num_subdomains = 0;

    // input/global connectivity, not owned
    const int *elem_conn = nullptr;

    // element -> subdomain
    std::vector<int> elem_sd_ind;

    // node -> subdomain incidence, with duplicate subdomains allowed before unique set use
    int node_elem_nnz = 0;
    std::vector<int> node_elem_rowp;
    std::vector<int> node_elem_ct;
    std::vector<int> node_sd_cols;

    // node classes
    std::vector<int> node_class_ind;
    std::vector<int> node_nsd;

    int nnodes_interior = 0;
    int nnodes_dirichlet_edge = 0;
    int nnodes_edge = 0;
    int nnodes_vertex = 0;

    // duplicated IEV layout
    int IEV_nnodes = 0;
    int IE_nnodes = 0;
    int I_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;

    std::vector<int> IEV_sd_ptr;     // size num_subdomains + 1
    std::vector<int> IEV_sd_ind;     // size IEV_nnodes
    std::vector<int> IEV_nodes;      // duplicated IEV node -> global node
    std::vector<int> IEV_elem_conn;  // elem connectivity in duplicated IEV numbering

    StructuredIEVSplitting() = default;

    StructuredIEVSplitting(int num_elements_, int num_nodes_, int nodes_per_elem_,
                           const int *elem_conn_, int nxe_, int nye_, int nxs_, int nys_,
                           int order_, bool close_hoop_ = false, bool track_dirichlet_ = false)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          elem_conn(elem_conn_) {
        setup_subdomains(nxe_, nye_, nxs_, nys_, order_, close_hoop_, track_dirichlet_);
    }

    void free() {
        elem_sd_ind.clear();

        node_elem_nnz = 0;
        node_elem_rowp.clear();
        node_elem_ct.clear();
        node_sd_cols.clear();

        node_class_ind.clear();
        node_nsd.clear();

        nnodes_interior = 0;
        nnodes_dirichlet_edge = 0;
        nnodes_edge = 0;
        nnodes_vertex = 0;

        IEV_nnodes = 0;
        IE_nnodes = 0;
        I_nnodes = 0;
        Vc_nnodes = 0;
        V_nnodes = 0;
        lam_nnodes = 0;

        IEV_sd_ptr.clear();
        IEV_sd_ind.clear();
        IEV_nodes.clear();
        IEV_elem_conn.clear();
    }

    CoarseStructuredSubdomains setup_coarse_structured_subdomains(int nxs2, int nys2) const {
        // returns data for multilevel BDDC coarser splitting
        if (elem_conn == nullptr) {
            throw std::runtime_error("StructuredIEVSplitting has no element connectivity");
        }

        if (num_elements != nxe * nye) {
            throw std::runtime_error("Structured mesh dimensions do not match num_elements");
        }

        if (nxs2 <= 0 || nys2 <= 0) {
            throw std::invalid_argument("Coarse subdomain counts must be positive");
        }

        // Build the next, coarser element partition independently. This leaves
        // the current fine splitting unchanged.
        StructuredIEVSplitting parent_splitting(num_elements, num_nodes, nodes_per_elem, elem_conn,
                                                nxe, nye, nxs2, nys2, order, close_hoop,
                                                track_dirichlet);

        CoarseStructuredSubdomains coarse;
        coarse.num_elements = num_subdomains;
        coarse.num_subdomains = parent_splitting.num_subdomains;

        // ------------------------------------------------------------
        // 1. Build coarse-node numbering from fine BDDC vertices.
        // ------------------------------------------------------------
        std::vector<int> global_to_coarse(num_nodes, -1);

        for (int gnode = 0; gnode < num_nodes; gnode++) {
            if (node_class_ind[gnode] != IEV_VERTEX) continue;

            global_to_coarse[gnode] = static_cast<int>(coarse.nodes.size());
            coarse.nodes.push_back(gnode);
        }

        coarse.num_nodes = static_cast<int>(coarse.nodes.size());

        if (coarse.num_nodes != Vc_nnodes) {
            throw std::runtime_error("Fine vertex count is inconsistent with Vc_nnodes");
        }

        // ------------------------------------------------------------
        // 2. Treat each fine subdomain as one coarse element.
        //
        // Use sets because a fine-grid vertex may be encountered through
        // several elements belonging to the same fine subdomain.
        // ------------------------------------------------------------
        std::vector<std::unordered_set<int>> coarse_element_nodes(coarse.num_elements);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int fine_sd = elem_sd_ind[ielem];

            if (fine_sd < 0 || fine_sd >= coarse.num_elements) {
                throw std::runtime_error("Invalid fine element-to-subdomain index");
            }

            const int *local_conn = &elem_conn[nodes_per_elem * ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                const int gnode = local_conn[lnode];

                if (node_class_ind[gnode] != IEV_VERTEX) continue;

                const int coarse_node = global_to_coarse[gnode];
                if (coarse_node < 0) {
                    throw std::runtime_error("Failed to map a fine vertex to a coarse node");
                }

                coarse_element_nodes[fine_sd].insert(coarse_node);
            }
        }

        // Flatten the connectivity into CSR form.
        coarse.elem_ptr.assign(coarse.num_elements + 1, 0);

        for (int celem = 0; celem < coarse.num_elements; celem++) {
            coarse.elem_ptr[celem + 1] =
                coarse.elem_ptr[celem] + static_cast<int>(coarse_element_nodes[celem].size());
        }

        coarse.elem_nnz = coarse.elem_ptr.back();
        coarse.elem_conn.resize(coarse.elem_nnz);

        for (int celem = 0; celem < coarse.num_elements; celem++) {
            // Sorting makes the connectivity deterministic.
            std::vector<int> nodes(coarse_element_nodes[celem].begin(),
                                   coarse_element_nodes[celem].end());

            std::sort(nodes.begin(), nodes.end());

            std::copy(nodes.begin(), nodes.end(),
                      coarse.elem_conn.begin() + coarse.elem_ptr[celem]);
        }

        // ------------------------------------------------------------
        // 3. Map each coarse element/fine subdomain to exactly one
        //    parent subdomain.
        // ------------------------------------------------------------
        coarse.elem_sd_ind.assign(coarse.num_elements, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int celem = elem_sd_ind[ielem];
            const int parent_sd = parent_splitting.elem_sd_ind[ielem];

            int &stored_parent = coarse.elem_sd_ind[celem];

            if (stored_parent < 0) {
                stored_parent = parent_sd;
            } else if (stored_parent != parent_sd) {
                throw std::invalid_argument(
                    "The requested structured partitions are not nested: "
                    "fine subdomain " +
                    std::to_string(celem) + " overlaps multiple parent subdomains");
            }
        }

        for (int celem = 0; celem < coarse.num_elements; celem++) {
            if (coarse.elem_sd_ind[celem] < 0) {
                throw std::runtime_error("Fine subdomain contains no elements");
            }
        }

        return coarse;
    }

   private:
    void setup_subdomains(int nxe_, int nye_, int nxs_, int nys_, int order_,
                          bool close_hoop_ = false, bool track_dirichlet_ = false) {
        // clear();

        nxe = nxe_;
        nye = nye_;
        nxs = nxs_;
        nys = nys_;
        order = order_;
        close_hoop = close_hoop_;
        track_dirichlet = track_dirichlet_;

        nx = order * nxe + 1;
        ny = close_hoop ? order * nye : (order * nye + 1);

        num_subdomains = nxs * nys;

        const int nxse = (nxe + nxs - 1) / nxs;
        const int nyse = (nye + nys - 1) / nys;

        // ------------------------------------------------------------
        // 1. element -> subdomain
        // ------------------------------------------------------------
        elem_sd_ind.resize(num_elements);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int ixe = ielem % nxe;
            int iye = ielem / nxe;

            int ixs = ixe / nxse;
            int iys = iye / nyse;

            if (ixs >= nxs) ixs = nxs - 1;
            if (iys >= nys) iys = nys - 1;

            elem_sd_ind[ielem] = ixs + iys * nxs;
        }

        // ------------------------------------------------------------
        // 2. node -> subdomain incidence
        // ------------------------------------------------------------
        build_node_subdomain_incidence();

        // ------------------------------------------------------------
        // 3. classify nodes
        // ------------------------------------------------------------
        classify_nodes();

        // ------------------------------------------------------------
        // 4. duplicated IEV nodal layout
        // ------------------------------------------------------------
        build_IEV_nodes();

        // ------------------------------------------------------------
        // 5. duplicated IEV element connectivity
        // ------------------------------------------------------------
        build_IEV_elem_conn();
    }

    void build_node_subdomain_incidence() {
        node_elem_nnz = 0;

        node_elem_rowp.assign(num_nodes + 1, 0);
        node_elem_ct.assign(num_nodes, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
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
            const int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];

                node_sd_cols[offset] = isd;
                temp_node_elem[gnode]++;
            }
        }
    }

    void classify_nodes() {
        node_class_ind.assign(num_nodes, IEV_INTERIOR);
        node_nsd.assign(num_nodes, 0);

        nnodes_interior = 0;
        nnodes_dirichlet_edge = 0;
        nnodes_edge = 0;
        nnodes_vertex = 0;

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

            int nsd = static_cast<int>(node_sds.size());
            node_nsd[inode] = nsd;

            int ix = inode % nx;
            int iy = inode / nx;

            bool on_x = (ix == 0) || (ix == nx - 1);
            bool on_y = ((iy == 0) || (iy == ny - 1)) && !close_hoop;
            bool on_bndry = on_x || on_y;

            if (nsd < 2) {
                node_class_ind[inode] = IEV_INTERIOR;
                nnodes_interior++;

                I_nnodes += 1;
                IE_nnodes += 1;
                IEV_nnodes += 1;
            } else if (nsd == 2) {
                if (on_bndry && track_dirichlet) {
                    node_class_ind[inode] = IEV_DIRICHLET_EDGE;
                    nnodes_dirichlet_edge++;

                    I_nnodes += nsd;
                    IE_nnodes += nsd;
                    IEV_nnodes += nsd;
                } else {
                    node_class_ind[inode] = IEV_EDGE;
                    nnodes_edge++;

                    IE_nnodes += nsd;
                    IEV_nnodes += nsd;
                    lam_nnodes += 1;
                }
            } else {
                node_class_ind[inode] = IEV_VERTEX;
                nnodes_vertex++;

                Vc_nnodes += 1;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }
    }

    void build_IEV_nodes() {
        IEV_sd_ptr.assign(num_subdomains + 1, 0);
        IEV_sd_ind.assign(IEV_nnodes, -1);
        IEV_nodes.assign(IEV_nnodes, -1);

        int IEV_ind = 0;
        std::vector<int> temp_completion(num_nodes, 0);

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::fill(temp_completion.begin(), temp_completion.end(), 0);
            IEV_sd_ptr[isd + 1] = IEV_sd_ptr[isd];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != isd) continue;

                const int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = local_elem_conn[lnode];

                    if (temp_completion[gnode]) continue;

                    if (IEV_ind >= IEV_nnodes) {
                        printf("ERROR: IEV_ind overflow: %d >= %d\n", IEV_ind, IEV_nnodes);
                        std::exit(1);
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
            printf("ERROR: IEV node count mismatch: built %d, expected %d\n", IEV_ind, IEV_nnodes);
            std::exit(1);
        }
    }

    void build_IEV_elem_conn() {
        IEV_elem_conn.assign(num_elements * nodes_per_elem, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            const int *local_elem_conn = &elem_conn[nodes_per_elem * ielem];
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = local_elem_conn[lnode];
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d, gnode %d\n",
                           ielem, gnode);
                    std::exit(1);
                }

                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }
    }
};