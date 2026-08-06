#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "domdec/subdomains/_iev.h"

template <class IEVSplitter>
class UnstructuredIEVSplitting {
   public:
    int num_elements = 0;
    int num_nodes = 0;
    int nodes_per_elem = 0;
    int target_sd_size = 0;
    int MAX_NUM_VERTEX_PER_SUBDOMAIN = 12;

    const int *elem_conn = nullptr;

    int num_subdomains = 0;

    std::vector<int> elem_sd_ind;

    int node_elem_nnz = 0;
    std::vector<int> node_elem_rowp;
    std::vector<int> node_elem_ct;
    std::vector<int> node_sd_cols;

    std::vector<int> node_class_ind;
    std::vector<int> node_nsd;

    int I_nnodes = 0;
    int IE_nnodes = 0;
    int IEV_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;

    std::vector<int> IEV_sd_ptr;
    std::vector<int> IEV_sd_ind;
    std::vector<int> IEV_nodes;
    std::vector<int> IEV_elem_conn;

    UnstructuredIEVSplitting(int num_elements_, int num_nodes_, int nodes_per_elem_,
                             const int *elem_conn_, int target_sd_size_ = 16)
        : num_elements(num_elements_),
          num_nodes(num_nodes_),
          nodes_per_elem(nodes_per_elem_),
          target_sd_size(target_sd_size_),
          elem_conn(elem_conn_) {
        setup_unstructured_subdomains();
    }

    void clear() {
        num_subdomains = 0;
        elem_sd_ind.clear();

        node_elem_nnz = 0;
        node_elem_rowp.clear();
        node_elem_ct.clear();
        node_sd_cols.clear();

        node_class_ind.clear();
        node_nsd.clear();

        I_nnodes = 0;
        IE_nnodes = 0;
        IEV_nnodes = 0;
        Vc_nnodes = 0;
        V_nnodes = 0;
        lam_nnodes = 0;

        IEV_sd_ptr.clear();
        IEV_sd_ind.clear();
        IEV_nodes.clear();
        IEV_elem_conn.clear();
    }

   private:
    void die(const char *msg) const {
        printf("ERROR[UnstructuredIEVSplitting]: %s\n", msg);
        std::exit(1);
    }

    int checked_node(int ielem, int lnode) const {
        int gnode = elem_conn[nodes_per_elem * ielem + lnode];
        if (gnode < 0 || gnode >= num_nodes) {
            printf(
                "ERROR[UnstructuredIEVSplitting]: invalid gnode %d at elem %d lnode %d "
                "(num_nodes=%d)\n",
                gnode, ielem, lnode, num_nodes);
            std::exit(1);
        }
        return gnode;
    }

    void setup_unstructured_subdomains() {
        clear();

        if (num_elements < 0 || num_nodes < 0 || nodes_per_elem <= 0) die("bad sizes");
        if (num_elements > 0 && elem_conn == nullptr) die("elem_conn is null");
        if (target_sd_size <= 0) target_sd_size = 1;

        std::vector<int> ne_ptr, ne_cols;
        build_node_element_adjacency_legacy(ne_ptr, ne_cols);

        std::vector<int> ee_ptr, ee_cols;
        build_element_element_adjacency_legacy(ne_ptr, ne_cols, ee_ptr, ee_cols);

        IEVSplitter splitter(num_elements, num_nodes, nodes_per_elem, elem_conn, target_sd_size);
        splitter.split(ne_ptr, ne_cols, ee_ptr, ee_cols);

        num_subdomains = splitter.num_subdomains;
        elem_sd_ind = std::move(splitter.elem_sd_ind);

        build_node_subdomain_incidence_legacy();
        classify_nodes_legacy();
        compute_max_vertices_per_subdomain_legacy();

        build_IEV_nodes_legacy();
        build_IEV_elem_conn_legacy();

        printf("UnstructuredIEVSplitting complete:\n");
        printf("  num_subdomains = %d\n", num_subdomains);
        printf("  IEV_nnodes     = %d\n", IEV_nnodes);
        printf("  Vc_nnodes      = %d\n", Vc_nnodes);
        printf("  MAX_NUM_VERTEX_PER_SUBDOMAIN = %d\n", MAX_NUM_VERTEX_PER_SUBDOMAIN);
    }

    void build_node_element_adjacency_legacy(std::vector<int> &ne_ptr, std::vector<int> &ne_cols) {
        int ne_nnz = 0;
        std::vector<int> ne_cts(num_nodes, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                ne_cts[gnode]++;
                ne_nnz++;
            }
        }

        ne_ptr.assign(num_nodes + 1, 0);
        for (int inode = 0; inode < num_nodes; inode++) {
            ne_ptr[inode + 1] = ne_ptr[inode] + ne_cts[inode];
        }

        std::fill(ne_cts.begin(), ne_cts.end(), 0);
        ne_cols.assign(ne_nnz, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int offset = ne_ptr[gnode] + ne_cts[gnode];
                ne_cols[offset] = ielem;
                ne_cts[gnode]++;
            }
        }
    }

    void build_element_element_adjacency_legacy(const std::vector<int> &ne_ptr,
                                                const std::vector<int> &ne_cols,
                                                std::vector<int> &ee_ptr,
                                                std::vector<int> &ee_cols) {
        std::vector<int> ee_cts(num_elements, 0);
        int ee_nnz = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;
                    ee_cts[ielem]++;
                    ee_nnz++;
                }
            }
        }

        ee_ptr.assign(num_elements + 1, 0);
        for (int ielem = 0; ielem < num_elements; ielem++) {
            ee_ptr[ielem + 1] = ee_ptr[ielem] + ee_cts[ielem];
        }

        std::fill(ee_cts.begin(), ee_cts.end(), 0);
        ee_cols.assign(ee_nnz, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                for (int jp = ne_ptr[gnode]; jp < ne_ptr[gnode + 1]; jp++) {
                    int jelem = ne_cols[jp];
                    if (jelem == ielem) continue;

                    int offset = ee_ptr[ielem] + ee_cts[ielem];
                    ee_cols[offset] = jelem;
                    ee_cts[ielem]++;
                }
            }
        }
    }

    void build_node_subdomain_incidence_legacy() {
        node_elem_nnz = 0;
        node_elem_rowp.assign(num_nodes + 1, 0);
        node_elem_ct.assign(num_nodes, 0);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
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
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int offset = node_elem_rowp[gnode] + temp_node_elem[gnode];
                node_sd_cols[offset] = isd;
                temp_node_elem[gnode]++;
            }
        }
    }

    void classify_nodes_legacy() {
        node_class_ind.assign(num_nodes, 0);
        node_nsd.assign(num_nodes, 0);

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

            int nsd = (int)node_sds.size();
            node_nsd[inode] = nsd;

            if (nsd < 2) {
                node_class_ind[inode] = IEV_INTERIOR;
                I_nnodes++;
                IE_nnodes++;
                IEV_nnodes++;
            } else if (nsd == 2) {
                node_class_ind[inode] = IEV_EDGE;
                lam_nnodes++;
                IE_nnodes += nsd;
                IEV_nnodes += nsd;
            } else {
                node_class_ind[inode] = IEV_VERTEX;
                Vc_nnodes++;
                V_nnodes += nsd;
                IEV_nnodes += nsd;
            }
        }
    }

    void compute_max_vertices_per_subdomain_legacy() {
        std::vector<int> nvertex(num_subdomains, 0);
        MAX_NUM_VERTEX_PER_SUBDOMAIN = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int node_class = node_class_ind[gnode];

                if (node_class == IEV_VERTEX) {
                    nvertex[isd]++;
                }
            }
        }

        for (int isd = 0; isd < num_subdomains; isd++) {
            MAX_NUM_VERTEX_PER_SUBDOMAIN = std::max(MAX_NUM_VERTEX_PER_SUBDOMAIN, nvertex[isd]);
        }
    }

    void build_IEV_nodes_legacy() {
        IEV_sd_ptr.assign(num_subdomains + 1, 0);
        IEV_sd_ind.assign(IEV_nnodes, 0);
        IEV_nodes.assign(IEV_nnodes, 0);

        int IEV_ind = 0;
        std::vector<int> temp_completion(num_nodes, 0);

        for (int isd = 0; isd < num_subdomains; isd++) {
            std::fill(temp_completion.begin(), temp_completion.end(), 0);
            IEV_sd_ptr[isd + 1] = IEV_sd_ptr[isd];

            for (int ielem = 0; ielem < num_elements; ielem++) {
                if (elem_sd_ind[ielem] != isd) continue;

                for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                    int gnode = checked_node(ielem, lnode);

                    if (temp_completion[gnode]) continue;

                    if (IEV_ind >= IEV_nnodes) {
                        die("IEV_ind exceeded IEV_nnodes in build_IEV_nodes_legacy");
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
            printf("ERROR[UnstructuredIEVSplitting]: IEV_ind %d != IEV_nnodes %d\n", IEV_ind,
                   IEV_nnodes);
            std::exit(1);
        }
    }

    void build_IEV_elem_conn_legacy() {
        IEV_elem_conn.assign(num_elements * nodes_per_elem, -1);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int isd = elem_sd_ind[ielem];

            for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
                int gnode = checked_node(ielem, lnode);
                int local_ind = -1;

                for (int jp = IEV_sd_ptr[isd]; jp < IEV_sd_ptr[isd + 1]; jp++) {
                    if (IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf(
                        "ERROR[UnstructuredIEVSplitting]: failed to find duplicated IEV node "
                        "for elem %d, isd %d, gnode %d\n",
                        ielem, isd, gnode);
                    std::exit(1);
                }

                IEV_elem_conn[nodes_per_elem * ielem + lnode] = local_ind;
            }
        }
    }
};