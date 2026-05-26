#pragma once

#include <algorithm>
#include <climits>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "cuda_utils.h"
#include "utils.h"

enum NodeType { INTERIOR = 0, INTERFACE = 1 };

class TacsComponentGPUPartitioner {
   public:
    TacsComponentGPUPartitioner(int ngpus_, int num_nodes_, int num_elements_, int nodes_per_elem_,
                                int *h_elem_conn_, int num_components_, int *h_elem_components_,
                                bool debug_ = false)
        : ngpus(ngpus_),
          num_nodes(num_nodes_),
          num_elements(num_elements_),
          num_components(num_components_),
          nodes_per_elem(nodes_per_elem_),
          h_elem_conn(h_elem_conn_),
          h_elem_components(h_elem_components_),
          debug(debug_) {
        split_elem_connectivity();
        assign_owned_nodes();
        build_owned_node_lists();
        build_local_node_maps();
        build_local_ghost_flags();
        build_ghost_node_maps();
        build_owned_local_maps();

        if (!debug) move_maps_to_device();
    }

    void free() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            if (d_local_elem_conn && d_local_elem_conn[g]) cudaFree(d_local_elem_conn[g]);
            if (d_node_gpu_ind && d_node_gpu_ind[g]) cudaFree(d_node_gpu_ind[g]);
            if (d_owned_nodes && d_owned_nodes[g]) cudaFree(d_owned_nodes[g]);
            if (d_owned_to_local_map && d_owned_to_local_map[g]) cudaFree(d_owned_to_local_map[g]);
            if (d_local_to_owned_map && d_local_to_owned_map[g]) cudaFree(d_local_to_owned_map[g]);

            if (h_local_elem_conn && h_local_elem_conn[g]) delete[] h_local_elem_conn[g];
            if (h_owned_nodes && h_owned_nodes[g]) delete[] h_owned_nodes[g];
            if (h_local_nodes && h_local_nodes[g]) delete[] h_local_nodes[g];
            if (h_owned_to_local_map && h_owned_to_local_map[g]) delete[] h_owned_to_local_map[g];
            if (h_local_to_owned_map && h_local_to_owned_map[g]) delete[] h_local_to_owned_map[g];
            if (h_is_local_ghost && h_is_local_ghost[g]) delete[] h_is_local_ghost[g];
        }

        int npairs = ngpus * ngpus;
        for (int idx = 0; idx < npairs; idx++) {
            int src = idx % ngpus;
            int dst = idx / ngpus;

            CHECK_CUDA(cudaSetDevice(src));
            if (d_srcred_map && d_srcred_map[idx]) cudaFree(d_srcred_map[idx]);

            CHECK_CUDA(cudaSetDevice(dst));
            if (d_dstred_map && d_dstred_map[idx]) cudaFree(d_dstred_map[idx]);

            if (h_srcred_map && h_srcred_map[idx]) delete[] h_srcred_map[idx];
            if (h_dstred_map && h_dstred_map[idx]) delete[] h_dstred_map[idx];
        }

        delete[] h_elem_assigned_gpu;

        delete[] local_nelems;
        delete[] h_local_elem_conn;
        delete[] d_local_elem_conn;

        delete[] h_node_gpu_ind;
        delete[] d_node_gpu_ind;

        delete[] owned_nnodes;
        delete[] owned_N;
        delete[] h_owned_nodes;
        delete[] d_owned_nodes;

        delete[] local_nnodes;
        delete[] local_N;
        delete[] h_local_nodes;

        delete[] ghost_nnodes;
        delete[] srcdest_nnodes;
        delete[] h_srcred_map;
        delete[] h_dstred_map;
        delete[] d_srcred_map;
        delete[] d_dstred_map;

        delete[] h_owned_to_local_map;
        delete[] d_owned_to_local_map;
        delete[] h_local_to_owned_map;
        delete[] d_local_to_owned_map;

        delete[] h_is_local_ghost;
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    int getLocalNumElements(const int g) const { return local_nelems[g]; }

    int find_owned_gpu_from_elem(int elem) const {
        if (elem < 0 || elem >= num_elements) return -1;
        return h_elem_assigned_gpu[elem];
    }

    int getNumOwnedNodes(const int g) const { return owned_nnodes[g]; }
    int getNumLocalNodes(const int g) const { return local_nnodes[g]; }

    int *getOwnedNodesPtr(const int g) const { return h_owned_nodes[g]; }
    int *getLocalNodesPtr(const int g) const { return h_local_nodes[g]; }

    void setElementComponents(const int *h_elem_comp, int **h_loc_elem_comp,
                              int **d_loc_elem_comp) {
        // printf("get loc elem components\n");
        int *elem_ctr = new int[ngpus];
        memset(elem_ctr, 0, ngpus * sizeof(int));
        for (int g = 0; g < ngpus; g++) {
            int local_nelems = getLocalNumElements(g);
            h_loc_elem_comp[g] = new int[local_nelems];
        }
        for (int e = 0; e < num_elements; e++) {
            int g = h_elem_assigned_gpu[e];
            int ered = elem_ctr[g];
            h_loc_elem_comp[g][ered] = h_elem_comp[e];
            elem_ctr[g]++;
        }

        for (int g = 0; g < ngpus; g++) {
            int local_nelems = getLocalNumElements(g);
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMalloc(&d_loc_elem_comp[g], local_nelems * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_loc_elem_comp[g], h_loc_elem_comp[g],
                                  local_nelems * sizeof(int), cudaMemcpyHostToDevice));
        }
        delete[] elem_ctr;
    }

   private:
    int get_num_interface_nodes(const std::vector<int> &current_comps, const int *n2e_ptr,
                                const int *n2e_vals) const {
        std::vector<char> in_comp(num_components, 0);
        for (int icomp : current_comps) in_comp[icomp] = 1;

        int num_interface = 0;

        for (int node = 0; node < num_nodes; node++) {
            bool has_inside = false;
            bool has_outside = false;

            for (int ip = n2e_ptr[node]; ip < n2e_ptr[node + 1]; ip++) {
                int ielem = n2e_vals[ip];
                int icomp = h_elem_components[ielem];

                if (in_comp[icomp])
                    has_inside = true;
                else
                    has_outside = true;

                if (has_inside && has_outside) {
                    num_interface++;
                    break;
                }
            }
        }

        return num_interface;
    }

    void split_elem_connectivity() {
        local_nelems = new int[ngpus];
        h_local_elem_conn = new int *[ngpus];
        d_local_elem_conn = new int *[ngpus];
        h_elem_assigned_gpu = new int[num_elements];

        std::memset(local_nelems, 0, ngpus * sizeof(int));
        std::memset(h_local_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(d_local_elem_conn, 0, ngpus * sizeof(int *));
        std::fill(h_elem_assigned_gpu, h_elem_assigned_gpu + num_elements, -1);

        int *n2e_cts = new int[num_nodes];
        std::memset(n2e_cts, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *lconn = &h_elem_conn[ielem * nodes_per_elem];
            for (int l = 0; l < nodes_per_elem; l++) {
                int node = lconn[l];
                n2e_cts[node]++;
            }
        }

        int *n2e_ptr = new int[num_nodes + 1];
        n2e_ptr[0] = 0;

        for (int inode = 0; inode < num_nodes; inode++) {
            n2e_ptr[inode + 1] = n2e_ptr[inode] + n2e_cts[inode];
        }

        int n2e_nnz = n2e_ptr[num_nodes];
        int *n2e_vals = new int[n2e_nnz];

        std::memset(n2e_cts, 0, num_nodes * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *lconn = &h_elem_conn[ielem * nodes_per_elem];
            for (int l = 0; l < nodes_per_elem; l++) {
                int node = lconn[l];
                int offset = n2e_ptr[node] + n2e_cts[node]++;
                n2e_vals[offset] = ielem;
            }
        }

        int *e2e_cts = new int[num_elements];
        std::memset(e2e_cts, 0, num_elements * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *lconn = &h_elem_conn[ielem * nodes_per_elem];
            std::vector<int> ctr(num_elements, 0);

            for (int l = 0; l < nodes_per_elem; l++) {
                int node = lconn[l];

                for (int jp = n2e_ptr[node]; jp < n2e_ptr[node + 1]; jp++) {
                    int jelem = n2e_vals[jp];
                    if (jelem == ielem) continue;

                    ctr[jelem]++;
                    if (ctr[jelem] == 2) e2e_cts[ielem]++;
                }
            }
        }

        int *e2e_ptr = new int[num_elements + 1];
        e2e_ptr[0] = 0;

        for (int ielem = 0; ielem < num_elements; ielem++) {
            e2e_ptr[ielem + 1] = e2e_ptr[ielem] + e2e_cts[ielem];
        }

        int e2e_nnz = e2e_ptr[num_elements];
        int *e2e_vals = new int[e2e_nnz];

        std::memset(e2e_cts, 0, num_elements * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int *lconn = &h_elem_conn[ielem * nodes_per_elem];
            std::vector<int> ctr(num_elements, 0);

            for (int l = 0; l < nodes_per_elem; l++) {
                int node = lconn[l];

                for (int jp = n2e_ptr[node]; jp < n2e_ptr[node + 1]; jp++) {
                    int jelem = n2e_vals[jp];
                    if (jelem == ielem) continue;

                    ctr[jelem]++;
                    if (ctr[jelem] == 2) {
                        int offset = e2e_ptr[ielem] + e2e_cts[ielem]++;
                        e2e_vals[offset] = jelem;
                    }
                }
            }
        }

        int *c2c_ctr = new int[num_components];
        std::memset(c2c_ctr, 0, num_components * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int icomp = h_elem_components[ielem];
            std::vector<char> seen(num_components, 0);

            for (int jp = e2e_ptr[ielem]; jp < e2e_ptr[ielem + 1]; jp++) {
                int jelem = e2e_vals[jp];
                int jcomp = h_elem_components[jelem];

                if (icomp == jcomp) continue;

                if (!seen[jcomp]) {
                    seen[jcomp] = 1;
                    c2c_ctr[icomp]++;
                }
            }
        }

        int *c2c_ptr = new int[num_components + 1];
        c2c_ptr[0] = 0;

        for (int icomp = 0; icomp < num_components; icomp++) {
            c2c_ptr[icomp + 1] = c2c_ptr[icomp] + c2c_ctr[icomp];
        }

        int c2c_nnz = c2c_ptr[num_components];
        int *c2c_vals = new int[c2c_nnz];

        std::memset(c2c_ctr, 0, num_components * sizeof(int));

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int icomp = h_elem_components[ielem];
            std::vector<char> seen(num_components, 0);

            for (int jp = e2e_ptr[ielem]; jp < e2e_ptr[ielem + 1]; jp++) {
                int jelem = e2e_vals[jp];
                int jcomp = h_elem_components[jelem];

                if (icomp == jcomp) continue;

                if (!seen[jcomp]) {
                    seen[jcomp] = 1;
                    int offset = c2c_ptr[icomp] + c2c_ctr[icomp]++;
                    c2c_vals[offset] = jcomp;
                }
            }
        }

        int *comp_gpu = new int[num_components];
        std::fill(comp_gpu, comp_gpu + num_components, -1);

        std::vector<int> comp_elem_cts(num_components, 0);
        std::vector<int> gpu_load(ngpus, 0);
        std::vector<std::vector<int>> gpu_comps(ngpus);

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int icomp = h_elem_components[ielem];
            comp_elem_cts[icomp]++;
        }

        int max_gpu_load = (num_elements + ngpus - 1) / ngpus;

        for (int seed = 0; seed < num_components; seed++) {
            if (comp_gpu[seed] >= 0) continue;

            int best_gpu = 0;
            for (int g = 1; g < ngpus; g++) {
                if (gpu_load[g] < gpu_load[best_gpu]) best_gpu = g;
            }

            comp_gpu[seed] = best_gpu;
            gpu_load[best_gpu] += comp_elem_cts[seed];
            gpu_comps[best_gpu].push_back(seed);

            bool added = true;
            while (added) {
                added = false;

                int best_comp = -1;
                int best_interface = INT_MAX;
                int best_conn = -1;

                for (int icomp = 0; icomp < num_components; icomp++) {
                    if (comp_gpu[icomp] >= 0) continue;
                    if (gpu_load[best_gpu] + comp_elem_cts[icomp] > max_gpu_load) continue;

                    int conn = 0;
                    for (int jp = c2c_ptr[icomp]; jp < c2c_ptr[icomp + 1]; jp++) {
                        int jcomp = c2c_vals[jp];
                        if (comp_gpu[jcomp] == best_gpu) conn++;
                    }

                    if (conn == 0) continue;

                    std::vector<int> trial_comps = gpu_comps[best_gpu];
                    trial_comps.push_back(icomp);

                    int interface = get_num_interface_nodes(trial_comps, n2e_ptr, n2e_vals);

                    if (interface < best_interface ||
                        (interface == best_interface && conn > best_conn)) {
                        best_interface = interface;
                        best_conn = conn;
                        best_comp = icomp;
                    }
                }

                if (best_comp >= 0) {
                    comp_gpu[best_comp] = best_gpu;
                    gpu_load[best_gpu] += comp_elem_cts[best_comp];
                    gpu_comps[best_gpu].push_back(best_comp);
                    added = true;
                }
            }
        }

        for (int icomp = 0; icomp < num_components; icomp++) {
            if (comp_gpu[icomp] >= 0) continue;

            int best_gpu = 0;
            int best_interface = INT_MAX;

            for (int g = 0; g < ngpus; g++) {
                std::vector<int> trial_comps = gpu_comps[g];
                trial_comps.push_back(icomp);

                int interface = get_num_interface_nodes(trial_comps, n2e_ptr, n2e_vals);

                if (interface < best_interface ||
                    (interface == best_interface && gpu_load[g] < gpu_load[best_gpu])) {
                    best_interface = interface;
                    best_gpu = g;
                }
            }

            comp_gpu[icomp] = best_gpu;
            gpu_load[best_gpu] += comp_elem_cts[icomp];
            gpu_comps[best_gpu].push_back(icomp);
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int icomp = h_elem_components[ielem];
            int g = comp_gpu[icomp];

            h_elem_assigned_gpu[ielem] = g;
            local_nelems[g]++;
        }

        std::vector<int> local_elem_cts(ngpus, 0);

        for (int g = 0; g < ngpus; g++) {
            int local_conn_size = local_nelems[g] * nodes_per_elem;
            h_local_elem_conn[g] = new int[local_conn_size];

            if (!debug) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaMalloc(&d_local_elem_conn[g], local_conn_size * sizeof(int)));
            }
        }

        for (int ielem = 0; ielem < num_elements; ielem++) {
            int g = h_elem_assigned_gpu[ielem];
            int local_e = local_elem_cts[g]++;

            std::memcpy(&h_local_elem_conn[g][local_e * nodes_per_elem],
                        &h_elem_conn[ielem * nodes_per_elem], nodes_per_elem * sizeof(int));
        }

        if (!debug) {
            for (int g = 0; g < ngpus; g++) {
                int local_conn_size = local_nelems[g] * nodes_per_elem;

                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaMemcpy(d_local_elem_conn[g], h_local_elem_conn[g],
                                      local_conn_size * sizeof(int), cudaMemcpyHostToDevice));
            }
        }

        delete[] n2e_cts;
        delete[] n2e_ptr;
        delete[] n2e_vals;

        delete[] e2e_cts;
        delete[] e2e_ptr;
        delete[] e2e_vals;

        delete[] c2c_ctr;
        delete[] c2c_ptr;
        delete[] c2c_vals;

        delete[] comp_gpu;
    }

    void assign_owned_nodes() {
        int *h_ne_cts = new int[num_nodes];
        std::memset(h_ne_cts, 0, num_nodes * sizeof(int));

        int ne_nnz = 0;

        for (int e = 0; e < num_elements; e++) {
            int *elem_nodes = &h_elem_conn[nodes_per_elem * e];

            for (int i = 0; i < nodes_per_elem; i++) {
                h_ne_cts[elem_nodes[i]]++;
                ne_nnz++;
            }
        }

        int *h_ne_ptr = new int[num_nodes + 1];
        h_ne_ptr[0] = 0;

        for (int n = 0; n < num_nodes; n++) {
            h_ne_ptr[n + 1] = h_ne_ptr[n] + h_ne_cts[n];
        }

        int *h_ne_elems = new int[ne_nnz];
        std::memset(h_ne_cts, 0, num_nodes * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            int *elem_nodes = &h_elem_conn[nodes_per_elem * e];

            for (int i = 0; i < nodes_per_elem; i++) {
                int node = elem_nodes[i];
                int offset = h_ne_ptr[node] + h_ne_cts[node]++;
                h_ne_elems[offset] = e;
            }
        }

        h_node_gpu_ind = new int[num_nodes];

        int *owned_node_cts = new int[ngpus];
        std::memset(owned_node_cts, 0, ngpus * sizeof(int));

        for (int n = 0; n < num_nodes; n++) {
            std::unordered_set<int> node_gpus;

            for (int ep = h_ne_ptr[n]; ep < h_ne_ptr[n + 1]; ep++) {
                int g = find_owned_gpu_from_elem(h_ne_elems[ep]);
                if (g >= 0) node_gpus.insert(g);
            }

            if (node_gpus.size() == 1) {
                int g = *node_gpus.begin();
                h_node_gpu_ind[n] = g;
                owned_node_cts[g]++;
            } else {
                h_node_gpu_ind[n] = -1;
            }
        }

        for (int n = 0; n < num_nodes; n++) {
            if (h_node_gpu_ind[n] != -1) continue;

            std::unordered_set<int> node_gpus;

            for (int ep = h_ne_ptr[n]; ep < h_ne_ptr[n + 1]; ep++) {
                int g = find_owned_gpu_from_elem(h_ne_elems[ep]);
                if (g >= 0) node_gpus.insert(g);
            }

            int best_gpu = -1;
            int best_ct = INT_MAX;

            for (int g : node_gpus) {
                if (owned_node_cts[g] < best_ct) {
                    best_ct = owned_node_cts[g];
                    best_gpu = g;
                }
            }

            h_node_gpu_ind[n] = best_gpu;
            owned_node_cts[best_gpu]++;
        }

        delete[] h_ne_cts;
        delete[] h_ne_ptr;
        delete[] h_ne_elems;
        delete[] owned_node_cts;
    }

    void build_owned_node_lists() {
        owned_nnodes = new int[ngpus];
        owned_N = new int[ngpus];
        h_owned_nodes = new int *[ngpus];
        d_owned_nodes = new int *[ngpus];

        std::memset(owned_nnodes, 0, ngpus * sizeof(int));
        std::memset(h_owned_nodes, 0, ngpus * sizeof(int *));
        std::memset(d_owned_nodes, 0, ngpus * sizeof(int *));

        for (int n = 0; n < num_nodes; n++) {
            int g = h_node_gpu_ind[n];
            owned_nnodes[g]++;
        }

        for (int g = 0; g < ngpus; g++) {
            owned_N[g] = owned_nnodes[g];
            h_owned_nodes[g] = new int[owned_nnodes[g]];
        }

        int *ct = new int[ngpus];
        std::memset(ct, 0, ngpus * sizeof(int));

        for (int n = 0; n < num_nodes; n++) {
            int g = h_node_gpu_ind[n];
            h_owned_nodes[g][ct[g]++] = n;
        }

        delete[] ct;
    }

    void build_local_node_maps() {
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        h_local_nodes = new int *[ngpus];

        std::memset(local_nnodes, 0, ngpus * sizeof(int));
        std::memset(local_N, 0, ngpus * sizeof(int));
        std::memset(h_local_nodes, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> node_set;

            int *conn = h_local_elem_conn[g];
            int conn_size = local_nelems[g] * nodes_per_elem;

            for (int i = 0; i < conn_size; i++) {
                node_set.insert(conn[i]);
            }

            std::vector<int> nodes(node_set.begin(), node_set.end());
            std::sort(nodes.begin(), nodes.end());

            local_nnodes[g] = static_cast<int>(nodes.size());
            local_N[g] = local_nnodes[g];

            h_local_nodes[g] = new int[local_nnodes[g]];

            for (int i = 0; i < local_nnodes[g]; i++) {
                h_local_nodes[g][i] = nodes[i];
            }
        }
    }

    void build_local_ghost_flags() {
        h_is_local_ghost = new bool *[ngpus];
        std::memset(h_is_local_ghost, 0, ngpus * sizeof(bool *));

        for (int g = 0; g < ngpus; g++) {
            h_is_local_ghost[g] = new bool[local_nnodes[g]];
            std::fill(h_is_local_ghost[g], h_is_local_ghost[g] + local_nnodes[g], false);
        }

        int **global_to_local = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            global_to_local[g] = new int[num_nodes];
            std::fill(global_to_local[g], global_to_local[g] + num_nodes, -1);

            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                int node = h_local_nodes[g][loc];
                global_to_local[g][node] = loc;
            }
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int dst_loc = 0; dst_loc < local_nnodes[dst]; dst_loc++) {
                int node = h_local_nodes[dst][dst_loc];
                int src = h_node_gpu_ind[node];

                if (src < 0 || src >= ngpus) continue;
                if (src == dst) continue;

                h_is_local_ghost[dst][dst_loc] = true;

                int src_loc = global_to_local[src][node];
                if (src_loc >= 0) h_is_local_ghost[src][src_loc] = true;
            }
        }

        for (int g = 0; g < ngpus; g++) {
            delete[] global_to_local[g];
        }

        delete[] global_to_local;
    }

    void build_ghost_node_maps() {
        int npairs = ngpus * ngpus;

        ghost_nnodes = new int[npairs];
        srcdest_nnodes = new int[npairs];
        h_srcred_map = new int *[npairs];
        h_dstred_map = new int *[npairs];
        d_srcred_map = new int *[npairs];
        d_dstred_map = new int *[npairs];

        std::memset(ghost_nnodes, 0, npairs * sizeof(int));
        std::memset(srcdest_nnodes, 0, npairs * sizeof(int));
        std::memset(h_srcred_map, 0, npairs * sizeof(int *));
        std::memset(h_dstred_map, 0, npairs * sizeof(int *));
        std::memset(d_srcred_map, 0, npairs * sizeof(int *));
        std::memset(d_dstred_map, 0, npairs * sizeof(int *));

        int *owned_pos = new int[num_nodes];
        std::fill(owned_pos, owned_pos + num_nodes, -1);

        for (int g = 0; g < ngpus; g++) {
            for (int i = 0; i < owned_nnodes[g]; i++) {
                owned_pos[h_owned_nodes[g][i]] = i;
            }
        }

        std::vector<int> *src_maps = new std::vector<int>[npairs];
        std::vector<int> *dst_maps = new std::vector<int>[npairs];

        for (int dst = 0; dst < ngpus; dst++) {
            for (int dst_loc = 0; dst_loc < local_nnodes[dst]; dst_loc++) {
                int node = h_local_nodes[dst][dst_loc];
                int src = h_node_gpu_ind[node];

                if (src == dst) continue;
                if (src < 0 || src >= ngpus) continue;

                int idx = pair_index(dst, src);

                src_maps[idx].push_back(owned_pos[node]);
                dst_maps[idx].push_back(dst_loc);
            }
        }

        for (int idx = 0; idx < npairs; idx++) {
            srcdest_nnodes[idx] = static_cast<int>(src_maps[idx].size());
            ghost_nnodes[idx] = srcdest_nnodes[idx];

            if (srcdest_nnodes[idx] == 0) continue;

            h_srcred_map[idx] = new int[srcdest_nnodes[idx]];
            h_dstred_map[idx] = new int[srcdest_nnodes[idx]];

            for (int i = 0; i < srcdest_nnodes[idx]; i++) {
                h_srcred_map[idx][i] = src_maps[idx][i];
                h_dstred_map[idx][i] = dst_maps[idx][i];
            }
        }

        delete[] owned_pos;
        delete[] src_maps;
        delete[] dst_maps;
    }

    void build_owned_local_maps() {
        h_owned_to_local_map = new int *[ngpus];
        d_owned_to_local_map = new int *[ngpus];

        h_local_to_owned_map = new int *[ngpus];
        d_local_to_owned_map = new int *[ngpus];

        std::memset(h_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(d_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(h_local_to_owned_map, 0, ngpus * sizeof(int *));
        std::memset(d_local_to_owned_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            h_owned_to_local_map[g] = new int[owned_nnodes[g]];
            h_local_to_owned_map[g] = new int[local_nnodes[g]];

            std::fill(h_local_to_owned_map[g], h_local_to_owned_map[g] + local_nnodes[g], -1);

            int *global_to_local = new int[num_nodes];
            std::fill(global_to_local, global_to_local + num_nodes, -1);

            for (int i = 0; i < local_nnodes[g]; i++) {
                int node = h_local_nodes[g][i];
                global_to_local[node] = i;
            }

            for (int i = 0; i < owned_nnodes[g]; i++) {
                int node = h_owned_nodes[g][i];
                int local = global_to_local[node];

                h_owned_to_local_map[g][i] = local;

                if (local >= 0) {
                    h_local_to_owned_map[g][local] = i;
                }
            }

            delete[] global_to_local;
        }
    }

    void move_maps_to_device() {
        d_node_gpu_ind = new int *[ngpus];
        std::memset(d_node_gpu_ind, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            CHECK_CUDA(cudaMalloc(&d_node_gpu_ind[g], num_nodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_node_gpu_ind[g], h_node_gpu_ind, num_nodes * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_owned_nodes[g], owned_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_owned_nodes[g], h_owned_nodes[g], owned_nnodes[g] * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_owned_to_local_map[g], owned_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_owned_to_local_map[g], h_owned_to_local_map[g],
                                  owned_nnodes[g] * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_to_owned_map[g], local_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_to_owned_map[g], h_local_to_owned_map[g],
                                  local_nnodes[g] * sizeof(int), cudaMemcpyHostToDevice));
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                if (srcdest_nnodes[idx] == 0) continue;

                CHECK_CUDA(cudaSetDevice(src));
                CHECK_CUDA(cudaMalloc(&d_srcred_map[idx], srcdest_nnodes[idx] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_srcred_map[idx], h_srcred_map[idx],
                                      srcdest_nnodes[idx] * sizeof(int), cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaSetDevice(dst));
                CHECK_CUDA(cudaMalloc(&d_dstred_map[idx], srcdest_nnodes[idx] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_dstred_map[idx], h_dstred_map[idx],
                                      srcdest_nnodes[idx] * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }

   public:
    int ngpus = 0;
    int num_nodes = 0;
    int num_elements = 0;
    int nodes_per_elem = 0;
    int num_components = 0;

    int *h_elem_conn = nullptr;
    int *h_elem_components = nullptr;

    bool debug = false;

    int *h_elem_assigned_gpu = nullptr;

    int *local_nelems = nullptr;
    int **h_local_elem_conn = nullptr;
    int **d_local_elem_conn = nullptr;

    int *h_node_gpu_ind = nullptr;
    int **d_node_gpu_ind = nullptr;

    int *owned_nnodes = nullptr;
    int *owned_N = nullptr;
    int **h_owned_nodes = nullptr;
    int **d_owned_nodes = nullptr;

    int *local_nnodes = nullptr;
    int *local_N = nullptr;
    int **h_local_nodes = nullptr;

    int **h_owned_to_local_map = nullptr;
    int **d_owned_to_local_map = nullptr;

    int **h_local_to_owned_map = nullptr;
    int **d_local_to_owned_map = nullptr;

    int *ghost_nnodes = nullptr;
    int *srcdest_nnodes = nullptr;
    int **h_srcred_map = nullptr;
    int **h_dstred_map = nullptr;
    int **d_srcred_map = nullptr;
    int **d_dstred_map = nullptr;

    bool **h_is_local_ghost = nullptr;
};