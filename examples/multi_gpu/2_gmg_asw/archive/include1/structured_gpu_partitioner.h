#pragma once
#include <algorithm>
#include <climits>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "cuda_utils.h"
#include "utils.h"

enum NodeType { INTERIOR = 0, INTERFACE = 1 };

class StructuredGPUPartitioner {
   public:
    StructuredGPUPartitioner(int ngpus_, int num_nodes_, int num_elements_, int nodes_per_elem_,
                             int *h_elem_conn_, bool debug_ = false)
        : ngpus(ngpus_),
          num_nodes(num_nodes_),
          num_elements(num_elements_),
          nodes_per_elem(nodes_per_elem_),
          h_elem_conn(h_elem_conn_),
          debug(debug_) {
        printf("temp\n");
        // printf("h_elem_conn[%d]: ", num_elements);
        // printVec<int>(4 * num_elements, h_elem_conn);

        printf("split element connectivity\n");
        split_elem_connectivity();
        printf("assign owned nodes\n");
        assign_owned_nodes();
        printf("build_owned_node_lists\n");
        build_owned_node_lists();
        printf("build_local_node_maps\n");
        build_local_node_maps();
        printf("build_ghost_node_maps\n");
        build_ghost_node_maps();
        printf("build_owned_local_maps\n");
        build_owned_local_maps();
        printf("move_maps_to_device\n");
        if (!debug) move_maps_to_device();
    }

    ~StructuredGPUPartitioner() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            if (d_local_elem_conn && d_local_elem_conn[g]) cudaFree(d_local_elem_conn[g]);
            if (d_node_gpu_ind && d_node_gpu_ind[g]) cudaFree(d_node_gpu_ind[g]);
            if (d_owned_nodes && d_owned_nodes[g]) cudaFree(d_owned_nodes[g]);
            if (d_owned_to_local_map && d_owned_to_local_map[g]) cudaFree(d_owned_to_local_map[g]);
            if (h_local_elem_conn && h_local_elem_conn[g]) delete[] h_local_elem_conn[g];
            if (h_owned_nodes && h_owned_nodes[g]) delete[] h_owned_nodes[g];
            if (h_local_nodes && h_local_nodes[g]) delete[] h_local_nodes[g];
            if (h_owned_to_local_map && h_owned_to_local_map[g]) delete[] h_owned_to_local_map[g];
        }

        int npairs = ngpus * ngpus;
        for (int i = 0; i < npairs; i++) {
            int src = i % ngpus;
            int dst = i / ngpus;

            CHECK_CUDA(cudaSetDevice(src));
            if (d_srcred_map && d_srcred_map[i]) cudaFree(d_srcred_map[i]);

            CHECK_CUDA(cudaSetDevice(dst));
            if (d_dstred_map && d_dstred_map[i]) cudaFree(d_dstred_map[i]);

            if (h_srcred_map && h_srcred_map[i]) delete[] h_srcred_map[i];
            if (h_dstred_map && h_dstred_map[i]) delete[] h_dstred_map[i];
        }

        delete[] start_elem;
        delete[] end_elem;
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
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }
    int getStartElem(const int g) { return start_elem[g]; }
    int getEndElem(const int g) { return end_elem[g]; }
    int getLocalNumElements(const int g) { return local_nelems[g]; }
    int find_owned_gpu_from_elem(int elem) const {
        for (int g = 0; g < ngpus; g++)
            if (start_elem[g] <= elem && elem < end_elem[g]) return g;
        return -1;
    }
    int getNumOwnedNodes(const int g) { return owned_nnodes[g]; }
    int getNumLocalNodes(const int g) { return local_nnodes[g]; }
    int *getOwnedNodesPtr(const int g) { return h_owned_nodes[g]; }
    int *getLocalNodesPtr(const int g) { return h_local_nodes[g]; }

   private:
    void split_elem_connectivity() {
        start_elem = new int[ngpus];
        end_elem = new int[ngpus];
        local_nelems = new int[ngpus];
        h_local_elem_conn = new int *[ngpus];
        d_local_elem_conn = new int *[ngpus];

        // printf("h_elem_conn[%d]: ", num_elements);
        // printVec<int>(num_elements * 4, h_elem_conn);

        for (int g = 0; g < ngpus; g++) {
            start_elem[g] = num_elements * g / ngpus;
            end_elem[g] = num_elements * (g + 1) / ngpus;
            local_nelems[g] = end_elem[g] - start_elem[g];

            int local_conn_size = local_nelems[g] * nodes_per_elem;
            int global_conn_offset = start_elem[g] * nodes_per_elem;

            h_local_elem_conn[g] = new int[local_conn_size];
            std::memcpy(h_local_elem_conn[g], &h_elem_conn[global_conn_offset],
                        local_conn_size * sizeof(int));

            printf("GPU[%d] - h_local_elem_conn[%d]: ", g, local_nelems[g]);
            printVec<int>(local_nelems[g] * 4, h_local_elem_conn[g]);

            if (!debug) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaMalloc(&d_local_elem_conn[g], local_conn_size * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_local_elem_conn[g], h_local_elem_conn[g],
                                      local_conn_size * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
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
        for (int n = 0; n < num_nodes; n++) h_ne_ptr[n + 1] = h_ne_ptr[n] + h_ne_cts[n];

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
            for (int ep = h_ne_ptr[n]; ep < h_ne_ptr[n + 1]; ep++)
                node_gpus.insert(find_owned_gpu_from_elem(h_ne_elems[ep]));

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
            for (int ep = h_ne_ptr[n]; ep < h_ne_ptr[n + 1]; ep++)
                node_gpus.insert(find_owned_gpu_from_elem(h_ne_elems[ep]));

            // std::vector<int> node_gpus_vec(node_gpus.begin(), node_gpus.end());
            // printf("node %d, node-gpus: ", n);
            // printVec<int>(node_gpus_vec.size(), node_gpus_vec.data());

            int best_gpu = -1, best_ct = INT_MAX;
            for (int g : node_gpus) {
                if (owned_node_cts[g] < best_ct) {
                    best_ct = owned_node_cts[g];
                    best_gpu = g;
                }
            }

            h_node_gpu_ind[n] = best_gpu;
            owned_node_cts[best_gpu]++;
        }

        // debug printout
        printf("h_node_gpu_ind: ");
        printVec<int>(num_nodes, h_node_gpu_ind);
        printf("owned_node_cts: ");
        printVec<int>(ngpus, owned_node_cts);

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

        for (int n = 0; n < num_nodes; n++) owned_nnodes[h_node_gpu_ind[n]]++;

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

        for (int g = 0; g < ngpus; g++) {
            printf("owned nodes on GPU[%d]\n", g);
            printVec<int>(owned_nnodes[g], h_owned_nodes[g]);
        }

        delete[] ct;
    }

    void build_local_node_maps() {
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        h_local_nodes = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> node_set;
            int *conn = h_local_elem_conn[g];
            int conn_size = local_nelems[g] * nodes_per_elem;

            for (int i = 0; i < conn_size; i++) node_set.insert(conn[i]);

            std::vector<int> nodes(node_set.begin(), node_set.end());
            std::sort(nodes.begin(), nodes.end());

            local_nnodes[g] = static_cast<int>(nodes.size());
            local_N[g] = local_nnodes[g];
            h_local_nodes[g] = new int[local_nnodes[g]];

            for (int i = 0; i < local_nnodes[g]; i++) h_local_nodes[g][i] = nodes[i];

            printf("local nodes on GPU[%d]\n", g);
            printVec<int>(local_nnodes[g], h_local_nodes[g]);
        }
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
            std::memset(global_to_local, -1, num_nodes * sizeof(int));

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
        std::memset(owned_pos, -1, num_nodes * sizeof(int));

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

            int src = idx % ngpus, dst = idx / ngpus;
            printf("h_srcredmap from GPU %d to %d\n", src, dst);
            printVec<int>(srcdest_nnodes[idx], h_srcred_map[idx]);

            printf("h_dstredmap from GPU %d to %d\n", src, dst);
            printVec<int>(srcdest_nnodes[idx], h_dstred_map[idx]);
        }

        delete[] owned_pos;
        delete[] src_maps;
        delete[] dst_maps;
    }

    void move_maps_to_device() {
        d_node_gpu_ind = new int *[ngpus];

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
    int ngpus, num_nodes, num_elements, nodes_per_elem;
    int *h_elem_conn = nullptr;
    bool debug;

    int *start_elem = nullptr;
    int *end_elem = nullptr;
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
};