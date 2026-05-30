#pragma once
#include <algorithm>
#include <cstring>
#include <vector>

#include "cuda_utils.h"
#include "utils.h"

template <class IEVPartition>
class SubdomainGPUPartitioner {
   public:
    SubdomainGPUPartitioner(int ngpus_, int num_nodes_, int *IE_nnodes_, int **IE_nodes_,
                            int **IEV_nodes_, IEVPartition *part_IEV_, bool same_IEV_order_ = true,
                            bool debug_ = false)
        : ngpus(ngpus_),
          num_nodes(num_nodes_),
          IE_nnodes_in(IE_nnodes_),
          IE_nodes_in(IE_nodes_),
          IEV_nodes_in(IEV_nodes_),
          part_IEV(part_IEV_),
          same_IEV_order(same_IEV_order_),
          debug(debug_) {
        // if (debug) printf("[SDPartition] begin ngpus=%d\n", ngpus);

        // if (debug) printf("[SDPartition] build_IE_offsets\n");
        build_IE_offsets();
        // if (debug) printf("[SDPartition] num_IE_nodes=%d\n", num_IE_nodes);

        // if (debug) printf("[SDPartition] build_IEV_to_IE_map\n");
        build_IEV_to_IE_map();

        // if (debug) printf("[SDPartition] build_owned_node_lists\n");
        build_owned_node_lists();

        // if (debug) printf("[SDPartition] build_local_node_lists\n");
        build_local_node_lists();

        // if (debug) {
        //     for (int g = 0; g < ngpus; g++) {
        //         printf("[SDPartition] gpu %d: owned=%d local=%d IE_in=%d IEV_local=%d\n", g,
        //                owned_nnodes[g], local_nnodes[g], IE_nnodes_in[g],
        //                part_IEV->getNumLocalNodes(g));
        //     }
        // }

        // if (debug) printf("[SDPartition] build_owned_local_maps\n");
        build_owned_local_maps();

        // if (debug) printf("[SDPartition] build_ghost_node_maps\n");
        build_ghost_node_maps();

        // if (debug) {
        //     int total_ghost = 0;
        //     for (int i = 0; i < ngpus * ngpus; i++) total_ghost += ghost_nnodes[i];
        //     printf("[SDPartition] total_ghost=%d\n", total_ghost);
        // }

        // if (debug) printf("[SDPartition] move_maps_to_device\n");
        move_maps_to_device();

        // if (debug) printf("[SDPartition] done\n");
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    int getNumOwnedNodes(const int g) const { return owned_nnodes[g]; }
    int getNumLocalNodes(const int g) const { return local_nnodes[g]; }

    int *getOwnedNodesPtr(const int g) const { return h_owned_nodes[g]; }
    int *getLocalNodesPtr(const int g) const { return h_local_nodes[g]; }

    int *getDeviceOwnedNodesPtr(const int g) const { return d_owned_nodes[g]; }
    int *getDeviceLocalNodesPtr(const int g) const { return d_local_nodes[g]; }

    int *getOwnedToLocalMapPtr(const int g) const { return h_owned_to_local_map[g]; }
    int *getLocalToOwnedMapPtr(const int g) const { return h_local_to_owned_map[g]; }

    int *getDeviceOwnedToLocalMapPtr(const int g) const { return d_owned_to_local_map[g]; }
    int *getDeviceLocalToOwnedMapPtr(const int g) const { return d_local_to_owned_map[g]; }

    int *getIEVToLocalMapPtr(const int g) const { return h_IEV_to_local_map[g]; }
    int *getLocalToIEVMapPtr(const int g) const { return h_local_to_IEV_map[g]; }
    int *getLocalToNRedMapPtr(const int g) const { return h_local_to_nred_map[g]; }

    int *getDeviceIEVToLocalMapPtr(const int g) const { return d_IEV_to_local_map[g]; }
    int *getDeviceLocalToIEVMapPtr(const int g) const { return d_local_to_IEV_map[g]; }
    int *getDeviceLocalToNRedMapPtr(const int g) const { return d_local_to_nred_map[g]; }

   private:
    void build_IE_offsets() {
        IE_offsets = new int[ngpus + 1];
        IE_offsets[0] = 0;

        for (int g = 0; g < ngpus; g++) {
            IE_offsets[g + 1] = IE_offsets[g] + IE_nnodes_in[g];
        }

        num_IE_nodes = IE_offsets[ngpus];
    }

    void build_IEV_to_IE_map() {
        num_IEV_nodes = 0;

        for (int g = 0; g < ngpus; g++) {
            int IEV_owned = part_IEV->getNumOwnedNodes(g);
            int IEV_local = part_IEV->getNumLocalNodes(g);

            int *IEV_owned_nodes = part_IEV->getOwnedNodesPtr(g);
            int *IEV_local_nodes = part_IEV->getLocalNodesPtr(g);

            for (int i = 0; i < IEV_owned; i++) {
                num_IEV_nodes = std::max(num_IEV_nodes, IEV_owned_nodes[i] + 1);
            }

            for (int i = 0; i < IEV_local; i++) {
                num_IEV_nodes = std::max(num_IEV_nodes, IEV_local_nodes[i] + 1);
            }
        }

        h_nred_to_IE = new int[num_IEV_nodes];
        h_node_gpu_ind = new int[num_IE_nodes];

        std::fill(h_nred_to_IE, h_nred_to_IE + num_IEV_nodes, -1);
        std::fill(h_node_gpu_ind, h_node_gpu_ind + num_IE_nodes, -1);

        for (int g = 0; g < ngpus; g++) {
            int IEV_owned = part_IEV->getNumOwnedNodes(g);
            int *IEV_owned_nodes = part_IEV->getOwnedNodesPtr(g);

            if (same_IEV_order) {
                int iev_cursor = 0;

                for (int i = 0; i < IE_nnodes_in[g]; i++) {
                    int ie = IE_offsets[g] + i;
                    int target_node = IE_nodes_in[g][i];

                    while (iev_cursor < IEV_owned && IEV_nodes_in[g][iev_cursor] != target_node) {
                        iev_cursor++;
                    }

                    if (iev_cursor >= IEV_owned) {
                        printf(
                            "[SDPartition] ERROR gpu %d: IE node %d not found in ordered owned IEV "
                            "list\n",
                            g, target_node);
                        exit(-1);
                    }

                    int iev_global = IEV_owned_nodes[iev_cursor];

                    h_nred_to_IE[iev_global] = ie;
                    h_node_gpu_ind[ie] = g;

                    iev_cursor++;
                }
            } else {
                for (int i = 0; i < IE_nnodes_in[g]; i++) {
                    int ie = IE_offsets[g] + i;
                    int target_node = IE_nodes_in[g][i];

                    int found = -1;

                    for (int iev = 0; iev < IEV_owned; iev++) {
                        if (IEV_nodes_in[g][iev] == target_node) {
                            found = iev;
                            break;
                        }
                    }

                    if (found < 0) {
                        printf(
                            "[SDPartition] ERROR gpu %d: IE node %d not found in unordered owned "
                            "IEV list\n",
                            g, target_node);
                        exit(-1);
                    }

                    int iev_global = IEV_owned_nodes[found];

                    h_nred_to_IE[iev_global] = ie;
                    h_node_gpu_ind[ie] = g;
                }
            }
        }
    }

    void build_owned_node_lists() {
        owned_nnodes = new int[ngpus];
        owned_N = new int[ngpus];

        h_owned_nodes = new int *[ngpus];
        d_owned_nodes = new int *[ngpus];

        std::memset(h_owned_nodes, 0, ngpus * sizeof(int *));
        std::memset(d_owned_nodes, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            std::vector<int> owned_ie_nodes;

            if (same_IEV_order) {
                // Original behavior: IE_nodes_in[g] order is trusted
                owned_ie_nodes.reserve(IE_nnodes_in[g]);

                for (int i = 0; i < IE_nnodes_in[g]; i++) {
                    owned_ie_nodes.push_back(IE_offsets[g] + i);
                }
            } else {
                // Important case:
                // Preserve the owned IEV ordering, which should already be E then V.
                // This makes the owned reduced vector use the same E+V split as local.
                int IEV_owned_nnodes = part_IEV->getNumOwnedNodes(g);
                int *h_IEV_owned_nodes = part_IEV->getOwnedNodesPtr(g);

                owned_ie_nodes.reserve(IE_nnodes_in[g]);

                for (int iev_owned = 0; iev_owned < IEV_owned_nnodes; iev_owned++) {
                    int iev_global = h_IEV_owned_nodes[iev_owned];

                    if (iev_global < 0 || iev_global >= num_IEV_nodes) continue;

                    int ie = h_nred_to_IE[iev_global];

                    if (ie < 0) continue;

                    // Only keep IE nodes owned by this subdomain
                    if (h_node_gpu_ind[ie] != g) continue;

                    owned_ie_nodes.push_back(ie);
                }

                if ((int)owned_ie_nodes.size() != IE_nnodes_in[g]) {
                    printf(
                        "[SDPartition] ERROR gpu %d: unordered owned E+V count mismatch: "
                        "got %d expected %d\n",
                        g, (int)owned_ie_nodes.size(), IE_nnodes_in[g]);
                    exit(-1);
                }
            }

            owned_nnodes[g] = static_cast<int>(owned_ie_nodes.size());
            owned_N[g] = owned_nnodes[g];

            h_owned_nodes[g] = new int[owned_nnodes[g]];

            for (int i = 0; i < owned_nnodes[g]; i++) {
                h_owned_nodes[g][i] = owned_ie_nodes[i];
            }
        }
    }

    // void build_owned_node_lists() {
    //     owned_nnodes = new int[ngpus];
    //     owned_N = new int[ngpus];

    //     h_owned_nodes = new int *[ngpus];
    //     d_owned_nodes = new int *[ngpus];

    //     std::memset(h_owned_nodes, 0, ngpus * sizeof(int *));
    //     std::memset(d_owned_nodes, 0, ngpus * sizeof(int *));

    //     for (int g = 0; g < ngpus; g++) {
    //         owned_nnodes[g] = IE_nnodes_in[g];
    //         owned_N[g] = owned_nnodes[g];

    //         h_owned_nodes[g] = new int[owned_nnodes[g]];

    //         for (int i = 0; i < owned_nnodes[g]; i++) {
    //             h_owned_nodes[g][i] = IE_offsets[g] + i;
    //         }
    //     }
    // }

    void build_local_node_lists() {
        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];

        h_local_nodes = new int *[ngpus];
        d_local_nodes = new int *[ngpus];

        h_IEV_to_local_map = new int *[ngpus];
        h_local_to_IEV_map = new int *[ngpus];
        h_local_to_nred_map = new int *[ngpus];

        d_IEV_to_local_map = new int *[ngpus];
        d_local_to_IEV_map = new int *[ngpus];
        d_local_to_nred_map = new int *[ngpus];

        std::memset(local_nnodes, 0, ngpus * sizeof(int));
        std::memset(h_local_nodes, 0, ngpus * sizeof(int *));
        std::memset(d_local_nodes, 0, ngpus * sizeof(int *));
        std::memset(h_IEV_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(h_local_to_IEV_map, 0, ngpus * sizeof(int *));
        std::memset(h_local_to_nred_map, 0, ngpus * sizeof(int *));
        std::memset(d_IEV_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(d_local_to_IEV_map, 0, ngpus * sizeof(int *));
        std::memset(d_local_to_nred_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            int IEV_local_nnodes = part_IEV->getNumLocalNodes(g);
            int *h_IEV_local_nodes = part_IEV->getLocalNodesPtr(g);

            h_IEV_to_local_map[g] = new int[IEV_local_nnodes];
            std::fill(h_IEV_to_local_map[g], h_IEV_to_local_map[g] + IEV_local_nnodes, -1);

            for (int iev_loc = 0; iev_loc < IEV_local_nnodes; iev_loc++) {
                int iev_global = h_IEV_local_nodes[iev_loc];

                if (iev_global >= 0 && iev_global < num_IEV_nodes &&
                    h_nred_to_IE[iev_global] >= 0) {
                    local_nnodes[g]++;
                }
            }

            local_N[g] = local_nnodes[g];

            h_local_nodes[g] = new int[local_nnodes[g]];
            h_local_to_IEV_map[g] = new int[local_nnodes[g]];
            h_local_to_nred_map[g] = new int[local_nnodes[g]];

            int ct = 0;

            for (int iev_loc = 0; iev_loc < IEV_local_nnodes; iev_loc++) {
                int iev_global = h_IEV_local_nodes[iev_loc];

                if (iev_global < 0 || iev_global >= num_IEV_nodes) continue;

                int ie = h_nred_to_IE[iev_global];

                if (ie < 0) continue;

                h_local_nodes[g][ct] = ie;
                h_local_to_IEV_map[g][ct] = iev_loc;
                h_local_to_nred_map[g][ct] = iev_global;
                h_IEV_to_local_map[g][iev_loc] = ct;

                ct++;
            }
        }
    }

    void build_owned_local_maps() {
        h_owned_to_local_map = new int *[ngpus];
        h_local_to_owned_map = new int *[ngpus];

        d_owned_to_local_map = new int *[ngpus];
        d_local_to_owned_map = new int *[ngpus];

        std::memset(h_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(h_local_to_owned_map, 0, ngpus * sizeof(int *));
        std::memset(d_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(d_local_to_owned_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            h_owned_to_local_map[g] = new int[owned_nnodes[g]];
            h_local_to_owned_map[g] = new int[local_nnodes[g]];

            std::fill(h_owned_to_local_map[g], h_owned_to_local_map[g] + owned_nnodes[g], -1);
            std::fill(h_local_to_owned_map[g], h_local_to_owned_map[g] + local_nnodes[g], -1);

            int *IE_to_local = new int[num_IE_nodes];
            std::fill(IE_to_local, IE_to_local + num_IE_nodes, -1);

            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                int ie = h_local_nodes[g][loc];

                if (ie < 0 || ie >= num_IE_nodes) {
                    printf("[SDPartition] ERROR gpu %d: bad local IE=%d\n", g, ie);
                    exit(-1);
                }

                IE_to_local[ie] = loc;
            }

            for (int i = 0; i < owned_nnodes[g]; i++) {
                int ie = h_owned_nodes[g][i];
                int loc = IE_to_local[ie];

                h_owned_to_local_map[g][i] = loc;

                if (loc >= 0) {
                    h_local_to_owned_map[g][loc] = i;
                }
            }

            delete[] IE_to_local;
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

        int *owned_pos = new int[num_IE_nodes];
        std::fill(owned_pos, owned_pos + num_IE_nodes, -1);

        for (int g = 0; g < ngpus; g++) {
            for (int i = 0; i < owned_nnodes[g]; i++) {
                owned_pos[h_owned_nodes[g][i]] = i;
            }
        }

        std::vector<int> *src_maps = new std::vector<int>[npairs];
        std::vector<int> *dst_maps = new std::vector<int>[npairs];

        for (int dst = 0; dst < ngpus; dst++) {
            for (int dst_loc = 0; dst_loc < local_nnodes[dst]; dst_loc++) {
                int ie = h_local_nodes[dst][dst_loc];
                int src = h_node_gpu_ind[ie];

                if (src == dst) continue;

                int idx = pair_index(dst, src);

                src_maps[idx].push_back(owned_pos[ie]);
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

    void move_maps_to_device() {
        d_node_gpu_ind = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            int IEV_local_nnodes = part_IEV->getNumLocalNodes(g);

            CHECK_CUDA(cudaSetDevice(g));

            CHECK_CUDA(cudaMalloc(&d_node_gpu_ind[g], num_IE_nodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_node_gpu_ind[g], h_node_gpu_ind, num_IE_nodes * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_owned_nodes[g], owned_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_owned_nodes[g], h_owned_nodes[g], owned_nnodes[g] * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_nodes[g], local_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_nodes[g], h_local_nodes[g], local_nnodes[g] * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_owned_to_local_map[g], owned_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_owned_to_local_map[g], h_owned_to_local_map[g],
                                  owned_nnodes[g] * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_to_owned_map[g], local_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_to_owned_map[g], h_local_to_owned_map[g],
                                  local_nnodes[g] * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_IEV_to_local_map[g], IEV_local_nnodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_IEV_to_local_map[g], h_IEV_to_local_map[g],
                                  IEV_local_nnodes * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_to_IEV_map[g], local_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_to_IEV_map[g], h_local_to_IEV_map[g],
                                  local_nnodes[g] * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc(&d_local_to_nred_map[g], local_nnodes[g] * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_local_to_nred_map[g], h_local_to_nred_map[g],
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
    int num_IE_nodes = 0;
    int num_IEV_nodes = 0;
    bool same_IEV_order = true;

    int *IE_nnodes_in = nullptr;
    int **IE_nodes_in = nullptr;
    int **IEV_nodes_in = nullptr;

    IEVPartition *part_IEV = nullptr;
    bool debug = false;

    int *IE_offsets = nullptr;
    int *h_nred_to_IE = nullptr;
    int *h_node_gpu_ind = nullptr;
    int **d_node_gpu_ind = nullptr;

    int *owned_nnodes = nullptr;
    int *owned_N = nullptr;
    int **h_owned_nodes = nullptr;
    int **d_owned_nodes = nullptr;

    int *local_nnodes = nullptr;
    int *local_N = nullptr;
    int **h_local_nodes = nullptr;
    int **d_local_nodes = nullptr;

    int **h_owned_to_local_map = nullptr;
    int **d_owned_to_local_map = nullptr;
    int **h_local_to_owned_map = nullptr;
    int **d_local_to_owned_map = nullptr;

    int **h_IEV_to_local_map = nullptr;
    int **d_IEV_to_local_map = nullptr;
    int **h_local_to_IEV_map = nullptr;
    int **d_local_to_IEV_map = nullptr;
    int **h_local_to_nred_map = nullptr;
    int **d_local_to_nred_map = nullptr;

    int *ghost_nnodes = nullptr;
    int *srcdest_nnodes = nullptr;

    int **h_srcred_map = nullptr;
    int **h_dstred_map = nullptr;
    int **d_srcred_map = nullptr;
    int **d_dstred_map = nullptr;
};