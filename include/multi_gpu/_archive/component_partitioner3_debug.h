#pragma once
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "cuda_utils.h"
#include "utils.h"

class TacsComponentGPUPartitioner {
   public:
    TacsComponentGPUPartitioner(int ngpus_, int num_nodes_, int num_elements_, int nodes_per_elem_,
                                int *h_elem_conn_, int num_components_, int *h_elem_components_,
                                bool debug_ = false)
        : ngpus(ngpus_),
          num_nodes(num_nodes_),
          num_elements(num_elements_),
          nodes_per_elem(nodes_per_elem_),
          num_components(num_components_),
          h_elem_conn(h_elem_conn_),
          h_elem_components(h_elem_components_),
          debug(debug_) {
        printf("\n========== TacsComponentGPUPartitioner constructor START ==========\n");
        printf(
            "[PART ctor] ngpus=%d num_nodes=%d num_elements=%d nodes_per_elem=%d num_components=%d "
            "debug=%d\n",
            ngpus, num_nodes, num_elements, nodes_per_elem, num_components, (int)debug);
        check_all_devices("constructor entry", true);

        printf("[PART ctor] split_elem_connectivity\n");
        split_elem_connectivity();
        check_all_devices("after split_elem_connectivity", true);

        printf("[PART ctor] assign_owned_nodes\n");
        assign_owned_nodes();
        check_all_devices("after assign_owned_nodes", true);

        printf("[PART ctor] build_owned_node_lists\n");
        build_owned_node_lists();
        check_all_devices("after build_owned_node_lists", true);

        printf("[PART ctor] build_local_node_maps\n");
        build_local_node_maps();
        check_all_devices("after build_local_node_maps", true);

        printf("[PART ctor] build_local_ghost_flags\n");
        build_local_ghost_flags();
        check_all_devices("after build_local_ghost_flags", true);

        printf("[PART ctor] build_ghost_node_maps\n");
        build_ghost_node_maps();
        check_all_devices("after build_ghost_node_maps", true);

        printf("[PART ctor] build_owned_local_maps\n");
        build_owned_local_maps();
        check_all_devices("after build_owned_local_maps", true);

        if (!debug) {
            printf("[PART ctor] move_maps_to_device\n");
            move_maps_to_device();
            check_all_devices("after move_maps_to_device", true);
        }

        debugCheck(false);
        printf("========== TacsComponentGPUPartitioner constructor DONE ==========\n\n");
    }

    void free() {
        printf("\n[PART free] START\n");

        for (int g = 0; g < ngpus; g++) {
            int dev = debug ? 0 : g;
            printf("[PART free] GPU[%d] dev=%d\n", g, dev);
            CHECK_CUDA(cudaSetDevice(dev));

            if (d_local_elem_conn && d_local_elem_conn[g]) cudaFree(d_local_elem_conn[g]);
            if (d_local_elems && d_local_elems[g]) cudaFree(d_local_elems[g]);
            if (d_node_gpu_ind && d_node_gpu_ind[g]) cudaFree(d_node_gpu_ind[g]);
            if (d_owned_nodes && d_owned_nodes[g]) cudaFree(d_owned_nodes[g]);
            if (d_owned_to_local_map && d_owned_to_local_map[g]) cudaFree(d_owned_to_local_map[g]);
            if (d_local_to_owned_map && d_local_to_owned_map[g]) cudaFree(d_local_to_owned_map[g]);

            if (h_local_elem_conn && h_local_elem_conn[g]) delete[] h_local_elem_conn[g];
            if (h_local_elems && h_local_elems[g]) delete[] h_local_elems[g];
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

            CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
            if (d_srcred_map && d_srcred_map[idx]) cudaFree(d_srcred_map[idx]);

            CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
            if (d_dstred_map && d_dstred_map[idx]) cudaFree(d_dstred_map[idx]);

            if (h_srcred_map && h_srcred_map[idx]) delete[] h_srcred_map[idx];
            if (h_dstred_map && h_dstred_map[idx]) delete[] h_dstred_map[idx];
        }

        delete[] h_elem_assigned_gpu;
        delete[] local_nelems;
        delete[] h_local_elem_conn;
        delete[] d_local_elem_conn;
        delete[] h_local_elems;
        delete[] d_local_elems;
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

        printf("[PART free] DONE\n\n");
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
    int *getLocalElemConnPtr(const int g) const { return h_local_elem_conn[g]; }
    int *getDeviceLocalElemConnPtr(const int g) const { return d_local_elem_conn[g]; }
    int *getLocalElementsPtr(const int g) const { return h_local_elems[g]; }
    int *getDeviceLocalElementsPtr(const int g) const { return d_local_elems[g]; }

    void setElementComponents(const int *h_elem_comp, int **h_loc_elem_comp,
                              int **d_loc_elem_comp) {
        printf("\n[PART setElementComponents] START\n");
        check_all_devices("setElementComponents entry", true);

        if (!h_elem_comp) {
            printf("[PART setElementComponents] ERROR h_elem_comp is null\n");
            exit(1);
        }
        if (!h_loc_elem_comp || !d_loc_elem_comp) {
            printf("[PART setElementComponents] ERROR output ptr arrays null h=%p d=%p\n",
                   (void *)h_loc_elem_comp, (void *)d_loc_elem_comp);
            exit(1);
        }

        for (int g = 0; g < ngpus; g++) {
            int nloc = local_nelems[g];
            printf("[PART setElementComponents] GPU[%d] nloc=%d h_local_elems=%p\n", g, nloc,
                   (void *)h_local_elems[g]);

            h_loc_elem_comp[g] = nullptr;
            d_loc_elem_comp[g] = nullptr;

            if (nloc <= 0) {
                printf("[PART setElementComponents] GPU[%d] nloc <= 0, skip\n", g);
                continue;
            }

            h_loc_elem_comp[g] = new int[nloc];

            for (int le = 0; le < nloc; le++) {
                int e = h_local_elems[g][le];

                if (e < 0 || e >= num_elements) {
                    printf("[PART setElementComponents] BAD e GPU[%d] le=%d e=%d num_elements=%d\n",
                           g, le, e, num_elements);
                    exit(1);
                }

                h_loc_elem_comp[g][le] = h_elem_comp[e];

                if (le < 5) {
                    printf("[PART setElementComponents] GPU[%d] le=%d global_e=%d comp=%d\n", g, le,
                           e, h_loc_elem_comp[g][le]);
                }
            }

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            check_cuda("setElementComponents before malloc", g, true);

            size_t bytes = (size_t)nloc * sizeof(int);
            printf("[PART setElementComponents] GPU[%d] malloc/copy bytes=%zu\n", g, bytes);

            CHECK_CUDA(cudaMalloc((void **)&d_loc_elem_comp[g], bytes));
            CHECK_CUDA(
                cudaMemcpy(d_loc_elem_comp[g], h_loc_elem_comp[g], bytes, cudaMemcpyHostToDevice));

            check_cuda("setElementComponents after copy", g, true);
        }

        check_all_devices("setElementComponents exit", true);
        printf("[PART setElementComponents] DONE\n\n");
    }

    bool debugCheck(bool print_maps = true) const {
        bool ok = true;

        printf("\n========== TacsComponentGPUPartitioner debugCheck ==========\n");
        printf(
            "ngpus=%d num_nodes=%d num_elements=%d nodes_per_elem=%d num_components=%d debug=%d\n",
            ngpus, num_nodes, num_elements, nodes_per_elem, num_components, (int)debug);

        int total_local_elems = 0;
        std::vector<int> elem_seen(num_elements, 0);

        for (int g = 0; g < ngpus; g++) {
            total_local_elems += local_nelems[g];

            printf("GPU[%d]: local_nelems=%d owned_nnodes=%d local_nnodes=%d\n", g, local_nelems[g],
                   owned_nnodes[g], local_nnodes[g]);

            if (print_maps) {
                printf("  local elems first 20: ");
                for (int i = 0; i < local_nelems[g] && i < 20; i++)
                    printf(" %d", h_local_elems[g][i]);
                printf("\n");

                printf("  owned nodes first 20: ");
                for (int i = 0; i < owned_nnodes[g] && i < 20; i++)
                    printf(" %d", h_owned_nodes[g][i]);
                printf("\n");

                printf("  local nodes first 20: ");
                for (int i = 0; i < local_nnodes[g] && i < 20; i++)
                    printf(" %d", h_local_nodes[g][i]);
                printf("\n");
            }

            for (int le = 0; le < local_nelems[g]; le++) {
                int e = h_local_elems[g][le];

                if (e < 0 || e >= num_elements) {
                    printf("[BAD] GPU[%d] local elem %d has invalid global elem %d\n", g, le, e);
                    ok = false;
                    continue;
                }

                elem_seen[e]++;

                if (h_elem_assigned_gpu[e] != g) {
                    printf("[BAD] elem %d local on GPU[%d], assigned GPU[%d]\n", e, g,
                           h_elem_assigned_gpu[e]);
                    ok = false;
                }

                for (int a = 0; a < nodes_per_elem; a++) {
                    int n = h_local_elem_conn[g][le * nodes_per_elem + a];
                    if (n < 0 || n >= num_nodes) {
                        printf("[BAD] GPU[%d] elem %d node slot %d invalid node %d\n", g, e, a, n);
                        ok = false;
                    }
                }
            }

            for (int i = 0; i < owned_nnodes[g]; i++) {
                int n = h_owned_nodes[g][i];
                if (n < 0 || n >= num_nodes || h_node_gpu_ind[n] != g) {
                    printf("[BAD] GPU[%d] owned node entry %d node=%d owner=%d\n", g, i, n,
                           (n >= 0 && n < num_nodes) ? h_node_gpu_ind[n] : -999);
                    ok = false;
                }
            }

            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                int n = h_local_nodes[g][loc];
                if (n < 0 || n >= num_nodes) {
                    printf("[BAD] GPU[%d] local node %d invalid global node %d\n", g, loc, n);
                    ok = false;
                }
            }
        }

        if (total_local_elems != num_elements) {
            printf("[BAD] total local elems = %d, expected %d\n", total_local_elems, num_elements);
            ok = false;
        }

        for (int e = 0; e < num_elements; e++) {
            if (elem_seen[e] != 1) {
                printf("[BAD] global elem %d appears %d times\n", e, elem_seen[e]);
                ok = false;
            }
            int c = h_elem_components[e];
            if (c < 0 || c >= num_components) {
                printf("[BAD] elem %d has invalid component %d\n", e, c);
                ok = false;
            }
        }

        int owned_total = 0;
        for (int g = 0; g < ngpus; g++) owned_total += owned_nnodes[g];

        if (owned_total != num_nodes) {
            printf("[BAD] total owned nodes = %d, expected %d\n", owned_total, num_nodes);
            ok = false;
        }

        int npairs = ngpus * ngpus;
        int total_ghost_pairs = 0;

        for (int idx = 0; idx < npairs; idx++) {
            int dst = idx / ngpus;
            int src = idx % ngpus;
            if (dst == src) continue;

            total_ghost_pairs += srcdest_nnodes[idx];

            if (srcdest_nnodes[idx] > 0) {
                printf("pair dst=%d src=%d srcdest_nnodes=%d\n", dst, src, srcdest_nnodes[idx]);
            }

            for (int k = 0; k < srcdest_nnodes[idx]; k++) {
                int src_owned = h_srcred_map[idx][k];
                int dst_loc = h_dstred_map[idx][k];

                if (src_owned < 0 || src_owned >= owned_nnodes[src]) {
                    printf("[BAD] pair dst=%d src=%d k=%d bad src_owned=%d\n", dst, src, k,
                           src_owned);
                    ok = false;
                    continue;
                }

                if (dst_loc < 0 || dst_loc >= local_nnodes[dst]) {
                    printf("[BAD] pair dst=%d src=%d k=%d bad dst_loc=%d\n", dst, src, k, dst_loc);
                    ok = false;
                    continue;
                }

                int src_node = h_owned_nodes[src][src_owned];
                int dst_node = h_local_nodes[dst][dst_loc];

                if (src_node != dst_node) {
                    printf("[BAD] pair dst=%d src=%d k=%d src_node=%d dst_node=%d mismatch\n", dst,
                           src, k, src_node, dst_node);
                    ok = false;
                }

                if (h_node_gpu_ind[dst_node] != src) {
                    printf("[BAD] pair dst=%d src=%d node=%d owner=%d\n", dst, src, dst_node,
                           h_node_gpu_ind[dst_node]);
                    ok = false;
                }
            }
        }

        printf("total ghost map entries = %d\n", total_ghost_pairs);
        printf("debugCheck result: %s\n", ok ? "PASS" : "FAIL");
        printf("============================================================\n\n");
        return ok;
    }

   private:
    void check_cuda(const char *stage, int g = -1, bool clear = true) const {
        cudaError_t err = clear ? cudaGetLastError() : cudaPeekAtLastError();
        if (err != cudaSuccess) {
            printf("[COMP PART CUDA ERROR] stage=%s GPU[%d] dev=%d error=%s\n", stage, g,
                   (g >= 0 ? (debug ? 0 : g) : -1), cudaGetErrorString(err));
            exit(1);
        }
    }

    void check_all_devices(const char *stage, bool clear = true) const {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            check_cuda(stage, g, clear);
        }
    }

    void check_component_ids() const {
        printf("[PART check_component_ids] START\n");

        if (!h_elem_components) {
            printf("[PART check_component_ids] ERROR h_elem_components null\n");
            exit(1);
        }

        for (int e = 0; e < num_elements; e++) {
            int c = h_elem_components[e];
            if (c < 0 || c >= num_components) {
                printf("[PART check_component_ids] BAD elem %d comp %d num_components %d\n", e, c,
                       num_components);
                exit(1);
            }
        }

        printf("[PART check_component_ids] DONE\n");
    }

    void check_elem_conn() const {
        printf("[PART check_elem_conn] START\n");

        if (!h_elem_conn) {
            printf("[PART check_elem_conn] ERROR h_elem_conn null\n");
            exit(1);
        }

        int bad = 0;
        for (int e = 0; e < num_elements; e++) {
            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];
                if (node < 0 || node >= num_nodes) {
                    if (bad < 20) {
                        printf("[PART check_elem_conn] BAD e=%d a=%d node=%d num_nodes=%d\n", e, a,
                               node, num_nodes);
                    }
                    bad++;
                }
            }
        }

        if (bad) {
            printf("[PART check_elem_conn] ERROR bad conn count=%d\n", bad);
            exit(1);
        }

        printf("[PART check_elem_conn] DONE\n");
    }

    int get_num_interface_nodes(const std::vector<int> &current_comps, const int *n2e_ptr,
                                const int *n2e_vals) const {
        std::vector<char> in_comp(num_components, 0);
        for (int icomp : current_comps) in_comp[icomp] = 1;

        int num_interface = 0;

        for (int node = 0; node < num_nodes; node++) {
            bool has_inside = false;
            bool has_outside = false;

            for (int ip = n2e_ptr[node]; ip < n2e_ptr[node + 1]; ip++) {
                int e = n2e_vals[ip];
                int c = h_elem_components[e];

                if (in_comp[c])
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
        printf("\n[PART split_elem_connectivity] START\n");
        check_component_ids();
        check_elem_conn();

        local_nelems = new int[ngpus];
        h_local_elem_conn = new int *[ngpus];
        d_local_elem_conn = new int *[ngpus];
        h_local_elems = new int *[ngpus];
        d_local_elems = new int *[ngpus];
        h_elem_assigned_gpu = new int[num_elements];

        std::memset(local_nelems, 0, ngpus * sizeof(int));
        std::memset(h_local_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(d_local_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(h_local_elems, 0, ngpus * sizeof(int *));
        std::memset(d_local_elems, 0, ngpus * sizeof(int *));
        std::fill(h_elem_assigned_gpu, h_elem_assigned_gpu + num_elements, -1);

        printf("[PART split] build n2e counts\n");

        int *n2e_cts = new int[num_nodes];
        std::memset(n2e_cts, 0, num_nodes * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];
                n2e_cts[node]++;
            }
        }

        int *n2e_ptr = new int[num_nodes + 1];
        n2e_ptr[0] = 0;
        for (int n = 0; n < num_nodes; n++) n2e_ptr[n + 1] = n2e_ptr[n] + n2e_cts[n];

        int n2e_nnz = n2e_ptr[num_nodes];
        printf("[PART split] n2e_nnz=%d\n", n2e_nnz);

        int *n2e_vals = new int[n2e_nnz];
        std::memset(n2e_cts, 0, num_nodes * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];
                int off = n2e_ptr[node] + n2e_cts[node]++;
                n2e_vals[off] = e;
            }
        }

        printf("[PART split] build e2e counts\n");

        int *e2e_cts = new int[num_elements];
        std::memset(e2e_cts, 0, num_elements * sizeof(int));

        std::vector<int> ctr(num_elements, 0);
        std::vector<int> touched;
        touched.reserve(128);

        int max_e2e = 0;

        for (int e = 0; e < num_elements; e++) {
            touched.clear();

            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];

                for (int jp = n2e_ptr[node]; jp < n2e_ptr[node + 1]; jp++) {
                    int je = n2e_vals[jp];
                    if (je == e) continue;

                    if (ctr[je] == 0) touched.push_back(je);
                    ctr[je]++;

                    if (ctr[je] == 2) e2e_cts[e]++;
                }
            }

            max_e2e = std::max(max_e2e, e2e_cts[e]);
            for (int je : touched) ctr[je] = 0;
        }

        int *e2e_ptr = new int[num_elements + 1];
        e2e_ptr[0] = 0;
        for (int e = 0; e < num_elements; e++) e2e_ptr[e + 1] = e2e_ptr[e] + e2e_cts[e];

        int e2e_nnz = e2e_ptr[num_elements];
        printf("[PART split] e2e_nnz=%d max_e2e=%d\n", e2e_nnz, max_e2e);

        int *e2e_vals = new int[e2e_nnz];
        std::memset(e2e_cts, 0, num_elements * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            touched.clear();

            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];

                for (int jp = n2e_ptr[node]; jp < n2e_ptr[node + 1]; jp++) {
                    int je = n2e_vals[jp];
                    if (je == e) continue;

                    if (ctr[je] == 0) touched.push_back(je);
                    ctr[je]++;

                    if (ctr[je] == 2) {
                        int off = e2e_ptr[e] + e2e_cts[e]++;
                        e2e_vals[off] = je;
                    }
                }
            }

            for (int je : touched) ctr[je] = 0;
        }

        printf("[PART split] build c2c\n");

        int *c2c_ctr = new int[num_components];
        std::memset(c2c_ctr, 0, num_components * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            int c = h_elem_components[e];
            std::vector<char> seen(num_components, 0);

            for (int jp = e2e_ptr[e]; jp < e2e_ptr[e + 1]; jp++) {
                int je = e2e_vals[jp];
                int jc = h_elem_components[je];

                if (c == jc) continue;

                if (!seen[jc]) {
                    seen[jc] = 1;
                    c2c_ctr[c]++;
                }
            }
        }

        int *c2c_ptr = new int[num_components + 1];
        c2c_ptr[0] = 0;
        for (int c = 0; c < num_components; c++) c2c_ptr[c + 1] = c2c_ptr[c] + c2c_ctr[c];

        int c2c_nnz = c2c_ptr[num_components];
        printf("[PART split] c2c_nnz=%d\n", c2c_nnz);

        int *c2c_vals = new int[c2c_nnz];
        std::memset(c2c_ctr, 0, num_components * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            int c = h_elem_components[e];
            std::vector<char> seen(num_components, 0);

            for (int jp = e2e_ptr[e]; jp < e2e_ptr[e + 1]; jp++) {
                int je = e2e_vals[jp];
                int jc = h_elem_components[je];

                if (c == jc) continue;

                if (!seen[jc]) {
                    seen[jc] = 1;
                    int off = c2c_ptr[c] + c2c_ctr[c]++;
                    c2c_vals[off] = jc;
                }
            }
        }

        int *comp_gpu = new int[num_components];
        std::fill(comp_gpu, comp_gpu + num_components, -1);

        std::vector<int> comp_elem_cts(num_components, 0);
        std::vector<int> gpu_load(ngpus, 0);
        std::vector<std::vector<int>> gpu_comps(ngpus);

        for (int e = 0; e < num_elements; e++) comp_elem_cts[h_elem_components[e]]++;

        int max_gpu_load = (num_elements + ngpus - 1) / ngpus;
        printf("[PART split] assign components max_gpu_load=%d\n", max_gpu_load);

        for (int seed = 0; seed < num_components; seed++) {
            if (comp_gpu[seed] >= 0) continue;

            int best_gpu = 0;
            for (int g = 1; g < ngpus; g++) {
                if (gpu_load[g] < gpu_load[best_gpu]) best_gpu = g;
            }

            comp_gpu[seed] = best_gpu;
            gpu_load[best_gpu] += comp_elem_cts[seed];
            gpu_comps[best_gpu].push_back(seed);

            printf("[PART split] seed comp=%d elems=%d -> GPU[%d]\n", seed, comp_elem_cts[seed],
                   best_gpu);

            bool added = true;
            while (added) {
                added = false;
                int best_comp = -1;
                int best_interface = INT_MAX;
                int best_conn = -1;

                for (int c = 0; c < num_components; c++) {
                    if (comp_gpu[c] >= 0) continue;
                    if (gpu_load[best_gpu] + comp_elem_cts[c] > max_gpu_load) continue;

                    int conn = 0;
                    for (int jp = c2c_ptr[c]; jp < c2c_ptr[c + 1]; jp++) {
                        int jc = c2c_vals[jp];
                        if (comp_gpu[jc] == best_gpu) conn++;
                    }

                    if (conn == 0) continue;

                    std::vector<int> trial = gpu_comps[best_gpu];
                    trial.push_back(c);
                    int interface = get_num_interface_nodes(trial, n2e_ptr, n2e_vals);

                    if (interface < best_interface ||
                        (interface == best_interface && conn > best_conn)) {
                        best_interface = interface;
                        best_conn = conn;
                        best_comp = c;
                    }
                }

                if (best_comp >= 0) {
                    comp_gpu[best_comp] = best_gpu;
                    gpu_load[best_gpu] += comp_elem_cts[best_comp];
                    gpu_comps[best_gpu].push_back(best_comp);
                    added = true;

                    printf(
                        "[PART split] add comp=%d elems=%d -> GPU[%d] load=%d interface=%d "
                        "conn=%d\n",
                        best_comp, comp_elem_cts[best_comp], best_gpu, gpu_load[best_gpu],
                        best_interface, best_conn);
                }
            }
        }

        for (int c = 0; c < num_components; c++) {
            if (comp_gpu[c] >= 0) continue;

            int best_gpu = 0;
            int best_interface = INT_MAX;

            for (int g = 0; g < ngpus; g++) {
                std::vector<int> trial = gpu_comps[g];
                trial.push_back(c);
                int interface = get_num_interface_nodes(trial, n2e_ptr, n2e_vals);

                if (interface < best_interface ||
                    (interface == best_interface && gpu_load[g] < gpu_load[best_gpu])) {
                    best_interface = interface;
                    best_gpu = g;
                }
            }

            comp_gpu[c] = best_gpu;
            gpu_load[best_gpu] += comp_elem_cts[c];
            gpu_comps[best_gpu].push_back(c);

            printf("[PART split] fallback comp=%d elems=%d -> GPU[%d] load=%d interface=%d\n", c,
                   comp_elem_cts[c], best_gpu, gpu_load[best_gpu], best_interface);
        }

        for (int c = 0; c < num_components; c++) {
            printf("[PART split] comp %d elems=%d assigned GPU[%d]\n", c, comp_elem_cts[c],
                   comp_gpu[c]);
        }

        for (int e = 0; e < num_elements; e++) {
            int g = comp_gpu[h_elem_components[e]];
            h_elem_assigned_gpu[e] = g;
            local_nelems[g]++;
        }

        for (int g = 0; g < ngpus; g++) {
            printf("[PART split] GPU[%d] local_nelems=%d gpu_load=%d\n", g, local_nelems[g],
                   gpu_load[g]);
        }

        for (int g = 0; g < ngpus; g++) {
            h_local_elem_conn[g] = nullptr;
            h_local_elems[g] = nullptr;

            if (local_nelems[g] > 0) {
                h_local_elem_conn[g] = new int[local_nelems[g] * nodes_per_elem];
                h_local_elems[g] = new int[local_nelems[g]];
            }

            if (!debug && local_nelems[g] > 0) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                check_cuda("split before local elem cudaMalloc", g, true);

                size_t conn_bytes = (size_t)local_nelems[g] * nodes_per_elem * sizeof(int);
                size_t elem_bytes = (size_t)local_nelems[g] * sizeof(int);

                printf(
                    "[PART split] GPU[%d] malloc d_local_elem_conn bytes=%zu d_local_elems "
                    "bytes=%zu\n",
                    g, conn_bytes, elem_bytes);

                CHECK_CUDA(cudaMalloc((void **)&d_local_elem_conn[g], conn_bytes));
                CHECK_CUDA(cudaMalloc((void **)&d_local_elems[g], elem_bytes));

                check_cuda("split after local elem cudaMalloc", g, true);
            }
        }

        std::vector<int> local_elem_cts(ngpus, 0);

        for (int e = 0; e < num_elements; e++) {
            int g = h_elem_assigned_gpu[e];
            int le = local_elem_cts[g]++;

            h_local_elems[g][le] = e;
            std::memcpy(&h_local_elem_conn[g][le * nodes_per_elem],
                        &h_elem_conn[e * nodes_per_elem], nodes_per_elem * sizeof(int));
        }

        if (!debug) {
            for (int g = 0; g < ngpus; g++) {
                if (local_nelems[g] <= 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                check_cuda("split before local elem cudaMemcpy", g, true);

                size_t conn_bytes = (size_t)local_nelems[g] * nodes_per_elem * sizeof(int);
                size_t elem_bytes = (size_t)local_nelems[g] * sizeof(int);

                printf("[PART split] GPU[%d] copy local elem conn bytes=%zu elems bytes=%zu\n", g,
                       conn_bytes, elem_bytes);

                CHECK_CUDA(cudaMemcpy(d_local_elem_conn[g], h_local_elem_conn[g], conn_bytes,
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_local_elems[g], h_local_elems[g], elem_bytes,
                                      cudaMemcpyHostToDevice));

                check_cuda("split after local elem cudaMemcpy", g, true);
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

        printf("[PART split_elem_connectivity] DONE\n\n");
    }

    void assign_owned_nodes() {
        printf("\n[PART assign_owned_nodes] START\n");

        int *h_ne_cts = new int[num_nodes];
        std::memset(h_ne_cts, 0, num_nodes * sizeof(int));

        int ne_nnz = 0;
        for (int e = 0; e < num_elements; e++) {
            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];
                h_ne_cts[node]++;
                ne_nnz++;
            }
        }

        printf("[PART assign_owned_nodes] ne_nnz=%d\n", ne_nnz);

        int *h_ne_ptr = new int[num_nodes + 1];
        h_ne_ptr[0] = 0;
        for (int n = 0; n < num_nodes; n++) h_ne_ptr[n + 1] = h_ne_ptr[n] + h_ne_cts[n];

        int *h_ne_elems = new int[ne_nnz];
        std::memset(h_ne_cts, 0, num_nodes * sizeof(int));

        for (int e = 0; e < num_elements; e++) {
            for (int a = 0; a < nodes_per_elem; a++) {
                int node = h_elem_conn[e * nodes_per_elem + a];
                int off = h_ne_ptr[node] + h_ne_cts[node]++;
                h_ne_elems[off] = e;
            }
        }

        h_node_gpu_ind = new int[num_nodes];

        int *owned_node_cts = new int[ngpus];
        std::memset(owned_node_cts, 0, ngpus * sizeof(int));

        int interface_nodes = 0;
        int isolated_nodes = 0;

        for (int n = 0; n < num_nodes; n++) {
            std::unordered_set<int> node_gpus;

            for (int ep = h_ne_ptr[n]; ep < h_ne_ptr[n + 1]; ep++) {
                int g = find_owned_gpu_from_elem(h_ne_elems[ep]);
                if (g >= 0) node_gpus.insert(g);
            }

            if (node_gpus.empty()) {
                h_node_gpu_ind[n] = 0;
                owned_node_cts[0]++;
                isolated_nodes++;
            } else if (node_gpus.size() == 1) {
                int g = *node_gpus.begin();
                h_node_gpu_ind[n] = g;
                owned_node_cts[g]++;
            } else {
                h_node_gpu_ind[n] = -1;
                interface_nodes++;
            }
        }

        printf("[PART assign_owned_nodes] interface_nodes=%d isolated_nodes=%d\n", interface_nodes,
               isolated_nodes);

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

            if (best_gpu < 0) best_gpu = 0;

            h_node_gpu_ind[n] = best_gpu;
            owned_node_cts[best_gpu]++;
        }

        for (int g = 0; g < ngpus; g++) {
            printf("[PART assign_owned_nodes] GPU[%d] tentative owned_node_cts=%d\n", g,
                   owned_node_cts[g]);
        }

        delete[] h_ne_cts;
        delete[] h_ne_ptr;
        delete[] h_ne_elems;
        delete[] owned_node_cts;

        printf("[PART assign_owned_nodes] DONE\n\n");
    }

    void build_owned_node_lists() {
        printf("\n[PART build_owned_node_lists] START\n");

        owned_nnodes = new int[ngpus];
        owned_N = new int[ngpus];
        h_owned_nodes = new int *[ngpus];
        d_owned_nodes = new int *[ngpus];

        std::memset(owned_nnodes, 0, ngpus * sizeof(int));
        std::memset(h_owned_nodes, 0, ngpus * sizeof(int *));
        std::memset(d_owned_nodes, 0, ngpus * sizeof(int *));

        for (int n = 0; n < num_nodes; n++) {
            int g = h_node_gpu_ind[n];

            if (g < 0 || g >= ngpus) {
                printf("[PART build_owned_node_lists] BAD owner node=%d owner=%d\n", n, g);
                exit(1);
            }

            owned_nnodes[g]++;
        }

        for (int g = 0; g < ngpus; g++) {
            owned_N[g] = owned_nnodes[g];
            h_owned_nodes[g] = owned_nnodes[g] > 0 ? new int[owned_nnodes[g]] : nullptr;
            printf("[PART build_owned_node_lists] GPU[%d] owned_nnodes=%d\n", g, owned_nnodes[g]);
        }

        int *ct = new int[ngpus];
        std::memset(ct, 0, ngpus * sizeof(int));

        for (int n = 0; n < num_nodes; n++) {
            int g = h_node_gpu_ind[n];
            h_owned_nodes[g][ct[g]++] = n;
        }

        delete[] ct;

        printf("[PART build_owned_node_lists] DONE\n\n");
    }

    void build_local_node_maps() {
        printf("\n[PART build_local_node_maps] START\n");

        local_nnodes = new int[ngpus];
        local_N = new int[ngpus];
        h_local_nodes = new int *[ngpus];

        std::memset(local_nnodes, 0, ngpus * sizeof(int));
        std::memset(local_N, 0, ngpus * sizeof(int));
        std::memset(h_local_nodes, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            std::unordered_set<int> node_set;

            int conn_size = local_nelems[g] * nodes_per_elem;
            for (int i = 0; i < conn_size; i++) {
                int node = h_local_elem_conn[g][i];

                if (node < 0 || node >= num_nodes) {
                    printf("[PART build_local_node_maps] BAD GPU[%d] conn i=%d node=%d\n", g, i,
                           node);
                    exit(1);
                }

                node_set.insert(node);
            }

            std::vector<int> nodes(node_set.begin(), node_set.end());
            std::sort(nodes.begin(), nodes.end());

            local_nnodes[g] = (int)nodes.size();
            local_N[g] = local_nnodes[g];
            h_local_nodes[g] = local_nnodes[g] > 0 ? new int[local_nnodes[g]] : nullptr;

            for (int i = 0; i < local_nnodes[g]; i++) h_local_nodes[g][i] = nodes[i];

            printf("[PART build_local_node_maps] GPU[%d] conn_size=%d local_nnodes=%d\n", g,
                   conn_size, local_nnodes[g]);
        }

        printf("[PART build_local_node_maps] DONE\n\n");
    }

    void build_local_ghost_flags() {
        printf("\n[PART build_local_ghost_flags] START\n");

        h_is_local_ghost = new bool *[ngpus];
        std::memset(h_is_local_ghost, 0, ngpus * sizeof(bool *));

        int **global_to_local = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            h_is_local_ghost[g] = local_nnodes[g] > 0 ? new bool[local_nnodes[g]] : nullptr;
            if (local_nnodes[g] > 0) {
                std::fill(h_is_local_ghost[g], h_is_local_ghost[g] + local_nnodes[g], false);
            }

            global_to_local[g] = new int[num_nodes];
            std::fill(global_to_local[g], global_to_local[g] + num_nodes, -1);

            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                int node = h_local_nodes[g][loc];
                global_to_local[g][node] = loc;
            }
        }

        int total_flags = 0;

        for (int node = 0; node < num_nodes; node++) {
            int count = 0;
            for (int g = 0; g < ngpus; g++) {
                if (global_to_local[g][node] >= 0) count++;
            }

            if (count <= 1) continue;

            for (int g = 0; g < ngpus; g++) {
                int loc = global_to_local[g][node];
                if (loc >= 0) {
                    h_is_local_ghost[g][loc] = true;
                    total_flags++;
                }
            }
        }

        for (int g = 0; g < ngpus; g++) {
            int nghost = 0;
            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                if (h_is_local_ghost[g][loc]) nghost++;
            }
            printf("[PART build_local_ghost_flags] GPU[%d] interface/local ghost flags=%d / %d\n",
                   g, nghost, local_nnodes[g]);
        }

        for (int g = 0; g < ngpus; g++) delete[] global_to_local[g];
        delete[] global_to_local;

        printf("[PART build_local_ghost_flags] total interface flags=%d\n", total_flags);
        printf("[PART build_local_ghost_flags] DONE\n\n");
    }

    void build_ghost_node_maps() {
        printf("\n[PART build_ghost_node_maps] START\n");

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
                int node = h_owned_nodes[g][i];
                owned_pos[node] = i;
            }
        }

        std::vector<int> *src_maps = new std::vector<int>[npairs];
        std::vector<int> *dst_maps = new std::vector<int>[npairs];

        for (int dst = 0; dst < ngpus; dst++) {
            for (int dst_loc = 0; dst_loc < local_nnodes[dst]; dst_loc++) {
                int node = h_local_nodes[dst][dst_loc];
                int src = h_node_gpu_ind[node];

                if (src < 0 || src >= ngpus || src == dst) continue;

                if (owned_pos[node] < 0) {
                    printf("[PART build_ghost_node_maps] BAD owned_pos node=%d src=%d dst=%d\n",
                           node, src, dst);
                    exit(1);
                }

                int idx = pair_index(dst, src);
                src_maps[idx].push_back(owned_pos[node]);
                dst_maps[idx].push_back(dst_loc);
            }
        }

        int total = 0;

        for (int idx = 0; idx < npairs; idx++) {
            int dst = idx / ngpus;
            int src = idx % ngpus;

            srcdest_nnodes[idx] = (int)src_maps[idx].size();
            ghost_nnodes[idx] = srcdest_nnodes[idx];
            total += srcdest_nnodes[idx];

            if (srcdest_nnodes[idx] > 0) {
                printf("[PART build_ghost_node_maps] pair dst=%d src=%d n=%d\n", dst, src,
                       srcdest_nnodes[idx]);
            }

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

        printf("[PART build_ghost_node_maps] total ghost map entries=%d\n", total);
        printf("[PART build_ghost_node_maps] DONE\n\n");
    }

    void build_owned_local_maps() {
        printf("\n[PART build_owned_local_maps] START\n");

        h_owned_to_local_map = new int *[ngpus];
        d_owned_to_local_map = new int *[ngpus];
        h_local_to_owned_map = new int *[ngpus];
        d_local_to_owned_map = new int *[ngpus];

        std::memset(h_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(d_owned_to_local_map, 0, ngpus * sizeof(int *));
        std::memset(h_local_to_owned_map, 0, ngpus * sizeof(int *));
        std::memset(d_local_to_owned_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            h_owned_to_local_map[g] = owned_nnodes[g] > 0 ? new int[owned_nnodes[g]] : nullptr;
            h_local_to_owned_map[g] = local_nnodes[g] > 0 ? new int[local_nnodes[g]] : nullptr;

            if (local_nnodes[g] > 0) {
                std::fill(h_local_to_owned_map[g], h_local_to_owned_map[g] + local_nnodes[g], -1);
            }

            int *global_to_local = new int[num_nodes];
            std::fill(global_to_local, global_to_local + num_nodes, -1);

            for (int loc = 0; loc < local_nnodes[g]; loc++) {
                int node = h_local_nodes[g][loc];
                global_to_local[node] = loc;
            }

            int owned_missing_local = 0;

            for (int i = 0; i < owned_nnodes[g]; i++) {
                int node = h_owned_nodes[g][i];
                int loc = global_to_local[node];

                h_owned_to_local_map[g][i] = loc;

                if (loc >= 0) {
                    h_local_to_owned_map[g][loc] = i;
                } else {
                    owned_missing_local++;
                }
            }

            delete[] global_to_local;

            printf("[PART build_owned_local_maps] GPU[%d] owned_missing_local=%d\n", g,
                   owned_missing_local);
        }

        printf("[PART build_owned_local_maps] DONE\n\n");
    }

    void move_maps_to_device() {
        printf("\n[PART move_maps_to_device] START\n");
        check_all_devices("move_maps entry", true);

        d_node_gpu_ind = new int *[ngpus];
        std::memset(d_node_gpu_ind, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            int dev = debug ? 0 : g;
            CHECK_CUDA(cudaSetDevice(dev));
            check_cuda("move_maps before per-gpu allocs", g, true);

            printf("[PART move_maps] GPU[%d] dev=%d num_nodes=%d owned_nnodes=%d local_nnodes=%d\n",
                   g, dev, num_nodes, owned_nnodes[g], local_nnodes[g]);

            CHECK_CUDA(cudaMalloc((void **)&d_node_gpu_ind[g], (size_t)num_nodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_node_gpu_ind[g], h_node_gpu_ind,
                                  (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));

            if (owned_nnodes[g] > 0) {
                CHECK_CUDA(
                    cudaMalloc((void **)&d_owned_nodes[g], (size_t)owned_nnodes[g] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_owned_nodes[g], h_owned_nodes[g],
                                      (size_t)owned_nnodes[g] * sizeof(int),
                                      cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaMalloc((void **)&d_owned_to_local_map[g],
                                      (size_t)owned_nnodes[g] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_owned_to_local_map[g], h_owned_to_local_map[g],
                                      (size_t)owned_nnodes[g] * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }

            if (local_nnodes[g] > 0) {
                CHECK_CUDA(cudaMalloc((void **)&d_local_to_owned_map[g],
                                      (size_t)local_nnodes[g] * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_local_to_owned_map[g], h_local_to_owned_map[g],
                                      (size_t)local_nnodes[g] * sizeof(int),
                                      cudaMemcpyHostToDevice));
            }

            check_cuda("move_maps after per-gpu copies", g, true);
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int n = srcdest_nnodes[idx];

                printf("[PART move_maps] pair dst=%d src=%d idx=%d n=%d\n", dst, src, idx, n);

                if (n <= 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                check_cuda("move_maps before d_srcred_map alloc", src, true);

                CHECK_CUDA(cudaMalloc((void **)&d_srcred_map[idx], (size_t)n * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_srcred_map[idx], h_srcred_map[idx], (size_t)n * sizeof(int),
                                      cudaMemcpyHostToDevice));

                check_cuda("move_maps after d_srcred_map copy", src, true);

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                check_cuda("move_maps before d_dstred_map alloc", dst, true);

                CHECK_CUDA(cudaMalloc((void **)&d_dstred_map[idx], (size_t)n * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_dstred_map[idx], h_dstred_map[idx], (size_t)n * sizeof(int),
                                      cudaMemcpyHostToDevice));

                check_cuda("move_maps after d_dstred_map copy", dst, true);
            }
        }

        check_all_devices("move_maps exit", true);
        printf("[PART move_maps_to_device] DONE\n\n");
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

    int **h_local_elems = nullptr;
    int **d_local_elems = nullptr;

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