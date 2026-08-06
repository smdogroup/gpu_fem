#include "include/poisson.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef CUDA_AWARE_MPI_DEFAULT
#define CUDA_AWARE_MPI_DEFAULT 0
#endif

// -----------------------------------------------------------------------------
// This file extends the original single-process multi-GPU ghosted BSR matvec to
// multiple MPI ranks. Each MPI rank owns a contiguous global node range and then
// subdivides that range across its local GPUs.
//
// Communication layout:
//   1. Same rank, different GPU: cudaMemcpyPeerAsync when possible.
//   2. Different rank: CUDA-aware MPI directly from/to device pointers if enabled.
//   3. Different rank fallback: device -> pinned host -> MPI -> pinned host -> device.
//
// Recommended launch examples:
//   mpicxx -O3 -fopenmp -DUSE_MPI -x c++ mpi_multi_gpu_matvec.cpp ...
//   mpirun -np 2 ./mpi_multi_gpu_matvec 4096 4 0
//
// Args:
//   argv[1] = grid dimension n, so scalar N = n*n. Default n = 128.
//   argv[2] = requested GPUs per rank. Default 4.
//   argv[3] = CUDA-aware MPI flag. 0 fallback host staging, 1 device MPI.
// -----------------------------------------------------------------------------

template <typename T>
MPI_Datatype mpiType();

template <>
MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }

template <>
MPI_Datatype mpiType<float>() { return MPI_FLOAT; }

template <typename T>
struct GhostCopy {
    int src_rank = -1;
    int src_gpu = -1;
    int src_local_node = -1;
    int dst_ext_node = -1;
    int global_node = -1;
};

template <typename T>
struct RemoteRecv {
    int src_rank = -1;
    int local_gpu = -1;
    int dst_ext_node = -1;
    int global_node = -1;
};

template <typename T>
struct RemoteSend {
    int dst_rank = -1;
    int local_gpu = -1;
    int src_local_node = -1;
    int global_node = -1;
};

// template <typename T>
// struct RankCommPlan {
//     int rank = -1;

//     std::vector<RemoteRecv<T>> recvs;
//     std::vector<RemoteSend<T>> sends;

//     T *h_send = nullptr;
//     T *h_recv = nullptr;
//     T *d_send = nullptr;
//     T *d_recv = nullptr;

//     int send_nodes = 0;
//     int recv_nodes = 0;
// };

template <typename T>
struct GPUData {
    int dev = 0;

    int row_start_node = 0;  // global block-node start for this local GPU
    int row_end_node = 0;    // global block-node end for this local GPU
    int local_nnodes = 0;
    int nghost = 0;
    int local_N = 0;
    int ext_N = 0;
    int nnzb_local = 0;

    std::vector<int> ghost_global_nodes;
    std::vector<GhostCopy<T>> ghost_copies;

    int *h_rowp = nullptr;
    int *h_cols = nullptr;
    T *h_vals = nullptr;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    T *d_vals = nullptr;

    T *d_x_owned = nullptr;
    T *d_x_ext = nullptr;
    T *d_y_owned = nullptr;

    cudaStream_t stream = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    cusparseMatDescr_t descrA = nullptr;
};

struct Owner {
    int rank = -1;
    int gpu = -1;
    int local_node = -1;
};

// -----------------------------------------------------------------------------
// Add near the top, after structs/includes
// -----------------------------------------------------------------------------

template <typename T>
__global__ void k_gather_remote_send(
    int n_nodes,
    int block_dim,
    const int *src_local_nodes,
    const T *x_owned,
    T *send_buf) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n_nodes * block_dim;
    if (tid >= N) return;

    int inode = tid / block_dim;
    int idof  = tid % block_dim;

    int src_node = src_local_nodes[inode];
    send_buf[inode * block_dim + idof] =
        x_owned[src_node * block_dim + idof];
}

template <typename T>
__global__ void k_scatter_remote_recv(
    int n_nodes,
    int block_dim,
    const int *dst_ext_nodes,
    const T *recv_buf,
    T *x_ext) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n_nodes * block_dim;
    if (tid >= N) return;

    int inode = tid / block_dim;
    int idof  = tid % block_dim;

    int dst_node = dst_ext_nodes[inode];
    x_ext[dst_node * block_dim + idof] =
        recv_buf[inode * block_dim + idof];
}

template <typename T>
struct PeerGpuLink {
    int peer_rank = -1;

    // local source GPU for sends, local destination GPU for receives
    int local_gpu = -1;

    // remote destination GPU for sends, remote source GPU for receives
    int remote_gpu = -1;

    int count = 0;

    // send-side map
    std::vector<int> h_send_src_nodes_vec;
    int *d_send_src_nodes = nullptr;

    // recv-side map
    std::vector<int> h_recv_dst_ext_nodes_vec;
    int *d_recv_dst_ext_nodes = nullptr;

    T *h_buf = nullptr;
    T *d_buf = nullptr;
};

template <typename T>
struct RankCommPlan {
    int rank = -1;

    // One contiguous message per peer/sourceGPU/destinationGPU.
    // send_links: this rank sends from local_gpu to peer rank's remote_gpu.
    // recv_links: this rank receives from peer rank's remote_gpu into local_gpu.
    std::vector<PeerGpuLink<T>> send_links;
    std::vector<PeerGpuLink<T>> recv_links;
};

static inline int ownerOfNode1D(int node, const std::vector<int> &starts,
                                const std::vector<int> &ends) {
    int lo = 0;
    int hi = (int)starts.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (node < starts[mid]) {
            hi = mid - 1;
        } else if (node >= ends[mid]) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

static Owner ownerOfGlobalNode(int node, const std::vector<int> &rank_starts,
                               const std::vector<int> &rank_ends,
                               const std::vector<std::vector<int>> &gpu_starts,
                               const std::vector<std::vector<int>> &gpu_ends) {
    Owner o;
    o.rank = ownerOfNode1D(node, rank_starts, rank_ends);
    if (o.rank < 0) return o;

    o.gpu = ownerOfNode1D(node, gpu_starts[o.rank], gpu_ends[o.rank]);
    if (o.gpu < 0) {
        o.rank = -1;
        return o;
    }

    o.local_node = node - gpu_starts[o.rank][o.gpu];
    return o;
}

static void buildGlobalPartition(int nnodes, int nranks, int ngpu_local,
                                 std::vector<int> &rank_starts,
                                 std::vector<int> &rank_ends,
                                 std::vector<std::vector<int>> &gpu_starts,
                                 std::vector<std::vector<int>> &gpu_ends) {
    rank_starts.resize(nranks);
    rank_ends.resize(nranks);
    gpu_starts.resize(nranks);
    gpu_ends.resize(nranks);

    for (int r = 0; r < nranks; r++) {
        rank_starts[r] = (r * nnodes) / nranks;
        rank_ends[r] = ((r + 1) * nnodes) / nranks;

        gpu_starts[r].resize(ngpu_local);
        gpu_ends[r].resize(ngpu_local);

        int r0 = rank_starts[r];
        int rn = rank_ends[r] - rank_starts[r];
        for (int g = 0; g < ngpu_local; g++) {
            gpu_starts[r][g] = r0 + (g * rn) / ngpu_local;
            gpu_ends[r][g] = r0 + ((g + 1) * rn) / ngpu_local;
        }
    }
}

template <typename T>
void extractLocalBSRRowsWithGhosts(GPUData<T> &gd,
                                   int my_rank,
                                   const std::vector<int> &rank_starts,
                                   const std::vector<int> &rank_ends,
                                   const std::vector<std::vector<int>> &gpu_starts,
                                   const std::vector<std::vector<int>> &gpu_ends,
                                   const int *rowp,
                                   const int *cols,
                                   const T *vals,
                                   int block_dim) {
    const int block_dim2 = block_dim * block_dim;
    const int row_start = gd.row_start_node;
    const int row_end = gd.row_end_node;
    const int local_nrows = row_end - row_start;

    const int start_nnz = rowp[row_start];
    const int end_nnz = rowp[row_end];
    const int nnzb_local = end_nnz - start_nnz;

    gd.h_rowp = (int *)malloc((local_nrows + 1) * sizeof(int));
    gd.h_cols = (int *)malloc(nnzb_local * sizeof(int));
    gd.h_vals = (T *)malloc(nnzb_local * block_dim2 * sizeof(T));
    gd.nnzb_local = nnzb_local;

    std::unordered_map<int, int> ghost_map;

    gd.h_rowp[0] = 0;
    for (int i = 0; i < local_nrows; i++) {
        gd.h_rowp[i + 1] = rowp[row_start + i + 1] - start_nnz;
    }

    for (int k = 0; k < nnzb_local; k++) {
        int global_col = cols[start_nnz + k];

        if (global_col >= row_start && global_col < row_end) {
            gd.h_cols[k] = global_col - row_start;
        } else {
            auto it = ghost_map.find(global_col);
            if (it == ghost_map.end()) {
                int ghost_id = (int)gd.ghost_global_nodes.size();
                ghost_map[global_col] = ghost_id;
                gd.ghost_global_nodes.push_back(global_col);

                Owner src = ownerOfGlobalNode(global_col, rank_starts, rank_ends,
                                              gpu_starts, gpu_ends);
                if (src.rank < 0) {
                    printf("ERROR: could not find owner for node %d\n", global_col);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                GhostCopy<T> cp;
                cp.src_rank = src.rank;
                cp.src_gpu = src.gpu;
                cp.src_local_node = src.local_node;
                cp.dst_ext_node = local_nrows + ghost_id;
                cp.global_node = global_col;
                gd.ghost_copies.push_back(cp);

                gd.h_cols[k] = local_nrows + ghost_id;
            } else {
                gd.h_cols[k] = local_nrows + it->second;
            }
        }
    }

#pragma omp parallel for if (nnzb_local * block_dim2 > 20000)
    for (int k = 0; k < nnzb_local * block_dim2; k++) {
        gd.h_vals[k] = vals[start_nnz * block_dim2 + k];
    }

    gd.nghost = (int)gd.ghost_global_nodes.size();
    gd.local_nnodes = local_nrows;
    gd.local_N = gd.local_nnodes * block_dim;
    gd.ext_N = (gd.local_nnodes + gd.nghost) * block_dim;
}

template <typename T>
void setupGhostedMultiGPU(std::vector<GPUData<T>> &gpus,
                          int my_rank,
                          int nranks,
                          int ngpu,
                          int N,
                          int nnodes,
                          int block_dim,
                          int block_dim2,
                          const int *rowp,
                          const int *cols,
                          const T *vals,
                          std::vector<int> &rank_starts,
                          std::vector<int> &rank_ends,
                          std::vector<std::vector<int>> &gpu_starts,
                          std::vector<std::vector<int>> &gpu_ends) {
    gpus.resize(ngpu);

    buildGlobalPartition(nnodes, nranks, ngpu, rank_starts, rank_ends, gpu_starts, gpu_ends);

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(g));

        gpus[g].dev = g;
        gpus[g].row_start_node = gpu_starts[my_rank][g];
        gpus[g].row_end_node = gpu_ends[my_rank][g];

        extractLocalBSRRowsWithGhosts<T>(gpus[g], my_rank, rank_starts, rank_ends,
                                         gpu_starts, gpu_ends, rowp, cols, vals,
                                         block_dim);

        CHECK_CUDA(cudaStreamCreate(&gpus[g].stream));

        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_rowp,
                              (gpus[g].local_nnodes + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_cols,
                              gpus[g].nnzb_local * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_vals,
                              gpus[g].nnzb_local * block_dim2 * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_x_owned,
                              gpus[g].local_N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_x_ext,
                              gpus[g].ext_N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&gpus[g].d_y_owned,
                              gpus[g].local_N * sizeof(T)));

        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_rowp, gpus[g].h_rowp,
                                   (gpus[g].local_nnodes + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, gpus[g].stream));
        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_cols, gpus[g].h_cols,
                                   gpus[g].nnzb_local * sizeof(int),
                                   cudaMemcpyHostToDevice, gpus[g].stream));
        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_vals, gpus[g].h_vals,
                                   gpus[g].nnzb_local * block_dim2 * sizeof(T),
                                   cudaMemcpyHostToDevice, gpus[g].stream));

        CHECK_CUSPARSE(cusparseCreate(&gpus[g].cusparseHandle));
        CHECK_CUSPARSE(cusparseSetStream(gpus[g].cusparseHandle, gpus[g].stream));
        CHECK_CUSPARSE(cusparseCreateMatDescr(&gpus[g].descrA));
        CHECK_CUSPARSE(cusparseSetMatType(gpus[g].descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(gpus[g].descrA, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));

        printf("rank %d GPU %d owns block rows [%d, %d), local = %d, ghosts = %d, local nnzb = %d\n",
               my_rank, g, gpus[g].row_start_node, gpus[g].row_end_node,
               gpus[g].local_nnodes, gpus[g].nghost, gpus[g].nnzb_local);
    }
}

template <typename T>
void scatterOwnedXToGPUs(std::vector<GPUData<T>> &gpus,
                         int ngpu,
                         int block_dim,
                         const T *h_x) {
#pragma omp parallel for if (ngpu > 1)
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int scalar_start = gpus[g].row_start_node * block_dim;
        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_x_owned, &h_x[scalar_start],
                                   gpus[g].local_N * sizeof(T),
                                   cudaMemcpyHostToDevice, gpus[g].stream));
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }
}

template <typename T>
void gatherOwnedYFromGPUs(std::vector<GPUData<T>> &gpus,
                          int ngpu,
                          int block_dim,
                          T *h_y) {
#pragma omp parallel for if (ngpu > 1)
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        int scalar_start = gpus[g].row_start_node * block_dim;
        CHECK_CUDA(cudaMemcpyAsync(&h_y[scalar_start], gpus[g].d_y_owned,
                                   gpus[g].local_N * sizeof(T),
                                   cudaMemcpyDeviceToHost, gpus[g].stream));
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }
}

template <typename T>
void buildRankCommPlans(std::vector<GPUData<T>> &gpus,
                        int my_rank,
                        int nranks,
                        int ngpu,
                        int block_dim,
                        const std::vector<int> &rank_starts,
                        const std::vector<int> &rank_ends,
                        const std::vector<std::vector<int>> &gpu_starts,
                        const std::vector<std::vector<int>> &gpu_ends,
                        std::vector<RankCommPlan<T>> &plans) {
    plans.resize(nranks);
    for (int r = 0; r < nranks; r++) {
        plans[r].rank = r;
    }

    // Request triples:
    //   [global_node, dst_gpu_on_requesting_rank, dst_ext_node_on_requesting_rank]
    std::vector<std::vector<int>> outgoing_req_ints(nranks);

    for (int dst_gpu = 0; dst_gpu < ngpu; dst_gpu++) {
        for (const auto &cp : gpus[dst_gpu].ghost_copies) {
            if (cp.src_rank == my_rank) continue;

            auto &v = outgoing_req_ints[cp.src_rank];
            v.push_back(cp.global_node);
            v.push_back(dst_gpu);
            v.push_back(cp.dst_ext_node);
        }
    }

    std::vector<int> send_counts(nranks, 0);
    std::vector<int> recv_counts(nranks, 0);

    for (int r = 0; r < nranks; r++) {
        send_counts[r] = (int)outgoing_req_ints[r].size();
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> send_displs(nranks + 1, 0);
    std::vector<int> recv_displs(nranks + 1, 0);

    for (int r = 0; r < nranks; r++) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }

    std::vector<int> send_buf(send_displs[nranks]);
    std::vector<int> recv_buf(recv_displs[nranks]);

    for (int r = 0; r < nranks; r++) {
        std::copy(outgoing_req_ints[r].begin(),
                  outgoing_req_ints[r].end(),
                  send_buf.begin() + send_displs[r]);
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Build receive links.
    // This rank knows what it requested from each peer.
    // Group by:
    //   peer rank
    //   remote source GPU
    //   local destination GPU
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (int p = 0; p < send_counts[peer]; p += 3) {
            int global_node = send_buf[send_displs[peer] + p + 0];
            int local_dst_gpu = send_buf[send_displs[peer] + p + 1];
            int dst_ext_node = send_buf[send_displs[peer] + p + 2];

            Owner src = ownerOfGlobalNode(global_node, rank_starts, rank_ends,
                                          gpu_starts, gpu_ends);

            if (src.rank != peer) {
                printf("rank %d ERROR: bad owner for requested node %d\n",
                       my_rank, global_node);
                MPI_Abort(MPI_COMM_WORLD, 10);
            }

            PeerGpuLink<T> *link = nullptr;

            for (auto &L : plans[peer].recv_links) {
                if (L.peer_rank == peer &&
                    L.remote_gpu == src.gpu &&
                    L.local_gpu == local_dst_gpu) {
                    link = &L;
                    break;
                }
            }

            if (!link) {
                PeerGpuLink<T> L;
                L.peer_rank = peer;
                L.remote_gpu = src.gpu;
                L.local_gpu = local_dst_gpu;
                plans[peer].recv_links.push_back(L);
                link = &plans[peer].recv_links.back();
            }

            link->h_recv_dst_ext_nodes_vec.push_back(dst_ext_node);
        }
    }

    // ------------------------------------------------------------------
    // Build send links.
    // Peer requests arrived in recv_buf. They tell us:
    //   global_node to send
    //   remote destination GPU
    //   remote destination ext node
    //
    // We only need global_node + remote dst GPU for packing.
    // Group by:
    //   peer rank
    //   local source GPU
    //   remote destination GPU
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (int p = recv_displs[peer]; p < recv_displs[peer + 1]; p += 3) {
            int global_node = recv_buf[p + 0];
            int remote_dst_gpu = recv_buf[p + 1];

            int local_src_gpu = -1;
            int src_local_node = -1;

            for (int g = 0; g < ngpu; g++) {
                if (global_node >= gpus[g].row_start_node &&
                    global_node <  gpus[g].row_end_node) {
                    local_src_gpu = g;
                    src_local_node = global_node - gpus[g].row_start_node;
                    break;
                }
            }

            if (local_src_gpu < 0) {
                printf("rank %d ERROR: peer %d requested non-owned node %d\n",
                       my_rank, peer, global_node);
                MPI_Abort(MPI_COMM_WORLD, 11);
            }

            PeerGpuLink<T> *link = nullptr;

            for (auto &L : plans[peer].send_links) {
                if (L.peer_rank == peer &&
                    L.local_gpu == local_src_gpu &&
                    L.remote_gpu == remote_dst_gpu) {
                    link = &L;
                    break;
                }
            }

            if (!link) {
                PeerGpuLink<T> L;
                L.peer_rank = peer;
                L.local_gpu = local_src_gpu;
                L.remote_gpu = remote_dst_gpu;
                plans[peer].send_links.push_back(L);
                link = &plans[peer].send_links.back();
            }

            link->h_send_src_nodes_vec.push_back(src_local_node);
        }
    }

    // ------------------------------------------------------------------
    // Allocate device maps and contiguous buffers.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (auto &L : plans[peer].send_links) {
            L.count = (int)L.h_send_src_nodes_vec.size();
            if (L.count == 0) continue;

            CHECK_CUDA(cudaSetDevice(gpus[L.local_gpu].dev));

            CHECK_CUDA(cudaMalloc((void **)&L.d_send_src_nodes,
                                  L.count * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(L.d_send_src_nodes,
                                  L.h_send_src_nodes_vec.data(),
                                  L.count * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc((void **)&L.d_buf,
                                  L.count * block_dim * sizeof(T)));

            CHECK_CUDA(cudaHostAlloc((void **)&L.h_buf,
                                     L.count * block_dim * sizeof(T),
                                     cudaHostAllocPortable));
        }

        for (auto &L : plans[peer].recv_links) {
            L.count = (int)L.h_recv_dst_ext_nodes_vec.size();
            if (L.count == 0) continue;

            CHECK_CUDA(cudaSetDevice(gpus[L.local_gpu].dev));

            CHECK_CUDA(cudaMalloc((void **)&L.d_recv_dst_ext_nodes,
                                  L.count * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(L.d_recv_dst_ext_nodes,
                                  L.h_recv_dst_ext_nodes_vec.data(),
                                  L.count * sizeof(int),
                                  cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc((void **)&L.d_buf,
                                  L.count * block_dim * sizeof(T)));

            CHECK_CUDA(cudaHostAlloc((void **)&L.h_buf,
                                     L.count * block_dim * sizeof(T),
                                     cudaHostAllocPortable));
        }
    }
}

template <typename T>
void exchangeGhosts(std::vector<GPUData<T>> &gpus,
                    std::vector<RankCommPlan<T>> &plans,
                    int my_rank,
                    int nranks,
                    int ngpu,
                    int block_dim,
                    bool cuda_aware_mpi) {
    const int threads = 256;

    // ---------------------------------------------------------------
    // Copy owned part into x_ext.
    // ---------------------------------------------------------------
#pragma omp parallel for if (ngpu > 1)
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_x_ext,
                                   gpus[g].d_x_owned,
                                   gpus[g].local_N * sizeof(T),
                                   cudaMemcpyDeviceToDevice,
                                   gpus[g].stream));
    }

    // ---------------------------------------------------------------
    // Same-rank same-node GPU ghosts.
    // Keep this path as peer copies.
    // ---------------------------------------------------------------
    for (int dst = 0; dst < ngpu; dst++) {
        CHECK_CUDA(cudaSetDevice(gpus[dst].dev));

        for (const auto &cp : gpus[dst].ghost_copies) {
            if (cp.src_rank != my_rank) continue;

            int src = cp.src_gpu;

            int can_access = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access,
                                               gpus[dst].dev,
                                               gpus[src].dev));

            if (src != dst && can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(gpus[src].dev, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CHECK_CUDA(err);
                }
                cudaGetLastError();
            }

            if (src == dst) {
                CHECK_CUDA(cudaMemcpyAsync(
                    gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                    gpus[src].d_x_owned + cp.src_local_node * block_dim,
                    block_dim * sizeof(T),
                    cudaMemcpyDeviceToDevice,
                    gpus[dst].stream));
            } else if (can_access) {
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                    gpus[dst].dev,
                    gpus[src].d_x_owned + cp.src_local_node * block_dim,
                    gpus[src].dev,
                    block_dim * sizeof(T),
                    gpus[dst].stream));
            } else {
                T tmp[64];
                assert(block_dim <= 64);

                CHECK_CUDA(cudaSetDevice(gpus[src].dev));
                CHECK_CUDA(cudaMemcpy(tmp,
                                      gpus[src].d_x_owned + cp.src_local_node * block_dim,
                                      block_dim * sizeof(T),
                                      cudaMemcpyDeviceToHost));

                CHECK_CUDA(cudaSetDevice(gpus[dst].dev));
                CHECK_CUDA(cudaMemcpyAsync(
                    gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                    tmp,
                    block_dim * sizeof(T),
                    cudaMemcpyHostToDevice,
                    gpus[dst].stream));
            }
        }
    }

    // ---------------------------------------------------------------
    // Pack remote sends with one gather kernel per link.
    // ---------------------------------------------------------------
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (auto &L : plans[peer].send_links) {
            if (L.count == 0) continue;

            int g = L.local_gpu;
            CHECK_CUDA(cudaSetDevice(gpus[g].dev));

            int nthreads = L.count * block_dim;
            int blocks = (nthreads + threads - 1) / threads;

            k_gather_remote_send<T><<<blocks, threads, 0, gpus[g].stream>>>(
                L.count,
                block_dim,
                L.d_send_src_nodes,
                gpus[g].d_x_owned,
                L.d_buf);

            CHECK_CUDA(cudaGetLastError());

            if (!cuda_aware_mpi) {
                CHECK_CUDA(cudaMemcpyAsync(L.h_buf,
                                           L.d_buf,
                                           L.count * block_dim * sizeof(T),
                                           cudaMemcpyDeviceToHost,
                                           gpus[g].stream));
            }
        }
    }

    // Need send buffers ready before MPI.
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }

    // ---------------------------------------------------------------
    // MPI exchange.
    // Tags distinguish:
    //   sender local source GPU
    //   receiver local destination GPU
    // ---------------------------------------------------------------
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * nranks * ngpu * ngpu);

    auto tag = [ngpu](int src_gpu, int dst_gpu) {
        return 2000 + src_gpu * ngpu + dst_gpu;
    };

    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (auto &L : plans[peer].recv_links) {
            if (L.count == 0) continue;

            MPI_Request req;
            void *recv_ptr = cuda_aware_mpi ? (void *)L.d_buf : (void *)L.h_buf;

            MPI_Irecv(recv_ptr,
                      L.count * block_dim,
                      mpiType<T>(),
                      peer,
                      tag(L.remote_gpu, L.local_gpu),
                      MPI_COMM_WORLD,
                      &req);

            reqs.push_back(req);
        }
    }

    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (auto &L : plans[peer].send_links) {
            if (L.count == 0) continue;

            MPI_Request req;
            void *send_ptr = cuda_aware_mpi ? (void *)L.d_buf : (void *)L.h_buf;

            MPI_Isend(send_ptr,
                      L.count * block_dim,
                      mpiType<T>(),
                      peer,
                      tag(L.local_gpu, L.remote_gpu),
                      MPI_COMM_WORLD,
                      &req);

            reqs.push_back(req);
        }
    }

    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // ---------------------------------------------------------------
    // Unpack remote receives with one scatter kernel per link.
    // ---------------------------------------------------------------
    for (int peer = 0; peer < nranks; peer++) {
        if (peer == my_rank) continue;

        for (auto &L : plans[peer].recv_links) {
            if (L.count == 0) continue;

            int g = L.local_gpu;
            CHECK_CUDA(cudaSetDevice(gpus[g].dev));

            if (!cuda_aware_mpi) {
                CHECK_CUDA(cudaMemcpyAsync(L.d_buf,
                                           L.h_buf,
                                           L.count * block_dim * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           gpus[g].stream));
            }

            int nthreads = L.count * block_dim;
            int blocks = (nthreads + threads - 1) / threads;

            k_scatter_remote_recv<T><<<blocks, threads, 0, gpus[g].stream>>>(
                L.count,
                block_dim,
                L.d_recv_dst_ext_nodes,
                L.d_buf,
                gpus[g].d_x_ext);

            CHECK_CUDA(cudaGetLastError());
        }
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }
}

template <typename T>
void multiGpuGhostedBSRMatVec(std::vector<GPUData<T>> &gpus,
                              std::vector<RankCommPlan<T>> &plans,
                              int my_rank,
                              int nranks,
                              int ngpu,
                              int block_dim,
                              bool cuda_aware_mpi) {
    T alpha = 1.0;
    T beta = 0.0;

    exchangeGhosts<T>(gpus, plans, my_rank, nranks, ngpu, block_dim, cuda_aware_mpi);

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUSPARSE(cusparseDbsrmv(gpus[g].cusparseHandle,
                                      CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      gpus[g].local_nnodes,
                                      gpus[g].local_nnodes + gpus[g].nghost,
                                      gpus[g].nnzb_local,
                                      &alpha,
                                      gpus[g].descrA,
                                      gpus[g].d_vals,
                                      gpus[g].d_rowp,
                                      gpus[g].d_cols,
                                      block_dim,
                                      gpus[g].d_x_ext,
                                      &beta,
                                      gpus[g].d_y_owned));
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }
}

template <typename T>
void cleanupMultiGPU(std::vector<GPUData<T>> &gpus,
                     std::vector<RankCommPlan<T>> &plans,
                     int my_rank) {
    for (auto &gd : gpus) {
        CHECK_CUDA(cudaSetDevice(gd.dev));

        if (gd.d_rowp) cudaFree(gd.d_rowp);
        if (gd.d_cols) cudaFree(gd.d_cols);
        if (gd.d_vals) cudaFree(gd.d_vals);
        if (gd.d_x_owned) cudaFree(gd.d_x_owned);
        if (gd.d_x_ext) cudaFree(gd.d_x_ext);
        if (gd.d_y_owned) cudaFree(gd.d_y_owned);

        if (gd.descrA) cusparseDestroyMatDescr(gd.descrA);
        if (gd.cusparseHandle) cusparseDestroy(gd.cusparseHandle);
        if (gd.stream) cudaStreamDestroy(gd.stream);

        if (gd.h_rowp) free(gd.h_rowp);
        if (gd.h_cols) free(gd.h_cols);
        if (gd.h_vals) free(gd.h_vals);
    }

    // for (auto &p : plans) {
    //     if (p.rank == my_rank) continue;
    //     if (p.h_send) cudaFreeHost(p.h_send);
    //     if (p.h_recv) cudaFreeHost(p.h_recv);
    //     if (p.d_send) cudaFree(p.d_send);
    //     if (p.d_recv) cudaFree(p.d_recv);
    // }
    for (auto &p : plans) {
        if (p.rank == my_rank) continue;

        for (auto &L : p.send_links) {
            if (L.d_send_src_nodes) cudaFree(L.d_send_src_nodes);
            if (L.d_buf) cudaFree(L.d_buf);
            if (L.h_buf) cudaFreeHost(L.h_buf);
        }

        for (auto &L : p.recv_links) {
            if (L.d_recv_dst_ext_nodes) cudaFree(L.d_recv_dst_ext_nodes);
            if (L.d_buf) cudaFree(L.d_buf);
            if (L.h_buf) cudaFreeHost(L.h_buf);
        }
    }
}

int main(int argc, char **argv) {
    using T = double;

    MPI_Init(&argc, &argv);

    int my_rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int n = 128;
    int requested_gpus = 4;
    bool cuda_aware_mpi = CUDA_AWARE_MPI_DEFAULT != 0;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) requested_gpus = atoi(argv[2]);
    if (argc > 3) cuda_aware_mpi = atoi(argv[3]) != 0;

    int N = n * n;
    int block_dim = 2;
    int block_dim2 = block_dim * block_dim;
    int nz = 5 * N - 4 * n;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    int ngpu = std::min(requested_gpus, device_count);

    if (ngpu < 1) {
        if (my_rank == 0) printf("ERROR: no CUDA devices visible.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (my_rank == 0) {
        printf("N = %d, n = %d, MPI ranks = %d, requested GPUs/rank = %d, CUDA-aware MPI = %d\n",
               N, n, nranks, requested_gpus, (int)cuda_aware_mpi);
    }
    printf("rank %d using %d GPU(s) out of %d visible.\n", my_rank, ngpu, device_count);

    int *csr_rowp = (int *)malloc(sizeof(int) * (N + 1));
    int *csr_cols = (int *)malloc(sizeof(int) * nz);
    T *csr_vals = (T *)malloc(sizeof(T) * nz);
    T *rhs = (T *)malloc(sizeof(T) * N);
    T *x = (T *)malloc(sizeof(T) * N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        rhs[i] = 0.0;
        x[i] = sin(0.001 * i) + 0.01 * cos(0.07 * i);
    }

    genLaplaceCSR_threaded<T>(csr_rowp, csr_cols, csr_vals, N, nz, rhs);

    int *rowp = nullptr;
    int *cols = nullptr;
    T *vals = nullptr;
    int nnzb = 0;

    CSRtoBSR_threaded<T>(block_dim, N, csr_rowp, csr_cols, csr_vals,
                         &rowp, &cols, &vals, &nnzb);

    int nnodes = N / block_dim;
    int mb = nnodes;

    if (my_rank == 0) {
        printf("N = %d, nnodes = %d, CSR nz = %d, BSR nnzb = %d\n",
               N, nnodes, nz, nnzb);
    }

    // ---------------------------------------------------------------------
    // Single GPU reference on rank 0 only.
    // ---------------------------------------------------------------------
    T *y_single = (T *)calloc(N, sizeof(T));
    T *y_multi = (T *)calloc(N, sizeof(T));

    if (my_rank == 0) {
        CHECK_CUDA(cudaSetDevice(0));

        int *d_rowp = nullptr;
        int *d_cols = nullptr;
        T *d_vals = nullptr;
        T *d_x = nullptr;
        T *d_y = nullptr;

        CHECK_CUDA(cudaMalloc((void **)&d_rowp, (nnodes + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_cols, nnzb * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_vals, nnzb * block_dim2 * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cols, cols, nnzb * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_vals, vals, nnzb * block_dim2 * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyHostToDevice));

        cusparseHandle_t cusparseHandle = nullptr;
        cusparseMatDescr_t descrA = nullptr;
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
        CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

        T alpha = 1.0;
        T beta = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle,
                                      CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      mb,
                                      mb,
                                      nnzb,
                                      &alpha,
                                      descrA,
                                      d_vals,
                                      d_rowp,
                                      d_cols,
                                      block_dim,
                                      d_x,
                                      &beta,
                                      d_y));

        CHECK_CUDA(cudaMemcpy(y_single, d_y, N * sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(d_rowp);
        cudaFree(d_cols);
        cudaFree(d_vals);
        cudaFree(d_x);
        cudaFree(d_y);
        cusparseDestroyMatDescr(descrA);
        cusparseDestroy(cusparseHandle);
    }

    // ---------------------------------------------------------------------
    // MPI + ghosted multi-GPU matvec.
    // ---------------------------------------------------------------------
    std::vector<GPUData<T>> gpus;
    std::vector<int> rank_starts, rank_ends;
    std::vector<std::vector<int>> gpu_starts, gpu_ends;

    setupGhostedMultiGPU<T>(gpus, my_rank, nranks, ngpu, N, nnodes,
                            block_dim, block_dim2, rowp, cols, vals,
                            rank_starts, rank_ends, gpu_starts, gpu_ends);

    std::vector<RankCommPlan<T>> plans;
    // buildRankCommPlans<T>(gpus, my_rank, nranks, ngpu, block_dim, plans);
    buildRankCommPlans<T>(
        gpus,
        my_rank,
        nranks,
        ngpu,
        block_dim,
        rank_starts,
        rank_ends,
        gpu_starts,
        gpu_ends,
        plans
    );

    // scatterOwnedXToGPUs<T>(gpus, ngpu, block_dim, x);
    // multiGpuGhostedBSRMatVec<T>(gpus, plans, my_rank, nranks, ngpu, block_dim, cuda_aware_mpi);
    // gatherOwnedYFromGPUs<T>(gpus, ngpu, block_dim, y_multi);
    scatterOwnedXToGPUs<T>(gpus, ngpu, block_dim, x);

    // ------------------------------------------------------------------
    // timing
    // ------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    int nreps = 100;

    for (int rep = 0; rep < nreps; rep++) {
        multiGpuGhostedBSRMatVec<T>(
            gpus,
            plans,
            my_rank,
            nranks,
            ngpu,
            block_dim,
            cuda_aware_mpi
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    double avg_time = (t1 - t0) / nreps;

    // take worst rank timing
    double max_time = 0.0;

    MPI_Reduce(
        &avg_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (my_rank == 0) {
        printf("\nMPI + MultiGPU Ghosted BSR SpMV timing:\n");
        printf("  avg matvec time = %.6e sec\n", max_time);

        double dof_per_sec = ((double)N) / max_time;

        printf("  throughput      = %.3e DOF/sec\n", dof_per_sec);
    }

    gatherOwnedYFromGPUs<T>(gpus, ngpu, block_dim, y_multi);

    // Combine rank-owned slices into y_multi on rank 0.
    std::vector<int> recv_counts(nranks), recv_displs(nranks);
    for (int r = 0; r < nranks; r++) {
        int node_count = rank_ends[r] - rank_starts[r];
        recv_counts[r] = node_count * block_dim;
        recv_displs[r] = rank_starts[r] * block_dim;
    }

    T *y_global = nullptr;
    if (my_rank == 0) y_global = (T *)calloc(N, sizeof(T));

    MPI_Gatherv(y_multi + recv_displs[my_rank], recv_counts[my_rank], mpiType<T>(),
                y_global, recv_counts.data(), recv_displs.data(), mpiType<T>(),
                0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double diff2 = 0.0;
        double norm2 = 0.0;
        double max_abs = 0.0;

        for (int i = 0; i < N; i++) {
            double diff = (double)y_global[i] - (double)y_single[i];
            diff2 += diff * diff;
            norm2 += (double)y_single[i] * (double)y_single[i];
            max_abs = std::max(max_abs, std::abs(diff));
        }

        double rel_err = sqrt(diff2 / norm2);
        printf("\nMPI + ghosted multi-GPU BSR SpMV check:\n");
        printf("  rel L2 error = %.15e\n", rel_err);
        printf("  max abs err  = %.15e\n", max_abs);
        printf("  %s\n", rel_err < 1e-12 ? "PASS" : "FAIL");
    }

    cleanupMultiGPU<T>(gpus, plans, my_rank);

    free(csr_rowp);
    free(csr_cols);
    free(csr_vals);
    free(rhs);
    free(x);
    free(y_single);
    free(y_multi);
    if (y_global) free(y_global);

    delete[] rowp;
    delete[] cols;
    delete[] vals;

    MPI_Finalize();
    return 0;
}