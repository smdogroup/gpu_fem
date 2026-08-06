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

template <typename T>
struct RankCommPlan {
    int rank = -1;

    std::vector<RemoteRecv<T>> recvs;
    std::vector<RemoteSend<T>> sends;

    T *h_send = nullptr;
    T *h_recv = nullptr;
    T *d_send = nullptr;
    T *d_recv = nullptr;

    int send_nodes = 0;
    int recv_nodes = 0;
};

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
                        std::vector<RankCommPlan<T>> &plans) {
    plans.resize(nranks);
    for (int r = 0; r < nranks; r++) {
        plans[r].rank = r;
    }

    // Local ghost consumers. These are the values this rank must receive from
    // other ranks and place into each local GPU's extended vector.
    std::vector<int> request_counts(nranks, 0);
    for (int dst_gpu = 0; dst_gpu < ngpu; dst_gpu++) {
        for (const auto &cp : gpus[dst_gpu].ghost_copies) {
            if (cp.src_rank != my_rank) request_counts[cp.src_rank]++;
        }
    }

    std::vector<int> request_displs(nranks + 1, 0);
    for (int r = 0; r < nranks; r++) request_displs[r + 1] = request_displs[r] + request_counts[r];
    std::vector<int> request_nodes(request_displs[nranks]);
    std::vector<int> cursor = request_displs;

    for (int dst_gpu = 0; dst_gpu < ngpu; dst_gpu++) {
        for (const auto &cp : gpus[dst_gpu].ghost_copies) {
            if (cp.src_rank != my_rank) {
                int p = cursor[cp.src_rank]++;
                request_nodes[p] = cp.global_node;

                RemoteRecv<T> rr;
                rr.src_rank = cp.src_rank;
                rr.local_gpu = dst_gpu;
                rr.dst_ext_node = cp.dst_ext_node;
                rr.global_node = cp.global_node;
                plans[cp.src_rank].recvs.push_back(rr);
            }
        }
    }

    // Exchange request counts.
    std::vector<int> incoming_counts(nranks, 0);
    MPI_Alltoall(request_counts.data(), 1, MPI_INT,
                 incoming_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> incoming_displs(nranks + 1, 0);
    for (int r = 0; r < nranks; r++) incoming_displs[r + 1] = incoming_displs[r] + incoming_counts[r];
    std::vector<int> incoming_nodes(incoming_displs[nranks]);

    MPI_Alltoallv(request_nodes.data(), request_counts.data(), request_displs.data(), MPI_INT,
                  incoming_nodes.data(), incoming_counts.data(), incoming_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Incoming request nodes determine what this rank must send to each peer.
    for (int dst_rank = 0; dst_rank < nranks; dst_rank++) {
        for (int p = incoming_displs[dst_rank]; p < incoming_displs[dst_rank + 1]; p++) {
            int global_node = incoming_nodes[p];
            int src_gpu = -1;
            int src_local_node = -1;

            for (int g = 0; g < ngpu; g++) {
                if (global_node >= gpus[g].row_start_node && global_node < gpus[g].row_end_node) {
                    src_gpu = g;
                    src_local_node = global_node - gpus[g].row_start_node;
                    break;
                }
            }

            if (src_gpu < 0) {
                printf("rank %d ERROR: peer requested node %d, but it is not locally owned\n",
                       my_rank, global_node);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }

            RemoteSend<T> ss;
            ss.dst_rank = dst_rank;
            ss.local_gpu = src_gpu;
            ss.src_local_node = src_local_node;
            ss.global_node = global_node;
            plans[dst_rank].sends.push_back(ss);
        }
    }

    for (int r = 0; r < nranks; r++) {
        if (r == my_rank) continue;

        plans[r].send_nodes = (int)plans[r].sends.size();
        plans[r].recv_nodes = (int)plans[r].recvs.size();

        if (plans[r].send_nodes > 0) {
            CHECK_CUDA(cudaHostAlloc((void **)&plans[r].h_send,
                                     plans[r].send_nodes * block_dim * sizeof(T),
                                     cudaHostAllocPortable));
            CHECK_CUDA(cudaMalloc((void **)&plans[r].d_send,
                                  plans[r].send_nodes * block_dim * sizeof(T)));
        }
        if (plans[r].recv_nodes > 0) {
            CHECK_CUDA(cudaHostAlloc((void **)&plans[r].h_recv,
                                     plans[r].recv_nodes * block_dim * sizeof(T),
                                     cudaHostAllocPortable));
            CHECK_CUDA(cudaMalloc((void **)&plans[r].d_recv,
                                  plans[r].recv_nodes * block_dim * sizeof(T)));
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
    // First copy owned part into the beginning of x_ext on every local GPU.
#pragma omp parallel for if (ngpu > 1)
    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaMemcpyAsync(gpus[g].d_x_ext, gpus[g].d_x_owned,
                                   gpus[g].local_N * sizeof(T),
                                   cudaMemcpyDeviceToDevice, gpus[g].stream));
    }

    // Same-rank ghost copies stay on GPUs and do not touch MPI.
    for (int dst = 0; dst < ngpu; dst++) {
        CHECK_CUDA(cudaSetDevice(gpus[dst].dev));
        for (const auto &cp : gpus[dst].ghost_copies) {
            if (cp.src_rank == my_rank) {
                int src = cp.src_gpu;
                int can_access = 0;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, gpus[dst].dev, gpus[src].dev));
                if (src != dst && can_access) {
                    // Peer access may already be enabled; ignore cudaErrorPeerAccessAlreadyEnabled.
                    cudaError_t err = cudaDeviceEnablePeerAccess(gpus[src].dev, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        CHECK_CUDA(err);
                    }
                    cudaGetLastError();
                }

                if (src == dst) {
                    CHECK_CUDA(cudaMemcpyAsync(gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                                               gpus[src].d_x_owned + cp.src_local_node * block_dim,
                                               block_dim * sizeof(T), cudaMemcpyDeviceToDevice,
                                               gpus[dst].stream));
                } else if (can_access) {
                    CHECK_CUDA(cudaMemcpyPeerAsync(gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                                                   gpus[dst].dev,
                                                   gpus[src].d_x_owned + cp.src_local_node * block_dim,
                                                   gpus[src].dev,
                                                   block_dim * sizeof(T),
                                                   gpus[dst].stream));
                } else {
                    // Rare same-node fallback if peer access is unavailable.
                    T tmp[64];
                    assert(block_dim <= 64);
                    CHECK_CUDA(cudaSetDevice(gpus[src].dev));
                    CHECK_CUDA(cudaMemcpy(tmp, gpus[src].d_x_owned + cp.src_local_node * block_dim,
                                          block_dim * sizeof(T), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaSetDevice(gpus[dst].dev));
                    CHECK_CUDA(cudaMemcpyAsync(gpus[dst].d_x_ext + cp.dst_ext_node * block_dim,
                                               tmp, block_dim * sizeof(T),
                                               cudaMemcpyHostToDevice, gpus[dst].stream));
                }
            }
        }
    }

    // Pack remote send buffers. This uses tiny per-node copies; for production,
    // replace this with a GPU gather kernel per peer/rank.
    for (int r = 0; r < nranks; r++) {
        if (r == my_rank) continue;
        auto &p = plans[r];

        if (p.send_nodes > 0) {
            for (int i = 0; i < p.send_nodes; i++) {
                const auto &s = p.sends[i];
                CHECK_CUDA(cudaSetDevice(gpus[s.local_gpu].dev));
                CHECK_CUDA(cudaMemcpyAsync(p.h_send + i * block_dim,
                                           gpus[s.local_gpu].d_x_owned + s.src_local_node * block_dim,
                                           block_dim * sizeof(T), cudaMemcpyDeviceToHost,
                                           gpus[s.local_gpu].stream));
            }
        }
    }

    for (int g = 0; g < ngpu; g++) {
        CHECK_CUDA(cudaSetDevice(gpus[g].dev));
        CHECK_CUDA(cudaStreamSynchronize(gpus[g].stream));
    }

    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * nranks);

    for (int r = 0; r < nranks; r++) {
        if (r == my_rank) continue;
        auto &p = plans[r];

        if (p.recv_nodes > 0) {
            MPI_Request req;
            void *recv_ptr = cuda_aware_mpi ? (void *)p.d_recv : (void *)p.h_recv;
            MPI_Irecv(recv_ptr, p.recv_nodes * block_dim, mpiType<T>(), r, 1000 + my_rank,
                      MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }

        if (p.send_nodes > 0) {
            MPI_Request req;
            void *send_ptr = cuda_aware_mpi ? (void *)p.d_send : (void *)p.h_send;
            if (cuda_aware_mpi) {
                // If CUDA-aware MPI is requested, copy host staging into d_send first.
                // For production, pack d_send directly with a gather kernel.
                CHECK_CUDA(cudaMemcpy(p.d_send, p.h_send,
                                      p.send_nodes * block_dim * sizeof(T),
                                      cudaMemcpyHostToDevice));
            }
            MPI_Isend(send_ptr, p.send_nodes * block_dim, mpiType<T>(), r, 1000 + r,
                      MPI_COMM_WORLD, &req);
            reqs.push_back(req);
        }
    }

    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack remote receives into local GPUs' x_ext ghost slots.
    for (int r = 0; r < nranks; r++) {
        if (r == my_rank) continue;
        auto &p = plans[r];
        if (p.recv_nodes == 0) continue;

        if (cuda_aware_mpi) {
            CHECK_CUDA(cudaMemcpy(p.h_recv, p.d_recv,
                                  p.recv_nodes * block_dim * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < p.recv_nodes; i++) {
            const auto &rr = p.recvs[i];
            CHECK_CUDA(cudaSetDevice(gpus[rr.local_gpu].dev));
            CHECK_CUDA(cudaMemcpyAsync(gpus[rr.local_gpu].d_x_ext + rr.dst_ext_node * block_dim,
                                       p.h_recv + i * block_dim,
                                       block_dim * sizeof(T), cudaMemcpyHostToDevice,
                                       gpus[rr.local_gpu].stream));
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

    for (auto &p : plans) {
        if (p.rank == my_rank) continue;
        if (p.h_send) cudaFreeHost(p.h_send);
        if (p.h_recv) cudaFreeHost(p.h_recv);
        if (p.d_send) cudaFree(p.d_send);
        if (p.d_recv) cudaFree(p.d_recv);
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
    buildRankCommPlans<T>(gpus, my_rank, nranks, ngpu, block_dim, plans);

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