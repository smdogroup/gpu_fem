#pragma once

#include <lapacke.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <set>
#include <type_traits>
#include <vector>

#include "_rigid_modes.cuh"
#include "cuda_utils.h"
#include "fake_assembler.h"
#include "lapacke.h"
#include "linalg/vec.h"
#include "multigrid/prolongation/_unstructured.cuh"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T>
__global__ void k_set_identity_blocks(const int nblocks, const int block_dim,
                                      const int *d_block_map, T *vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_dim2 = block_dim * block_dim;
    const int nvals = nblocks * block_dim2;
    if (tid >= nvals) return;

    int ib = tid / block_dim2;
    int local = tid % block_dim2;
    int row = local / block_dim;
    int col = local % block_dim;
    int jp = d_block_map[ib];
    vals[block_dim2 * jp + local] = (row == col) ? T(1) : T(0);
}

template <typename T>
__global__ void k_zero_blocks(const int nblocks, const int block_dim, const int *d_block_map,
                              T *vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_dim2 = block_dim * block_dim;
    const int nvals = nblocks * block_dim2;
    if (tid >= nvals) return;

    int ib = tid / block_dim2;
    int local = tid % block_dim2;
    int jp = d_block_map[ib];
    vals[block_dim2 * jp + local] = T(0);
}

template <typename T>
__global__ void k_copy_node_modes_to_dense_blocks(const int nnodes, const int block_dim,
                                                  const int *d_block_map, const T *d_node_modes,
                                                  T *d_block_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_dim2 = block_dim * block_dim;
    const int nvals = nnodes * block_dim2;
    if (tid >= nvals) return;

    int inode = tid / block_dim2;
    int local = tid % block_dim2;
    int jp = d_block_map[inode];
    d_block_vals[block_dim2 * jp + local] = d_node_modes[block_dim2 * inode + local];
}

template <typename T>
__global__ void k_extract_root_modes(const int nroots, const int block_dim, const int *d_root_nodes,
                                     const T *d_B_vals, T *d_Bc_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_dim2 = block_dim * block_dim;
    const int nvals = nroots * block_dim2;
    if (tid >= nvals) return;

    int iroot = tid / block_dim2;
    int local = tid % block_dim2;
    int inode = d_root_nodes[iroot];
    d_Bc_vals[block_dim2 * iroot + local] = d_B_vals[block_dim2 * inode + local];
}

template <typename T>
__global__ void k_extract_block_column(const int nnodes, const int block_dim, const int icol,
                                       const T *d_block_vals, T *d_vec) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nvals = nnodes * block_dim;
    if (tid >= nvals) return;

    int inode = tid / block_dim;
    int irow = tid % block_dim;

    // row-major block storage: B_i(irow, icol)
    d_vec[tid] = d_block_vals[(inode * block_dim * block_dim) + irow * block_dim + icol];
}

template <typename T>
__global__ void k_insert_block_column(const int nnodes, const int block_dim, const int icol,
                                      const T *d_vec, T *d_block_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nvals = nnodes * block_dim;
    if (tid >= nvals) return;

    int inode = tid / block_dim;
    int irow = tid % block_dim;

    d_block_vals[(inode * block_dim * block_dim) + irow * block_dim + icol] = d_vec[tid];
}

// TODO: replace this with a GPU batched dense solve if you already have one.
// For now this is a host-side prototype helper that computes Ti = B_i * inv(Bc_a).
// template <typename T>
// static void compute_rootnode_dense_blocks_host(int nnodes, int block_dim,
//                                                const int *h_aggregate_ind, const bool *h_C_nodes,
//                                                const T *h_B_vals, const T *h_Bc_vals, T
//                                                *h_T_vals) {
//     std::vector<T> Binv(block_dim * block_dim);
//     std::vector<int> ipiv(block_dim);

//     for (int inode = 0; inode < nnodes; inode++) {
//         T *Tblock = &h_T_vals[block_dim * block_dim * inode];
//         if (h_C_nodes[inode]) {
//             for (int i = 0; i < block_dim; i++) {
//                 for (int j = 0; j < block_dim; j++) {
//                     Tblock[block_dim * i + j] = (i == j) ? T(1) : T(0);
//                 }
//             }
//             continue;
//         }

//         int agg = h_aggregate_ind[inode];
//         const T *Bblock = &h_B_vals[block_dim * block_dim * inode];
//         const T *Bcblock = &h_Bc_vals[block_dim * block_dim * agg];

//         std::memcpy(Binv.data(), Bcblock, block_dim * block_dim * sizeof(T));
//         std::memcpy(Tblock, Bblock, block_dim * block_dim * sizeof(T));

//         int n = block_dim;
//         int nrhs = block_dim;
//         int lda = block_dim;
//         int ldb = block_dim;

//         if constexpr (std::is_same_v<T, double>) {
//             LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, reinterpret_cast<double *>(Binv.data()),
//             lda,
//                           ipiv.data(), reinterpret_cast<double *>(Tblock), ldb);
//         } else {
//             // Prototype currently assumes double.
//         }

//         printf("node %d, B block\n", inode);
//         for (int row = 0; row < 6; row++) {
//             printVec<T>(6, &Bblock[6 * row]);
//         }
//         printf("node %d, Bc block\n", inode);
//         for (int row = 0; row < 6; row++) {
//             printVec<T>(6, &Bcblock[6 * row]);
//         }
//         printf("node %d, T block\n", inode);
//         for (int row = 0; row < 6; row++) {
//             printVec<T>(6, &Tblock[6 * row]);
//         }
//     }
// }

// Form A_pinv = A^+ using SVD, row-major storage.
// A is m x n, here we use m = n = block_dim, but this works generally.
static int compute_pseudoinverse_rowmajor_double(const double *A, int m, int n, double rcond,
                                                 double *A_pinv) {
    std::vector<double> Acopy(A, A + m * n);

    const int k = std::min(m, n);
    std::vector<double> S(k);
    std::vector<double> U(m * m);
    std::vector<double> VT(n * n);
    std::vector<double> superb(std::max(1, k - 1));

    // Acopy = U * diag(S) * VT
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, Acopy.data(), n, S.data(), U.data(),
                              m, VT.data(), n, superb.data());
    if (info != 0) {
        return info;
    }

    const double smax = (k > 0) ? S[0] : 0.0;
    const double tol = rcond * smax;

    // A^+ = V * diag(S^+) * U^T
    // VT is n x n, so V = VT^T.
    std::fill(A_pinv, A_pinv + n * m, 0.0);

    for (int p = 0; p < k; p++) {
        if (S[p] <= tol) {
            continue;
        }
        const double sinv = 1.0 / S[p];

        // outer product of column p of V and column p of U:
        // A_pinv += (1/S[p]) * V[:,p] * U[:,p]^T
        for (int i = 0; i < n; i++) {
            const double vip = VT[p * n + i];  // V(i,p) = VT(p,i)
            for (int j = 0; j < m; j++) {
                const double ujp = U[j * m + p];  // U(j,p)
                A_pinv[i * m + j] += sinv * vip * ujp;
            }
        }
    }

    return 0;
}

// Row-major C = A * B
static void dense_matmul_rowmajor_double(const double *A, const double *B, double *C, int m, int k,
                                         int n) {
    // A: m x k, B: k x n, C: m x n
    std::fill(C, C + m * n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            const double aip = A[i * k + p];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += aip * B[p * n + j];
            }
        }
    }
}

// Host-side prototype helper that computes Ti = B_i * pinv(Bc_a).
template <typename T>
static void compute_rootnode_dense_blocks_host(int nnodes, int block_dim,
                                               const int *h_aggregate_ind, const bool *h_C_nodes,
                                               const T *h_B_vals, const T *h_Bc_vals, T *h_T_vals,
                                               double rcond = 1e-12) {
    if constexpr (!std::is_same_v<T, double>) {
        printf("compute_rootnode_dense_blocks_host currently only supports double\n");
        return;
    }

    std::vector<double> Bc_pinv(block_dim * block_dim);
    std::vector<double> Ttmp(block_dim * block_dim);

    for (int inode = 0; inode < nnodes; inode++) {
        T *Tblock = &h_T_vals[block_dim * block_dim * inode];

        if (h_C_nodes[inode]) {
            for (int i = 0; i < block_dim; i++) {
                for (int j = 0; j < block_dim; j++) {
                    Tblock[block_dim * i + j] = (i == j) ? T(1) : T(0);
                }
            }
            continue;
        }

        const int agg = h_aggregate_ind[inode];
        const T *Bblock = &h_B_vals[block_dim * block_dim * inode];
        const T *Bcblock = &h_Bc_vals[block_dim * block_dim * agg];

        int info = compute_pseudoinverse_rowmajor_double(
            reinterpret_cast<const double *>(Bcblock), block_dim, block_dim, rcond, Bc_pinv.data());

        if (info != 0) {
            printf("WARNING: pseudoinverse SVD failed for node %d agg %d with info = %d\n", inode,
                   agg, info);
            std::fill(Tblock, Tblock + block_dim * block_dim, T(0));
            continue;
        }

        dense_matmul_rowmajor_double(reinterpret_cast<const double *>(Bblock), Bc_pinv.data(),
                                     Ttmp.data(), block_dim, block_dim, block_dim);

        for (int i = 0; i < block_dim * block_dim; i++) {
            Tblock[i] = static_cast<T>(Ttmp[i]);
        }

        // printf("node %d, B block\n", inode);
        // for (int row = 0; row < block_dim; row++) {
        //     printVec<T>(block_dim, &Bblock[block_dim * row]);
        // }

        // printf("node %d, Bc block\n", inode);
        // for (int row = 0; row < block_dim; row++) {
        //     printVec<T>(block_dim, &Bcblock[block_dim * row]);
        // }

        // printf("node %d, T block\n", inode);
        // for (int row = 0; row < block_dim; row++) {
        //     printVec<T>(block_dim, &Tblock[block_dim * row]);
        // }
    }
}

template <typename T, class FAKE_ASSEMBLER, class Smoother, bool ORTHOG_PROJECTOR = true,
          bool LINE_SEARCH = false>
class RootNodeAMG : public BaseSolver {
   public:
    // using Assembler = FakeAssembler<T>;
    using Assembler = FAKE_ASSEMBLER;
    using CoarseMG = RootNodeAMG<T, FAKE_ASSEMBLER, Smoother, ORTHOG_PROJECTOR>;
    using CoarseDirect = CusparseMGDirectLU<T, Assembler>;

    RootNodeAMG(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                Smoother *smoother_, int nnodes_, BsrMat<DeviceVec<T>> kmat_,
                BsrMat<DeviceVec<T>> kmat_free_, DeviceVec<T> rigid_body_modes_,
                DeviceVec<int> d_bcs_, int coarse_node_threshold_ = 6000,
                T sparse_threshold_ = 0.15, T omegaJac_ = 0.7, int nsmooth_ = 1, int level_ = 0,
                int rbm_nsmooth_ = 1, int prol_nsmooth_ = 3,
                std::string coarsening_type_ = "standard")
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          smoother(smoother_),
          kmat(kmat_),
          kmat_free(kmat_free_),
          nnodes(nnodes_),
          rigid_body_modes(rigid_body_modes_),
          coarse_node_threshold(coarse_node_threshold_),
          sparse_threshold(sparse_threshold_),
          level(level_),
          nsmooth(nsmooth_),
          rbm_nsmooth(rbm_nsmooth_),
          prol_nsmooth(prol_nsmooth_),
          coarsening_type(coarsening_type_) {
        auto ctor_t0 = clock_type::now();

        auto d_kmat_bsr_data = kmat.getBsrData();
        d_kmat_vals = kmat.getVec().getPtr();
        d_kmat_free_vals = kmat_free.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_rows = d_kmat_bsr_data.rows;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        block_dim = d_kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;
        N = nnodes * block_dim;
        omegaJac = omegaJac_;
        d_bcs = d_bcs_;
        coarsening_type = "standard";

        // initCuda();
        // build_cf_pattern();
        // build_rootnode_aggregates();

        // is_coarse_mg = num_aggregates > coarse_node_threshold;
        // compute_prolongation_nz_pattern();
        // compute_coarse_grid_nz_pattern();
        // compute_coarse_problem();

        {
            auto t0 = clock_type::now();
            initCuda();
            sync_if_needed();
            print_setup_time("initCuda", elapsed_sec(t0, clock_type::now()));
        }

        {
            auto t0 = clock_type::now();
            build_cf_pattern();
            sync_if_needed();
            print_setup_time("build_cf_pattern", elapsed_sec(t0, clock_type::now()));
        }

        {
            auto t0 = clock_type::now();
            build_rootnode_aggregates();
            sync_if_needed();
            print_setup_time("build_rootnode_aggregates", elapsed_sec(t0, clock_type::now()));
        }

        is_coarse_mg = num_aggregates > coarse_node_threshold;

        {
            auto t0 = clock_type::now();
            compute_prolongation_nz_pattern();
            sync_if_needed();
            print_setup_time("compute_prolongation_nz_pattern", elapsed_sec(t0, clock_type::now()));
        }

        {
            auto t0 = clock_type::now();
            compute_coarse_grid_nz_pattern();
            sync_if_needed();
            print_setup_time("compute_coarse_grid_nz_pattern", elapsed_sec(t0, clock_type::now()));
        }

        {
            auto t0 = clock_type::now();
            compute_coarse_problem();
            sync_if_needed();
            print_setup_time("compute_coarse_problem", elapsed_sec(t0, clock_type::now()));
        }

        print_setup_time("constructor total", elapsed_sec(ctor_t0, clock_type::now()));
    }

    int get_total_nnzb() {
        int c_nnzb = P_nnzb * 2 + kmat_nnzb;
        if (is_coarse_mg) {
            c_nnzb += coarse_mg->get_total_nnzb();
        } else {
            c_nnzb += coarse_direct->get_nnzb();
        }
        return c_nnzb;
    }
    T get_operator_complexity(int nofill_nnzb) {
        return T(get_total_nnzb()) * 1.0 / T(nofill_nnzb);
    }

    void compute_coarse_problem() {
        // compute_prolongator_values();
        // compute_coarse_grid_values();
        auto t0 = clock_type::now();
        compute_prolongator_values();
        sync_if_needed();
        print_setup_time("compute_prolongator_values", elapsed_sec(t0, clock_type::now()));

        t0 = clock_type::now();
        compute_coarse_grid_values();
        sync_if_needed();
        print_setup_time("compute_coarse_grid_values", elapsed_sec(t0, clock_type::now()));
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // TODO
    }
    void factor() {}
    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one
    void set_cycle_type(std::string cycle_) {}
    void set_matrix_nsmooth(int nsmooth_) { prol_nsmooth = nsmooth_; }
    void set_rbm_nsmooth(int nsmooth_) {
        // printf("set rbm nsmooth %d\n", nsmooth_);
        rbm_nsmooth = nsmooth_;
    }

    void build_coarse_system(Assembler coarse_assembler, Smoother *coarse_smoother) {
        if (!is_coarse_mg) {
            coarse_direct =
                new CoarseDirect(cublasHandle, cusparseHandle, coarse_assembler, coarse_kmat);
        } else {
            auto no_bcs = DeviceVec<int>(0);
            coarse_mg = new CoarseMG(cublasHandle, cusparseHandle, coarse_smoother, num_aggregates,
                                     coarse_kmat, coarse_free_kmat, d_Bc_vec, no_bcs,
                                     coarse_node_threshold, sparse_threshold, omegaJac, nsmooth,
                                     level + 1, rbm_nsmooth, prol_nsmooth, coarsening_type);
        }
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_inner_soln, 0, N * sizeof(T));

        // pre-smooth
        this->smoother->smoothDefect(d_rhs_vec, d_inner_soln_vec, nsmooth);

        // restrict residual/defect to coarse grid
        d_coarse_rhs_vec.zeroValues();
        int nprods = P_nnzb * block_dim2;
        dim3 block0(32), grid0((nprods + 31) / 32);
        k_bsrmv_transpose<T><<<grid0, block0>>>(P_nnzb, block_dim, d_prolong_rows, d_prolong_cols,
                                                d_prolong_vals, d_rhs_vec.getPtr(),
                                                d_coarse_rhs_vec.getPtr());
        CHECK_CUDA(cudaDeviceSynchronize());

        // coarse solve
        if (!is_coarse_mg) {
            coarse_direct->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        } else {
            coarse_mg->solve(d_coarse_rhs_vec, d_coarse_soln_vec);
        }

        // prolong coarse correction: d_temp = P * e_c
        T a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes,
            num_aggregates, P_nnzb, &a, descrKmat, d_prolong_vals, d_prolong_rowp, d_prolong_cols,
            block_dim, d_coarse_soln_vec.getPtr(), &b, d_temp));

        // compute K * d_temp into d_temp2
        a = 1.0;
        b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));

        // optional 1D line search on coarse correction
        T omega = 1.0;
        if constexpr (LINE_SEARCH) {
            T sT_rhs = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_rhs, 1, d_temp, 1, &sT_rhs));

            T sT_Ks = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));

            // avoid divide-by-zero / pathological tiny denominator
            if (fabs(sT_Ks) > 1e-30) {
                omega = sT_rhs / sT_Ks;
                // omega = std::clamp(omega, omega_min, omega_max);
            } else {
                omega = 1.0;
            }
        }

        // apply coarse correction to solution
        a = omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_inner_soln, 1));

        // update defect: r <- r - omega * K * d_temp
        a = -omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_rhs, 1));

        // post-smooth
        this->smoother->smoothDefect(d_rhs_vec, d_inner_soln_vec, nsmooth);

        cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);
        return false;
    }

    BsrData get_coarse_bsr_data() { return coarse_kmat_bsr_data; }
    int get_num_aggregates() { return num_aggregates; }
    bool get_coarse_mg() { return is_coarse_mg; }
    BsrMat<DeviceVec<T>> get_coarse_kmat() { return coarse_kmat; }

   private:
    void initCuda() {
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        int *h_kmat_diagp = new int[nnodes];
        for (int i = 0; i < nnodes; i++) {
            h_kmat_diagp[i] = -1;
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                if (h_kmat_cols[jp] == i) {
                    h_kmat_diagp[i] = jp;
                    break;
                }
            }
        }
        d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

        d_diag_norms = DeviceVec<T>(nnodes).getPtr();
        d_strength_indicator = DeviceVec<bool>(kmat_nnzb).getPtr();

        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();
    }

    void build_cf_pattern() {
        k_get_diag_norms<T>
            <<<nnodes, 32>>>(nnodes, d_kmat_diagp, block_dim, d_kmat_free_vals, d_diag_norms);
        k_compute_strength_bools<T><<<kmat_nnzb, 32>>>(kmat_nnzb, block_dim, d_diag_norms,
                                                       d_kmat_rows, d_kmat_cols, d_kmat_free_vals,
                                                       sparse_threshold, d_strength_indicator);
        CHECK_CUDA(cudaDeviceSynchronize());

        h_strength_indicator =
            DeviceVec<bool>(kmat_nnzb, d_strength_indicator).createHostVec().getPtr();

        int strength_nnz = 0;
        for (int ib = 0; ib < kmat_nnzb; ib++) {
            if (h_strength_indicator[ib]) strength_nnz++;
        }

        h_strength_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_strength_cols = HostVec<int>(strength_nnz).getPtr();
        h_strength_rowp[0] = 0;
        for (int i = 0; i < nnodes; i++) {
            h_strength_rowp[i + 1] = h_strength_rowp[i];
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                if (h_strength_indicator[jp]) {
                    h_strength_cols[h_strength_rowp[i + 1]++] = h_kmat_cols[jp];
                }
            }
        }

        if (coarsening_type == "standard") {
            form_cf_splitting();  // standard coarsening (makes it really slow to compute higher DOF
            //                       // eventually with high fillin to coarse)
        } else if (coarsening_type == "aggressive_A2") {
            form_aggressive_A2_cf_splitting();  // aggressive A2 coarsening
        }

        Nc = num_coarse_nodes * block_dim;
        d_coarse_rhs_vec = DeviceVec<T>(Nc);
        d_coarse_rhs = d_coarse_rhs_vec.getPtr();
        d_coarse_soln_vec = DeviceVec<T>(Nc);
        d_coarse_soln = d_coarse_soln_vec.getPtr();
    }

    void build_strength_transpose() {
        h_strength_tr_rowp = HostVec<int>(nnodes + 1).getPtr();
        int strength_nnz = h_strength_rowp[nnodes];
        h_strength_tr_cols = HostVec<int>(strength_nnz).getPtr();
        int *row_cts = HostVec<int>(nnodes).getPtr();
        std::memset(row_cts, 0, nnodes * sizeof(int));
        std::memset(h_strength_tr_rowp, 0, (nnodes + 1) * sizeof(int));

        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                row_cts[h_strength_cols[jp]]++;
            }
        }
        for (int i = 0; i < nnodes; i++) {
            h_strength_tr_rowp[i + 1] = h_strength_tr_rowp[i] + row_cts[i];
        }
        std::memset(row_cts, 0, nnodes * sizeof(int));
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                int j = h_strength_cols[jp];
                int ip = h_strength_tr_rowp[j] + row_cts[j]++;
                h_strength_tr_cols[ip] = i;
            }
        }
    }

    // void form_cf_splitting() {
    //     h_C_nodes = HostVec<bool>(nnodes).getPtr();
    //     h_F_nodes = HostVec<bool>(nnodes).getPtr();
    //     bool *h_U = HostVec<bool>(nnodes).getPtr();
    //     std::memset(h_C_nodes, 0, nnodes * sizeof(bool));
    //     std::memset(h_F_nodes, 0, nnodes * sizeof(bool));
    //     std::memset(h_U, 1, nnodes * sizeof(bool));

    //     build_strength_transpose();
    //     std::vector<int> LAM(nnodes, 0);
    //     for (int i = 0; i < nnodes; i++) {
    //         LAM[i] = h_strength_tr_rowp[i + 1] - h_strength_tr_rowp[i];
    //     }

    //     int num_unassigned = nnodes;
    //     while (num_unassigned > 0) {
    //         int best_i = -1, best_lam = -1;
    //         for (int i = 0; i < nnodes; i++) {
    //             if (h_U[i] && LAM[i] > best_lam) {
    //                 best_lam = LAM[i];
    //                 best_i = i;
    //             }
    //         }
    //         if (best_i < 0) break;

    //         if (LAM[best_i] == 0) {
    //             h_C_nodes[best_i] = true;
    //             h_U[best_i] = false;
    //             num_unassigned--;
    //             continue;
    //         }

    //         int i = best_i;
    //         h_C_nodes[i] = true;
    //         h_U[i] = false;
    //         num_unassigned--;

    //         for (int jp = h_strength_tr_rowp[i]; jp < h_strength_tr_rowp[i + 1]; jp++) {
    //             int j = h_strength_tr_cols[jp];
    //             if (!h_U[j]) continue;
    //             h_F_nodes[j] = true;
    //             h_U[j] = false;
    //             num_unassigned--;
    //             for (int kp = h_strength_rowp[j]; kp < h_strength_rowp[j + 1]; kp++) {
    //                 int k = h_strength_cols[kp];
    //                 if (h_U[k]) LAM[k] += 2;
    //             }
    //         }
    //         for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
    //             int j = h_strength_cols[jp];
    //             if (h_U[j]) LAM[j] -= 1;
    //         }
    //     }

    //     num_coarse_nodes = 0;
    //     for (int i = 0; i < nnodes; i++) {
    //         if (h_C_nodes[i]) num_coarse_nodes++;
    //     }

    //     h_coarse_id = HostVec<int>(nnodes).getPtr();
    //     h_coarse_nodes = HostVec<int>(num_coarse_nodes).getPtr();
    //     int ic = 0;
    //     for (int i = 0; i < nnodes; i++) {
    //         if (h_C_nodes[i]) {
    //             h_coarse_id[i] = ic;
    //             h_coarse_nodes[ic] = i;
    //             ic++;
    //         } else {
    //             h_coarse_id[i] = -1;
    //         }
    //     }
    //     num_aggregates = num_coarse_nodes;

    //     // DEBUG:
    //     // printf("C_nodes: ");
    //     // for (int i = 0; i < nnodes; i++) {
    //     //     if (h_C_nodes[i]) printf("%d ", i);
    //     // }
    //     // printf("\n");
    //     // printf("F_nodes: ");
    //     // for (int i = 0; i < nnodes; i++) {
    //     //     if (h_F_nodes[i]) printf("%d ", i);
    //     // }
    //     // printf("\n");
    // }

    static void run_standard_cf_splitting_from_strength(int n, const int *strength_rowp,
                                                        const int *strength_cols,
                                                        const int *strength_tr_rowp,
                                                        const int *strength_tr_cols, bool *C_nodes,
                                                        bool *F_nodes) {
        bool *U = HostVec<bool>(n).getPtr();
        std::memset(C_nodes, 0, n * sizeof(bool));
        std::memset(F_nodes, 0, n * sizeof(bool));
        for (int i = 0; i < n; i++) U[i] = true;

        std::vector<int> LAM(n, 0);
        for (int i = 0; i < n; i++) {
            LAM[i] = strength_tr_rowp[i + 1] - strength_tr_rowp[i];
        }

        int num_unassigned = n;
        while (num_unassigned > 0) {
            int best_i = -1;
            int best_lam = -1;

            for (int i = 0; i < n; i++) {
                if (U[i] && LAM[i] > best_lam) {
                    best_lam = LAM[i];
                    best_i = i;
                }
            }
            if (best_i < 0) break;

            if (LAM[best_i] == 0) {
                C_nodes[best_i] = true;
                U[best_i] = false;
                num_unassigned--;
                continue;
            }

            int i = best_i;
            C_nodes[i] = true;
            U[i] = false;
            num_unassigned--;

            // Mark transpose-neighbors as fine
            for (int jp = strength_tr_rowp[i]; jp < strength_tr_rowp[i + 1]; jp++) {
                int j = strength_tr_cols[jp];
                if (!U[j]) continue;

                F_nodes[j] = true;
                U[j] = false;
                num_unassigned--;

                // Promote neighbors of newly created F-point
                for (int kp = strength_rowp[j]; kp < strength_rowp[j + 1]; kp++) {
                    int k = strength_cols[kp];
                    if (U[k]) {
                        LAM[k] += 2;
                    }
                }
            }

            // Decrease LAM of forward neighbors of the new C-point
            for (int jp = strength_rowp[i]; jp < strength_rowp[i + 1]; jp++) {
                int j = strength_cols[jp];
                if (U[j]) {
                    LAM[j] -= 1;
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Helper: pack adjacency lists into CSR
    // -----------------------------------------------------------------------------
    static void adjacency_to_csr(const std::vector<std::vector<int>> &adj, std::vector<int> &rowp,
                                 std::vector<int> &cols) {
        int n = static_cast<int>(adj.size());
        rowp.resize(n + 1);
        rowp[0] = 0;
        for (int i = 0; i < n; i++) {
            rowp[i + 1] = rowp[i] + static_cast<int>(adj[i].size());
        }

        cols.resize(rowp[n]);
        for (int i = 0; i < n; i++) {
            int p = rowp[i];
            for (int j : adj[i]) {
                cols[p++] = j;
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Existing standard CF splitting, now rewritten to use helper
    // -----------------------------------------------------------------------------
    void form_cf_splitting() {
        h_C_nodes = HostVec<bool>(nnodes).getPtr();
        h_F_nodes = HostVec<bool>(nnodes).getPtr();

        build_strength_transpose();

        run_standard_cf_splitting_from_strength(nnodes, h_strength_rowp, h_strength_cols,
                                                h_strength_tr_rowp, h_strength_tr_cols, h_C_nodes,
                                                h_F_nodes);

        num_coarse_nodes = 0;
        for (int i = 0; i < nnodes; i++) {
            if (h_C_nodes[i]) num_coarse_nodes++;
        }

        h_coarse_id = HostVec<int>(nnodes).getPtr();
        h_coarse_nodes = HostVec<int>(num_coarse_nodes).getPtr();

        int ic = 0;
        for (int i = 0; i < nnodes; i++) {
            if (h_C_nodes[i]) {
                h_coarse_id[i] = ic;
                h_coarse_nodes[ic] = i;
                ic++;
            } else {
                h_coarse_id[i] = -1;
            }
        }

        num_aggregates = num_coarse_nodes;
    }

    // -----------------------------------------------------------------------------
    // New aggressive A2 coarsening based on the attached Python
    // -----------------------------------------------------------------------------
    void form_aggressive_A2_cf_splitting() {
        // ---------------------------------------------------------
        // Stage 1: standard coarsening on original graph
        // ---------------------------------------------------------
        h_C_nodes = HostVec<bool>(nnodes).getPtr();
        h_F_nodes = HostVec<bool>(nnodes).getPtr();

        build_strength_transpose();

        run_standard_cf_splitting_from_strength(nnodes, h_strength_rowp, h_strength_cols,
                                                h_strength_tr_rowp, h_strength_tr_cols, h_C_nodes,
                                                h_F_nodes);

        // Gather global coarse nodes from stage 1
        std::vector<int> coarse_nodes_vec;
        coarse_nodes_vec.reserve(nnodes);
        for (int i = 0; i < nnodes; i++) {
            if (h_C_nodes[i]) {
                coarse_nodes_vec.push_back(i);
            }
        }

        const int n_coarse = static_cast<int>(coarse_nodes_vec.size());

        // Trivial fallback
        if (n_coarse == 0) {
            num_coarse_nodes = 0;
            h_coarse_id = HostVec<int>(nnodes).getPtr();
            for (int i = 0; i < nnodes; i++) h_coarse_id[i] = -1;
            h_coarse_nodes = HostVec<int>(0).getPtr();
            num_aggregates = 0;
            return;
        }

        // Map global coarse node -> local coarse index
        std::vector<int> coarse_map(nnodes, -1);
        for (int ic = 0; ic < n_coarse; ic++) {
            coarse_map[coarse_nodes_vec[ic]] = ic;
        }

        // ---------------------------------------------------------
        // Stage 2: build second-level C=>F=>C adjacency with
        // at least two distinct paths
        // ---------------------------------------------------------
        std::vector<std::vector<int>> strength2(nnodes);
        std::vector<std::vector<int>> strength_tr2(nnodes);

        for (int i = 0; i < nnodes; i++) {
            if (!h_C_nodes[i]) continue;

            // Forward second-level: i -> f -> j, where f is fine and j is coarse
            {
                std::vector<int> counts(nnodes, 0);

                for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                    int f = h_strength_cols[jp];
                    if (h_C_nodes[f]) continue;  // only first hop to F

                    for (int kp = h_strength_rowp[f]; kp < h_strength_rowp[f + 1]; kp++) {
                        int j = h_strength_cols[kp];
                        if (h_C_nodes[j]) {
                            counts[j]++;
                        }
                    }
                }

                for (int j = 0; j < nnodes; j++) {
                    if (counts[j] >= 2) {
                        strength2[i].push_back(j);
                    }
                }
            }

            // Transpose second-level: i <- f <- j using transpose graph
            {
                std::vector<int> counts_tr(nnodes, 0);

                for (int jp = h_strength_tr_rowp[i]; jp < h_strength_tr_rowp[i + 1]; jp++) {
                    int f = h_strength_tr_cols[jp];
                    if (h_C_nodes[f]) continue;  // only first hop to F

                    for (int kp = h_strength_tr_rowp[f]; kp < h_strength_tr_rowp[f + 1]; kp++) {
                        int j = h_strength_tr_cols[kp];
                        if (h_C_nodes[j]) {
                            counts_tr[j]++;
                        }
                    }
                }

                for (int j = 0; j < nnodes; j++) {
                    if (counts_tr[j] >= 2) {
                        strength_tr2[i].push_back(j);
                    }
                }
            }
        }

        // ---------------------------------------------------------
        // Restrict second-level graph to coarse nodes only
        // ---------------------------------------------------------
        std::vector<std::vector<int>> rstrength2(n_coarse);
        std::vector<std::vector<int>> rstrength_tr2(n_coarse);

        for (int ilocal = 0; ilocal < n_coarse; ilocal++) {
            int iglobal = coarse_nodes_vec[ilocal];

            for (int jglobal : strength2[iglobal]) {
                int jlocal = coarse_map[jglobal];
                if (jlocal >= 0) {
                    rstrength2[ilocal].push_back(jlocal);
                }
            }

            for (int jglobal : strength_tr2[iglobal]) {
                int jlocal = coarse_map[jglobal];
                if (jlocal >= 0) {
                    rstrength_tr2[ilocal].push_back(jlocal);
                }
            }
        }

        // Convert restricted adjacency to CSR
        std::vector<int> rstrength2_rowp, rstrength2_cols;
        std::vector<int> rstrength_tr2_rowp, rstrength_tr2_cols;

        adjacency_to_csr(rstrength2, rstrength2_rowp, rstrength2_cols);
        adjacency_to_csr(rstrength_tr2, rstrength_tr2_rowp, rstrength_tr2_cols);

        // ---------------------------------------------------------
        // Stage 3: standard coarsening on restricted coarse graph
        // ---------------------------------------------------------
        bool *C2 = HostVec<bool>(n_coarse).getPtr();
        bool *F2 = HostVec<bool>(n_coarse).getPtr();
        for (int i = 0; i < n_coarse; i++) {
            C2[i] = false;
            F2[i] = false;
        }

        run_standard_cf_splitting_from_strength(n_coarse, rstrength2_rowp.data(),
                                                rstrength2_cols.data(), rstrength_tr2_rowp.data(),
                                                rstrength_tr2_cols.data(), C2, F2);

        // ---------------------------------------------------------
        // Stage 4: map decisions back to original graph
        //
        // Python logic:
        // if F2[local_idx], demote that old coarse node to fine
        // ---------------------------------------------------------
        for (int ilocal = 0; ilocal < n_coarse; ilocal++) {
            int iglobal = coarse_nodes_vec[ilocal];
            if (F2[ilocal]) {
                h_C_nodes[iglobal] = false;
                h_F_nodes[iglobal] = true;
            }
        }

        // Rebuild coarse indexing
        num_coarse_nodes = 0;
        for (int i = 0; i < nnodes; i++) {
            if (h_C_nodes[i]) num_coarse_nodes++;
        }

        h_coarse_id = HostVec<int>(nnodes).getPtr();
        h_coarse_nodes = HostVec<int>(num_coarse_nodes).getPtr();

        int ic = 0;
        for (int i = 0; i < nnodes; i++) {
            if (h_C_nodes[i]) {
                h_coarse_id[i] = ic;
                h_coarse_nodes[ic] = i;
                ic++;
            } else {
                h_coarse_id[i] = -1;
            }
        }

        num_aggregates = num_coarse_nodes;
    }

    void build_rootnode_aggregates() {
        h_aggregate_ind = HostVec<int>(nnodes).getPtr();
        std::memset(h_aggregate_ind, -1, nnodes * sizeof(int));

        std::vector<std::vector<int>> aggregate_groups(num_coarse_nodes);
        std::vector<int> root_of_agg(num_coarse_nodes, -1);
        for (int ic = 0; ic < num_coarse_nodes; ic++) {
            int root = h_coarse_nodes[ic];
            root_of_agg[ic] = root;
            aggregate_groups[ic].push_back(root);
            h_aggregate_ind[root] = ic;
        }

        for (int i = 0; i < nnodes; i++) {
            if (h_aggregate_ind[i] != -1) continue;
            std::vector<int> candidate_aggs;
            for (int ic = 0; ic < num_coarse_nodes; ic++) {
                int root = root_of_agg[ic];
                bool strong_to_root = false;
                for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                    if (h_strength_cols[jp] == root) {
                        strong_to_root = true;
                        break;
                    }
                }
                if (strong_to_root) candidate_aggs.push_back(ic);
            }
            if (candidate_aggs.empty()) continue;

            int best_agg = candidate_aggs[0];
            int best_size = aggregate_groups[best_agg].size();
            for (int agg : candidate_aggs) {
                int sz = aggregate_groups[agg].size();
                if (sz < best_size) {
                    best_size = sz;
                    best_agg = agg;
                }
            }
            aggregate_groups[best_agg].push_back(i);
            h_aggregate_ind[i] = best_agg;
        }

        for (int i = 0; i < nnodes; i++) {
            if (h_aggregate_ind[i] != -1) continue;
            std::set<int> nb_aggs;
            for (int jp = h_strength_rowp[i]; jp < h_strength_rowp[i + 1]; jp++) {
                int j = h_strength_cols[jp];
                if (h_aggregate_ind[j] != -1) nb_aggs.insert(h_aggregate_ind[j]);
            }
            if (nb_aggs.empty()) continue;

            int best_agg = *nb_aggs.begin();
            int best_size = aggregate_groups[best_agg].size();
            for (int agg : nb_aggs) {
                int sz = aggregate_groups[agg].size();
                if (sz < best_size) {
                    best_size = sz;
                    best_agg = agg;
                }
            }
            aggregate_groups[best_agg].push_back(i);
            h_aggregate_ind[i] = best_agg;
        }

        // printf("node aggregates: \n");
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     printf("aggregate %d: ", iagg);
        //     for (int i = 0; i < nnodes; i++) {
        //         if (h_aggregate_ind[i] == iagg) printf("%d ", i);
        //     }
        //     printf("\n");
        // }

        d_aggregate_ind = HostVec<int>(nnodes, h_aggregate_ind).createDeviceVec().getPtr();
        d_root_nodes = HostVec<int>(num_aggregates, h_coarse_nodes).createDeviceVec().getPtr();
    }

    void compute_prolongation_nz_pattern() {
        h_root_block_map = HostVec<int>(num_aggregates).getPtr();
        h_node_block_map = HostVec<int>(nnodes).getPtr();

        std::vector<int> prolong_rowp(nnodes + 1, 0);
        std::vector<int> prolong_cols;

        for (int i = 0; i < nnodes; i++) {
            std::set<int> unique_cols;

            if (h_C_nodes[i]) {
                // C-row gets identity only
                unique_cols.insert(h_coarse_id[i]);
            } else {
                // ------------------------------------------------------------
                // First part: direct A_FC pattern
                // ------------------------------------------------------------
                for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                    int j = h_kmat_cols[jp];
                    if (h_C_nodes[j]) {
                        unique_cols.insert(h_coarse_id[j]);
                    }
                }

                // ------------------------------------------------------------
                // Second part: A_FF * A_FC fill
                // i(F) -> j(F) -> k(C)
                // ------------------------------------------------------------
                for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                    int j = h_kmat_cols[jp];
                    if (!h_F_nodes[j]) continue;

                    for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j + 1]; kp++) {
                        int k = h_kmat_cols[kp];
                        if (h_C_nodes[k]) {
                            unique_cols.insert(h_coarse_id[k]);
                        }
                    }
                }
            }
            // also add the aggregate index here..
            int my_agg = h_aggregate_ind[i];
            if (my_agg >= 0) {
                unique_cols.insert(my_agg);
            }

            prolong_rowp[i + 1] = prolong_rowp[i] + static_cast<int>(unique_cols.size());
            for (int col : unique_cols) {
                prolong_cols.push_back(col);
            }
        }

        P_nnzb = static_cast<int>(prolong_cols.size());

        h_prolong_rowp = HostVec<int>(nnodes + 1).getPtr();
        h_prolong_rows = HostVec<int>(P_nnzb).getPtr();
        h_prolong_cols = HostVec<int>(P_nnzb).getPtr();

        memcpy(h_prolong_rowp, prolong_rowp.data(), (nnodes + 1) * sizeof(int));
        memcpy(h_prolong_cols, prolong_cols.data(), P_nnzb * sizeof(int));

        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                h_prolong_rows[jp] = i;
            }
        }

        // ------------------------------------------------------------
        // Build maps into the actual prolongation block indices
        // ------------------------------------------------------------
        for (int i = 0; i < nnodes; i++) {
            h_node_block_map[i] = -1;
        }
        for (int agg = 0; agg < num_aggregates; agg++) {
            h_root_block_map[agg] = -1;
        }

        for (int i = 0; i < nnodes; i++) {
            int my_agg = h_aggregate_ind[i];

            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int col = h_prolong_cols[jp];

                // map each node row to its "home" aggregate block
                if (col == my_agg) {
                    h_node_block_map[i] = jp;
                }

                // map each root aggregate to its exact injection block
                if (h_C_nodes[i] && col == h_coarse_id[i]) {
                    h_root_block_map[h_coarse_id[i]] = jp;
                }
            }
        }

        // optional sanity checks
        for (int i = 0; i < nnodes; i++) {
            if (h_node_block_map[i] < 0) {
                printf("ERROR: h_node_block_map[%d] not found (agg=%d)\n", i, h_aggregate_ind[i]);
            }
        }
        for (int agg = 0; agg < num_aggregates; agg++) {
            if (h_root_block_map[agg] < 0) {
                printf("ERROR: h_root_block_map[%d] not found\n", agg);
            }
        }

        d_prolong_rowp = HostVec<int>(nnodes + 1, h_prolong_rowp).createDeviceVec().getPtr();
        d_prolong_rows = HostVec<int>(P_nnzb, h_prolong_rows).createDeviceVec().getPtr();
        d_prolong_cols = HostVec<int>(P_nnzb, h_prolong_cols).createDeviceVec().getPtr();
        d_prolong_vals = DeviceVec<T>(P_nnzb * block_dim2).getPtr();
        d_Z_vec = DeviceVec<T>(P_nnzb * block_dim2);
        d_Z_vals = d_Z_vec.getPtr();
        d_node_block_map = HostVec<int>(nnodes, h_node_block_map).createDeviceVec().getPtr();
        d_root_block_map =
            HostVec<int>(num_aggregates, h_root_block_map).createDeviceVec().getPtr();
    }

    template <bool startup = true>
    void _compute_diag_vals() {
        // first need to construct rowp and cols for diagonal (fairly easy)

        // startup section
        int ndiag_vals = block_dim * block_dim * nnodes;
        // printf("diag vals part 1: startup\n");
        if constexpr (startup) {
            int *h_diag_rowp = new int[nnodes + 1];
            diag_inv_nnzb = nnodes;
            int *h_diag_cols = new int[nnodes];
            h_diag_rowp[0] = 0;

            for (int i = 0; i < nnodes; i++) {
                h_diag_rowp[i + 1] = i + 1;
                h_diag_cols[i] = i;
            }

            // now copy to device
            d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
            d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

            // create the bsr data object on device
            d_diag_bsr_data = BsrData(nnodes, block_dim, diag_inv_nnzb, d_diag_rowp, d_diag_cols,
                                      nullptr, nullptr, false);
            delete[] h_diag_rowp;
            delete[] h_diag_cols;

            // now allocate DeviceVec for the values
            d_diag_vec = DeviceVec<T>(ndiag_vals);
            d_diag_LU_vals = d_diag_vec.getPtr();  // just copy these pointers..
        }                                          // end of startup
        // CHECK_CUDA(cudaDeviceSynchronize());

        // regular jacobi preconditioner
        // printf("diag vals part 2: copy diag values from kmat\n");
        //  zero previous values (to get new Dinv, in case optimization or nonlinear problem)
        d_diag_vec.zeroValues();  // this is vector for the opinter d_diag_LU_vals (confusing, can
                                  // fix later
        k_copyBlockDiagFromBsrMat<T><<<(ndiag_vals + 31) / 32, 32>>>(
            nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // ilu0 factoriation
        // printf("diag vals part 3: perform ILU0 startup\n");
        if constexpr (startup) {
            // create M matrix object (for full numeric factorization)
            cusparseCreateMatDescr(&descr_M);
            cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseCreateBsrilu02Info(&info_M);

            // init L matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_L);
            cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
            cusparseCreateBsrsv2Info(&info_L);

            // init U matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_U);
            cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
            cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseCreateBsrsv2Info(&info_U);

            // symbolic and numeric factorizations
            CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, &pBufferSize_M));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, &pBufferSize_L));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, &pBufferSize_U));
            pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
            // cudaMalloc((void **)&pBuffer, pBufferSize);
            cudaMalloc(&pBuffer, pBufferSize);

            // perform ILU symbolic factorization on L
            CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, policy_M, pBuffer));
            status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
            if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
                printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
            }

            // analyze sparsity patern of L for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, policy_L, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());

            // analyze sparsity pattern of U for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, policy_U, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("diag vals part 4 : ILU0 numeric factorization\n");
        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                                         info_M, policy_M, pBuffer));
        // CHECK_CUDA(cudaDeviceSynchronize());
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        // startup part of Dinv linear operator
        if constexpr (startup) {
            d_dinv_vec = DeviceVec<T>(ndiag_vals);
            d_dinv_vals = d_dinv_vec.getPtr();
        }

        // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
        // to get effective matrix.. need six temp vectors..
        // printf("diag vals part 5: compute Dinv by applying triang solves 6 times\n");
        for (int i = 0; i < block_dim; i++) {
            // set d_temp to ei (one of e1 through e6 per block)
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            dim3 block(32);
            dim3 grid((nnodes + 31) / 32);
            k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

            // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
                pBuffer));  // prob only need U^-1 part for block diag.. TBD

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

            // now copy temp2 into columns of new operator
            dim3 grid2((N + 31) / 32);
            k_setLUinv_operator<T>
                <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vec.getPtr());
        }  // this works!
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    void compute_prolongator_values() {
        // 1) candidate blocks B from supplied rigid body modes
        // rigid_body_modes is assumed shape (nnodes * block_dim * block_dim)
        d_B_vec = DeviceVec<T>(nnodes * block_dim2);
        d_B_vals = d_B_vec.getPtr();
        CHECK_CUDA(cudaMemcpy(d_B_vals, rigid_body_modes.getPtr(), nnodes * block_dim2 * sizeof(T),
                              cudaMemcpyDeviceToDevice));

        _compute_diag_vals<true>();
        compute_matmat_prod_nz_pattern();

        // // DEBUG
        // T *h_B_vals0 = d_B_vec.createHostVec().getPtr();
        // for (int i = 0; i < nnodes; i++) {
        //     printf("B0[node %d]\n", i);
        //     for (int row = 0; row < 6; row++) {
        //         printVec<T>(6, &h_B_vals0[36 * i + 6 * row]);
        //     }
        // }

        // 2) optional Jacobi smoothing of candidate blocks B
        smooth_rigid_body_modes();

        // DEBUG
        // T *h_B_vals = d_B_vec.createHostVec().getPtr();
        // for (int i = 0; i < nnodes; i++) {
        //     printf("B[node %d]\n", i);
        //     for (int row = 0; row < 6; row++) {
        //         printVec<T>(6, &h_B_vals[36 * i + 6 * row]);
        //     }
        // }

        // 3) coarse candidate blocks Bc = B at root nodes
        d_Bc_vec = DeviceVec<T>(num_aggregates * block_dim2);
        d_Bc_vals = d_Bc_vec.getPtr();
        int nBcVals = num_aggregates * block_dim2;
        k_extract_root_modes<T><<<(nBcVals + 31) / 32, 32>>>(num_aggregates, block_dim,
                                                             d_root_nodes, d_B_vals, d_Bc_vals);
        CHECK_CUDA(cudaDeviceSynchronize());

        // T *h_Bc_vals = d_Bc_vec.createHostVec().getPtr();
        // for (int i = 0; i < num_aggregates; i++) {
        //     printf("Bc[iagg %d]\n", i);
        //     for (int row = 0; row < 6; row++) {
        //         printVec<T>(6, &h_Bc_vals[36 * i + 6 * row]);
        //     }
        // }

        // 4) host-side dense prototype for tentative root-node blocks
        auto h_B = DeviceVec<T>(nnodes * block_dim2, d_B_vals).createHostVec().getPtr();
        auto h_Bc = DeviceVec<T>(num_aggregates * block_dim2, d_Bc_vals).createHostVec().getPtr();
        std::vector<T> h_T(nnodes * block_dim2, 0.0);
        compute_rootnode_dense_blocks_host(nnodes, block_dim, h_aggregate_ind, h_C_nodes, h_B, h_Bc,
                                           h_T.data());

        // scatter tentative prolongator to larger nonzero pattern of P (to allow energy-smoothing)
        std::vector<T> h_P(P_nnzb * block_dim2, T(0));

        for (int i = 0; i < nnodes; i++) {
            int jp = h_node_block_map[i];
            if (jp < 0) {
                printf("ERROR: h_node_block_map[%d] invalid in compute_prolongator_values\n", i);
                continue;
            }

            const T *Tsrc = &h_T[i * block_dim2];
            T *Pdst = &h_P[jp * block_dim2];

            for (int k = 0; k < block_dim2; k++) {
                Pdst[k] = Tsrc[k];
            }
        }

        CHECK_CUDA(cudaMemcpy(d_prolong_vals, h_P.data(), P_nnzb * block_dim2 * sizeof(T),
                              cudaMemcpyHostToDevice));

        // 5) enforce exact root injection
        int nRootVals = num_aggregates * block_dim2;
        k_set_identity_blocks<T><<<(nRootVals + 31) / 32, 32>>>(num_aggregates, block_dim,
                                                                d_root_block_map, d_prolong_vals);
        CHECK_CUDA(cudaDeviceSynchronize());

        // DEBUG : printout tentative prolongator
        // T *h_T_vals = DeviceVec<T>(P_nnzb * block_dim2, d_prolong_vals).createHostVec().getPtr();
        // for (int i = 0; i < nnodes; i++) {
        //     for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
        //         int jc = h_prolong_cols[jp];
        //         int j = h_coarse_nodes[jc];
        //         printf("T(i %d, j %d)\n", i, j);
        //         for (int row = 0; row < 6; row++) {
        //             printVec<T>(6, &h_T_vals[36 * jp + 6 * row]);
        //         }
        //     }
        // }

        // 6) optional prolongation smoothing: P <- P - omega D^{-1} A P,
        // but every update is projected so that U * Bc = 0, then re-apply P_CC = I.
        if (prol_nsmooth > 0) {
            d_Z_vec.zeroValues();

            const int ndiag_vals = nnodes * block_dim2;
            auto free_var_vec = DeviceVec<bool>(N);
            free_var_vec.setFullVecToConstValue(true);
            d_free_dof = free_var_vec.getPtr();

            dim3 OP_block(32), OP_grid(nnodes);
            dim3 PKP_block(216), PKP_grid(nnzb_prod);
            dim3 DP_block(216), DP_grid(P_nnzb);
            dim3 add_block(64);

            // printf("INIT T prolong\n");
            // T *h_T_vals =
            //     DeviceVec<T>(P_nnzb * block_dim2, d_prolong_vals).createHostVec().getPtr();
            // for (int i = 0; i < nnodes; i++) {
            //     for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
            //         int jc = h_prolong_cols[jp];
            //         int j = h_coarse_nodes[jc];
            //         printf("T(i %d, j %d)\n", i, j);
            //         for (int row = 0; row < 6; row++) {
            //             printVec<T>(6, &h_T_vals[36 * jp + 6 * row]);
            //         }
            //     }
            // }

            for (int isweep = 0; isweep < prol_nsmooth; isweep++) {
                // Z <- A * P
                d_Z_vec.zeroValues();
                T a = 1.0;
                k_compute_mat_mat_prod<T><<<PKP_grid, PKP_block>>>(
                    nnzb_prod, block_dim, a, d_K_prodBlocks, d_P_prodBlocks, d_Z_prodBlocks,
                    d_kmat_vals, d_prolong_vals, d_Z_vec.getPtr());
                CHECK_CUDA(cudaDeviceSynchronize());

                // printf("Z = A * P\n");
                // T *h_Z0_vals = DeviceVec<T>(P_nnzb * block_dim2,
                // d_Z_vals).createHostVec().getPtr(); for (int i = 0; i < nnodes; i++) {
                //     for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                //         int jc = h_prolong_cols[jp];
                //         int j = h_coarse_nodes[jc];
                //         printf("A*P(i %d, j %d)\n", i, j);
                //         for (int row = 0; row < 6; row++) {
                //             printVec<T>(6, &h_Z0_vals[36 * jp + 6 * row]);
                //         }
                //     }
                // }

                if constexpr (ORTHOG_PROJECTOR) {
                    // Project the update Z so that Z * Bc = 0 row-wise before adding into P.
                    if (isweep == 0) {
                        d_SU_vec = DeviceVec<T>(ndiag_vals);
                        d_SU_vals = d_SU_vec.getPtr();
                        d_UTU_vec = DeviceVec<T>(ndiag_vals);
                        d_UTU_vals = d_UTU_vec.getPtr();
                        d_UTUinv_vec = DeviceVec<T>(ndiag_vals);
                        d_UTUinv_vals = d_UTUinv_vec.getPtr();
                        d_SU_vec.zeroValues();
                        d_UTU_vec.zeroValues();
                        d_UTUinv_vec.zeroValues();
                    }

                    // // compute free variables
                    // auto free_var_vec = DeviceVec<bool>(N);
                    // free_var_vec.setFullVecToConstValue(
                    //     true);  // set all to default true meaning free var
                    // d_free_dof = free_var_vec.getPtr();

                    // SU_i = Z_i * Bc_{agg(i)},   UTU_i = Bc_{agg(i)}^T * Bc_{agg(i)}
                    k_orthog_projector_computeSU<T><<<OP_grid, OP_block>>>(
                        nnodes, block_dim, d_Bc_vals, d_free_dof, d_prolong_rowp, d_prolong_cols,
                        d_Z_vec.getPtr(), d_SU_vals);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // printf("SU vals\n");
                    // T *h_SU_vals =
                    //     DeviceVec<T>(nnodes * block_dim2, d_SU_vals).createHostVec().getPtr();
                    // for (int i = 0; i < nnodes; i++) {
                    //     printf("SU(i %d)\n", i);
                    //     for (int row = 0; row < 6; row++) {
                    //         printVec<T>(6, &h_SU_vals[36 * i + 6 * row]);
                    //     }
                    // }

                    k_orthog_projector_computeUTU<T>
                        <<<OP_grid, OP_block>>>(nnodes, block_dim, d_Bc_vals, d_free_dof,
                                                d_prolong_rowp, d_prolong_cols, d_UTU_vals);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    // printf("UTU vals\n");
                    // T *h_UTU_vals =
                    //     DeviceVec<T>(nnodes * block_dim2, d_UTU_vals).createHostVec().getPtr();
                    // for (int i = 0; i < nnodes; i++) {
                    //     printf("UTU(i %d)\n", i);
                    //     for (int row = 0; row < 6; row++) {
                    //         printVec<T>(6, &h_UTU_vals[36 * i + 6 * row]);
                    //     }
                    // }

                    CUSPARSE::perform_ilu0_factorization(
                        cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer, nnodes,
                        diag_inv_nnzb, block_dim, d_UTU_vals, d_diag_rowp, d_diag_cols, trans_L,
                        trans_U, policy_L, policy_U, dir);
                    CHECK_CUDA(cudaDeviceSynchronize());

                    for (int i = 0; i < block_dim; i++) {
                        cudaMemset(d_temp, 0, N * sizeof(T));
                        dim3 block(32), grid((nnodes + 31) / 32);
                        k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);
                        CHECK_CUDA(cudaDeviceSynchronize());

                        const double alpha = 1.0;
                        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                            cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L,
                            d_UTU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp,
                            d_resid, policy_L, pBuffer));
                        CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                            cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U,
                            d_UTU_vals, d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid,
                            d_temp2, policy_U, pBuffer));
                        CHECK_CUDA(cudaDeviceSynchronize());

                        dim3 grid2((N + 31) / 32);
                        k_setLUinv_operator<T>
                            <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_UTUinv_vals);
                        CHECK_CUDA(cudaDeviceSynchronize());
                    }

                    // Z <- Z - Z*Bc*(Bc^T Bc)^{-1}*Bc^T
                    k_orthog_projector_removeRowSums<T><<<OP_grid, OP_block>>>(
                        nnodes, block_dim, d_Bc_vals, d_free_dof, d_prolong_rowp, d_prolong_cols,
                        d_SU_vals, d_UTUinv_vals, d_Z_vec.getPtr());
                    CHECK_CUDA(cudaDeviceSynchronize());
                }  // end of orthogonal projector

                // Z <- D^{-1} Z
                k_compute_Dinv_P_mmprod<T><<<DP_grid, DP_block>>>(P_nnzb, block_dim, d_dinv_vals,
                                                                  d_prolong_rows, d_Z_vec.getPtr());
                CHECK_CUDA(cudaDeviceSynchronize());

                // P <- P - omega * Z
                T scale = -omegaJac;
                k_add_colored_submat_PFP<T><<<DP_grid, add_block>>>(
                    P_nnzb, block_dim, scale, 0, d_Z_vec.getPtr(), d_prolong_vals);
                CHECK_CUDA(cudaDeviceSynchronize());

                // Always restore exact root-node injection after every update.
                k_set_identity_blocks<T><<<(nRootVals + 31) / 32, 32>>>(
                    num_aggregates, block_dim, d_root_block_map, d_prolong_vals);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }

        k_set_identity_blocks<T><<<(nRootVals + 31) / 32, 32>>>(num_aggregates, block_dim,
                                                                d_root_block_map, d_prolong_vals);
        CHECK_CUDA(cudaDeviceSynchronize());

        // _apply_dirichlet_bcs();
    }

    void smooth_rigid_body_modes() {
        if (rbm_nsmooth <= 0) return;

        DeviceVec<T> d_x_vec(N), d_y_vec(N), d_z_vec(N);
        T *d_x = d_x_vec.getPtr();
        T *d_y = d_y_vec.getPtr();
        T *d_z = d_z_vec.getPtr();

        dim3 block(32);
        dim3 grid_vec((N + 31) / 32);

        // printf("RBM smooth with omega %.4f, nsmooth %d\n", omegaJac, rbm_nsmooth);

        for (int isweep = 0; isweep < rbm_nsmooth; isweep++) {
            for (int icol = 0; icol < block_dim; icol++) {
                // x <- column icol of B
                k_extract_block_column<T>
                    <<<grid_vec, block>>>(nnodes, block_dim, icol, d_B_vals, d_x);
                CHECK_CUDA(cudaDeviceSynchronize());

                // y = A * x
                T a = 1.0, b = 0.0;
                d_y_vec.zeroValues();
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                              kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                              d_kmat_cols, block_dim, d_x, &b, d_y));

                // z = D^{-1} y
                d_z_vec.zeroValues();
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                              diag_inv_nnzb, &a, descrKmat, d_dinv_vals,
                                              d_diag_rowp, d_diag_cols, block_dim, d_y, &b, d_z));

                // x <- x - omega * z
                T alpha = -omegaJac;
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_z, 1, d_x, 1));

                // write smoothed column back into B
                k_insert_block_column<T>
                    <<<grid_vec, block>>>(nnodes, block_dim, icol, d_x, d_B_vals);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
    }

    void _apply_dirichlet_bcs() {
        int nbcs = d_bcs.getSize();
        if (nbcs == 0) return;
        dim3 block(32), grid((nbcs + 31) / 32);
        apply_mat_bcs_P_kernel<T, DeviceVec>
            <<<grid, block>>>(d_bcs, block_dim, d_prolong_rowp, d_prolong_cols, d_prolong_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void compute_coarse_grid_nz_pattern() {
        // 1) compute P^T nonzero pattern (restriction)
        std::vector<int> prolong_tr_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> prolong_tr_cols;                 // column indices for P

        h_prolong_tr_row_cts = HostVec<int>(num_aggregates).getPtr();
        h_prolong_tr_rowp = HostVec<int>(num_aggregates + 1).getPtr();
        h_prolong_tr_cols = HostVec<int>(P_nnzb).getPtr();

        // printf("coarse_grid_nz 1 - get P^T pattern\n");

        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                h_prolong_tr_row_cts[j]++;
            }
        }

        for (int i = 0; i < num_aggregates; i++) {
            h_prolong_tr_rowp[i + 1] = h_prolong_tr_rowp[i] + h_prolong_tr_row_cts[i];
        }

        // reset to zero
        memset(h_prolong_tr_row_cts, 0, num_aggregates * sizeof(int));
        for (int i = 0; i < nnodes; i++) {
            // loop through cols
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];
                int ip = h_prolong_tr_rowp[j] + h_prolong_tr_row_cts[j];
                h_prolong_tr_cols[ip] = i;
                h_prolong_tr_row_cts[j]++;
            }
        }
        // printf("h_prolong_tr_rowp: ");
        // printVec<int>(num_aggregates + 1, h_prolong_tr_rowp);
        // printf("h_prolong_tr_cols: ");
        // printVec<int>(P_nnzb, h_prolong_tr_cols);

        // also compute map between P and P^T block storage since I don't have that storage
        int *h_prolong_tr_map = new int[P_nnzb];  // PT block input => P block output
        for (int iagg = 0; iagg < num_aggregates; iagg++) {
            for (int jp = h_prolong_tr_rowp[iagg]; jp < h_prolong_tr_rowp[iagg + 1]; jp++) {
                int jnode = h_prolong_tr_cols[jp];

                // now find equivalent block ind in prolong
                for (int kp = h_prolong_rowp[jnode]; kp < h_prolong_rowp[jnode + 1]; kp++) {
                    int kagg = h_prolong_cols[kp];
                    if (kagg == iagg) {
                        h_prolong_tr_map[jp] = kp;
                        // printf("h_prolong_tr_map on (node %d, agg %d) with jp %d => kp %d\n",
                        // jnode,
                        //        iagg, jp, kp);
                        // break;
                    }
                }
            }
        }
        // printf("h_prolong_tr_map: ");
        // printVec<int>(P_nnzb, h_prolong_tr_map);

        // 2) compute A*P nonzero pattern
        // printf("coarse_grid_nz 2 - compute A*P pattern\n");
        std::vector<int> AP_rowp(nnodes + 1, 0);  // row pointer array for P
        std::vector<int> AP_cols;                 // column indices for P
        for (int i = 0; i < nnodes; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // Add the tentative prolongation pattern of row i (usually the "diagonal" entry).
            for (int kp = h_prolong_rowp[i]; kp < h_prolong_rowp[i + 1]; kp++) {
                uniqueIndices.insert(h_prolong_cols[kp]);
            }
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                int j = h_kmat_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity,
                // that is j).
                for (int kp = h_prolong_rowp[j]; kp < h_prolong_rowp[j + 1]; kp++) {
                    uniqueIndices.insert(h_prolong_cols[kp]);
                }
            }

            // The number of entries for row i is the size of the set.
            AP_rowp[i + 1] = AP_rowp[i] + uniqueIndices.size();

            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                AP_cols.push_back(col);
            }
        }

        // 3) compute P^T * (AP) nz pattern now
        // printf("coarse_grid_nz 3 - compute P^T * A * P pattern\n");
        // printf("\tnum agg = %d\n", num_aggregates);
        int num_coarse = num_aggregates;
        std::vector<int> PTAP_rowp(num_coarse + 1, 0);
        std::vector<int> PTAP_cols;
        for (int i = 0; i < num_coarse; i++) {
            // Use a set to gather unique column indices.
            std::set<int> uniqueIndices;
            // For every neighbor j of i (from the kmat data), add j's tentative pattern.
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];
                // For row j in the tentative pattern, add all its entries (for the identity,
                // that is j).
                for (int kp = AP_rowp[j]; kp < AP_rowp[j + 1]; kp++) {
                    uniqueIndices.insert(AP_cols[kp]);
                }
            }

            // The number of entries for row i is the size of the set.
            PTAP_rowp[i + 1] = PTAP_rowp[i] + uniqueIndices.size();

            // Append the sorted (unique) entries to the prolongator's column array.
            // (std::set iterates in sorted order by default.)
            for (int col : uniqueIndices) {
                PTAP_cols.push_back(col);
            }
        }

        PTAP_nnzb = PTAP_rowp[num_coarse] * 1;
        h_PTAP_rowp = HostVec<int>(num_coarse + 1).getPtr();
        h_PTAP_cols = HostVec<int>(PTAP_nnzb).getPtr();
        memcpy(h_PTAP_rowp, PTAP_rowp.data(), (num_coarse + 1) * sizeof(int));
        memcpy(h_PTAP_cols, PTAP_cols.data(), PTAP_nnzb * sizeof(int));

        // now compute LU fillin for direct solve if necessary..
        if (!is_coarse_mg) {
            // printf("MAKING coarse LU pattern for direct solve\n");
            auto c_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, h_PTAP_rowp,
                                      h_PTAP_cols, nullptr, nullptr, true);
            c_bsr_data.compute_full_LU_pattern(10.0, false);
            // now get new nnzb, rowp and cols
            delete[] h_PTAP_rowp;
            delete[] h_PTAP_cols;
            PTAP_nnzb = c_bsr_data.nnzb;
            h_PTAP_rowp = c_bsr_data.rowp;
            h_PTAP_cols = c_bsr_data.cols;
        } else {
            // is coarse MG
            auto c_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, h_PTAP_rowp,
                                      h_PTAP_cols, nullptr, nullptr, true);
            c_bsr_data.compute_nofill_pattern();
            // now get new nnzb, rowp and cols
            delete[] h_PTAP_rowp;
            delete[] h_PTAP_cols;
            PTAP_nnzb = c_bsr_data.nnzb;
            h_PTAP_rowp = c_bsr_data.rowp;
            h_PTAP_cols = c_bsr_data.cols;
        }

        d_PTAP_rowp = HostVec<int>(num_coarse + 1, h_PTAP_rowp).createDeviceVec().getPtr();
        d_PTAP_cols = HostVec<int>(PTAP_nnzb, h_PTAP_cols).createDeviceVec().getPtr();
        // assign Kc or PTAP coarse grid matrix values
        d_PTAP_vec = DeviceVec<T>(block_dim2 * PTAP_nnzb);
        d_PTAP_vals = d_PTAP_vec.getPtr();
        d_PTAP_free_vec = DeviceVec<T>(block_dim2 * PTAP_nnzb);
        d_PTAP_free_vals = d_PTAP_free_vec.getPtr();

        // 4) compute nonzero product block pattern..
        // printf("coarse_grid_nz 4 - compute P^T * A * P 6x6 block triple-mat prod patterns\n");
        PTAP_nnzb_prod = 0;
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k + 1]; lp++) {
                        int l = h_prolong_cols[lp];
                        PTAP_nnzb_prod++;
                    }
                }
            }
        }
        // printf("\tPTAP_nnzb_prod = %d\n", PTAP_nnzb_prod);
        h_PTAP_Kc_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P1_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();
        h_PTAP_P2_blocks = HostVec<int>(PTAP_nnzb_prod).getPtr();

        int inzb_prod = 0;
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_prolong_tr_rowp[i]; jp < h_prolong_tr_rowp[i + 1]; jp++) {
                int j = h_prolong_tr_cols[jp];

                for (int kp = h_kmat_rowp[j]; kp < h_kmat_rowp[j + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int lp = h_prolong_rowp[k]; lp < h_prolong_rowp[k + 1]; lp++) {
                        int l = h_prolong_cols[lp];

                        // find block entry in PTAP matrix
                        int _mp = -1;
                        for (int mp = h_PTAP_rowp[i]; mp < h_PTAP_rowp[i + 1]; mp++) {
                            int m = h_PTAP_cols[mp];
                            if (m == l) {
                                _mp = mp;
                            }
                        }
                        if (_mp < 0) {
                            printf(
                                "BAD PTAP MAP: coarse row %d missing coarse col %d (j=%d k=%d "
                                "lp=%d kp=%d jp=%d)\n",
                                i, l, j, k, lp, kp, jp);
                            fflush(stdout);
                            abort();
                        }
                        int jp_untr =
                            h_prolong_tr_map[jp];  // P^T to P storage since we don't store P^T
                        h_PTAP_Kc_blocks[inzb_prod] = _mp;      // output Kc
                        h_PTAP_P1_blocks[inzb_prod] = jp_untr;  // transpose P
                        h_PTAP_K_blocks[inzb_prod] = kp;        // K
                        h_PTAP_P2_blocks[inzb_prod] = lp;       // P on right
                        inzb_prod++;
                    }
                }
            }
        }
        // printVec<int>(PTAP_nnzb, h_PTAP_cols);

        // put prod blocks on GPU
        // printf("coarse_grid_nz 5 - allocate block prod patterns on GPU\n");
        d_PTAP_Kc_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_Kc_blocks).createDeviceVec().getPtr();
        d_PTAP_P1_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_P1_blocks).createDeviceVec().getPtr();
        d_PTAP_K_blocks = HostVec<int>(PTAP_nnzb_prod, h_PTAP_K_blocks).createDeviceVec().getPtr();
        d_PTAP_P2_blocks =
            HostVec<int>(PTAP_nnzb_prod, h_PTAP_P2_blocks).createDeviceVec().getPtr();
    }

    // void compute_coarse_grid_nz_pattern_v2() {
    //     /* compute coarse grid NZ pattern using two separate matrix-matrix products this time */

    //     // this one intends to get P^T * A * P triple product blocks for kernel
    //     // but we still call it first as it also computes Ac pattern
    //     compute_coarse_grid_nz_pattern();

    //     // 1) first compute pattern of A * P
    //     std::vector<int> AP_rowp(nnodes + 1, 0);
    //     std::vector<int> AP_cols;
    //     for (int i = 0; i < nnodes; i++) {
    //         std::set<int> unique_cols;

    //         for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
    //             int j = h_kmat_cols[jp];
    //             for (int kp = h_prolong_rowp[j]; kp < h_prolong_rowp[j + 1]; kp++) {
    //                 int k = h_prolong_cols[kp];
    //                 unique_cols.insert(k);
    //             }
    //         }

    //         AP_rowp[i + 1] = AP_rowp[i] + static_cast<int>(unique_cols.size());
    //         for (int col : unique_cols) {
    //             AP_cols.push_back(col);
    //         }
    //     }

    //     AP_nnzb = static_cast<int>(prolong_cols.size());

    //     h_AP_rowp = HostVec<int>(nnodes + 1).getPtr();
    //     h_AP_rows = HostVec<int>(AP_nnzb).getPtr();
    //     h_AP_cols = HostVec<int>(AP_nnzb).getPtr();

    //     memcpy(h_AP_rowp, AP_rowp.data(), (nnodes + 1) * sizeof(int));
    //     memcpy(h_AP_cols, AP_cols.data(), AP_nnzb * sizeof(int));

    //     for (int i = 0; i < nnodes; i++) {
    //         for (int jp = h_AP_rowp[i]; jp < h_AP_rowp[i + 1]; jp++) {
    //             h_AP_rows[jp] = i;
    //         }
    //     }
    //     d_AP_rowp = HostVec<int>(nnodes + 1, h_AP_rowp).createDeviceVec().getPtr();
    //     d_AP_rows = HostVec<int>(AP_nnzb, h_AP_rows).getPtr();
    //     d_AP_cols = HostVec<int>(AP_nnzb, h_AP_cols).getPtr();

    //     // 2) now compute mat-mat product patterns of K*P into KP pattern matrix Z
    //     nnzb_prod = 0;
    //     for (int i = 0; i < nnodes; i++) {
    //         for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
    //             int j = h_prolong_cols[jp];  // (P_F)_{ij} output
    //             // now inner loop k for K_{ik} * P_{kj}
    //             for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
    //                 int k = h_kmat_cols[kp];

    //                 // check P_{kj} nz
    //                 bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add
    //                 K*P
    //                                       // fillin (for better prolong)
    //                 for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
    //                     int j2 = h_prolong_cols[jp2];
    //                     if (j2 == j) {
    //                         nz_Pkj = true;
    //                     }
    //                 }
    //                 if (!nz_Pkj) continue;
    //                 // otherwise, we do have a valid nz product here
    //                 nnzb_prod++;
    //             }
    //         }
    //     }
    //     // printf("nnzb_prod = %d\n", nnzb_prod);
    //     // now allocate the block indices of the product
    //     int *h_PF_blocks = new int[nnzb_prod];
    //     int *h_K_blocks = new int[nnzb_prod];
    //     int *h_P_blocks = new int[nnzb_prod];
    //     memset(h_PF_blocks, 0, nnzb_prod * sizeof(int));
    //     memset(h_K_blocks, 0, nnzb_prod * sizeof(int));
    //     memset(h_P_blocks, 0, nnzb_prod * sizeof(int));
    //     int inz_prod = 0;
    //     for (int i = 0; i < nnodes; i++) {
    //         for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
    //             int j = h_prolong_cols[jp];  // (P_F)_{ij} output
    //             // now inner loop k for K_{ik} * P_{kj}
    //             for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
    //                 int k = h_kmat_cols[kp];

    //                 // check P_{kj} nz
    //                 bool nz_Pkj = false;
    //                 int _jp2 = -1;
    //                 for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
    //                     int j2 = h_prolong_cols[jp2];
    //                     if (j2 == j) {
    //                         nz_Pkj = true;
    //                         _jp2 = jp2;
    //                     }
    //                 }
    //                 if (!nz_Pkj) continue;
    //                 // otherwise, we do have a valid nz product here
    //                 h_PF_blocks[inz_prod] = jp;
    //                 h_K_blocks[inz_prod] = kp;
    //                 h_P_blocks[inz_prod] = _jp2;
    //                 inz_prod++;
    //             }
    //         }
    //     }
    //     // now allocate onto the device
    //     d_Z_prodBlocks = HostVec<int>(nnzb_prod, h_PF_blocks).createDeviceVec().getPtr();
    //     d_K_prodBlocks = HostVec<int>(nnzb_prod, h_K_blocks).createDeviceVec().getPtr();
    //     d_P_prodBlocks = HostVec<int>(nnzb_prod, h_P_blocks).createDeviceVec().getPtr();
    // }

    void compute_matmat_prod_nz_pattern() {
        // get pointers

        nnzb_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add K*P
                                          // fillin (for better prolong)
                    for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
                        int j2 = h_prolong_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    nnzb_prod++;
                }
            }
        }
        // printf("nnzb_prod = %d\n", nnzb_prod);
        // now allocate the block indices of the product
        int *h_PF_blocks = new int[nnzb_prod];
        int *h_K_blocks = new int[nnzb_prod];
        int *h_P_blocks = new int[nnzb_prod];
        memset(h_PF_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_K_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_P_blocks, 0, nnzb_prod * sizeof(int));
        int inz_prod = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = h_prolong_rowp[i]; jp < h_prolong_rowp[i + 1]; jp++) {
                int j = h_prolong_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;
                    int _jp2 = -1;
                    for (int jp2 = h_prolong_rowp[k]; jp2 < h_prolong_rowp[k + 1]; jp2++) {
                        int j2 = h_prolong_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                            _jp2 = jp2;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    h_PF_blocks[inz_prod] = jp;
                    h_K_blocks[inz_prod] = kp;
                    h_P_blocks[inz_prod] = _jp2;
                    inz_prod++;
                }
            }
        }
        // now allocate onto the device
        d_Z_prodBlocks = HostVec<int>(nnzb_prod, h_PF_blocks).createDeviceVec().getPtr();
        d_K_prodBlocks = HostVec<int>(nnzb_prod, h_K_blocks).createDeviceVec().getPtr();
        d_P_prodBlocks = HostVec<int>(nnzb_prod, h_P_blocks).createDeviceVec().getPtr();

        // printf("DEBUG: PF_nnzb = %d, nnzb_prod %d\n", P_nnzb, nnzb_prod);
    }

    void compute_coarse_grid_values() {
        // 1) compute coarse grid Galerkin product Ac = P^T * A * P, and Ac_free = P^T * Afree *
        // Ap
        cudaMemset(d_PTAP_vals, 0.0, PTAP_nnzb * block_dim2 * sizeof(T));
        cudaMemset(d_PTAP_free_vals, 0.0, PTAP_nnzb * block_dim2 * sizeof(T));

        // cudaPointerAttributes attr;

        //         auto check_ptr = [&](const char *name, const void *ptr) {
        //             auto err = cudaPointerGetAttributes(&attr, ptr);
        //             printf("%s ptr=%p  err=%s", name, ptr, cudaGetErrorString(err));
        // #if CUDART_VERSION >= 10000
        //             if (err == cudaSuccess)
        //                 printf(" type=%d\n", (int)attr.type);
        //             else
        //                 printf("\n");
        // #else
        //             if (err == cudaSuccess)
        //                 printf(" memoryType=%d\n", (int)attr.memoryType);
        //             else
        //                 printf("\n");
        // #endif
        //             fflush(stdout);
        //         };

        // check_ptr("d_PTAP_Kc_blocks", d_PTAP_Kc_blocks);
        // check_ptr("d_PTAP_P1_blocks", d_PTAP_P1_blocks);
        // check_ptr("d_PTAP_K_blocks", d_PTAP_K_blocks);
        // check_ptr("d_PTAP_P2_blocks", d_PTAP_P2_blocks);
        // check_ptr("d_prolong_vals", d_prolong_vals);
        // check_ptr("d_kmat_vals", d_kmat_vals);
        // check_ptr("d_PTAP_vals", d_PTAP_vals);

        // cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, 0);
        // printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
        // printf("maxGridSize = %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1],
        //        prop.maxGridSize[2]);
        // printf("sharedMemPerBlock = %zu\n", prop.sharedMemPerBlock);
        // fflush(stdout);

        // _debug_device_prodmap();

        // printf("\tcompute coarse grid Galerkin product with nprod %d\n", PTAP_nnzb_prod);
        // k_compute_PTAP_product6<T><<<PTAP_nnzb_prod, 64>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("compute matmat prod to Ac\n");

        // k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);
        // CHECK_CUDA(cudaDeviceSynchronize());

        // printf("[AMG L%d] PTAP launch: nnodes=%d aggs=%d PTAP_nnzb=%d PTAP_prod=%d\n", level,
        //        nnodes, num_aggregates, PTAP_nnzb, PTAP_nnzb_prod);
        // fflush(stdout);

        // cudaError_t old_err = cudaGetLastError();  // clears stale error
        // printf("[AMG L%d] cleared pre-launch err = %s\n", level, cudaGetErrorString(old_err));
        // fflush(stdout);

        // k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
        //     PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
        //     d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);

        k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
            PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
            d_PTAP_P2_blocks, d_prolong_vals, d_kmat_vals, d_PTAP_vals);

        // auto err = cudaPeekAtLastError();
        // printf("[AMG L%d] PTAP post-launch err = %s\n", level, cudaGetErrorString(err));
        // fflush(stdout);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("[AMG L%d] PTAP sync done\n", level);
        // fflush(stdout);

        // printf("compute matmat prod to Ac_free\n");

        // temp just
        k_compute_PTAP_product6_v2<T><<<PTAP_nnzb_prod, 216>>>(
            PTAP_nnzb_prod, block_dim, d_PTAP_Kc_blocks, d_PTAP_P1_blocks, d_PTAP_K_blocks,
            d_PTAP_P2_blocks, d_prolong_vals, d_kmat_free_vals, d_PTAP_free_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
        // CHECK_CUDA(cudaMemcpy(d_PTAP_free_vals, d_PTAP_vals, PTAP_nnzb * block_dim2 *
        // sizeof(T),
        //                       cudaMemcpyDeviceToDevice));
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tdone with coarse grid free Galerkin product\n");

        // printf("\n\n");
        // printf("CHECK fine grid kmat values\n");
        // T *h_Avals = DeviceVec<T>(kmat_nnzb * 36, d_kmat_vals).createHostVec().getPtr();
        // // printf("tentative prolongator: ");
        // // printVec<T>(36 * P_nnzb, h_prolong_vals);
        // for (int inode = 0; inode < nnodes; inode++) {
        //     for (int jp = h_kmat_rowp[inode]; jp < h_kmat_rowp[inode + 1]; jp++) {
        //         int jnode = h_kmat_cols[jp];
        //         printf("A mat on (node %d, node %d): \n", inode, jnode);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_Avals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }

        // printf("\n\n");
        // printf("CHECK Coarse grid Galerkin values\n");
        // T *h_PTAP_vals = DeviceVec<T>(PTAP_nnzb * 36, d_PTAP_vals).createHostVec().getPtr();
        // // printf("tentative prolongator: ");
        // // printVec<T>(36 * P_nnzb, h_prolong_vals);
        // for (int iagg = 0; iagg < num_aggregates; iagg++) {
        //     for (int jp = h_PTAP_rowp[iagg]; jp < h_PTAP_rowp[iagg + 1]; jp++) {
        //         int jagg = h_PTAP_cols[jp];
        //         printf("PTAP mat on (agg %d, agg %d): \n", iagg, jagg);
        //         for (int irow = 0; irow < 6; irow++) {
        //             printVec<T>(6, &h_PTAP_vals[36 * jp + 6 * irow]);
        //         }
        //     }
        // }

        // get the rows for coarse kmat
        h_PTAP_rows = new int[PTAP_nnzb];
        memset(h_PTAP_rows, 0, PTAP_nnzb * sizeof(int));
        for (int i = 0; i < num_aggregates; i++) {
            for (int jp = h_PTAP_rowp[i]; jp < h_PTAP_rowp[i + 1]; jp++) {
                int j = h_PTAP_cols[jp];
                h_PTAP_rows[jp] = i;
            }
        }
        d_PTAP_rows = HostVec<int>(PTAP_nnzb, h_PTAP_rows).createDeviceVec().getPtr();

        // now make a coarse grid galerkin matrix
        coarse_kmat_bsr_data = BsrData(num_aggregates, block_dim, PTAP_nnzb, d_PTAP_rowp,
                                       d_PTAP_cols, nullptr, nullptr, false);
        coarse_kmat_bsr_data.rows = d_PTAP_rows;
        coarse_kmat = BsrMat<DeviceVec<T>>(coarse_kmat_bsr_data, d_PTAP_vec);
        coarse_free_kmat = BsrMat<DeviceVec<T>>(coarse_kmat_bsr_data, d_PTAP_free_vec);
    }

   public:
    Smoother *smoother = nullptr;
    bool is_coarse_mg = false;
    CoarseMG *coarse_mg = nullptr;
    CoarseDirect *coarse_direct = nullptr;

   private:
    bool print_timing = true;
    // bool print_timing = false;

    using clock_type = std::chrono::steady_clock;

    static double elapsed_sec(const clock_type::time_point &t0, const clock_type::time_point &t1) {
        return std::chrono::duration<double>(t1 - t0).count();
    }

    static void sync_if_needed() { CHECK_CUDA(cudaDeviceSynchronize()); }

    void print_setup_time(const char *name, double tsec) const {
        if (print_timing) {
            printf("\t[RootNodeAMG L%d] %-32s %.6e s\n", level, name, tsec);
        }
    }

   private:
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    BsrMat<DeviceVec<T>> kmat, kmat_free, coarse_kmat, coarse_free_kmat;
    DeviceVec<T> rigid_body_modes, d_rhs_vec, d_inner_soln_vec, d_coarse_rhs_vec, d_coarse_soln_vec,
        d_temp_vec, d_B_vec, d_Bc_vec, d_diag_vec;
    DeviceVec<int> d_bcs;

    cusparseMatDescr_t descrKmat{}, descr_M{}, descr_L{}, descr_U{};
    bsrsv2Info_t info_L{}, info_U{};
    bsrilu02Info_t info_M{};
    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    cusparseStatus_t status{};
    int structural_zero{}, numerical_zero{};
    int pBufferSize_M = 0, pBufferSize_L = 0, pBufferSize_U = 0, pBufferSize = 0;
    void *pBuffer = nullptr;

    int nnodes = 0, N = 0, Nc = 0, kmat_nnzb = 0, block_dim = 0, block_dim2 = 0;
    int P_nnzb = 0, diag_inv_nnzb = 0, num_aggregates = 0, num_coarse_nodes = 0;
    int coarse_node_threshold = 0, level = 0, nsmooth = 1, rbm_nsmooth = 1, prol_nsmooth = 1;
    T sparse_threshold = 0, omegaJac = 0;

    T *d_kmat_vals = nullptr, *d_kmat_free_vals = nullptr, *d_temp = nullptr;
    T *d_rhs = nullptr, *d_inner_soln = nullptr, *d_coarse_rhs = nullptr, *d_coarse_soln = nullptr;
    T *d_diag_norms = nullptr, *d_diag_LU_vals = nullptr, *d_B_vals = nullptr, *d_Bc_vals = nullptr;
    bool *d_strength_indicator = nullptr;
    int *d_kmat_rowp = nullptr, *d_kmat_rows = nullptr, *d_kmat_cols = nullptr,
        *d_kmat_diagp = nullptr;
    int *d_prolong_rowp = nullptr, *d_prolong_rows = nullptr, *d_prolong_cols = nullptr;
    int *d_aggregate_ind = nullptr, *d_root_nodes = nullptr, *d_node_block_map = nullptr,
        *d_root_block_map = nullptr;
    int *d_diag_rowp = nullptr, *d_diag_cols = nullptr;
    T *d_prolong_vals = nullptr, *d_Z_vals = nullptr;
    DeviceVec<T> d_Z_vec;

    int *h_kmat_rowp = nullptr, *h_kmat_cols = nullptr;
    bool *h_strength_indicator = nullptr;
    int *h_strength_rowp = nullptr, *h_strength_cols = nullptr;
    int *h_strength_tr_rowp = nullptr, *h_strength_tr_cols = nullptr;
    bool *h_C_nodes = nullptr, *h_F_nodes = nullptr;
    int *h_coarse_id = nullptr, *h_coarse_nodes = nullptr;
    int *h_aggregate_ind = nullptr;
    int *h_prolong_rowp = nullptr, *h_prolong_rows = nullptr, *h_prolong_cols = nullptr;
    int *h_node_block_map = nullptr, *h_root_block_map = nullptr;

    // coarse grid galerkin product
    int *h_prolong_tr_row_cts, *h_prolong_tr_rowp, *h_prolong_tr_cols;
    int PTAP_nnzb;
    int *h_PTAP_rowp, *h_PTAP_cols;
    int *d_PTAP_rowp, *d_PTAP_cols;
    int PTAP_nnzb_prod;
    int *h_PTAP_Kc_blocks, *h_PTAP_P1_blocks, *h_PTAP_K_blocks, *h_PTAP_P2_blocks;
    int *d_PTAP_Kc_blocks, *d_PTAP_P1_blocks, *d_PTAP_K_blocks, *d_PTAP_P2_blocks;
    T *d_PTAP_vals, *d_PTAP_free_vals;
    DeviceVec<T> d_PTAP_vec, d_PTAP_free_vec;
    int *h_PTAP_rows, *d_PTAP_rows;

    int *d_P_prodBlocks, *d_K_prodBlocks, *d_Z_prodBlocks;
    int nnzb_prod;

    BsrData d_diag_bsr_data, coarse_kmat_bsr_data;
    bool *d_free_dof;

    std::string coarsening_type;

    DeviceVec<T> d_dinv_vec;
    T *d_dinv_vals;
    DeviceVec<T> d_SU_vec, d_UTU_vec, d_UTUinv_vec;
    T *d_SU_vals, *d_UTU_vals, *d_UTUinv_vals;
    T *d_temp2, *d_resid;

    // new pointers for new mat-mat prod
    int AP_nnzb, *h_AP_rowp, *h_AP_cols, *h_AP_rows;
    int *d_AP_rowp, *d_AP_cols, *d_AP_rows;
    int nnzb_left_prod, *d_Z_leftProdBlocks, *d_K_leftProdBlocks, *d_P_leftProdBlocks;
    int nnzb_right_prod, *d_Z_rightProdBlocks, *d_K_rightProdBlocks, *d_P_rightProdBlocks;
};

// }  // namespace rootnode_amg
