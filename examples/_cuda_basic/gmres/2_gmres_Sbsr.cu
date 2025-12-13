#include "_gmres_utils.h"
#include "_mat_utils.h"

int main() {
    // double BSR mv routine doesn't work (see archive)
    // so need to use float instead for BSR matrix
    using T = float;

    // case inputs
    // -----------

    constexpr bool test_mult = false;
    constexpr bool check_mat_data = false;
    int N = 16; // 16384
    int n_iter = min(N, 200);
    constexpr bool use_precond = true;
    constexpr bool debug = false;
    int checkpoint = -1;
    T abs_tol = 1e-8, rel_tol = 1e-8;

    // NOTE : starting with BSR matrix of block size 1 (just to demonstrate the correct cusparse methods for BSR)

    // initialize data
    // ---------------
    
    int *csr_rowp, *csr_cols;
    // int M = N;
    T *csr_vals, *rhs, *x;
    int nz = 5 * N - 4 * (int)sqrt((double)N);

    // allocate rowp, cols on host
    csr_rowp = (int*)malloc(sizeof(int) * (N + 1));
    csr_cols = (int*)malloc(sizeof(int) * nz);
    csr_vals = (T*)malloc(sizeof(T) * nz);
    x = (T*)malloc(sizeof(T) * N);
    rhs = (T*)malloc(sizeof(T) * N);

    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
        rhs[i] = 0.0;    
    }

    // initialize data
    genLaplaceCSR<T>(csr_rowp, csr_cols, csr_vals, N, nz, rhs);
    // now rhs is not zero

    // convert to BSR
    int *rowp, *cols, nnzb;
    T *vals;
    int mb = N /2;
    int block_dim = 2;
    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals, &rowp, &cols, &vals, &nnzb);

    // printf("nnzb = %d\n", nnzb);
    // printf("bsr_rowp:");
    // printVec<int>(N/2+1, rowp);
    // printf("bsr_cols:");
    // printVec<int>(nnzb, cols);
    // printf("bsr_vals:");
    // printVec<T>(nnzb * 4, vals);

    // transfer data to the device
    int *d_rowp, *d_cols;
    T *d_vals, *d_x, *d_rhs, *d_vals_ILU0;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N/2+1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, 4 * nnzb * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, 4 * nnzb * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));

    // copy data for the matrix over to device
    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, (N/2+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, 4 * nnzb * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs, N * sizeof(T), cudaMemcpyHostToDevice));

    if constexpr (check_mat_data) {
        // also print out d_rowp, d_cols, d_vals to double check their values on host
        int nr = N/2+1;
        int *h_rowp = new int[nr];
        int *h_cols = new int[nnzb];
        T *h_vals = new T[4*nnzb];

        printf("mb %d, nnzb %d, N %d, block_dim %d\n", mb, nnzb, N, block_dim);
        
        CHECK_CUDA(cudaMemcpy(h_rowp, d_rowp, nr * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_cols, d_cols, nnzb * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_vals, d_vals, (4 * nnzb) * sizeof(T), cudaMemcpyDeviceToHost));
        printf("d_rowp:");
        printVec<int>(nr, h_rowp);
        printf("d_cols:");
        printVec<int>(nnzb, h_cols);
        printf("d_vals:");
        printVec<T>(4 * nnzb, h_vals);
    }

    // create temp vec objects
    // -----------------------
    
    T *d_resid, *d_tmp, *d_w;
    CHECK_CUDA(cudaMalloc((void **)&d_resid, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_tmp, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_w, N * sizeof(T)));

    // create initial cusparse and cublas objects
    // ------------------------------------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // wrap dense vectors into cusparse dense vector objects
    // -----------------------------------------------------

    cusparseDnVecDescr_t vec_rhs, vec_tmp, vec_w;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_rhs, N, d_rhs, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_tmp, N, d_tmp, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w, N, d_w, CUDA_R_32F));
    
    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // first test the matrix-vec product on GPU
    // ----------------------------------------
    
    T a = 1.0, b = 0.0;
    if constexpr (test_mult) {
        assert(4 == N);
        T *my_vec = new T[4];
        for (int i = 0; i < 4; i++) {
            my_vec[i] = i+1;
        }

        T *d_my_vec;
        CHECK_CUDA(cudaMalloc((void **)&d_my_vec, 4 * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_my_vec, my_vec, 4 * sizeof(T), cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t vec_my_vec = NULL;
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_my_vec, N, d_my_vec, CUDA_R_32F));    

        // A * my_vec => tmp1
        // BSR MV here
        CHECK_CUSPARSE(cusparseSbsrmv(
            cusparseHandle, 
            CUSPARSE_DIRECTION_ROW,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            mb, mb, nnzb,
            &a, descrA,
            d_vals, d_rowp, d_cols,
            block_dim,
            d_my_vec,
            &b,
            d_tmp
        ));

        // should get this result in tmp1
        // t1=array([  1,  -3,  -7, -11]) [this works now with float!]

        // copy data back to host x vec (just a test of matmult here)
        CHECK_CUDA(cudaMemcpy(x, d_tmp, N * sizeof(T), cudaMemcpyDeviceToHost));
        if (debug && N == 4) {
            printf("A*my_vec=");
            printVec<T>(4, x);
        }

        return 0;
    }
    
    // create ILU(0) preconditioner
    // ----------------------------

    // copy data over to preconditioner
    CHECK_CUDA(cudaMemcpy(d_vals_ILU0, d_vals, 4 * nnzb * sizeof(T), cudaMemcpyDeviceToDevice));

    cusparseMatDescr_t descr_M = 0, descr_L = 0, descr_U = 0;
    bsrilu02Info_t info_M = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    void *pBuffer = 0;
    int structural_zero, numerical_zero;

    // TODO : change to different solve policy later so parallelizes better
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL, 
        policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL, policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE, trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    printf("here1\n");

    // create M matrix objects (for full numeric fact)
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseCreateBsrilu02Info(&info_M);

    // init L matrix objects (for triangular solve)
    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL); // need general for ilu
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    cusparseCreateBsrsv2Info(&info_L);
    
    // init U matrix objects (for triangular solve)
    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL); // need general for ilu
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseCreateBsrsv2Info(&info_U);

    // symbolic and numeric factorizations for the preconditioner

    if constexpr (use_precond) {
        // create buffers for efficient GPU analysis
        CHECK_CUSPARSE(cusparseSbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M, 
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_M, &pBufferSize_M));
        CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, &pBufferSize_L))
        CHECK_CUSPARSE(cusparseSbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, &pBufferSize_U))
        pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));
        CHECK_CUDA(cudaMalloc((void **)&pBuffer, pBufferSize));

        // printf("here2-1\n");
        // printf("mb %d, nnzb %d, block_dim %d, pBufferSize %d\n", mb, nnzb, block_dim, pBufferSize);
        // T *h_vals_ILU0 = new T[4 * nnzb];
        // CHECK_CUDA(cudaMemcpy(h_vals_ILU0, d_vals_ILU0, 4 * nnzb * sizeof(T), cudaMemcpyDeviceToHost));
        // printf("h_vals_ILU0:");
        // printVec<T>(4 * nnzb, h_vals_ILU0);
        // int *h_rowp = new int[N/2+1];
        // int *h_cols = new int[nnzb];
        // CHECK_CUDA(cudaMemcpy(h_rowp, d_rowp, (N/2 + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(h_cols, d_cols, nnzb * sizeof(int), cudaMemcpyDeviceToHost));
        // printf("h_rowp:");
        // printVec<int>(N/2+1, h_rowp);
        // printf("h_cols:");
        // printVec<int>(nnzb, h_cols);

        // perform ILU symbolic factorization on L
        CHECK_CUSPARSE(cusparseSbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_M, 
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_M, policy_M, pBuffer));
        // printf("here2-1-2\n");
        cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
        if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
            printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
        }

        // analyze sparsity pattern of L for efficient triangular solves
        CHECK_CUSPARSE(cusparseSbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L, 
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, policy_L, pBuffer));

        // analyze sparsity pattern of U for efficient triangular solves
        CHECK_CUSPARSE(cusparseSbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U, 
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, policy_U, pBuffer));

        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseSbsrilu02(cusparseHandle, dir, mb, nnzb, descr_L, 
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_M, policy_M, pBuffer));
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        // printf("pre device synchronize\n");
        CHECK_CUDA(cudaDeviceSynchronize());
    } // end of symbolic and numeric factorizations of the preconditioner (and buffers)    

    // printf("here3\n");

    if constexpr (debug && use_precond) {
        // print A matrix values::
        T *h_A = new T[4 * nnzb];
        T *h_M = new T[4 * nnzb];
        CHECK_CUDA(cudaMemcpy(h_A, d_vals, 4 * nnzb * sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_M, d_vals_ILU0, 4 * nnzb * sizeof(T), cudaMemcpyDeviceToHost));
        
        printf("h_A:");
        // print out data in 
        int ct = 0;
        for (int row = 0; row < N; row++) {
            int brow = row / 2;
            int inner_row = row % 2;
            for (int j = rowp[brow]; j < rowp[brow+1]; j++) {
                for (int inner_col = 0; inner_col < 2; inner_col++) {
                    T val = h_A[4 * j + 2 * inner_row + inner_col];
                    printf("%.9e,", val);
                    ct += 1;
                    if (ct % 4 == 0) printf("\n");
                }
            }
        }
        printf("\n");
        printf("h_M:");
        ct = 0;
        // print out data in 
        for (int row = 0; row < N; row++) {
            int brow = row / 2;
            int inner_row = row % 2;
            for (int j = rowp[brow]; j < rowp[brow+1]; j++) {
                for (int inner_col = 0; inner_col < 2; inner_col++) {
                    T val = h_M[4 * j + 2 * inner_row + inner_col];
                    printf("%.9e,", val);
                    ct += 1;
                    if (ct % 4 == 0) printf("\n");
                }
            }
        }
        printf("\n");
        // printf("h_M:");
        // printVec<T>(4 * nnzb, h_M);

        // test LU solve on each unit vector..
    }

    // GMRES solve now with BSR matrix
    // -------------------------------

    printf("checkpt\n");

    // initialize GMRES data, some on host, some on GPU
    // host GMRES data
    T g[n_iter+1], cs[n_iter], ss[n_iter];
    T H[(n_iter+1)*(n_iter)];

    // GMRES device data
    T *d_Vmat, *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter+1) * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    // use single cusparseDnVecDescr_t of size N and just update it's values occasionally
    cusparseDnVecDescr_t vec_V;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_32F));
    // other strategy is to make pointer array of cusparseDnVecDescr_t vecs
    // update with void *col_ptr = static_cast<void*>(&d_Vmat[k * N]);
    //             cusparseDnVecSetValues(vec_V, col_ptr)

    // GMRES algorithm
    // ----------------------------

    int jj = n_iter - 1;

    // apply precond to rhs if in use
    if constexpr (use_precond) {
        // print part of initial vec_rhs
        int NPRINT = N;
        T *h_rhs = new T[NPRINT];
        if (debug) {
            CHECK_CUDA(cudaMemcpy(h_rhs, d_rhs, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
            printf("b:");
            printVec<T>(NPRINT, h_rhs);
        }

        // zero vec_tmp
        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));
        
        // ILU solve U^-1 L^-1 * b
        // L^-1 * b => tmp
        a = 1.0;
        CHECK_CUSPARSE(cusparseSbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_rhs, d_tmp, policy_L, pBuffer));

        if (debug) {
            CHECK_CUDA(cudaMemcpy(h_rhs, d_tmp, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
            printf("L^-1 * b:");
            printVec<T>(NPRINT, h_rhs);
        }

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseSbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
            d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp, d_rhs, policy_U, pBuffer));

        if (debug) {
            CHECK_CUDA(cudaMemcpy(h_rhs, d_rhs, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
            printf("U^-1 * L^-1 * b:");
            printVec<T>(NPRINT, h_rhs);
        }

        // CHECK_CUDA(cudaMemcpy(h_rhs, d_rhs, NPRINT * sizeof(T), cudaMemcpyDeviceToHost));
        // printf("precond vec_rhs:");
        // printVec<T>(NPRINT, h_rhs);
    }

    // GMRES initial residual
    // assumes here d_X is 0 initially => so r0 = b - Ax = b
    T beta;
    CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_rhs, 1, &beta));
    printf("GMRES init resid = %.9e\n", beta);
    g[0] = beta;

    if (debug && checkpoint == 1) return 0;

    // set v0 = r0 / beta (unit vec)
    a = 1.0 / beta;
    CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_rhs, 1, &d_Vmat[0], 1));

    T *h_v0 = new T[N];
    CHECK_CUDA(cudaMemcpy(h_v0, d_Vmat, N * sizeof(T), cudaMemcpyDeviceToHost));
    // print vec
    if (debug) {
        printf("r0:");
        printVec<T>(4, h_v0);
    }

    // then begin main GMRES iteration loop!
    for (int j = 0; j < n_iter; j++) {
        // zero this vec
        // CHECK_CUDA(cudaMemset(&d_Vmat[j * N], 0.0, N * sizeof(T)));

        // get vj and copy it into the cusparseDnVec_t
        void *vj_col = static_cast<void*>(&d_Vmat[j * N]);
        CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vj_col));

        // w = A * vj + 0 * w
        // BSR matrix multiply here MV
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseSbsrmv(
            cusparseHandle, 
            CUSPARSE_DIRECTION_ROW,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            mb, mb, nnzb,
            &a, descrA,
            d_vals, d_rowp, d_cols,
            block_dim,
            &d_Vmat[j * N],
            &b,
            d_w
        ));

        if (debug && N <= 16) {
            T *h_w = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("w=A*v:");
            printVec<T>(N, h_w);

            // if (j == 0) return 0;
        }

        if constexpr (use_precond) {
            // U^-1 L^-1 * w => w precond solve here
            a = 1.0;
            CHECK_CUSPARSE(cusparseSbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
            // U^-1 * tmp => into rhs
            CHECK_CUSPARSE(cusparseSbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));
        }

        // double check and print the value of 
        if (debug && N <= 16) {
            T *h_w = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("h_w[%d] pre-GS:", j);
            printVec<T>(N, h_w);

            // if (j == 0) return 0;
        }

        if (debug && checkpoint == 2) return 0;

        // now update householder matrix
        for (int i = 0 ; i < j+1; i++) {
            // get vi column
            void *vi_col = static_cast<void*>(&d_Vmat[i * N]);
            CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vi_col));

            T w_vi_dot;
            CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &w_vi_dot));

            // H_ij = vi dot w
            H[n_iter * i + j] = w_vi_dot;

            if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[n_iter * i + j]);
            
            // w -= Hij * vi
            a = -H[n_iter * i + j];
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
        }

        // double check and print the value of 
        if (debug && N <= 16) {
            T *h_w = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
            printf("h_w[%d] post GS:", j);
            printVec<T>(N, h_w);

            // if (j == 0) return 0;
        }

        if (debug && checkpoint == 3) return 0;

        // norm of w
        T nrm_w;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_w, 1, &nrm_w));

        // H_{j+1,j}
        H[n_iter * (j+1) + j] = nrm_w;

        // v_{j+1} column unit vec = w / H_{j+1,j}
        a = 1.0 / H[n_iter * (j+1) + j];
        CHECK_CUBLAS(cublasScopy(cublasHandle, N, d_w, 1, &d_Vmat[(j+1) * N], 1));
        CHECK_CUBLAS(cublasSscal(cublasHandle, N, &a, &d_Vmat[(j+1) * N], 1));

        if (debug && N <= 16) {
            T *h_tmp = new T[N];
            // printf("checkpt3\n");
            CHECK_CUDA(cudaMemcpy(h_tmp, &d_Vmat[(j+1) * N], N * sizeof(T), cudaMemcpyDeviceToHost));
            // printf("next V:");
            // printVec<T>(N, h_tmp);

            // if (j == 0) return 0;
        }

        // then givens rotations to elim householder matrix
        for (int i = 0; i < j; i++) {
            T temp = H[i * n_iter + j];
            H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i+1) + j];
            H[n_iter * (i+1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i+1) + j];
        }

        T Hjj = H[n_iter * j + j], Hj1j = H[n_iter * (j+1) + j];
        cs[j] = Hjj / sqrt(Hjj * Hjj + Hj1j * Hj1j);
        ss[j] = cs[j] * Hj1j / Hjj;

        T g_temp = g[j];
        g[j] *= cs[j];
        g[j+1] = -ss[j] * g_temp;

        // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
        printf("GMRES iter %d : resid %.9e\n", j, abs(g[j+1]));

        if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j+1]);

        H[n_iter * j + j] = cs[j] * H[n_iter * j + j] + ss[j] * H[n_iter * (j+1) + j];
        H[n_iter * (j+1) + j] = 0.0;

        if (abs(g[j+1]) < (abs_tol + beta * rel_tol)) {
            printf("GMRES converged in %d iterations to %.9e resid\n", j+1, g[j+1]);
            jj = j;
            break;
        }

        // should I use givens rotations or nrm_w for convergence? I think givens rotations
        // if (abs(nrm_w) < (abs_tol + beta * rel_tol)) {
        //     printf("GMRES converged in %d iterations to %.9e resid\n", j+1, nrm_w);
        //     jj = j;
        //     break;
        // }

    }

    // now solve Householder triangular system
    // only up to size jj+1 x jj+1 where we exited on iteration jj
    T *Hred = new T[(jj+1) * (jj+1)];
    for (int i = 0; i < jj+1; i++) {
        for (int j = 0; j < jj+1; j++) {
            // in-place transpose to be compatible with column-major cublasDtrsv later on
            Hred[(jj+1) * i + j] = H[n_iter * j + i];

            // Hred[(jj+1) * i + j] = H[n_iter * i + j];
        }
    }

    // now print out Hred
    if (debug) {
        printf("Hred:");
        printVec<T>((jj+1) * (jj+1), Hred);
        printf("gred:");
        printVec<T>((jj+1), g);
    }

    // now copy data from Hred host to device
    T *d_Hred;
    CHECK_CUDA(cudaMalloc(&d_Hred, (jj+1) * (jj+1) * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_Hred, Hred, (jj+1) * (jj+1) * sizeof(T), cudaMemcpyHostToDevice));

    // also create gred vector on the device
    T *d_gred;
    CHECK_CUDA(cudaMalloc(&d_gred, (jj+1) * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_gred, g, (jj+1) * sizeof(T), cudaMemcpyHostToDevice));

    // now solve Householder system H * y = g
    // T *d_y;
    // CHECK_CUDA(cudaMalloc(&d_y, (jj+1) * sizeof(T)));
    CHECK_CUBLAS(cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 
        CUBLAS_DIAG_NON_UNIT, jj+1, d_Hred, jj+1, d_gred, 1));
    // writes g => y inplace

    // now copy back to the host
    T *h_y = new T[jj+1];
    CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj+1) * sizeof(T), cudaMemcpyDeviceToHost));

    if (debug && N <= 16) {
        printf("yred:");
        printVec<T>((jj+1), h_y);
    }

    // now compute the matrix product soln = V * y one column at a time
    // zero solution (d_x is already zero)
    for (int j = 0; j < jj+1; j++) {
        a = h_y[j];
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_x, 1));
    }

    // TODO : now compute the residual again
    // resid = b - A * x
    // resid = -1 * A * vj + 1 * w
    // T float_neg_one = -1.0;
    // CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &float_neg_one, matA,
    //     vec_rhs, &floatone, vec_tmp, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
    //     d_bufferMV));

    // now copy solution back to host
    CHECK_CUDA(cudaMemcpy(x, d_x, N * sizeof(T), cudaMemcpyDeviceToHost));

    // now print solution
    if (N <= 16) {
        printf("GMRES no precond soln:");
        printVec<T>(N, x);
    }

    if (N == 16) {
        // compare against truth from python solver
        T ref[N] = {0.45454545, 0.59469697, 0.59469697, 0.45454545, 0.22348485,
            0.32954545, 0.32954545, 0.22348485, 0.10984848, 0.17045455,
            0.17045455, 0.10984848, 0.04545455, 0.0719697 , 0.0719697 ,
            0.04545455};
        T abs_diff[N], tot_abs_diff = 0.0;
        for (int i = 0; i < N; i++) {
            abs_diff[i] = abs(ref[i] - x[i]);
            tot_abs_diff += abs_diff[i];
        }
        printf("tot diff against truth = %.9e\n", tot_abs_diff);
    }

};