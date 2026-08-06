

template <typename T>
void direct_cholesky_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                           bool can_print = false) {
    /* direct Cholesky solve => performs Chol factorization K = LL^T and then triangular solves
        Best for solving with same matrix and multiple rhs vectors such as aeroelastic + linear
       structures. */

    // example in NVIDIA website, https://docs.nvidia.com/cuda/cusparse/
    // with search: For example, suppose A is a real m-by-m matrix, where m=mb*blockDim.
    // The following code solves precondition system M*y = x, where M is the product of Cholesky
    // factorization L and its transpose.

    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // NOTE : need to ensure the rowp, cols and values are of cholesky pattern here
    // that is no cols entries in upper triangular portion and values match only diag + lower too
    // there is a method to call bsr_mat.switch_cholesky_pattern() or TODO : you could make cholesky
    // pattern from start and assemble like that

    if (can_print) {
        printf("begin cusparse direct LU solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    const BsrData &bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    T *d_vals = mat.getPtr();
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();

    // temp vector
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0;
    bsrsv2Info_t info_L = 0, info_LT = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_LT = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_LT = CUSPARSE_OPERATION_TRANSPOSE;
    // const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_ichol0_factorization<T>(handle, descr_L, info_L, info_LT, &pBuffer, mb, nnzb,
                                              block_dim, d_vals, d_rowp, d_cols, trans_L, trans_LT,
                                              policy_L, policy_LT, dir);
    // CUSPARSE::debug_call(mb, nnzb, block_dim, d_rowp, d_cols);

    // Forward solve: L*z = rhs
    T alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_L, d_rhs, d_temp, policy_L,
                                         pBuffer));

    // Backward solve: L^T*soln = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_LT, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_LT, d_temp, d_soln,
                                         policy_LT, pBuffer));

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_LT);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

template <typename T>
void perform_ichol0_factorization(cusparseHandle_t handle, cusparseMatDescr_t &descr_L,
                                  bsrsv2Info_t &info_L, bsrsv2Info_t &info_LT,
                                  void **pBuffer,  // allows return of allocated pointer
                                  int mb, int nnzb, int blockDim, T *vals, const int *rowp,
                                  const int *cols, const cusparseOperation_t trans_L,
                                  const cusparseOperation_t trans_LT,
                                  const cusparseSolvePolicy_t policy_L,
                                  const cusparseSolvePolicy_t policy_LT,
                                  const cusparseDirection_t dir) {
    // performs symbolic and numeric Chol and Incomplete Chol factorizations in cusparse
    // no direct ichol0, it just reuses ILU(0) but using sym matrix properties and only half the
    // data

    // temp objects for the factorization
    cusparseMatDescr_t descr_M = 0;
    bsric02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_LT, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;

    // create M matrix object (for full numeric factorization)
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);  // sym not supported in Dsbsrilu02
    cusparseCreateBsric02Info(&info_M);

    // init L matrix objects (for triangular solve)
    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);  // sym not supported in Dsbsrilu02
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_LT);

    // auto h_rowp2 = DeviceVec<int>(mb + 1, rowp).createHostVec();
    // auto h_cols2 = DeviceVec<int>(nnzb, cols).createHostVec();
    // printf("h_rowp4:");
    // printVec<int>(h_rowp2.getSize(), h_rowp2.getPtr());
    // printf("h_cols4:");
    // printVec<int>(h_cols2.getSize(), h_cols2.getPtr());

    // symbolic and numeric factorizations
    CHECK_CUSPARSE(cusparseDbsric02_bufferSize(handle, dir, mb, nnzb, descr_M, vals, rowp, cols,
                                               blockDim, info_M, &pBufferSize_M));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, vals, rowp,
                                              cols, blockDim, info_L, &pBufferSize_L));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(handle, dir, trans_LT, mb, nnzb, descr_L, vals, rowp,
                                              cols, blockDim, info_LT, &pBufferSize_LT));
    pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_LT));
    cudaMalloc(pBuffer, pBufferSize);

    // Incomplete Cholesky Symbolic Analysis
    CHECK_CUSPARSE(cusparseDbsric02_analysis(handle, dir, mb, nnzb, descr_M, vals, rowp, cols,
                                             blockDim, info_M, policy_M, *pBuffer));
    status = cusparseXbsric02_zeroPivot(handle, info_M, &structural_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("[IChol] A(%d,%d) is missing (structural zero)\n", structural_zero, structural_zero);
    }

    // Analyze for triangular solve with L
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, vals, rowp,
                                            cols, blockDim, info_L, policy_L, *pBuffer));

    // Analyze for triangular solve with LT
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(handle, dir, trans_LT, mb, nnzb, descr_L, vals, rowp,
                                            cols, blockDim, info_LT, policy_LT, *pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Numeric Incomplete Cholesky Factorization
    CHECK_CUSPARSE(cusparseDbsric02_solve(handle, dir, mb, nnzb, descr_M, vals, rowp, cols,
                                          blockDim, info_M, policy_M, *pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    status = cusparseXbsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("[IChol] block L(%d,%d) is not invertible (numerical zero)\n", numerical_zero,
               numerical_zero);
    }
}