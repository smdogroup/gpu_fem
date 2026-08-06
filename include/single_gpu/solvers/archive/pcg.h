
// TODO : need to debug this before I compile with it..int
int _gpu_CG(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m,
            cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matL, Vec d_B, Vec d_X, Vec d_R,
            Vec d_R_aux, Vec d_P, Vec d_T, Vec d_tmp, void *d_bufferMV, int maxIterations,
            double tolerance) {
    // source:
    //
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/cg/cg_example.c

    const double zero = 0.0;
    const double one = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice))
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA,
                                d_X.vec, &one, d_R.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                d_bufferMV))
    //--------------------------------------------------------------------------
    // ### 2 ### R_i_aux = L^-1 L^-T R_i
    size_t bufferSizeL, bufferSizeLT;
    void *d_bufferL, *d_bufferLT;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
    //    (a) L^-1 tmp => R_i_aux    (triangular solver)
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                           matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL))
    CHECK_CUDA(cudaMalloc(&d_bufferL, bufferSizeL))
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                         matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL))
    CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
    CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matL,
                                      d_R.vec, d_tmp.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                      spsvDescrL))

    //    (b) L^-T R_i => tmp    (triangular solver)
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrLT))
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL,
                                           d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT))
    CHECK_CUDA(cudaMalloc(&d_bufferLT, bufferSizeLT))
    CHECK_CUSPARSE(cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL,
                                         d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT))
    CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
    CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL,
                                      d_tmp.vec, d_R_aux.vec, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                      spsvDescrLT))
    //--------------------------------------------------------------------------
    // ### 3 ### P0 = R0_aux
    CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice))
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
    double threshold = tolerance * nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta))
    //--------------------------------------------------------------------------
    // ### 4 ### repeat until convergence based on max iterations and
    //           and relative residual
    for (int i = 0; i < maxIterations; i++) {
        printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
        //     (a) T  = A * P_i
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                    d_P.vec, &zero, d_T.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                    d_bufferMV))
        //     (b) denominator = (T, P_i)
        double denominator;
        CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1, &denominator))
        //     (c) alpha = delta / denominator
        double alpha = delta / denominator;
        PRINT_INFO(delta)
        PRINT_INFO(denominator)
        PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 6 ###  X_i+1 = X_i + alpha * P
        //    (a) X_i+1 = -alpha * T + X_i
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1, d_X.ptr, 1))
        //----------------------------------------------------------------------
        // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
        //    (a) R_i+1 = -alpha * T + R_i
        double minus_alpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1, d_R.ptr, 1))
        //----------------------------------------------------------------------
        // ### 8 ###  check ||R_i+1|| < threshold
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
        PRINT_INFO(nrm_R)
        if (nrm_R < threshold) break;
        //----------------------------------------------------------------------
        // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
        //    (a) L^-1 R_i+1 => tmp    (triangular solver)
        CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
        CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                          matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))
        //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matL,
                                          d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT))
        //----------------------------------------------------------------------
        // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
        //    (a) delta_new => (R_i+1, R_aux_i+1)
        double delta_new;
        CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta_new))
        //    (b) beta => delta_new / delta
        double beta = delta_new / delta;
        PRINT_INFO(delta_new)
        PRINT_INFO(beta)
        delta = delta_new;
        //----------------------------------------------------------------------
        // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
        //    (a) P = beta * P
        CHECK_CUBLAS(cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1))
        //    (b) P = R_aux + P
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &one, d_R_aux.ptr, 1, d_P.ptr, 1))
    }
    //--------------------------------------------------------------------------
    printf("Check Solution\n");  // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice))
    // R = -A * X + R
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA,
                                d_X.vec, &one, d_R.vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                d_bufferMV))
    // check ||R||
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrL))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrLT))
    CHECK_CUDA(cudaFree(d_bufferL))
    CHECK_CUDA(cudaFree(d_bufferLT))
    return EXIT_SUCCESS;
}

// template <typename T>
// void iterative_PCG_solve(BsrMat<DeviceVec<T>> mat, DeviceVec<T> rhs,
//                          DeviceVec<T> soln, int maxIterations = 10000,
//                          double tol = 1e-12) {

//     BsrData bsr_data = mat.getBsrData();
//     int mb = bsr_data.nnodes;
//     int nnzb = bsr_data.nnzb;
//     int blockDim = bsr_data.block_dim;
//     int *d_rowPtr = bsr_data.rowPtr;
//     int *d_colPtr = bsr_data.colPtr;
//     int m = mb * blockDim;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();

//     // make the kitchen sink of temporary vectors
//     DeviceVec<T> resid = DeviceVec<T>(soln.getSize());
//     T *d_resid = resid.getPtr();

//     DeviceVec<T> resid_aux = DeviceVec<T>(soln.getSize());
//     T *d_resid_aux = resid_aux.getPtr();

//     DeviceVec<T> p_vec = DeviceVec<T>(soln.getSize());
//     T *d_P = p_vec.getPtr();

//     DeviceVec<T> T_vec = DeviceVec<T>(soln.getSize());
//     T *d_T = T_vec.getPtr();

//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     // note this changes the mat data to be LU (but that's the whole
//     point
//     // of LU solve is for repeated linear solves we now just do
//     triangular
//     // solves)
//     T *d_values = mat.getPtr();

//     // cuSPARSE handle and matrix descriptor
//     cusparseHandle_t cusparseHandle;
//     cusparseMatDescr_t descr;
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
//     cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//     cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

//     cublasHandle_t cublasHandle = NULL;
//     CHECK_CUBLAS(cublasCreate(&cublasHandle));

//     // Setup ILU preconditioner
//     bsrilu02Info_t iluInfo;
//     CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&iluInfo));

//     cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
//     void *buffer;
//     size_t bufferSize;
//     CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
//         cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr,
//         d_values, d_rowPtr, d_colPtr, blockDim, iluInfo, &bufferSize));
//     CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

//     // Analyze and compute ILU
//     CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
//         cusparseHandle, CUSPARSE_DIRECTION_ROW, mb, nnzb, descr,
//         d_values, d_rowPtr, d_colPtr, blockDim, iluInfo, policy,
//         buffer));

//     CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle,
//     CUSPARSE_DIRECTION_ROW, mb,
//                                      nnzb, descr, d_values, d_rowPtr,
//                                      d_colPtr, blockDim, iluInfo, policy,
//                                      buffer));

//     cusparseIndexBase_t baseIdx = CUSPARSE_INDEX_BASE_ZERO;
//     cusparseSpMatDescr_t matA, matL;
//     int *d_L_rows = d_A_rows;
//     int *d_L_columns = d_A_columns;
//     cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
//     cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

//     cusparseSpMatDescr_t matA;
//     CHECK_CUSPARSE(cusparseCreateBsr(
//         &matA, mb, mb, nnzb, blockDim, blockDim, d_rowPtr, d_colPtr,
//         d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F, CUSPARSE_ORDER_ROW));

//     CHECK_CUSPARSE(cusparseCreateBsr(&matA, m, m, nnz, d_A_rows,
//     d_A_columns,
//                                      d_A_values, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_32I, baseIdx,
//                                      CUDA_R_64F))
//     // L
//     CHECK_CUSPARSE(cusparseCreateBsr(&matL, m, m, nnz, d_L_rows,
//     d_L_columns,
//                                      d_L_values, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_32I, baseIdx,
//                                      CUDA_R_64F))
//     CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL,
//     CUSPARSE_SPMAT_FILL_MODE,
//                                              &fill_lower,
//                                              sizeof(fill_lower)))
//     CHECK_CUSPARSE(cusparseSpMatSetAttribute(
//         matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit,
//         sizeof(diag_non_unit)))

//     // resources,
//     // https://
//     //
//     docs.nvidia.com/cuda/cuda-samples/index.html#cusolversp-linear-solver-%5B/url%5D
//     // github repo #1 for PCG,
//     //
//     https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/cg
//     // github repo #2 for PCG,
//     // https://
//     //
//     github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradientPrecond

//     printf("CG loop:\n");
//     _gpu_CG(cublasHandle, cusparseHandle, m, matA, matL, d_rhs, d_soln,
//     d_resid,
//             d_resid_aux, d_P, d_T, d_temp, d_bufferMV, maxIterations,
//             tolerance);

//     // Clean up
//     CHECK_CUDA(cudaFree(buffer));
//     CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
//     CHECK_CUBLAS(cublasDestroy(cublasHandle));
// }