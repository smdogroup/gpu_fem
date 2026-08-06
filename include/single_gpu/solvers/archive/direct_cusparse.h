
// void direct_chol_solve(CsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs,
//                        DeviceVec<T> &soln) {
//     cusolverSpHandle_t handle = NULL;
//     cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
//     cudaStream_t stream = NULL;
//     cusparseMatDescr_t descrA = NULL;

//     CsrData csr_data = mat.getCsrData();

//     int rowsA = csr_data.num_global; // number of rows of A
//     int colsA = csr_data.num_global; // number of columns of A
//     int nnzA = csr_data.nnz;         // number of nonzeros of A
//     // int baseA = 0;  // base index in CSR format

//     int *d_rowPtr = csr_data.rowPtr;
//     int *d_colPtr = csr_data.colPtr;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();
//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     // note mat data will change and contain LU
//     // however, this is good, allows for faster repeated linear solves
//     // which is the whole benefit of direct solves
//     T *d_values = mat.getPtr();

//     // CSR(A)

//     double tol = 1.e-12;
//     // can change reordering types
//     // [symrcm (symamd or csrmetisnd)] if reorder is 1 (2, or 3),
//     // int reorder = 0;     // no reordering
//     int reorder = 1;
//     int singularity = 0; // -1 if A is invertible under tol.

//     CHECK_CUSOLVER(cusolverSpCreate(&handle));
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUDA(cudaStreamCreate(&stream));
//     CHECK_CUSOLVER(cusolverSpSetStream(handle, stream));
//     CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, stream));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
//     CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//     CHECK_CUSPARSE((cusparseSetMatIndexBase(descrA,
//     CUSPARSE_INDEX_BASE_ZERO)));

//     // // verify if A has symmetric pattern or not
//     // int issym = 0;
//     // CHECK_CUSOLVER(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
//     //                                        h_csrRowPtrA, h_csrRowPtrA + 1,
//     //                                        h_csrColIndA, &issym));
//     // if (!issym) {
//     //     printf("Error: A has no symmetric pattern, please use LU or QR
//     \n");
//     //     exit(EXIT_FAILURE);
//     // }

//     CHECK_CUSOLVER(cusolverSpDcsrlsvchol(handle, rowsA, nnzA, descrA,
//     d_values,
//                                          d_rowPtr, d_colPtr, d_rhs, tol,
//                                          reorder, d_x, &singularity));
//     CHECK_CUDA(cudaDeviceSynchronize());
//     if (0 <= singularity) {
//         printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
//                singularity, tol);
//     }

//     // Clean slate
//     if (handle) {
//         CHECK_CUSOLVER(cusolverSpDestroy(handle));
//     }
//     if (cusparseHandle) {
//         CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     }
//     if (stream) {
//         CHECK_CUDA(cudaStreamDestroy(stream));
//     }
//     if (descrA) {
//         CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
//     }
//     if (d_csrValA) {
//         CHECK_CUDA(cudaFree(d_csrValA));
//     }
//     if (d_csrRowPtrA) {
//         CHECK_CUDA(cudaFree(d_csrRowPtrA));
//     }
//     if (d_csrColIndA) {
//         CHECK_CUDA(cudaFree(d_csrColIndA));
//     }
//     if (d_x) {
//         CHECK_CUDA(cudaFree(d_x));
//     }
//     if (d_b) {
//         CHECK_CUDA(cudaFree(d_b));
//     }
// }

// void direct_chol_solve(CsrMat<HostVec<T>> &mat, HostVec<T> &rhs,
//                        HostVec<T> &soln) {
//     cusolverSpHandle_t handle = NULL;
//     cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
//     cudaStream_t stream = NULL;
//     cusparseMatDescr_t descrA = NULL;

//     CsrData csr_data = mat.getCsrData();

//     int rowsA = csr_data.num_global; // number of rows of A
//     int colsA = csr_data.num_global; // number of columns of A
//     int nnzA = csr_data.nnz;         // number of nonzeros of A
//     // int baseA = 0;  // base index in CSR format

//     int *h_rowPtr = csr_data.rowPtr;
//     int *h_colPtr = csr_data.colPtr;
//     T *h_rhs = rhs.getPtr();
//     T *h_soln = soln.getPtr();
//     HostVec<T> temp = HostVec<T>(soln.getSize());
//     T *h_temp = temp.getPtr();

//     // note mat data will change and contain LU
//     // however, this is good, allows for faster repeated linear solves
//     // which is the whole benefit of direct solves
//     T *h_values = mat.getPtr();

//     // CSR(A)

//     double tol = 1.e-12;
//     // can change reordering types
//     // [symrcm (symamd or csrmetisnd)] if reorder is 1 (2, or 3),
//     // int reorder = 0;     // no reordering
//     int reorder = 1;
//     int singularity = 0; // -1 if A is invertible under tol.

//     CHECK_CUSOLVER(cusolverSpCreate(&handle));
//     CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
//     CHECK_CUDA(cudaStreamCreate(&stream));
//     CHECK_CUSOLVER(cusolverSpSetStream(handle, stream));
//     CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, stream));
//     CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
//     CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//     CHECK_CUSPARSE((cusparseSetMatIndexBase(descrA,
//     CUSPARSE_INDEX_BASE_ZERO)));

//     // // verify if A has symmetric pattern or not
//     // int issym = 0;
//     // CHECK_CUSOLVER(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
//     //                                        h_csrRowPtrA, h_csrRowPtrA + 1,
//     //                                        h_csrColIndA, &issym));
//     // if (!issym) {
//     //     printf("Error: A has no symmetric pattern, please use LU or QR
//     \n");
//     //     exit(EXIT_FAILURE);
//     // }

//     CHECK_CUSOLVER(cusolverSpDcsrlsvcholHost(
//         handle, rowsA, nnzA, descrA, h_values, h_rowPtr, h_colPtr, h_rhs,
//         tol, reorder, h_soln, &singularity));

//     if (0 <= singularity) {
//         printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
//                singularity, tol);
//     }

//     // Clean slate
//     if (handle) {
//         CHECK_CUSOLVER(cusolverSpDestroy(handle));
//     }
//     if (cusparseHandle) {
//         CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
//     }
//     if (stream) {
//         CHECK_CUDA(cudaStreamDestroy(stream));
//     }
//     if (descrA) {
//         CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
//     }

//     // TODO : delete temp vecs down here
// }

// template <typename T>
// void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs,
//                      DeviceVec<T> &soln) {

//     // copy important inputs for Bsr structure out of BsrMat
//     // TODO : was trying to make some of these const but didn't accept it
//     in
//     // final solve
//     BsrData bsr_data = mat.getBsrData();
//     int mb = bsr_data.nnodes;
//     int nnzb = bsr_data.nnzb;
//     int blockDim = bsr_data.block_dim;
//     int *d_rowPtr = bsr_data.rowPtr;
//     int *d_colPtr = bsr_data.colPtr;
//     T *d_rhs = rhs.getPtr();
//     T *d_soln = soln.getPtr();
//     DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
//     T *d_temp = temp.getPtr();

//     //
//     // https: //
//     //
//     developer.nvidia.com/blog/accelerated-solution-sparse-linear-systems/

//     // copy kmat data vec since gets modified during LU
//     // otherwise we can't compute residual properly K * u - f
//     T *d_values = mat.getVec().copyVec().getPtr();

//     /*
//     Cusparse documentation
//     The function cusparseSpSM_bufferSize() returns the size of the
//     workspace needed by cusparseSpSM_analysis() and cusparseSpSM_solve().
//     The function cusparseSpSM_analysis() performs the analysis phase,
//     while cusparseSpSM_solve() executes the solve phase for a sparse
//     triangular linear system. The opaque data structure spsmDescr is used
//     to share information among all functions. The function
//     cusparseSpSM_updateMatrix() updates spsmDescr with new matrix values.
//     */

//     // Initialize cuSPARSE handle
//     cusparseHandle_t handle;
//     CHECK_CUSPARSE(cusparseCreate(&handle));

//     // Create a cuSPARSE matrix descriptor
// cusparseSpMatDescr_t matA;
// CHECK_CUSPARSE(cusparseCreateBsr(
//     &matA, mb, mb, nnzb, blockDim, blockDim, d_rowPtr, d_colPtr,
//     d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F, CUSPARSE_ORDER_ROW));

//     // TODO : need to manually convert from BSR to CSR myself before
//     doing
//     // this.. temporarily convert to CSR format
//     // cusparseSpMatDescr_t matA_CSR;
//     // CHECK_CUSPARSE(cusparseCreateCsr(&matA_CSR, brows, bcols, nnz,
//     //                                  csrRowOffsets, csrColInd,
//     csrValues,
//     //                                  CUSPARSE_INDEX_32I,
//     CUSPARSE_INDEX_32I,
//     //                                  CUSPARSE_INDEX_BASE_ZERO,
//     CUDA_R_64F));

//     // temporarily convert the matrix to CSR for factorization?
//     // I suppose we could do that here instead of C++ later..
//     // cusparseSpMatDescr_t matA_CSR;

//     // Create a dense matrix descriptor for the right-hand side vector
//     cusparseDnMatDescr_t matB;
//     CHECK_CUSPARSE(cusparseCreateDnMat(&matB, mb, 1, mb, d_rhs,
//     CUDA_R_64F,
//                                        CUSPARSE_ORDER_ROW));

//     // Create a dense matrix descriptor for the result vector
//     cusparseDnMatDescr_t matC;
//     CHECK_CUSPARSE(cusparseCreateDnMat(&matC, mb, 1, mb, d_soln,
//     CUDA_R_64F,
//                                        CUSPARSE_ORDER_ROW));

//     // Create sparse matrix solve descriptor
//     cusparseSpSMDescr_t spsmDescr;
//     CHECK_CUSPARSE(cusparseSpSM_createDescr(&spsmDescr));

//     // Choose algorithm for sparse matrix solve
//     cusparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;

//     // create buffer size for LU factorization
//     size_t bufferSize;
//     double alpha = 1.0;
//     const void *alpha_ptr = reinterpret_cast<const void *>(&alpha);

//     CHECK_CUSPARSE(cusparseSpSM_bufferSize(
//         handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_ptr, matA, matB, matC,
//         CUDA_R_64F, alg, spsmDescr, &bufferSize));

//     // create buffer for sparse matrix solve
//     void *d_buffer;
//     CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

//     // do analysis to get in A in LU format
//     cusparseSpSM_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                           CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_ptr,
//                           matA, matB, matC, CUDA_R_64F, alg, spsmDescr,
//                           d_buffer);

//     CHECK_CUSPARSE(cusparseSpSM_solve(handle,
//     CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                       alpha_ptr, matA, matB, matC,
//                                       CUDA_R_64F, alg, spsmDescr));

//     // CHECK_CUSPARSE(cusparseSpSM_analysis(
//     //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //     CUSPARSE_OPERATION_NON_TRANSPOSE, matA, nullptr, CUDA_R_64F,
//     //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr,
//     //     &bufferSizeSM)); // spSMDescr, &bufferSizeSM) // nullptr,
//     //     &bufferSizeSM)

//     // // Allocate buffer for analysis
//     // void *d_bufferSM;
//     // CHECK_CUDA(cudaMalloc(&d_bufferSM, bufferSizeSM));

//     // // LU analysis step
//     // CHECK_CUSPARSE(cusparseSpSM_analysis(
//     //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //     CUSPARSE_OPERATION_NON_TRANSPOSE, matA, nullptr, CUDA_R_64F,
//     //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr, d_bufferSM));

//     // // Create descriptors for L and U (triangular structure)
//     // cusparseSpMatDescr_t matL, matU;

//     // // Lower triangular matrix (L)
//     // CHECK_CUSPARSE(cusparseCreateBlockedSparseMat(
//     //     &matL, mb, mb, nnzb, d_rowPtr, d_colPtr, d_values, blockDim,
//     //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
//     // CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL,
//     CUSPARSE_SPMAT_TRIANGULAR,
//         // CUSPARSE_SPMAT_TRIANGULAR_LOWER));

//         // // Upper triangular matrix (U)
//         // CHECK_CUSPARSE(cusparseCreateBlockedSparseMat(
//         //     &matU, mb, mb, nnzb, d_rowPtr, d_colPtr, d_values,
//         blockDim,
//         //     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
//         CUDA_R_64F));
//         // CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU,
//         CUSPARSE_SPMAT_TRIANGULAR,
//         // CUSPARSE_SPMAT_TRIANGULAR_UPPER));

//         // // Solution for L*y = f  (y is d_temp, f is d_rhs)
//         // // Solution for U*x = y  (x is d_soln, y is d_rhs)

//         // // Perform LU factorization (in place update of d_values)
//         // CHECK_CUSPARSE(cusparseSpSM_solve(
//         //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         //     CUSPARSE_OPERATION_NON_TRANSPOSE, matL, d_rhs, d_temp,
//         //     CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr,
//         d_bufferSM));

//         // CHECK_CUSPARSE(cusparseSpSM_solve(
//         //     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         //     CUSPARSE_OPERATION_NON_TRANSPOSE, matU, d_temp, d_soln,
//         CUDA_R_64F,
//         //     CUSPARSE_SPSM_ALG_DEFAULT, spSMDescr, d_bufferSM));

//         // Cleanup
//         CHECK_CUSPARSE(cusparseSpSM_destroyDescr(spsmDescr));
//     CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
//     CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
//     CHECK_CUSPARSE(cusparseDestroySpMat(matA));
//     CHECK_CUDA(cudaFree(d_buffer));

//     // Destroy cuSPARSE handle
//     CHECK_CUSPARSE(cusparseDestroy(handle));
// }
