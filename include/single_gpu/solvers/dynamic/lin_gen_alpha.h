#pragma once
/*
CUDA BSR double-precision solver for Linear elastic unsteady solve (with Gen-alpha method)

Generalized-alpha method for structural dynamics solutions
    "A Time Integration Algorithm for Structural Dynamics With Improved Numerical Dissipation: The
Generalized-α Method" by J. Chung, G.M. Hulbert 1993
    https://asmedigitalcollection.asme.org/appliedmechanics/article/60/2/371/423023/A-Time-Integration-Algorithm-for-Structural

Choosing to use the HHT-alpha variant of gen-alpha such that:
    alpha_f = 1/3 (from choice rho_infty = 1/2)
    alpha_m = 0
    beta = 0.25 * (1 - alpha_m + alpha_f)^2 = 4/9
    gamma = 1/2 - alpha_m + alpha_f = 2/3
*/

#include <chrono>

#include "linalg/bsr_mat.h"
#include "linalg/vec.h"
#include "mesh/vtk_writer.h"

#ifdef USE_GPU

class LGAIntegrator {
    // LGAIntergrator stands for Linear Gen-Alpha Integrator
   public:
    // could go back and template this later, we'll see
    using T = double;
    using Vec = DeviceVec<T>;
    using Mat = BsrMat<DeviceVec<T>>;

    LGAIntegrator(Mat &mass_mat, Mat &kmat, Vec &_forces, int num_dof, int num_timesteps, double dt,
                  int print_freq_ = 10)
        : ndof(num_dof), num_timesteps(num_timesteps), dt(dt) {
        // intialize the disps, vel and accelerations
        assert(_forces.getSize() == (num_timesteps * ndof));

        // only storing final displacements for each timestep
        // vel and accel are stored in circular buffer (cycles in mod 2 where each one is)
        disp = Vec(num_timesteps * ndof).getPtr();
        vel = Vec(2 * ndof).getPtr();
        accel = Vec(2 * ndof).getPtr();

        // util vectors
        update = Vec(ndof).getPtr();
        temp_vec = Vec(ndof);
        temp = temp_vec.getPtr();
        rhs_vec = Vec(ndof);
        rhs = rhs_vec.getPtr();
        res_vec = Vec(ndof);

        forces = _forces.getPtr();

        // initialize constants for HHT-alpha variant
        alpha_f = 1.0 / 3.0;
        // alpha_f = 0.0;
        alpha_m = 0.0;
        beta = 0.25 * (1.0 - alpha_m + alpha_f) * (1.0 - alpha_m + alpha_f);
        gamma = 0.5 - alpha_m + alpha_f;

        printf("alpha_f %.3f, alpha_m %.3f, beta %.3f, gamma %.3f\n", alpha_f, alpha_m, beta,
               gamma);

        print_freq = print_freq_;

        _initialize_LU(mass_mat, kmat);
    }

    void solve(bool can_print = true) {
        /* full primal solve method here */
        // iterate one less time than the number of timesteps (since )
        for (int itime = 0; itime < num_timesteps - 1; itime++) {
            iterate(itime, can_print);
        }
        printf("done with solve\n");
    }

    void iterate(int itime, bool can_print = true) {
        /* first compute the residual as the rhs for new update */
        auto start = std::chrono::high_resolution_clock::now();

        if (itime >= (num_timesteps - 1)) {
            printf(
                "time step %d past the max number of timesteps %d was input into the "
                "LinearGenAlphaIntegrator\n",
                itime, num_timesteps);
            return;
        }

        // copy loads from this timestep into the rhs (l.c. of forces at the prev and current
        // timestep)
        rhs_vec.zeroValues();
        // approximately F((1-alpha_f) * t_{n+1} + alpha_f * t_n)
        T a = 1 - alpha_f;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &forces[(itime + 1) * ndof], 1, rhs, 1));
        a = alpha_f;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &forces[itime * ndof], 1, rhs, 1));
        CHECK_CUDA(
            cudaMemcpy(rhs, &forces[ndof * itime], ndof * sizeof(T), cudaMemcpyDeviceToDevice));

        // permute the loads at this timestep
        rhs_vec.permuteData(block_dim, iperm);

        // circular indices for i (accel of prev timestep) vs f (accel of next timestep)
        int i = itime % 2;
        int f = (itime + 1) % 2;

        // print rhs vec
        // T *h_temp = new T[ndof];
        // CHECK_CUDA(cudaMemcpy(h_temp, rhs, ndof * sizeof(T), cudaMemcpyDeviceToHost));
        // // printVec<T>(54, h_temp);
        // for (int i = 0; i < ndof; i++) {
        //     if (abs(h_temp[i]) > 0.0) printf("h_temp[%d] nz with val %.4e\n", i, h_temp[i]);
        // }

        // add in disp, vec, accel terms (in same permuted order as matrix, will have to perm later
        // for visualization)

        // 1) subtract in alpha_m * M * a_n
        a = -alpha_m;
        T b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descr_M,
                                      Mvals, d_rowp, d_cols, block_dim, &accel[ndof * i], &b, rhs));

        // 2) subtract in (1-alpha_f) * (1/2 - beta) * dt^2 * K * an
        a = -1.0 * (1.0 - alpha_f) * (0.5 - beta) * dt * dt, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descr_K,
                                      Kvals, d_rowp, d_cols, block_dim, &accel[ndof * i], &b, rhs));

        // 3) subtract in (1-alpha_f) * dt * K * vn
        a = -1.0 * (1.0 - alpha_f) * dt, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descr_K,
                                      Kvals, d_rowp, d_cols, block_dim, &vel[ndof * i], &b, rhs));

        // 4) subtract in K * disp_n
        a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
            &a, descr_K, Kvals, d_rowp, d_cols, block_dim, &disp[ndof * itime], &b, rhs));

        /* then compute the new accel with the LU solve */
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             LUvals, d_rowp, d_cols, block_dim, info_L, rhs, temp,
                                             policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             LUvals, d_rowp, d_cols, block_dim, info_U, temp,
                                             &accel[f * ndof], policy_U, pBuffer));

        /* now compute vel updates */
        // v_{n+1} = v_n + dt * (1-gamma) * a_n + dt * gamma * a_{n+1}
        CHECK_CUDA(
            cudaMemcpy(&vel[f * ndof], &vel[i * ndof], ndof * sizeof(T), cudaMemcpyDeviceToDevice));
        a = dt * (1 - gamma);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, &vel[f * ndof], 1));
        a = dt * gamma;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &accel[f * ndof], 1, &vel[f * ndof], 1));

        /* and update the disps
              (it is still inv permuted, not ready for disp here.. see later) */
        // d_{n+1} = d_n + dt * v_n + dt^2 * (1/2-beta) * a_n + dt^2 * beta * a_{n+1}
        CHECK_CUDA(cudaMemcpy(&disp[(itime + 1) * ndof], &disp[itime * ndof], ndof * sizeof(T),
                              cudaMemcpyDeviceToDevice));
        a = dt;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, ndof, &a, &vel[i * ndof], 1, &disp[(itime + 1) * ndof], 1));
        a = dt * dt * (0.5 - beta);
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, &disp[(itime + 1) * ndof], 1));
        a = dt * dt * beta;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, ndof, &a, &accel[f * ndof], 1, &disp[(itime + 1) * ndof], 1));

        /* write log output to terminal */
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iterate_time = stop - start;
        if (itime % print_freq == 0 && can_print)
            printf("LGAIntegrator : step %d / %d in %.4e sec\n", itime + 1, num_timesteps,
                   iterate_time.count());
    }

    std::string _getTimeString(int itime) {
        if (itime < 10) {
            return "00" + std::to_string(itime);
        } else if (itime < 100) {
            return "0" + std::to_string(itime);
        } else {
            return std::to_string(itime);
        }
    }

    template <class Assembler>
    void writeToVTK(Assembler &assembler, std::string base_filename, int stride = 1) {
        printf("here1\n");
        auto h_temp = temp_vec.createHostVec();
        printf("begin write to vtk\n");
        int ct = 0;
        for (int itime = 0; itime < num_timesteps; itime += stride, ct++) {
            // permute the disps so in visualization order (not solve ordering)
            CHECK_CUDA(
                cudaMemcpy(temp, &disp[itime * ndof], ndof * sizeof(T), cudaMemcpyDeviceToDevice));
            temp_vec.permuteData(block_dim, perm);
            temp_vec.copyValuesTo(h_temp);  // copy to host

            // then print to VTK with vis order disps in temp vec
            std::string time_string = _getTimeString(ct);
            std::string filename = base_filename + "_" + time_string + ".vtk";
            printToVTK<Assembler, HostVec<T>>(assembler, h_temp, filename);
        }
    }

    void free() { /* free all CUDA objects */
        CHECK_CUDA(cudaFree(pBuffer));
        CHECK_CUDA(cudaFree(disp));
        CHECK_CUDA(cudaFree(vel));
        CHECK_CUDA(cudaFree(accel));
        CHECK_CUDA(cudaFree(forces));
        CHECK_CUDA(cudaFree(update));
        CHECK_CUDA(cudaFree(temp));
        CHECK_CUDA(cudaFree(rhs));
        CHECK_CUDA(cudaFree(LUvals));
        cusparseDestroyMatDescr(descr_L);
        cusparseDestroyMatDescr(descr_U);
        cusparseDestroyMatDescr(descr_M);
        cusparseDestroyMatDescr(descr_K);
        cusparseDestroyBsrsv2Info(info_L);
        cusparseDestroyBsrsv2Info(info_U);
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
    }

   private:
    void _initialize_LU(Mat &mass_mat, Mat &kmat) {
        // initialize CUDA LU data so we can hold the LU factor in place
        // assumes kmat and mass_mat have full LU sparsity

        // get perm, iperm, block_data out of the matrix
        perm = mass_mat.getPerm();
        iperm = mass_mat.getIPerm();
        block_dim = mass_mat.getBlockDim();
        mat_nnz = mass_mat.get_nnz();
        BsrData bsr_data = mass_mat.getBsrData();
        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;

        // copy device pointers out of the matrices
        Mvals = mass_mat.getPtr();
        Kvals = kmat.getPtr();

        // create handles for CUDA - cusparse and cublas
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // compute the l.c. of M and K matrix for new accel updates
        // TBD : later include C matrix for damping?
        // l.c. is MK = (1-alpha_m) * M + (1-alpha_f) * dt^2 * beta
        LUvals = Vec(mat_nnz).getPtr();
        double a = 1 - alpha_m;  // first we add (1-alpha_m) * M
        printf("a %.4e for adding Mmat to LUvals\n", a);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, mat_nnz, &a, Mvals, 1, LUvals, 1));
        a = 0.5 * (1 - alpha_f) * dt * dt * beta;  // first we add (1-alpha_f) * dt * dt * beta * K
        printf("a %.4e for adding Kmat to LUvals\n", a);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, mat_nnz, &a, Kvals, 1, LUvals, 1));

        // compute the LU factor of the MK matrix stored in LUvals
        CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                             &pBuffer, mb, nnzb, block_dim, LUvals, d_rowp, d_cols,
                                             trans_L, trans_U, policy_L, policy_U, dir);

        // create a matrix descriptors for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));  // first for mass matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_K));  // second for stiffness matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_K, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_K, CUSPARSE_INDEX_BASE_ZERO));

        printf("done with LU init part\n");

        // initial acceleration a_0 = M^-1 * (F(0) - C*v - K*d) but C = 0 here
        a = 1.0;
        CHECK_CUDA(cudaMemcpy(rhs, forces, ndof * sizeof(T), cudaMemcpyDeviceToDevice));
        rhs_vec.permuteData(block_dim, iperm);
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             LUvals, d_rowp, d_cols, block_dim, info_L, rhs, temp,
                                             policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             LUvals, d_rowp, d_cols, block_dim, info_U, temp, accel,
                                             policy_U, pBuffer));
    }

    // timestep and gen-alpha info
    double dt;
    int num_timesteps, print_freq;
    T alpha_f, alpha_m, beta, gamma;

    // general vec data
    int block_dim, mat_nnz, ndof, nnzb, mb;
    int *perm, *iperm;
    int *d_rowp, *d_cols;
    T *disp, *vel, *accel, *forces;
    T *update, *temp, *rhs;
    Vec rhs_vec, temp_vec, res_vec;

    // matrix data
    T *Mvals, *Kvals, *LUvals;

    // general CUDA data
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

    // CUDA data for LU factor of (alpha * M + beta * K)^-1 approx U^-1 L^-1
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // CUDA data for M and K matrices for mat-mul and residual estimates
    cusparseMatDescr_t descr_M = 0;  // mass matrix descriptor for SpMV
    cusparseMatDescr_t descr_K = 0;  // stiffness matrix descriptor for SpMV
};

#endif