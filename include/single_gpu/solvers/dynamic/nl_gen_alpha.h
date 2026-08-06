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

template <class Mat, class Vec>
using LinearSolver = void (*)(Mat &, Vec &, Vec &, bool);

template <class Assembler>
class NLGAIntegrator {
    // NLGAIntergrator stands for NonLinear Gen-Alpha Integrator
   public:
    // could go back and template this later, we'll see
    using T = double;
    using Vec = DeviceVec<T>;
    using Mat = BsrMat<DeviceVec<T>>;

    NLGAIntegrator(LinearSolver<Mat,Vec> &linear_solve, Assembler &assembler, Mat &mass_mat, Mat &kmat,
                   Vec &_forces, int num_dof, int num_timesteps, double dt, int print_freq_ = 10, 
                   T rel_tol = 1e-8, T abs_tol = 1e-8, int max_newton_steps = 30, bool linear_print = false)
        : ndof(num_dof),
          num_timesteps(num_timesteps),
          dt(dt),
          linear_solve(linear_solve),
          max_newton_steps(max_newton_steps), rel_tol(rel_tol), abs_tol(abs_tol),
          mass_mat(mass_mat), kmat(kmat), linear_print(linear_print),
          assembler(assembler) {
        // intialize the disps, vel and accelerations
        assert(_forces.getSize() == (num_timesteps * ndof));

        // only storing final displacements for each timestep
        // vel and accel are stored in circular buffer (cycles in mod 2 where each one is)
        disp = Vec(num_timesteps * ndof).getPtr();
        vel = Vec(2 * ndof).getPtr();
        accel = Vec(2 * ndof).getPtr();

        // util vectors
        disp_star_vec = Vec(ndof);
        disp_star = disp_star_vec.getPtr();
        update_vec = Vec(ndof);
        update = update_vec.getPtr();
        accel_star_vec = Vec(ndof);
        accel_star = accel_star_vec.getPtr();

        temp_vec = Vec(ndof);
        temp = temp_vec.getPtr();
        rhs_vec = Vec(ndof);
        rhs = rhs_vec.getPtr();
        
        forces = _forces.getPtr();

        // initialize constants for HHT-alpha variant
        alpha_f = 1.0 / 3.0;
        // alpha_f = 0.0;
        alpha_m = 0.0;
        beta = 0.25 * (1.0 - alpha_m + alpha_f) * (1.0 - alpha_m + alpha_f);
        gamma = 0.5 - alpha_m + alpha_f;

        // ovr_alpha is such that disp_star += update and accel_star += ovr_alpha * update
        //     where disp_star is the l.c. of this and next timestep disps (from gen-alpha equations)
        // the final residual equation becomes:
        //     M * accel_star + K(disp_star) * disp_star = F_{ext}
        // OR  (K(disp_star) + ovr_alpha * update) * update = R_{init} [so ovr_alpha shows up as l.c. of matrices]
        ovr_alpha = (1-alpha_m) / 0.5 / (1.0 - alpha_f) / dt / dt / beta;

        printf("alpha_f %.3f, alpha_m %.3f, beta %.3f, gamma %.3f\n", alpha_f, alpha_m, beta,
               gamma);
        printf("ovr_alpha = %.4e\n", ovr_alpha);

        print_freq = print_freq_;

        _initialize();
    }

    void solve(bool can_print = true) {
        /* full primal solve method here */
        disp_star_vec.zeroValues();
        accel_star_vec.zeroValues();

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

        // don't reset guess, just keep guess from last timestep (best soln rn) => 
        // the previous init_guess code caused divergence (so some bug in it)
        // _init_guess(itime); 
        update_vec.zeroValues();

        int ct = 0;
        for (int inewton = 0; inewton < max_newton_steps; inewton++, ct++) {
            // assemble the jacobian again (with new displacements)
            _update_guess(itime);
            _update_jacobian(itime);
            _update_rhs(itime);

            // check convergence
            bool converged = _check_convergence(itime, inewton);
            if (converged) break;

            // linear solve to update delta_accel (change in accel from prev to next timestep)
            update_vec.zeroValues(); 

            // rhs 
            rhs_vec.permuteData(block_dim, perm); // put back in vis order

            linear_solve(A, rhs_vec, update_vec, linear_print);

            rhs_vec.permuteData(block_dim, iperm); // put back in solve order
            update_vec.permuteData(block_dim, iperm); // put back in solve order
        }

        _compute_next_timesteps(itime);

        /* write log output to terminal */
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iterate_time = stop - start;
        if (itime % print_freq == 0 && can_print)
            printf("LGAIntegrator : step %d / %d in %.4e sec and %d newton steps\n", itime + 1, num_timesteps,
                    iterate_time.count(), ct);
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

    void free() {
        /* free all CUDA objects */
        CHECK_CUDA(cudaFree(disp));
        CHECK_CUDA(cudaFree(vel));
        CHECK_CUDA(cudaFree(accel));
        CHECK_CUDA(cudaFree(forces));
        CHECK_CUDA(cudaFree(update));
        CHECK_CUDA(cudaFree(disp_star));
        CHECK_CUDA(cudaFree(accel_star));
        CHECK_CUDA(cudaFree(temp));
        CHECK_CUDA(cudaFree(rhs));
        CHECK_CUDA(cudaFree(Avals));
        cusparseDestroyMatDescr(descr_M);
        cusparseDestroyMatDescr(descr_K);
        cusparseDestroyMatDescr(descr_A);
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
    }

   private:
    void _init_guess(int itime) {
        /* init disp_star and accel_star (for linear case) */
        disp_star_vec.zeroValues();
        accel_star_vec.zeroValues();

        // // circular indices for i (accel of prev timestep) vs f (accel of next timestep)
        // int i = itime % 2;
        // int f = (itime + 1) % 2;

        // // compute d_{n+1} = d_n + v_n * dt + 1/2 * dt^2 * a_n (init guess)
        // CHECK_CUDA(cudaMemcpy(&disp[(itime + 1) * ndof], &disp[itime * ndof], ndof * sizeof(T),
        //                       cudaMemcpyDeviceToDevice));
        // T a = dt;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &vel[i * ndof], 1, &disp[(itime + 1) * ndof], 1));
        // a = 0.5 * dt * dt;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, &disp[(itime + 1) * ndof], 1));
        
        // // now compute disp_star = (1-alpha_f) * d_{n+1} + alpha_f * d_n
        // a = alpha_f;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &disp[itime * ndof], 1, disp_star, 1));
        // a = 1.0 - alpha_f;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &disp[(itime+1) * ndof], 1, disp_star, 1));

        // // now compute a_{n+1} as a_n init guess
        // CHECK_CUDA(
        //     cudaMemcpy(&accel[f * ndof], &accel[i * ndof], ndof * sizeof(T), cudaMemcpyDeviceToDevice));
    
        // // then compute accel_star init guess
        // a = alpha_m;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, accel_star, 1));
        // a = 1.0 - alpha_m;
        // CHECK_CUBLAS(
        //     cublasDaxpy(cublasHandle, ndof, &a, &accel[f * ndof], 1, accel_star, 1));
    }

    void _update_guess(int itime) {
        /* update disp_star, accel_star */
        
        // update applied directly to disp_star (since that's what we iterate on)
        T a = 1.0;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, ndof, &a, update, 1, disp_star, 1));
            

        // update also applied to accel_star since they are lin dependent
        a = ovr_alpha;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, ndof, &a, update, 1, accel_star, 1));
    }

    void _update_jacobian(int itime) {
        /* update nonlinear K matrix and M and then the full jacobian (l.c. of K and M) */

        // change back to vis order temporarily to set variables into assembler
        disp_star_vec.permuteData(block_dim, perm);
        assembler.set_variables(disp_star_vec);
        disp_star_vec.permuteData(block_dim, iperm);

        assembler.add_jacobian(temp_vec, kmat);
        assembler.add_mass_jacobian(temp_vec, mass_mat);
        assembler.apply_bcs(kmat);
        assembler.apply_bcs(mass_mat);

        // now update the A matrix l.c. of M and K for accel updates LHS solve
        // l.c. is A = K(d^*) + M * ovr_alpha
        A.zeroValues();
        T a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, mat_nnz, &a, Kvals, 1, Avals, 1));
        a = ovr_alpha;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, mat_nnz, &a, Mvals, 1, Avals, 1));
    }

    void _update_rhs(int itime) {
        /* compute the initial residual for this current guess */
        // residual eqn is M * accel_star + K(disp_star) * disp_star = F_{ext}

        // first add in F_{ext} in the solve ordering (from vis ordering so perm)
        rhs_vec.zeroValues();
        _apply_loads(itime);
        // permute the loads to the solve permutation order (were unpermuted originally and in vis ordering)
        rhs_vec.permuteData(block_dim, iperm);

        // for debugging
        // T load_norm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, ndof, rhs, 1, &load_norm));
        // printf("\tload norm = %.4e\n", load_norm);

        // then subtract M * accel_star
        T a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descr_M,
                                      Mvals, d_rowp, d_cols, block_dim, accel_star, &b, rhs));

        // then subtract K_{nl} * disp_star
        a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descr_K,
                                      Kvals, d_rowp, d_cols, block_dim, disp_star, &b, rhs));
    }

    void _apply_loads(int itime) {
        // copy loads from this timestep into the rhs (l.c. of forces at the prev and current
        // timestep)
        // approximately F((1-alpha_f) * t_{n+1} + alpha_f * t_n)
        T a = 1 - alpha_f;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &forces[(itime + 1) * ndof], 1, rhs, 1));
        a = alpha_f;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &forces[itime * ndof], 1, rhs, 1));
    }

    bool _check_convergence(int itime, int inewton) {
        /* check the norm of the rhs (the residual for accel update) */
        T resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, ndof, rhs, 1, &resid_norm));

        if (inewton == 0) init_resid_norm = resid_norm;

        printf("\tnewton %d : resid %.4e\n", inewton, resid_norm);

        T ub = abs(init_resid_norm) * rel_tol + abs_tol;
        return resid_norm < ub;
    }

    void _compute_next_timesteps(int itime) {
        /* from final disp_star, accel_star compute next disp, vel, accel */

        // circular indices for i (accel of prev timestep) vs f (accel of next timestep)
        int i = itime % 2;
        int f = (itime + 1) % 2;

        // d_{n+1} = 1/(1-alpha_f) * [disp_star - alpha_f * d_n]
        CHECK_CUDA(cudaMemset(&disp[(itime+1) * ndof], 0.0, ndof * sizeof(T)));
        T a = 1.0 / (1.0 - alpha_f);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, disp_star, 1, &disp[(itime+1) * ndof], 1));
        a *= -alpha_f;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &disp[itime * ndof], 1, &disp[(itime+1) * ndof], 1));

        // a_{n+1} = 1/(1-alpha_m) * [accel_star - alpha_m * a_n]
        CHECK_CUDA(cudaMemset(&accel[f * ndof], 0.0, ndof * sizeof(T)));
        a = 1.0 / (1.0 - alpha_m);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, accel_star, 1, &accel[f * ndof], 1));
        a *= -alpha_m;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, &accel[f * ndof], 1));

        // v_{n+1} = v_n + dt * (1-gamma) * a_n + dt * gamma * a_{n+1}
        CHECK_CUDA(
            cudaMemcpy(&vel[f * ndof], &vel[i * ndof], ndof * sizeof(T), cudaMemcpyDeviceToDevice));
        a = dt * (1 - gamma);
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &accel[i * ndof], 1, &vel[f * ndof], 1));
        a = dt * gamma;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, ndof, &a, &accel[f * ndof], 1, &vel[f * ndof], 1));
    }    

    void _initialize() {
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
        auto Avec = Vec(mat_nnz);
        Avals = Avec.getPtr();

        // create the A matrix for LHS of accel update
        A = BsrMat(bsr_data, Avec);

        // create handles for CUDA - cusparse and cublas
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // create a matrix descriptors for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));  // first for mass matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_K));  // second for stiffness matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_K, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_K, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_A));  // second for stiffness matrix
        CHECK_CUSPARSE(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));
    }

    // timestep and gen-alpha info
    double dt;
    int num_timesteps, print_freq, max_newton_steps;
    T alpha_f, alpha_m, beta, gamma, ovr_alpha;
    T abs_tol, rel_tol;
    T init_resid_norm;
    bool linear_print;

    // general vec data
    int block_dim, mat_nnz, ndof, nnzb, mb;
    int *perm, *iperm;
    int *d_rowp, *d_cols;
    T *disp, *vel, *accel, *forces;
    T *temp, *rhs, *update, *disp_star, *accel_star;
    Vec rhs_vec, temp_vec, update_vec, disp_star_vec, accel_star_vec;

    // matrix data
    T *Mvals, *Kvals, *Avals;
    Mat mass_mat, kmat, A;

    // linear solver
    LinearSolver<Mat,Vec> linear_solve;

    // Assembler
    Assembler assembler;

    // general CUDA data
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

    // CUDA data for M and K matrices for mat-mul and residual estimates
    cusparseMatDescr_t descr_M = 0;  // mass matrix descriptor for SpMV
    cusparseMatDescr_t descr_K = 0;  // stiffness matrix descriptor for SpMV
    cusparseMatDescr_t descr_A = 0;
};

#endif // USE_GPU