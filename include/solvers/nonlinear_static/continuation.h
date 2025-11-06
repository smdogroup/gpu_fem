// implement Ali's TACS continuation solver from TACS CPU for TACS GPU
// implemented by Sean Engelstad, Nov 4th 2025

template <typename T, class Vec, class Assembler, class InnerSolver>
class NonlinearContinuationSolver {
   public:
    NonlinearContinuationSolver(cublasHandle_t &cublasHandle_, Assembler &assembler_,
                                InnerSolver *inner_solver_, bool use_predictor_ = true)
        : cublasHandle(cublasHandle_) {
        inner_solver = inner_solver_, assembler = assembler_;  //, cublasHandle = cublasHandle_;

        prev_state = assembler.createVarsVec();
        state = assembler.createVarsVec();
        nvars = assembler.get_num_vars();
        use_predictor = use_predictor_;

        // circular storage of predictor history
        if (use_predictor) {
            n_predictor = 4;  // num predictor states to hold
            u_hist = DeviceVec<T>(n_predictor * nvars).getPtr();
            lam_hist = new T[n_predictor];
        }
    }

    // need func to set new RHS sometimes?

    bool solve(Vec &u_inout, T lambda0 = 0.2, T inner_atol = 1e-8) {
        u_inout.copyValuesTo(state);  // totally permuted

        // TBD add energy min restart and other options
        // energy_min_restart

        resetPredictor();

        // set (u,lam) = (0,0) as iniial

        lambda = lambda0;
        T dlambda = lambda0;  // initial step size
        bool fail = true;

        // for (int icont = 0; icont < 5; icont++) {
        for (int icont = 0; icont < 20; icont++) {
            // save u0 to go back to if inner solver fails
            state.copyValuesTo(prev_state);

            printf("cont step %d => lambda %.4e\n", icont, lambda);

            // use nonlinear predictor (only really uses it if >= 3 states currently)
            predictNextState(lambda, state);

            // call inner solver
            bool final_step = lambda == 1.0;
            T inner_rtol = final_step ? 1e-8 : 1e-3;
            // T inner_atol = 1e-8;
            bool inner_conv = inner_solver->solve(lambda, inner_rtol, inner_atol, state);

            // DEBUG
            // ======================
            // if (!inner_conv) {
            //     // prev_state.copyValuesTo(state); // don't reset this
            //     break;
            // }
            // ======================

            // update load factors adaptively and reset state if needed
            if (!inner_conv) {
                resetPredictor();
                prev_state.copyValuesTo(state);  // reset state
                lambda -= dlambda;               // go back to prev lambda
                dlambda *= 0.5;                  // reduce step size
            } else {
                // inner solve passed
                if (lambda == 1.0) {
                    // we succeeded the whole solve, break and exit
                    fail = false;
                    break;
                } else {
                    // otherwise adaptively update step size
                    int inewton_steps = inner_solver->get_num_newton_steps();
                    T Rlam = sqrt(8.0 / inewton_steps);  // increases adaptive step size to hit
                                                         // target of 8 newton steps per inner solve
                    Rlam = std::clamp(Rlam, 0.5, 2.0);   // clips so not huge change suddenly
                    dlambda *= Rlam;

                    // record (u,lam) in predictor states
                    recordForPredictor(lambda, state);
                }
            }

            // clip load step size then update load factor
            dlambda = std::clamp(dlambda, 1e-4, 1.0 - lambda);
            lambda += dlambda;
        }

        // store state to out
        state.copyValuesTo(u_inout);
        if (fail) printf("continuation solver failed to conv\n");
        return fail;
    }

    T get_last_lambda() { return lambda; }

   private:
    void energy_min_restart() {
        /* TODO : restart from previous u0, but new rhs now */
        // TBD implement from Ali's NL solver
    }

    void resetPredictor() {
        if (!use_predictor) return;
        // totally reset predictor states
        i_hist = 0, n_hist = 0;  // gives start and stop of current history (stored circularly),
                                 // stop is inclusive so here is 0
        cudaMemset(u_hist, 0.0, n_predictor * nvars * sizeof(T));
        memset(lam_hist, 0.0, n_predictor * sizeof(T));

        // add (u,lam) = (0,0) as an equillib state
        n_hist = 1;
    }

    void predictNextState(T lambda, Vec &vec) {
        /* compute new u prediction for new lambda with nonlinear lagrange poly interp */
        if (!use_predictor) return;
        vec.zeroValues();

        int nuse = std::min(n_hist, 3);  // seems to work better with this
        // int nuse = n_hist;
        if (nuse < 3) return;  // need at least three points to capture curvature effects!
        // printf("use predictor\n");

        // indices of last nuse points in λ order
        int start = (i_hist + n_hist - nuse + n_predictor) % n_predictor;
        printf("using predictor\n");

        for (int i = 0; i < nuse; i++) {
            int icirc = (start + i) % n_predictor;
            T coeff = 1.0;
            for (int j = 0; j < nuse; j++) {
                int jcirc = (start + j) % n_predictor;
                if (i != j) {
                    coeff *= (lambda - lam_hist[jcirc]) / (lam_hist[icirc] - lam_hist[jcirc]);
                }
            }
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &coeff, &u_hist[icirc * nvars], 1,
                                     vec.getPtr(), 1));
        }
    }

    void recordForPredictor(T lambda, Vec &vec) {
        if (!use_predictor) return;
        int pos = (i_hist + n_hist) % n_predictor;
        CHECK_CUDA(cudaMemcpy(&u_hist[pos * nvars], vec.getPtr(), nvars * sizeof(T),
                              cudaMemcpyDeviceToDevice));
        lam_hist[pos] = lambda;

        if (n_hist < n_predictor)
            n_hist++;
        else
            i_hist = (i_hist + 1) % n_predictor;
    }

    Vec prev_state, state;
    InnerSolver *inner_solver;
    Assembler assembler;
    int nvars;
    bool use_predictor;

    cublasHandle_t &cublasHandle;

    int i_hist, n_hist, n_predictor;
    T *u_hist;
    T *lam_hist;
    T lambda;
};
