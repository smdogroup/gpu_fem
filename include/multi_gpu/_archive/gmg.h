#pragma once

#include <vector>

#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/multigpu_context.h"

template <typename T, class PARTITION, class ASSEMBLER, class SMOOTHER, class PROLONGATION,
          class COARSE_SOLVER>
class MultiGPUGeometricMultigrid {
   public:
    using VEC = GPUvec<T, PARTITION>;
    using MAT = GPUbsrmat<T, PARTITION>;

    MultiGPUGeometricMultigrid(MultiGPUContext *ctx_, std::vector<ASSEMBLER *> assemblers_,
                               std::vector<MAT *> mats_, std::vector<SMOOTHER *> smoothers_,
                               std::vector<PROLONGATION *> prolongations_,
                               COARSE_SOLVER *coarse_solver_, int MAX_STEPS_ = 500, T rtol_ = 1e-6,
                               T atol_ = 1e-30, bool PRINT_ = false, int print_freq_ = 10,
                               T line_search_min_ = 1e-2, T line_search_max_ = 2.0)
        : ctx(ctx_),
          assemblers(assemblers_),
          mats(mats_),
          smoothers(smoothers_),
          prolongations(prolongations_),
          coarse_solver(coarse_solver_) {
        nlevels = assemblers.size();
        printf("created GMG with nlevels=%d\n", nlevels);
        MAX_STEPS = MAX_STEPS_;
        line_search_min = line_search_min_;
        line_search_max = line_search_max_;
        print_freq = print_freq_;
        PRINT = PRINT_;
        rtol = rtol_, atol = atol_;

        // make vecs on each level
        // printf("create GMG vecs\n");
        for (int level = 0; level < nlevels; level++) {
            // printf("create GMG vecs on level %d\n", level);
            d_defects.push_back(assemblers[level]->createGPUVec());
            d_solns.push_back(assemblers[level]->createGPUVec());
            d_temp.push_back(assemblers[level]->createGPUVec());
            d_temp_defect.push_back(assemblers[level]->createGPUVec());
        }
        // printf("\tdone creating GMG vecs\n");
    }

    void solve(VEC *rhs, VEC *soln) {
        // V-cycle solve here..
        OBSERVED_STEPS = 0;

        // printf("Vcyc solve\n");

        // somehow set fine grid defect from rhs
        rhs->copyTo(d_defects[0]);
        d_solns[0]->zeroAll();
        T init_defect_norm = d_defects[0]->norm();
        T converged_nrm = atol + rtol * init_defect_norm;
        T final_defect_norm = init_defect_norm;
        // V-cycle steps
        for (int STEP = 0; STEP < MAX_STEPS; STEP++) {
            // restrict + pre-smooth from fine to coarse
            for (int level = 0; level < nlevels - 1; level++) {
                // pre-smooth (solve here is equivalent to smoothDefect)
                printf("Vcyc step %d: pre-smooth on level %d\n", STEP, level);
                smoothers[level]->smoothDefect(d_defects[level], d_solns[level]);
                CHECK_CUDA(cudaGetLastError());
                ctx->sync();

                // restrict
                printf("Vcyc step %d: restrict on level %d\n", STEP, level);
                prolongations[level]->restrict_vec(d_defects[level], d_defects[level + 1]);
                assemblers[level + 1]->apply_bcs(d_defects[level + 1]);
                d_solns[level + 1]->zeroAll();  // clear coarse soln
                CHECK_CUDA(cudaGetLastError());
                ctx->sync();
            }

            // coarse solve
            printf("Vcyc step %d: coarse direct solve\n", STEP);
            coarse_solver->solve(d_defects[nlevels - 1], d_solns[nlevels - 1]);

            // prolong + post-smooth back up from coarse to fine
            for (int level = nlevels - 2; level >= 0; level--) {
                printf("Vcyc step %d: prolong line search on level %d\n", STEP, level);
                prolongate_line_search(level);
                CHECK_CUDA(cudaGetLastError());
                ctx->sync();

                printf("Vcyc step %d: post-smooth on level %d\n", STEP, level);
                smoothers[level]->smoothDefect(d_defects[level], d_solns[level]);
                CHECK_CUDA(cudaGetLastError());
                ctx->sync();
            }

            // convergence check
            printf("Vcyc step %d: outer convergence check\n", STEP);
            T defect_nrm = d_defects[0]->norm();
            final_defect_norm = defect_nrm;
            if (STEP % print_freq == 0 && PRINT) {
                printf("V-cycle step %d, ||defect|| = %.3e\n", STEP, defect_nrm);
            }
            if (defect_nrm < converged_nrm) {
                OBSERVED_STEPS = STEP + 1;
                if (PRINT) {
                    printf(
                        "V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                        OBSERVED_STEPS, defect_nrm, init_defect_norm);
                }
                break;
            }
        }

        // copy final solution back to output
        d_solns[0]->copyTo(soln);
    }

    void checkP0(const char *tag) {
        if (prolongations.size() > 0 && prolongations[0]) {
            printf("CHECK P0: %s\n", tag);
            prolongations[0]->debug_check_P(tag);
            CHECK_CUDA(cudaGetLastError());
            ctx->sync();
        }
    }

    void prolongate_line_search(int level) {
        prolongations[level]->prolongate(d_solns[level + 1], d_temp[level]);
        assemblers[level]->apply_bcs(d_temp[level]);

        // line search on d_temp
        // printf("Vcyc step %d: line search on level %d\n", STEP, level);
        mats[level]->mult(d_temp[level], d_temp_defect[level]);
        T dp1 = d_temp[level]->dotProd(d_defects[level]);
        T dp2 = d_temp[level]->dotProd(d_temp_defect[level]);
        T omega = dp1 / dp2;
        // printf("omega = %.4e\n", omega);
        omega = std::max(line_search_min, std::min(line_search_max, omega));
        d_solns[level]->axpy(omega, d_temp[level]);
        d_defects[level]->axpy(-omega, d_temp_defect[level]);
    }

    void free() {
        // TBD
    }

    //    private:
    int nlevels;
    MultiGPUContext *ctx;
    std::vector<ASSEMBLER *> assemblers;
    std::vector<MAT *> mats;
    std::vector<SMOOTHER *> smoothers;
    std::vector<PROLONGATION *> prolongations;
    COARSE_SOLVER *coarse_solver;
    int MAX_STEPS;

    // also need some vectors at each level too?
    std::vector<VEC *> d_defects;
    std::vector<VEC *> d_solns;
    std::vector<VEC *> d_temp;
    std::vector<VEC *> d_temp_defect;

    T line_search_min, line_search_max, rtol, atol;
    int print_freq, OBSERVED_STEPS;
    bool PRINT;
};
