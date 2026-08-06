#pragma once
#include "multigrid/solvers/solve_utils.h"

template <typename T, class DOMDEC, class KRYLOV>
class DomDecKrylovWrapper : public BaseSolver {
   public:
    DomDecKrylovWrapper(DOMDEC *fetidp_, KRYLOV *krylov_) : fetidp(fetidp_), krylov(krylov_) {
        nlam = fetidp->getLambdaSize();
        N = krylov->get_num_dof();

        // initialize the vecs
        lam_rhs = DeviceVec<T>(nlam);
        lam_soln = DeviceVec<T>(nlam);

        auto bsr_data = fetidp->kmat->getBsrData();
        block_dim = bsr_data.block_dim;
        d_iperm = bsr_data.iperm;
        d_perm = bsr_data.perm;
    }
    ~DomDecKrylovWrapper(){};  // must have virtual destructor?
    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // NOTE : rhs is ignored here since we have to compute residual in separate place for
        // res_IEV (FETI-DP case) and still want to use same original call structure in
        // inexact_newton

        // fetidp->set_global_rhs(rhs); // this method doesn't work (needs element->IEV)
        // don't think set global rhs and then solving different linear system each time works?
        // cause needs element-level loads.. right? not equiv linear system cause rhs
        fetidp->get_lam_rhs(lam_rhs);
        bool fail = krylov->solve(lam_rhs, lam_soln, check_conv);
        fetidp->get_global_soln(lam_soln, soln);

        // permute from global to solve/perm ordering to return here..
        soln.permuteData(block_dim, d_iperm);

        return fail;  // for not fail
    }
    void compute_feti_residual(T lambdaE, T lambdaI, DeviceVec<T> vars) {
        // compute res = lambdaE * fext - lambdaI * fint where fint = kmat(u) * u
        // but here we compute it in the FETI u_IEV, f_IEV decomposed system instead..
        fetidp->set_IEV_residual(lambdaE, lambdaI, vars);
    }
    void add_res_IEV(T a, T b, DeviceVec<T> &_res_IEV) {
        // compute res = lambdaE * fext - lambdaI * fint where fint = kmat(u) * u
        // but here we compute it in the FETI u_IEV, f_IEV decomposed system instead..
        fetidp->add_res_IEV(a, b, _res_IEV);
    }
    void get_res_IEV(DeviceVec<T> &_res_IEV) { fetidp->get_res_IEV(_res_IEV); }
    void update_after_assembly(DeviceVec<T> &vars) {
        // krylov calls for DOMDEC preconditioner (so don't need to call on that)
        // printf("update after assembly in DomDecKrylovWrapper\n");
        krylov->update_after_assembly(vars);
        // printf("\tdone with update after assembly in DomDecKrylovWrapper\n");
    }
    void factor() {}
    void set_print(bool print) override { krylov->set_print(print); }
    void set_abs_tol(T atol) override { krylov->set_abs_tol(atol); }
    void set_rel_tol(T rtol) override { krylov->set_rel_tol(rtol); }
    void set_cycle_type(std::string cycle_) override { krylov->set_cycle_type(cycle_); }
    int get_num_iterations() override { return krylov->get_num_iterations(); }
    void free() {}

   public:
    DOMDEC *fetidp;
    KRYLOV *krylov;

   private:
    int nlam, N;
    int block_dim, *d_perm, *d_iperm;
    DeviceVec<T> lam_rhs, lam_soln;
    // DeviceVec<T> glob_rhs, glob_soln;
};
