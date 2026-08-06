#pragma once
#include <cassert>
#include <string>

#include "mesh/vtk_writer.h"

// Combined nonlinear + adjoint interface
// - primal solve uses Continuation
// - adjoint solve uses LinearSolver
// - optional BDDC IEV adjoint path is supported

template <typename T, class Assembler, class LinearSolver, class Continuation, class BDDC>
class TACSNLBddcInterface {
   public:
    using Vec = typename Assembler::template VecType<T>;
    using MyFunction = typename Assembler::MyFunction;

    TACSNLBddcInterface(cublasHandle_t &cublasHandle_, Continuation *nl_solver_,
                        LinearSolver *linear_solver_, BDDC *bddc_, Assembler &assembler_,
                        bool print = true, bool include_adjoint_vars = true,
                        bool use_bddc_adjoint_ieV = true, T inner_frtol_ = 1e-6)
        : cublasHandle(cublasHandle_),
          nl_solver(nl_solver_),
          linear_solver(linear_solver_),
          bddc(bddc_),
          assembler(assembler_),
          print(print),
          include_adjoint_vars(include_adjoint_vars),
          use_bddc_adjoint_ieV(use_bddc_adjoint_ieV),
          inner_frtol(inner_frtol_) {
        // create vectors
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();

        if (include_adjoint_vars) {
            dfdu = assembler.createVarsVec();
            psi = assembler.createVarsVec();
        }

        // BDDC IEV data only needed for special adjoint path
        if (use_bddc_adjoint_ieV) {
            assert(bddc != nullptr);

            d_IEV_elem_conn = bddc->get_IEV_conn();
            d_xpts_IEV = bddc->get_IEV_xpts();
            d_vars_IEV = bddc->get_IEV_vars();
            IEV_nnodes = bddc->get_num_IEV_nodes();
            dfdu_IEV = DeviceVec<T>(6 * IEV_nnodes);
        }
    }

    int get_num_nodes() { return assembler.get_num_nodes(); }
    int get_num_dvs() { return assembler.get_num_dvs(); }

    Vec getStructDisps() { return vars.removeRotationalDOF(); }
    void getStructDisps(Vec &us_xyz) { vars.removeRotationalDOF(us_xyz); }

    Assembler &getAssembler() { return assembler; }

    void resetSoln() {
        vars.zeroValues();
        assembler.set_variables(vars);
    }

    void writeSoln(std::string filename) {
        auto h_vars = vars.createHostVec();
        printToVTK<Assembler, HostVec<T>>(assembler, h_vars, filename);
    }

    T evalFunction(MyFunction &func) {
        if (!func.setup) {
            assembler.setupFunction(func);
        }
        assembler.evalFunction(func);
        return func.value;
    }

    bool solve() {
        // Nonlinear primal solve only
        T lambda0 = 0.2;
        T inner_atol = 1e-8;
        T lambdaf = 1.0;
        T inner_crtol = 1e-3;

        bool fail = nl_solver->solve(vars, lambda0, inner_atol, lambdaf, inner_crtol, inner_frtol);

        // update assembler state after nonlinear solve
        assembler.set_variables(vars);

        return fail;
    }

    void copy_solution_in(Vec &soln_in) {
        soln_in.copyValuesTo(soln);
        soln.copyValuesTo(vars);
        assembler.set_variables(vars);
    }

    void copy_solution_out(Vec &soln_out) { vars.copyValuesTo(soln_out); }

    void set_design_variables(Vec &x) {
        // Usually no reset here for nonlinear continuation
        assembler.set_design_variables(x);

        // If continuation has its own design-variable update hook, call it here
        // nl_solver->set_design_variables(x);
    }

    void solve_adjoint(MyFunction &func, const Vec *adj_rhs = nullptr) {
        assert(include_adjoint_vars);

        if (!func.setup) {
            assembler.setupFunction(func);
        }

        int nvars = assembler.get_num_vars();
        T a;

        if (func.has_adjoint) {
            dfdu.zeroValues();
            assembler.evalFunctionSVSens(func, dfdu);

            // ------------------------------------------------------------
            // Optional BDDC IEV adjoint RHS path
            // ------------------------------------------------------------
            if (use_bddc_adjoint_ieV) {
                assert(bddc != nullptr);

                dfdu_IEV.zeroValues();

                // First call updates internal IEV vars from current state
                a = -1.0;
                bddc->set_IEV_adjoint_rhs(vars, dfdu_IEV, a);

                assembler.evalFunctionSVSens_BDDC_IEV(func, d_IEV_elem_conn, d_xpts_IEV, d_vars_IEV,
                                                      dfdu_IEV);

                // Second call pushes assembled dfdu_IEV into BDDC internal rhs/res
                a = -1.0;
                bddc->set_IEV_adjoint_rhs(vars, dfdu_IEV, a);
            }

            // Standard nodal adjoint RHS: -df/du + adj_rhs
            a = -1.0;
            CHECK_CUBLAS(cublasDscal(cublasHandle, nvars, &a, dfdu.getPtr(), 1));

            if (adj_rhs) {
                a = 1.0;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nvars, &a, adj_rhs->getPtr(), 1, dfdu.getPtr(), 1));
            }

            assembler.apply_bcs(dfdu);

            // ------------------------------------------------------------
            // Linearized adjoint solve
            // LinearSolver should behave like Krylov here
            // ------------------------------------------------------------
            bool check_conv = true;
            linear_solver->solve(dfdu, psi, check_conv);

            assembler.apply_bcs(psi);
        }

        func.dv_sens.zeroValues();
        assembler.evalFunctionDVSens(func);

        if (func.has_adjoint) {
            assembler.evalFunctionAdjResProduct(psi, func);
        }
    }

    void writeAdjointSolution(std::string filename) {
        psi.copyValuesTo(vars);
        writeSoln(filename);
    }

    void free() {
        soln.free();
        res.free();
        vars.free();
        rhs.free();

        if (linear_solver) {
            linear_solver->free();
        }

        if (include_adjoint_vars) {
            dfdu.free();
            psi.free();
        }
    }

   protected:
    Continuation *nl_solver = nullptr;
    LinearSolver *linear_solver = nullptr;
    BDDC *bddc = nullptr;

    Assembler assembler;

    Vec soln, res, vars, rhs;
    Vec dfdu, psi;

    bool print = true;
    bool include_adjoint_vars = true;
    bool use_bddc_adjoint_ieV = true;

    cublasHandle_t &cublasHandle;
    T inner_frtol = 1e-6;

    // BDDC IEV adjoint data
    DeviceVec<int> d_IEV_elem_conn;
    int IEV_nnodes = 0;
    DeviceVec<T> d_xpts_IEV, d_vars_IEV, dfdu_IEV;
};