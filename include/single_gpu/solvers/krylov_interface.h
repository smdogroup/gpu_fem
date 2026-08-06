#pragma once
#include <string>

#include "mesh/vtk_writer.h"

// like a TacsInterface for the multigrid solvers
// can do function evals, set dvs, forward + adjoint solves, gradients, etc.

template <typename T, class Assembler, class Krylov>
class TacsKrylovInterface {
   public:
    using Vec = typename Assembler::template VecType<T>;
    // using Mat = typename Assembler::template MatType<Vec>;
    using MyFunction = typename Assembler::MyFunction;

    TacsKrylovInterface(Krylov &krylov_, Assembler &assembler_, BsrMat<DeviceVec<T>> &kmat_,
                        bool print = true, bool include_adjoint_vars = true)
        : krylov(krylov_),
          assembler(assembler_),
          include_adjoint_vars(include_adjoint_vars),
          kmat(kmat_),
          print(print) {
        // create vectors
        loads = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();

        if (include_adjoint_vars) {
            dfdu = assembler.createVarsVec();
            psi = assembler.createVarsVec();
            CHECK_CUBLAS(cublasCreate(&cublasHandle));
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
        /* evaluate a function */
        if (!func.setup) assembler.setupFunction(func);
        assembler.evalFunction(func);
        return func.value;
    }

    void solve(Vec &struct_loads) {
        // copy loads
        this->loads.zeroValues();
        struct_loads.copyValuesTo(this->loads);
        this->assembler.apply_bcs(this->loads);

        // make this more general later..
        bool check_conv = true;
        krylov.solve(this->loads, this->soln, check_conv);
        this->soln.copyValuesTo(this->vars);

        // set new variables into assembler (for output function evals)
        this->assembler.set_variables(this->vars);
    }

    void copy_solution_in(Vec &soln_in) {
        soln_in.copyValuesTo(this->soln);
        this->soln.copyValuesTo(this->vars);
        this->assembler.set_variables(this->vars);
    }
    void copy_solution_out(Vec &soln_out) { this->soln.copyValuesTo(soln_out); }

    void set_design_variables(Vec &x) {
        this->resetSoln();
        this->assembler.set_design_variables(x);
        this->_update_assembly();
    }

    void _update_assembly() {
        /* update kmat with new design */
        if (this->print) printf("updating assembly\n");
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);

        // additional assembly steps like update factor, smoother, etc.
        krylov.update_after_assembly(this->vars);
    }

    void solve_adjoint(MyFunction &func, const Vec *adj_rhs = nullptr) {
        /* adjoint analysis for a single function */
        assert(include_adjoint_vars);
        int nvars = assembler.get_num_vars();
        T a;
        if (!func.setup) assembler.setupFunction(func);

        // solve adjoint system K^T * psi = -dfdu + adj_rhs
        if (func.has_adjoint) {
            dfdu.zeroValues();
            assembler.evalFunctionSVSens(func, dfdu);
            a = -1.0;
            CHECK_CUBLAS(cublasDscal(cublasHandle, nvars, &a, dfdu.getPtr(), 1));
            if (adj_rhs) {
                a = 1.0;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nvars, &a, adj_rhs->getPtr(), 1, dfdu.getPtr(), 1));
            }
            assembler.apply_bcs(dfdu);  // zero RHS part too?

            // make this more general later.. solves adjoint problem here
            bool check_conv = true;
            krylov.solve(this->dfdu, this->psi);
            assembler.apply_bcs(psi);  // dirichlet boundary conditions
        }

        func.dv_sens.zeroValues();
        assembler.evalFunctionDVSens(func);
        if (func.has_adjoint) assembler.evalFunctionAdjResProduct(psi, func);
    }

    void writeAdjointSolution(std::string filename) {
        psi.copyValuesTo(vars);
        writeSoln(filename);
    }

    void free() {
        loads.free();
        soln.free();
        res.free();
        vars.free();
        rhs.free();
        krylov.free();
        if (include_adjoint_vars) {
            dfdu.free();
            psi.free();
        }
    }

   protected:
    Krylov krylov;
    Assembler assembler;
    BsrMat<DeviceVec<T>> kmat;
    Vec loads, soln, res, vars, rhs;
    Vec dfdu, psi;
    bool include_adjoint_vars;
    cublasHandle_t cublasHandle = NULL;
    bool print;

    T rtol = 1e-6, atol = 1e-6;
    T omega = 1.0;
    int pre_smooth = 1, post_smooth = 1, n_cycles = 200, print_freq = 3;
    bool double_smooth = true;
};