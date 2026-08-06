#pragma once
#include <string>

#include "mesh/vtk_writer.h"

// supported classes
// TacsLinearStatic
// TacsNonlinearStaticNewton
// required methods: solve(fs) => us

template <class Mat, class Vec>
using LinearSolveFunc = void (*)(Mat &, Vec &, Vec &, bool, bool);

template <typename T, class Assembler>
class BaseTacsStatic {
   public:
    using Vec = typename Assembler::template VecType<T>;
    using Mat = typename Assembler::template MatType<Vec>;
    using MyFunction = typename Assembler::MyFunction;

    using LinearSolve = LinearSolveFunc<Mat, Vec>;

    BaseTacsStatic(Assembler &assembler, Mat &kmat, LinearSolve linear_solve, bool print = true,
                   bool include_adjoint_vars = true)
        : assembler(assembler),
          kmat(kmat),
          linear_solve(linear_solve),
          include_adjoint_vars(include_adjoint_vars),
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

    void set_design_variables(Vec &x) { assembler.set_design_variables(x); }

    T evalFunction(MyFunction &func) {
        /* evaluate a function */
        if (!func.setup) assembler.setupFunction(func);
        assembler.evalFunction(func);
        return func.value;
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
            linear_solve(kmat, dfdu, psi, print,
                         true);        // TODO : will need transpose kmat assembly later
            assembler.apply_bcs(psi);  // dirichlet boundary conditions
        }

        func.dv_sens.zeroValues();
        assembler.evalFunctionDVSens(func);
        if (func.has_adjoint) assembler.evalFunctionAdjResProduct(psi, func);
        // DVsens is then stored in func.dv_sens
        // TODO : get XptSens also
        // TODO : could add coupled adjoint jac product here too
    }

    void free() {
        loads.free();
        soln.free();
        res.free();
        vars.free();
        rhs.free();
        if (include_adjoint_vars) {
            dfdu.free();
            psi.free();
        }
    }

    // void solve(Vec &struct_loads);  // virtual

   protected:
    Assembler assembler;
    LinearSolve linear_solve;
    Mat kmat;
    Vec loads, soln, res, vars, rhs;
    Vec dfdu, psi;
    bool include_adjoint_vars;
    cublasHandle_t cublasHandle = NULL;
    bool print;
};

template <typename T, class Assembler>
class TacsLinearStatic : public BaseTacsStatic<T, Assembler> {
   public:
    using Base = BaseTacsStatic<T, Assembler>;
    using Vec = typename Base::Vec;
    using Mat = typename Base::Mat;
    using LinearSolve = typename Base::LinearSolve;

    TacsLinearStatic(Assembler &assembler, Mat &kmat, LinearSolve linear_solve, bool print = false)
        : BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve, print) {
        // compute the linear kmat on construction
        assembler.set_variables(this->vars);
        assembler.add_jacobian(this->res, this->kmat);
        assembler.apply_bcs(this->kmat);
    }

    void solve(Vec &struct_loads) {
        // copy loads
        this->loads.zeroValues();
        struct_loads.copyValuesTo(this->loads);
        this->assembler.apply_bcs(this->loads);

        // solve the linear system
        this->soln.zeroValues();
        this->linear_solve(this->kmat, this->loads, this->soln, this->print, true);
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
        this->assembler.add_jacobian(this->res, this->kmat);
        this->assembler.apply_bcs(this->kmat);
    }

   private:
    bool print;
};

// TODO : later make separate classes or settings to use Newton solver vs. Riks solver here?

template <typename T, class Assembler>
class TacsNonlinearStaticNewton : public BaseTacsStatic<T, Assembler> {
   public:
    using Base = BaseTacsStatic<T, Assembler>;
    using Vec = typename Base::Vec;
    using Mat = typename Base::Mat;
    using LinearSolve = typename Base::LinearSolve;

    TacsNonlinearStaticNewton(Assembler &assembler, Mat &kmat, LinearSolve linear_solve,
                              int num_load_factors, int num_newton, bool print = false,
                              T abs_tol = 1e-8, T rel_tol = 1e-6, bool write_vtk = false,
                              std::string outputFilePrefix = "tacs_output")
        : BaseTacsStatic<T, Assembler>(assembler, kmat, linear_solve, print) {
        // store the nonlinear newton solve settings
        this->num_load_factors = num_load_factors;
        this->num_newton = num_newton;
        this->abs_tol = abs_tol;
        this->rel_tol = rel_tol;
        this->outputFilePrefix = outputFilePrefix;
        this->write_vtk = write_vtk;
    }

    void solve(Vec &struct_loads) {
        // copy loads
        this->loads.zeroValues();
        struct_loads.copyValuesTo(this->loads);
        this->assembler.apply_bcs(this->loads);

        // reset things:
        this->soln.zeroValues();
        this->res.zeroValues();
        this->rhs.zeroValues();

        // TODO : add continuation strategy where min_load_factor is not zero each time, and looser
        // solves at start for now just full solve
        T min_load_factor = 1.0 / num_load_factors;
        T max_load_factor = 1.0;

        // now do Newton solve
        this->soln.zeroValues();
        newton_solve(this->linear_solve, this->kmat, this->loads, this->soln, this->assembler,
                     this->res, this->rhs, this->vars, num_load_factors, min_load_factor,
                     max_load_factor, num_newton, abs_tol, rel_tol, outputFilePrefix, this->print);
    }

   private:
    int num_load_factors, num_newton;
    T abs_tol, rel_tol;
    std::string outputFilePrefix;
    bool write_vtk;
};
