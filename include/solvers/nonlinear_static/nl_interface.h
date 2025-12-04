#pragma once
#include <string>

#include "mesh/vtk_writer.h"

// like a TacsInterface for the multigrid solvers
// can do function evals, set dvs, forward + adjoint solves, gradients, etc.

template <typename T, class Assembler, class LinearSolver, class Continuation>
class TACSNLInterface {
   public:
    using Vec = typename Assembler::template VecType<T>;
    // using Mat = typename Assembler::template MatType<Vec>;
    using MyFunction = typename Assembler::MyFunction;

    TacsMGInterface(Continuation &nl_solver_, Assembler &assembler_, LinearSolver &mg_,
                    Vec &struct_loads, bool print = true, bool include_adjoint_vars = true)
        : nl_solver(nl_solver),
          assembler(assembler_),
          mg(mg_),
          include_adjoint_vars(include_adjoint_vars),
          print(print) {
        // create vectors
        loads = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();

        // copy loads here
        struct_loads.copyValuesTo(loads);
        this->assembler.apply_bcs(this->loads);

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

    void set_loads(Vec &struct_loads) {
        // set constant loads in
        this->loads()
    }

    void solve() {
        // do nonlinear continuation solve
        this->nl_solver.solve(this->vars);  // inout updates the vars

        // set new variables into assembler (for output function evals)
        this->assembler.set_variables(this->vars);
    }

    void copy_solution_in(Vec &soln_in) {
        soln_in.copyValuesTo(this->soln);
        this->soln.copyValuesTo(this->vars);
        this->assembler.set_variables(this->vars);
    }
    void copy_solution_out(Vec &soln_out) { this->vars.copyValuesTo(soln_out); }

    void set_design_variables(Vec &x) {
        this->resetSoln();
        this->assembler.set_design_variables(x);
        this->mg.set_design_variables(x);
        this->_update_assembly();
    }

    void _update_assembly() {
        /* update kmat with new design */
        if (this->print) printf("updating assembly\n");
        // for nonlinear case this will change and have to be Galerkin GMG not new coarse grid
        // matrices, TBD on that though
        int ilevel = 0;
        // auto &res = mg.grids[ilevel].d_temp_vec;
        auto &kmat = mg.grids[ilevel].Kmat;
        auto &i_assembler = mg.grids[ilevel].assembler;
        i_assembler.add_jacobian_fast(kmat);
        i_assembler.apply_bcs(kmat);
        // for (int ilevel = 0; ilevel < mg.getNumLevels(); ilevel++) {
        //     auto &res = mg.grids[ilevel].d_temp_vec;
        //     auto &kmat = mg.grids[ilevel].Kmat;
        //     auto &i_assembler = mg.grids[ilevel].assembler;
        //     i_assembler.add_jacobian(res, kmat);
        //     i_assembler.apply_bcs(kmat);
        // }
        // redundant

        // additional assembly steps like update factor, smoother, etc.
        this->mg.update_after_assembly(this->vars);
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
            mg.grids[0].zeroSolution();
            mg.grids[0].setDefect(this->dfdu);
            // bool inner_print = false, inner_time = false;
            mg.solve();
            mg.grids[0].getSolution(this->psi);
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
        mg.free();
        if (include_adjoint_vars) {
            dfdu.free();
            psi.free();
        }
    }

   protected:
    Continuation nl_solver;
    Assembler assembler;
    LinearSolver mg;  // usually multigrid (need it separately for linear adjoint solves)
    Vec loads, soln, res, vars, rhs;
    Vec dfdu, psi;
    bool include_adjoint_vars;
    cublasHandle_t cublasHandle = NULL;
    bool print;
};