#pragma once
#include <string>

#include "mesh/vtk_writer.h"

// like a TacsInterface for the multigrid solvers
// can do function evals, set dvs, forward + adjoint solves, gradients, etc.

template <typename T, class Assembler, class Multigrid>
class TacsMGInterface {
   public:
    using Vec = typename Assembler::template VecType<T>;
    // using Mat = typename Assembler::template MatType<Vec>;
    using MyFunction = typename Assembler::MyFunction;

    TacsMGInterface(Multigrid &mg, bool print = true, bool include_adjoint_vars = true)
        : mg(mg),
          assembler(mg.grids[0].assembler),
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

    // TODO : have this method write into the inner MG solver..
    // void set_mg_solver_settings(T rtol_, T atol_, int n_cycles_, int pre_smooth_, int
    // post_smooth_,
    //                             int print_freq_ = 1, bool double_smooth_ = true, T omega_ = 1.0)
    //                             {
    //     rtol = rtol_, atol = atol_, omega = omega_;
    //     n_cycles = n_cycles_, pre_smooth = pre_smooth_, post_smooth = post_smooth_,
    //     print_freq = print_freq_;
    //     double_smooth = double_smooth_;
    // }

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
        mg.grids[0].zeroSolution();
        mg.grids[0].setDefect(this->loads);
        // bool inner_print = false, inner_time = false;
        mg.solve();
        // mg.vcycle_solve(0, pre_smooth, post_smooth, n_cycles, inner_print, atol, rtol, omega,
        //                 double_smooth, print_freq, inner_time);
        mg.grids[0].getSolution(this->soln);
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
        this->mg.set_design_variables(x);
        this->_update_assembly();
    }

    void _update_assembly() {
        /* update kmat with new design */
        if (this->print) printf("updating assembly\n");
        // for nonlinear case this will change and have to be Galerkin GMG not new coarse grid
        // matrices, TBD on that though
        // just do top levels, MG will do other levels
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
        mg.update_after_assembly(this->vars);
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
    Multigrid mg;
    Assembler assembler;
    Vec loads, soln, res, vars, rhs;
    Vec dfdu, psi;
    bool include_adjoint_vars;
    cublasHandle_t cublasHandle = NULL;
    bool print;

    // mg solver settings
    T rtol = 1e-6, atol = 1e-6;
    T omega = 1.0;
    int pre_smooth = 1, post_smooth = 1, n_cycles = 200, print_freq = 3;
    bool double_smooth = true;
};