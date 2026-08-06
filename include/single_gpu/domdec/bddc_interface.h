#pragma once
#include <string>

#include "mesh/vtk_writer.h"

// like a TacsInterface for the multigrid solvers
// can do function evals, set dvs, forward + adjoint solves, gradients, etc.

template <typename T, class Assembler, class Krylov, class BDDC>
class TacsBDDCInterface {
   public:
    using Vec = typename Assembler::template VecType<T>;
    // using Mat = typename Assembler::template MatType<Vec>;
    using MyFunction = typename Assembler::MyFunction;

    TacsBDDCInterface(Krylov *krylov_, BDDC *bddc_, Assembler &assembler_,
                      BsrMat<DeviceVec<T>> &kmat_, bool print = true,
                      bool include_adjoint_vars = true)
        : krylov(krylov_),
          bddc(bddc_),
          assembler(assembler_),
          include_adjoint_vars(include_adjoint_vars),
          print(print) {
        // create vectors
        loads = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        rhs = assembler.createVarsVec();
        kmat = kmat_;

        if (include_adjoint_vars) {
            dfdu = assembler.createVarsVec();
            psi = assembler.createVarsVec();
            CHECK_CUBLAS(cublasCreate(&cublasHandle));
        }

        // device vecs with same pointers (so don't need copying)
        d_IEV_elem_conn = bddc->get_IEV_conn();
        d_xpts_IEV = bddc->get_IEV_xpts();
        d_vars_IEV = bddc->get_IEV_vars();
        IEV_nnodes = bddc->get_num_IEV_nodes();
        dfdu_IEV = DeviceVec<T>(6 * IEV_nnodes);
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

        bddc->set_IEV_linear_rhs(this->vars);
        // make this more general later..
        bool check_conv = true;
        // NOTE : BDDC internal uses fext (but Krylov still needs this vec)
        krylov->solve(this->loads, this->soln, check_conv);
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
        // printf("krylov update after assembly inside BDDC interface\n");
        krylov->update_after_assembly(this->vars);
        // printf("\tdone with krylov update after assembly inside BDDC interface\n");
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

            // setup internal BDDC rhs compatible with external dfdu (nodal), with IEV-splitting
            dfdu_IEV.zeroValues();
            // first one basically just sets vars into IEV_vars
            a = -1.0;  // negative sign for adjoint (-df/du_IEV is rhs)
            bddc->set_IEV_adjoint_rhs(this->vars, dfdu_IEV, a);
            assembler.evalFunctionSVSens_BDDC_IEV(func, d_IEV_elem_conn, d_xpts_IEV, d_vars_IEV,
                                                  dfdu_IEV);
            // second call sets dfdu_IEV into rhs_IEV and res_IEV internally
            a = -1.0;  // negative sign for adjoint (-df/du_IEV is rhs)
            bddc->set_IEV_adjoint_rhs(this->vars, dfdu_IEV, a);

            // // TEMP DEBUG
            // int *h_IEV_elem_conn = d_IEV_elem_conn.createHostVec().getPtr();
            // printf("h_IEV_elem_conn (%d): ", d_IEV_elem_conn.getSize());
            // printVec<int>(d_IEV_elem_conn.getSize(), h_IEV_elem_conn);

            // T *h_xpts_IEV = d_xpts_IEV.createHostVec().getPtr();
            // T *h_vars_IEV = d_vars_IEV.createHostVec().getPtr();
            // T *h_dfdu_IEV = dfdu_IEV.createHostVec().getPtr();
            // printf("h_xpts_IEV (%d): ", d_xpts_IEV.getSize());
            // printVec<T>(d_xpts_IEV.getSize(), h_xpts_IEV);

            // for (int IEV_node = 0; IEV_node < IEV_nnodes; IEV_node++) {
            //     T *xpt = &h_xpts_IEV[3 * IEV_node];
            //     T *var = &h_vars_IEV[6 * IEV_node];
            //     T *dfdu_pt = &h_dfdu_IEV[6 * IEV_node];
            //     printf("IEV_node %d: \n", IEV_node);
            //     printf("\txpt ");
            //     printVec<T>(3, xpt);
            //     printf("\tvar ");
            //     printVec<T>(6, var);
            //     printf("\tdfdu_pt ");
            //     printVec<T>(6, dfdu_pt);
            // }

            // printf("h_vars_IEV (%d): ", d_vars_IEV.getSize());
            // printVec<T>(d_vars_IEV.getSize(), h_vars_IEV);

            // printf("h_dfdu_IEV (%d): ", IEV_nnodes);
            // printVec<T>(6 * IEV_nnodes, h_dfdu_IEV);

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
            // BDDC also has internal dfdu rhs setup..
            // printf("adjoint solve\n");
            krylov->solve(this->dfdu, this->psi, check_conv);
            // printf("\tdone with adjoint solve\n");
            assembler.apply_bcs(psi);  // dirichlet boundary conditions

            // check that K * psi - dfdu = 0 residual (or close to that)
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
        krylov->free();
        if (include_adjoint_vars) {
            dfdu.free();
            psi.free();
        }
    }

   protected:
    Krylov *krylov;
    BDDC *bddc;
    Assembler assembler;
    Vec loads, soln, res, vars, rhs;
    Vec dfdu, psi;
    bool include_adjoint_vars;
    cublasHandle_t cublasHandle = NULL;
    bool print;
    BsrMat<DeviceVec<T>> kmat;

    DeviceVec<int> d_IEV_elem_conn;
    int IEV_nnodes;
    DeviceVec<T> d_xpts_IEV, d_vars_IEV, dfdu_IEV;

    T rtol = 1e-6, atol = 1e-6;
    T omega = 1.0;
    int pre_smooth = 1, post_smooth = 1, n_cycles = 200, print_freq = 3;
    bool double_smooth = true;
};