#include "_src/nl_wing.h"

/* goal is to view the L3 wing mesh CF transfer when it fails (is magnitude wrong?)
   why is the V-cycle breaking down */

int main() {

    int level = 3;
    double force = 4e7, omegaMC = 0.7, SR = 10.0;
    bool use_predictor = false, debug_gmg = true, kmg_print = false, nl_debug = true;
    int nsmooth = 4, ninnercyc = 2, n_krylov = 50;
    double omegaLS_min = 0.5, omegaLS_max = 2.0;

    auto solver = NonlinearWingGPUSolver(
        level, force, omegaMC, SR, use_predictor, kmg_print, 
        nl_debug, debug_gmg, nsmooth, ninnercyc, n_krylov,
        omegaLS_min, omegaLS_max
    );

    using T = double;

    int nvars = solver.get_num_vars();
    HostVec<T> u0(nvars), uf(nvars);

    // call the nonlinear cont solve (which prev fails), and this doesn't seem to run well in python
    // for some reason (multiple calls, so thus doing it in .cu)

    T lam0 = 0.2, lamf = 1.0, inner_atol = 1e-4;
    solver.continuationSolve(u0.getPtr(), uf.getPtr(), lam0, lamf, inner_atol);

    bool fail = true;
    if (fail) {
        printf("NL cont solve failed, doing debug V-cycle CF solve now\n");

        HostVec<T> i_defect(nvars), ism_defect(nvars), cf_soln(nvars);
        HostVec<T> ch_defect(nvars), fsm_defect(nvars), lu_soln(nvars);

        T lam = 1.0;
        solver.setGridDefect(uf.getPtr(), lam, true);
        solver.getCoarseFineStep(i_defect.getPtr(), ism_defect.getPtr(), cf_soln.getPtr(), 
            ch_defect.getPtr(), fsm_defect.getPtr(), lu_soln.getPtr());

        // write CF step at nonlinear state here
        solver.writeSolution("out/_wing_idefect_1.vtk", i_defect.getPtr());
        solver.writeSolution("out/_wing_ismdefect_1.vtk", ism_defect.getPtr());
        solver.writeSolution("out/_wing_cfsoln_1.vtk", cf_soln.getPtr());
        solver.writeSolution("out/_wing_lusoln_1.vtk", lu_soln.getPtr());
        solver.writeSolution("out/_wing_chdefect_1.vtk", ch_defect.getPtr());
        solver.writeSolution("out/_wing_fdefect_1.vtk", fsm_defect.getPtr());

        // TODO : then run getCF step about zero disps too to compare?

    } else {
        printf("NL cont solve passed, no debug CF solve\n");
    }

    return 0;
};
