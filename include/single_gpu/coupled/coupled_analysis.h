#pragma once
#include "utils.h"

// TODO : finish FuntofemCoupledAnalysis solver
// TODO : make an Mphys version as well (so we can compare)

template <typename T, class Vec, class StructSolver, class AeroSolver, class Transfer>
class FuntofemCoupledAnalysis {
    // an aeroelastic coupled analysis, later can change it to include other couplings if desired
   public:
    FuntofemCoupledAnalysis(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer,
                            int num_coupled_steps, bool demo = false)
        : struct_solver(struct_solver),
          aero_solver(aero_solver),
          transfer(transfer),
          num_coupled_steps(num_coupled_steps) {
        // initialize coupled states
        ns = struct_solver.get_num_nodes();
        na = aero_solver.get_num_surf_nodes();
        us = Vec(3 * ns);
        ua = Vec(3 * na);
        fs = Vec(3 * ns);
        fa = Vec(3 * na);
        fs_ext = Vec(6 * ns);
        this->demo = demo;
    }

    void solve_forward() {
        // TODO : add optional uncoupled phase here?

        us.zeroValues();  // initial zero struct disps to start loop

        for (int icoupled = 0; icoupled < num_coupled_steps; icoupled++) {
            printf("\ncoupled step %d / %d\n------------------\n", icoupled, num_coupled_steps);
            // block Gauss-seidel strategy
            ua = transfer.transferDisps(us);
            aero_solver.solve(ua);  // or iterate
            fa = aero_solver.getAeroLoads();
            fs = transfer.transferLoads(fa);
            fs.addRotationalDOF(fs_ext);
            // TODO : need some way to continue the struct soln better for coupled scheme
            // for now just reset the struct variables since I'm starting from scratch
            if (demo) struct_solver.resetSoln();
            struct_solver.solve(fs_ext);
            struct_solver.getStructDisps(us);
        }
    }

    void free() {
        us.free();
        ua.free();
        fs.free();
        fa.free();
        fs_ext.free();
    }

   private:
    int num_coupled_steps;
    StructSolver struct_solver;
    AeroSolver aero_solver;
    Transfer transfer;
    bool demo;

    int ns, na;
    Vec us, ua, fs, fa, fs_ext;
};