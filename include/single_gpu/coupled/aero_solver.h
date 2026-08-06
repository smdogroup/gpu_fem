#pragma once

template <typename T, class Vec>
class BaseAeroSolver {
    // TODO:
};

template <typename T, class Vec>
class FixedAeroSolver {
    // supplies fixed aero loads each time
   public:
    FixedAeroSolver(int na_surf, Vec &fa_full) : na_surf(na_surf) {
        fa = fa_full.removeRotationalDOF();
    }

    void solve(Vec &ua) {}
    Vec& getAeroLoads() { return fa; }

    int get_num_surf_nodes() { return na_surf; }

    void free() {
        fa.free();
    }

   private:
    int na_surf;
    Vec fa;
};

template <typename T, class Vec>
class Fun3dAeroSolver {
   public:
    // TODO : make constructor with any additional inputs like FUN3D wrapper
    // Fun3dAeroSolver(int na_surf) : na_surf(na_surf);

    void solve(Vec &ua) {
        // TODO put FUN3D wrapper analysis in here
        return;
    }
    Vec& getAeroLoads() { return fa; }

    void free() {
        fa.free();
    }

   private:
    int na_surf;
    Vec fa;
};