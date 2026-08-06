#pragma once
#include <stdexcept>
#include <string>

#include "linalg/vec.h"

template <typename T, template <typename> class Vec>
class AnalysisFunction {
   public:
    AnalysisFunction(std::string name, bool has_adjoint = true)
        : name(std::move(name)), setup(false), value(0.0), has_adjoint(has_adjoint) {}
    std::string name;
    T value;
    bool setup;
    bool has_adjoint;

    void check_setup() const {
        if (!setup) {
            throw std::runtime_error("Function is not set up properly!");
        }
    }

    void init_sens(int num_dvs, int num_xpts) {
        dv_sens = Vec<T>(num_dvs);
        xpt_sens = Vec<T>(num_xpts);
        setup = true;
    }

    void free() {
        dv_sens.free();
        xpt_sens.free();
    }

    // makes it an abstract class
    virtual ~AnalysisFunction() = default;

    //    private:
    // should we make this hostvec later?
    Vec<T> dv_sens;
    Vec<T> xpt_sens;
};

template <typename T, template <typename> class Vec>
class KSFailure : public AnalysisFunction<T, Vec> {
   public:
    KSFailure(T rho_KS, T safetyFactor = 1.0)
        : AnalysisFunction<T, Vec>("ksfailure", true), rho_KS(rho_KS), safetyFactor(safetyFactor) {}
    T rho_KS, safetyFactor;
};

template <typename T, template <typename> class Vec>
class Mass : public AnalysisFunction<T, Vec> {
   public:
    Mass() : AnalysisFunction<T, Vec>("mass", false) {}
};