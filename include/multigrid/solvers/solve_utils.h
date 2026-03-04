#pragma once

class BaseSolver {
   public:
    using T = double;
    BaseSolver() = default;
    virtual ~BaseSolver() = default;  // must have virtual destructor?
    virtual bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) = 0;
    virtual void update_after_assembly(DeviceVec<T> &vars) = 0;
    virtual void factor() = 0;
    virtual void set_abs_tol(T atol) = 0;
    virtual void set_rel_tol(T atol) = 0;
    virtual int get_num_iterations() = 0;
    virtual void set_print(bool print) = 0;
    virtual void free() = 0;
    virtual void set_cycle_type(std::string cycle_) = 0;
};

class SolverOptions {
   public:
    using T = double;
    SolverOptions() = default;

    SolverOptions(T omega_, int nsmooth_, int ncycles_, bool symmetric_ = false, T atol_ = 1e-6,
                  T rtol_ = 1e-6, int print_freq_ = 1, bool debug_ = false, bool print_ = false,
                  bool inner_print_ = false) {
        omega = omega_;
        nsmooth = nsmooth_, ncycles = ncycles_;
        symmetric = symmetric_;
        atol = atol_, rtol = rtol_;
        print_freq = print_freq_;
        debug = debug_, print = print_;
        inner_print = inner_print_;
    }

    SolverOptions(const SolverOptions &other_options) {
        // copy constructor
        omega = other_options.omega;
        atol = other_options.atol;
        rtol = other_options.rtol;
        nsmooth = other_options.nsmooth;
        ncycles = other_options.ncycles;
        print_freq = other_options.print_freq;
        symmetric = other_options.symmetric;
        debug = other_options.debug;
        print = other_options.print;
        inner_print = other_options.inner_print;
    }

    // data for object
    T omega = 1.0, atol = 1e-6, rtol = 1e-6;
    int nsmooth = 2, ncycles = 2;
    int print_freq = 1, ilevel = -1;
    bool symmetric = false, debug = false, print = false, inner_print = false;
};