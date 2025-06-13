#pragma once

#include <cstring>

#include "chrono"
#include "cuda_utils.h"
#include "element/base_elem_group.h"
#include "mesh/TACSMeshLoader.h"

// linear algebra formats
#include "linalg/bsr_data.h"
#include "linalg/vec.h"
#include "optimization/analysis_function.h"

template <typename T_, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
class ElementAssembler {
   public:
    using T = T_;
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Mat = Mat_<Vec<T>>;
    using MyFunction = AnalysisFunction<T, Vec>;

    template <typename U>
    using VecType = Vec<U>;

    template <typename V>
    using MatType = Mat_<V>;

    static constexpr int32_t geo_nodes_per_elem = Geo::num_nodes;
    static constexpr int32_t vars_nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t spatial_dim = Geo::spatial_dim;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

    // function declarations (to make easier to use)
    // ------------------------
    ElementAssembler(int32_t num_geo_nodes_, int32_t num_vars_nodes_, int32_t num_elements_,
                     HostVec<int32_t> &geo_conn, HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                     HostVec<int> &bcs, HostVec<Data> &physData, int32_t num_components_ = 0,
                     HostVec<int> elem_component = HostVec<int>(0));
    void moveBsrDataToDevice();
    static ElementAssembler createFromBDF(TACSMeshLoader &mesh_loader, Data single_data);
    __HOST__ void apply_bcs(Vec<T> &vec, bool can_print = false);
    void apply_bcs(Mat &mat, bool can_print = false);
#ifdef USE_GPU
    DeviceVec<T> createVarsVec(T *data = nullptr, bool randomize = false, bool can_print = false);
#else
    HostVec<T> createVarsVec(T *data = nullptr, bool randomize = false);
#endif

    void set_variables(Vec<T> &newVars);
    void set_design_variables(Vec<T> &newDVs);
    void add_energy(T *glob_U, bool can_print = false);
    void add_residual(Vec<T> &res, bool can_print = false);
    void add_jacobian(Vec<T> &res, Mat &mat, bool can_print = false);

    // optimization
    void setupFunction(MyFunction &func);
    void evalFunction(MyFunction &func);
    // used in adjoint solve and total derivatives
    void evalFunctionDVSens(MyFunction &func);
    void evalFunctionSVSens(const MyFunction &func, Vec<T> &dfdu);
    void evalFunctionXptSens(MyFunction &func);
    void evalFunctionAdjResProduct(const Vec<T> &psi, MyFunction &func);

    // optimization utils
    void _compute_adjResProduct(const Vec<T> &psi, Vec<T> &dfdx);
    void _compute_ks_failure_SVsens(T rho_KS, T safetyFactor, Vec<T> &dfdu, T *_max_fail = nullptr,
                                    T *_sumexp_fail = nullptr);
    void _compute_ks_failure_DVsens(T rho_KS, T safetyFactor, Vec<T> &dfdu, T *_max_fail = nullptr,
                                    T *_sumexp_fail = nullptr);
    void _compute_mass_DVsens(Vec<T> &dfdx);

    // visualization
    void compute_visualization_states(Vec<T> &dvs_out, Vec<T> &fail_index, Vec<T> &strains,
                                      Vec<T> &stresses);

    // util functions
    BsrData &getBsrData() { return bsr_data; }
    Vec<T> getXpts() { return xpts; }
    Vec<int> getBCs() { return bcs; }
    Vec<int> getConn() { return vars_conn; }
    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    int get_num_dvs() { return num_components * Data::ndvs_per_comp; }
    int get_num_components() { return num_components; }
    int get_num_nodes() { return num_vars_nodes; }
    int get_num_elements() { return num_elements; }
    int get_num_vis_stresses() { return Phys::vars_per_node * num_elements; }
    void get_elem_components(Vec<int> &elem_comp_out) {
        elem_components.copyValuesTo(elem_comp_out);
    }

    HostVec<T> createVarsHostVec(T *data, bool randomize);
    void setBsrData(BsrData new_bsr_data) { this->bsr_data = new_bsr_data; }

    // private functions
    T _compute_ks_failure(T rho_KS, T safetyFactor, bool smooth = true, T *_max_fail = nullptr,
                          T *_sumexp_fail = nullptr);
    T _compute_mass();

    void permuteVec(Vec<T> vec) {
        if (bsr_data.perm) {
            vec.permuteData(bsr_data.block_dim, bsr_data.perm);
        } else {
            printf("bsr data has no iperm pointer\n");
        }
    }

    void invPermuteVec(Vec<T> vec) {
        if (bsr_data.iperm) {
            vec.permuteData(bsr_data.block_dim, bsr_data.iperm);
        } else {
            printf("bsr data has no iperm pointer\n");
        }
    }

    // ------------------------
    // end of function declaration section

    void free() {
        geo_conn.free();
        vars_conn.free();
        bcs.free();
        elem_components.free();
        xpts.free();
        vars.free();
        physData.free();
        bsr_data.free();
        dvs.free();
    }

   private:
    int32_t num_geo_nodes;
    int32_t num_vars_nodes;
    int32_t num_elements;  // Number of elements of this type
    int32_t num_components;

    Vec<int32_t> geo_conn, vars_conn;
    Vec<int> bcs, elem_components;
    Vec<T> xpts, vars, dvs;
    Vec<Data> physData;
    BsrData bsr_data;
};  // end of ElementAssembler class declaration

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssembler<T, ElemGroup, Vec, Mat>::ElementAssembler(
    int32_t num_geo_nodes_, int32_t num_vars_nodes_, int32_t num_elements_,
    HostVec<int32_t> &geo_conn, HostVec<int32_t> &vars_conn, HostVec<T> &xpts, HostVec<int> &bcs,
    HostVec<Data> &physData, int32_t num_components_, HostVec<int> elem_components)
    : num_geo_nodes(num_geo_nodes_),
      num_vars_nodes(num_vars_nodes_),
      num_elements(num_elements_),
      num_components(num_components_) {
    // keeping inputs as HostVec even if running on device eventually here
    // std::unique, std::sort, and std::vector not directly supported on GPU
    // there are some options like thrust::sort, thrust::device_vector,
    // thrust::unique but I would also need to launch kernel for BSR..
    // should be cheap enough to just do on host now and then
    // createDeviceVec here

    int32_t num_vars = get_num_vars();
    this->vars = Vec<T>(num_vars);

    // on host (TODO : if need to deep copy entries to device?)
    // TODO : should probably do factorization explicitly instead of
    // implicitly upon construction
    bsr_data = BsrData(num_elements, num_vars_nodes, Basis::num_nodes, Phys::vars_per_node,
                       vars_conn.getPtr());

    int ndvs = get_num_dvs();

#ifdef USE_GPU

    // convert everything to device vecs
    this->geo_conn = geo_conn.createDeviceVec();
    this->vars_conn = vars_conn.createDeviceVec();
    this->xpts = xpts.createDeviceVec();
    this->bcs = bcs.createDeviceVec();
    this->physData = physData.createDeviceVec(false);
    this->elem_components = elem_components.createDeviceVec();
    this->dvs = DeviceVec<T>(ndvs);

#else  // not USE_GPU

    // on host just copy normally
    this->geo_conn = geo_conn;
    this->vars_conn = vars_conn;
    this->xpts = xpts;
    this->bcs = bcs;
    this->physData = physData;
    this->elem_components = elem_components;
    this->dvs = HostVec<T>(ndvs);

#endif  // end of USE_GPU or not USE_GPU check
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssembler<T, ElemGroup, Vec, Mat> ElementAssembler<T, ElemGroup, Vec, Mat>::createFromBDF(
    TACSMeshLoader &mesh_loader, Data single_data) {
    int vars_per_node = Phys::vars_per_node;  // input

    int num_nodes, num_elements, num_bcs, num_components;
    int *elem_conn, *bcs, *elem_components;
    T *xpts;

    mesh_loader.getAssemblerCreatorData(vars_per_node, num_nodes, num_elements, num_bcs,
                                        num_components, elem_conn, bcs, elem_components, xpts);

    // make HostVec objects here for Assembler
    HostVec<int> elem_conn_vec(vars_nodes_per_elem * num_elements, elem_conn);
    HostVec<int> bcs_vec(num_bcs, bcs);
    HostVec<int> elem_components_vec(num_elements, elem_components);
    HostVec<T> xpts_vec(spatial_dim * num_nodes, xpts);
    HostVec<Data> physData_vec(num_elements, single_data);

    printf("num_components = %d\n", num_components);

    // call base constructor
    return ElementAssembler(num_nodes, num_nodes, num_elements, elem_conn_vec, elem_conn_vec,
                            xpts_vec, bcs_vec, physData_vec, num_components, elem_components_vec);
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::moveBsrDataToDevice() {
#ifdef USE_GPU
    this->bsr_data = bsr_data.createDeviceBsrData();
#endif
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
T ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_mass() {
    using Quadrature = typename ElemGroup::Quadrature;
    constexpr int32_t elems_per_block = ElemGroup::res_block.x;

#ifdef USE_GPU

    // temporary mass device pointer (for adding up total mass)
    DeviceVec<T> d_mass(1);

    dim3 block(32, Quadrature::num_quad_pts);
    int nblocks = (num_elements + block.x - 1) / block.x;
    dim3 grid(nblocks);

    compute_mass_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, xpts, physData, d_mass.getPtr());

    CHECK_CUDA(cudaDeviceSynchronize());

    T *h_mass = d_mass.createHostVec().getPtr();
    return h_mass[0];

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_mass_DVsens(Vec<T> &dfdx) {
    using Quadrature = typename ElemGroup::Quadrature;
#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    compute_mass_DVsens_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, elem_components, geo_conn, xpts, physData, dfdx);

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
T ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_ks_failure(T rho_KS, T safetyFactor,
                                                                bool smooth, T *_max_fail,
                                                                T *_sumexp_fail) {
    using Quadrature = typename ElemGroup::Quadrature;

#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    // first compute the max failure index (not KS), so we can prevent overflow
    DeviceVec<T> d_max_fail(1);
    compute_max_failure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                          safetyFactor, d_max_fail.getPtr());
    T h_max_fail = d_max_fail.createHostVec()[0];

    // then do sum KS max fail
    DeviceVec<T> d_sum_ksfail(1);
    compute_ksfailure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                          safetyFactor, h_max_fail, d_sum_ksfail.getPtr());
    // add back global non-smooth max (overflow prevention)
    T sumexp_ks_fail = d_sum_ksfail.createHostVec()[0];
    T h_ksmax_fail = h_max_fail + log(sumexp_ks_fail) / rho_KS;

    // copy states (so can reuse for derivatives)
    if (_max_fail) {
        *_max_fail = h_max_fail;
    }
    if (_sumexp_fail) {
        *_sumexp_fail = sumexp_ks_fail;
    }

    // need smooth case for optim
    if (smooth) {
        return h_ksmax_fail;
    } else {
        return h_max_fail;
    }

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::compute_visualization_states(Vec<T> &dvs_out,
                                                                            Vec<T> &fail_index,
                                                                            Vec<T> &strains,
                                                                            Vec<T> &stresses) {
    using Quadrature = typename ElemGroup::Quadrature;

    dvs.copyValuesTo(dvs_out);

#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    vis_failure_index_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, fail_index);

    vis_strains_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, strains);

    vis_stresses_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, stresses);

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_ks_failure_DVsens(T rho_KS, T safetyFactor,
                                                                          Vec<T> &dfdx,
                                                                          T *_max_fail,
                                                                          T *_sumexp_fail) {
    using Quadrature = typename ElemGroup::Quadrature;

#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    // rerun old states from forward (if not provided)
    T h_max_fail, h_sumexp_ks_fail;
    if (_max_fail) {
        h_max_fail = *_max_fail;
    } else {
        // first compute the max failure index (not KS), so we can prevent overflow
        DeviceVec<T> d_max_fail(1);
        compute_max_failure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                              safetyFactor, d_max_fail.getPtr());
        h_max_fail = d_max_fail.createHostVec()[0];
    }

    if (_sumexp_fail) {
        h_sumexp_ks_fail = *_sumexp_fail;
    } else {
        // second, do sum ks fail (needed for denom of KS derivs)
        DeviceVec<T> d_sum_ksfail(1);
        compute_ksfailure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                              safetyFactor, h_max_fail, d_sum_ksfail.getPtr());
        // add back global non-smooth max (overflow prevention)
        h_sumexp_ks_fail = d_sum_ksfail.createHostVec()[0];
    }

    // now compute the DVsens gradient
    compute_ksfailure_DVsens_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, elem_components, geo_conn, vars_conn, xpts, vars, physData,
                          rho_KS, safetyFactor, h_max_fail, h_sumexp_ks_fail, dfdx);

    CHECK_CUDA(cudaDeviceSynchronize());

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_ks_failure_SVsens(T rho_KS, T safetyFactor,
                                                                          Vec<T> &dfdu,
                                                                          T *_max_fail,
                                                                          T *_sumexp_fail) {
    using Quadrature = typename ElemGroup::Quadrature;

#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    // dfdu.zeroValues();

    // rerun old states from forward (if not provided)
    T h_max_fail, h_sumexp_ks_fail;
    if (_max_fail) {
        h_max_fail = *_max_fail;
    } else {
        // first compute the max failure index (not KS), so we can prevent overflow
        DeviceVec<T> d_max_fail(1);
        compute_max_failure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                              safetyFactor, d_max_fail.getPtr());
        h_max_fail = d_max_fail.createHostVec()[0];
    }

    if (_sumexp_fail) {
        h_sumexp_ks_fail = *_sumexp_fail;
    } else {
        // second, do sum ks fail (needed for denom of KS derivs)
        DeviceVec<T> d_sum_ksfail(1);
        compute_ksfailure_kernel<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                              safetyFactor, h_max_fail, d_sum_ksfail.getPtr());
        // add back global non-smooth max (overflow prevention)
        h_sumexp_ks_fail = d_sum_ksfail.createHostVec()[0];
    }

    // now compute the SVsens gradient
    compute_ksfailure_SVsens_kernel<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, rho_KS,
                          safetyFactor, h_max_fail, h_sumexp_ks_fail, dfdu);

    CHECK_CUDA(cudaDeviceSynchronize());

#endif
};

// doesn't do anything yet, TODO to write it
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::evalFunctionXptSens(MyFunction &func) {
    func.check_setup();
    // then check function through if statement and make call
    // func.xpt_sens
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::_compute_adjResProduct(const Vec<T> &psi,
                                                                      Vec<T> &dfdx) {
    using Quadrature = typename ElemGroup::Quadrature;
    // apply bcs to the adjoint vector first
    // so that dRe/dxe doesn't contribute to fixed bc terms
    // psi.apply_bcs();

#ifdef USE_GPU

    const int elems_per_block = 32;
    dim3 block(Quadrature::num_quad_pts, elems_per_block);
    int nblocks = (num_elements + block.y - 1) / block.y;
    dim3 grid(nblocks);

    // very similar kernel to the residual call
    // add into dfdx
    compute_adjResProduct_kernel<T, ElemGroup, Data, elems_per_block, Vec><<<grid, block>>>(
        num_elements, elem_components, geo_conn, vars_conn, xpts, vars, physData, psi, dfdx);

#endif
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
__HOST__ void ElementAssembler<T, ElemGroup, Vec, Mat>::apply_bcs(Vec<T> &vec, bool can_print) {
    if (can_print) {
        printf("apply bcs to vector\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    vec.apply_bcs(bcs);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished apply bcs vec in %.4e seconds\n", dt);
    }
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::apply_bcs(Mat &mat, bool can_print) {
    if (can_print) {
        printf("apply bcs to matrix\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    mat.apply_bcs(bcs);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished apply bcs matrix in %.4e sec\n", dt);
    }
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
HostVec<T> ElementAssembler<T, ElemGroup, Vec, Mat>::createVarsHostVec(T *data, bool randomize) {
    HostVec<T> h_vec;
    if (data == nullptr) {
        h_vec = HostVec<T>(get_num_vars());
    } else {
        h_vec = HostVec<T>(get_num_vars(), data);
    }
    if (randomize) {
        h_vec.randomize();
    }
    return h_vec;
}

#ifdef USE_GPU
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
DeviceVec<T> ElementAssembler<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize,
                                                                     bool can_print) {
    if (can_print) {
        printf("begin create vars host vec\n");
    }
    auto h_vec = createVarsHostVec(data, randomize);
    if (can_print) {
        printf("inner checkpt 2\n");
    }
    auto d_vec = h_vec.createDeviceVec(true, can_print);
    if (can_print) {
        printf("inner checkpt 3\n");
    }
    return d_vec;
}
#else
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
HostVec<T> ElementAssembler<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize) {
    return createVarsHostVec(data, randomize);
}
#endif

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::set_variables(Vec<T> &newVars) {
    /* set the state variables u => assembler */
    // vars is not reordered, permutations for Kmat, res only happen on assembly
    newVars.copyValuesTo(this->vars);
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::set_design_variables(Vec<T> &newDVs) {
// call kernel function to update the physData of each element, component by component
#ifdef USE_GPU

    newDVs.copyValuesTo(dvs);

    constexpr int elems_per_block = 32;
    dim3 block(elems_per_block);
    int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
    dim3 grid(nblocks);

    set_design_variables_gpu<T, elems_per_block, Data, Vec>
        <<<grid, block>>>(num_elements, newDVs, elem_components, physData);

    CHECK_CUDA(cudaDeviceSynchronize());

#else
    printf("host vec design variables set not supported yet\n");
#endif
}

//  template <class ExecParameters>
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::add_energy(T *glob_U, bool can_print) {
    auto start = std::chrono::high_resolution_clock::now();
    if (can_print) {
        printf("begin add_energy\n");
    }

    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
    dim3 block = ElemGroup::energy_block;
    int nblocks = (num_elements + block.x - 1) / block.x;
    dim3 grid(nblocks);
    constexpr int32_t elems_per_block = ElemGroup::energy_block.x;

    add_energy_gpu<T, ElemGroup, Data, elems_per_block, Vec>
        <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, glob_U);
    CHECK_CUDA(cudaDeviceSynchronize());
#else   // USE_GPU
    ElemGroup::template add_energy_cpu<Data, Vec>(num_elements, geo_conn, vars_conn, xpts, vars,
                                                  physData, glob_U);
#endif  // USE_GPU

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished assembly in %.4e sec\n", dt);
    }
};

//  template <class ExecParameters>
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::add_residual(Vec<T> &res, bool can_print) {
    auto start = std::chrono::high_resolution_clock::now();
    if (can_print) {
        printf("begin add_residual\n");
    }

    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename ElemGroup::Quadrature;

    res.zeroValues();

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
    constexpr bool use_custom_kernel = false;

    if constexpr (use_custom_kernel) {
        // customized add_residual_gpu for this element
        // ElemGroup::template opt_add_residual_gpu<Data, Vec>(num_elements, geo_conn, vars_conn,
        // xpts,
        //                                                     vars, physData, res);

        dim3 block = ElemGroup::res_block;
        int nblocks = (num_elements + block.y - 1) / block.y;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.y;

        // TODO : figure out how to make this call from the ElemGroup..
        fast_drill_strain_add_residual<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        // shell_elem_add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec>
        //     <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);
    } else {
        // const int elems_per_block = 32;
        // dim3 block(elems_per_block, Quadrature::num_quad_pts);
        // int nblocks = (num_elements + block.x - 1) / block.x;
        // dim3 grid(nblocks);
        // add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec>
        //     <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        // v2 quadpts every 4 threads now + warp reduction
        // const int elems_per_block = 32;  // 32;
        // dim3 block(Quadrature::num_quad_pts, elems_per_block);
        // int nblocks = (num_elements + block.y - 1) / block.y;
        // dim3 grid(nblocks);
        // add_residual_gpu_v2<T, ElemGroup, Data, elems_per_block, Vec>
        //     <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        // v3
        // const int elems_per_block = 32;  // 32;
        // dim3 block(Quadrature::num_quad_pts, elems_per_block);
        // int nblocks = (num_elements + block.y - 1) / block.y;
        // dim3 grid(nblocks);
        // add_residual_gpu_v3<T, ElemGroup, Data, elems_per_block, Vec>
        //     <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        // v5 - try changing elems per block (still v2 kernel best)
        const int elems_per_block = 32;  // 32;
        dim3 block(Quadrature::num_quad_pts, elems_per_block);
        int nblocks = (num_elements + block.y - 1) / block.y;
        dim3 grid(nblocks);
        add_residual_gpu_v2<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        // v6 - don't parallelize over quadpts
        // const int elems_per_block = 32;  // 32;
        // dim3 block(elems_per_block);
        // int nblocks = (num_elements + block.x - 1) / block.x;
        // dim3 grid(nblocks);
        // add_residual_gpu_v6<T, ElemGroup, Data, elems_per_block, Vec>
        //     <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

#else   // USE_GPU
    ElemGroup::template add_residual_cpu<Data, Vec>(num_elements, geo_conn, vars_conn, xpts, vars,
                                                    physData, res);
#endif  // USE_GPU

    // permute residual (new => old rows see tests/reordering/README.md)
    // this->permuteVec(res);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> add_resid_time = stop - start;
    if (can_print) {
        printf("\tfinished add_residual in %.4e\n", add_resid_time.count());
    }
};

//  template <class ExecParameters>
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
void ElementAssembler<T, ElemGroup, Vec, Mat_>::add_jacobian(
    Vec<T> &res, Mat_<Vec<T>> &mat,
    bool can_print) {  // TODO : make this Vec here..
    auto start = std::chrono::high_resolution_clock::now();
    if (can_print) {
        printf("begin add_jacobian\n");
    }

    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Mat = Mat_<Vec<T>>;
    using Quadrature = typename ElemGroup::Quadrature;

    res.zeroValues();
    mat.zeroValues();

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

    // dim3 block = ElemGroup::jac_block;
    // int nblocks = (num_elements + block.x - 1) / block.x;
    // dim3 grid(nblocks);
    // constexpr int32_t elems_per_block = ElemGroup::jac_block.x;

    // v1 or baseline
    // const int elems_per_block = 1;
    // dim3 block(elems_per_block, 24, Quadrature::num_quad_pts);
    // int nblocks = (num_elements + block.x - 1) / block.x;
    // dim3 grid(nblocks);
    // add_jacobian_gpu<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
    //     num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    // v2
    // const int elems_per_block = 1;
    // dim3 block(Quadrature::num_quad_pts, 24, elems_per_block);
    // int nblocks = (num_elements + block.z - 1) / block.z;
    // dim3 grid(nblocks);
    // add_jacobian_gpu_v2<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
    //     num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    // v3
    // const int elems_per_block = 1;
    // dim3 block(Quadrature::num_quad_pts, 24, elems_per_block);
    // int nblocks = (num_elements + block.z - 1) / block.z;
    // dim3 grid(nblocks);
    // add_jacobian_gpu_v3<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
    //     num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    // // v4 - no adding into resid
    const int elems_per_block = 1;
    dim3 block(Quadrature::num_quad_pts, 24, elems_per_block);
    int nblocks = (num_elements + block.z - 1) / block.z;
    dim3 grid(nblocks);
    add_jacobian_gpu_v4<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
        num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    // v5 - change block size (same as v4 kernel)
    // const int elems_per_block = 1;
    // dim3 block(Quadrature::num_quad_pts, 24, elems_per_block);
    // int nblocks = (num_elements + block.z - 1) / block.z;
    // dim3 grid(nblocks);
    // add_jacobian_gpu_v4<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
    //     num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    // v6 - change block size (same as v4 kernel)
    // const int elems_per_block = 1;
    // dim3 block(1, 24, elems_per_block);
    // int nblocks = (num_elements + block.z - 1) / block.z;
    // dim3 grid(nblocks);
    // add_jacobian_gpu_v6<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
    //     num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

    CHECK_CUDA(cudaDeviceSynchronize());

#else  // CPU data
    // maybe a way to call add_residual_kernel as same method on CPU
    // with elems_per_block = 1
    ElemGroup::template add_jacobian_cpu<Data, Vec, Mat>(num_vars_nodes, num_elements, geo_conn,
                                                         vars_conn, xpts, vars, physData, res, mat);
#endif

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> add_jac_time = stop - start;
    if (can_print) {
        printf("\tfinished add_jacobian in %.4e sec\n", add_jac_time.count());
    }
};

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::setupFunction(MyFunction &func) {
    // setup num DVs, xpt of new mesh potentially
    int num_dvs = get_num_dvs();
    int num_xpts = get_num_xpts();
    func.init_sens(num_dvs, num_xpts);
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::evalFunction(MyFunction &func) {
    func.check_setup();
    if (func.name == "mass") {
        func.value = _compute_mass();
    } else if (func.name == "ksfailure") {
        auto *ks_func = dynamic_cast<KSFailure<T, Vec> *>(&func);
        if (!ks_func) {
            throw std::runtime_error("Invalid function type: expected KSFailure");
        }
        ks_func->value = _compute_ks_failure(ks_func->rho_KS, ks_func->safetyFactor);
    }
}

// plan to solve adjoint system outside of assembler, maybe can make a template or static method
// later
template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::evalFunctionDVSens(MyFunction &func) {
    // df/dx partial term (not total derivative
    func.check_setup();
    if (func.name == "mass") {
        _compute_mass_DVsens(func.dv_sens);

    } else if (func.name == "ksfailure") {
        auto *ks_func = dynamic_cast<KSFailure<T, Vec> *>(&func);
        if (!ks_func) {
            throw std::runtime_error("Invalid function type: expected KSFailure");
        }
        _compute_ks_failure_DVsens(ks_func->rho_KS, ks_func->safetyFactor, func.dv_sens);
    }
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::evalFunctionSVSens(const MyFunction &func,
                                                                  Vec<T> &dfdu) {
    // df/du partial term
    func.check_setup();
    if (func.name == "mass") {
        // pass non-adjoint function
    } else if (func.name == "ksfailure") {
        // Attempt safe downcast
        auto *ks_func = dynamic_cast<const KSFailure<T, Vec> *>(&func);
        if (!ks_func) {
            throw std::runtime_error("Invalid function type: expected KSFailure");
        }
        _compute_ks_failure_SVsens(ks_func->rho_KS, ks_func->safetyFactor, dfdu);
    }
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssembler<T, ElemGroup, Vec, Mat>::evalFunctionAdjResProduct(const Vec<T> &psi,
                                                                         MyFunction &func) {
    func.check_setup();
    // add into dfdx that is df/dx += psi^T dR/dx
    _compute_adjResProduct(psi, func.dv_sens);
}