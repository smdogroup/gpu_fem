#pragma once

#include <cstring>

#include "chrono"
#include "cuda_utils.h"
#include "mesh/TACSMeshLoader.h"

// linear algebra formats
#include "linalg/bsr_data.h"
#include "linalg/vec.h"

template <typename T_, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
class ElementAssemblerV1 {
   public:
    using T = T_;
    using Geo = typename ElemGroup::Geo;
    using Basis = typename ElemGroup::Basis;
    using Phys = typename ElemGroup::Phys;
    using Data = typename Phys::Data;
    using Mat = Mat_<Vec<T>>;

    template <typename U>
    using VecType = Vec<U>;

    template <typename V>
    using MatType = Mat_<V>;

    static constexpr int32_t geo_nodes_per_elem = Geo::num_nodes;
    static constexpr int32_t vars_nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t spatial_dim = Geo::spatial_dim;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

    void add_residual(Vec<T> &res, bool can_print = false) {
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print) {
            printf("begin add_residual\n");
        }

        using Phys = typename ElemGroup::Phys;
        using Data = typename Phys::Data;

        res.zeroValues();

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU
        dim3 block = ElemGroup::res_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::res_block.x;

        add_residual_gpu<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid, block>>>(num_elements, geo_conn, vars_conn, xpts, vars, physData, res);

        CHECK_CUDA(cudaDeviceSynchronize());

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

    void add_jacobian(Vec<T> &res, Mat_<Vec<T>> &mat,
                      bool can_print) {  // TODO : make this Vec here..
        auto start = std::chrono::high_resolution_clock::now();
        if (can_print) {
            printf("begin add_jacobian\n");
        }

        using Phys = typename ElemGroup::Phys;
        using Data = typename Phys::Data;
        using Mat = Mat_<Vec<T>>;

        res.zeroValues();
        mat.zeroValues();

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

        dim3 block = ElemGroup::jac_block;
        int nblocks = (num_elements + block.x - 1) / block.x;
        dim3 grid(nblocks);
        constexpr int32_t elems_per_block = ElemGroup::jac_block.x;

        add_jacobian_gpu<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
            num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

        CHECK_CUDA(cudaDeviceSynchronize());
#endif
    };

    // function declarations (to make easier to use)
    // ------------------------
    ElementAssemblerV1(int32_t num_geo_nodes_, int32_t num_vars_nodes_, int32_t num_elements_,
                       HostVec<int32_t> &geo_conn, HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                       HostVec<int> &bcs, HostVec<Data> &physData, int32_t num_components_ = 0,
                       HostVec<int> elem_component = HostVec<int>(0));
    void moveBsrDataToDevice();
    static ElementAssemblerV1 createFromBDF(TACSMeshLoader<T> &mesh_loader, Data single_data);
    __HOST__ void apply_bcs(Vec<T> &vec, bool can_print = false);
    void apply_bcs(Mat &mat, bool can_print = false);
#ifdef USE_GPU
    DeviceVec<T> createVarsVec(T *data = nullptr, bool randomize = false, bool can_print = false);
#else
    HostVec<T> createVarsVec(T *data = nullptr, bool randomize = false);
#endif

    void set_variables(Vec<T> &newVars);

    // util functions
    BsrData &getBsrData() { return bsr_data; }
    Vec<T> getXpts() { return xpts; }
    Vec<int> getBCs() { return bcs; }
    Vec<int> getConn() { return vars_conn; }
    int get_num_xpts() { return num_geo_nodes * spatial_dim; }
    int get_num_vars() { return num_vars_nodes * vars_per_node; }
    int get_num_nodes() { return num_vars_nodes; }
    int get_num_elements() { return num_elements; }
    int get_num_dvs() { return num_components * Phys::num_dvs; }
    HostVec<T> createVarsHostVec(T *data, bool randomize);
    void setBsrData(BsrData new_bsr_data) { this->bsr_data = new_bsr_data; }

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
    }

   private:
    int32_t num_geo_nodes;
    int32_t num_vars_nodes;
    int32_t num_elements;  // Number of elements of this type
    int32_t num_components;

    Vec<int32_t> geo_conn, vars_conn;
    Vec<int> bcs, elem_components;
    Vec<T> xpts, vars;
    Vec<Data> physData;
    BsrData bsr_data;
};  // end of ElementAssemblerV1 class declaration

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssemblerV1<T, ElemGroup, Vec, Mat>::ElementAssemblerV1(
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

#ifdef USE_GPU

    // convert everything to device vecs
    this->geo_conn = geo_conn.createDeviceVec();
    this->vars_conn = vars_conn.createDeviceVec();
    this->xpts = xpts.createDeviceVec();
    this->bcs = bcs.createDeviceVec();
    this->physData = physData.createDeviceVec(false);
    this->elem_components = elem_components.createDeviceVec();

#else  // not USE_GPU

    // on host just copy normally
    this->geo_conn = geo_conn;
    this->vars_conn = vars_conn;
    this->xpts = xpts;
    this->bcs = bcs;
    this->physData = physData;
    this->elem_components = elem_components;

#endif  // end of USE_GPU or not USE_GPU check
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssemblerV1<T, ElemGroup, Vec, Mat>
ElementAssemblerV1<T, ElemGroup, Vec, Mat>::createFromBDF(TACSMeshLoader<T> &mesh_loader,
                                                          Data single_data) {
    int vars_per_node = Phys::vars_per_node;  // input

    int num_nodes, num_elements, num_bcs, num_components;
    int *elem_conn, *bcs, *elem_components;
    T *xpts;

    mesh_loader.getAssemblerCreatorData(vars_per_node, num_nodes, num_elements, num_bcs,
                                        num_components, elem_conn, bcs, elem_components, xpts);

    // make HostVec objects here for Assembler
    HostVec<int> elem_conn_vec(vars_nodes_per_elem * num_elements, elem_conn);
    HostVec<int> bcs_vec(num_bcs, bcs);
    HostVec<int> elem_components_vec(num_components, elem_components);
    HostVec<T> xpts_vec(spatial_dim * num_nodes, xpts);
    HostVec<Data> physData_vec(num_elements, single_data);

    // call base constructor
    return ElementAssemblerV1(num_nodes, num_nodes, num_elements, elem_conn_vec, elem_conn_vec,
                              xpts_vec, bcs_vec, physData_vec, num_components, elem_components_vec);
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssemblerV1<T, ElemGroup, Vec, Mat>::moveBsrDataToDevice() {
#ifdef USE_GPU
    this->bsr_data = bsr_data.createDeviceBsrData();
#endif
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
__HOST__ void ElementAssemblerV1<T, ElemGroup, Vec, Mat>::apply_bcs(Vec<T> &vec, bool can_print) {
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
void ElementAssemblerV1<T, ElemGroup, Vec, Mat>::apply_bcs(Mat &mat, bool can_print) {
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
HostVec<T> ElementAssemblerV1<T, ElemGroup, Vec, Mat>::createVarsHostVec(T *data, bool randomize) {
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
DeviceVec<T> ElementAssemblerV1<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize,
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
HostVec<T> ElementAssemblerV1<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize) {
    return createVarsHostVec(data, randomize);
}
#endif

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssemblerV1<T, ElemGroup, Vec, Mat>::set_variables(Vec<T> &newVars) {
    // vars is not reordered, permutations for Kmat, res only happen on assembly
    newVars.copyValuesTo(this->vars);
}
