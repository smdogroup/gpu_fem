#pragma once

#include <cstring>

#include "chrono"
#include "cuda_utils.h"
#include "mesh/TACSMeshLoader.h"
#include "shell_transforms.cuh"

// linear algebra formats
#include "linalg/bsr_data.h"
#include "linalg/vec.h"

template <typename T_, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat_>
class ElementAssemblerV3 {
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
        // if (can_print) {
        //     printf("begin add_residual\n");
        // }

        using Phys = typename ElemGroup::Phys;
        using Data = typename Phys::Data;

        res.zeroValues();

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

        // fewer elems can slightly speedup by increasing occupancy on SM
        constexpr int32_t elems_per_block = 32;  // 8 is faster

        constexpr int32_t kernel_option = ElemGroup::kernel_option;

        if constexpr (kernel_option == 1) {
            dim3 block(16, 32, 1);  // faster with (4,32,1) memory access so that's why I switch to
            // shared_v2 aka shared2
            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);
            drill_strain_residual_shared<T, ElemGroup, Data, elems_per_block, Vec><<<grid, block>>>(
                num_elements, geo_conn, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq, res);
        } else if (kernel_option == 2) {
            // again leads to slow mem access of 9e-4 sec vs 2e-4 for (4,32,1) so faster compute but
            // slower memory although, I can do warp reduction and bcast using just (4,32,1) so see
            // kernel_options 3 and 4 for improvement
            dim3 block(16, 32, 1);
            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);
            drill_strain_residual_local<T, ElemGroup, Data, elems_per_block, Vec><<<grid, block>>>(
                num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq, res);
        } else if (kernel_option == 3) {
            printf("launch kernel 3\n");
            // WAY faster 1e-3 to 2.7e-4 with fewer elems per block
            // really need launch parms since this is platform dependent probably
            // constexpr int32_t elems_per_block2 = 4;
            constexpr int32_t elems_per_block2 = 8;
            // constexpr int32_t elems_per_block2 = 16;
            // constexpr int32_t elems_per_block2 = 32;
            dim3 block(4, elems_per_block2, 1);
            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);

            drill_strain_residual_local2<T, ElemGroup, Data, elems_per_block2, Vec>
                <<<grid, block>>>(num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq,
                                  res);
        } else if (kernel_option == 4) {
            // change here is we use less shared mem (no longer shared mem of block_res)
            // this should help with jac later!
            // slightly faster at 2.2655e-4 vs 2.7e-4 than kernel option 3!
            printf("launch kernel 4\n");
            constexpr int32_t elems_per_block2 = 8;
            // constexpr int32_t elems_per_block2 = 16;
            // constexpr int32_t elems_per_block2 = 32;
            dim3 block(4, elems_per_block2, 1);

            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);

            drill_strain_residual_local3<T, ElemGroup, Data, elems_per_block2, Vec>
                <<<grid, block>>>(num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq,
                                  res);
        } else if (kernel_option == 5) {
            /* try here switching to shared mem for XdinvTn, Tmatn since I'm well under shared mem
            occupancy now with block_res no longer here goal is to reduce registers and see if that
            helps improve waves per SM occupancy */
            printf("launch kernel 5\n");
            constexpr int32_t elems_per_block2 = 8;
            // constexpr int32_t elems_per_block2 = 16;
            // constexpr int32_t elems_per_block2 = 32;
            dim3 block(4, elems_per_block2, 1);
            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);

            drill_strain_residual_shared3<T, ElemGroup, Data, elems_per_block2, Vec>
                <<<grid, block>>>(num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq,
                                  res);
        } else if (kernel_option == 6) {
            /* goal is to see if simpler code but a little more loading is comparable registers,
             * etc. */
            printf("launch kernel 6\n");
            constexpr int32_t elems_per_block2 = 8;
            // constexpr int32_t elems_per_block2 = 16;
            // constexpr int32_t elems_per_block2 = 32;
            dim3 block(4, elems_per_block2, 1);
            int nblocks = (num_elements + elems_per_block - 1) / elems_per_block;
            dim3 grid(nblocks);

            drill_strain_residual_local3_simple<T, ElemGroup, Data, elems_per_block2, Vec>
                <<<grid, block>>>(num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq,
                                  res);
        }
        // TODO : do one all with oneshot compute, no pre-computing xpts shell transforms

        CHECK_CUDA(cudaDeviceSynchronize());

#endif  // USE_GPU

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> add_resid_time = stop - start;
        if (can_print) {
            printf("add_residual in %.4e\n", add_resid_time.count());
        }
    };

    void add_jacobian(Vec<T> &res, Mat_<Vec<T>> &mat,
                      bool can_print = false) {  // TODO : make this Vec here..
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

        // constexpr int32_t mat_cols_per_block = 8;
        constexpr int32_t mat_cols_per_block = 24;
        // constexpr int32_t mat_cols_per_block = 32;

        dim3 block(4, mat_cols_per_block, 1);  // (4,8,1)
        int num_elem_cols = num_elements * 24;
        // int num_elem_cols = num_elements;
        int nblocks = (num_elem_cols + mat_cols_per_block - 1) / mat_cols_per_block;
        printf("nblocks %d, nelements %d\n", nblocks, num_elements);
        dim3 grid(nblocks);

        drill_strain_jac<T, ElemGroup, Data, mat_cols_per_block, Vec, Mat>
            <<<grid, block>>>(num_elements, vars_conn, vars, physData, Tmatn, XdinvTn, detXdq, mat);

        CHECK_CUDA(cudaDeviceSynchronize());
#endif

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> add_jac_time = stop - start;
        if (can_print) {
            printf("add_jac in %.4e\n", add_jac_time.count());
        }
    };

    void _compute_shell_transforms(bool can_print = true) {
        /* computes the shell transforms at each new design */
        auto start = std::chrono::high_resolution_clock::now();
        // if (can_print) {
        //     printf("begin compute shell transforms\n");
        // }

        using Phys = typename ElemGroup::Phys;
        using Data = typename Phys::Data;

// input is either a device array when USE_GPU or a host array if not USE_GPU
#ifdef USE_GPU

        // launch kernel to compute nodal transforms
        static constexpr int elems_per_block = 32;
        dim3 block1(elems_per_block, 4);
        int nblocks1 = (num_elements + block1.x - 1) / block1.x;
        dim3 grid1(nblocks1);

        compute_shell_nodal_transforms<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid1, block1>>>(num_elements, geo_conn, xpts, physData, Tmatn, XdinvTn);

        // launch kernel to compute quadpt element transforms / detXd (TBD)
        // static constexpr int elems_per_block = 32;
        dim3 block2(elems_per_block, 4);
        int nblocks2 = (num_elements + block2.x - 1) / block2.x;
        dim3 grid2(nblocks2);
        compute_shell_quadpt_transforms<T, ElemGroup, Data, elems_per_block, Vec>
            <<<grid2, block2>>>(num_elements, geo_conn, xpts, physData, detXdq);
        // TBD on adding XdinvTq and Tmatq to this computation

        CHECK_CUDA(cudaDeviceSynchronize());

#endif  // USE_GPU

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> add_resid_time = stop - start;
        if (can_print) {
            printf("compute shell transforms in %.4e\n", add_resid_time.count());
        }
    };

    // function declarations (to make easier to use)
    // ------------------------
    ElementAssemblerV3(int32_t num_geo_nodes_, int32_t num_vars_nodes_, int32_t num_elements_,
                       HostVec<int32_t> &geo_conn, HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                       HostVec<int> &bcs, HostVec<Data> &physData, int32_t num_components_ = 0,
                       HostVec<int> elem_component = HostVec<int>(0));
    void moveBsrDataToDevice();
    static ElementAssemblerV3 createFromBDF(TACSMeshLoader<T> &mesh_loader, Data single_data);
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

    // additional place to hold shell transform data for faster assembly
    Vec<T> Tmatn, XdinvTn, detXdq;

};  // end of ElementAssemblerV3 class declaration

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssemblerV3<T, ElemGroup, Vec, Mat>::ElementAssemblerV3(
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

    // create temporary data for kernels
    this->Tmatn = DeviceVec<T>(36 * num_elements);
    this->XdinvTn = DeviceVec<T>(36 * num_elements);
    this->detXdq = DeviceVec<T>(4 * num_elements);

#else  // not USE_GPU

    // on host just copy normally
    this->geo_conn = geo_conn;
    this->vars_conn = vars_conn;
    this->xpts = xpts;
    this->bcs = bcs;
    this->physData = physData;
    this->elem_components = elem_components;

#endif  // end of USE_GPU or not USE_GPU check

    // compute on each design and init the shell transform data
    this->_compute_shell_transforms(false);  // true or false to print or not
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
ElementAssemblerV3<T, ElemGroup, Vec, Mat>
ElementAssemblerV3<T, ElemGroup, Vec, Mat>::createFromBDF(TACSMeshLoader<T> &mesh_loader,
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
    return ElementAssemblerV3(num_nodes, num_nodes, num_elements, elem_conn_vec, elem_conn_vec,
                              xpts_vec, bcs_vec, physData_vec, num_components, elem_components_vec);
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssemblerV3<T, ElemGroup, Vec, Mat>::moveBsrDataToDevice() {
#ifdef USE_GPU
    this->bsr_data = bsr_data.createDeviceBsrData();
#endif
}

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
__HOST__ void ElementAssemblerV3<T, ElemGroup, Vec, Mat>::apply_bcs(Vec<T> &vec, bool can_print) {
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
void ElementAssemblerV3<T, ElemGroup, Vec, Mat>::apply_bcs(Mat &mat, bool can_print) {
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
HostVec<T> ElementAssemblerV3<T, ElemGroup, Vec, Mat>::createVarsHostVec(T *data, bool randomize) {
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
DeviceVec<T> ElementAssemblerV3<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize,
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
HostVec<T> ElementAssemblerV3<T, ElemGroup, Vec, Mat>::createVarsVec(T *data, bool randomize) {
    return createVarsHostVec(data, randomize);
}
#endif

template <typename T, typename ElemGroup, template <typename> class Vec,
          template <typename> class Mat>
void ElementAssemblerV3<T, ElemGroup, Vec, Mat>::set_variables(Vec<T> &newVars) {
    // vars is not reordered, permutations for Kmat, res only happen on assembly
    newVars.copyValuesTo(this->vars);
}

//  template <class ExecParameters>
// template <typename T, typename ElemGroup, template <typename> class Vec,
//           template <typename> class Mat_>
// void ElementAssemblerV3<T, ElemGroup, Vec, Mat_>::add_jacobian(
//     Vec<T> &res, Mat_<Vec<T>> &mat,
//     bool can_print) {  // TODO : make this Vec here..
//     auto start = std::chrono::high_resolution_clock::now();
//     if (can_print) {
//         printf("begin add_jacobian\n");
//     }

//     using Phys = typename ElemGroup::Phys;
//     using Data = typename Phys::Data;
//     using Mat = Mat_<Vec<T>>;

//     res.zeroValues();
//     mat.zeroValues();

// // input is either a device array when USE_GPU or a host array if not USE_GPU
// #ifdef USE_GPU

//     dim3 block = ElemGroup::jac_block;
//     int nblocks = (num_elements + block.x - 1) / block.x;
//     dim3 grid(nblocks);
//     constexpr int32_t elems_per_block = ElemGroup::jac_block.x;

//     add_jacobian_gpu<T, ElemGroup, Data, elems_per_block, Vec, Mat><<<grid, block>>>(
//         num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, vars, physData, res, mat);

//     CHECK_CUDA(cudaDeviceSynchronize());
// #endif
// };
