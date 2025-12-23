#pragma once

#include "../../assembler.h"
#include "../shell/a2d/a2dsymmatrotateframe.h"
#include "../shell/strains/basic_utils.h"
#include "../shell/strains/strain_types.h"
#include "_plate.cuh"
#include "a2dcore.h"
#include "cuda_utils.h"
#include "physics/_plate_utils.h"

// fully integrated here means we don't reduced integrated or MITC (mixed integrated)
// the transverse shear or membrane strains (in order to have better multigrid performance)
// only certain bases (chebyshev) can use it without shear locking..

// NOTE : the way it is written, it should only be solving plate bending problems (not membrane)
// because asymptotic transformation in plane not done correctly yet..

template <typename T, class Basis_, class Phys_, template <typename> class Vec_,
          template <typename> class Mat_>
class AsymptoticIsogeometricPlateAssembler
    : public ElementAssembler<AsymptoticIsogeometricPlateAssembler<T, Basis_, Phys_, Vec_, Mat_>, T,
                              Basis_, Phys_, Vec_, Mat_> {
   public:
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using Data = typename Phys_::Data;
    using Assembler = AsymptoticIsogeometricPlateAssembler<T, Basis_, Phys_, Vec_, Mat_>;
    using Base = ElementAssembler<Assembler, T, Basis_, Phys_, Vec_, Mat_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Mat = Mat_<Vec_<T>>;

    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * num_nodes;
    static constexpr int32_t dof_per_elem = vars_per_node * num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

// TODO : way to make this more general if num_quad_pts is not a multiple of 3?
// some if constexpr stuff on type of Basis?
#ifdef USE_GPU
    static constexpr dim3 energy_block = dim3(32, num_quad_pts, 1);
    static constexpr dim3 res_block = dim3(32, num_quad_pts, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    // default constructor
    AsymptoticIsogeometricPlateAssembler() = default;

    // constructor
    AsymptoticIsogeometricPlateAssembler(int32_t num_geo_nodes, int32_t num_vars_nodes,
                                         int32_t num_elements, HostVec<int32_t> &geo_conn,
                                         HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                                         HostVec<int> &bcs, HostVec<Data> &compData,
                                         int32_t num_components = 1,
                                         HostVec<int> elem_component = HostVec<int>(1))
        : Base(num_geo_nodes, num_vars_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
               compData, num_components, elem_component) {}

    template <int elems_per_block = 1>
    void add_jacobian_fast(Mat &mat) {
        // method for testing out faster jacobian GPU

        mat.zeroValues();
        // int cols_per_elem = (Quadrature::num_quad_pts <= 4) ? 24 : 9;
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);
        dim3 block(num_quad_pts, cols_per_elem,
                   elems_per_block);  // better order for consecutive threads and mem reads
        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nelem_cols = this->num_elements * dof_per_elem;
        int nblocks = (nelem_cols + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        k_add_aigplate_jacobian_fast<T, elems_per_block, Assembler, Data, Vec_, Mat>
            <<<grid, block>>>(this->num_vars_nodes, this->num_elements, cols_per_elem,
                              this->elem_components, this->geo_conn, this->vars_conn, this->xpts,
                              this->vars, this->compData, mat);

        CHECK_CUDA(cudaDeviceSynchronize());
        // #endif
    }

    template <int elems_per_block = 8>
    void add_residual_fast(Vec_<T> &res) {
        // method for testing out faster jacobian GPU

        res.zeroValues();
        dim3 block(num_quad_pts,
                   elems_per_block);  // better order for consecutive threads and mem reads
        int nblocks = (this->num_elements + elems_per_block - 1) / elems_per_block;
        dim3 grid(nblocks);

        k_add_aigplate_residual_fast<T, elems_per_block, Assembler, Data, Vec_><<<grid, block>>>(
            this->num_vars_nodes, this->num_elements, this->elem_components, this->geo_conn,
            this->vars_conn, this->xpts, this->vars, this->compData, res);

        CHECK_CUDA(cudaDeviceSynchronize());
        // #endif
    }

    void asymptotic_rhs_transform(Vec_<T> fext) {
        // host side function to rescale right hand side for asymptotic correctness

        dim3 block(32);
        int nblocks = (this->num_nodes + 31) / 32;
        dim3 grid(nblocks);

        k_asymptotic_rhs_transform<T, Assembler, Data, Vec_><<<grid, block>>>(
            this->num_vars_nodes, this->num_elements, this->elem_components, this->compData, fext);
    }

    void recover_asymptotic_solution(Vec_<T> soln) {
        // subtract first derivatives of rotations to w(x,y) for asymptotic correctness
    }

    // ==================================================================================================
    // ==================================================================================================
    // HELPER FUNCTIONS
    // ==================================================================================================
    // ==================================================================================================

    template <class Data>
    __DEVICE__ static void _asymptotic_lhs_transform(T xpts[xpts_per_elem], Data &compData) {
        // rescale xpts by 1/thickness
        for (int i = 0; i < xpts_per_elem; i++) {
            xpts[i] /= compData.thick;
        }

        // then replace thickness with 1 (so that weak form is thickness independent)
        compData.thick = 1.0;
    }

    template <class Data>
    __DEVICE__ static void _asymptotic_rhs_transform(T fext_node[vars_per_node], Data &compData) {
        T G = compData.E / 2.0 / (1 + compData.nu);
        fext_node[2] *= compData.thick;  // / G;  // just w force is rescaled by thickness
        // don't think I need to rescale by mu.. (only thickness needs to be eliminated)
        // see p. 3 of "Asymptotically accurate and locking-free finite element implementation
        // of first order shear deformation theory for plates"

        // for membrane part: rescale by thicknesses too
        fext_node[0] *= compData.thick;
        fext_node[1] *= compData.thick;
    }

    template <class Data>
    __DEVICE__ static void _asymptotic_soln_update(T fext_node[vars_per_node], Data &compData) {
        T G = compData.E / 2.0 / (1 + compData.nu);
        fext_node[2] *= compData.thick;  // / G;  // just w force is rescaled by thickness
        // don't think I need to rescale by mu.. (only thickness needs to be eliminated)
        // see p. 3 of "Asymptotically accurate and locking-free finite element implementation
        // of first order shear deformation theory for plates"

        // for membrane part: rescale by thicknesses too
        fext_node[0] *= compData.thick;
        fext_node[1] *= compData.thick;
    }

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void add_element_quadpt_residual_fast(  // __noinline__
        const int iquad, const bool bndry[4], const T xpts[xpts_per_elem], const Data &compData,
        const T vars[dof_per_elem], T res[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;

        // 0) startup of this element residual
        // -----------------------------------
        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        // 1/4 the quad weight (see examples/asym_iga/_python_demo/2_iga/3_plate.py since we've
        // halfed the comp domain
        weight *= 0.25;
        T Xdinv[4];
        T detJ = getTransformMatrix<T, Basis>(pt, bndry, xpts, Xdinv);
        T scale = detJ * weight;

        T XdinvT[9] = {0};  // 3x3 version of rotation matrix for tying strains
        XdinvT[0] = Xdinv[0], XdinvT[3] = Xdinv[2];
        XdinvT[1] = Xdinv[1], XdinvT[4] = Xdinv[3];
        XdinvT[8] = 1.0;

        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        //     printf("Xdinv: ");
        //     printVec<T>(4, Xdinv);

        //     printf("XdinvT: ");
        //     printVec<T>(9, XdinvT);
        // }

        // data to store in forwards + backwards section
        // static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (bending) {
            A2D::ADObj<A2D::Vec<T, 3>> ek;

            computeBendingStrain<T, vars_per_node, Basis>(pt, bndry, Xdinv, vars, ek.value());
            Phys::computeBendingStress(scale, compData, ek.value(), ek.bvalue());
            computeBendingStrainTranspose<T, vars_per_node, Basis>(pt, bndry, Xdinv, ek.bvalue(),
                                                                   res);
        }

        if constexpr (tying) {
            A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

            A2D::SymMat<T, 3> gty;
            computeFullTyingStrain<T, vars_per_node, Basis>(pt, bndry, xpts, vars, gty.get_data());
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
            computeEngineerTyingStrains<T>(e0ty.value());

            Phys::computeTyingStress(scale, compData, e0ty.value(), e0ty.bvalue());

            computeEngineerTyingStrains<T>(e0ty.bvalue());
            A2D::SymMat<T, 3> gty_bar;
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                gty_bar.get_data());
            computeFullTyingStrainSens<T, vars_per_node, Basis>(pt, bndry, xpts, gty_bar.get_data(),
                                                                res);
        }

        if constexpr (drill) {
            A2D::ADObj<A2D::Vec<T, 1>> et;
            computeDrillStrain<T, vars_per_node, Basis>(pt, bndry, Xdinv, vars,
                                                        et.value().get_data());
            Phys::computeDrillStress(scale, compData, et.value().get_data(),
                                     et.bvalue().get_data());
            computeDrillStrainTranspose<T, vars_per_node, Basis>(pt, bndry, Xdinv,
                                                                 et.bvalue().get_data(), res);
        }
    }

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void add_element_quadpt_jacobian_col_fast(  // __noinline__
        const int iquad, const bool bndry[4], const T xpts[xpts_per_elem], const Data &compData,
        const T vars[dof_per_elem], const T pvars[dof_per_elem], T matCol[dof_per_elem]) {
        static constexpr bool is_nonlinear = Phys::is_nonlinear;
        if constexpr (is_nonlinear) {
            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
                printf(
                    "WARNING: doesn't do nonlinear physics with AIG plate element yet.. tying "
                    "strains not "
                    "fully nonlinear. Kmat reverted to linear.\n");
            }
            // return;
        }

        // since it's linear just call residual array on pvars input
        add_element_quadpt_residual_fast<Data, strain>(iquad, bndry, xpts, compData, pvars, matCol);
    }

    // ------------------------------------------------------------
    // inactive methods from main assembler
    // ------------------------------------------------------------

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data compData, T &Uelem) {}

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T res[dof_per_elem]) {}

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const bool bndry[4], const Data compData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {}

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_mass_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T accel[dof_per_elem], const Data compData, T res[dof_per_elem]) {}

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_mass_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T accel[dof_per_elem], const Data compData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_adj_res_product(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, const T psi[dof_per_elem],
        T loc_dv_sens[]) {}

    template <class Data>
    __HOST_DEVICE__ static void _compute_element_quadpt_strains(
        const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
        const Data &compData, A2D::Mat<T, 3, 3> &u0x, A2D::Mat<T, 3, 3> &u1x,
        A2D::SymMat<T, 3> &e0ty, A2D::Vec<T, 1> &et) {}

    template <class Data>
    __HOST_DEVICE__ static void get_element_quadpt_failure_index(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
        T &fail_index) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_dv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T loc_dv_sens[]) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_sv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T dfdu_local[]) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T strains[vars_per_node]) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_stresses(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T stresses[vars_per_node]) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_mass(bool active_thread, int iquad,
                                                            const T xpts[], const Data compData,
                                                            T *output) {}

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_dmass_dx(bool active_thread, int iquad,
                                                                const T xpts[], const Data compData,
                                                                T *dm_dxlocal) {}
};