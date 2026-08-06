#pragma once

#include "a2dcore.h"
#include "element/shell/a2d/a2dsymmatrotateframe.h"
#include "element/shell/strains/_all.h"
#include "gpu_assembler.h"

template <typename T, class Partitioner, class Director_, class Basis_, class Phys_>
class GPU_MITCShellAssembler
    : public GPUElementAssembler<GPU_MITCShellAssembler<T, Partitioner, Director_, Basis_, Phys_>,
                                 T, Partitioner, Basis_, Phys_> {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using Data = typename Phys_::Data;
    using Assembler = GPU_MITCShellAssembler<T, Partitioner, Director_, Basis_, Phys_>;
    using Base = GPUElementAssembler<Assembler, T, Partitioner, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;

    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * num_nodes;
    static constexpr int32_t dof_per_elem = vars_per_node * num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    // default constructor
    GPU_MITCShellAssembler() = default;

    // constructor
    GPU_MITCShellAssembler(MultiGPUContext *ctx_, int32_t num_nodes_, int32_t num_elements_,
                           HostVec<int> *h_elem_conn_, HostVec<T> *xpts, HostVec<int> *bcs_,
                           HostVec<Data> *compData, int32_t num_components_ = 1,
                           HostVec<int> *elem_component = new HostVec<int>(1))
        : Base(ctx_, num_nodes_, num_elements_, h_elem_conn_, xpts, bcs_, compData, num_components_,
               elem_component) {}

    Partitioner *getPartitioner() { return this->part; }

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void add_element_quadpt_residual_fast(
        const T pt[2], const T &scale, const T xpts[xpts_per_elem], const T fn[xpts_per_elem],
        const T XdinvT[9], const T Tmat[9], const T XdinvzT[9], const Data &compData,
        const T vars[dof_per_elem], T res[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (bending) {
            A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
            A2D::ADObj<A2D::Vec<T, 3>> ek;

            // forward
            {
                T d[3 * num_nodes];
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                computeBendingDispGrad<T, vars_per_node, Basis>(pt, vars, d, Tmat, XdinvT, XdinvzT,
                                                                u0x.value().get_data(),
                                                                u1x.value().get_data());

                computeBendingStrain<T, is_nonlinear>(
                    u0x.value().get_data(), u1x.value().get_data(), ek.value().get_data());
            }
            // __syncthreads();

            // 1st order brev (only need ek.bvalue, so no additional steps here)
            {
                Phys::computeBendingStress(scale, compData, ek.value(), ek.bvalue());

                computeBendingStrainSens<T, is_nonlinear>(
                    ek.bvalue().get_data(), u0x.value().get_data(), u1x.value().get_data(),
                    u0x.bvalue().get_data(), u1x.bvalue().get_data());

                A2D::Vec<T, 3 * num_nodes> d_bar;
                // TODO : change to Hrev when I add nonlinear back in for this part
                computeBendingDispGradSens<T, vars_per_node, Basis>(
                    pt, Tmat, XdinvT, XdinvzT, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
                    d_bar.get_data(), res);

                Director::template computeDirectorSens<vars_per_node, num_nodes>(
                    fn, d_bar.get_data(), res);
            }
        }

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            {
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                T ety[Basis::num_all_tying_points];
                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
                A2D::SymMat<T, 3> gty;
                interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

                A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

                computeEngineerTyingStrains<T>(e0ty.value());
                // __syncthreads();
            }

            // 1st order brev
            A2D::SymMat<T, 3> gty_bar;
            {
                Phys::computeTyingStress(scale, compData, e0ty.value(), e0ty.bvalue());

                computeEngineerTyingStrains<T>(e0ty.bvalue());

                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                    gty_bar.get_data());

                A2D::Vec<T, 3 * num_nodes> d_bar;
                A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
                interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
                computeMITCTyingStrainSens<T, Phys, Basis, is_nonlinear>(
                    xpts, fn, vars, d, ety_bar.get_data(), res, d_bar.get_data());

                Director::template computeDirectorSens<vars_per_node, num_nodes>(
                    fn, d_bar.get_data(), res);
            }
        }

        // just show the linear case pvalue and hvalue rn
        if constexpr (drill) {
            A2D::ADObj<A2D::Vec<T, 1>> et;

            // pforward
            ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(pt, Tmat, XdinvT, vars,
                                                                           et.value().get_data());
            // __syncthreads();

            // compute drill stress
            Phys::computeDrillStress(scale, compData, et.value().get_data(),
                                     et.bvalue().get_data());
            // __syncthreads();

            // hreverse for drill
            ShellComputeDrillStrainFastSens<T, vars_per_node, Basis, Director>(
                pt, Tmat, XdinvT, et.bvalue().get_data(), res);
        }

    }  // add_element_quadpt_residual_fast

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void add_element_quadpt_jacobian_col_fast(  // __noinline__ is slower actually
        const T pt[2], const T &scale, const T xpts[xpts_per_elem], const T fn[xpts_per_elem],
        const T XdinvT[9], const T Tmat[9], const T XdinvzT[9], const Data &compData,
        const T vars[dof_per_elem], const T pvars[dof_per_elem], T matCol[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (bending) {
            A2D::A2DObj<A2D::Mat<T, 3, 3>> u0x, u1x;
            A2D::A2DObj<A2D::Vec<T, 3>> ek;

            // forward
            if constexpr (is_nonlinear) {
                T d[3 * num_nodes];
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                computeBendingDispGrad<T, vars_per_node, Basis>(pt, vars, d, Tmat, XdinvT, XdinvzT,
                                                                u0x.value().get_data(),
                                                                u1x.value().get_data());

                computeBendingStrain<T, is_nonlinear>(
                    u0x.value().get_data(), u1x.value().get_data(), ek.value().get_data());
            }
            // __syncthreads();

            // just code in the linear part right now
            // pforward
            T p_d[3 * num_nodes];
            {
                Director::template computeDirector<vars_per_node, num_nodes>(pvars, fn, p_d);

                computeBendingDispGrad<T, vars_per_node, Basis>(pt, pvars, p_d, Tmat, XdinvT,
                                                                XdinvzT, u0x.pvalue().get_data(),
                                                                u1x.pvalue().get_data());

                computeBendingStrainHfwd<T, is_nonlinear>(
                    u0x.value().get_data(), u1x.value().get_data(), u0x.pvalue().get_data(),
                    u1x.pvalue().get_data(), ek.pvalue().get_data());
            }
            // __syncthreads();

            // 1st order brev (only need ek.bvalue, so no additional steps here)
            if constexpr (is_nonlinear) {
                Phys::computeBendingStress(scale, compData, ek.value(), ek.bvalue());
            }

            // 2nd order hrev
            {
                Phys::computeBendingStress(scale, compData, ek.pvalue(), ek.hvalue());

                computeBendingStrainHrev<T, is_nonlinear>(
                    ek.hvalue().get_data(), ek.bvalue().get_data(), u0x.value().get_data(),
                    u1x.value().get_data(), u0x.pvalue().get_data(), u1x.pvalue().get_data(),
                    u0x.hvalue().get_data(), u1x.hvalue().get_data());

                A2D::Vec<T, 3 * num_nodes> d_hat;
                // TODO : change to Hrev when I add nonlinear back in for this part
                computeBendingDispGradSens<T, vars_per_node, Basis>(
                    pt, Tmat, XdinvT, XdinvzT, u0x.hvalue().get_data(), u1x.hvalue().get_data(),
                    d_hat.get_data(), matCol);

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    fn, d_hat.get_data(), matCol);
            }
        }

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

                T ety[Basis::num_all_tying_points];
                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
                A2D::SymMat<T, 3> gty;
                interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

                A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

                computeEngineerTyingStrains<T>(e0ty.value());
                // __syncthreads();
            }

            // pforward section
            // -------------------------------
            T p_d[3 * num_nodes];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(pvars, fn, p_d);

                T p_ety[Basis::num_all_tying_points];
                computeMITCTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, pvars, p_d, p_ety);
                A2D::SymMat<T, 3> p_gty;
                interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());

                A2D::SymMatRotateFrame<T, 3>(XdinvT, p_gty, e0ty.pvalue());

                computeEngineerTyingStrains<T>(e0ty.pvalue());
            }
            // __syncthreads();

            // 1st order brev
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
            if constexpr (is_nonlinear) {
                Phys::computeTyingStress(scale, compData, e0ty.value(), e0ty.bvalue());

                computeEngineerTyingStrains<T>(e0ty.bvalue());

                A2D::SymMat<T, 3> gty_bar;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                    gty_bar.get_data());

                interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
                // __syncthreads();
            }

            // 2nd order hrev
            {
                Phys::computeTyingStress(scale, compData, e0ty.pvalue(), e0ty.hvalue());

                computeEngineerTyingStrains<T>(e0ty.hvalue());

                A2D::SymMat<T, 3> gty_hat;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.hvalue().get_data(),
                                                    gty_hat.get_data());

                A2D::Vec<T, 3 * num_nodes> d_hat;
                A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
                interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());
                computeMITCTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, pvars, p_d,
                                                           ety_bar.get_data(), ety_hat.get_data(),
                                                           matCol, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    fn, d_hat.get_data(), matCol);
            }
        }

        // just show the linear case pvalue and hvalue rn
        if constexpr (drill) {
            A2D::A2DObj<A2D::Vec<T, 1>> et;

            // pforward
            ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(pt, Tmat, XdinvT, pvars,
                                                                           et.pvalue().get_data());
            // __syncthreads();

            // compute drill stress
            Phys::computeDrillStress(scale, compData, et.pvalue().get_data(),
                                     et.hvalue().get_data());
            // __syncthreads();

            // hreverse for drill
            ShellComputeDrillStrainFastSens<T, vars_per_node, Basis, Director>(
                pt, Tmat, XdinvT, et.hvalue().get_data(), matCol);
        }

    }  // add_element_quadpt_jacobian_col
};