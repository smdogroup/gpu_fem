#pragma once

#include "a2d/_a2dshell.h"
#include "a2dcore.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroupV1 {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroupV1<T, Director_, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
    static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;

// TODO : way to make this more general if num_quad_pts is not a multiple of 3?
// some if constexpr stuff on type of Basis?
#ifdef USE_GPU
    static constexpr dim3 energy_block = dim3(32, num_quad_pts, 1);
    static constexpr dim3 res_block = dim3(32, num_quad_pts, 1);
    // static constexpr dim3 res_block = dim3(64, num_quad_pts, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
    // static constexpr dim3 jac_block = dim3(dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    template <class Data, int debug_mode = -1>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem])

    {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T d[3 * num_nodes];   // needed for reverse mode, nonlinear case
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

            // get the scale for disp grad sens of the energy
            T scale = detXd * weight;

            // debug and testing modes (1 - drill, 2 - tying, 3 - bending)
            if constexpr (debug_mode == 2 || debug_mode == 3) {
                et.value()[0] = 0.0;
            }
            if constexpr (debug_mode == 1 || debug_mode == 3) {
                for (int i = 0; i < 6; i++) {
                    e0ty.value()[i] = 0.0;
                }
            }
            if constexpr (debug_mode == 1 || debug_mode == 2) {
                for (int i = 0; i < 9; i++) {
                    u0x.value()[i] = 0.0;
                    u1x.value()[i] = 0.0;
                }
            }

            // compute energy + energy-dispGrad sensitivites with physics
            Phys::template computeWeakRes<T>(physData, scale, u0x, u1x, e0ty, et);

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
        // T ety_bar[Basis::num_all_tying_points];
        ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            e0ty.bvalue(), res, d_bar.get_data(), ety_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), res,
                                               d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(), res);

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T scale;              // scale for energy derivatives
        T d[3 * num_nodes];   // need directors in reverse for nonlinear strains
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;
        A2D::A2DObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward section
        {
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.value().get_data(),
                u1x.value().get_data(), e0ty.value());

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope

        // hforward section (pvalue's)
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        T p_d[3 * num_nodes];
        {
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, et.pvalue().get_data());

            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                             p_d);

            T p_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                   p_ety);

            ShellComputeDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, p_ety,
                u0x.pvalue().get_data(), u1x.pvalue().get_data(), e0ty.pvalue());

        }  // end of hforward scope

        // derivatives over disp grad to strain energy portion
        // ---------------------
        Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, e0ty, et);
        // ---------------------
        // begin reverse blocks from strain energy => physical disp grad sens

        // breverse (1st order derivs)
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
        {
            A2D::Vec<T, 3 * num_nodes> d_bar;  // zeroes out on init
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), res, d_bar.get_data(), ety_bar.get_data());

            computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), res,
                                                   d_bar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                             res);

            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

        }  // end of breverse scope (1st order derivs)

        // hreverse (2nd order derivs)
        {
            A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
            A2D::Vec<T, 3 * num_nodes> d_hat;                  // zeroes out on init
            ShellComputeDispGradHrev<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), e0ty.hvalue(), matCol, d_hat.get_data(),
                ety_hat.get_data());

            computeTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                   ety_bar.get_data(), ety_hat.get_data(), matCol,
                                                   d_hat.get_data());

            Director::template computeDirectorHrev<vars_per_node, num_nodes>(fn, d_hat.get_data(),
                                                                             matCol);

            ShellComputeDrillStrainHrev<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);
        }  // end of hreverse scope (2nd order derivs)
    }      // add_element_quadpt_jacobian_col
};