#pragma once

#include "../base_elem_group.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "basis.h"
#include "director.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_>
class ShellElementGroup : public BaseElementGroup<ShellElementGroup<T, Director_, Basis_, Phys_>, T,
                                                  typename Basis_::Geo, Basis_, Phys_> {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroup<T, Director_, Basis_, Phys_>;
    using Base = BaseElementGroup<ElemGroup, T, Geo, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Base::xpts_per_elem;
    static constexpr int32_t dof_per_elem = Base::dof_per_elem;
    static constexpr int32_t num_quad_pts = Base::num_quad_pts;
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

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data physData, T &Uelem) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        A2D::ADObj<T> _Uelem;
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
            T d[3 * num_nodes];
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

            // compute energy + energy-dispGrad sensitivites with physics
            Phys::template computeStrainEnergy<T>(physData, scale, u0x, u1x, e0ty, et, _Uelem);

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Uelem += _Uelem.value();
    }

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

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col_no_resid(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T matCol[dof_per_elem]) {
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

        T res[24];  // dummy resid (should get compiled away since not used anywhere)
        // breverse (1st order derivs)
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
        if constexpr (is_nonlinear) {
            A2D::Vec<T, 3 * num_nodes> d_bar;  // zeroes out on init
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), res, d_bar.get_data(), ety_bar.get_data());

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

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_gmat_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars0[dof_per_elem], const T path[dof_per_elem], const Data physData,
        T matCol[dof_per_elem]) {
        if (!active_thread) return;

        // compute the stability matrix G
        using T2 = A2D::ADScalar<T, 1>;

        // copy variables over and set
        // directional deriv in for path deriv of vars
        T2 vars[dof_per_elem];
        for (int i = 0; i < dof_per_elem; i++) {
            vars[i].value = vars0[i];
            vars[i].deriv = path[i];
        }

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T scale;              // scale for energy derivatives
        T d[3 * num_nodes];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Mat<T2, 3, 3>> u0x, u1x;
        A2D::A2DObj<A2D::SymMat<T2, 3>> e0ty;
        A2D::A2DObj<A2D::Vec<T2, 1>> et;

        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // TODO : need to add flexible type for all vars-related like et
            // here needs to be T2 type

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.value().get_data());

            // TODO : need to add flexible type for all vars-related like et
            // here needs to be T2 type

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
            scale = detXd * weight;

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // compute forward projection vectors (like hforward in A2D)
        // ---------------------------------------------------------
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        {
            // goal of this section is to compute pvalue()'s of u0x, u1x, e0ty,
            // et in order to do projected Hessian reversal
            // TODO : if nonlinear, may need to recompute / store some projected
            // hessians to reverse also TODO : may need to write dot version of
            // these guys if some formulations are nonlinear (if linear same
            // method call)

            // compute the projected drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, et.pvalue().get_data());

            // compute projected directors
            T p_d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                         p_d);

            // compute tying strain projection
            T p_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                p_vars.get_data(), p_d, p_ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, p_ety,
                u0x.pvalue().get_data(), u1x.pvalue().get_data(), e0ty.pvalue());
        }

        // now we have pvalue()'s set into each in/out var => get projected
        // hessian hvalue()'s now with reverse mode AD below
        Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, e0ty, et);

        // residual backprop section (1st order derivs)
        {
            A2D::Vec<T, 3 * num_nodes> d_bar;
            T ety_bar[Basis::num_all_tying_points];
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), matCol, d_bar.get_data(), ety_bar);

            // drill strain sens
            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), matCol);

            // backprop tying strain sens ety_bar to d_bar and res
            computeTyingStrainSens<Phys, Basis>(xpts, fn, vars, d, ety_bar, matCol,
                                                d_bar.get_data());

            // directors back to residuals
            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                             matCol);

            // TODO : rotation constraint sens for some director classes (zero
            // for linear rotation)
        }  // end of 1st order deriv section

        // proj hessian backprop section (2nd order derivs)
        // -----------------------------------------------------
        {
            // TODO : make method versions of these guys for proj hessians
            // specifically (if nonlinear terms)
            A2D::Vec<T, 3 * num_nodes> d_hbar;
            T ety_hbar[Basis::num_all_tying_points];
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), e0ty.hvalue(), matCol, d_hbar.get_data(), ety_hbar);

            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);

            computeTyingStrainHrev<Phys, Basis>(xpts, fn, vars, d, ety_hbar, matCol,
                                                d_hbar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_hbar.get_data(),
                                                                             matCol);

            // TODO : rotation constraint sens for some director classes (zero
            // for linear rotation)
        }  // end of 2nd order deriv section

    }  // add_element_quadpt_gmat_col

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_adj_res_product(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T psi[dof_per_elem], T loc_dv_sens[])

    {
        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // intermediate strain states
        A2D::Mat<T, 3, 3> u0x, u1x, psi_u0x, psi_u1x;
        A2D::SymMat<T, 3> e0ty, psi_e0ty;
        A2D::Vec<T, 1> et, psi_et;
        T scale = 0.0, detXd = 0.0;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        T d[3 * num_nodes];
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.get_data(), u1x.get_data(), e0ty);

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // hforward scope block for psi[u] => psi[E] the strain level equiv adjoint
        // ------------------------------------
        {
            // compute the interpolated drill strain
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, psi, fn, psi_et.get_data());

            // compute directors (linearized Hfwd)
            T psi_d[3 * num_nodes];
            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(psi, fn, psi_d);

            // compute tying strain (hfwd version, so linearized)
            T psi_ety[Basis::num_all_tying_points];
            computeTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, psi, psi_d, psi_ety);

            // compute all shell displacement gradients (linearized Hfwd version)
            ShellComputeDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, psi, fn, psi_d, psi_ety, psi_u0x.get_data(),
                psi_u1x.get_data(), psi_e0ty);

        }  // end of hforward scope block for psi
        // ------------------------------------

        // want psi[u]^T d^2Pi/du/dx = psi[E]^T d^2Pi/dE/dx
        // instead of backprop sensitivities, hfwd and compute product on the strains
        Phys::template compute_strain_adjoint_res_product<T>(
            physData, scale, u0x, u1x, e0ty, et, psi_u0x, psi_u1x, psi_e0ty, psi_et, loc_dv_sens);

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
        const Data &physData, A2D::Mat<T, 3, 3> &u0x, A2D::Mat<T, 3, 3> &u1x,
        A2D::SymMat<T, 3> &e0ty, A2D::Vec<T, 1> &et) {
        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strains
        // ------------------------------------------------
        {
            // compute node normals fn
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            // compute the interpolated drill strain
            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            T d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, ety, u0x.get_data(), u1x.get_data(), e0ty);

        }  // end of forward scope block for strains
        // ------------------------------------------------
    }

    template <class Data>
    __HOST_DEVICE__ static void get_element_quadpt_failure_index(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        T &fail_index) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndex(physData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                           fail_index);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_dv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T loc_dv_sens[]) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndexDVSens(physData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                                 fail_sens, loc_dv_sens);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_sv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T dfdu_local[]) {
        if (!active_thread) return;

        // if (threadIdx.x == 0 && threadIdx.y == 3 && blockIdx.x == 0) {
        //     printf("xpts:");
        //     printVec<T>(12, xpts);
        //     printf("vars:");
        //     printVec<T>(24, vars);
        //     printf("rhoKS %.4e, SF %.4e\n", rhoKS, safetyFactor);
        // }

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward scope block for strain energy
        // ------------------------------------------------
        T d[3 * num_nodes];
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

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Phys::template computeFailureIndexSVSens<T>(physData, rhoKS, safetyFactor, fail_sens, u0x,
                                                    u1x, e0ty, et);

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
        // T ety_bar[Basis::num_all_tying_points];
        ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
            pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            e0ty.bvalue(), dfdu_local, d_bar.get_data(), ety_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        computeTyingStrainSens<T, Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), dfdu_local,
                                               d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                         dfdu_local);

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), dfdu_local);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T strains[vars_per_node])

    {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Vec<T, 9>> E, S;
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        T scale = 1.0;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x.value(), u1x.value(),
                                             e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(physData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Ef = E.value();
        for (int i = 0; i < 6; i++) {
            strains[i] = Ef[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_stresses(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, T stresses[vars_per_node]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // in-out of forward & backwards section
        A2D::ADObj<A2D::Vec<T, 9>> E, S;
        A2D::ADObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;
        A2D::ADObj<A2D::Vec<T, 1>> et;
        T scale = 1.0;

        // get strains and then failure index
        compute_element_quadpt_strains<Data>(iquad, xpts, vars, physData, u0x.value(), u1x.value(),
                                             e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(physData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Sf = S.value();
        for (int i = 0; i < 6; i++) {
            stresses[i] = Sf[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_mass(bool active_thread, int iquad,
                                                            const T xpts[], const Data physData,
                                                            T *output) {
        // compute int[rho * thick] dA
        if (!active_thread) return;

        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // compute shell node normals
        T fn[3 * Basis::num_nodes];  // node normals
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute detXd to transform dxideta to dA
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        T Xd[9];
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // compute area density quadpt contribution (then int across area with element sums)
        *output = weight * detXd * physData.rho * physData.thick;
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_dmass_dx(bool active_thread, int iquad,
                                                                const T xpts[], const Data physData,
                                                                T *dm_dxlocal) {
        // mass = int[rho * thick] dA summed over all elements
        // then return dmass/dx for x this element thickness
        // since one DV here just put in first
        if (!active_thread) return;

        T pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        // compute shell node normals
        T fn[3 * Basis::num_nodes];  // node normals
        ShellComputeNodeNormals<T, Basis>(xpts, fn);

        // compute detXd to transform dxideta to dA
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        T Xd[9];
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // only one local DV in isotropic shell (panel thickness)
        dm_dxlocal[0] = weight * detXd * physData.rho;
    }
};