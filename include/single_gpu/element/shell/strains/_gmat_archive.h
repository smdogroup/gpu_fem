#pragma once
#include "../../../cuda_utils.h"

// archive file for analytic Gmatrix.. TBD

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
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, d, gty, u0x.value().get_data(),
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
            computeMITCTyingStrainHfwd<Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                p_vars.get_data(), p_d, p_ety);
            A2D::SymMat<T, 3> p_gty;
            interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());

            // compute all shell displacement gradients
            T detXd = ShellComputeDispGrad<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, p_vars.get_data(), fn, p_d, p_gty,
                u0x.pvalue().get_data(), u1x.pvalue().get_data(), e0ty.pvalue());
        }

        // now we have pvalue()'s set into each in/out var => get projected
        // hessian hvalue()'s now with reverse mode AD below
        Phys::template computeWeakJacobianCol<T>(physData, scale, u0x, u1x, e0ty, et);

        // residual backprop section (1st order derivs)
        {
            A2D::Vec<T, 3 * num_nodes> d_bar;
            A2D::SymMat<T, 3> gty_bar;
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue(), matCol, d_bar.get_data(), gty_bar);

            // drill strain sens
            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.bvalue().get_data(), matCol);

            // backprop tying strain sens ety_bar to d_bar and res
            A2D::Vec<T, Basis::num_all_tying_points>  ety_bar;
            interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
            computeMITCTyingStrainSens<Phys, Basis>(xpts, fn, vars, d, ety_bar.get_data(), matCol,
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
            A2D::SymMat<T, 3> gty_hbar;
            ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(
                pt, physData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), e0ty.hvalue(), matCol, d_hbar.get_data(), gty_hbar);

            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, physData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);

            A2D::Vec<T, Basis::num_all_tying_points>  ety_hbar;
            interpTyingStrainTranspose<T, Basis>(pt, gty_hbar.get_data(), ety_hbar.get_data());
            computeMITCTyingStrainHrev<Phys, Basis>(xpts, fn, vars, d, ety_hbar.get_data(), matCol,
                                                d_hbar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_hbar.get_data(),
                                                                             matCol);

            // TODO : rotation constraint sens for some director classes (zero
            // for linear rotation)
        }  // end of 2nd order deriv section

    }  // add_element_quadpt_gmat_col