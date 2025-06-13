#pragma once

#include "a2dcore.h"
#include "shell_utils.h"

template <typename T, class Director_, class Basis_, class Phys_, bool full_strain_ = true>
class ShellElementGroupV2 {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using ElemGroup = ShellElementGroupV2<T, Director_, Basis_, Phys_>;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;

    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * Geo::num_nodes;
    static constexpr int32_t dof_per_elem = Phys::vars_per_node * Basis::num_nodes;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    static constexpr int32_t num_nodes = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr bool full_strain = full_strain_;

// TODO : way to make this more general if num_quad_pts is not a multiple of 3?
// some if constexpr stuff on type of Basis?
#ifdef USE_GPU
    static constexpr dim3 energy_block = dim3(32, num_quad_pts, 1);
    static constexpr dim3 res_block = dim3(num_quad_pts, 32, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    template <class Data>
    __HOST_DEVICE__ static void compute_nodal_shell_transforms(const bool active_thread,
                                                               const int inode,
                                                               const T xpts[xpts_per_elem],
                                                               const Data physData, T Tmat[9],
                                                               T XdinvT[9]) {
        if (!active_thread) return;

        ShellComputeNodalTransforms<T, Data, Basis>(inode, xpts, physData, Tmat, XdinvT);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_quadpt_shell_transforms(const bool active_thread,
                                                                const int iquad,
                                                                const T xpts[xpts_per_elem],
                                                                const Data physData, T Tmat[9],
                                                                T XdinvT[9], T XdinvzT[9]) {
        if (!active_thread) return;

        ShellComputeQuadptTransforms<T, Data, Basis, Quadrature>(iquad, xpts, physData, Tmat,
                                                                 XdinvT, XdinvzT);
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_drill_strain_quadpt_residual_fast(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[36], const T XdinvT[36],
        T res[dof_per_elem]) {
        /*
        new fast drill strain residual
        goal: all methods use shared memory or registers only, no local memory
        I will then go back and see if I can make A2D more GPU friendly at some point..
        Trying to us epre-computed Tmat, XdinvT to use less registers here..
        */

        if (!active_thread) return;

        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        A2D::ADObj<A2D::Vec<T, 1>> et;

        // compute the interpolated drill strain
        {
            ShellComputeDrillStrainFast<T, vars_per_node, Basis, Director>(
                quad_pt, xpts, vars, Tmat, XdinvT, et.value().get_data()[0]);
        }

        // need to get scale = detXd * weight somehow
        T detXd = Basis::getDetXd(quad_pt, xpts);
        T scale = detXd * weight;

        // backprop from strain energy to et
        Phys::template compute_drill_strain_grad<T>(physData, scale, et);

        // // backprop from drill strain to residual
        ShellComputeDrillStrainSensFast<T, vars_per_node, Basis, Director>(
            quad_pt, xpts, vars, Tmat, XdinvT, et.bvalue().get_data(), res);

        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("\tinside drill strain resid\n");
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_tying_strain_quadpt_residual_fast(
        bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &physData, T XdinvT[9], T res[dof_per_elem]) {
        if (!active_thread) return;

        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        A2D::ADObj<A2D::SymMat<T, 3>> e0ty;

        // forward scope block
        {
            // computes and interps tying strains in oneshot, faster than split up
            A2D::SymMat<T, 3> gty;
            computeInterpTyingStrainFast<T, Phys, Basis, Director>(quad_pt, xpts, vars,
                                                                   gty.get_data());

            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
        }

        // backprop from strain energy to tying strain gradient in physics
        T detXd = Basis::getDetXd(quad_pt, xpts);
        T scale = detXd * weight;

        Phys::template compute_tying_strain_midplane_grad<T>(physData, scale, e0ty);
        Phys::template compute_tying_strain_transverse_grad<T>(physData, scale, e0ty);

        // reverse scope block
        {
            A2D::SymMat<T, 3> gty_bar;
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                gty_bar.get_data());

            computeInterpTyingStrainFastSens<T, Phys, Basis, Director>(quad_pt, xpts, vars,
                                                                       gty_bar.get_data(), res);
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void _add_bending_strain_quadpt_residual_fast(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data physData, const T Tmat[9], const T XdinvT[9],
        const T XdinvzT[9], T res[]) {
        if (!active_thread) return;

        // prelim
        T quad_pt[2];
        T weight = Quadrature::getQuadraturePoint(iquad, quad_pt);
        A2D::ADObj<A2D::Vec<T, 3>> ek;

        // forward scope block
        T u0x[9], u1x[9];  // needed for backprop (should become registers hopefully)
        computeBendingStrains<T, vars_per_node, Basis, Director, Phys::is_nonlinear>(
            quad_pt, xpts, vars, Tmat, XdinvT, XdinvzT, u0x, u1x, ek.value().get_data());

        // now compute the strains to stresses
        T detXd = Basis::getDetXd(quad_pt, xpts);
        T scale = detXd * weight;
        Phys::template compute_bending_strain_grad<T>(physData, scale, ek);

        // reverse scope block
        computeBendingStrainSens<T, vars_per_node, Basis, Director, Phys::is_nonlinear>(
            quad_pt, xpts, vars, Tmat, XdinvT, XdinvzT, ek.bvalue().get_data(), u0x, u1x, res);
    }
};