#pragma once

#include "../../assembler.h"
#include "_shell.cuh"
#include "a2d/a2dsymmatrotateframe.h"
#include "a2dcore.h"
#include "strains/_all.h"

template <typename T, class Director_, class Basis_, class Phys_, template <typename> class Vec_,
          template <typename> class Mat_>
class MITCShellAssembler
    : public ElementAssembler<MITCShellAssembler<T, Director_, Basis_, Phys_, Vec_, Mat_>, T,
                              Basis_, Phys_, Vec_, Mat_> {
   public:
    using Director = Director_;
    using Basis = Basis_;
    using Geo = typename Basis::Geo;
    using Phys = Phys_;
    using Data = typename Phys_::Data;
    using Assembler = MITCShellAssembler<T, Director_, Basis_, Phys_, Vec_, Mat_>;
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
    // static constexpr dim3 res_block = dim3(64, num_quad_pts, 1);
    static constexpr dim3 jac_block = dim3(1, dof_per_elem, num_quad_pts);
    // static constexpr dim3 jac_block = dim3(dof_per_elem, num_quad_pts);
#endif  // USE_GPU

    // default constructor
    MITCShellAssembler() = default;

    // constructor
    MITCShellAssembler(int32_t num_geo_nodes, int32_t num_vars_nodes, int32_t num_elements,
                       HostVec<int32_t> &geo_conn, HostVec<int32_t> &vars_conn, HostVec<T> &xpts,
                       HostVec<int> &bcs, HostVec<Data> &compData, int32_t num_components = 1,
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
        int nblocks =
            (this->num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        k_add_jacobian_fast<T, elems_per_block, Assembler, Data, Vec_, Mat><<<grid, block>>>(
            this->num_vars_nodes, this->num_elements, cols_per_elem, this->elem_components,
            this->geo_conn, this->vars_conn, this->xpts, this->vars, this->compData, mat);

        // CHECK_CUDA(cudaDeviceSynchronize());
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

        k_add_residual_fast<T, elems_per_block, Assembler, Data, Vec_><<<grid, block>>>(
            this->num_vars_nodes, this->num_elements, this->elem_components, this->geo_conn,
            this->vars_conn, this->xpts, this->vars, this->compData, res);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // #endif
    }

    template <class LoadMagnitude, int elems_per_block = 8>
    void add_fext_fast(const LoadMagnitude &load, T load_mag, Vec_<T> &fext,
                       T xinplane_frac = 0.0) {
        fext.zeroValues();

        dim3 block(num_quad_pts, elems_per_block);
        int nblocks = (this->num_elements + elems_per_block - 1) / elems_per_block;
        dim3 grid(nblocks);

        k_add_fext_fast<T, elems_per_block, Assembler, Data, LoadMagnitude, Vec_><<<grid, block>>>(
            this->num_elements, load, this->elem_components, this->geo_conn, this->vars_conn,
            this->xpts, this->compData, load_mag, fext, xinplane_frac);

        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    template <int elems_per_block = 1>
    void add_lockstrain_jacobian_fast(Mat &mat) {
        // fine-fine tying strain product G_f^T G_f

        mat.zeroValues();
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);
        // dim3 block(
        //     num_quad_pts,
        //     cols_per_elem,     // no num_quad_pts here, because uses tying points not num_quadpts
        //     elems_per_block);  // better order for consecutive threads and mem reads
        dim3 block(
            1, cols_per_elem,  // no num_quad_pts here, because uses tying points not num_quadpts
            elems_per_block);  // better order for consecutive threads and mem reads
        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nblocks =
            (this->num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        k_add_lockstrain_jacobian_fast<T, elems_per_block, Assembler, Data, Vec_, Mat>
            <<<grid, block>>>(this->num_vars_nodes, this->num_elements, cols_per_elem,
                              this->elem_components, this->geo_conn, this->vars_conn, this->xpts,
                              this->vars, this->compData, mat);

        CHECK_CUDA(cudaDeviceSynchronize());
        // #endif
    }

    template <int elems_per_block = 1>
    void add_lockstrain_fc_jacobian_fast(Assembler &coarse_assembler, int *d_fc_elem_map,
                                         Mat &mat) {
        // FC prolong mat product G_f^T * P_gam * G_c

        mat.zeroValues();
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);
        // dim3 block(
        //     num_quad_pts,
        //     cols_per_elem,     // no num_quad_pts here, because uses tying points not num_quadpts
        //     elems_per_block);  // better order for consecutive threads and mem reads
        dim3 block(
            1, cols_per_elem,  // no num_quad_pts here, because uses tying points not num_quadpts
            elems_per_block);  // better order for consecutive threads and mem reads
        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nblocks =
            (this->num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;
        dim3 grid(nblocks);

        auto d_c_conn = coarse_assembler.getConn();
        auto d_c_xpts = coarse_assembler.getXpts();
        auto d_c_vars = coarse_assembler.getVars();

        k_add_lockstrain_fc_jacobian_fast<T, elems_per_block, Assembler, Data, Vec_, Mat>
            <<<grid, block>>>(this->num_vars_nodes, coarse_assembler.num_nodes, this->num_elements,
                              coarse_assembler.num_elements, d_fc_elem_map, cols_per_elem,
                              this->elem_components, this->geo_conn, this->vars_conn, d_c_conn,
                              d_c_conn, this->xpts, this->vars, d_c_xpts, d_c_vars, this->compData,
                              mat);

        CHECK_CUDA(cudaDeviceSynchronize());
        // #endif
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_energy(const bool active_thread, const int iquad,
                                                          const T xpts[xpts_per_elem],
                                                          const T vars[dof_per_elem],
                                                          const Data compData, T &Uelem) {
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
                pt, compData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            T d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T XdinvT[9];
            T detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.value().get_data(),
                u1x.value().get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

            // get the scale for disp grad sens of the energy
            T scale = detXd * weight;

            // compute energy + energy-dispGrad sensitivites with physics
            Phys::template computeStrainEnergy<T>(compData, scale, u0x, u1x, e0ty, et, _Uelem);

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Uelem += _Uelem.value();
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T res[dof_per_elem]) {
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
                pt, compData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute and interp tying strain
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T XdinvT[9];
            T detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.value().get_data(),
                u1x.value().get_data());

            // rotate the tying strains
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

            // get the scale for disp grad sens of the energy
            T scale = detXd * weight;

            // compute energy + energy-dispGrad sensitivites with physics
            Phys::template computeWeakRes<T>(compData, scale, u0x, u1x, e0ty, et);

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        T XdinvT[9];
        computeBendingDispGradSens<T, vars_per_node, Basis, Data>(
            pt, compData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            XdinvT, res, d_bar.get_data());

        // transpose rotate the tying strains
        A2D::SymMat<T, 3> gty_bar;
        A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gty_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
        interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
        computeMITCTyingStrainSens<T, Phys, Basis, is_nonlinear>(
            xpts, fn, vars, d, ety_bar.get_data(), res, d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(), res);

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, compData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_mass_residual(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T accel[dof_per_elem], const Data compData, T res[dof_per_elem]) {
        if (!active_thread) return;

        T fn[3 * num_nodes];       // node normals
        T pt[2];                   // quadrature point
        T scale;                   // scale for energy derivatives
        T d_accel[3 * num_nodes];  // need director accel in reverse for nonlinear strains
        T weight = Quadrature::getQuadraturePoint(iquad, pt);

        ShellComputeNodeNormals<T, Basis>(xpts, fn);
        T detXd = getDetXd<T, Basis>(pt, xpts, fn);
        scale = weight * detXd;
        Director::template computeDirector<vars_per_node, num_nodes>(accel, fn, d_accel);

        T moments[3];
        Phys::template getMassMoments(compData, moments);

        // evaluate the second time derivatives (interpolated to the quadpt)
        T u0_accel[3], d0_accel[3];
        Basis::template interpFields<vars_per_node, 3>(pt, accel, u0_accel);
        Basis::template interpFields<3, 3>(pt, d_accel, d0_accel);

        // now backprop from kinetic energy
        A2D::Vec<T, 3> du0_accel;
        A2D::VecAddCore<T, 3>(scale * moments[0], u0_accel, du0_accel.get_data());
        A2D::VecAddCore<T, 3>(scale * moments[1], d0_accel, du0_accel.get_data());
        Basis::template interpFieldsTranspose<vars_per_node, 3>(pt, du0_accel.get_data(), res);

        A2D::Vec<T, 3> dd0_accel;
        A2D::Vec<T, 12> dd_accel;  // director accel backprop
        A2D::VecAddCore<T, 3>(scale * moments[1], u0_accel, dd0_accel.get_data());
        A2D::VecAddCore<T, 3>(scale * moments[2], d0_accel, dd0_accel.get_data());
        Basis::template interpFieldsTranspose<3, 3>(pt, dd0_accel.get_data(), dd_accel.get_data());

        // backprop through directors to residual
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, dd_accel.get_data(),
                                                                         res);
    }

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {
        // keep in mind max of ~256 floats on single thread

        if (!active_thread) return;

        // data to store in forwards + backwards section
        T fn[3 * num_nodes];  // node normals
        T pt[2];              // quadrature point
        T scale;              // scale for energy derivatives
        T d[3 * num_nodes];   // need directors in reverse for nonlinear strains
        T weight = Quadrature::getQuadraturePoint(iquad, pt);
        T XdinvT[9];

        // in-out of forward & backwards section
        A2D::A2DObj<A2D::Mat<T, 3, 3>> u0x, u1x;
        A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;
        A2D::A2DObj<A2D::Vec<T, 1>> et;
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        // forward section
        {
            ShellComputeNodeNormals<T, Basis>(xpts, fn);

            ShellComputeDrillStrain<T, vars_per_node, Data, Basis, Director>(
                pt, compData.refAxis, xpts, vars, fn, et.value().get_data());

            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // get the MITC tying strains
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // get the bending strains
            T detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.value().get_data(),
                u1x.value().get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope

        // hforward section (pvalue's)
        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;  // p_vars is unit vector for current column to compute
        T p_d[3 * num_nodes];
        {
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, compData.refAxis, xpts, p_vars.get_data(), fn, et.pvalue().get_data());

            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(p_vars.get_data(), fn,
                                                                             p_d);

            // compute forward derivs of MITC tying strains
            T p_ety[Basis::num_all_tying_points];
            computeMITCTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                       p_ety);
            A2D::SymMat<T, 3> p_gty;
            interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());

            // forward derivs of bending strains
            computeBendingDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, p_vars.get_data(), fn, p_d, XdinvT,
                u0x.pvalue().get_data(), u1x.pvalue().get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, p_gty, e0ty.pvalue());

        }  // end of hforward scope

        // derivatives over disp grad to strain energy portion
        // ---------------------
        Phys::template computeWeakJacobianCol<T>(compData, scale, u0x, u1x, e0ty, et);
        // ---------------------
        // begin reverse blocks from strain energy => physical disp grad sens

        // breverse (1st order derivs)
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
        {
            A2D::Vec<T, 3 * num_nodes> d_bar;  // zeroes out on init
            computeBendingDispGradSens<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), XdinvT, res, d_bar.get_data());

            // transpose rotate the tying strains (frame transform)
            A2D::SymMat<T, 3> gty_bar;
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                gty_bar.get_data());

            // backprop from tying strains sens
            interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
            computeMITCTyingStrainSens<T, Phys, Basis, is_nonlinear>(
                xpts, fn, vars, d, ety_bar.get_data(), res, d_bar.get_data());

            Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                             res);

            ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
                pt, compData.refAxis, xpts, vars, fn, et.bvalue().get_data(), res);

        }  // end of breverse scope (1st order derivs)

        // hreverse (2nd order derivs)
        {
            A2D::Vec<T, 3 * num_nodes> d_hat;  // zeroes out on init
            T XdinvT[9];
            computeBendingDispGradHrev<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, u0x.hvalue().get_data(),
                u1x.hvalue().get_data(), XdinvT, matCol, d_hat.get_data());

            // // transpose rotate the tying strains
            A2D::SymMat<T, 3> gty_hat;
            A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.hvalue().get_data(),
                                                gty_hat.get_data());

            // backprop from tying strains
            A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
            interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());
            computeMITCTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, p_vars.get_data(), p_d,
                                                       ety_bar.get_data(), ety_hat.get_data(),
                                                       matCol, d_hat.get_data());

            Director::template computeDirectorHrev<vars_per_node, num_nodes>(fn, d_hat.get_data(),
                                                                             matCol);

            ShellComputeDrillStrainHrev<T, vars_per_node, Data, Basis, Director>(
                pt, compData.refAxis, xpts, vars, fn, et.hvalue().get_data(), matCol);
        }  // end of hreverse scope (2nd order derivs)
    }      // add_element_quadpt_jacobian_col

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

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void
    add_element_lockstrain_jacobian_col_fast_v1(  // __noinline__ is slower actually
        const T pt[2], const T &scale, const T xpts[xpts_per_elem], const T fn[xpts_per_elem],
        const T XdinvT[9], const T Tmat[9], const T XdinvzT[9], const Data &compData,
        const T vars[dof_per_elem], const T pvars[dof_per_elem], T matCol[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            T ety[Basis::num_all_tying_points];
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);
                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
                A2D::SymMat<T, 3> gty;
                interpTyingStrain<T, Basis>(pt, ety, gty.get_data());
                A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
                computeEngineerTyingStrains<T>(e0ty.value());
                __syncthreads();
            }

            // pforward section
            // -------------------------------
            T p_d[3 * num_nodes];
            T p_ety[Basis::num_all_tying_points];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(pvars, fn, p_d);
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
                Phys::computeIdentityTyingStress(scale, e0ty.value(), e0ty.bvalue());
                // Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, ety,
                //                                      ety_bar.get_data());

                computeEngineerTyingStrains<T>(e0ty.bvalue());
                A2D::SymMat<T, 3> gty_bar;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                                                    gty_bar.get_data());
                interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
                // __syncthreads();
            }

            // 2nd order hrev
            {
                Phys::computeIdentityTyingStress(scale, e0ty.pvalue(), e0ty.hvalue());

                computeEngineerTyingStrains<T>(e0ty.hvalue());
                A2D::SymMat<T, 3> gty_hat;
                A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.hvalue().get_data(),
                                                    gty_hat.get_data());
                A2D::Vec<T, 3 * num_nodes> d_hat;
                A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
                interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());
                // Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, p_ety,
                //                                      ety_hat.get_data());
                computeMITCTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, pvars, p_d,
                                                           ety_bar.get_data(), ety_hat.get_data(),
                                                           matCol, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    fn, d_hat.get_data(), matCol);
            }
        }

    }  // add_lockstrain_jacobian_col

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void
    add_element_lockstrain_fc_jacobian_row_fast_v1(  // __noinline__ is slower actually
        const T pt[2], const T &scale, const T fine_xpts[xpts_per_elem],
        const T coarse_xpts[xpts_per_elem], const T fine_fn[xpts_per_elem],
        const T coarse_fn[xpts_per_elem], const T fine_XdinvT[9], const T fine_Tmat[9],
        const T fine_XdinvzT[9], const T coarse_XdinvT[9], const T coarse_Tmat[9],
        const T coarse_XdinvzT[9], const Data &compData, const T fine_vars[dof_per_elem],
        const T coarse_vars[dof_per_elem], const T pvars[dof_per_elem], T matRow[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;
        // should be FINE is like p_vars and the input side, COARSE DOF and xpts for output side

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T fine_d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            T ety[Basis::num_all_tying_points];
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(fine_vars, fine_fn,
                                                                             fine_d);
                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(fine_xpts, fine_fn, fine_vars,
                                                                     fine_d, ety);
                A2D::SymMat<T, 3> gty;
                interpTyingStrain<T, Basis>(pt, ety, gty.get_data());
                A2D::SymMatRotateFrame<T, 3>(fine_XdinvT, gty, e0ty.value());
                computeEngineerTyingStrains<T>(e0ty.value());
                __syncthreads();
            }

            T coarse_d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(coarse_vars, coarse_fn,
                                                                             coarse_d);
            }

            // pforward section
            // -------------------------------
            T p_d[3 * num_nodes];
            T p_ety[Basis::num_all_tying_points];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(pvars, fine_fn,
                                                                                 p_d);

                computeMITCTyingStrainHfwd<T, Phys, Basis>(fine_xpts, fine_fn, fine_vars, fine_d,
                                                           pvars, p_d, p_ety);
                A2D::SymMat<T, 3> p_gty;
                interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());
                A2D::SymMatRotateFrame<T, 3>(fine_XdinvT, p_gty, e0ty.pvalue());
                computeEngineerTyingStrains<T>(e0ty.pvalue());
            }
            // __syncthreads();

            // 1st order brev
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
            if constexpr (is_nonlinear) {
                // Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, ety,
                //                                      ety_bar.get_data());

                Phys::computeIdentityTyingStress(scale, e0ty.value(), e0ty.bvalue());
                computeEngineerTyingStrains<T>(e0ty.bvalue());
                A2D::SymMat<T, 3> gty_bar;
                A2D::SymMat3x3RotateFrameReverse<T>(coarse_XdinvT, e0ty.bvalue().get_data(),
                                                    gty_bar.get_data());
                interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
                __syncthreads();
            }

            // 2nd order hrev
            {
                Phys::computeIdentityTyingStress(scale, e0ty.pvalue(), e0ty.hvalue());
                computeEngineerTyingStrains<T>(e0ty.hvalue());
                A2D::SymMat<T, 3> gty_hat;
                A2D::SymMat3x3RotateFrameReverse<T>(coarse_XdinvT, e0ty.hvalue().get_data(),
                                                    gty_hat.get_data());
                A2D::Vec<T, 3 * num_nodes> d_hat;
                A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
                interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());

                // Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, p_ety,
                //                                      ety_hat.get_data());
                computeMITCTyingStrainHrev<T, Phys, Basis>(
                    coarse_xpts, coarse_fn, coarse_vars, coarse_d, pvars, p_d, ety_bar.get_data(),
                    ety_hat.get_data(), matRow, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    coarse_fn, d_hat.get_data(), matRow);
            }
        }

    }  // add_element_lockstrain_fc_jacobian_row_fast

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void
    add_element_lockstrain_jacobian_col_fast_v2(  // __noinline__ is slower actually
        const T pt[2], const T &scale, const T xpts[xpts_per_elem], const T fn[xpts_per_elem],
        const T XdinvT[9], const T Tmat[9], const T XdinvzT[9], const Data &compData,
        const T vars[dof_per_elem], const T pvars[dof_per_elem], T matCol[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to
            // energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            T ety[Basis::num_all_tying_points];
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);
                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
                // A2D::SymMat<T, 3> gty;
                // interpTyingStrain<T, Basis>(pt, ety, gty.get_data());
                // A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());
                // computeEngineerTyingStrains<T>(e0ty.value());
                // __syncthreads();
            }

            // pforward section
            // -------------------------------
            T p_d[3 * num_nodes];
            T p_ety[Basis::num_all_tying_points];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(pvars, fn, p_d);

                computeMITCTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, pvars, p_d, p_ety);

                // TEMP process for plate linear case: converts to physical coords
                // normally needs XdinvT^T * strains * XdinvT (and with g_{13} uses XdinvT_{11} and
                // XdinvT_{33} in this product where XdinvT_{33} = 1 so still works)
                p_ety[5] *= XdinvT[4];
                p_ety[6] *= XdinvT[4];
                p_ety[7] *= XdinvT[0];
                p_ety[8] *= XdinvT[0];

                // A2D::SymMat<T, 3> p_gty;
                // interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());
                // A2D::SymMatRotateFrame<T, 3>(XdinvT, p_gty, e0ty.pvalue());
                // computeEngineerTyingStrains<T>(e0ty.pvalue());

                // DEBUG: check the tying strains strain-disp matrix
                // if (blockIdx.x == 0) {
                //     int trv_shear[4] = {5, 6, 7, 8};
                //     for (int i = 0; i < 4; i++) {
                //         int j = trv_shear[i];
                //         printf("ideriv %d, trv shear %d => %.4e\n", threadIdx.y, i, p_ety[j]);
                //     }
                // }
            }
            // __syncthreads();

            // 1st order brev
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
            if constexpr (is_nonlinear) {
                // Phys::computeIdentityTyingStress(scale, e0ty.value(), e0ty.bvalue());
                Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, ety,
                                                     ety_bar.get_data());

                //     computeEngineerTyingStrains<T>(e0ty.bvalue());
                // A2D::SymMat<T, 3> gty_bar;
                // A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(),
                //                                     gty_bar.get_data());
                // interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(),
                // ety_bar.get_data();
                // // __syncthreads();
            }

            // 2nd order hrev
            {
                // Phys::computeIdentityTyingStress(scale, e0ty.pvalue(), e0ty.hvalue());

                // computeEngineerTyingStrains<T>(e0ty.hvalue());
                // A2D::SymMat<T, 3> gty_hat;
                // A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.hvalue().get_data(),
                //                                     gty_hat.get_data());
                A2D::Vec<T, 3 * num_nodes> d_hat;
                A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
                // interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());
                Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, p_ety,
                                                     ety_hat.get_data());
                // TEMP process for plate linear case: converts to physical coords
                ety_hat[5] *= XdinvT[4];
                ety_hat[6] *= XdinvT[4];
                ety_hat[7] *= XdinvT[0];
                ety_hat[8] *= XdinvT[0];
                computeMITCTyingStrainHrev<T, Phys, Basis>(xpts, fn, vars, d, pvars, p_d,
                                                           ety_bar.get_data(), ety_hat.get_data(),
                                                           matCol, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    fn, d_hat.get_data(), matCol);
            }
        }

    }  // add_lockstrain_jacobian_col

    template <class Data, STRAIN strain = ALL>
    __DEVICE__ static void
    add_element_lockstrain_fc_jacobian_row_fast_v2(  // __noinline__ is slower actually
        const T pt[2], const T &scale, const T fine_xpts[xpts_per_elem],
        const T coarse_xpts[xpts_per_elem], const T fine_fn[xpts_per_elem],
        const T coarse_fn[xpts_per_elem], const T fine_XdinvT[9], const T fine_Tmat[9],
        const T fine_XdinvzT[9], const T coarse_XdinvT[9], const T coarse_Tmat[9],
        const T coarse_XdinvzT[9], const Data &compData, const T fine_vars[dof_per_elem],
        const T coarse_vars[dof_per_elem], const T pvars[dof_per_elem], T matRow[dof_per_elem]) {
        constexpr bool bending = strain == BENDING || strain == ALL;
        constexpr bool tying = strain == TYING || strain == ALL;
        constexpr bool drill = strain == DRILL || strain == ALL;
        // should be FINE is like p_vars and the input side, COARSE DOF and xpts for output side

        // data to store in forwards + backwards section
        static constexpr bool is_nonlinear = Phys::is_nonlinear;

        if constexpr (tying) {
            // // TODO : only need 1st order obj not 2nd order here since e0ty is linear to
            // energy
            // // nonlinear part of tying strains happens in earlier step before e0ty
            A2D::A2DObj<A2D::SymMat<T, 3>> e0ty;

            // forward section
            // --------------------------------
            T fine_d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            T ety[Basis::num_all_tying_points];
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(fine_vars, fine_fn,
                                                                             fine_d);

                computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(fine_xpts, fine_fn, fine_vars,
                                                                     fine_d, ety);
                // A2D::SymMat<T, 3> gty;
                // interpTyingStrain<T, Basis>(pt, ety, gty.get_data());
                // A2D::SymMatRotateFrame<T, 3>(fine_XdinvT, gty, e0ty.value());
                // computeEngineerTyingStrains<T>(e0ty.value());
                // __syncthreads();
            }

            T coarse_d[3 * num_nodes];  // need directors in reverse for nonlinear strains
            if constexpr (is_nonlinear) {
                Director::template computeDirector<vars_per_node, num_nodes>(coarse_vars, coarse_fn,
                                                                             coarse_d);
            }

            // pforward section
            // -------------------------------
            T p_d[3 * num_nodes];
            T p_ety[Basis::num_all_tying_points];
            {
                Director::template computeDirectorHfwd<vars_per_node, num_nodes>(pvars, fine_fn,
                                                                                 p_d);

                computeMITCTyingStrainHfwd<T, Phys, Basis>(fine_xpts, fine_fn, fine_vars, fine_d,
                                                           pvars, p_d, p_ety);
                // A2D::SymMat<T, 3> p_gty;
                // interpTyingStrain<T, Basis>(pt, p_ety, p_gty.get_data());
                // A2D::SymMatRotateFrame<T, 3>(fine_XdinvT, p_gty, e0ty.pvalue());
                // computeEngineerTyingStrains<T>(e0ty.pvalue());

                // TEMP process for plate linear case: converts to physical coords
                // normally needs XdinvT^T * strains * XdinvT (and with g_{13} uses XdinvT_{11} and
                // XdinvT_{33} in this product where XdinvT_{33} = 1 so still works)
                p_ety[5] *= fine_XdinvT[4];
                p_ety[6] *= fine_XdinvT[4];
                p_ety[7] *= fine_XdinvT[0];
                p_ety[8] *= fine_XdinvT[0];
            }
            // __syncthreads();

            // 1st order brev
            A2D::Vec<T, Basis::num_all_tying_points> ety_bar;  // zeroes out on init
            if constexpr (is_nonlinear) {
                Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, ety,
                                                     ety_bar.get_data());

                // Phys::computeIdentityTyingStress(scale, e0ty.value(), e0ty.bvalue());
                // computeEngineerTyingStrains<T>(e0ty.bvalue());
                // A2D::SymMat<T, 3> gty_bar;
                // A2D::SymMat3x3RotateFrameReverse<T>(coarse_XdinvT, e0ty.bvalue().get_data(),
                //                                     gty_bar.get_data());
                // interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
                // __syncthreads();
            }

            // 2nd order hrev
            {
                // Phys::computeIdentityTyingStress(scale, e0ty.pvalue(), e0ty.hvalue());
                // computeEngineerTyingStrains<T>(e0ty.hvalue());
                // A2D::SymMat<T, 3> gty_hat;
                // A2D::SymMat3x3RotateFrameReverse<T>(coarse_XdinvT, e0ty.hvalue().get_data(),
                //                                     gty_hat.get_data());
                A2D::Vec<T, 3 * num_nodes> d_hat;
                A2D::Vec<T, Basis::num_all_tying_points> ety_hat;  // zeroes out on init
                // interpTyingStrainTranspose<T, Basis>(pt, gty_hat.get_data(), ety_hat.get_data());

                Phys::computeIdentityTyingPtStresses(scale, Basis::num_all_tying_points, p_ety,
                                                     ety_hat.get_data());
                // TEMP process for plate linear case: converts to physical coords
                ety_hat[5] *= coarse_XdinvT[4];
                ety_hat[6] *= coarse_XdinvT[4];
                ety_hat[7] *= coarse_XdinvT[0];
                ety_hat[8] *= coarse_XdinvT[0];
                computeMITCTyingStrainHrev<T, Phys, Basis>(
                    coarse_xpts, coarse_fn, coarse_vars, coarse_d, pvars, p_d, ety_bar.get_data(),
                    ety_hat.get_data(), matRow, d_hat.get_data());

                Director::template computeDirectorHrev<vars_per_node, num_nodes>(
                    coarse_fn, d_hat.get_data(), matRow);
            }
        }

    }  // add_element_lockstrain_fc_jacobian_row_fast

    template <class Data>
    __HOST_DEVICE__ static void add_element_quadpt_mass_jacobian_col(
        const bool active_thread, const int iquad, const int ivar, const T xpts[xpts_per_elem],
        const T accel[dof_per_elem], const Data compData, T res[dof_per_elem],
        T matCol[dof_per_elem]) {
        // since it's linear, it should be very similar to the residual, just that we're doing
        // projected hessians, so you need to do the residual on a p_vars input basically

        A2D::Vec<T, dof_per_elem> p_vars;
        p_vars[ivar] = 1.0;
        // this should be equivalent to assuming KE = 1/2 accel^T M acce (global and for each
        // element) resid = M * accel and to get a column of M you can plug in a cartesian basis vec
        // to the residual, resid(ej) = M * ej

        add_element_quadpt_mass_residual(active_thread, iquad, xpts, accel, compData,
                                         res);  // take this out later in speedup assembly kernels
        add_element_quadpt_mass_residual(active_thread, iquad, xpts, p_vars.get_data(), compData,
                                         matCol);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_adj_res_product(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, const T psi[dof_per_elem], T loc_dv_sens[])

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
                pt, compData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T XdinvT[9];
            detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.get_data(), u1x.get_data());

            // rotate the tying strains
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty);

            // get the scale for disp grad sens of the energy
            scale = detXd * weight;

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        // hforward scope block for psi[u] => psi[E] the strain level equiv adjoint
        // ------------------------------------
        {
            // compute the interpolated drill strain
            ShellComputeDrillStrainHfwd<T, vars_per_node, Data, Basis, Director>(
                pt, compData.refAxis, xpts, psi, fn, psi_et.get_data());

            // compute directors (linearized Hfwd)
            T psi_d[3 * num_nodes];
            Director::template computeDirectorHfwd<vars_per_node, num_nodes>(psi, fn, psi_d);

            // compute tying strain (hfwd version, so linearized)
            T psi_ety[Basis::num_all_tying_points];
            computeMITCTyingStrainHfwd<T, Phys, Basis>(xpts, fn, vars, d, psi, psi_d, psi_ety);
            A2D::SymMat<T, 3> psi_gty;
            interpTyingStrain<T, Basis>(pt, psi_ety, psi_gty.get_data());

            // compute all shell displacement gradients (linearized Hfwd version)
            T XdinvT[9];
            computeBendingDispGradHfwd<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, psi, fn, psi_d, XdinvT, psi_u0x.get_data(),
                psi_u1x.get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, psi_gty, psi_e0ty);

        }  // end of hforward scope block for psi
        // ------------------------------------

        // want psi[u]^T d^2Pi/du/dx = psi[E]^T d^2Pi/dE/dx
        // instead of backprop sensitivities, hfwd and compute product on the strains
        Phys::template compute_strain_adjoint_res_product<T>(
            compData, scale, u0x, u1x, e0ty, et, psi_u0x, psi_u1x, psi_e0ty, psi_et, loc_dv_sens);

    }  // end of method add_element_quadpt_residual

    template <class Data>
    __HOST_DEVICE__ static void _compute_element_quadpt_strains(
        const int iquad, const T xpts[xpts_per_elem], const T vars[dof_per_elem],
        const Data &compData, A2D::Mat<T, 3, 3> &u0x, A2D::Mat<T, 3, 3> &u1x,
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
                pt, compData.refAxis, xpts, vars, fn, et.get_data());

            // compute directors
            T d[3 * num_nodes];
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T XdinvT[9];
            T detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.get_data(), u1x.get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty);

        }  // end of forward scope block for strains
        // ------------------------------------------------
    }

    template <class Data>
    __HOST_DEVICE__ static void get_element_quadpt_failure_index(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
        T &fail_index) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        // get strains and then failure index
        _compute_element_quadpt_strains<Data>(iquad, xpts, vars, compData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndex(compData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                           fail_index);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_dv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
        const T &fail_sens, T loc_dv_sens[]) {
        if (!active_thread) return;

        // in-out of forward & backwards section
        A2D::Mat<T, 3, 3> u0x, u1x;
        A2D::SymMat<T, 3> e0ty;
        A2D::Vec<T, 1> et;

        _compute_element_quadpt_strains<Data>(iquad, xpts, vars, compData, u0x, u1x, e0ty, et);

        Phys::template computeFailureIndexDVSens(compData, u0x, u1x, e0ty, et, rhoKS, safetyFactor,
                                                 fail_sens, loc_dv_sens);
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_failure_sv_sens(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data &compData, const T &rhoKS, const T &safetyFactor,
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
                pt, compData.refAxis, xpts, vars, fn, et.value().get_data());

            // compute directors
            Director::template computeDirector<vars_per_node, num_nodes>(vars, fn, d);

            // compute tying strain
            T ety[Basis::num_all_tying_points];
            computeMITCTyingStrain<T, Phys, Basis, is_nonlinear>(xpts, fn, vars, d, ety);
            A2D::SymMat<T, 3> gty;
            interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

            // compute all shell displacement gradients
            T XdinvT[9];
            T detXd = computeBendingDispGrad<T, vars_per_node, Basis, Data>(
                pt, compData.refAxis, xpts, vars, fn, d, XdinvT, u0x.value().get_data(),
                u1x.value().get_data());

            // rotate the tying strains with XdinvT frame
            A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty.value());

        }  // end of forward scope block for strain energy
        // ------------------------------------------------

        Phys::template computeFailureIndexSVSens<T>(compData, rhoKS, safetyFactor, fail_sens, u0x,
                                                    u1x, e0ty, et);

        // beginning of backprop section to final residual derivatives
        // -----------------------------------------------------

        // compute disp grad sens u0x_bar, u1x_bar, e0ty_bar => res, d_bar,
        // ety_bar
        A2D::Vec<T, 3 * num_nodes> d_bar;
        T XdinvT[9];
        computeBendingDispGradSens<T, vars_per_node, Basis, Data>(
            pt, compData.refAxis, xpts, vars, fn, u0x.bvalue().get_data(), u1x.bvalue().get_data(),
            XdinvT, dfdu_local, d_bar.get_data());

        // transpose rotate the tying strains
        A2D::SymMat<T, 3> gty_bar;
        A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty.bvalue().get_data(), gty_bar.get_data());

        // backprop tying strain sens ety_bar to d_bar and res
        A2D::Vec<T, Basis::num_all_tying_points> ety_bar;
        interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar.get_data());
        computeMITCTyingStrainSens<T, Phys, Basis, is_nonlinear>(
            xpts, fn, vars, d, ety_bar.get_data(), dfdu_local, d_bar.get_data());

        // directors back to residuals
        Director::template computeDirectorSens<vars_per_node, num_nodes>(fn, d_bar.get_data(),
                                                                         dfdu_local);

        // drill strain sens
        ShellComputeDrillStrainSens<T, vars_per_node, Data, Basis, Director>(
            pt, compData.refAxis, xpts, vars, fn, et.bvalue().get_data(), dfdu_local);

        // TODO : rotation constraint sens for some director classes (zero for
        // linear rotation)
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_strains(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T strains[vars_per_node])

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
        _compute_element_quadpt_strains<Data>(iquad, xpts, vars, compData, u0x.value(), u1x.value(),
                                              e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(compData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Ef = E.value();
        for (int i = 0; i < 6; i++) {
            strains[i] = Ef[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_stresses(
        const bool active_thread, const int iquad, const T xpts[xpts_per_elem],
        const T vars[dof_per_elem], const Data compData, T stresses[vars_per_node]) {
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
        _compute_element_quadpt_strains<Data>(iquad, xpts, vars, compData, u0x.value(), u1x.value(),
                                              e0ty.value(), et.value());

        // compute energy + energy-dispGrad sensitivites with physics
        Phys::template computeQuadptStresses<T>(compData, scale, u0x, u1x, e0ty, et, E, S);

        // now copy strains out
        A2D::Vec<T, 9> &Sf = S.value();
        for (int i = 0; i < 6; i++) {
            stresses[i] = Sf[i];
        }
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_mass(bool active_thread, int iquad,
                                                            const T xpts[], const Data compData,
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
        assembleFrame<T>(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // compute area density quadpt contribution (then int across area with element sums)
        *output = weight * detXd * compData.rho * compData.thick;
    }

    template <class Data>
    __HOST_DEVICE__ static void compute_element_quadpt_dmass_dx(bool active_thread, int iquad,
                                                                const T xpts[], const Data compData,
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
        assembleFrame<T>(Xxi, Xeta, n0, Xd);

        // compute detXd finally for jacobian dA conversion
        T detXd = A2D::MatDetCore<T, 3>(Xd);

        // only one local DV in isotropic shell (panel thickness)
        dm_dxlocal[0] = weight * detXd * compData.rho;
    }
};