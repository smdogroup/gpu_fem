#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_fetidp.cuh"
#include "cuda_utils.h"
#include "domdec/fetidp_assembler.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_>
class BddcSolver : public FetidpSolver<T, ShellAssembler_, Vec_, Mat_> {
    // adapted from FETI-DP solver in same folder

   public:
    using ShellAssembler = ShellAssembler_;
    using Director = typename ShellAssembler::Director;
    using Basis = typename ShellAssembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename ShellAssembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;
    using FADType = typename A2D::ADScalar<T, 1>;
    using Vec = Vec_<T>;
    using Mat = Mat_<Vec_<T>>;
    using BsrMatType = BsrMat<DeviceVec<T>>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    BddcSolver() = default;

    BddcSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
               ShellAssembler &assembler_, BsrMatType &kmat_, bool print_timing_ = false,
               bool warnings_ = true)
        : FetidpSolver<T, ShellAssembler_, Vec_, Mat_>(cublasHandle_, cusparseHandle_, assembler_,
                                                       kmat_, print_timing_) {
        warnings = warnings_;
    }

    ~BddcSolver() { this->clear_host_data(); }

    int getLambdaSize() const { return ngam * this->block_dim; }

    void setup_unstructured_subdomains(int target_sd_size = 16) {
        // call base FETI-DP setup
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_unstructured_subdomains(target_sd_size);

        // TODO: BDDC unique maps/weights for edge averaging

        // build vectors of size gam
        n_edge = this->lam_nnodes;
        this->ngam = n_edge + this->Vc_nnodes;
        gam_nodes = new int[this->ngam];

        for (int i = 0; i < n_edge; i++) {
            gam_nodes[i] = this->lam_nodes[i];
        }
        for (int i = n_edge; i < ngam; i++) {
            gam_nodes[i] = this->Vc_nodes[i - n_edge];
        }
        // printf("gam_nodes (nE %d, nV %d): ", n_edge, this->Vc_nnodes);
        // printVec<int>(ngam, gam_nodes);

        this->temp_lam = Vec(this->lam_nnodes * this->block_dim);
        this->temp_lam2 = Vec(this->lam_nnodes * this->block_dim);
    }

    void setup_structured_subdomains(int nxe_, int nye_, int nxs_, int nys_,
                                     bool close_hoop = false, bool track_dirichlet = false) {
        // call base FETI-DP setup
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_structured_subdomains(
            nxe_, nye_, nxs_, nys_, close_hoop, track_dirichlet);

        // TODO: BDDC unique maps/weights for edge averaging

        // build vectors of size gam
        n_edge = this->lam_nnodes;
        this->ngam = n_edge + this->Vc_nnodes;
        gam_nodes = new int[this->ngam];

        for (int i = 0; i < n_edge; i++) {
            gam_nodes[i] = this->lam_nodes[i];
        }
        for (int i = n_edge; i < ngam; i++) {
            gam_nodes[i] = this->Vc_nodes[i - n_edge];
        }
        // printf("gam_nodes (nE %d, nV %d): ", n_edge, this->Vc_nnodes);
        // printVec<int>(ngam, gam_nodes);

        this->temp_lam = Vec(this->lam_nnodes * this->block_dim);
        this->temp_lam2 = Vec(this->lam_nnodes * this->block_dim);
    }

    void setup_tacs_component_subdomains(int nxse_, int nyse_, int MOD_WRAPAROUND = -1,
                                         T wrap_frac = 1.0, bool track_dirichlet = false) {
        // call base FETI-DP setup
        bool compute_jump = false;
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_tacs_component_subdomains(
            nxse_, nyse_, MOD_WRAPAROUND, wrap_frac, compute_jump, track_dirichlet);

        // TODO: BDDC unique maps/weights for edge averaging
        // printf("\tdone with BDDC outer setup wing subdomains\n");

        // build vectors of size gam
        n_edge = this->lam_nnodes;
        this->ngam = n_edge + this->Vc_nnodes;
        // printf("n_edge %d, ngam %d\n", n_edge, this->ngam);
        gam_nodes = new int[this->ngam];

        for (int i = 0; i < n_edge; i++) {
            gam_nodes[i] = this->lam_nodes[i];
        }
        for (int i = n_edge; i < ngam; i++) {
            gam_nodes[i] = this->Vc_nodes[i - n_edge];
        }
        // printf("gam_nodes (nE %d, nV %d): ", n_edge, this->Vc_nnodes);
        // printVec<int>(ngam, gam_nodes);

        this->temp_lam = Vec(this->lam_nnodes * this->block_dim);
        this->temp_lam2 = Vec(this->lam_nnodes * this->block_dim);
    }

    void setup_coarse_structured_subdomains(const int nxe_, const int nye_, const int nxs_,
                                            const int nys_, const int nxs2_, const int nys2_,
                                            const bool close_hoop, int &coarse_num_elements,
                                            int &coarse_num_nodes, int &coarse_elem_nnz,
                                            int *&coarse_elem_ptr, int *&coarse_elem_conn,
                                            int *&coarse_elem_sd_ind) {
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_coarse_structured_subdomains(
            nxe_, nye_, nxs_, nys_, nxs2_, nys2_, close_hoop, coarse_num_elements, coarse_num_nodes,
            coarse_elem_nnz, coarse_elem_ptr, coarse_elem_conn, coarse_elem_sd_ind);
    }

    void setup_coarse_tacs_component_subdomains(const int nxse_, const int nyse_, const int nxse2_,
                                                const int nyse2_, const int MOD_WRAPAROUND,
                                                const T wrap_frac, int &coarse_num_elements,
                                                int &coarse_num_nodes, int &coarse_elem_nnz,
                                                int *&coarse_elem_ptr, int *&coarse_elem_conn,
                                                int *&coarse_elem_sd_ind) {
        bool compute_jump = false;
        FetidpSolver<T, ShellAssembler_, Vec_, Mat_>::setup_coasre_tacs_component_subdomains(
            nxse_, nyse_, nxse2_, nyse2_, MOD_WRAPAROUND, wrap_frac, compute_jump,
            coarse_num_elements, coarse_num_nodes, coarse_elem_nnz, coarse_elem_ptr,
            coarse_elem_conn, coarse_elem_sd_ind);
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // copy vars to d_vars (NL update)
        // printf("copyValues to d_vars\n");
        vars.copyValuesTo(this->d_vars);
        // printf("assemble subdomains\n");
        this->assemble_subdomains();
        // printf("subdomainIEsolver factor\n");
        this->subdomainIESolver->factor();
        // printf("subdomainISolver factor\n");
        this->subdomainISolver->factor();
        // printf("assemble coarse problem\n");
        this->assemble_coarse_problem();
        // equiv to factor (but more general for coarse preconditioners)
        // coasre problem doesn't need NL update (uses matrix, so d_coarse_vars always zero)
        // printf("coarse solver update after assembly\n");
        this->coarseSolver->update_after_assembly(this->d_coarse_vars);
        // this->coarseSolver->factor();
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, DeviceVec<T> vars) {
        // res_IEV(u_IEV) = lambdaE * fext_IEV - lambdaI * fint_IEV

        // temp debug
        // printf("set_IEV_residual\n");
        // T *h_xpts = this->d_xpts.createHostVec().getPtr();
        // printf("h_xpts: ");
        // printVec<T>(3 * this->num_nodes, h_xpts);
        // int *h_IE_nsd = DeviceVec<int>(this->IE_nnodes, this->d_IE_nsd).createHostVec().getPtr();
        // printf("h_IE_nsd: ");
        // printVec<int>(this->IE_nnodes, h_IE_nsd);
        // int *h_vertex_nsd =
        //     DeviceVec<int>(this->Vc_nnodes, this->d_vertex_nsd).createHostVec().getPtr();
        // printf("h_vertex_nsd: ");
        // printVec<int>(this->Vc_nnodes, h_vertex_nsd);

        this->addVec_globalToIEV(this->d_xpts, this->d_IEV_xpts, 3, 1.0, 0.0);
        this->addVec_globalToIEV(vars, this->d_IEV_vars, this->block_dim, 1.0, 0.0);

        // T *h_IEV_xpts = this->d_IEV_xpts.createHostVec().getPtr();
        // printf("h_IEV_xpts on GPU[0]: ");
        // printVec<T>(3 * this->IEV_nnodes, h_IEV_xpts);

        this->fint_IEV.zeroValues();

        dim3 block(num_quad_pts, elems_per_block);
        dim3 grid(this->num_elements);

        k_add_residual_fast<T, elems_per_block, ShellAssembler, Data, Vec_>
            <<<grid, block>>>(this->IEV_nnodes, this->num_elements, this->d_elem_components,
                              this->d_IEV_elem_conn, this->d_IEV_elem_conn, this->d_IEV_xpts,
                              this->d_IEV_vars, this->d_compData, this->fint_IEV);
        this->fint_IEV.apply_bcs(this->d_IEV_bcs);

        T a = -lambdaI;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, this->block_dim * this->IEV_nnodes, &a,
                                 this->fint_IEV.getPtr(), 1));

        this->fint_IEV.copyValuesTo(this->res_IEV);
        a = lambdaE;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, this->block_dim * this->IEV_nnodes, &a,
                                 this->fext_IEV.getPtr(), 1, this->res_IEV.getPtr(), 1));
    }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           BaseSolver *coarseSolver_, BaseSolver *subdomainIKrylov = nullptr) {
        // subdomainIKrylov matrix can be avoided only if use full fillin on K_II
        this->subdomainIESolver = subdomainIESolver_;
        this->subdomainISolver = subdomainISolver_;
        this->coarseSolver = coarseSolver_;
        this->subdomainIKrylov = subdomainIKrylov;
    }

    void get_lam_rhs(DeviceVec<T> &gam_rhs) {
        gam_rhs.zeroValues();

        // printf("FineBDDC :: pre-synchronize\n");

        this->res_IEV.copyValuesTo(this->f_IEV);
        // printf("[BDDC-get_lam_rhs]: res_IEV\n");
        // this->printDeviceNodeVec(this->IEV_nnodes, this->res_IEV.getPtr());
        // printf("[BDDC-get_lam_rhs]: f_IEV\n");
        // this->printDeviceNodeVec(this->IEV_nnodes, this->f_IEV.getPtr());

        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoI(this->f_IE, this->f_I, 1.0, 0.0);
        // this part here must use full fillin / Krylov solve on K_II subdomain parallel matrix
        // printf("[BDDC-get_lam_rhs]: f_I\n");
        // this->printDeviceNodeVec(this->I_nnodes, this->f_I.getPtr());

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("FineBDDC::get_lam_rhs - pre solveSubdomainI\n");

        this->solveSubdomainIKrylov(this->f_I, this->u_I);
        // printf("[BDDC-get_lam_rhs]: u_I\n");
        // this->printDeviceNodeVec(this->I_nnodes, this->u_I.getPtr());

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("FineBDDC :: post solveSubdomainIKrylov\n");
        // T *h_u_I = this->u_I.createHostVec().getPtr();
        // int nvals = this->u_I.getSize();
        // printf("FineBDDC::get_lam_rhs - u_I\n");
        // printVec<T>(nvals, h_u_I);

        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("FineBDDC :: pre kmat_IEV SpMV\n");
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 1.0, this->f_IEV);
        // printf("[BDDC-get_lam_rhs]: f_IEV2\n");
        // this->printDeviceNodeVec(this->IEV_nnodes, this->f_IEV.getPtr());

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("FineBDDC :: post kmat_IEV SpMV\n");

        this->addVecIEVtoGam(this->f_IEV, gam_rhs, 1.0, 0.0);
        // printf("[BDDC-get_lam_rhs]: final_gam_rhs\n");
        // this->printDeviceNodeVec(ngam, gam_rhs.getPtr());
    }

    void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        // outer mat_vec method, calls timing or non-timing version
        if (this->print_timing) {
            _mat_vec_timing(gam_in, gam_out);
        } else {
            _mat_vec(gam_in, gam_out);
        }
    }

    bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // outer mat_vec method, calls timing or non-timing version
        if (this->print_timing) {
            return _solve_timing(gam_rhs, gam, check_conv);
        } else {
            return _solve(gam_rhs, gam, check_conv);
        }
    }

    void _mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        // non-profiled method
        gam_out.zeroValues();
        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-mat_vec]: gam_in\n");
        //     this->printDeviceNodeVec(ngam, gam_in.getPtr());
        // }

        this->addVecGamtoIEV(gam_in, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, 1.0, 0.0, this->f_IEV);

        // solve rest of local Schur complement
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoI(this->f_IE, this->f_I, 1.0, 0.0);
        this->solveSubdomainI(this->f_I, this->u_I);

        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 1.0, this->f_IEV);
        this->addVecIEVtoGam(this->f_IEV, gam_out, 1.0, 0.0);
        // if (debug) {
        //     printf("[BDDC-mat_vec]: gam_out\n");
        //     this->printDeviceNodeVec(ngam, gam_out.getPtr());
        // }
    }

    bool _solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // does edge averaging (not vertex averaging since that's primal S_VV)
        const bool SCALED = true;

        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-solve]: gam_rhs\n");
        //     this->printDeviceNodeVec(ngam, gam_rhs.getPtr());
        // }

        // printf("BDDCsolve: addVecGamtoIEV\n");
        this->addVecGamtoIEV<SCALED>(gam_rhs, this->f_IEV, 1.0, 0.0);
        // if (debug) {
        //     printf("[BDDC-solve]: f_IEV\n");
        //     this->printDeviceNodeVec(this->IEV_nnodes, this->f_IEV.getPtr());
        // }

        // debug check initial V_rhs
        // printf("BDDCsolve: addVecIEVtoVc\n");
        this->template addVecIEVtoVc<SCALED>(this->f_IEV, this->f_V, 1.0, 0.0);

        // IE solve
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->f_IE, this->f_IEV, -1.0, 1.0);  // remove IE part
        this->zeroInteriorIE(this->f_IE);

        // printf("BDDCsolve: solveSubdomainIE\n");
        this->solveSubdomainIE(this->f_IE, this->u_IE);

        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);

        // coarse solve
        // printf("BDDCsolve: solveCoarse\n");
        this->addVecIEVtoVc(this->f_IEV, this->f_V, 1.0, 1.0);

        // if (debug) {
        //     printf("[BDDC-solve]: f_V\n");
        //     this->printDeviceNodeVec(this->Vc_nnodes, this->f_V.getPtr());
        // }
        this->solveCoarse(this->f_V, this->u_V);
        // if (debug) {
        //     printf("[BDDC-solve]: u_V\n");
        //     this->printDeviceNodeVec(this->Vc_nnodes, this->u_V.getPtr());
        // }

        // harmonic extension back to edge space
        this->addVecVctoIEV(this->u_V, this->temp_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->temp_IEV, -1.0, 0.0, this->f_IEV);
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->u_IE.zeroValues();

        // printf("BDDCsolve: solveSubdomainIE\n");
        this->solveSubdomainIE(this->f_IE, this->u_IE);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 1.0);
        this->template addVecVctoIEV<SCALED>(this->u_V, this->u_IEV, 1.0, 1.0);

        // now IEV to gam with averaging
        // printf("BDDCsolve: addVecIEVtoGam\n");
        this->addVecIEVtoGam<SCALED>(this->u_IEV, gam, 1.0, 0.0);

        // if (debug) {
        //     printf("[BDDC-solve]: gam\n");
        //     this->printDeviceNodeVec(ngam, gam.getPtr());
        // }

        return false;  // fail = false
    }

    void get_global_soln(DeviceVec<T> &gam, DeviceVec<T> &soln) {
        // recover global solution from interface DOF
        soln.zeroValues();
        const bool SCALED = true;

        // set IE values from interface to IEV subdomains
        this->addVecGamtoIEV(gam, this->u_IEV, 1.0, 0.0);

        // first add the solved E and V DOF into IE and V parts (no scaling 1.0)
        this->template addVecIEVtoVc<SCALED>(this->u_IEV, this->u_V, 1.0, 0.0);
        this->addVecIEVtoI(this->res_IEV, this->f_I, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);
        this->addVecIEVtoI(this->f_IEV, this->f_I, 1.0, 1.0);
        // this part here must use full fillin / Krylov solve on K_II subdomain parallel matrix
        this->solveSubdomainIKrylov(this->f_I, this->u_I);
        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);

        // add from gam the edges into u_IE
        this->addVecGamtoIEV(gam, this->u_IEV, 1.0, 0.0);
        this->addVecIEVtoIE(this->u_IEV, this->u_IE, 1.0, 1.0);

        this->addGlobalSoln(this->u_IE, this->u_V, soln);
    }

    BsrMat<DeviceVec<T>> *getKmatI() { return this->kmat_I; }
    BsrMat<DeviceVec<T>> *getCoarseSVVmat() { return this->S_VV; }
    int getInvars() { return this->I_nnodes * this->block_dim; }
    DeviceVec<T> getCoarseXpts() {
        // get vertex xpts for AMG to solve coarse BDDC problem
        T *h_xpts = this->d_xpts.createHostVec().getPtr();
        T *h_coarse_xpts = new T[3 * this->Vc_nnodes];
        for (int ivc = 0; ivc < this->Vc_nnodes; ivc++) {
            int glob_node = this->Vc_nodes[ivc];
            for (int idim = 0; idim < 3; idim++) {
                h_coarse_xpts[3 * ivc + idim] = h_xpts[3 * glob_node + idim];
            }
        }
        auto d_coarse_xpts = HostVec<T>(3 * this->Vc_nnodes, h_coarse_xpts).createDeviceVec();
        return d_coarse_xpts;
    }
    int getCoarseNumNodes() { return this->Vc_nnodes; }

   protected:
    // additional utilities
    void zeroIEinIEV(DeviceVec<T> &vec_IEV) {
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, -1.0, 1.0);
    }

    void zeroIinIEV(DeviceVec<T> &vec_IEV) {
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoI(this->temp_IE, this->temp_I, 1.0, 0.0);
        this->addVecIEtoI(this->temp_I, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, -1.0, 1.0);
    }

    void solveSubdomainIKrylov(Vec &rhs_in, Vec &sol_out) {
        auto _bsr_data = this->kmat_I->getBsrData();
        int *d_perm = _bsr_data.perm, *d_iperm = _bsr_data.iperm;
        rhs_in.copyValuesTo(this->rhs_I_perm);
        this->rhs_I_perm.permuteData(this->block_dim, d_iperm);

        if (subdomainIKrylov == nullptr) {
            if (warnings)
                printf(
                    "WARNING : must have full fillin if using non-Krylov solver for full K_II^-1 "
                    "solve\n\tIf want ILU(k) fillin, use set_inner_solvers to set subdomainIKrylov "
                    "solver.\n");
            this->subdomainISolver->solve(this->rhs_I_perm, this->sol_I_perm);
        } else {
            // do want to check convergence for this one
            subdomainIKrylov->solve(this->rhs_I_perm, this->sol_I_perm, true);
        }

        this->sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(this->block_dim, d_perm);
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(const DeviceVec<T> &vec_IEV, DeviceVec<T> &vec_gam, T alpha, T beta) {
        // add from IEV to unique EV indices (lam plus V DOF)

        int gam_size = ngam * this->block_dim;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, gam_size, &beta, vec_gam.getPtr(), 1));

        // add IEV to lam first (edge DOF)
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, alpha, 0.0);
        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-addVecIEVtoGam]: temp_IE\n");
        //     this->printDeviceNodeVec(this->IE_nnodes, this->temp_IE.getPtr());
        // }

        addVecIEtoGam(this->temp_IE, this->temp_lam, 1.0, 0.0);
        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-addVecIEVtoGam]: temp_lam\n");
        //     this->printDeviceNodeVec(this->lam_nnodes, this->temp_lam.getPtr());
        // }

        int edge_size = this->lam_nnodes * this->block_dim;
        if constexpr (SCALED) {
            dim3 block(32), grid((edge_size + 31) / 32);
            k_subdomain_normalize_vec_inout<T><<<grid, block>>>(
                this->lam_nnodes, this->block_dim, this->d_edge_nsd, this->temp_lam.getPtr());
        }

        // T a = (SCALED ? 0.5 : 1.0);
        T a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, edge_size, &a, this->temp_lam.getPtr(), 1,
                                 vec_gam.getPtr(), 1));

        // now add Vc part in next block in vector
        this->addVecIEVtoVc(vec_IEV, this->temp_V, alpha, 0.0);

        T *d_vec_gam = vec_gam.getPtr();
        a = 1.0;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, V_size, &a, this->temp_V.getPtr(), 1,
                                 &d_vec_gam[edge_size], 1));
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(DeviceVec<T> &vec_gam, DeviceVec<T> &vec_IEV, T alpha, T beta) {
        int IEV_size = this->block_dim * this->IEV_nnodes;
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, IEV_size, &beta, vec_IEV.getPtr(), 1));

        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: vec_gam\n");
        //     this->printDeviceNodeVec(ngam, vec_gam.getPtr());
        // }

        this->temp_lam.zeroValues();
        this->temp_V.zeroValues();

        // T a = (SCALED ? 0.5 : 1.0) * alpha;
        T a = alpha;
        int edge_size = this->lam_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, edge_size, &a, vec_gam.getPtr(), 1,
                                 this->temp_lam.getPtr(), 1));
        if constexpr (SCALED) {
            dim3 block(32), grid((edge_size + 31) / 32);
            k_subdomain_normalize_vec_inout<T><<<grid, block>>>(
                this->lam_nnodes, this->block_dim, this->d_edge_nsd, this->temp_lam.getPtr());
        }

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: temp_lam\n");
        //     this->printDeviceNodeVec(this->lam_nnodes, temp_lam.getPtr());
        // }

        addVecGamtoIE(this->temp_lam, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, 1.0, 0.0);

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: vec_IEV\n");
        //     this->printDeviceNodeVec(this->IEV_nnodes, vec_IEV.getPtr());
        // }

        a = alpha;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(this->cublasHandle, V_size, &a, &vec_gam.getPtr()[edge_size], 1,
                                 this->temp_V.getPtr(), 1));

        // if (debug) {
        //     printf("[BDDC-addVecGamtoIEV]: temp_V\n");
        //     this->printDeviceNodeVec(this->Vc_nnodes, this->temp_V.getPtr());
        // }

        this->addVecVctoIEV(this->temp_V, vec_IEV, 1.0, 1.0);
    }

    void addVecGamtoIE(const DeviceVec<T> &gam, DeviceVec<T> &vec_IE, T a, T b) {
        // map(a * x) + b * y => y
        CHECK_CUBLAS(cublasDscal(this->cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));
        int nvals = this->IE_nnodes * this->block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecGamtoIE<T><<<grid, block>>>(this->IE_nnodes, this->block_dim, this->d_IE_to_lam_map,
                                            gam.getPtr(), vec_IE.getPtr(), a);
    }

    void addVecIEtoGam(DeviceVec<T> &vec_IE, DeviceVec<T> &gam, T a, T b) {
        // map(a * x) + b * y => y
        // bool debug = true;
        // if (debug) {
        //     printf("[BDDC-addVecIEtoGam]: vec_IE\n");
        //     this->printDeviceNodeVec(this->IE_nnodes, vec_IE.getPtr());
        //     printf("[BDDC-addVecIEtoGam]: pre_gam\n");
        //     this->printDeviceNodeVec(ngam, gam.getPtr());
        // }

        CHECK_CUBLAS(cublasDscal(this->cublasHandle, gam.getSize(), &b, gam.getPtr(), 1));
        int nvals = this->IE_nnodes * this->block_dim;
        dim3 block(32), grid((nvals + 31) / 32);
        k_addVecIEtoGam<T><<<grid, block>>>(this->IE_nnodes, this->block_dim, this->d_IE_to_lam_map,
                                            vec_IE.getPtr(), gam.getPtr(), a);
        CHECK_CUDA(cudaDeviceSynchronize());

        // if (debug) {
        //     printf("[BDDC-addVecIEtoGam]: post_gam\n");
        //     this->printDeviceNodeVec(ngam, gam.getPtr());
        // }
    }

   protected:
    void _mat_vec_timing(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        cudaEvent_t e_start, e_stop;
        float t_gam_to_iev_and_kmat = 0.0f;
        float t_subdomainI_prep = 0.0f;
        float t_subdomainI_solve = 0.0f;
        float t_back_substitution = 0.0f;

        const bool do_timing = this->print_timing;  // or just `print_timing` if in scope

        if (do_timing) {
            CHECK_CUDA(cudaEventCreate(&e_start));
            CHECK_CUDA(cudaEventCreate(&e_stop));
        }

        gam_out.zeroValues();

        // -----------------------------------
        // 1) gam -> IEV and first kmat apply
        // -----------------------------------
        if (do_timing) CHECK_CUDA(cudaEventRecord(e_start));

        this->addVecGamtoIEV(gam_in, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, 1.0, 0.0, this->f_IEV);

        if (do_timing) {
            CHECK_CUDA(cudaEventRecord(e_stop));
            CHECK_CUDA(cudaEventSynchronize(e_stop));
            CHECK_CUDA(cudaEventElapsedTime(&t_gam_to_iev_and_kmat, e_start, e_stop));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // -----------------------------------
        // 2) prepare I solve
        // -----------------------------------
        if (do_timing) CHECK_CUDA(cudaEventRecord(e_start));

        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoI(this->f_IE, this->f_I, 1.0, 0.0);

        if (do_timing) {
            CHECK_CUDA(cudaEventRecord(e_stop));
            CHECK_CUDA(cudaEventSynchronize(e_stop));
            CHECK_CUDA(cudaEventElapsedTime(&t_subdomainI_prep, e_start, e_stop));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // -----------------------------------
        // 3) subdomain I solve
        // -----------------------------------
        if (do_timing) CHECK_CUDA(cudaEventRecord(e_start));

        this->solveSubdomainI(this->f_I, this->u_I);

        if (do_timing) {
            CHECK_CUDA(cudaEventRecord(e_stop));
            CHECK_CUDA(cudaEventSynchronize(e_stop));
            CHECK_CUDA(cudaEventElapsedTime(&t_subdomainI_solve, e_start, e_stop));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // -----------------------------------
        // 4) back-substitution and final kmat
        // -----------------------------------
        if (do_timing) CHECK_CUDA(cudaEventRecord(e_start));

        this->addVecItoIE(this->u_I, this->u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 1.0, this->f_IEV);
        this->addVecIEVtoGam(this->f_IEV, gam_out, 1.0, 0.0);

        if (do_timing) {
            CHECK_CUDA(cudaEventRecord(e_stop));
            CHECK_CUDA(cudaEventSynchronize(e_stop));
            CHECK_CUDA(cudaEventElapsedTime(&t_back_substitution, e_start, e_stop));
            CHECK_CUDA(cudaDeviceSynchronize());

            printf("mat_vec timing breakdown:\n");
            printf("\tgam->IEV + kmat_IEV             : %.6f ms\n", t_gam_to_iev_and_kmat);
            printf("\tprep subdomainI rhs            : %.6f ms\n", t_subdomainI_prep);
            printf("\tsolveSubdomainI                : %.6f ms\n", t_subdomainI_solve);
            printf("\tback-subst + kmat + IEV->gam   : %.6f ms\n", t_back_substitution);
            printf("\ttotal                          : %.6f ms\n",
                   t_gam_to_iev_and_kmat + t_subdomainI_prep + t_subdomainI_solve +
                       t_back_substitution);

            CHECK_CUDA(cudaEventDestroy(e_start));
            CHECK_CUDA(cudaEventDestroy(e_stop));
        }
    }

    bool _solve_timing(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // does edge averaging (not vertex averaging since that's primal S_VV)
        const bool SCALED = true;

        cudaEvent_t s1, e1, s2, e2, s3, e3, s4, e4, s5, e5, s6, e6;
        float t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f, t5 = 0.0f, t6 = 0.0f;

        if (this->print_timing) {
            cudaEventCreate(&s1);
            cudaEventCreate(&e1);
            cudaEventCreate(&s2);
            cudaEventCreate(&e2);
            cudaEventCreate(&s3);
            cudaEventCreate(&e3);
            cudaEventCreate(&s4);
            cudaEventCreate(&e4);
            cudaEventCreate(&s5);
            cudaEventCreate(&e5);
            cudaEventCreate(&s6);
            cudaEventCreate(&e6);
        }

        // -----------------------------------
        // 1) Initial lift/scatter from gamma
        // -----------------------------------
        if (this->print_timing) cudaEventRecord(s1);

        this->addVecGamtoIEV<SCALED>(gam_rhs, this->f_IEV, 1.0, 0.0);

        // debug check initial V_rhs
        this->template addVecIEVtoVc<SCALED>(this->f_IEV, this->f_V, 1.0, 0.0);

        // IE solve setup
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->f_IE, this->f_IEV, -1.0, 1.0);  // remove IE part
        this->zeroInteriorIE(this->f_IE);

        if (this->print_timing) {
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t1, s1, e1);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ------------------------
        // 2) First IE subdomain solve
        // ------------------------
        if (this->print_timing) cudaEventRecord(s2);

        this->solveSubdomainIE(this->f_IE, this->u_IE);

        if (this->print_timing) {
            cudaEventRecord(e2);
            cudaEventSynchronize(e2);
            cudaEventElapsedTime(&t2, s2, e2);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ------------------------
        // 3) Build and solve coarse problem
        // ------------------------
        if (this->print_timing) cudaEventRecord(s3);

        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);

        // coarse solve
        this->addVecIEVtoVc(this->f_IEV, this->f_V, 1.0, 1.0);
        this->solveCoarse(this->f_V, this->u_V);

        if (this->print_timing) {
            cudaEventRecord(e3);
            cudaEventSynchronize(e3);
            cudaEventElapsedTime(&t3, s3, e3);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ------------------------
        // 4) Harmonic extension setup back to IE
        // ------------------------
        if (this->print_timing) cudaEventRecord(s4);

        this->addVecVctoIEV(this->u_V, this->temp_IEV, 1.0, 0.0);
        this->sparseMatVec(*this->kmat_IEV, this->temp_IEV, -1.0, 0.0, this->f_IEV);
        this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
        this->u_IE.zeroValues();

        if (this->print_timing) {
            cudaEventRecord(e4);
            cudaEventSynchronize(e4);
            cudaEventElapsedTime(&t4, s4, e4);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ------------------------
        // 5) Second IE subdomain solve
        // ------------------------
        if (this->print_timing) cudaEventRecord(s5);

        this->solveSubdomainIE(this->f_IE, this->u_IE);

        if (this->print_timing) {
            cudaEventRecord(e5);
            cudaEventSynchronize(e5);
            cudaEventElapsedTime(&t5, s5, e5);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ------------------------
        // 6) Final accumulation / averaging back to gamma
        // ------------------------
        if (this->print_timing) cudaEventRecord(s6);

        this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 1.0);
        this->template addVecVctoIEV<SCALED>(this->u_V, this->u_IEV, 1.0, 1.0);

        // now IEV to gam with averaging
        this->addVecIEVtoGam<SCALED>(this->u_IEV, gam, 1.0, 0.0);

        if (this->print_timing) {
            cudaEventRecord(e6);
            cudaEventSynchronize(e6);
            cudaEventElapsedTime(&t6, s6, e6);
            CHECK_CUDA(cudaDeviceSynchronize());

            printf("BDDCSolver::solve timing breakdown:\n");
            printf("\t  gamma -> IEV/IE setup        : %.6f ms\n", t1);
            printf("\t  solveSubdomainIE #1          : %.6f ms\n", t2);
            printf("\t  coarse residual + solve      : %.6f ms\n", t3);
            printf("\t  harmonic extension setup     : %.6f ms\n", t4);
            printf("\t  solveSubdomainIE #2          : %.6f ms\n", t5);
            printf("\t  final IEV/V -> gamma         : %.6f ms\n", t6);
            printf("\t  total                        : %.6f ms\n", t1 + t2 + t3 + t4 + t5 + t6);

            cudaEventDestroy(s1);
            cudaEventDestroy(e1);
            cudaEventDestroy(s2);
            cudaEventDestroy(e2);
            cudaEventDestroy(s3);
            cudaEventDestroy(e3);
            cudaEventDestroy(s4);
            cudaEventDestroy(e4);
            cudaEventDestroy(s5);
            cudaEventDestroy(e5);
            cudaEventDestroy(s6);
            cudaEventDestroy(e6);
        }

        return false;  // fail = false
    }

    bool warnings;
    int ngam, n_edge;
    int gam_offset;
    DeviceVec<T> temp_lam, temp_lam2;
    int *gam_nodes;
    BaseSolver *subdomainIKrylov;
};