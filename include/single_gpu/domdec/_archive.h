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
    // adapted from FETI-DP assembler in same folder

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
               ShellAssembler &assembler_, BsrMatType &kmat_, bool print_timing_ = false)
        : FetidpSolver<T, ShellAssembler_, Vec_, Mat_>(cublasHandle_, cusparseHandle_, assembler_,
                                                       kmat_, print_timing_) {}

    ~BddcSolver() { this->clear_host_data(); }
    int getLambdaSize() const { return this->lam_nnodes * this->block_dim; }

    void setup_structured_subdomains(int nxe_, int nye_, int nxs_, int nys_,
                                     bool close_hoop = false) {
        // call FETI-DP method as well (does this work as subcall)
        this->setup_structured_subdomains(nxe_, nye_, nxs_, nys_, close_hoop);

        // TODO : BDDC unique maps and weights, edge averaging part..
        // maybe don't do it here, do 0.5 for now later (and for wings will have to change this code
        // more)

        // build vectors of size gam (edge non-repeated + vertices non-repeated, gam size) and
        // maps.. or just store concatenated form..
        n_edge = this->nlam;
        ngam = n_edge + this->Vc_nnodes;

        temp_lam = Vec(this->lam_nnodes * block_dim);
        temp_lam2 = Vec(this->lam_nnodes * block_dim);
    }

    void get_interface_rhs(DeviceVec<T> &gam_rhs) {
        // must use this-> to get data and methods from subclass
        gam_rhs.zeroValues();
        res_IEV.copyValuesTo(f_IEV);
        this->addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        this->addVecIEtoI(f_IE, f_I, 1.0, 0.0);
        this->solveSubdomainI(f_I, u_I);
        this->addVecItoIE(u_I, u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        zeroIinIEV(f_IEV);  // f_I removed from IEV
        this->sparseMatVec(*kmat_IEV, u_IEV, -1.0, 1.0, f_IEV);
        addVecIEVtoGam(f_IEV, gam_rhs, 1.0, 0.0);
    }

    void mat_vec(const DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        gam_out.zeroValues();

        addVecGamtoIEV(gam_in, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);

        // solve rest of local Schur complement
        this->addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        this->addVecIEtoI(f_IE, f_I, 1.0, 0.0);
        this->solveSubdomainI(f_I, u_I);
        this->addVecItoIE(u_I, u_IE, 1.0, 0.0);
        this->addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        this->sparseMatVec(*kmat_IEV, u_IEV, -1.0, 1.0, f_IEV);
        this->addVecIEVtoGam(f_IEV, gam_out, 1.0, 0.0);
    }

    bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // gam.zeroValues();
        // does edge averaging (not vertex averaging cause that's primal S_VV)
        const bool SCALED = true;

        // similar to FETI-DP mat_vec (flipped), but a bit different
        addVecGamtoIEV<SCALED>(gam_rhs, f_IEV, 1.0, 0.0);

        // IE solve
        this->addVecIEVtoIE(gam_rhs, f_IE, 1.0, 0.0);
        this->addVecIEtoIEV(f_IE, f_IEV, -1.0, 1.0);  // remove IE part
        this->zeroInterior(f_IE);
        this->solveSubdomainIE(f_IE, u_IE);
        this->addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        this->sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);

        // coarse solve
        this->addVecIEVtoVc(f_IEV, f_V, 1.0, 0.0);
        this->solveCoarse(f_V, u_V);

        // harmonic extension back to edge space (keeping previous edge terms
        this->addVecVctoIEV(u_V, temp_IEV, 1.0, 0.0);
        this->sparseMatVec(*kmat_IEV, temp_IEV, f_IEV, 1.0, 1.0);
        this->addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        this->solveSubdomainIE(f_IE, u_IE);
        this->addVecIEtoIEV(u_IE, u_IEV, -1.0, 1.0);

        // now IEV to lam with averaging
        addVecIEVtoGam<SCALED>(u_IEV, gam, 1.0, 0.0);
        return false;  // fail = false
    }

    void get_global_soln(const DeviceVec<T> &gam, DeviceVec<T> &soln) {
        // recover global solution from interface DOF
        soln.zeroValues();

        // set IE values from interface to IEV subdomains
        addVecGamtoIEV(gam_in, u_IEV, 1.0, 0.0);
        this->addVecIEVtoVc(u_IEV, u_V, 1.0, 0.0);
        this->addVecIEVtoIE(res_IEV, f_IE, 1.0, 0.0);  // add I residual in only
        this->zeroInteriorIE(f_IE);                    // remove edge part of loads
        this->sparseMatVec(*kmat_IEV, temp_IEV, f_IEV, -1.0, 0.0);
        this->addVecIEVtoIE(f_IEV, f_IE, 1.0, 1.0);
        this->addVecIEtoI(f_IE, f_I, 1.0, 0.0);
        this->solveSubdomainI(f_I, u_I);
        this->addVecItoIE(u_I, u_IE, 1.0, 0.0);
        this->addGlobalSoln(u_IE, u_V, soln);
    }

   private:
    // additional utilities
    void zeroIEinIEV(DeviceVec<T> &vec_IEV) {  // use other util functions to accomplish
        this->addVecIEVtoIE(vec_IEV, temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(temp_IE, vec_IEV, -1.0, 1.0);
    }

    void zeroIinIEV(DeviceVec<T> &vec_IEV) {  // use other util functions to accomplish
        this->addVecIEVtoIE(vec_IEV, temp_IE, 1.0, 0.0);
        zeroInteriorIE(temp_IE);
        this->addVecIEtoIEV(temp_IE, vec_IEV, -1.0, 1.0);
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(DeviceVec<T> &vec_IEV, DeviceVec<T> &vec_gam, T alpha, T beta) {
        // add from IEV to just EV unique indices (like lam plus V DOF)
        // TODO : need new maps for this, shouldn't be too hard, can reuse some of lam for it.. and
        // then just concatenate Vertex V DOF

        // FOR now just scale edge parts by 1/2 when SCALED = true (FUTURE WORK with wings will
        // generalize)
        // SCALED is kept as template arg (cause might call different code paths later)

        // add IEV to lam first (edge DOF)
        this->addVecIEVtoIE(vec_IEV, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoLam(this->temp_IE, temp_la, 1.0, 0.0);
        T a = SCALED ? 0.5 : 1.0;
        int edge_size = this->lam_nnodes * this->block_dim;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, edge_size, &a, temp_lam.getPtr(), 1, vec_gam.getPtr(), 1));

        // now add Vc part in next block in vector (concatenated together)
        this->addVecIEVtoVc(vec_IEV, this->temp_V, 1.0,
                            0.0);  // no repeats here on either vec, so no scaling
        a = 1.0;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, V_size, &a, this->temp_V.getPtr(), 1,
                                 &vec_gam.getPtr()[lam_size], 1));
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(DeviceVec<T> &vec_gam, DeviceVec<T> &vec_IEV, T alpha, T beta) {
        // add from IEV to just EV unique indices (like lam plus V DOF)

        // add lam part of gam to IEV first (edge DOF)

        T a = SCALED ? 0.5 : 1.0;
        int edge_size = this->lam_nnodes * this->block_dim;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, edge_size, &a, vec_gam.getPtr(), 1, temp_lam.getPtr(), 1));

        this->addVecLamtoIE(temp_lam, this->temp_IE, 1.0, 0.0);
        this->addVecIEtoIEV(this->temp_IE, vec_IEV, 1.0, 0.0);

        // Vc from gam to Vc then to IEV
        a = 1.0;
        int V_size = this->Vc_nnodes * this->block_dim;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, V_size, &a, &vec_gam.getPtr()[lam_size], 1,
                                 this->temp_V.getPtr(), 1, , 1));
        this->addVecVctoIEV(this->temp_V, vec_IEV, 1.0,
                            1.0);  // no repeats here on either vec, so no scaling
    }

    // private data
   private:
    int ngam, n_edge;
    int gam_offset;
    DeviceVec<T> temp_lam, temp_lam2;
};

void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
    gam_out.zeroValues();

    // const T *h_gam_in = gam_in.createHostVec().getPtr();
    // printf("h_gam_in_mat_vec:\n");
    // for (int ilam = 0; ilam < ngam; ilam++) {
    //     int iglob = gam_nodes[ilam];
    //     printf("igam %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_gam_in[lam_dof]);
    //     }
    //     printf("\n");
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

    // const T *h_gam_soln = gam_out.createHostVec().getPtr();
    // printf("h_gam_out_mat_vec:\n");
    // for (int ilam = 0; ilam < ngam; ilam++) {
    //     int iglob = gam_nodes[ilam];
    //     printf("igam %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_gam_soln[lam_dof]);
    //     }
    //     printf("\n");
    // }
}

bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
    // does edge averaging (not vertex averaging since that's primal S_VV)
    const bool SCALED = true;

    // const T *h_gam_rhs = gam_rhs.createHostVec().getPtr();
    // printf("h_gam_rhs-pc:\n");
    // for (int ilam = 0; ilam < ngam; ilam++) {
    //     int iglob = gam_nodes[ilam];
    //     printf("igam %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_gam_rhs[lam_dof]);
    //     }
    //     printf("\n");
    // }

    // similar to FETI-DP mat_vec (flipped), but a bit different
    this->addVecGamtoIEV<SCALED>(gam_rhs, this->f_IEV, 1.0, 0.0);

    // const T *h_fIEV = this->f_IEV.createHostVec().getPtr();
    // printf("h_fIEV-pc with #IEV = %d:\n", this->IEV_nnodes);
    // for (int ilam = 0; ilam < this->IEV_nnodes; ilam++) {
    //     int iglob = this->IEV_nodes[ilam];
    //     printf("iIEV %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_fIEV[lam_dof]);
    //     }
    //     printf("\n");
    // }

    // debug check initial V_rhs
    // for vertices in rectangular part (will need to change this for wing case here)
    // TODO : change this part for wing case here..
    // this->addVecIEVtoVc(this->f_IEV, this->f_V, 0.25, 0.0);
    this->template addVecIEVtoVc<SCALED>(this->f_IEV, this->f_V, 1.0, 0.0);

    // IE solve
    this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);

    // const T *h_fIE0 = this->f_IE.createHostVec().getPtr();
    // printf("h_fIE0-pc with #IE = %d:\n", this->IE_nnodes);
    // for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    //     int iglob = this->IE_nodes[ilam];
    //     printf("iIE %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_fIE0[lam_dof]);
    //     }
    //     printf("\n");
    // }

    this->addVecIEtoIEV(this->f_IE, this->f_IEV, -1.0, 1.0);  // remove IE part
    this->zeroInteriorIE(this->f_IE);

    // const T *h_fIE = this->f_IE.createHostVec().getPtr();
    // printf("h_fIE-pc with #IE = %d:\n", this->IE_nnodes);
    // for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    //     int iglob = this->IE_nodes[ilam];
    //     printf("iIE %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_fIE[lam_dof]);
    //     }
    //     printf("\n");
    // }

    this->solveSubdomainIE(this->f_IE, this->u_IE);

    // const T *h_uIE = this->u_IE.createHostVec().getPtr();
    // printf("h_uIE-pc:\n");
    // for (int ilam = 0; ilam < this->IE_nnodes; ilam++) {
    //     int iglob = this->IE_nodes[ilam];
    //     printf("iIE %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_uIE[lam_dof]);
    //     }
    //     printf("\n");
    // }

    this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 0.0);
    this->sparseMatVec(*this->kmat_IEV, this->u_IEV, -1.0, 0.0, this->f_IEV);

    // coarse solve
    this->addVecIEVtoVc(this->f_IEV, this->f_V, 1.0, 1.0);
    this->solveCoarse(this->f_V, this->u_V);

    // harmonic extension back to edge space
    this->addVecVctoIEV(this->u_V, this->temp_IEV, 1.0, 0.0);
    this->sparseMatVec(*this->kmat_IEV, this->temp_IEV, -1.0, 0.0, this->f_IEV);
    this->addVecIEVtoIE(this->f_IEV, this->f_IE, 1.0, 0.0);
    this->u_IE.zeroValues();

    this->solveSubdomainIE(this->f_IE, this->u_IE);
    this->addVecIEtoIEV(this->u_IE, this->u_IEV, 1.0, 1.0);

    // add u_V into u_IEV
    // TODO : generalize better than 0.25 here (for wing case)
    // this->addVecVctoIEV(this->u_V, this->u_IEV, 0.25, 1.0);
    this->template addVecVctoIEV<SCALED>(this->u_V, this->u_IEV, 1.0, 1.0);

    // now IEV to gam with averaging
    this->addVecIEVtoGam<SCALED>(this->u_IEV, gam, 1.0, 0.0);

    // const T *h_gam = gam.createHostVec().getPtr();
    // printf("h_gam-pc:\n");
    // for (int ilam = 0; ilam < ngam; ilam++) {
    //     int iglob = gam_nodes[ilam];
    //     printf("igam %d, glob node %d: ", ilam, iglob);
    //     for (int idof = 2; idof < 5; idof++) {
    //         int lam_dof = this->block_dim * ilam + idof;
    //         printf("%.6e,", h_gam[lam_dof]);
    //     }
    //     printf("\n");
    // }

    return false;  // fail = false
}