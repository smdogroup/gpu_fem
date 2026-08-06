#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "_fetidp.cuh"
#include "cuda_utils.h"
#include "domdec/bddc_assembler.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "multigrid/amg/fake_assembler.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_, class CoarseSolver>
class CoarseBddcSolver
    : public BddcSolver<T, FakeAssembler<T, ShellAssembler_>, Vec_, Mat_, CoarseSolver> {
    // adapted from BDDC

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
    using FAssembler = FakeAssembler<T, ShellAssembler>;
    using FineBDDC = BddcSolver<T, FAssembler, Vec_, Mat_, CoarseSolver>;

   public:
    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    CoarseBddcSolver() = default;

    CoarseBddcSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                     FAssembler &fassembler, BsrMatType &kmat_, int elem_nnz_, int *h_elem_ptr_,
                     int *h_elem_conn_, int *h_elem_sd_ind_, bool print_timing_ = false,
                     bool warnings_ = true, int MAX_NUM_VERTEX_PER_SUBDOMAIN_ = 6)
        : FineBDDC(cublasHandle_, cusparseHandle_, fassembler, kmat_, print_timing_, warnings_) {
        // auto _bsr_data = kmat_.getBsrData();
        // auto fake_assembler = FAssembler(_bsr_data, num_nodes_, num_elements_);
        // FineBDDC(cublasHandle_, cusparseHandle_, fake_assembler, kmat_, print_timing_,
        // warnings_);

        // more general element connectivity here.. for wing cases
        this->is_fine_bddc = false;
        this->elem_nnz = elem_nnz_;
        this->elem_ptr = h_elem_ptr_;
        this->elem_conn = h_elem_conn_;
        this->elem_sd_ind = h_elem_sd_ind_;  // subdomain splitting (solved on fine mesh rn)

        // get num subdomains
        std::unordered_set<int> unique_sd;
        for (int i = 0; i < this->num_elements; i++) {
            int i_subdomain = this->elem_sd_ind[i];
            unique_sd.insert(i_subdomain);
        }
        this->num_subdomains = unique_sd.size();

        // default max num vertex per subdomain (for plate + wing problems)
        this->MAX_NUM_VERTEX_PER_SUBDOMAIN = MAX_NUM_VERTEX_PER_SUBDOMAIN_;

        // since we have elem_sd_ind, this is doable now immediately
        // and no Dirichlet nodes on coarse problem (with current problems)
        printf("setup subdomains in coarseBddc constructor\n");
        setup_subdomains();
        printf("\tsetup subdomains in coarseBddc constructor\n");
    }

    ~CoarseBddcSolver() { this->clear_host_data(); }

    void update_after_assembly(DeviceVec<T> &vars) {
        // TBD make this method work
        assemble_coarse_subdomains();  // instead of regular assemble subdomains
        this->subdomainIESolver->factor();
        this->subdomainISolver->factor();
        this->assemble_coarse_problem();
        // equiv to factor (but more general for coarse preconditioners)
        // coasre problem doesn't need NL update (uses matrix, so d_coarse_vars always zero)
        this->coarseSolver->update_after_assembly(this->d_coarse_vars);
        // this->coarseSolver->factor();
    }

    template <int elems_per_block = 8>
    void set_IEV_residual(T lambdaE, T lambdaI, DeviceVec<T> vars) {
        // TBD make this method work for Coarse BDDC
    }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           CoarseSolver *coarseSolver_, BaseSolver *subdomainIKrylov = nullptr) {
        // subdomainIKrylov matrix can be avoided only if use full fillin on K_II
        this->subdomainIESolver = subdomainIESolver_;
        this->subdomainISolver = subdomainISolver_;
        this->coarseSolver = coarseSolver_;
        this->subdomainIKrylov = subdomainIKrylov;
    }
    // void get_lam_rhs(DeviceVec<T> &gam_rhs) {}
    // void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {}
    // bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {}
    // void get_global_soln(DeviceVec<T> &gam, DeviceVec<T> &soln) {}

    void get_IEV_nodes(int &_IEV_nnodes, int *&_IEV_nodes) {
        printf("IEV nodes from coarse BDDC with #nodes=%d and #IEV_nodes=%d\n", this->num_nodes,
               this->IEV_nnodes);
        printVec<int>(this->IEV_nnodes, this->IEV_nodes);
        _IEV_nnodes = this->IEV_nnodes;
        _IEV_nodes = this->IEV_nodes;
    }

    void setup_matrix_sparsity() {
        // USER must call this routine..
        printf(
            "NOTE : FETI-DP doesn't support permutations yet on subdomains.. TBD later on "
            "that\n");
        printf(
            "\tJust does full fillin currently of each matrix used for inner linear "
            "solves\n");

        // do fillin of IE and I matrices (later also do coarse matrix)
        this->IE_rowp = this->IE_bsr_data.rowp;
        this->IE_cols = this->IE_bsr_data.cols;
        this->IE_nnzb = this->IE_bsr_data.nnzb;
        this->IE_perm = this->IE_bsr_data.perm;
        this->IE_iperm = this->IE_bsr_data.iperm;

        // host
        this->I_rowp = this->I_bsr_data.rowp;
        this->I_cols = this->I_bsr_data.cols;
        this->I_nnzb = this->I_bsr_data.nnzb;
        this->I_perm = this->I_bsr_data.perm;
        this->I_iperm = this->I_bsr_data.iperm;

        // recompute rows after potential fillin
        delete[] this->IE_rows;
        delete[] this->I_rows;
        this->IE_rows = new int[this->IE_nnzb];
        this->I_rows = new int[this->I_nnzb];

        for (int inode = 0; inode < this->IE_nnodes; inode++) {
            for (int jp = this->IE_rowp[inode]; jp < this->IE_rowp[inode + 1]; jp++) {
                this->IE_rows[jp] = inode;
            }
        }
        for (int inode = 0; inode < this->I_nnodes; inode++) {
            for (int jp = this->I_rowp[inode]; jp < this->I_rowp[inode + 1]; jp++) {
                this->I_rows[jp] = inode;
            }
        }
        this->IE_bsr_data.rows = this->IE_rows;
        this->I_bsr_data.rows = this->I_rows;

        // now move sparsity to the device
        // and no fillin for IEV matrix cause it isn't used for linear solves
        this->IEV_bsr_data.rows = this->IEV_rows;
        this->d_IEV_bsr_data = this->IEV_bsr_data.createDeviceBsrData();
        this->d_IEV_vals = DeviceVec<T>(this->block_dim2 * this->IEV_nnzb);
        this->kmat_IEV = new BsrMatType(this->d_IEV_bsr_data, this->d_IEV_vals);

        this->d_IE_bsr_data = this->IE_bsr_data.createDeviceBsrData();
        this->d_IE_vals = DeviceVec<T>(this->block_dim2 * this->IE_nnzb);
        this->kmat_IE = new BsrMatType(this->d_IE_bsr_data, this->d_IE_vals);

        this->d_I_bsr_data = this->I_bsr_data.createDeviceBsrData();
        this->d_I_vals = DeviceVec<T>(this->block_dim2 * this->I_nnzb);
        this->kmat_I = new BsrMatType(this->d_I_bsr_data, this->d_I_vals);

        this->d_IE_perm = this->d_IE_bsr_data.perm;
        this->d_IE_iperm = this->d_IE_bsr_data.iperm;
        this->d_I_perm = this->d_I_bsr_data.perm;
        this->d_I_iperm = this->d_I_bsr_data.iperm;

        // -----------------------------------------
        // IEV => IE kmat block copy map
        // -----------------------------------------

        this->kmat_IEnofill_map = new int[this->IE_nofill_nnzb];
        this->kmat_IEtoIEV_map = new int[this->IE_nofill_nnzb];
        memset(this->kmat_IEnofill_map, -1, this->IE_nofill_nnzb * sizeof(int));
        memset(this->kmat_IEtoIEV_map, -1, this->IE_nofill_nnzb * sizeof(int));

        int nofill_ind = 0;
        for (int i = 0; i < this->IE_nnodes; i++) {
            int i_perm = this->IE_iperm[i];  // VIS to solve order
            for (int jp = this->IE_rowp[i_perm]; jp < this->IE_rowp[i_perm + 1]; jp++) {
                int j_perm = this->IE_cols[jp];
                int j = this->IE_perm[j_perm];  // solve to VIS order

                // find equivalent nz block of IEV rowp
                int i_IEV = this->IEVtoIE_imap[i];
                int j_IEV = this->IEVtoIE_imap[j];

                for (int kp = this->IEV_rowp[i_IEV]; kp < this->IEV_rowp[i_IEV + 1]; kp++) {
                    int k = this->IEV_cols[kp];
                    if (k == j_IEV) {
                        this->kmat_IEnofill_map[nofill_ind] = jp;
                        this->kmat_IEtoIEV_map[nofill_ind] = kp;
                        nofill_ind++;
                    }
                }
            }
        }

        this->d_kmat_IEtoIEV_map =
            HostVec<int>(this->IE_nofill_nnzb, this->kmat_IEtoIEV_map).createDeviceVec().getPtr();
        this->d_kmat_IEnofill_map =
            HostVec<int>(this->IE_nofill_nnzb, this->kmat_IEnofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV => I kmat block copy map
        // -----------------------------------------

        this->kmat_Inofill_map = new int[this->I_nofill_nnzb];
        this->kmat_ItoIEV_map = new int[this->I_nofill_nnzb];
        memset(this->kmat_Inofill_map, -1, this->I_nofill_nnzb * sizeof(int));
        memset(this->kmat_ItoIEV_map, -1, this->I_nofill_nnzb * sizeof(int));

        nofill_ind = 0;
        for (int i = 0; i < this->I_nnodes; i++) {
            int i_perm = this->I_iperm[i];  // VIS to solve order
            for (int jp = this->I_rowp[i_perm]; jp < this->I_rowp[i_perm + 1]; jp++) {
                int j_perm = this->I_cols[jp];
                int j = this->I_perm[j_perm];

                // find equivalent nz block of IEV rowp
                int i_IEV = this->IEVtoI_imap[i];
                int j_IEV = this->IEVtoI_imap[j];

                for (int kp = this->IEV_rowp[i_IEV]; kp < this->IEV_rowp[i_IEV + 1]; kp++) {
                    int k = this->IEV_cols[kp];
                    if (k == j_IEV) {
                        this->kmat_Inofill_map[nofill_ind] = jp;
                        this->kmat_ItoIEV_map[nofill_ind] = kp;
                        nofill_ind++;
                    }
                }
            }
        }

        this->d_kmat_ItoIEV_map =
            HostVec<int>(this->I_nofill_nnzb, this->kmat_ItoIEV_map).createDeviceVec().getPtr();
        this->d_kmat_Inofill_map =
            HostVec<int>(this->I_nofill_nnzb, this->kmat_Inofill_map).createDeviceVec().getPtr();

        // -----------------------------------------
        // get S_VV matrix sparsity / nonzero pattern (nofill first)
        // -----------------------------------------

        // reverse map of global => reduced Vc nodes
        this->Vc_node_imap = new int[this->num_nodes];
        memset(this->Vc_node_imap, -1, this->num_nodes * sizeof(int));
        for (int vnode = 0; vnode < this->Vc_nnodes; vnode++) {
            int glob_node = this->Vc_nodes[vnode];
            this->Vc_node_imap[glob_node] = vnode;
        }

        // build unique adjacency per coarse row
        std::vector<std::unordered_set<int>> Svv_adj(this->Vc_nnodes);

        for (int i_subdomain = 0; i_subdomain < this->num_subdomains; i_subdomain++) {
            std::unordered_set<int> sd_Vc_nodeset;

            for (int ielem = 0; ielem < this->num_elements; ielem++) {
                int j_subdomain = this->elem_sd_ind[ielem];
                if (i_subdomain != j_subdomain) {
                    continue;
                }

                for (int elemp = this->elem_ptr[ielem]; elemp < this->elem_ptr[ielem + 1];
                     elemp++) {
                    int gnode = this->elem_conn[elemp];
                    int node_class = this->node_class_ind[gnode];

                    if (node_class == VERTEX) {
                        int vnode = this->Vc_node_imap[gnode];
                        if (vnode >= 0) {
                            sd_Vc_nodeset.insert(vnode);
                        }
                    }
                }
            }

            std::vector<int> sd_Vc_nodes(sd_Vc_nodeset.begin(), sd_Vc_nodeset.end());

            // add unique couplings for this subdomain
            for (int i : sd_Vc_nodes) {
                for (int j : sd_Vc_nodes) {
                    Svv_adj[i].insert(j);
                }
            }
        }

        // row counts from unique adjacency
        int *Svv_rowcts = new int[this->Vc_nnodes];
        memset(Svv_rowcts, 0, this->Vc_nnodes * sizeof(int));
        for (int i = 0; i < this->Vc_nnodes; i++) {
            Svv_rowcts[i] = static_cast<int>(Svv_adj[i].size());
        }

        // fill rowp
        this->Svv_rowp = new int[this->Vc_nnodes + 1];
        memset(this->Svv_rowp, 0, (this->Vc_nnodes + 1) * sizeof(int));
        for (int i = 0; i < this->Vc_nnodes; i++) {
            this->Svv_rowp[i + 1] = this->Svv_rowp[i] + Svv_rowcts[i];
        }
        this->Svv_nnzb = this->Svv_rowp[this->Vc_nnodes];

        // fill cols
        this->Svv_cols = new int[this->Svv_nnzb];
        memset(this->Svv_cols, 0, this->Svv_nnzb * sizeof(int));

        for (int i = 0; i < this->Vc_nnodes; i++) {
            int jp = this->Svv_rowp[i];
            for (int j : Svv_adj[i]) {
                this->Svv_cols[jp++] = j;
            }

            std::sort(&this->Svv_cols[this->Svv_rowp[i]], &this->Svv_cols[this->Svv_rowp[i + 1]]);
        }

        this->Svv_rows = new int[this->Svv_nnzb];
        for (int i = 0; i < this->Vc_nnodes; i++) {
            for (int jp = this->Svv_rowp[i]; jp < this->Svv_rowp[i + 1]; jp++) {
                this->Svv_rows[jp] = i;
            }
        }

        delete[] Svv_rowcts;

        printf("Svv_rowp with nnzb %d: ", this->Svv_nnzb);
        printVec<int>(this->Vc_nnodes + 1, this->Svv_rowp);
        printf("Svv_cols: ");
        printVec<int>(this->Svv_nnzb, this->Svv_cols);

        // -----------------------------------------
        // build Svv matrix sparsity and do fillin
        // -----------------------------------------

        this->Svv_bsr_data = BsrData(this->Vc_nnodes, this->block_dim, this->Svv_nnzb,
                                     this->Svv_rowp, this->Svv_cols);
        this->Svv_bsr_data.rows = this->Svv_rows;
        this->Svv_nofill_nnzb = this->Svv_nnzb;
        this->SVV_perm = this->Svv_bsr_data.perm;
        this->SVV_iperm = this->Svv_bsr_data.iperm;
    }

    void set_global_rhs(DeviceVec<T> &rhs) {
        // somehow need to take global RHS and redistribute it to the IEV nodes (with inverse
        // scaling)
        const bool scaled = true;
        this->template addVec_globalToIEV<scaled>(rhs, this->fext_IEV, this->block_dim, 1.0, 0.0);
        // fext_IEV.apply_bcs(d_IEV_bcs);
        // no BCs in coarse BDDC yet (and for all cases we do, not yet)
        this->fext_IEV.copyValuesTo(this->res_IEV);
    }

    void get_lam_rhs(DeviceVec<T> &gam_rhs) {
        // printf("coarse BDDC get_lam_rhs\n");
        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("post synchronize in CBDDC::get_lam_rhs\n");

        FineBDDC::get_lam_rhs(gam_rhs);

        // CHECK_CUDA(cudaDeviceSynchronize());
        // T *h_gam_rhs = gam_rhs.createHostVec().getPtr();
        // int nvals = gam_rhs.getSize();
        // printf("coarseBddc : h_gam_rhs\n");
        // printVec<T>(nvals, h_gam_rhs);
    }

    void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        // printf("coarse BDDC mat-vec\n");
        FineBDDC::mat_vec(gam_in, gam_out);
    }
    bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        // printf("coarse BDDC solve\n");
        bool fail = FineBDDC::solve(gam_rhs, gam, check_conv);
        // printf("\tdone with coarse BDDC solve\n");
        return fail;
    }
    void get_global_soln(DeviceVec<T> &gam, DeviceVec<T> &soln) {
        FineBDDC::get_global_soln(gam, soln);
    }

    BsrMat<DeviceVec<T>> *getKmatI() { return this->kmat_I; }
    int getInvars() { return this->I_nnodes * this->block_dim; }
    BsrMat<DeviceVec<T>> *getKmatIEV() { return this->kmat_IEV; }
    BsrMat<DeviceVec<T>> *getSVVmat() { return this->S_VV; }
    int *getIEVnodes() { return this->IEV_nodes; }
    int *getIEVsdPtr() { return this->IEV_sd_ptr; }
    int *getIEVsdInd() { return this->IEV_sd_ind; }

    void assemble_coarse_subdomains() {
        // assemble subdomains without assembly of S_VV (or this fine grid matrix)
        // since that comes from higher level

        // copy entries from kmat_IEV to kmat_IE, kmat_I and S_VV matrices
        this->kmat_IE->zeroValues();
        this->kmat_I->zeroValues();
        this->copyKmat_IEVtoIE();
        this->copyKmat_IEVtoI();
    }

   private:
    // private helper methods

    void setup_subdomains() {
        // build subdomains from an existing elem_sd_ind splitting
        //  the elem_sd_ind splitting is built from a hierarchical splitting on fine mesh

        // -----------------------------------------
        // node -> subdomain incidence
        // -----------------------------------------
        this->node_elem_nnz = 0;
        this->node_elem_rowp = new int[this->num_nodes + 1];
        this->node_elem_ct = new int[this->num_nodes];
        std::memset(this->node_elem_rowp, 0, (this->num_nodes + 1) * sizeof(int));
        std::memset(this->node_elem_ct, 0, this->num_nodes * sizeof(int));

        for (int ielem = 0; ielem < this->num_elements; ielem++) {
            for (int elemp = elem_ptr[ielem]; elemp < elem_ptr[ielem + 1]; elemp++) {
                int gnode = this->elem_conn[elemp];
                this->node_elem_ct[gnode]++;
                this->node_elem_nnz++;
            }
        }

        for (int inode = 0; inode < this->num_nodes; inode++) {
            this->node_elem_rowp[inode + 1] =
                this->node_elem_rowp[inode] + this->node_elem_ct[inode];
        }

        int *temp_node_elem = new int[this->num_nodes];
        this->node_sd_cols = new int[this->node_elem_nnz];
        std::memset(temp_node_elem, 0, this->num_nodes * sizeof(int));
        std::memset(this->node_sd_cols, 0, this->node_elem_nnz * sizeof(int));

        for (int ielem = 0; ielem < this->num_elements; ielem++) {
            int subdomain_ind = this->elem_sd_ind[ielem];
            for (int elemp = this->elem_ptr[ielem]; elemp < this->elem_ptr[ielem + 1]; elemp++) {
                int gnode = this->elem_conn[elemp];
                int offset = this->node_elem_rowp[gnode] + temp_node_elem[gnode];
                this->node_sd_cols[offset] = subdomain_ind;
                temp_node_elem[gnode]++;
            }
        }
        delete[] temp_node_elem;

        printf("elem_conn: ");
        printVec<int>(this->elem_nnz, this->elem_conn);

        // -----------------------------------------
        // classify nodes
        // -----------------------------------------
        this->node_class_ind = new int[this->num_nodes];
        std::memset(this->node_class_ind, 0, this->num_nodes * sizeof(int));

        this->node_nsd = new int[this->num_nodes];
        std::memset(this->node_nsd, 0, this->num_nodes * sizeof(int));
        this->I_nnodes = 0, this->IE_nnodes = 0, this->IEV_nnodes = 0;
        this->Vc_nnodes = 0, this->V_nnodes = 0, this->lam_nnodes = 0;

        for (int inode = 0; inode < this->num_nodes; inode++) {
            std::unordered_set<int> node_sds;
            for (int jp = this->node_elem_rowp[inode]; jp < this->node_elem_rowp[inode + 1]; jp++) {
                node_sds.insert(this->node_sd_cols[jp]);
            }

            int nsd = static_cast<int>(node_sds.size());
            this->node_nsd[inode] = nsd;

            // no dirichlet nodes so just based on nsd (classification)
            if (nsd < 2) {
                this->node_class_ind[inode] = INTERIOR;
                this->I_nnodes++, this->IE_nnodes++, this->IEV_nnodes++;
            } else if (nsd == 2) {
                this->node_class_ind[inode] = EDGE;
                this->lam_nnodes++;
                this->IE_nnodes += nsd;
                this->IEV_nnodes += nsd;
            } else {
                this->node_class_ind[inode] = VERTEX;
                this->Vc_nnodes++;  //, lam_nnodes++;
                this->V_nnodes += nsd;
                this->IEV_nnodes += nsd;
            }
        }
        // printf("node_nsd: ");
        // printVec<int>(this->num_nodes, this->node_nsd);
        // printf("node_class_ind: ");
        // printVec<int>(this->num_nodes, this->node_class_ind);
        // printf("IEV_nnodes %d\n", this->IEV_nnodes);
        // printf("num subdomains %d\n", this->num_subdomains);

        // -----------------------------------------
        // build duplicated IEV nodal layout
        // -----------------------------------------
        this->IEV_sd_ptr = new int[this->num_subdomains + 1];
        this->IEV_sd_ind = new int[this->IEV_nnodes];
        this->IEV_nodes = new int[this->IEV_nnodes];

        std::memset(this->IEV_sd_ptr, 0, (this->num_subdomains + 1) * sizeof(int));

        int IEV_ind = 0;
        int *temp_completion = new int[this->num_nodes];

        for (int i_subdomain = 0; i_subdomain < this->num_subdomains; i_subdomain++) {
            std::memset(temp_completion, 0, this->num_nodes * sizeof(int));
            this->IEV_sd_ptr[i_subdomain + 1] = this->IEV_sd_ptr[i_subdomain];

            for (int ielem = 0; ielem < this->num_elements; ielem++) {
                if (this->elem_sd_ind[ielem] != i_subdomain) continue;

                for (int elemp = this->elem_ptr[ielem]; elemp < this->elem_ptr[ielem + 1];
                     elemp++) {
                    int gnode = this->elem_conn[elemp];
                    if (temp_completion[gnode] == 1) continue;

                    // printf("fill IEV_ind %d, gnode %d, i_subdomain %d\n", IEV_ind, gnode,
                    //        i_subdomain);

                    this->IEV_nodes[IEV_ind] = gnode;
                    this->IEV_sd_ind[IEV_ind] = i_subdomain;
                    this->IEV_sd_ptr[i_subdomain + 1]++;
                    IEV_ind++;
                    temp_completion[gnode] = 1;
                }
            }
        }
        delete[] temp_completion;

        // printf("IEV_sd_ptr: ");
        // printVec<int>(this->num_subdomains + 1, this->IEV_sd_ptr);
        // printf("IEV_sd_ind: ");
        // printVec<int>(this->IEV_nnodes, this->IEV_sd_ind);
        // printf("IEV_nodes: ");
        // printVec<int>(this->IEV_nnodes, this->IEV_nodes);

        // -----------------------------------------
        // build IEV element connectivity
        // -----------------------------------------
        this->IEV_elem_conn = new int[this->elem_nnz];
        for (int ielem = 0; ielem < this->num_elements; ielem++) {
            int i_subdomain = this->elem_sd_ind[ielem];

            for (int elemp = this->elem_ptr[ielem]; elemp < this->elem_ptr[ielem + 1]; elemp++) {
                int gnode = this->elem_conn[elemp];
                int local_ind = -1;

                for (int jp = this->IEV_sd_ptr[i_subdomain]; jp < this->IEV_sd_ptr[i_subdomain + 1];
                     jp++) {
                    if (this->IEV_nodes[jp] == gnode) {
                        local_ind = jp;
                        break;
                    }
                }

                if (local_ind < 0) {
                    printf("ERROR: failed to find duplicated IEV node for elem %d node %d\n", ielem,
                           gnode);
                }
                // printf("fill gnode %d, IEV_node %d, elemp %d into IEV elem_conn\n", gnode,
                //        local_ind, elemp);
                this->IEV_elem_conn[elemp] = local_ind;
            }
        }

        // printf("IEV_ind %d\n", IEV_ind);
        // printf("IEV_nodes %d: ", IEV_nnodes);
        // printVec<int>(IEV_nnodes, IEV_nodes);
        // printf("IEV_conn: ");
        // printVec<int>(this->elem_nnz, this->IEV_elem_conn);

        // printf("num elements %d\n", this->num_elements);
        // for (int i = 0; i < this->num_elements; i++) {
        // printf("coarse_elem %d (IEV_elem_conn): ", i);
        //     for (int elemp = this->elem_ptr[i]; elemp < this->elem_ptr[i + 1]; elemp++) {
        //         int iev = this->IEV_elem_conn[elemp];
        //         printf("%d ", iev);
        //     }
        //     printf("\n");
        // }

        // for (int iev = 0; iev < IEV_nnodes; iev++) {
        //     int isd = IEV_sd_ind[iev];
        //     int gnode = IEV_nodes[iev];
        //     int iclass = node_class_ind[gnode];
        //     printf("iev %d, isd %d, gnode %d, class %d\n", iev, isd, gnode, iclass);
        // }

        // -----------------------------------------
        // IE and I nodal lists
        // -----------------------------------------
        this->IE_nodes = new int[this->IE_nnodes];
        this->I_nodes = new int[this->I_nnodes];
        std::memset(this->IE_nodes, 0, this->IE_nnodes * sizeof(int));
        std::memset(this->I_nodes, 0, this->I_nnodes * sizeof(int));
        this->IE_interior = new bool[this->IE_nnodes];
        this->IE_general_edge = new bool[this->IE_nnodes];

        int IE_ind = 0;
        int I_ind = 0;
        for (int inode = 0; inode < this->IEV_nnodes; inode++) {
            int gnode = this->IEV_nodes[inode];
            int node_class = this->node_class_ind[gnode];

            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                this->IE_interior[IE_ind] = node_class == INTERIOR || node_class == DIRICHLET_EDGE;
                this->IE_general_edge[IE_ind] = node_class == DIRICHLET_EDGE || node_class == EDGE;
                this->IE_nodes[IE_ind++] = gnode;
            }
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                this->I_nodes[I_ind++] = gnode;
            }
        }
        this->d_IE_interior =
            HostVec<bool>(this->IE_nnodes, this->IE_interior).createDeviceVec().getPtr();
        this->d_IE_general_edge =
            HostVec<bool>(this->IE_nnodes, this->IE_general_edge).createDeviceVec().getPtr();
        this->d_IE_nodes = HostVec<int>(this->IE_nnodes, this->IE_nodes).createDeviceVec().getPtr();

        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("I_nodes %d: ", I_nnodes);
        // printVec<int>(I_nnodes, I_nodes);

        // -----------------------------------------
        // build IEV sparsity from duplicated connectivity
        // -----------------------------------------
        // printf("build BSR data\n");
        // instead build IEV rowp, cols from IEV_elem_conn (without nodes_per_elem)
        // build unique column set for each row
        std::vector<std::set<int>> row_sets(this->IEV_nnodes);

        for (int ielem = 0; ielem < this->num_elements; ielem++) {
            // printf("elem %d: ", ielem);
            for (int elemp = this->elem_ptr[ielem]; elemp < this->elem_ptr[ielem + 1]; elemp++) {
                int IEV_row = this->IEV_elem_conn[elemp];
                for (int elemq = this->elem_ptr[ielem]; elemq < this->elem_ptr[ielem + 1];
                     elemq++) {
                    int IEV_col = this->IEV_elem_conn[elemq];
                    // printf("(%d, %d) ", IEV_row, IEV_col);
                    row_sets[IEV_row].insert(IEV_col);
                }
            }
            // printf("\n");
        }

        // build rowp
        this->IEV_rowp = new int[this->IEV_nnodes + 1];
        this->IEV_rowp[0] = 0;
        for (int i = 0; i < this->IEV_nnodes; i++) {
            this->IEV_rowp[i + 1] = this->IEV_rowp[i] + row_sets[i].size();
        }

        this->IEV_nnzb = this->IEV_rowp[this->IEV_nnodes];
        this->IEV_cols = new int[this->IEV_nnzb];

        // fill cols
        for (int i = 0; i < this->IEV_nnodes; i++) {
            int offset = this->IEV_rowp[i];
            for (int col : row_sets[i]) {
                this->IEV_cols[offset++] = col;
            }
        }

        // this->IEV_bsr_data =
        //     BsrData(this->num_elements, this->IEV_nnodes, nodes_per_elem, block_dim,
        //     IEV_elem_conn);
        // // printf("\tdone build BSR data\n");
        // this->IEV_rowp = this->IEV_bsr_data.rowp;
        // this->IEV_cols = this->IEV_bsr_data.cols;
        // this->IEV_nnzb = this->IEV_bsr_data.nnzb;
        this->IEV_rows = new int[this->IEV_nnzb];
        for (int inode = 0; inode < this->IEV_nnodes; inode++) {
            for (int jp = this->IEV_rowp[inode]; jp < this->IEV_rowp[inode + 1]; jp++) {
                this->IEV_rows[jp] = inode;
            }
        }
        this->IEV_bsr_data = BsrData(this->IEV_nnodes, this->block_dim, this->IEV_nnzb,
                                     this->IEV_rowp, this->IEV_cols);
        this->IEV_bsr_data.rows = this->IEV_rows;
        printf("IEV_rowp with nnzb %d: ", this->IEV_nnzb);
        printVec<int>(this->IEV_nnodes + 1, this->IEV_rowp);
        printf("IEV_cols: ");
        printVec<int>(this->IEV_nnzb, this->IEV_cols);

        // -----------------------------------------
        // reduced rowp arrays
        // -----------------------------------------
        this->IE_rowp = new int[this->IE_nnodes + 1];
        this->I_rowp = new int[this->I_nnodes + 1];
        std::memset(this->IE_rowp, 0, (this->IE_nnodes + 1) * sizeof(int));
        std::memset(this->I_rowp, 0, (this->I_nnodes + 1) * sizeof(int));

        int IE_row = 0;
        int I_row = 0;
        for (int row = 0; row < this->IEV_nnodes; row++) {
            int gnode_row = this->IEV_nodes[row];
            int class_row = this->node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            if (typeI_row) this->I_rowp[I_row + 1] = this->I_rowp[I_row];
            if (typeIE_row) this->IE_rowp[IE_row + 1] = this->IE_rowp[IE_row];

            for (int jp = this->IEV_rowp[row]; jp < this->IEV_rowp[row + 1]; jp++) {
                int col = this->IEV_cols[jp];
                int gnode_col = this->IEV_nodes[col];
                int class_col = this->node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) this->I_rowp[I_row + 1]++;
                if (typeIE_row && typeIE_col) this->IE_rowp[IE_row + 1]++;
            }

            if (typeI_row) I_row++;
            if (typeIE_row) IE_row++;
        }

        this->I_nnzb = this->I_rowp[this->I_nnodes];
        this->IE_nnzb = this->IE_rowp[this->IE_nnodes];

        // printf("I_rowp nnzb %d: ", I_nnzb);
        // printVec<int>(I_nnodes + 1, I_rowp);
        // printf("IE_nodes %d: ", IE_nnodes);
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("IE_rowp nnzb %d: ", IE_nnzb);
        // printVec<int>(IE_nnodes + 1, IE_rowp);

        this->IE_rows = new int[this->IE_nnzb];
        for (int inode = 0; inode < this->IE_nnodes; inode++) {
            for (int jp = this->IE_rowp[inode]; jp < this->IE_rowp[inode + 1]; jp++) {
                this->IE_rows[jp] = inode;
            }
        }
        this->I_rows = new int[this->I_nnzb];
        for (int inode = 0; inode < this->I_nnodes; inode++) {
            for (int jp = this->I_rowp[inode]; jp < this->I_rowp[inode + 1]; jp++) {
                this->I_rows[jp] = inode;
            }
        }

        // -----------------------------------------
        // IEV -> IE map
        // -----------------------------------------
        this->IEVtoIE_map = new int[this->IEV_nnodes];
        std::memset(this->IEVtoIE_map, -1, this->IEV_nnodes * sizeof(int));
        this->IEVtoIE_imap = new int[this->IE_nnodes];

        IE_ind = 0;
        for (int inode = 0; inode < this->IEV_nnodes; inode++) {
            int gnode = this->IEV_nodes[inode];
            int node_class = this->node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE || node_class == EDGE) {
                this->IEVtoIE_imap[IE_ind] = inode;
                this->IEVtoIE_map[inode] = IE_ind++;
            }
        }

        // printf("IEVtoIE_map: ");
        // printVec<int>(IEV_nnodes, IEVtoIE_map);

        // put on device
        this->d_IEVtoIE_imap =
            HostVec<int>(this->IE_nnodes, this->IEVtoIE_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // IEV -> I map
        // -----------------------------------------
        this->IEVtoI_map = new int[this->IEV_nnodes];
        this->IEVtoI_imap = new int[this->I_nnodes];
        std::memset(this->IEVtoI_map, -1, this->IEV_nnodes * sizeof(int));

        I_ind = 0;
        for (int inode = 0; inode < this->IEV_nnodes; inode++) {
            int gnode = this->IEV_nodes[inode];
            int node_class = this->node_class_ind[gnode];
            if (node_class == INTERIOR || node_class == DIRICHLET_EDGE) {
                this->IEVtoI_imap[I_ind] = inode;
                this->IEVtoI_map[inode] = I_ind++;
            }
        }

        // printf("IEVtoI_map: ");
        // printVec<int>(IEV_nnodes, IEVtoI_map);

        // put on device
        this->d_IEVtoI_imap =
            HostVec<int>(this->I_nnodes, this->IEVtoI_imap).createDeviceVec().getPtr();

        // -----------------------------------------
        // reduced column arrays
        // -----------------------------------------
        this->IE_cols = new int[this->IE_nnzb];
        this->I_cols = new int[this->I_nnzb];

        I_ind = 0;
        IE_ind = 0;
        for (int row = 0; row < this->IEV_nnodes; row++) {
            int gnode_row = this->IEV_nodes[row];
            int class_row = this->node_class_ind[gnode_row];
            bool typeI_row = (class_row == INTERIOR || class_row == DIRICHLET_EDGE);
            bool typeIE_row = (typeI_row || class_row == EDGE);

            for (int jp = this->IEV_rowp[row]; jp < this->IEV_rowp[row + 1]; jp++) {
                int col = this->IEV_cols[jp];
                int gnode_col = this->IEV_nodes[col];
                int class_col = this->node_class_ind[gnode_col];
                bool typeI_col = (class_col == INTERIOR || class_col == DIRICHLET_EDGE);
                bool typeIE_col = (typeI_col || class_col == EDGE);

                if (typeI_row && typeI_col) {
                    this->I_cols[I_ind++] = this->IEVtoI_map[col];
                }
                if (typeIE_row && typeIE_col) {
                    this->IE_cols[IE_ind++] = this->IEVtoIE_map[col];
                }
            }
        }

        // printf("I_cols: ");
        // printVec<int>(I_nnzb, I_cols);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);

        this->d_IEV_elem_conn = HostVec<int>(this->elem_nnz, this->IEV_elem_conn).createDeviceVec();

        // -----------------------------------------
        // IEV -> Vc map (coarse non-repeated vertices)
        // -----------------------------------------

        // make a list of the reduced Vc_ind (takes global node and figures out it's Vc_ind)
        std::unordered_set<int> Vc_nodeset;
        for (int i = 0; i < this->IEV_nnodes; i++) {
            int gnode = this->IEV_nodes[i];
            int node_class = this->node_class_ind[gnode];
            if (node_class == VERTEX) {
                Vc_nodeset.insert(gnode);
            }
        }
        std::vector<int> Vc_nodes_vec(Vc_nodeset.begin(), Vc_nodeset.end());
        std::sort(Vc_nodes_vec.begin(), Vc_nodes_vec.end());
        // printf("Vc_nodes %d: ", Vc_nodes_vec.size());
        // printVec<int>(Vc_nodes_vec.size(), Vc_nodes_vec.data());

        int *Vc_inodes = new int[this->num_nodes];  // takes Vc global node => Vc red node
        memset(Vc_inodes, -1, this->num_nodes * sizeof(int));
        for (int i = 0; i < Vc_nodes_vec.size(); i++) {
            int j = Vc_nodes_vec[i];
            Vc_inodes[j] = i;
        }

        // printf("Vc_inodes: ");
        // printVec<int>(num_nodes, Vc_inodes);

        this->d_Vc_nodes =
            HostVec<int>(this->Vc_nnodes, Vc_nodes_vec.data()).createDeviceVec().getPtr();
        this->Vc_nodes = DeviceVec<int>(this->Vc_nnodes, this->d_Vc_nodes).createHostVec().getPtr();

        // printf("post move to device of Vc_nodes\n");

        // set VcToV_imap and IEVtoV_imap now
        this->VctoV_imap = new int[this->V_nnodes];
        std::memset(this->VctoV_imap, -1, this->V_nnodes * sizeof(int));
        this->IEVtoV_imap = new int[this->V_nnodes];
        std::memset(this->IEVtoV_imap, -1, this->V_nnodes * sizeof(int));

        int V_ind = 0;
        for (int inode = 0; inode < this->IEV_nnodes; inode++) {
            int gnode = this->IEV_nodes[inode];
            int node_class = this->node_class_ind[gnode];
            if (node_class == VERTEX) {
                int Vc_ind = Vc_inodes[gnode];
                this->VctoV_imap[V_ind] = Vc_ind;
                this->IEVtoV_imap[V_ind] = inode;
                V_ind++;
            }
        }

        // printf("IEVtoV_imap %d: ", V_nnodes);
        // printVec<int>(V_nnodes, IEVtoV_imap);

        // put on device
        this->d_IEVtoV_imap =
            HostVec<int>(this->V_nnodes, this->IEVtoV_imap).createDeviceVec().getPtr();
        this->d_VctoV_imap =
            HostVec<int>(this->V_nnodes, this->VctoV_imap).createDeviceVec().getPtr();

        // Build all remaining maps/permutations:
        //  - jump operator ownership/sign maps

        // printf("compute jump operators\n");
        bool square_domain = false;  // should be false for BDDC
        this->_compute_jump_operators(square_domain);
        // printf("allocate workspace\n");
        this->allocate_workspace();
        // printf("\tdone with allocate workspace\n");

        // ---------------------------------------------------
        // Save ORIGINAL nofill sparsity for IE and I
        // before fill-in / AMD reordering
        // ---------------------------------------------------
        this->IE_bsr_data =
            BsrData(this->IE_nnodes, this->block_dim, this->IE_nnzb, this->IE_rowp, this->IE_cols);
        this->IE_bsr_data.rows = this->IE_rows;
        this->IE_nofill_nnzb = this->IE_nnzb;
        this->IE_perm = this->IE_bsr_data.perm, this->IE_iperm = this->IE_bsr_data.iperm;

        // sparsity before the permutation
        // printf("\nIE SPARSITY BEFORE PERMUTATION\n");
        // for (int inode = 0; inode < IE_nnodes; inode++) {
        //     printf("(");
        //     int grow = IE_nodes[IE_perm[inode]];
        //     printf("%d, ", grow);
        //     for (int jp = IE_rowp[inode]; jp < IE_rowp[inode + 1]; jp++) {
        //         int j = IE_cols[jp];
        //         int gcol = IE_nodes[IE_perm[j]];
        //         printf("%d ", gcol);
        //     }
        //     printf(")\n");
        // }
        // printf("\n\n");

        // printf("pre-perm IE_rowp: ");
        // printVec<int>(IE_nnodes + 1, IE_rowp);
        // printf("IE_cols: ");
        // printVec<int>(IE_nnzb, IE_cols);
        // printf("IE_nodes: ");
        // printVec<int>(IE_nnodes, IE_nodes);
        // printf("\n");

        this->I_bsr_data =
            BsrData(this->I_nnodes, this->block_dim, this->I_nnzb, this->I_rowp, this->I_cols);
        this->I_bsr_data.rows = this->I_rows;
        this->I_nofill_nnzb = this->I_nnzb;
        // printf("\tdone with setup wing subdomains\n");

        // setup BDDC states
        // build vectors of size gam
        this->n_edge = this->lam_nnodes;
        this->ngam = this->n_edge + this->Vc_nnodes;
        this->gam_nodes = new int[this->ngam];

        for (int i = 0; i < this->n_edge; i++) {
            this->gam_nodes[i] = this->lam_nodes[i];
        }
        for (int i = this->n_edge; i < this->ngam; i++) {
            this->gam_nodes[i] = this->Vc_nodes[i - this->n_edge];
        }
        // printf("gam_nodes (nE %d, nV %d): ", n_edge, this->Vc_nnodes);
        // printVec<int>(ngam, gam_nodes);

        this->temp_lam = Vec(this->lam_nnodes * this->block_dim);
        this->temp_lam2 = Vec(this->lam_nnodes * this->block_dim);
    }

   private:
    // private data
    int elem_nnz, *elem_ptr;
};