#pragma once

#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "cuda_utils.h"
#include "element/shell/_shell.cuh"
#include "linalg/bsr_data.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class ShellAssembler_, template <typename> class Vec_,
          template <typename> class Mat_, class IEVSplit>
class BddcSolver : public BaseSolver {
   public:
    using ShellAssembler = ShellAssembler_;
    using Director = typename ShellAssembler::Director;
    using Basis = typename ShellAssembler::Basis;
    using Geo = typename Basis::Geo;
    using Phys = typename ShellAssembler::Phys;
    using Data = typename Phys::Data;
    using Quadrature = typename Basis::Quadrature;
    using Vec = Vec_<T>;
    using Mat = Mat_<Vec_<T>>;
    using BsrMatType = BsrMat<DeviceVec<T>>;

    static constexpr int32_t nodes_per_elem = Basis::num_nodes;
    static constexpr int32_t vars_per_node = Phys::vars_per_node;
    static constexpr int32_t xpts_per_elem = Geo::spatial_dim * nodes_per_elem;
    static constexpr int32_t dof_per_elem = vars_per_node * nodes_per_elem;
    static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

    BddcSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
               ShellAssembler &assembler_, BsrMatType &kmat_, const IEVSplit &split_,
               bool print_timing_ = false)
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          assembler(assembler_),
          kmat(&kmat_),
          split(split_),
          print_timing(print_timing_) {
        num_elements = assembler.get_num_elements();
        num_nodes = assembler.get_num_nodes();
        N = num_nodes * vars_per_node;

        auto kmat_bsr_data = kmat->getBsrData();
        kmat_nnzb = kmat_bsr_data.nnzb;
        block_dim = kmat_bsr_data.block_dim;
        block_dim2 = block_dim * block_dim;

        d_xpts = assembler.getXpts();
        d_vars = assembler.getVars();
        d_elem_components = assembler.getElemComponents();
        d_compData = assembler.getCompData();

        import_splitting();

        build_IE_I_V_maps();
        build_IEV_sparsity();
        allocate_vectors();

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));
    }

    ~BddcSolver() { clear_host_data(); }

    int get_num_iterations() { return 1; }
    int getLambdaSize() const { return ngam * block_dim; }

    void set_inner_solvers(BaseSolver *subdomainIESolver_, BaseSolver *subdomainISolver_,
                           BaseSolver *coarseSolver_, BaseSolver *subdomainIKrylov_ = nullptr) {
        subdomainIESolver = subdomainIESolver_;
        subdomainISolver = subdomainISolver_;
        coarseSolver = coarseSolver_;
        subdomainIKrylov = subdomainIKrylov_;
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        vars.copyValuesTo(d_vars);

        assemble_subdomains();

        subdomainIESolver->factor();
        subdomainISolver->factor();

        assemble_coarse_problem();

        // More general than factor(), works if coarseSolver is iterative/preconditioned.
        coarseSolver->update_after_assembly(d_coarse_vars);
    }

    void factor() override {}

    void mat_vec(DeviceVec<T> &gam_in, DeviceVec<T> &gam_out) {
        gam_out.zeroValues();

        addVecGamtoIEV(gam_in, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, 1.0, 0.0, f_IEV);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        addVecIEtoI(f_IE, f_I, 1.0, 0.0);

        solveSubdomainI(f_I, u_I);

        addVecItoIE(u_I, u_IE, 1.0, 0.0);
        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);

        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 1.0, f_IEV);

        addVecIEVtoGam(f_IEV, gam_out, 1.0, 0.0);
    }

    bool solve(DeviceVec<T> gam_rhs, DeviceVec<T> gam, bool check_conv = false) {
        constexpr bool SCALED = true;

        addVecGamtoIEV<SCALED>(gam_rhs, f_IEV, 1.0, 0.0);

        addVecIEVtoVc<SCALED>(f_IEV, f_V, 1.0, 0.0);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);
        addVecIEtoIEV(f_IE, f_IEV, -1.0, 1.0);
        zeroInteriorIE(f_IE);

        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, u_IEV, -1.0, 0.0, f_IEV);

        addVecIEVtoVc(f_IEV, f_V, 1.0, 1.0);
        solveCoarse(f_V, u_V);

        addVecVctoIEV(u_V, temp_IEV, 1.0, 0.0);
        sparseMatVec(*kmat_IEV, temp_IEV, -1.0, 0.0, f_IEV);

        addVecIEVtoIE(f_IEV, f_IE, 1.0, 0.0);

        u_IE.zeroValues();
        solveSubdomainIE(f_IE, u_IE);

        addVecIEtoIEV(u_IE, u_IEV, 1.0, 1.0);
        addVecVctoIEV<SCALED>(u_V, u_IEV, 1.0, 1.0);

        addVecIEVtoGam<SCALED>(u_IEV, gam, 1.0, 0.0);

        return false;
    }

   private:
    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    cusparseMatDescr_t descrK = nullptr;

    ShellAssembler &assembler;
    BsrMatType *kmat = nullptr;
    const IEVSplit &split;

    bool print_timing = false;

    int num_elements = 0;
    int num_nodes = 0;
    int N = 0;

    int block_dim = 0;
    int block_dim2 = 0;
    int kmat_nnzb = 0;

    int num_subdomains = 0;

    int I_nnodes = 0;
    int IE_nnodes = 0;
    int IEV_nnodes = 0;
    int Vc_nnodes = 0;
    int V_nnodes = 0;
    int lam_nnodes = 0;
    int ngam = 0;
    int n_edge = 0;

    int *elem_sd_ind = nullptr;
    int *node_class_ind = nullptr;
    int *node_nsd = nullptr;

    int *IEV_sd_ptr = nullptr;
    int *IEV_sd_ind = nullptr;
    int *IEV_nodes = nullptr;
    int *IEV_elem_conn = nullptr;

    int *IE_nodes = nullptr;
    int *I_nodes = nullptr;
    int *Vc_nodes = nullptr;
    int *gam_nodes = nullptr;

    int *IEVtoIE_map = nullptr;
    int *IEVtoI_map = nullptr;
    int *IEVtoV_imap = nullptr;
    int *VctoV_imap = nullptr;

    BsrData IEV_bsr_data;
    int *IEV_rowp = nullptr;
    int *IEV_cols = nullptr;
    int *IEV_rows = nullptr;
    int IEV_nnzb = 0;

    BsrMatType *kmat_IEV = nullptr;

    BaseSolver *subdomainIESolver = nullptr;
    BaseSolver *subdomainISolver = nullptr;
    BaseSolver *coarseSolver = nullptr;
    BaseSolver *subdomainIKrylov = nullptr;

    T *d_xpts = nullptr;
    T *d_vars = nullptr;
    int *d_elem_components = nullptr;
    Data *d_compData = nullptr;

    DeviceVec<int> d_IEV_elem_conn;
    int *d_IEVtoIE_imap = nullptr;
    int *d_IEVtoI_imap = nullptr;
    int *d_IEVtoV_imap = nullptr;
    int *d_VctoV_imap = nullptr;
    int *d_Vc_nodes = nullptr;

    Vec d_IEV_xpts, d_IEV_vars;
    Vec fext_IEV, fint_IEV, res_IEV;
    Vec f_IEV, u_IEV, temp_IEV;
    Vec f_IE, u_IE;
    Vec f_I, u_I;
    Vec f_V, u_V;
    Vec temp_lam, temp_lam2;
    Vec d_coarse_vars;

   private:
    void import_splitting() {
        num_subdomains = split.num_subdomains;

        I_nnodes = split.I_nnodes;
        IE_nnodes = split.IE_nnodes;
        IEV_nnodes = split.IEV_nnodes;
        Vc_nnodes = split.Vc_nnodes;
        V_nnodes = split.V_nnodes;
        lam_nnodes = split.lam_nnodes;

        elem_sd_ind = copy_vec(split.elem_sd_ind);
        node_class_ind = copy_vec(split.node_class_ind);
        node_nsd = copy_vec(split.node_nsd);

        IEV_sd_ptr = copy_vec(split.IEV_sd_ptr);
        IEV_sd_ind = copy_vec(split.IEV_sd_ind);
        IEV_nodes = copy_vec(split.IEV_nodes);
        IEV_elem_conn = copy_vec(split.IEV_elem_conn);

        d_IEV_elem_conn =
            HostVec<int>(num_elements * nodes_per_elem, IEV_elem_conn).createDeviceVec();
    }

    static int *copy_vec(const std::vector<int> &v) {
        if (v.empty()) return nullptr;

        int *out = new int[v.size()];
        std::memcpy(out, v.data(), v.size() * sizeof(int));
        return out;
    }

    void build_IE_I_V_maps() {
        IE_nodes = new int[IE_nnodes];
        I_nodes = new int[I_nnodes];

        IEVtoIE_map = new int[IEV_nnodes];
        IEVtoI_map = new int[IEV_nnodes];

        std::memset(IEVtoIE_map, -1, IEV_nnodes * sizeof(int));
        std::memset(IEVtoI_map, -1, IEV_nnodes * sizeof(int));

        int IE_ind = 0;
        int I_ind = 0;

        for (int iev = 0; iev < IEV_nnodes; iev++) {
            int gnode = IEV_nodes[iev];
            int cls = node_class_ind[gnode];

            bool is_I = (cls == INTERIOR || cls == DIRICHLET_EDGE);
            bool is_IE = is_I || cls == EDGE;

            if (is_IE) {
                IE_nodes[IE_ind] = gnode;
                IEVtoIE_map[iev] = IE_ind;
                IE_ind++;
            }

            if (is_I) {
                I_nodes[I_ind] = gnode;
                IEVtoI_map[iev] = I_ind;
                I_ind++;
            }
        }

        build_Vc_and_gam_maps();
    }

    void build_Vc_and_gam_maps() {
        std::unordered_set<int> Vc_set;

        for (int iev = 0; iev < IEV_nnodes; iev++) {
            int gnode = IEV_nodes[iev];
            if (node_class_ind[gnode] == VERTEX) {
                Vc_set.insert(gnode);
            }
        }

        std::vector<int> Vc_vec(Vc_set.begin(), Vc_set.end());
        std::sort(Vc_vec.begin(), Vc_vec.end());

        Vc_nodes = new int[Vc_vec.size()];
        for (int i = 0; i < (int)Vc_vec.size(); i++) {
            Vc_nodes[i] = Vc_vec[i];
        }

        std::vector<int> Vc_inodes(num_nodes, -1);
        for (int i = 0; i < (int)Vc_vec.size(); i++) {
            Vc_inodes[Vc_vec[i]] = i;
        }

        IEVtoV_imap = new int[V_nnodes];
        VctoV_imap = new int[V_nnodes];

        int V_ind = 0;
        for (int iev = 0; iev < IEV_nnodes; iev++) {
            int gnode = IEV_nodes[iev];
            if (node_class_ind[gnode] == VERTEX) {
                IEVtoV_imap[V_ind] = iev;
                VctoV_imap[V_ind] = Vc_inodes[gnode];
                V_ind++;
            }
        }

        d_IEVtoV_imap = HostVec<int>(V_nnodes, IEVtoV_imap).createDeviceVec().getPtr();
        d_VctoV_imap = HostVec<int>(V_nnodes, VctoV_imap).createDeviceVec().getPtr();

        n_edge = lam_nnodes;
        ngam = n_edge + Vc_nnodes;

        gam_nodes = new int[ngam];

        // NOTE:
        // Fill first n_edge entries from your edge/lam nodes.
        // If your splitting object stores lam_nodes, use that.
        // Otherwise build them from global EDGE nodes.
        int e = 0;
        for (int inode = 0; inode < num_nodes; inode++) {
            if (node_class_ind[inode] == EDGE) {
                if (e < n_edge) gam_nodes[e++] = inode;
            }
        }

        for (int i = 0; i < Vc_nnodes; i++) {
            gam_nodes[n_edge + i] = Vc_nodes[i];
        }

        d_Vc_nodes = HostVec<int>(Vc_nnodes, Vc_nodes).createDeviceVec().getPtr();
    }

    void build_IEV_sparsity() {
        IEV_bsr_data = BsrData(num_elements, IEV_nnodes, nodes_per_elem, block_dim, IEV_elem_conn);

        IEV_rowp = IEV_bsr_data.rowp;
        IEV_cols = IEV_bsr_data.cols;
        IEV_nnzb = IEV_bsr_data.nnzb;

        IEV_rows = new int[IEV_nnzb];

        for (int i = 0; i < IEV_nnodes; i++) {
            for (int jp = IEV_rowp[i]; jp < IEV_rowp[i + 1]; jp++) {
                IEV_rows[jp] = i;
            }
        }
    }

    void allocate_vectors() {
        d_IEV_xpts = Vec(3 * IEV_nnodes);
        d_IEV_vars = Vec(block_dim * IEV_nnodes);

        fext_IEV = Vec(block_dim * IEV_nnodes);
        fint_IEV = Vec(block_dim * IEV_nnodes);
        res_IEV = Vec(block_dim * IEV_nnodes);

        f_IEV = Vec(block_dim * IEV_nnodes);
        u_IEV = Vec(block_dim * IEV_nnodes);
        temp_IEV = Vec(block_dim * IEV_nnodes);

        f_IE = Vec(block_dim * IE_nnodes);
        u_IE = Vec(block_dim * IE_nnodes);

        f_I = Vec(block_dim * I_nnodes);
        u_I = Vec(block_dim * I_nnodes);

        f_V = Vec(block_dim * Vc_nnodes);
        u_V = Vec(block_dim * Vc_nnodes);

        temp_lam = Vec(block_dim * lam_nnodes);
        temp_lam2 = Vec(block_dim * lam_nnodes);

        d_coarse_vars = Vec(block_dim * Vc_nnodes);
    }

    void clear_host_data() {
        delete[] elem_sd_ind;
        delete[] node_class_ind;
        delete[] node_nsd;

        delete[] IEV_sd_ptr;
        delete[] IEV_sd_ind;
        delete[] IEV_nodes;
        delete[] IEV_elem_conn;

        delete[] IE_nodes;
        delete[] I_nodes;
        delete[] Vc_nodes;
        delete[] gam_nodes;

        delete[] IEVtoIE_map;
        delete[] IEVtoI_map;
        delete[] IEVtoV_imap;
        delete[] VctoV_imap;

        delete[] IEV_rows;

        if (descrK) {
            cusparseDestroyMatDescr(descrK);
            descrK = nullptr;
        }
    }

    // ---------------------------------------------------------------------
    // Core BDDC/FETI-DP helper operations
    // ---------------------------------------------------------------------

    void assemble_subdomains() {
        addVec_globalToIEV(d_xpts, d_IEV_xpts, 3, 1.0, 0.0);
        addVec_globalToIEV(d_vars, d_IEV_vars, block_dim, 1.0, 0.0);

        kmat_IEV->zeroValues();

        const int elems_per_block = 1;
        int cols_per_elem = (Basis::order == 1 ? 24 : Basis::order == 2 ? 9 : 4);

        dim3 block(num_quad_pts, cols_per_elem, 1);

        int elem_cols_per_block = cols_per_elem * elems_per_block;
        int nblocks = (num_elements * dof_per_elem + elem_cols_per_block - 1) / elem_cols_per_block;

        dim3 grid(nblocks);

        k_add_jacobian_fast<T, elems_per_block, ShellAssembler, Data, Vec_, BsrMatType>
            <<<grid, block>>>(IEV_nnodes, num_elements, cols_per_elem, d_elem_components,
                              d_IEV_elem_conn, d_IEV_elem_conn, d_IEV_xpts, d_IEV_vars, d_compData,
                              *kmat_IEV);

        kmat_IEV->apply_bcs(d_IEV_bcs);

        kmat_IE->zeroValues();
        kmat_I->zeroValues();

        copyKmat_IEVtoIE();
        copyKmat_IEVtoI();
    }

    void assemble_coarse_problem() {
        if (print_timing) {
            _assemble_coarse_problem_timing();
            return;
        }

        S_VV->zeroValues();

        if (S_VV_MLIEV) {
            S_VV_MLIEV->zeroValues();
        }

        copyKmat_IEVtoSvv();
        computeSvvInverseTerm();

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void sparseMatVec(BsrMatType &A, Vec &x, T alpha, T beta, Vec &y) {
        auto bsr_data = A.getBsrData();

        int mb = bsr_data.mb;
        int nb = bsr_data.nb;
        int nnzb = bsr_data.nnzb;

        T *d_vals = A.getVec().getPtr();
        int *d_rowp = bsr_data.rowp;
        int *d_cols = bsr_data.cols;
        int *perm = bsr_data.perm;
        int *iperm = bsr_data.iperm;

        x.permuteData(block_dim, iperm);
        y.permuteData(block_dim, iperm);

        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
            &alpha, descrK, d_vals, d_rowp, d_cols, block_dim, x.getPtr(), &beta, y.getPtr()));

        x.permuteData(block_dim, perm);
        y.permuteData(block_dim, perm);
    }

    void solveSubdomainIE(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainIESolver) {
            printf("ERROR: subdomain IE solver is null\n");
            return;
        }

        auto bsr_data = kmat_IE->getBsrData();
        int *d_perm = bsr_data.perm;
        int *d_iperm = bsr_data.iperm;

        rhs_in.copyValuesTo(rhs_IE_perm);
        rhs_IE_perm.permuteData(block_dim, d_iperm);

        subdomainIESolver->solve(rhs_IE_perm, sol_IE_perm);

        sol_IE_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveSubdomainI(Vec &rhs_in, Vec &sol_out) {
        if (!subdomainISolver) {
            printf("ERROR: subdomain I solver is null\n");
            return;
        }

        auto bsr_data = kmat_I->getBsrData();
        int *d_perm = bsr_data.perm;
        int *d_iperm = bsr_data.iperm;

        rhs_in.copyValuesTo(rhs_I_perm);
        rhs_I_perm.permuteData(block_dim, d_iperm);

        subdomainISolver->solve(rhs_I_perm, sol_I_perm);

        sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveSubdomainIKrylov(Vec &rhs_in, Vec &sol_out) {
        auto bsr_data = kmat_I->getBsrData();
        int *d_perm = bsr_data.perm;
        int *d_iperm = bsr_data.iperm;

        rhs_in.copyValuesTo(rhs_I_perm);
        rhs_I_perm.permuteData(block_dim, d_iperm);

        if (subdomainIKrylov == nullptr) {
            if (warnings) {
                printf(
                    "WARNING: using subdomainISolver for full K_II^-1 solve. "
                    "Use set_inner_solvers(..., subdomainIKrylov) if using ILU(k) or another "
                    "inexact local preconditioner.\n");
            }

            subdomainISolver->solve(rhs_I_perm, sol_I_perm);
        } else {
            subdomainIKrylov->solve(rhs_I_perm, sol_I_perm, true);
        }

        sol_I_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void solveCoarse(Vec &rhs_in, Vec &sol_out) {
        if (!coarseSolver) {
            printf("ERROR: coarse solver is null\n");
            return;
        }

        auto bsr_data = S_VV->getBsrData();
        int *d_perm = bsr_data.perm;
        int *d_iperm = bsr_data.iperm;

        rhs_in.copyValuesTo(rhs_Vc_perm);
        rhs_Vc_perm.permuteData(block_dim, d_iperm);

        bool check_conv = true;
        coarseSolver->solve(rhs_Vc_perm, sol_Vc_perm, check_conv);

        sol_Vc_perm.copyValuesTo(sol_out);
        sol_out.permuteData(block_dim, d_perm);
    }

    void zeroInteriorIE(DeviceVec<T> &x) {
        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_zeroInterior<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_interior, x.getPtr());
    }

    template <bool SCALED = false>
    void addVecIEVtoGam(const DeviceVec<T> &vec_IEV, DeviceVec<T> &vec_gam, T alpha, T beta) {
        int gam_size = ngam * block_dim;

        CHECK_CUBLAS(cublasDscal(cublasHandle, gam_size, &beta, vec_gam.getPtr(), 1));

        addVecIEVtoIE(vec_IEV, temp_IE, alpha, 0.0);
        addVecIEtoGam(temp_IE, temp_lam, 1.0, 0.0);

        int edge_size = lam_nnodes * block_dim;

        if constexpr (SCALED) {
            dim3 block(32);
            dim3 grid((edge_size + 31) / 32);

            k_subdomain_normalize_vec_inout<T>
                <<<grid, block>>>(lam_nnodes, block_dim, d_edge_nsd, temp_lam.getPtr());
        }

        T a = 1.0;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, edge_size, &a, temp_lam.getPtr(), 1, vec_gam.getPtr(), 1));

        addVecIEVtoVc(vec_IEV, temp_V, alpha, 0.0);

        T *d_vec_gam = vec_gam.getPtr();
        int V_size = Vc_nnodes * block_dim;

        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, V_size, &a, temp_V.getPtr(), 1, &d_vec_gam[edge_size], 1));
    }

    template <bool SCALED = false>
    void addVecGamtoIEV(const DeviceVec<T> &vec_gam, DeviceVec<T> &vec_IEV, T alpha, T beta) {
        int IEV_size = block_dim * IEV_nnodes;

        CHECK_CUBLAS(cublasDscal(cublasHandle, IEV_size, &beta, vec_IEV.getPtr(), 1));

        temp_lam.zeroValues();
        temp_V.zeroValues();

        T a = alpha;
        int edge_size = lam_nnodes * block_dim;

        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, edge_size, &a, vec_gam.getPtr(), 1, temp_lam.getPtr(), 1));

        if constexpr (SCALED) {
            dim3 block(32);
            dim3 grid((edge_size + 31) / 32);

            k_subdomain_normalize_vec_inout<T>
                <<<grid, block>>>(lam_nnodes, block_dim, d_edge_nsd, temp_lam.getPtr());
        }

        addVecGamtoIE(temp_lam, temp_IE, 1.0, 0.0);
        addVecIEtoIEV(temp_IE, vec_IEV, 1.0, 0.0);

        int V_size = Vc_nnodes * block_dim;

        CHECK_CUBLAS(cublasDaxpy(cublasHandle, V_size, &a, &vec_gam.getPtr()[edge_size], 1,
                                 temp_V.getPtr(), 1));

        addVecVctoIEV(temp_V, vec_IEV, 1.0, 1.0);
    }

    void addVecGamtoIE(const DeviceVec<T> &gam, DeviceVec<T> &vec_IE, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));

        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecGamtoIE<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, gam.getPtr(),
                                            vec_IE.getPtr(), a);
    }

    void addVecIEtoGam(const DeviceVec<T> &vec_IE, DeviceVec<T> &gam, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, gam.getSize(), &b, gam.getPtr(), 1));

        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecIEtoGam<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, vec_IE.getPtr(),
                                            gam.getPtr(), a);
    }

    template <bool scaled = false>
    void addVec_globalToIEV(Vec &x_global, Vec &y_iev, int vars_per_node_in, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, y_iev.getSize(), &b, y_iev.getPtr(), 1));

        u_IE.zeroValues();
        u_V.zeroValues();

        int nvals = IE_nnodes * vars_per_node_in;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVec_GlobalToIE<T, scaled><<<grid, block>>>(
            IE_nnodes, vars_per_node_in, d_IE_nodes, d_IE_nsd, x_global.getPtr(), u_IE.getPtr(), a);

        int nvals2 = Vc_nnodes * vars_per_node_in;
        dim3 grid2((nvals2 + 31) / 32);

        k_addVec_GlobaltoVc<T, scaled><<<grid2, block>>>(Vc_nnodes, vars_per_node_in, d_Vc_nodes,
                                                         d_vertex_nsd, x_global.getPtr(),
                                                         u_V.getPtr(), a);

        addVecIEtoIEV(u_IE, y_iev, 1.0, 0.0, vars_per_node_in);
        addVecVctoIEV(u_V, y_iev, 1.0, 1.0, vars_per_node_in);
    }

    void addVecIEVtoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecSmallerOut<T>
            <<<grid, block>>>(IE_nnodes, block_dim, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecItoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = I_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecSmallerIn<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    void addVecIEVtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = I_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecSmallerOut<T>
            <<<grid, block>>>(I_nnodes, block_dim, d_IEVtoI_imap, x.getPtr(), y.getPtr(), a);
    }

    template <bool scaled = false>
    void addVecIEVtoVc(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = V_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVec_IEVtoVc<T><<<grid, block>>>(V_nnodes, block_dim, d_IEVtoV_imap, d_VctoV_imap,
                                             x.getPtr(), y.getPtr(), a);

        if constexpr (scaled) {
            int Vc_nvals = Vc_nnodes * block_dim;
            dim3 grid2((Vc_nvals + 31) / 32);

            k_subdomain_normalize_vec_inout<T>
                <<<grid2, block>>>(Vc_nnodes, block_dim, d_vertex_nsd, y.getPtr());
        }
    }

    void addVecIEtoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        if (vars_per_node == -1) {
            vars_per_node = block_dim;
        }

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = IE_nnodes * vars_per_node;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecSmallerIn<T>
            <<<grid, block>>>(IE_nnodes, vars_per_node, d_IEVtoIE_imap, x.getPtr(), y.getPtr(), a);
    }

    template <bool scaled = false>
    void addVecVctoIEV(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b, int vars_per_node = -1) {
        CHECK_CUDA(cudaMemcpy(temp_V.getPtr(), x.getPtr(), Vc_nnodes * block_dim * sizeof(T),
                              cudaMemcpyDeviceToDevice));

        if constexpr (scaled) {
            int Vc_nvals = Vc_nnodes * block_dim;
            dim3 block(32);
            dim3 grid((Vc_nvals + 31) / 32);

            k_subdomain_normalize_vec_inout<T>
                <<<grid, block>>>(Vc_nnodes, block_dim, d_vertex_nsd, temp_V.getPtr());
        }

        if (vars_per_node == -1) {
            vars_per_node = block_dim;
        }

        CHECK_CUBLAS(cublasDscal(cublasHandle, y.getSize(), &b, y.getPtr(), 1));

        int nvals = V_nnodes * vars_per_node;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVec_VctoIEV<T><<<grid, block>>>(V_nnodes, vars_per_node, d_IEVtoV_imap, d_VctoV_imap,
                                             temp_V.getPtr(), y.getPtr(), a);
    }

    void addVecItoIE(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        addVecItoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoIE(temp_IEV, y, 1.0, b);
    }

    void addVecIEtoI(const DeviceVec<T> &x, DeviceVec<T> &y, T a, T b) {
        addVecIEtoIEV(x, temp_IEV, a, 0.0);
        addVecIEVtoI(temp_IEV, y, 1.0, b);
    }

    void addVecLamtoIE(const DeviceVec<T> &lam, DeviceVec<T> &vec_IE, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, vec_IE.getSize(), &b, vec_IE.getPtr(), 1));

        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecLamtoIE<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            lam.getPtr(), vec_IE.getPtr(), a);
    }

    void addVecIEtoLam(const DeviceVec<T> &vec_IE, DeviceVec<T> &lam, T a, T b) {
        CHECK_CUBLAS(cublasDscal(cublasHandle, lam.getSize(), &b, lam.getPtr(), 1));

        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVecIEtoLam<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_to_lam_map, d_IE_to_lam_vec,
                                            vec_IE.getPtr(), lam.getPtr(), a);
    }

    void addGlobalSoln(const DeviceVec<T> &u_IE, const DeviceVec<T> &u_V, DeviceVec<T> &soln) {
        int nvals = IE_nnodes * block_dim;
        dim3 block(32);
        dim3 grid((nvals + 31) / 32);

        k_addVec_IEtoGlobal<T><<<grid, block>>>(IE_nnodes, block_dim, d_IE_nodes, d_IE_nsd,
                                                u_IE.getPtr(), soln.getPtr(), 1.0);

        int nvals2 = Vc_nnodes * block_dim;
        dim3 grid2((nvals2 + 31) / 32);

        k_addVec_VctoGlobal<T>
            <<<grid2, block>>>(Vc_nnodes, block_dim, d_Vc_nodes, u_V.getPtr(), soln.getPtr(), 1.0);
    }
};