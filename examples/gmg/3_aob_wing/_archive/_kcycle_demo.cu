
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/fea.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

/* argparse options:
[mg/direct/debug] [--level int]
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

std::string time_string(int itime) {
    std::string _time = std::to_string(itime);
    if (itime < 10) {
        return "00" + _time;
    } else if (itime < 100) {
        return "0" + _time;
    } else {
        return _time;
    }
}

void solve_linear_pcg_kcycle_gmg(MPI_Comm &comm, int level, double SR, int nsmooth, int ncycles) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // old smoothers
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2_JUNCTION;
    const SCALER scaler = LINE_SEARCH; // inner V-cycle still need these updates

    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Basis, is_bsr>; 

    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    using MG = GeometricMultigridSolver<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    GRID *fine_grid;

    // make each wing multigrid object..
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.01;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // TODO : run optimized design from AOB case

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder = true;
        // printf("reorder %d\n", reorder);
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        printf("begin unstructured map\n");
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = mg.grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting v cycle solve\n");
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    // best was V(4,4) before
    // bool print = false;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    T omega = 1.5; // good GS-SSOR parameter (speedups up convergence)

    // ----------------------------------------------------------------------------
    // PCG solve

    /* 1) PCG setup (or allocate work arrays) */
    auto &mat = mg.grids[0].Kmat;
    auto soln = mg.grids[0].d_soln;
    auto rhs = mg.grids[0].d_rhs;
    
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    int *d_rowp = bsr_data.rowp;
    int *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    T *d_vals = mat.getPtr();

    cublasHandle_t &cublasHandle = mg.grids[0].cublasHandle;
    cusparseHandle_t &cusparseHandle = mg.grids[0].cusparseHandle;
    
    int N = soln.getSize();
    T *d_rhs = rhs.getPtr();
    T *d_x = soln.getPtr(); // soln starts out at zero

    // description of the A matrix
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // make temp vecs
    T *d_tmp = DeviceVec<T>(N).getPtr();
    auto d_resid_vec = DeviceVec<T>(N);
    T *d_resid = d_resid_vec.getPtr();
    T *d_p = DeviceVec<T>(N).getPtr();
    T *d_w = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();
    T *d_zprev = DeviceVec<T>(N).getPtr();

    /* 2) begin PCG solve with GMG preconditioner (no restarts in this version, not much point in PCG cause low # temp vecs) */
    int n_iter = 100;
    bool can_print = true;
    int print_freq = 1;
    // int print_freq = 5;
    // int n_cycles = 2;
    // int n_cycles = 4;
    // int n_cycles = 10;

    // NOTE : I'm implementing first here a left-precond flexible PCG
    // no need for right-precond, this is the true residual despite left-precond (nice feature in PCG)

    // compute r_0 = b - Ax
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
    T a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                    d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

    // compute |r_0|
    T init_resid_norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
    if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

    // copy z => p
    CHECK_CUDA(cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));

    T rho_prev, rho; // coefficients that we need to remember
    bool converged = false;


    // if constexpr (pcg_method == 1) {
        // from this document, https://www.netlib.org/templates/templates.pdf
        // doesn't include corrections if 

        // inner loop
        for (int j = 0; j < n_iter; j++) {

            /* inner 1) solve Mz = r for z (precond) */
            // ----------------------------------------

            // set the defect of Vcycle to the residual (permuted), and set soln to zero
            cudaMemcpy(mg.grids[0].d_defect.getPtr(), d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemset(mg.grids[0].d_soln.getPtr(), 0.0, N * sizeof(T));

            // only so many steps of outer V-cycle
            bool inner_print = false, double_smooth = true;
            int print_freq = 1;
            // bool symmetric = true;
            bool symmetric = false; // this is tsronger smoother and doesn't really help PCG? some reason
            mg.vcycle_solve(0, pre_smooth, post_smooth, ncycles, inner_print, atol, rtol, omega, double_smooth, print_freq, symmetric); // V-cycle with precond actually faster than F-cycle..
            // mg.fcycle_solve(0, pre_smooth, post_smooth, ncycles, inner_print, atol, rtol, omega, double_smooth, print_freq, symmetric);

            // copy out of fine grid temp vec into z the prolong solution
            cudaMemcpy(d_z, mg.grids[0].d_soln.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);

            // // write precond solution
            // int *d_perm = mg.grids[0].d_perm;
            // auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
            // printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");

            /* 2) compute dot products, and p vec */
            // -------------------------------------
            
            // if fletcher-reeves method
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));

            if (j == 0) {
                // first iteration, p = z
                cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // compute beta
                T beta = rho / rho_prev;

                // p_new = z + beta * p in two steps
                a = beta;  // p *= beta scalar
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
                a = 1.0;  // p += z
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));
            }

            // store rho for next iteration (prev), only used in this part
            rho_prev = rho;

            /* 3) compute w = A * p mat-vec product */
            // ----------------------------------------

            // w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                            descrA, d_vals, d_rowp, d_cols, block_dim, d_p, &b, d_w));

            /* 4) update x and r using dot products */
            // ---------------------------------------

            // alpha = <r,z> / <w,p> = rho / <w,p>
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            T alpha = rho / wp0;

            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

            // r -= alpha * w
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));

            // copy z into zprev (for polak-riberre formula)
            cudaMemcpy(d_zprev, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);

            // check for convergence
            T resid_norm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
            if (can_print && (j % print_freq == 0)) printf("PCG [%d] = %.8e\n", j, resid_norm);

            if (abs(resid_norm) < (atol + init_resid_norm * rtol)) {
                converged = true;
                if (can_print)
                    printf("\nPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
                break;
            }
        }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = mg.get_memory_usage_mb();
    printf("wingbox PCG K-cycle with GMG precond solve:\n");
    printf("\tndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    // double check with true resid nrm
    T resid_nrm = mg.grids[0].getResidNorm();
    printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    bool is_gmres = false;
    int level = 3; // level mesh to solve..
    double SR = 50.0;
    // int nsmooth = 4;
    int nsmooth = 6; // typically faster right now
    int ncycles = 4; // how many V-cycles for preconditioner..

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else if (strcmp(arg, "--ncycles") == 0) {
            if (i + 1 < argc) {
                ncycles = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--SR double] [--nsmooth int]" << std::endl;
            return 1;
        }
    }

    if (is_gmres) {
        // solve_linear_gmres_kcycle_gmg(comm, level, SR, nsmooth, ncycles);
        return 0;
    } else {
        solve_linear_pcg_kcycle_gmg(comm, level, SR, nsmooth, ncycles);
    }
    

    MPI_Finalize();
    return 0;
};
