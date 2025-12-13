#include "../plate/_src/_plate_utils.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

/* command line args:
    [linear|nonlinear] [--iterative] [--nxe int]
    * currently nxe must be multiple of 5

    examples:
    ./static.out linear --iterative --nxe 10    to run linear, GMRES solve with nxe=10
    ./static.out nonlinear   to run nonlinear
    if --iterative is not used, full_LU direct solve instead
*/

void solve_linear(double slenderness, int nxe, double qorder, double diag_frac = 1e-10) {
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / slenderness, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 5, nye_per_comp = nye/5; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    bool print = true;

    /*
    RCM and reorderings actually hurt GMRES performance on the plate case
    because the matrix already has a nice banded structure => RCM increases bandwidth (which means it just doesn't work well for this problem
    as it's whole point is to decrease matrix bandwidth)
    */

    // bsr_data.AMD_reordering();
    // bsr_data.RCM_reordering();

    if (qorder > 0) {
        printf("Qorder with p = %.4e\n", qorder);
        // bsr_data.RCM_reordering();
        bsr_data.qorder_reordering(qorder);
    } else if (qorder == 0.0) {
        printf("random reordering\n");
        bsr_data.random_reordering();
    } else {
        printf("Qorder input < 0, so no qorder\n");
    }
        
    // bsr_data.compute_ILUk_pattern(5, fillin);
    bsr_data.compute_nofill_pattern();
    // bsr_data.compute_ILUk_pattern(0, fillin);
    // bsr_data.compute_full_LU_pattern(fillin, print); // reordered full LU here for debug

    // printf("perm:");
    // printVec<int>(bsr_data.nnodes, bsr_data.perm);
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    /* old way to modify matrix diags */

    // auto d_kmat_vec = kmat.getVec();
    // auto h_kmat_vec = d_kmat_vec.createHostVec();
    // int nnodes = bsr_data.nnodes, nnzb = bsr_data.nnzb;
    // int *d_rowp = bsr_data.rowp, *d_cols = bsr_data.cols;
    // int *rowp = DeviceVec<int>(nnodes, d_rowp).createHostVec().getPtr();
    // int *cols = DeviceVec<int>(nnzb, d_cols).createHostVec().getPtr();

    // double diag_add = 1e4; // worked well pure add for SR = 10, but changed to diag_frac method

    // double diag_frac = 1e-4; // BEST here (works for all SR evenly)
    // double diag_frac = 1e-5;
    // double diag_frac = 1e-6;

    // for (int row = 0; row < nnodes; row++) {
    //     for (int jp = rowp[row]; jp < rowp[row+1]; jp++) {
    //         // int col = cols[jp];

    //         // compute trace mag
    //         double trace_abs = 0.0;
    //         for (int i = 0; i < 6; i++) {
    //             trace_abs += abs(h_kmat_vec[36 * jp + 6 * i + i]); // add small diags here in each 6x6 block nodal matrix
    //         }

            
    //         for (int i = 0; i < 6; i++) {
    //             // T val = h_kmat_vec[36 * jp + 6 * i + i];
    //             // T sign = val > 0.0 ? 1.0 : -1.0;
    //             // T sign = 1.0;
    //             h_kmat_vec[36 * jp + 6 * i + i] += diag_frac * trace_abs; // add small diags here in each 6x6 block nodal matrix
    //         }
    //     }
    // }
    // h_kmat_vec.copyValuesTo(d_kmat_vec);

    /* new equiv way to add the matrix block diags */
    // kmat.add_diag_to_each_block(diag_frac);

    // NO, but you can't just add diag nugget to whole matrix (can't change original, can only change precond one, otherwise it makes matrix more benign)
    // unrealistically, doing it inside GMRES solve right now then (may add new option)

    // solve the linear system
    int n_iter = 400, max_iter = n_iter;
    // T abs_tol = 1e-11, rel_tol = 1e-14;
    T abs_tol = 1e-8, rel_tol = 1e-8;
    // bool print = true;
    CUSPARSE::GMRES_solve<T>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print, false, 10, true, diag_frac);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "plate.vtk");

    // check the residual of the system
    assembler.set_variables(soln);
    assembler.add_residual(res); // internal residual
    auto rhs = assembler.createVarsVec();
    CUBLAS::axpy(1.0, loads, rhs);
    CUBLAS::axpy(-1.0, res, rhs); // rhs = loads - f_int
    assembler.apply_bcs(rhs);
    double resid_norm = CUBLAS::get_vec_norm(rhs);
    printf("resid_norm = %.4e\n", resid_norm);
}

int main(int argc, char **argv) {
    // input ----------
    // bool run_linear = true;
    int nxe = 100;  // default value
    double SR = 10.0; // default slenderness
    double qorder = -1.0; // example values 1.0, 0.5, 0.25
    double diag_frac = 0.0; // default diag nugget fraction

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        // if (strcmp(arg, "linear") == 0) {
        //     run_linear = true;
        // } else if (strcmp(arg, "nonlinear") == 0) {
        //     run_linear = false;
        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--diag") == 0) {
            if (i + 1 < argc) {
                diag_frac = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --diag\n";
                return 1;
            }
        } else if (strcmp(arg, "--qorder") == 0) {
            if (i + 1 < argc) {
                qorder = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --qorder\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--SR value] [--nxe value] [--qorder value]" << std::endl;
            return 1;
        }
    }

    solve_linear(SR, nxe, qorder, diag_frac);
    return 0;
};