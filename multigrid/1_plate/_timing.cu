// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// chebyshev fully integrated element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Basis, class Assembler>
void plate_timings(int nxe, double SR) {
    auto start0 = std::chrono::high_resolution_clock::now();
    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_0 = std::chrono::high_resolution_clock::now();
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> create_time0 = end_0 - start_0;
    printf("time to create the plate assembler object %.2e\n", create_time0.count());

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;

    auto start_1 = std::chrono::high_resolution_clock::now();
    // bsr_data.AMD_reordering();
    int num_colors, *_color_rowp;
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> reorder_time = end_1 - start_1;
    // printf("time to do AMD reordering %.2e\n", reorder_time.count());
    printf("time to do MC reordering %.2e\n", reorder_time.count());

    auto start_2 = std::chrono::high_resolution_clock::now();
    // bsr_data.compute_full_LU_pattern(fillin, print);
    bsr_data.compute_nofill_pattern();
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> full_LU_time = end_2 - start_2;
    // printf("time to do compute full LU pattern %.2e\n", full_LU_time.count());
    printf("time to do nofill pattern %.2e\n", full_LU_time.count());

    auto start_3 = std::chrono::high_resolution_clock::now();
    assembler.moveBsrDataToDevice();
    auto end_3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> move_bsr_device_time = end_3 - start_3;
    printf("time to move bsr data to device %.2e\n", move_bsr_device_time.count());


    auto start_4 = std::chrono::high_resolution_clock::now();
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto end_4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> other_init_time1 = end_4 - start_4;
    printf("other init time like bcs and get loads %.2e\n", other_init_time1.count());

    // assemble the kmat
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_as = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_as = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assemb_time = end_as - start_as;

    // apply bcs
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    // solve the linear system
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;

    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double nz_time = startup_time.count() - assemb_time.count();
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    
    printf("plate direct solve, ndof %d :\n", ndof);
    printf("\tusing Basis %s, Assembler %s\n", typeid(Basis).name(), typeid(Assembler).name());
    printf("\tassembly time %.2e, other startup time %.2e / total time %.2e\n", assemb_time.count(), nz_time, total);
    printf("\tdirect solve time %.2e / total time %.2e, with mem (MB) %.2e\n", solve_time.count(), total, mem_mb);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    // int nxe = 256; // default value
    int nxe = 1024;
    double SR = 100.0; // default
    std::string elem_type = 'MITC4'; // 'MITC4', 'CFI4', 'CFI9'

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--elem") == 0) {
            if (i + 1 < argc) {
                elem_type = argv[++i];
            } else {
                std::cerr << "Missing value for --elem\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    if (elem_type == 'MITC4') {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        plate_timings<T, Assembler>(nxe, SR);
    } else if (elem_type == 'CFI4') {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        plate_timings<T, Assembler>(nxe, SR);
    } else if (elem_type == 'CFI9') {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        plate_timings<T, Assembler>(nxe, SR);
    } else (
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    )

    return 0;

    
}