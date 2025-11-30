// #include "../../examples/plate/_src/_plate_utils.h"
#include "multigrid/utils/fea.h" // these plate assemblers can do higher order
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include <chrono>

// chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
// #include "element/shell/basis/lagrange_basis.h"
// #include "element/shell/mitc_shell.h"

template <typename Quad, typename Basis>
void time_assembly(int nxe) {
    // run the plate problem to time the assembly (nofill)
    using T = double;   

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_nz = std::chrono::high_resolution_clock::now();

    using Director = LinearizedRotation<T>;
    using Geo = typename Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC higher order is really slow to assemble (cause tying so we just do CFI higher order)
    // using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;
    using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    auto& bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();

    // get the loads
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Basis, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    int nvars = assembler.get_num_vars();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_nz = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nz_time = end_nz - start_nz;
    printf("\n=============================\n\n");
    printf("plate mesh with %d x %d elems or %d NDOF (nofill pattern)\n", nxe, nxe, nvars);
    printf("\tnonzero pattern in %.5e seconds\n", nz_time.count());

    // assemble the kmat
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_jac = std::chrono::high_resolution_clock::now();
    // const int cols_per_elem = 4;
    // // const int cols_per_elem = 9;
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_jac = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> jac_time = end_jac - start_jac;
    printf("\tjacobian assembly in %.5e seconds\n", jac_time.count());

    // assemble the residual
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_res = std::chrono::high_resolution_clock::now();
    assembler.add_residual_fast(res);
    assembler.apply_bcs(res);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_res = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> res_time = end_res - start_res;
    printf("\tresidual assembly in %.5e seconds\n", res_time.count());

    // // estimate throughput on jacobian assembly
    // // double peak_gflops = 108*64*2*1.41; // for A100
    // double peak_gflops = 1250; // gflops for 3090Ti GPU (double precision so about 10x lower)
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);
    // kernel_assemble<<<grid, block>>>(...);
    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float ms = 0.0f;
    // cudaEventElapsedTime(&ms, start, stop);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // double flops = num_elements * flops_per_element;
    // double achieved_gflops = flops / (ms * 1e6);
    // double percent = 100.0 * achieved_gflops / peak_gflops;

    // printf("Assembly: %.2f GFLOPS (%.1f%% of peak)\n",
    //     achieved_gflops, percent);

}

template <typename T>
void run_with_order(int order, int nxe) {
    switch(order) {

        case 1: {
            using Quad = QuadLinearQuadrature<T>;
            using Basis = ChebyshevQuadBasis<T, Quad, 1>;
            time_assembly<Quad, Basis>(nxe);
            break;
        }

        case 2: {
            using Quad = QuadQuadraticQuadrature<T>;
            using Basis = ChebyshevQuadBasis<T, Quad, 2>;
            time_assembly<Quad, Basis>(nxe);
            break;
        }

        case 3: {
            using Quad = QuadCubicQuadrature<T>;
            using Basis = ChebyshevQuadBasis<T, Quad, 3>;
            time_assembly<Quad, Basis>(nxe);
            break;
        }

        default:
            printf("ERROR: Unsupported order %d\n", order);
            exit(1);
    }
}


int main(int argc, char** argv) {
    using T = double;

    int ORDER = (argc > 1 ? atoi(argv[1]) : 1);

    int nxe_vals[7] = {16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 7; i++) {
        run_with_order<T>(ORDER, nxe_vals[i]);
    }
}