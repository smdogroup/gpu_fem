// use multi-GPU partition typically for assembler
// BUT: if on coarsest problem, vector uses multi-GPU partition
// and assembler will use single-GPU partition
// and then copy into LU pattern of SingleGPUDirectLU class

#include "assembler/gpu_assembler.h"
#include "smoothers/gpu_asw.h"
#include "assembler/gpu_mitc_shell.h"
#include "solvers/gpu_pcg.h"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/gpu_print_vtk.h"
#include "utils/fea.h"
#include "utils/multigpu_context.h"
#include "partition/component_partitioner.h"
#include "prolongation/gpu_uprolong.h"
#include "solvers/sgpu_direct.h"
#include "solvers/gmg.h"

#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"

#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK_ALL(ctx, msg)                                                       \
    do {                                                                               \
        printf("\n[MAIN CUDA CHECK] %s\n", msg);                                       \
        for (int _g = 0; _g < (ctx)->ngpus; _g++) {                                    \
            CHECK_CUDA(cudaSetDevice((ctx)->debug ? 0 : _g));                          \
            cudaError_t _err = cudaGetLastError();                                     \
            if (_err != cudaSuccess) {                                                 \
                printf("[MAIN CUDA ERROR] %s GPU[%d] dev=%d : %s\n", msg, _g,          \
                       (ctx)->debug ? 0 : _g, cudaGetErrorString(_err));               \
                exit(1);                                                               \
            }                                                                          \
        }                                                                              \
    } while (0)

#define CUDA_SYNC_CHECK_ALL(ctx, msg)                                                  \
    do {                                                                               \
        printf("\n[MAIN CUDA SYNC CHECK] %s\n", msg);                                  \
        (ctx)->sync();                                                                 \
        CUDA_CHECK_ALL(ctx, msg);                                                      \
    } while (0)

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T>
T get_max_disp(DeviceVec<T> &d_soln, int idof = 2) {
    T *h_soln = d_soln.createHostVec().getPtr();
    int nvars = d_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;

    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }

    return my_max;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // int level = 2;
    int level = 3;
    double SR = 300;

    if (argc > 1) {
        level = atoi(argv[1]);
    }

    using T = double;
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using Partitioner = TacsComponentGPUPartitioner;
    using Mat = GPUbsrmat<T, Partitioner>;
    using Vec = GPUvec<T, Partitioner>;

    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = GPU_MITCShellAssembler<T, Partitioner, Director, Basis, Physics>;

    using ASW = MultiGPUElementASW<T, Partitioner>;
    using Prolongation = MultiGPUUnstructuredProlongation<T, Assembler, Partitioner>;
    using CoarseSolver = SingleGPUDirectLU<T, Partitioner, Partitioner>;
    using GMG =
        MultiGPUGeometricMultigrid<T, Partitioner, Assembler, ASW, Prolongation, CoarseSolver>;

    using PCG = GPU_PCG<T, Partitioner, GMG>;

    const int block_dim = Physics::vars_per_node;

    auto ctx = new MultiGPUContext();
    int ngpus_override = 1;
    auto sgpu_ctx = new MultiGPUContext(ngpus_override);

    int device_count = ctx->ngpus;
    printf("#GPUs = %d\n", device_count);

    std::vector<Assembler *> assemblers;
    std::vector<Mat *> mats;
    std::vector<ASW *> smoothers;
    std::vector<Prolongation *> prolongations;

    Assembler *coarse_assembler = nullptr;
    CoarseSolver *coarse_solver = nullptr;

    Vec *fine_rhs = nullptr;
    Vec *fine_soln = nullptr;
    Vec *fine_test = nullptr;
    int fine_N = 0;

    // bool debug = true;
    bool debug = false;

    int lev_min = debug ? (level - 1) : 0;

    for (int i = level; i >= lev_min; i--) {
        TACSMeshLoader mesh_loader{comm};

        std::string fname = std::string(std::getenv("HOME")) +
                            "/git/gpu_fem/examples/gmg/3_aob_wing/meshes/aob_wing_L" +
                            std::to_string(i) + ".bdf";

        mesh_loader.scanBDFFile(fname.c_str());

        double E = 70e9;
        double nu = 0.3;
        double thick = 2.0 / SR;

        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
    CUDA_CHECK_ALL(ctx, "before create assembler");

    auto assembler = Assembler::createFromBDF(ctx, mesh_loader, Data(E, nu, thick));
    CUDA_SYNC_CHECK_ALL(ctx, "after create assembler");

        int nvars = assembler->get_num_vars();
        int nnodes = assembler->get_num_nodes();

        HostVec<T> h_loads(nvars);
        double load_mag = 1e3;
        double *my_loads = h_loads.getPtr();

        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        assemblers.push_back(assembler);

        auto part = assembler->getPartitioner();

        // auto kmat = new GPUbsrmat<T, Partitioner>(ctx, part, block_dim);
        // auto rhs = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        // auto soln = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        printf("[MAIN] create kmat\n");
        auto kmat = new GPUbsrmat<T, Partitioner>(ctx, part, block_dim);
        CUDA_SYNC_CHECK_ALL(ctx, "after create kmat");

        printf("[MAIN] create rhs\n");
        auto rhs = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        CUDA_SYNC_CHECK_ALL(ctx, "after create rhs");

        printf("[MAIN] create soln\n");
        auto soln = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        CUDA_SYNC_CHECK_ALL(ctx, "after create soln");

        int N = assembler->get_num_vars();

        // rhs->setValuesFromHost(my_loads);

        // assembler->add_jacobian(kmat);
        // assembler->apply_bcs(kmat);
        // assembler->apply_bcs(rhs);

        // auto test_vec = new GPUvec<T, Partitioner>(ctx, part, block_dim)

        printf("[MAIN] rhs setValuesFromHost\n");
        rhs->setValuesFromHost(my_loads);
        CUDA_SYNC_CHECK_ALL(ctx, "after rhs setValuesFromHost");

        printf("[MAIN] add_jacobian\n");
        assembler->add_jacobian(kmat);
        CUDA_SYNC_CHECK_ALL(ctx, "after add_jacobian");

        printf("[MAIN] apply_bcs(kmat)\n");
        assembler->apply_bcs(kmat);
        CUDA_SYNC_CHECK_ALL(ctx, "after apply_bcs(kmat)");

        printf("[MAIN] apply_bcs(rhs)\n");
        assembler->apply_bcs(rhs);
        CUDA_SYNC_CHECK_ALL(ctx, "after apply_bcs(rhs)");

        printf("[MAIN] create test_vec\n");
        auto test_vec = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        CUDA_SYNC_CHECK_ALL(ctx, "after create test_vec");;

        ctx->sync();

        mats.push_back(kmat);

        if (i == level) {
            fine_rhs = rhs;
            fine_soln = assembler->createGPUVec();
            fine_N = N;
            fine_test = test_vec;
        }

        T omega = 0.15;
        int nsmooth = 4;

        // auto smoother = new ASW(ctx, part, kmat, omega, nsmooth);
        // smoother->factor();

        printf("[MAIN] create ASW smoother\n");
        CUDA_CHECK_ALL(ctx, "before ASW constructor");
        auto smoother = new ASW(ctx, part, kmat, omega, nsmooth);
        CUDA_SYNC_CHECK_ALL(ctx, "after ASW constructor");

        printf("[MAIN] factor ASW smoother\n");
        smoother->factor();
        CUDA_SYNC_CHECK_ALL(ctx, "after ASW factor");

        smoothers.push_back(smoother);

        if (i == lev_min) {
            TACSMeshLoader mesh_loader2{comm};
            mesh_loader2.scanBDFFile(fname.c_str());

            auto sgpu_assembler =
                Assembler::createFromBDF(sgpu_ctx, mesh_loader2, Data(E, nu, thick));

            coarse_assembler = sgpu_assembler;

            auto sgpu_part = sgpu_assembler->getPartition();

            auto sgpu_mat = new GPUbsrmat<T, Partitioner>(sgpu_ctx, sgpu_part, block_dim);

            sgpu_assembler->add_jacobian(sgpu_mat);
            sgpu_assembler->apply_bcs(sgpu_mat);

            coarse_solver = new CoarseSolver(ctx, part, sgpu_part, sgpu_mat);
            coarse_solver->factor();
        }

        if (i == level) {
            fine_rhs->setValuesFromHost(my_loads);
            assemblers[0]->apply_bcs(fine_rhs);
        }

        CHECK_CUDA(cudaGetLastError());
        ctx->sync();
    }

    int nlevels = assemblers.size();

    for (int i = 0; i < nlevels - 1; i++) {
        auto fine_assembler_i = assemblers[i];
        auto coarse_assembler_i = assemblers[i + 1];

        auto fine_part_i = fine_assembler_i->getPartition();
        auto coarse_part_i = coarse_assembler_i->getPartition();

        printf("level %d: create prolongation\n", i);

        int ELEM_MAX = 10;

        auto prolongation =
            new Prolongation(ctx, fine_part_i, coarse_part_i, fine_assembler_i,
                             coarse_assembler_i, block_dim, mats[i], mats[i + 1], ELEM_MAX);

        prolongations.push_back(prolongation);
    }

    auto fine_assembler = assemblers[0];
    auto fine_part = fine_assembler->getPartition();
    auto fine_kmat = mats[0];

    int NSTEPS = 1;
    T rtol = 1e-6;
    T atol = 1e-30;
    T LS_min = 1e-2;
    T LS_max = 2.0;
    bool PRINT = false;
    // bool PRINT = true;
    int print_freq = 10;

    auto gmg = new GMG(ctx, assemblers, mats, smoothers, prolongations, coarse_solver, NSTEPS,
                       rtol, atol, PRINT, print_freq, LS_min, LS_max);

    if (level <= 3 && debug) {
        int max_node = 20;

        T nrm1 = fine_rhs->norm();
        printf("fine rhs (nrm = %.8e)\n", nrm1);
        fine_rhs->printValuesOnHost(max_node);

        auto fine_defect = assemblers[0]->createGPUVec();
        fine_rhs->copyTo(fine_defect);

        smoothers[0]->smoothDefect(fine_defect, fine_soln);

        T nrm2 = fine_defect->norm();
        printf("fine defect after pre-smooth (nrm = %.8e)\n", nrm2);
        fine_defect->printValuesOnHost(max_node);

        T nrm3 = fine_soln->norm();
        printf("fine soln after pre-smooth (nrm = %.8e)\n", nrm3);
        fine_soln->printValuesOnHost(max_node);

        auto crs_defect = assemblers[1]->createGPUVec();

        prolongations[0]->restrict_vec(fine_defect, crs_defect);
        assemblers[1]->apply_bcs(crs_defect);

        T nrm4 = crs_defect->norm();
        printf("crs defect (nrm = %.8e)\n", nrm4);
        crs_defect->printValuesOnHost(max_node);

        auto crs_soln = assemblers[1]->createGPUVec();

        coarse_solver->solve(crs_defect, crs_soln);

        T nrm5 = crs_soln->norm();
        printf("crs soln (nrm = %.8e)\n", nrm5);
        crs_soln->printValuesOnHost(max_node);

        auto fine_temp = assemblers[0]->createGPUVec();
        auto fine_soln_dbg = assemblers[0]->createGPUVec();

        crs_soln->copyTo(gmg->d_solns[1]);
        fine_defect->copyTo(gmg->d_defects[0]);
        fine_soln_dbg->copyTo(gmg->d_solns[0]);

        prolongations[0]->prolongate(crs_soln, fine_soln_dbg);

        T nrm6 = fine_soln_dbg->norm();
        printf("prolong fine soln (nrm = %.8e)\n", nrm6);
        fine_soln_dbg->printValuesOnHost(max_node);

        gmg->prolongate_line_search(0);

        gmg->d_solns[0]->copyTo(fine_soln_dbg);

        T nrm7 = fine_soln_dbg->norm();
        printf("prolongfine soln w line search (nrm = %.8e)\n", nrm7);
        fine_soln_dbg->printValuesOnHost(max_node);

        return 0;
    }

    auto pc = gmg;

    auto pcg = new PCG(ctx, fine_part, fine_kmat, pc, fine_N, block_dim);

    int max_iter = 500;
    int print_freq2 = 10;
    T rtol2 = 1e-6;
    T atol2 = 1e-30;
    bool can_print = true;

    int exp_iters =
        pcg->solve(fine_rhs, fine_soln, max_iter, atol2, rtol2, print_freq2, can_print);

    printf("print to VTK\n");

    T *h_soln = new T[fine_N];
    memset(h_soln, 0, fine_N * sizeof(T));

    fine_soln->getValuesToHost(h_soln);

    printToVTK_v2<T, Assembler>(*fine_assembler, h_soln, "./out/wing_gmg6.vtk");

    printf("\tdone with print to VTK\n");

    MPI_Finalize();
    return 0;
}