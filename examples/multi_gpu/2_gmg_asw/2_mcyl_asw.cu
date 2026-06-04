#include "assembler/gpu_assembler.h"
#include "smoothers/gpu_asw.h"
#include "assembler/gpu_mitc_shell.h"
#include "solvers/gpu_pcg.h"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/gpu_print_vtk.h"
#include "utils/fea.h"
#include "utils/multigpu_context.h"
#include "partition/structured_gpu_partitioner.h"


// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/basis/lagrange_basis.h"

#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include <string>
#include <chrono>


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
    int nxe = 128; // default
    double SR = 1e1;
    // double SR = 1e3;
    double pressure = 8e6;

    // just type ./2_multi_asw.out 256 >> out.txt for instance (no --nxe)
    if (argc > 1) {
        nxe = atoi(argv[1]);
    }

    // ---------------------------------------------
    // type declarations
    // ---------------------------------------------

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true; // this is a nonlinear GMG case
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using Partitioner = StructuredGPUPartitioner;

    // have to use MITC4 shells cause this is before diff element types in paper
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = GPU_MITCShellAssembler<T, Partitioner, Director, Basis, Physics>;

    // preconditioner and solver
    using ASW = MultiGPUElementASW<T, Partitioner>;
    using PCG = GPU_PCG<T, Partitioner, ASW>;

    const int block_dim = Physics::vars_per_node;

    // ---------------------------------------------
    // start multi GPU device context
    // ---------------------------------------------
    auto ctx = new MultiGPUContext();

    // ctx->ngpus = 4; // debug

    int device_count = ctx->ngpus;
    printf("#GPUs = %d\n", device_count);


    // ---------------------------------------------
    // create FEA problem
    // ---------------------------------------------

    double L = 1.0, R = 0.5, thick = L / SR;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    printf("create GPU cylinder assembler\n");
    auto assembler = createGPUCylinderAssembler<Assembler>(ctx, nxe, nxe, L, R, E, nu, thick, 
        imperfection, imp_x, imp_hoop);
    constexpr bool compressive = false;
    const int load_case = 3; // petal and chirp load
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    printf("create cylinder loads\n");
    T *my_loads = getCylinderLoads2<T, Basis, Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // ---------------------------------------------
    // get mesh partitioner
    printf("get mesh partitioner\n");
    auto part = assembler->getPartitioner();
    
    // build matrix and vectors
    // ---------------------------------------------
    printf("make GPUbsrmat\n");
    auto kmat = new GPUbsrmat<T, Partitioner>(ctx, part, block_dim);
    printf("make GPUvecs\n");
    auto rhs = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    auto soln = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    int N = assembler->get_num_vars();


    // ---------------------------------------------
    // assemble the jacobian and get rhs
    // ---------------------------------------------
    printf("rhs->setValuesFromHost\n");
    rhs->setValuesFromHost(my_loads);
    printf("add jacobian\n");
    assembler->add_jacobian(kmat);

    printf("add jacobian post-sync\n");
    ctx->sync();

    if (nxe * nxe <= 100) {
        printf("kmat before bcs\n");
        assembler->printMatrixOnHost(kmat);
    }

    printf("apply bcs to kmat\n");
    assembler->apply_bcs(kmat);
    printf("apply bcs to rhs\n");
    assembler->apply_bcs(rhs);

    if (nxe * nxe <= 30) {
        printf("kmat after bcs\n");
        assembler->printMatrixOnHost(kmat);
    }

    if (nxe * nxe <= 100) {
        T rhs_norm = rhs->norm();
        printf("rhs vec after bcs with norm %.4e\n", rhs_norm);
        rhs->printValuesOnHost();
    }

    // ------------------------------
    // prelim DEBUG part 1
    // ------------------------------

    auto test_vec = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    kmat->mult(rhs, test_vec);

    if (nxe * nxe <= 100) {
        T test_nrm = test_vec->norm();
        printf("test mat-vec with nrm %.8e\n", test_nrm);
        test_vec->printValuesOnHost();
    }

    // ---------------------------------------------
    // build the ASW preconditioner
    // ---------------------------------------------
    T omega = 0.2;
    int nsmooth = 2;
    printf("build ASW preconditioner\n");
    auto pc = new ASW(ctx, part, kmat, omega, nsmooth);
    printf("build PCG\n");
    const char *precond_name = "ASW";
    auto pcg = new PCG(ctx, part, kmat, pc, N, block_dim, precond_name);
    printf("\tdone build PCG\n");

    // ---------------------------------------------
    // perform the linear solve
    // ---------------------------------------------

    // factor before solve
    printf("factor ASW precond\n");
    pc->factor();

    // test precond
    if (nxe * nxe <= 100) {
        test_vec->zero();
        test_vec->zeroLocal();
        pc->solve(rhs, test_vec);

        T test_nrm = test_vec->norm();
        printf("test precond-vec with nrm %.8e\n", test_nrm);
        test_vec->printValuesOnHost();

        printf("ASW printMatValues\n");
        pc->printSubdomainMatValues();

        // reset host values after debug
        rhs->setValuesFromHost(my_loads);
    }


    // then solve
    int max_iter = 500, print_freq = 3;
    T rtol = 1e-6, atol = 1e-30;
    bool can_print = true;

    printf("begin PCG solve\n");
    int exp_iters = pcg->solve(rhs, soln, max_iter, atol, rtol, print_freq, can_print);

    // ---------------------------------------------
    // get solution and print to VTK on host
    // ---------------------------------------------

    // get host solution
    T *h_soln = new T[N];
    memset(h_soln, 0, N * sizeof(T));
    soln->getValuesToHost(h_soln);
    printToVTK_v2<T, Assembler>(*assembler, h_soln, "./out/cyl_mgpu.vtk");
};