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
#include "partition/structured_gpu_partitioner.h"
#include "prolongation/gpu_sprolong.h"
#include "solvers/sgpu_direct.h"
#include "solvers/gmg.h"


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
    // int nxe = 64;
    // int nxe = 8;

    // double SR = 1e1;
    double SR = 1e2;
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
    using Mat = GPUbsrmat<T, Partitioner>;
    using Vec = GPUvec<T, Partitioner>;

    // have to use MITC4 shells cause this is before diff element types in paper
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = GPU_MITCShellAssembler<T, Partitioner, Director, Basis, Physics>;

    // multigrid objects
    using ASW = MultiGPUElementASW<T, Partitioner>;
    using Prolongation = MultiGPUStructuredProlongation<T, Partitioner, Basis, CYLINDER>;
    using CoarseSolver = SingleGPUDirectLU<T, Partitioner, Partitioner>;
    using GMG = MultiGPUGeometricMultigrid<T, Partitioner, Assembler, ASW, Prolongation, CoarseSolver>;

    // outer krylov
    using PCG = GPU_PCG<T, Partitioner, GMG>;

    const int block_dim = Physics::vars_per_node;


    int nxe_min = (nxe > 32) ? 32 : (nxe / 2);

    // ---------------------------------------------
    // start multi GPU device context
    // ---------------------------------------------
    auto ctx = new MultiGPUContext();
    int ngpus_override = 1; // for single gpu format
    auto sgpu_ctx = new MultiGPUContext(ngpus_override); // for coarse direct solver

    int device_count = ctx->ngpus;
    printf("#GPUs = %d\n", device_count);


    // create multigrid mesh hierarchy
    std::vector<Assembler*> assemblers;
    std::vector<Mat*> mats;
    std::vector<ASW*> smoothers;
    std::vector<Prolongation*> prolongations;
    std::vector<int> nxe_vec;
    Assembler* coarse_assembler;
    CoarseSolver *coarse_solver;

    Vec *fine_rhs, *fine_soln, *fine_test;
    int fine_N;

    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {

        nxe_vec.push_back(c_nxe);

        // create FEA assembler
        double L = 1.0, R = 0.5, thick = L / SR;
        double E = 70e9, nu = 0.3;
        // double rho = 2500, ys = 350e6;
        bool imperfection = false; // option for geom imperfection
        int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
        printf("nxe=%d: create GPU cylinder assembler\n", c_nxe);
        auto assembler = createGPUCylinderAssembler<Assembler>(ctx, c_nxe, c_nxe, L, R, E, nu, thick, 
            imperfection, imp_x, imp_hoop);
        constexpr bool compressive = false;
        const int load_case = 3; // petal and chirp load
        double uniform_force = pressure * 1.0 * 1.0;
        double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
        nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
        double Q = 1.0; // load magnitude
        printf("\tcreate cylinder loads\n");
        T *my_loads = getCylinderLoads2<T, Basis, Physics, load_case>(c_nxe, c_nxe, L, R, nodal_loads);

        assemblers.push_back(assembler);

        
        // ---------------------------------------------
        // get mesh partitioner
        // printf("\tget mesh partitioner\n");
        auto part = assembler->getPartitioner();
        
        // build matrix and vectors
        // ---------------------------------------------
        // printf("\tmake GPUbsrmat\n");
        auto kmat = new GPUbsrmat<T, Partitioner>(ctx, part, block_dim);
        // printf("\tmake GPUvecs\n");
        auto rhs = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        auto soln = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        int N = assembler->get_num_vars();

        // ---------------------------------------------
        // assemble the jacobian and get rhs
        // ---------------------------------------------
        // printf("\trhs->setValuesFromHost\n");
        rhs->setValuesFromHost(my_loads);
        printf("\tadd jacobian\n");
        assembler->add_jacobian(kmat);
        assembler->apply_bcs(kmat);
        assembler->apply_bcs(rhs);

        auto test_vec = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        kmat->mult(rhs, test_vec);

        // if (nxe * nxe <= 100) {
        //     // printf("kmat before bcs\n");
        //     // assembler->printMatrixOnHost(kmat);
        //     T rhs_norm = rhs->norm();
        //     printf("rhs vec [nxe=%d] after bcs with norm %.4e\n", c_nxe, rhs_norm);
        //     rhs->printValuesOnHost();

        //     T test_nrm = test_vec->norm();
        //     printf("test mat-vec [nxe=%d] with nrm %.8e\n", c_nxe, test_nrm);
        //     test_vec->printValuesOnHost();
        // }
        
        ctx->sync();

        mats.push_back(kmat);
        if (c_nxe == nxe) {
            fine_rhs = rhs;
            fine_soln = assembler->createGPUVec();
            fine_N = N;
            fine_test = test_vec;
        }

        // ---------------------------------------------
        // build the ASW smoother
        // ---------------------------------------------
        // T omega = 0.2;
        T omega = 0.15;
        int nsmooth = 2;
        // printf("\tbuild ASW preconditioner\n");
        auto smoother = new ASW(ctx, part, kmat, omega, nsmooth);
        // printf("\tASW->factor()\n");
        smoother->factor();
        smoothers.push_back(smoother);

        // if (nxe * nxe <= 100) {
        //     test_vec->zero();
        //     test_vec->zeroLocal();
        //     smoother->solve(rhs, test_vec);

        //     T test_nrm = test_vec->norm();
        //     printf("test precond-vec [nxe=%d] with nrm %.8e\n", c_nxe, test_nrm);
        //     test_vec->printValuesOnHost();

        //     // reset host values after debug
        //     rhs->setValuesFromHost(my_loads);
        // }

        // build coarse solver
        if (c_nxe == nxe_min) {
            // rebuild assembler in SingleGPU partition format
            // TODO : need some way to build single GPU partitioned assembler here..
            // printf("coarse mesh: create single GPU cylinder assembler\n");
            auto sgpu_assembler = createGPUCylinderAssembler<Assembler>(sgpu_ctx, c_nxe, c_nxe, L, R, E, nu, thick, 
                imperfection, imp_x, imp_hoop);
            coarse_assembler = sgpu_assembler;
            auto sgpu_part = sgpu_assembler->getPartition();
            // TODO : does this make a matrix on a singleGPU?
            // need a different context too?
            // printf("\tcoarse mesh: single GPU add jacobian\n");
            auto sgpu_mat = new GPUbsrmat<T, Partitioner>(sgpu_ctx, sgpu_part, block_dim);
            sgpu_assembler->add_jacobian(sgpu_mat);
            sgpu_assembler->apply_bcs(sgpu_mat);

            // printf("\tcoarse mesh: build CoarseSolver\n");
            coarse_solver = new CoarseSolver(ctx, part, sgpu_part, sgpu_mat);
            coarse_solver->factor();

            // if (nxe * nxe <= 100) {
            //     test_vec->zero();
            //     test_vec->zeroLocal();
            //     sgpu_mat->mult(rhs, test_vec);

            //     T test_nrm = test_vec->norm();
            //     printf("test SingleGPU coarse mat-vec [nxe=%d] with nrm %.8e\n", c_nxe, test_nrm);
            //     test_vec->printValuesOnHost();

            //     test_vec->zero();
            //     test_vec->zeroLocal();
            //     coarse_solver->solve(rhs, test_vec);
            //     T test_nrm2 = test_vec->norm();
            //     printf("test SingleGPU coarse LUsolve-vec [nxe=%d] with nrm %.8e\n", c_nxe, test_nrm2);
            //     test_vec->printValuesOnHost();

            //     // reset host values after debug
            //     rhs->setValuesFromHost(my_loads);
            // }
        }

        if (c_nxe == nxe) {
            fine_rhs->setValuesFromHost(my_loads);
            assemblers[0]->apply_bcs(fine_rhs);
        }
    }

    // now build prolongations (on fine-coarse pairs)
    int nlevels = assemblers.size();
    for (int level = 0; level < nlevels - 1; level++) {
        auto fine_assembler = assemblers[level];
        auto coarse_assembler = assemblers[level+1];

        int nxe_fine = nxe_vec[level], nxe_coarse = nxe_vec[level+1];
        auto fine_part = fine_assembler->getPartition();
        auto coarse_part = coarse_assembler->getPartition();

        printf("level %d: create prolongation\n", level);
        auto prolongation = new Prolongation(ctx, fine_part, coarse_part, 
            nxe_fine, nxe_coarse, block_dim, mats[level], mats[level+1]);
        // printf("\tdone create prolongation on level %d\n", level);
        prolongations.push_back(prolongation);
    }

    auto fine_assembler = assemblers[0];
    auto fine_part = fine_assembler->getPartition();
    auto fine_kmat = mats[0];


    // -------------------------------------------
    // now build final GMG object
    int NSTEPS = 1; // for K-cycle GMG just one V-cycle per solve
    // int NSTEPS = 100;
    T rtol = 1e-6, atol = 1e-30, LS_min = 1e-2, LS_max = 2.0;
    bool PRINT = false; // no print on Vcyc for K-cycle GMG
    int print_freq = 10; // but not printing anyways
    // printf("Build GMG object\n");
    auto gmg = new GMG(ctx, assemblers, mats, smoothers, prolongations, coarse_solver, 
        NSTEPS, rtol, atol, PRINT, print_freq, LS_min, LS_max);

    // -------------------------------------------
    // DEBUG section
    // -------------------------------------------

    if (nxe * nxe < 400) {
        printf("fine_rhs\n");
        fine_rhs->printValuesOnHost();
        auto fine_defect = assemblers[0]->createGPUVec();
        fine_rhs->copyTo(fine_defect);

        // pre-smooth
        smoothers[0]->smoothDefect(fine_defect, fine_soln);
        printf("fine_defect after pre-smooth\n");
        fine_defect->printValuesOnHost();
        printf("fine_soln after pre-smooth\n");
        fine_soln->printValuesOnHost();

        auto crs_defect = assemblers[1]->createGPUVec();
        prolongations[0]->restrict_vec(fine_defect, crs_defect);
        assemblers[1]->apply_bcs(crs_defect);
        printf("crs defect after restrict\n");
        crs_defect->printValuesOnHost();

        // do coarse soln
        auto crs_soln = assemblers[1]->createGPUVec();
        coarse_solver->solve(crs_defect, crs_soln);
        printf("crs soln\n");
        crs_soln->printValuesOnHost();

        // prolongate
        auto fine_temp = assemblers[0]->createGPUVec();
        auto fine_soln = assemblers[0]->createGPUVec();
        crs_soln->copyTo(gmg->d_solns[1]);
        fine_defect->copyTo(gmg->d_defects[0]);
        fine_soln->copyTo(gmg->d_solns[0]);

        prolongations[0]->prolongate(crs_soln, fine_soln);
        printf("fine soln w reg prolong\n");
        fine_soln->printValuesOnHost();

        gmg->prolongate_line_search(0);
        gmg->d_solns[0]->copyTo(fine_soln);
        printf("fine soln with prolong line search\n");
        fine_soln->printValuesOnHost();
    }

    // return;

    // ------------------------------------------
    // now build Krylov PCG
    // ------------------------------------------

    auto pc = gmg;   
    // printf("build PCG\n");

    // debug
    // gmg->MAX_STEPS = 100;
    // gmg->solve(fine_rhs, fine_soln);

    const char *precond_name = "GMG-ASW(struct)";
    auto pcg = new PCG(ctx, fine_part, fine_kmat, pc, fine_N, block_dim, precond_name);
    // printf("\tdone build PCG\n");

    // ---------------------------------------------
    // perform the linear solve
    // ---------------------------------------------

    // PCG solver settings
    int max_iter = 500, print_freq2 = 3;
    T rtol2 = 1e-6, atol2 = 1e-30;
    bool can_print = true;

    // printf("begin PCG solve\n");
    int exp_iters = pcg->solve(fine_rhs, fine_soln, max_iter, atol2, rtol2, print_freq2, can_print);

    // ---------------------------------------------
    // get solution and print to VTK on host
    // ---------------------------------------------

    // get host solution
    T *h_soln = new T[fine_N];
    memset(h_soln, 0, fine_N * sizeof(T));
    fine_soln->getValuesToHost(h_soln);
    printToVTK_v2<T, Assembler>(*fine_assembler, h_soln, "./out/cyl_gmg4.vtk");

    // FREE
    // TODO : free section
};