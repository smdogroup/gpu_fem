#include "assembler/gpu_assembler.h"
#include "smoothers/gpu_asw.h"
#include "assembler/gpu_mitc_shell.h"
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

// BDDC imports
#include "domdec/subdomains/unstructured_iev.h"
#include "partition/subdomain_partitioner.h"
#include "domdec/bddc_lu.h"
#include "solvers/gpu_pcg_matfree.h"

// unstruct splitting methods
// #include "domdec/subdomains/unstruct/edge_expand.h"
// #include "domdec/subdomains/unstruct/edge_expand2.h"
// #include "domdec/subdomains/unstruct/flood.h"
#include "domdec/subdomains/unstruct/v1.h"
// #include "domdec/subdomains/unstruct/corner_sep.h"

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
T get_max_disp(HostVec<T> h_soln, int idof = 2) {
    T *h_soln_ptr = h_soln.getPtr();
    int nvars = h_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln_ptr[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}


template <typename T>
struct ObliqueCylinderLoad {

    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        T x_hat = x / 1.0; // assumes L = 1.0
        T th = atan2(y, z);
        T th_hat = th / 2 / M_PI;
        T mag = 1.0e2 *
                (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                    0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);
        return mag;
    }
};


int main(int argc, char **argv) {
    // NOTE : this version uses inner direct solvers

    // shell type
    using T = double;
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    using Partitioner = StructuredGPUPartitioner;
    using SDPartitioner = SubdomainGPUPartitioner<Partitioner>;

    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = GPU_MITCShellAssembler<T, Partitioner, Director, Basis, Physics>;

    // splitter methods
    using Splitter = UnstructSplitterV1;
    // // using Splitter = UnstructFloodFillSplitter;
    // // using Splitter = UnstructEdgeFillSplitter;
    // // using Splitter = UnstructEdgeFillSplitterV2;
    // using Splitter = CornerSeparatingSplitter;

    // preconditioner and solver
    using IEVSplit = UnstructuredIEVSplitting<Splitter>;
    using BDDC = MultiGPUBDDC_LUSolver<T, Assembler, Partitioner, IEVSplit>;
    using PCG_MatFree = GPU_PCGMatfree<T, SDPartitioner, BDDC, BDDC>;

    int nxe = 6, nxe_subdomain_size = 2;
    // int nxe = 256, nxe_subdomain_size = 4; // 8 subdomains slightly faster (cause shrinks coarse problem) for local + HPC
    // int nxe = 512, nxe_subdomain_size = 4; // 8 subdomains slightly faster (cause shrinks coarse problem) for local + HPC
    T thick = 1e-3;
    T mag = 1.0e2;

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
        } else if (strcmp(arg, "--thick") == 0) {
            if (i + 1 < argc) {
                thick = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --thick\n";
                return 1;
            }
        } else if (strcmp(arg, "--mag") == 0) {
            if (i + 1 < argc) {
                mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --mag\n";
                return 1;
            }
        } else if (strcmp(arg, "--subdomain") == 0) {
            if (i + 1 < argc) {
                nxe_subdomain_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --subdomain\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/krylov] [--nxe value] [--SR value] [--nsmooth int]" << std::endl;
            return 1;
        }
    }


    // =================

    // ---------------------------------------------
    // start multi GPU device context
    // ---------------------------------------------
    const int block_dim = Physics::vars_per_node;
    auto ctx = new MultiGPUContext();

    // ctx->ngpus = 4; // debug

    int device_count = ctx->ngpus;
    printf("#GPUs = %d\n", device_count);

    // ---------------------------------------------
    // create FEA problem
    // ---------------------------------------------

    double L = 1.0, R = 0.5;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    // printf("create GPU cylinder assembler\n");
    auto assembler = createGPUCylinderAssembler<Assembler>(ctx, nxe, nxe, L, R, E, nu, thick, 
        imperfection, imp_x, imp_hoop);

    // ---------------------------------------------
    // get mesh partitioner
    // printf("get mesh partitioner\n");
    auto part = assembler->getPartitioner();
    
    // build matrix and vectors
    // ---------------------------------------------
    // printf("make GPUbsrmat\n");
    auto kmat = new GPUbsrmat<T, Partitioner>(ctx, part, block_dim);
    // printf("make GPUvecs\n");
    // auto rhs = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    auto soln = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    auto vars = new GPUvec<T, Partitioner>(ctx, part, block_dim);
    int N = assembler->get_num_vars();


    // ---------------------------------------------
    // assemble the jacobian and get rhs
    // ---------------------------------------------
    // printf("rhs->setValuesFromHost\n");
    // rhs->setValuesFromHost(my_loads);
    // printf("add jacobian\n");
    assembler->add_jacobian(kmat);

    // printf("add jacobian post-sync\n");
    ctx->sync();
    // printf("apply bcs to kmat\n");
    assembler->apply_bcs(kmat);
    // printf("apply bcs to rhs\n");
    // assembler->apply_bcs(rhs);

    // ---------------------------

    // build IEV subdomain splitting
    int num_elements = assembler->get_num_elements();
    int num_nodes = assembler->get_num_nodes();
    int nodes_per_elem = Basis::num_nodes;
    int *h_elem_conn = part->h_elem_conn; // comes directly from mesh loader on host + root
    int target_sd_size = nxe_subdomain_size * nxe_subdomain_size;
    // printf("[MAIN] make IEV splitting class\n");
    auto split = new IEVSplit(num_elements, num_nodes, nodes_per_elem, h_elem_conn,
                            target_sd_size);
    // printf("[MAIN] done making IEV splitting class\n");

    // printf("[MAIN] make BDDC class\n");
    bool debug = true;
    // bool debug = false;
    auto bddc = new BDDC(ctx, part, assembler, kmat, split, debug);
    // printf("[MAIN] done making BDDC class\n");

    // printf("[MAIN] add_subdomain_fext\n");
    ObliqueCylinderLoad<T> load;
    bddc->add_subdomain_fext(load, mag);
    // printf("[MAIN] done add_subdomain_fext\n");

    // printf("[MAIN] set_IEV_residual\n");
    bddc->set_IEV_residual(1.0, 0.0, vars);
    // printf("[MAIN] done set_IEV_residual\n");

    // printf("[MAIN] update_after_assembly\n");
    bddc->update_after_assembly(vars);
    // printf("[MAIN] done update_after_assembly\n");

    // lambda rhs (TODO : fix this later so it can do multi-GPU)
    // VecType<T> gam_rhs(bddc->getLambdaSize(0));
    // VecType<T> gam(bddc->getLambdaSize(0));
    auto gam_rhs = bddc->createGamVec();
    auto gam_soln = bddc->createGamVec();
    // printf("[MAIN] get_lam_rhs\n");
    bddc->get_lam_rhs(gam_rhs);
    // printf("[MAIN] done get_lam_rhs\n");

    // return; // debug

    auto sd_part = bddc->get_part_gam();
    
    // build the matrix-free PCG solver now
    // printf("[MAIN] build PCG_MatFree\n");
    const char *precond_name = "BDDC-Interface";
    auto pcg = new PCG_MatFree(ctx, sd_part, bddc, bddc, N, block_dim, precond_name);
    // printf("[MAIN] done build PCG_MatFree\n");

    // then solve
    int max_iter = 500;
    // int max_iter = 5; // temp debug
    int print_freq = 1;
    T rtol = 1e-6, atol = 1e-30;
    bool can_print = true;

    // printf("begin PCG solve\n");
    int exp_iters = pcg->solve(gam_rhs, gam_soln, max_iter, atol, rtol, print_freq, can_print);

    bddc->get_global_soln(gam_soln, soln);

    // ---------------------------------------------
    // get solution and print to VTK on host
    // ---------------------------------------------

    // get host solution
    T *h_soln = new T[N];
    memset(h_soln, 0, N * sizeof(T));
    soln->getValuesToHost(h_soln);
    printToVTK_v2<T, Assembler>(*assembler, h_soln, "./out/cyl_mgpu.vtk");

    return 0;
}
