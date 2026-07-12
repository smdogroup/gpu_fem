#include "assembler/gpu_assembler.h"
#include "assembler/gpu_mitc_shell.h"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "partition/structured_gpu_partitioner.h"
#include "smoothers/gpu_asw.h"
#include "utils/fea.h"
#include "utils/gpu_print_vtk.h"
#include "utils/multigpu_context.h"

// shell imports
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// BDDC imports
#include "domdec/bddc_lu.h"
#include "domdec/subdomains/unstructured_iev.h"
#include "partition/subdomain_partitioner.h"
#include "solvers/gpu_pcg_matfree.h"

// unstruct splitting methods
#include <chrono>
#include <string>

#include "domdec/subdomains/unstruct/corner_sep.h"
#include "domdec/subdomains/unstruct/corner_sep_opt.h"
#include "domdec/subdomains/unstruct/corner_sep_opt2.h"
#include "domdec/subdomains/unstruct/corner_sep_opt3.h"
#include "domdec/subdomains/unstruct/metis_sep.h"
#include "domdec/subdomains/unstruct/edge_expand.h"
#include "domdec/subdomains/unstruct/edge_expand2.h"
#include "domdec/subdomains/unstruct/flood.h"
#include "domdec/subdomains/unstruct/macro_elem.h"
#include "domdec/subdomains/unstruct/macro_elem2.h"
#include "domdec/subdomains/unstruct/v1.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

template <typename T>
struct ObliqueCylinderLoad {
    __HOST_DEVICE__
    T operator()(T x, T y, T z) const {
        T x_hat = x;
        T th = atan2(y, z);

        T load_mag =
            1.0e2 *
            (0.3 * cos(5.0 * th + 2.0 * M_PI * x_hat) +
             0.7 * cos(10.0 * th + M_PI / 6.0 +
                       5.3 * M_PI * x_hat)) *
            sin(5.0 * M_PI * x_hat + x_hat * x_hat);

        return load_mag;
    }
};

template <typename Splitter>
int run_solver(int nxe, int target_sd_size, double thick, double mag) {
    // NOTE: this version uses inner direct solvers

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
    using Assembler =
        GPU_MITCShellAssembler<T, Partitioner, Director, Basis, Physics>;

    using IEVSplit = UnstructuredIEVSplitting<Splitter>;
    using BDDC =
        MultiGPUBDDC_LUSolver<T, Assembler, Partitioner, IEVSplit>;
    using PCG_MatFree =
        GPU_PCGMatfree<T, SDPartitioner, BDDC, BDDC>;

    const int block_dim = Physics::vars_per_node;
    auto ctx = new MultiGPUContext();

    printf("#GPUs = %d\n", ctx->ngpus);

    double L = 1.0, R = 0.5;
    double E = 70e9, nu = 0.3;
    bool imperfection = false;
    int imp_x = 1, imp_hoop = 1;

    auto assembler = createGPUCylinderAssembler<Assembler>(
        ctx, nxe, nxe, L, R, E, nu, thick,
        imperfection, imp_x, imp_hoop);

    auto part = assembler->getPartitioner();

    auto kmat = new GPUbsrmat<T, Partitioner>(
        ctx, part, block_dim);
    auto soln = new GPUvec<T, Partitioner>(
        ctx, part, block_dim);
    auto vars = new GPUvec<T, Partitioner>(
        ctx, part, block_dim);

    int N = assembler->get_num_vars();

    assembler->add_jacobian(kmat);
    ctx->sync();
    assembler->apply_bcs(kmat);

    auto split = new IEVSplit(
        assembler->get_num_elements(),
        assembler->get_num_nodes(),
        Basis::num_nodes,
        part->h_elem_conn,
        target_sd_size);

    bool debug = true;
    auto bddc =
        new BDDC(ctx, part, assembler, kmat, split, debug);

    ObliqueCylinderLoad<T> load;
    bddc->add_subdomain_fext(load, mag);
    bddc->set_IEV_residual(1.0, 0.0, vars);
    bddc->update_after_assembly(vars);

    auto gam_rhs = bddc->createGamVec();
    auto gam_soln = bddc->createGamVec();

    bddc->get_lam_rhs(gam_rhs);

    auto sd_part = bddc->get_part_gam();

    auto pcg = new PCG_MatFree(
        ctx, sd_part, bddc, bddc, N, block_dim,
        "BDDC-Interface");

    int max_iter = 500;
    int print_freq = 1;
    T rtol = 1e-6;
    T atol = 1e-30;
    bool can_print = true;

    pcg->solve(
        gam_rhs, gam_soln, max_iter, atol, rtol,
        print_freq, can_print);

    bddc->get_global_soln(gam_soln, soln);

    T *h_soln = new T[N]();
    soln->getValuesToHost(h_soln);

    printToVTK_v2<T, Assembler>(
        *assembler, h_soln, "./out/1_struct_cyl.vtk");

    delete[] h_soln;
    return 0;
}

int main(int argc, char **argv) {
    int nxe = 100;
    // here subdomain is #elements in subdomain (not side of something)
    int subdomain = 49; // default a bit larger to work better for METIS
    double thick = 1e-3;
    double mag = 1e2;
    std::string splitter = "metis-opt";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--nxe" && i + 1 < argc) {
            nxe = std::atoi(argv[++i]);
        } else if (arg == "--subdomain" && i + 1 < argc) {
            subdomain = std::atoi(argv[++i]);
        } else if (arg == "--thick" && i + 1 < argc) {
            thick = std::atof(argv[++i]);
        } else if (arg == "--mag" && i + 1 < argc) {
            mag = std::atof(argv[++i]);
        } else if (arg == "--splitter" && i + 1 < argc) {
            splitter = argv[++i];
        } else {
            std::cerr
                << "Usage: " << argv[0]
                << " [--splitter name]"
                << " [--nxe int]"
                << " [--subdomain int]"
                << " [--thick value]"
                << " [--mag value]\n";
            return 1;
        }
    }

#define RUN_SPLITTER(name, type)                    \
    if (splitter == name)                           \
        return run_solver<type>(                    \
            nxe, subdomain, thick, mag)

    RUN_SPLITTER("metis", MetisCornerOptSplitter<false>);
    RUN_SPLITTER("metis-opt", MetisCornerOptSplitter<true>);
    RUN_SPLITTER("v1", UnstructSplitterV1);
    RUN_SPLITTER("flood", UnstructFloodFillSplitter);
    RUN_SPLITTER("edge", UnstructEdgeFillSplitter);
    RUN_SPLITTER("edge2", UnstructEdgeFillSplitterV2);
    RUN_SPLITTER("corner", CornerSeparatingSplitter);
    RUN_SPLITTER("macro", UnstructMacroElementSplitter);
    RUN_SPLITTER("compact", UnstructCompactPatchSplitter);
    RUN_SPLITTER("opt", GlobalLocalCornerOptSplitter);
    RUN_SPLITTER("opt2", GlobalLocalCornerOptSplitterV2);
    RUN_SPLITTER("opt3", GlobalLocalCornerOptSplitterV3);

#undef RUN_SPLITTER

    std::cerr << "Unknown splitter: " << splitter << '\n';
    std::cerr
        << "Options: metis, metis-opt, v1, flood, edge, edge2, corner, "
        << "macro, compact, opt, opt2, opt3\n";

    return 1;
}