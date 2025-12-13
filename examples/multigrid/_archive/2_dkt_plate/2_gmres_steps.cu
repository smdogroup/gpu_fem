/* investigate near identity load maps with Qorder + GMRES */

// hoping to see whether certain parts of the region or near boundary are causing the 
// issues in ILU(0) factor with Q-order

// demo the DKT kirchoff geometric multigrid solves..
#include "include/plate_assembler.h"
#include "include/_utils.h"
#include <chrono>

/* command line args:
    [direct/qorder] [--nxe int]
    * nxe must be power of 2

    examples:
    ./1_ilu.out direct --nxe 512  runs direct full LU solve on 512^2 elem grid
    ./1_ilu.out qorder --nxe 1024  runs ILU(0) qorder GMRES solve on 1024^2 elem grid
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

int main(int argc, char **argv) {
    using T = double;

    int nxe = 32;

    // 2048 is largest we can solve..

    // read argparse
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
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--nxe value]" << std::endl;
            return 1;
        }
    }

    T E = 2e7, thick = 0.01, nu = 0.3;
    T load_mag = 1000.0 / nxe / nxe; // so norm among diff nxe sizes..

    auto start0 = std::chrono::high_resolution_clock::now();
    auto assembler = DKTPlateAssembler(nxe, E, thick, nu, load_mag);

    // orderings and fillin for Q-order
    // TBD : be able to choose RCM, no order, others as well..
    int rcm_iters = 0; // no rcm iters because low bandwidth already
    double qorder_p = 0.5;
    if (nxe < 64) qorder_p = 2.0;
    assembler.h_bsr_data.qorder_reordering(qorder_p, rcm_iters);  
    assembler.h_bsr_data.compute_nofill_pattern();

    // then call the full initialize which also does assembly
    assembler.assemble();

    // now call the routine that checks near-identity map step by step with ILU(0) 
    // right-precond GMRES
    int n_iters = 10;
    assembler.apply_near_identity_gmres_map(n_iters, "out/");

    return 0;
}
