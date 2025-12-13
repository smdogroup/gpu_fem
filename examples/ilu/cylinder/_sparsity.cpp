#include "_cylinder_utils.h"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include <cstdlib>  // for std::atoi, std::stod
#include <iostream>
#include <string>

#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

/**

    ## Example command line inputs:

    # No ordering, no fill
    ./program none nofill

    # RCM ordering with 5 RCM iters and ILU(k=2)
    ./program RCM ILUk 5 2

    # Q-ordering with p_factor=4 and ILU(k=3)
    ./program qorder ILUk 4 3

    # Q-ordering with full LU
    ./program qorder LU 4

 **/

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./program <ordering> <fill> [p_factor or k]\n";
        std::cerr << "Example: ./program qorder ILUk 3\n";
        return 1;
    }

    std::string ordering = argv[1];   // "none", "RCM", or "qorder"
    std::string fill_type = argv[2];  // "nofill", "ILUk", or "LU"

    int rcm_iters = 5;
    double p_factor = 0;
    int k = 0;

    if (ordering == "qorder") {
        if (argc < 4) {
            std::cerr << "Error: qorder requires a p_factor\n";
            return 1;
        }
        p_factor = std::stod(argv[3]);
    }
    if (ordering == "RCM") {
        if (argc < 4) {
            std::cerr << "Error: RCM requires a number of iterations\n";
            return 1;
        }
        rcm_iters = std::atoi(argv[3]);
    }

    if (fill_type == "ILUk") {
        if ((ordering != "qorder" && argc < 4) || (ordering == "qorder" && argc < 5)) {
            std::cerr << "Error: ILUk requires a value for k\n";
            return 1;
        }
        // ILUk's k value is argv[3] if ordering ≠ qorder, argv[4] if ordering = qorder
        bool multi_input_order = ordering == "RCM" || ordering == "qorder";
        k = std::atoi(argv[multi_input_order ? 4 : 3]);
    }

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

    // geometry of cylinder
    double L = 1.0;
    double Lr = 3.0;
    double rt = 100.0;  // r/t the cylinder slenderness
    double R = L / Lr;
    double thick = R / rt;

    // mesh settings
    int nxe_0 = 10, nhe_0 = 10;
    int refinement = 3;
    int nxe = nxe_0 * refinement, nhe = nhe_0 * refinement;

    // material properties (aluminum)
    double E = 70e9, nu = 0.3;

    // make the cylinder
    auto assembler = createCylinderAssembler<Assembler>(nxe, nhe, L, R, E, nu, thick);

    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;

    // Apply reordering
    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return 1;
    }

    // Apply fill pattern
    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(k);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return 1;
    }

    write_to_csv<int>(bsr_data.nnodes + 1, bsr_data.rowp, "out/rowp.csv");
    write_to_csv<int>(bsr_data.nnzb, bsr_data.cols, "out/cols.csv");

    return 0;
}
