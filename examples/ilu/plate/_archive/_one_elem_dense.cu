#include "_plate_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
    using T = double;

    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data>;

    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

    int nxe = 1;
    int nye = nxe;
    // reduced Lx, Ly dimensions so that Kelem should be the same
    double Lx = 2.0/2.0, Ly = 1.0/2.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // init variables u
    auto vars = assembler.createVarsVec();
    assembler.set_variables(vars);

    // setup matrix & vecs
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    DenseMat<VecType<T>> kmat(assembler.get_num_vars());  

    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    auto h_kmat = kmat.createHostVec();
    
    // write the solution to binary file so I can read it in in python
    write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(), "csv/kelem-dense.csv");
    return 0;
};