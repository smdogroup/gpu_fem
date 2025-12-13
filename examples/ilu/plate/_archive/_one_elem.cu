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
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    int nxe = 1;
    int nye = nxe;

    int other_nxe = 3;

    // reduced Lx, Ly dimensions so that Kelem should be the same
    double Lx = 2.0/other_nxe, Ly = 1.0/other_nxe, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);
    assembler.symbolic_factorization(1.0, true);

    // init variables u
    auto vars = assembler.createVarsVec();
    assembler.set_variables(vars);

    // setup matrix & vecs
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

    auto start = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    auto h_kmat = kmat.createHostVec();
    auto bsrData = kmat.getBsrData();

    // copy kmat values back to host
    DeviceVec<int32_t> d_rowPtr(bsrData.nnodes, bsrData.rowPtr);
    auto h_rowPtr = d_rowPtr.createDeviceVec();
    DeviceVec<int32_t> d_colPtr(bsrData.nnzb, bsrData.colPtr);
    auto h_colPtr = d_colPtr.createDeviceVec();
    
    // write the solution to binary file so I can read it in in python
    write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(), "csv/kelem.csv");
    return 0;
};