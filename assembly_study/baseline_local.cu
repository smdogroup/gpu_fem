#include "linalg/_linalg.h"
#include <chrono>

// shell imports
#include "include/v1/v1.h"
#include "include/time_assembler.h"

template <class Assembler>
Assembler createPlateAssembler(int nxe, int nye, double Lx, double Ly, double E, double nu,
                               double thick) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;

    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    - In the very thin-walled regime (low thick) becomes
    CPT or Kirchoff plate theory with no transverse shear effects
    - PDE for Kirchoff plate theory, linear static analysis
        D * nabla^4 w = q(x,y)
        w = 0, simply supported
    - if transverse loads q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
      [one half-wave each direction], then solution is:
        w(x,y) = A * sin(pi * x / a) * sin(pi * y / b)
        with A = Q / D / pi^4 / (1/a^4 + 1 / b^4 + 2 / a^2 b^2)

    - simply supported BCs are:
        on negative x2 edge: dof 23
        on negative x1 edge: dof 13
        on (0,0) corner : dof 123456
        on pos x2 edge: dof 3
        on pos x1 edge: dof 3
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;
    int num_elements = nxe * nye;

    // printf("checkpoint 1\n");

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // (0,0) corner with dof 123456
    for (int idof = 0; idof < 6; idof++) {
        my_bcs.push_back(idof);
    }
    // negative x2 edge with dof 23
    for (int ix = 1; ix < nnx; ix++) {
        int iy = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode + 1);  // dof 2 for v
        my_bcs.push_back(6 * inode + 2);  // dof 3 for w
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);
        my_bcs.push_back(6 * inode + 2);

        // pos x1 edge
        ix = nnx - 1;
        inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode + 2);  // corresp dof 3 for w
    }
    // pos x2 edge
    for (int ix = 1; ix < nnx - 1; ix++) {
        int iy = nny - 1;
        int inode = nnx * iy + ix;
        // printf("new bc = %d\n", 6 * inode + 2);
        my_bcs.push_back(6 * inode + 2);  // corresp dof 3 for w
    }

    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // printf("checkpoint 2 - post bcs\n");

    // printf("bcs: ");
    // printVec<int>(my_bcs.size(), bcs.getPtr());

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            // TODO : issue with defining conn out of order like this, needs to
            // be sorted now?""
            int nodes[] = {nnx * iye + ixe, nnx * iye + ixe + 1, nnx * (iye + 1) + ixe,
                           nnx * (iye + 1) + ixe + 1};
            for (int inode = 0; inode < Basis::num_nodes; inode++) {
                elem_conn[Basis::num_nodes * ielem + inode] = nodes[inode];
            }
        }
    }

    // printf("checkpoint 3 - post elem_conn\n");

    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the x-coordinates of the panel
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = Lx / nxe;
    T dy = Ly / nye;
    for (int iy = 0; iy < nny; iy++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * iy + ix;
            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            xpt_node[0] = dx * ix;
            xpt_node[1] = dy * iy;
            xpt_node[2] = 0.0;
        }
    }

    // printf("checkpoint 4 - post xpts\n");

    HostVec<Data> physData(num_elements, Data(E, nu, thick));

    // printf("checkpoint 5 - create physData\n");

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData);

    // printf("checkpoint 6 - create assembler\n");

    return assembler;
}

void time_linear_static(int nxe, std::string ordering, std::string fill_type, bool LU_solve = true, int ILU_k = 5,
    double p_factor = 1.0, bool print = true, bool write_vtk = false, bool debug = false) {
    // run the plate problem to time the linear static
    using T = double;   
    int rcm_iters = 5;

    auto start = std::chrono::high_resolution_clock::now();


    constexpr bool is_nonlinear = false; // true
    using Quad = QuadLinearQuadratureV1<T>;
    using Director = LinearizedRotationV1<T>;
    using Basis = ShellQuadBasisV1<T, Quad, 2>;
    using Data = ShellIsotropicDataV1<T, false>;
    using Physics = IsotropicShellV1<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroupV1<T, Director, Basis, Physics>;
    using Assembler = ElementAssemblerV1<T, ElemGroup, VecType, BsrMat>;

    int nye = nxe;
    double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005;
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0

    if (debug) printf("before reordering\n");

    if (ordering == "RCM") {
        bsr_data.RCM_reordering(rcm_iters);
    } else if (ordering == "AMD") {
        bsr_data.AMD_reordering();
    } else if (ordering == "qorder") {
        bsr_data.qorder_reordering(p_factor, rcm_iters, debug);
    } else if (ordering != "none") {
        std::cerr << "Unknown ordering: " << ordering << "\n";
        return;
    }

    if (debug) printf("before fillin\n");
    if (fill_type == "nofill") {
        bsr_data.compute_nofill_pattern();
    } else if (fill_type == "ILUk") {
        bsr_data.compute_ILUk_pattern(ILU_k, fillin, debug);
    } else if (fill_type == "LU") {
        bsr_data.compute_full_LU_pattern(fillin);
    } else {
        std::cerr << "Unknown fill type: " << fill_type << "\n";
        return;
    }

    assembler.moveBsrDataToDevice();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    if (debug) printf("before assembly\n");
    assembler.apply_bcs(res); // warmup call
    assembler.add_residual(res, print);
    assembler.add_jacobian(res, kmat, print);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);
}

int main() {
    printf("nxe, nnodes, ordering, fillin, solve, tot_time (s)\n");

    time_linear_static(100, "AMD", "ILUk", true, 0);
};