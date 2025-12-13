#include "assembler.h"
#include "chrono"
#include "linalg/bsr_mat.h"
#include "linalg/vec.h"
#include "shell/shell.h"

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
    using Assembler = ElementAssembler<T, ElemGroup, HostVec, BsrMat>;

    printf("running!\n");

    int num_geo_nodes = 1e2;
    int num_vars_nodes = 1e2;
    int num_elements = 1e3;

    // make fake element connectivity for testing
    int N = Geo::num_nodes * num_elements;
    HostVec<int32_t> geo_conn(N);
    geo_conn.randomize(num_geo_nodes);
    make_unique_conn(num_elements, Geo::num_nodes, num_geo_nodes,
                     geo_conn.getPtr());

    // randomly generate the connectivity for the variables / basis
    int N2 = Basis::num_nodes * num_elements;
    HostVec<int32_t> vars_conn(N2);
    vars_conn.randomize(num_vars_nodes);
    make_unique_conn(num_elements, Basis::num_nodes, num_vars_nodes,
                     vars_conn.getPtr());

    // set the xpts randomly for this example
    int32_t num_xpts = Geo::spatial_dim * num_geo_nodes;
    HostVec<T> xpts(num_xpts);
    xpts.randomize();

    // initialize ElemData
    double E = 70e9, nu = 0.3, t = 0.005; // aluminum plate
    // initialize same data object for each element
    HostVec<Data> physData(num_elements, Data(E, nu, t));

    // make the assembler
    Assembler assembler(num_geo_nodes, num_vars_nodes, num_elements, geo_conn,
                        vars_conn, xpts, physData);

    // init variables u
    int32_t num_vars = assembler.get_num_vars();
    HostVec<T> vars(num_vars);
    vars.randomize();
    assembler.set_variables(vars);

    // init residual and kmat objects
    HostVec<T> res(num_vars);
    auto kmat = createBsrMat<Assembler, HostVec<T>>(assembler);

    // time add residual method
    auto start = std::chrono::high_resolution_clock::now();

    assembler.add_jacobian(res, kmat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // print some data of the host residual
    printf("res: ");
    printVec<double>(24, res.getPtr());
    printf("kmat: ");
    printVec<double>(24, kmat.getPtr());

    printf("took %d microseconds to run add residual\n", (int)duration.count());

    return 0;
};
