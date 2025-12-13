#include "assembler.h"
#include "shell/shell.h"

int main(void) {
    using T = double;

    // using assembler = ElementAssembler;

    // test the quadrature
    using Quadrature = QuadLinearQuadrature<T>;
    int ind = 0;
    T pt[2];
    T weight = Quadrature::getQuadraturePoint(ind, pt);
    // printf("%d : pt[0] = %.8f, pt[1] = %.8f, weight = %.8f\n", ind, pt[0], pt[1], weight);

    // test the basis
    using Basis = ShellQuadBasis<T, Quadrature, 2>;
    int num_nodes = Basis::num_nodes;
    T xpts[3*num_nodes], X[3];
    // debug temporarily change to pt = {-1,-1} first node
    pt[0] = -1;
    pt[1] = -1;
    for (int i = 0; i < 3*num_nodes; i++) {
      xpts[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    Basis::interpFields<3,3>(pt, xpts, X);
    for (int d = 0; d < 3; d++) {
      printf("Xpts[%d] = %.8e\n", d, xpts[d]);
      printf("X[%d] = %.8e\n", d, X[d]);
    }

    // make ref axis object
    constexpr bool has_ref_axis = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    Data my_data;
    T E = 70e9, nu = 0.3, thick = 0.005;
    T ref_axis[] = {1.0, 0.0, 0.0};
    my_data = Data(E, nu, thick, ref_axis);
    printf("Data : E=%.8e nu=%.8e thick=%.8e\n", my_data.E, my_data.nu, my_data.thick);
    printf("Data ref_axis : %.8e %.8e %.8e\n", my_data.refAxis[0], my_data.refAxis[1], my_data.refAxis[2]);

    return 0;
}