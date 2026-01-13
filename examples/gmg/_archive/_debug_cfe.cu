// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/mitc_shell.h"
#include "element/shell/fint_shell.h"

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void debug_chebyshev_basis() {
    // debug parts of chebyshev basis..
    // order of the chebyshev element
    const int order = 1; 
    // const int order = 2; // need at least order 2 to improve multigrid..
    // const int order = 3;

    using T = double;   
    // using Quad = QuadLinearQuadrature<T>;
    using Quad = QuadQuadraticQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ChebyshevQuadBasis<T, Quad, order>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = FullyIntegratedShellElementGroup<T, Director, Basis, Physics>;

    int n = order + 1;

    // check chebyshev GP gauss points
    for (int i = 0; i < n; i++) {
        T xi_i = Basis::get_chebyshev_gauss_point(i);
        printf("1D chebyshev GPs[i=%d] = %.2e\n", i, xi_i);
    }

    // now check basis function properties..
    for (int i = 0; i < n; i++) {
        T xi_i = Basis::get_chebyshev_gauss_point(i);
        for (int j = 0; j < n; j++) {
            T Nij = Basis::eval_chebyshev_1d(xi_i, j);
            printf("N%d at xi[%d] = %.2e\n", j, i, Nij);
        }
    }

    // test derivs of the chebyshev basis
    T xi_0 = 0.23242111;
    T h = 1e-4;
    for (int i = 0; i < n; i++) {
        T f1 = Basis::eval_chebyshev_1d(xi_0, i);
        T f2 = Basis::eval_chebyshev_1d(xi_0 + h, i);
        T df_FD = (f2 - f1) / h;

        T df_AN = Basis::eval_chebyshev_1d_grad(xi_0, i);
        printf("Cheb N[%d] fd test: FD %.8e, AN %.8e\n", i, df_FD, df_AN);
    }

    // try assembling an element
    const int n_xpts = 3 * Basis::num_nodes;
    const int n_vars = 6 * Basis::num_nodes;
    T xpts[n_xpts], vars[n_vars], res[n_vars];
    memset(xpts, 0.0, n_xpts * sizeof(T));
    memset(vars, 0.0, n_vars * sizeof(T));
    memset(res, 0.0, n_vars * sizeof(T));

    for (int i = 0; i < Basis::num_nodes; i++) {
        int ix = i % n, iy = i / n;
        xpts[3 * i] = 0.5 * ix;
        xpts[3 * i + 1] = 0.5 * iy;
    }

    T E = 2e7, nu = 0.3, thick = 0.1;
    auto data = Data(E, nu, thick);
    
    const int n_mat = n_vars * n_vars;
    T kelem[n_mat];
    memset(kelem, 0.0, n_mat * sizeof(T));
    T matCol[n_vars];
    bool active_thread = true;

    for (int iquad = 0; iquad < Quad::num_quad_pts; iquad++) {
        for (int icol = 0; icol < n_vars; icol++) {
            memset(matCol, 0.0, n_vars * sizeof(T));

            // printf("iquad %d, icol %d: \n", iquad, icol);

            // call element quadpt jacobian col
            ElemGroup::add_element_quadpt_jacobian_col<Data>(active_thread, iquad, icol, xpts, vars, data, res, matCol);

            // temp debug
            // return;

            // now add into kelem
            for (int irow = 0; irow < n_vars; irow++) {
                kelem[n_vars * irow + icol] += matCol[irow];
            }
        }
    }

    // print the element stiffness matrix
    printf("\n\nChebyshev fully integrated Kelem:\n");
    for (int irow = 0; irow < n_vars; irow++) {
        printf("Kelem[%d,:] : ", irow);
        printVec<T>(n_vars, &kelem[n_vars * irow]);
    }

}

void debug_lagrange_basis() {

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;

    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = MITCShellElementGroup<T, Director, Basis, Physics>;

    // try assembling an element
    const int n_xpts = 3 * Basis::num_nodes;
    const int n_vars = 6 * Basis::num_nodes;
    T xpts[n_xpts], vars[n_vars], res[n_vars];
    memset(xpts, 0.0, n_xpts * sizeof(T));
    memset(vars, 0.0, n_vars * sizeof(T));
    memset(res, 0.0, n_vars * sizeof(T));

    for (int i = 0; i < Basis::num_nodes; i++) {
        int ix = i % 2, iy = i / 2;
        xpts[3 * i] = 0.5 * ix;
        xpts[3 * i + 1] = 0.5 * iy;
    }

    T E = 2e7, nu = 0.3, thick = 0.1;
    auto data = Data(E, nu, thick);
    
    const int n_mat = n_vars * n_vars;
    T kelem[n_mat];
    memset(kelem, 0.0, n_mat * sizeof(T));
    T matCol[n_vars];
    bool active_thread = true;

    for (int iquad = 0; iquad < Quad::num_quad_pts; iquad++) {
        for (int icol = 0; icol < n_vars; icol++) {
            memset(matCol, 0.0, n_vars * sizeof(T));

            // printf("iquad %d, icol %d: \n", iquad, icol);

            // call element quadpt jacobian col
            ElemGroup::add_element_quadpt_jacobian_col<Data>(active_thread, iquad, icol, xpts, vars, data, res, matCol);

            // temp debug
            // return;

            // now add into kelem
            for (int irow = 0; irow < n_vars; irow++) {
                kelem[n_vars * irow + icol] += matCol[irow];
            }
        }
    }

    // print the element stiffness matrix
    printf("\n\nLagrange fully integrated Kelem:\n");
    for (int irow = 0; irow < n_vars; irow++) {
        printf("Kelem[%d,:] : ", irow);
        printVec<T>(n_vars, &kelem[n_vars * irow]);
    }

}

int main(int argc, char **argv) {

    debug_chebyshev_basis();
    debug_lagrange_basis();
    return 0;
}