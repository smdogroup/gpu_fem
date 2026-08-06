#pragma once
#include "a2dcore.h"
#include "quadrature.h"

// technically nx = 2 is linear.. change this later
template <typename T, class Quadrature_, int _order = 1>
class LagrangeQuadBasis {
   public:
    using Quadrature = Quadrature_;
    static constexpr int32_t order = _order;
    static constexpr int32_t nx = order + 1;
    static constexpr bool ISOGEOM = false;

    // Required for loading solution data
    static constexpr int32_t num_nodes = nx * nx;
    static constexpr int32_t param_dim = 2;

    // MITC method => number of tying points for each strain component
    static constexpr int32_t mitcLevel2 = nx * (nx - 1);
    static constexpr int32_t mitcLevel3 = (nx - 1) * (nx - 1);
    static constexpr int32_t num_tying_components = 5;  // five gij components for tying strains
    static constexpr int32_t num_all_tying_points = 4 * mitcLevel2 + mitcLevel3;

    __HOST_DEVICE__ static constexpr int32_t num_tying_points(int icomp) {
        // g11, g22, g12, g23, g13, g33=0 (not included g33)
        int32_t _num_tying_points[] = {mitcLevel2, mitcLevel2, mitcLevel3, mitcLevel2, mitcLevel2};
        return _num_tying_points[icomp];
    }

    __HOST_DEVICE__ static constexpr int32_t tying_point_offsets(int icomp) {
        // g11, g22, g12, g23, g13, g33=0 (not included g33)
        int32_t _tying_point_offsets[] = {0, mitcLevel2, 2 * mitcLevel2,
                                          2 * mitcLevel2 + mitcLevel3, 3 * mitcLevel2 + mitcLevel3};
        return _tying_point_offsets[icomp];
    }

    // isoperimetric has same #nodes geometry class inside it
    class LinearQuadGeo {
       public:
        // Required for loading nodal coordinates
        static constexpr int32_t spatial_dim = 3;

        // Required for knowning number of spatial coordinates per node
        static constexpr int32_t num_nodes = nx * nx;

        // Number of quadrature points
        static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

        // Data-size = spatial_dim * number of nodes
        // static constexpr int32_t geo_data_size = 5 * num_quad_pts;
    };  // end of class LinearQuadGeo
    using Geo = LinearQuadGeo;

    __HOST_DEVICE__ static T getGaussPoint(int i) {
        // evenly spaced gauss-points
        T pt[2] = {0};
        Quadrature::getQuadraturePoint(i, pt);
        return pt[0];
    }

    // generic evalBasis call for interps in multigrid and FEA
    __HOST_DEVICE__ static void getBasis(const T pt[2], T N[num_nodes]) {
        lagrangeLobatto2D(pt[0], pt[1], N);
    }

    // shape functions here
    __HOST_DEVICE__ static void lagrangeLobatto1D(const T u, T *N) {
        if constexpr (nx == 2) {
            // Linear
            N[0] = 0.5 * (1.0 - u);
            N[1] = 0.5 * (1.0 + u);
        }

        else if constexpr (nx == 3) {
            // Quadratic Lobatto: nodes = [-1, 0, 1]
            N[0] = 0.5 * u * (u - 1.0);
            N[1] = 1.0 - u * u;
            N[2] = 0.5 * u * (u + 1.0);
        }

        else if constexpr (nx == 4) {
            // Cubic Lobatto: nodes = [-1, -1/3, 1/3, 1]
            const T a = u;
            N[0] = -(9.0 / 16.0) * (a + 1.0) * (a + 1.0 / 3.0) * (a - 1.0 / 3.0);
            N[1] = (27.0 / 16.0) * (a + 1.0) * (a + 1.0 / 3.0) * (a - 1.0);
            N[2] = -(27.0 / 16.0) * (a + 1.0) * (a - 1.0 / 3.0) * (a - 1.0);
            N[3] = (9.0 / 16.0) * (a - 1.0 / 3.0) * (a + 1.0 / 3.0) * (a - 1.0);
        }
    }

    __HOST_DEVICE__ static void lagrangeLobatto1DGrad(const T u, T *N, T *Nd) {
        if constexpr (nx == 2) {
            N[0] = 0.5 * (1.0 - u);
            N[1] = 0.5 * (1.0 + u);

            Nd[0] = -0.5;
            Nd[1] = 0.5;
        }

        else if constexpr (nx == 3) {
            // Quadratic
            N[0] = 0.5 * u * (u - 1.0);
            N[1] = 1.0 - u * u;
            N[2] = 0.5 * u * (u + 1.0);

            Nd[0] = u - 0.5;
            Nd[1] = -2.0 * u;
            Nd[2] = u + 0.5;
        }

        else if constexpr (nx == 4) {
            // Cubic
            const T a = u;

            N[0] = -(9.0 / 16.0) * (a + 1.0) * (a + 1.0 / 3.0) * (a - 1.0 / 3.0);
            N[1] = (27.0 / 16.0) * (a + 1.0) * (a + 1.0 / 3.0) * (a - 1.0);
            N[2] = -(27.0 / 16.0) * (a + 1.0) * (a - 1.0 / 3.0) * (a - 1.0);
            N[3] = (9.0 / 16.0) * (a - 1.0 / 3.0) * (a + 1.0 / 3.0) * (a - 1.0);

            Nd[0] = -(9.0 / 16.0) * ((a + 1.0) * (a + 1.0 / 3.0) + (a + 1.0) * (a - 1.0 / 3.0) +
                                     (a + 1.0 / 3.0) * (a - 1.0 / 3.0));
            Nd[1] = (27.0 / 16.0) * ((a + 1.0) * (a + 1.0 / 3.0) + (a + 1.0) * (a - 1.0) +
                                     (a + 1.0 / 3.0) * (a - 1.0));
            Nd[2] = -(27.0 / 16.0) * ((a + 1.0) * (a - 1.0 / 3.0) + (a + 1.0) * (a - 1.0) +
                                      (a - 1.0 / 3.0) * (a - 1.0));
            Nd[3] = (9.0 / 16.0) * ((a - 1.0 / 3.0) * (a + 1.0 / 3.0) +
                                    (a - 1.0 / 3.0) * (a - 1.0) + (a + 1.0 / 3.0) * (a - 1.0));
        }
    }

    __HOST_DEVICE__ static void lagrangeLobatto2D(const T xi, const T eta, T *N) {
        // compute N_i(u) shape function values at each node
        T na[nx], nb[nx];
        lagrangeLobatto1D(xi, na);
        lagrangeLobatto1D(eta, nb);

        // now compute 2D N_a(xi) * N_b(eta) for each node
        for (int ieta = 0; ieta < nx; ieta++) {
            for (int ixi = 0; ixi < nx; ixi++) {
                N[nx * ieta + ixi] = na[ixi] * nb[ieta];
            }
        }
    }  // end of lagrangeLobatto2D

    __HOST_DEVICE__ static void lagrangeLobatto2DGrad(const T xi, const T eta, T *N, T *dNdxi,
                                                      T *dNdeta) {
        // compute N_i(u) shape function values at each node
        T na[nx], nb[nx];
        T dna[nx], dnb[nx];
        lagrangeLobatto1DGrad(xi, na, dna);
        lagrangeLobatto1DGrad(eta, nb, dnb);

        // now compute 2D N_a(xi) * N_b(eta) for each node
        for (int ieta = 0; ieta < nx; ieta++) {
            for (int ixi = 0; ixi < nx; ixi++) {
                N[nx * ieta + ixi] = na[ixi] * nb[ieta];
                dNdxi[nx * ieta + ixi] = dna[ixi] * nb[ieta];
                dNdeta[nx * ieta + ixi] = na[ixi] * dnb[ieta];
            }
        }
    }  // end of lagrangeLobatto2D_grad

    __HOST_DEVICE__ static void assembleFrame(const T a[], const T b[], const T c[], T frame[]) {
        for (int i = 0; i < 3; i++) {
            frame[3 * i] = a[i];
            frame[3 * i + 1] = b[i];
            frame[3 * i + 2] = c[i];
        }
    }

    __HOST_DEVICE__ static void extractFrame(const T frame[], T a[], T b[], T c[]) {
        for (int i = 0; i < 3; i++) {
            a[i] = frame[3 * i];
            b[i] = frame[3 * i + 1];
            c[i] = frame[3 * i + 2];
        }
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        // get xi, eta coordinates of each node point
        pt[0] = -1.0 + (2.0 / (nx - 1)) * (n % nx);
        pt[1] = -1.0 + (2.0 / (nx - 1)) * (n / nx);
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]) {
        T N[num_nodes];
        lagrangeLobatto2D(pt[0], pt[1], N);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            field[ifield] = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                field[ifield] += N[inode] * values[inode * vars_per_node + ifield];
            }
        }
    }  // end of interpFields method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[], T dxi[],
                                                 T deta[]) {
        T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
        lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            dxi[ifield] = 0.0;
            deta[ifield] = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dxi[ifield] += dNdxi[inode] * values[inode * vars_per_node + ifield];
                deta[ifield] += dNdeta[inode] * values[inode * vars_per_node + ifield];
            }
        }
    }  // end of interpFieldsGrad method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsMixedGrad(const T pt[], const T values[], T d2mixed[]) {
        // Compute 1D basis and gradients
        T na[nx], nb[nx];
        T dna[nx], dnb[nx];

        lagrangeLobatto1DGrad(pt[0], na, dna);
        lagrangeLobatto1DGrad(pt[1], nb, dnb);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            T val = 0.0;

            // Loop over 2D nodes: N(xi_i, eta_j) = na[i] * nb[j]
            for (int j = 0; j < nx; j++) {
                for (int i = 0; i < nx; i++) {
                    const int inode = nx * j + i;

                    // Mixed derivative:  dN/dxi * dN/deta = dna[i] * dnb[j]
                    const T Nmixed = dna[i] * dnb[j];

                    val += Nmixed * values[inode * vars_per_node + ifield];
                }
            }

            d2mixed[ifield] = val;
        }
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsMixedGradTranspose(const T pt[], const T d2mixed_b[],
                                                               T values_b[]) {
        // Compute 1D basis and gradients
        T na[nx], nb[nx];
        T dna[nx], dnb[nx];
        lagrangeLobatto1DGrad(pt[0], na, dna);
        lagrangeLobatto1DGrad(pt[1], nb, dnb);

        // Loop over 2D nodes to accumulate adjoints
        for (int j = 0; j < nx; j++) {
            for (int i = 0; i < nx; i++) {
                const int inode = nx * j + i;
                T coeff = dna[i] * dnb[j];  // mixed derivative

                for (int ifield = 0; ifield < num_fields; ifield++) {
                    values_b[inode * vars_per_node + ifield] += coeff * d2mixed_b[ifield];
                }
            }
        }
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsTranspose(const T pt[], const T field_bar[],
                                                      T values_bar[]) {
        T N[num_nodes];  // TODO : double check this method (can we store less
                         // floats here also?)
        lagrangeLobatto2D(pt[0], pt[1], N);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] += field_bar[ifield] * N[inode];
            }
        }
    }  // end of interpFieldsTranspose method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[], const T dxi_bar[],
                                                          const T deta_bar[], T values_bar[]) {
        T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
        lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] +=
                    dxi_bar[ifield] * dNdxi[inode] + deta_bar[ifield] * dNdeta[inode];
            }
        }
    }  // end of interpFieldsGrad method

    __HOST_DEVICE__ static void getTyingKnots(T red_knots[], T full_knots[]) {
        // reduced integration points (not the same as lagrange-lobatto)
        if constexpr (nx == 2) {
            T a = 1.0;  // standard MITC
            // T a = 1.0 / sqrt(3.0);  // debugging for lock-aware prolong

            full_knots[0] = -a;
            full_knots[1] = a;

            red_knots[0] = 0.0;
        }

        if constexpr (nx == 3) {
            full_knots[0] = -0.774596669241483;
            full_knots[1] = 0.0;
            full_knots[2] = 0.774596669241483;

            red_knots[0] = -0.577350269189626;
            red_knots[1] = 0.577350269189626;
        }

        if constexpr (nx == 4) {
            full_knots[0] = -0.861136311594053;
            full_knots[1] = -0.339981043584856;
            full_knots[2] = 0.339981043584856;
            full_knots[3] = 0.861136311594053;

            red_knots[0] = -0.774596669241483;
            red_knots[1] = 0.0;
            red_knots[2] = 0.774596669241483;
        }
    }

    // section for tying point interpolations
    template <int icomp>
    __HOST_DEVICE__ static void getTyingPoint(const int n, T pt[]) {
        T red_knots[nx * (nx - 1)], full_knots[nx * nx];
        getTyingKnots(red_knots, full_knots);

        // use constexpr here to prevent warp divergence
        if constexpr (icomp == 0 || icomp == 4) {  // g11 or g13
            // 1-dir uses reduced knots for 1j strains
            // also nx*(nx-1) matrix but nx-1 columns
            pt[0] = red_knots[n % (nx - 1)];
            pt[1] = full_knots[n / (nx - 1)];
        } else if constexpr (icomp == 1 || icomp == 3) {  // g22 or g23
            // 2-dir uses reduced knots for 2j strains
            // also nx*(nx-1) matrix but nx columns
            pt[0] = full_knots[n % nx];
            pt[1] = red_knots[n / nx];
        } else if constexpr (icomp == 2) {  // g12
            // 1-dir and 2-dir use reduced knots here
            pt[0] = red_knots[n % (nx - 1)];
            pt[1] = red_knots[n / (nx - 1)];
        }
    }

    // replaces TacsLagrangeShapeFunction from TACS CPU
    template <int tying_nx>
    __HOST_DEVICE__ static void lagrange1D_tying(const T u, const T knots[], T *N) {
        // does lagrange interpolation from tying points to quadrature
        // so needs general lagrange polynomial rule (not interp from nodes but from tying points)
        if constexpr (tying_nx == 1) {
            N[0] = 1.0;
        } else if constexpr (tying_nx == 2) {
            N[0] = (u - knots[1]) / (knots[0] - knots[1]);
            N[1] = (u - knots[0]) / (knots[1] - knots[0]);
        } else if constexpr (tying_nx == 3) {
            N[0] =
                (u - knots[1]) * (u - knots[2]) / ((knots[0] - knots[1]) * (knots[0] - knots[2]));
            N[1] =
                (u - knots[0]) * (u - knots[2]) / ((knots[1] - knots[0]) * (knots[1] - knots[2]));
            N[2] =
                (u - knots[0]) * (u - knots[1]) / ((knots[2] - knots[0]) * (knots[2] - knots[1]));
        } else if constexpr (tying_nx == 4) {
            N[0] = (u - knots[1]) * (u - knots[2]) * (u - knots[3]) /
                   ((knots[0] - knots[1]) * (knots[0] - knots[2]) * (knots[0] - knots[3]));
            N[1] = (u - knots[0]) * (u - knots[2]) * (u - knots[3]) /
                   ((knots[1] - knots[0]) * (knots[1] - knots[2]) * (knots[1] - knots[3]));
            N[2] = (u - knots[0]) * (u - knots[1]) * (u - knots[3]) /
                   ((knots[2] - knots[0]) * (knots[2] - knots[1]) * (knots[2] - knots[3]));
            N[3] = (u - knots[0]) * (u - knots[1]) * (u - knots[2]) /
                   ((knots[3] - knots[0]) * (knots[3] - knots[1]) * (knots[3] - knots[2]));
        }
    }

    template <int icomp>
    __HOST_DEVICE__ static void getTyingInterp(const T pt[], T N[]) {
        // get 1d knot vectors
        T red_knots[(nx - 1)], full_knots[nx];
        getTyingKnots(red_knots, full_knots);

        // compute reduced space interp for MITC9 (see paper
        // https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.4399)

        T na[nx], nb[nx];
        lagrange1D_tying<nx>(pt[0], full_knots, na);
        lagrange1D_tying<nx>(pt[1], full_knots, nb);

        T nar[nx], nbr[nx];
        lagrange1D_tying<nx - 1>(pt[0], red_knots, nar);
        lagrange1D_tying<nx - 1>(pt[1], red_knots, nbr);

        if constexpr (icomp == 0 || icomp == 4) {
            // g11 or g13
            for (int j = 0; j < nx; j++) {
                for (int i = 0; i < nx - 1; i++, N++) {
                    N[0] = nar[i] * nb[j];
                }
            }
        } else if constexpr (icomp == 1 || icomp == 3) {
            // g22 or g23
            for (int j = 0; j < nx - 1; j++) {
                for (int i = 0; i < nx; i++, N++) {
                    N[0] = na[i] * nbr[j];
                }
            }
        } else if constexpr (icomp == 2) {
            // g12
            for (int j = 0; j < nx - 1; j++) {
                for (int i = 0; i < nx - 1; i++, N++) {
                    N[0] = nar[i] * nbr[j];
                }
            }
        }

    }  // end of getTyingInterp

};  // end of class ShellQuadBasis
