#pragma once
#include <fstream>
#include <vector>

#include "assembler.h"
#include "linalg/vec.h"
#include "math.h"
#include "utils.h"

template <class Assembler>
Assembler createPlateAssembler(int nxe, int nye, double Lx, double Ly, double E, double nu,
                               double thick, double rho = 2500, double ys = 350e6,
                               int nxe_per_comp = 1, int nye_per_comp = 1,
                               bool theta_ss_bc = true) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;
    using Physics = typename Assembler::Phys;
    const int order = Basis::order;

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

    assert(nxe % nxe_per_comp == 0);
    assert(nye % nye_per_comp == 0);

    // number of nodes per direction
    int nnx = Basis::ISOGEOM ? nxe + order : order * nxe + 1;
    int nny = Basis::ISOGEOM ? nye + order : order * nye + 1;
    int num_nodes = nnx * nny;
    int num_elements = nxe * nye;

    // printf("num nodes %d, num_elements %d\n", num_nodes, num_elements);

    // printf("checkpoint 1\n");

    constexpr bool IS_HR_ELEM = Physics::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Physics::vars_per_node;

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // (0,0) corner with dof 123456
    for (int idof = 0; idof < vpn; idof++) {
        if (idof == 3 || idof == 4) {
            if (theta_ss_bc) my_bcs.push_back(idof);
        } else {
            my_bcs.push_back(idof);
        }
    }
    // negative x2 (or y) edge with dof 23
    for (int ix = 1; ix < nnx; ix++) {
        int iy = 0;
        int inode = nnx * iy + ix;
        if constexpr (IS_HR_ELEM) {
            // also constrain [v0, v1, v2, v3, v4] = strain-disp of [e11, e12, e22, gam13, gam23]
            // with v1(e12) constrained like u and v and e11,e22 like u,v and gam13, gam23 like thx,
            my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like v)
            my_bcs.push_back(vpn * inode + 2);  // v2 equiv to e22 strain-gap disp (zero like v)
            my_bcs.push_back(vpn * inode +
                             4);  // v3 equiv to gam13 strain-gap disp (zero like w, thy), checked
                                  // coupling in nodal matrix (this is right)
        }

        my_bcs.push_back(vpn * inode + offset + 1);                   // dof 2 for v
        my_bcs.push_back(vpn * inode + offset + 2);                   // dof 3 for w
        if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 4);  // dof 4 for thy
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(vpn * inode + offset);                       // u
        my_bcs.push_back(vpn * inode + offset + 2);                   // w
        if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 3);  // dof 3 for thx
        if constexpr (IS_HR_ELEM) {
            my_bcs.push_back(vpn * inode + 0);  // v0 equiv to e11 strain-gap disp (zero like u)
            my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like u)
            my_bcs.push_back(vpn * inode + 3);  // v4 equiv to gam23 strain-gap disp (zero like
            // w, thy). cje
        }

        // pos x1 edge
        ix = nnx - 1;
        inode = nnx * iy + ix;
        my_bcs.push_back(vpn * inode + offset + 2);                   // corresp dof 3 for w
        if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 3);  // corresp dof 3 for thx
        // no HR constraints needed on positive edges
        if constexpr (IS_HR_ELEM) {
            // in-plane BCs a bit weird still (needed v12 or v1 = 0 on positive x1,x2 edges but not
            // v0 and v2) my_bcs.push_back(vpn * inode + 0);
            // v0 equiv to e11 strain-gap disp (zero like u)
            // my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like u)
            my_bcs.push_back(vpn * inode + 3);  // v4 equiv to gam23 strain-gap disp (zero like
            // thx)
        }
    }
    // pos x2 edge (up to one before upper-right corner)
    for (int ix = 1; ix < nnx - 1; ix++) {
        int iy = nny - 1;
        int inode = nnx * iy + ix;
        // printf("new bc = %d\n", 6 * inode + 2);
        if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 4);  // dof 5 for thy
        my_bcs.push_back(vpn * inode + offset + 2);                   // corresp dof 3 for w
        // no HR constraints needed on positive edges
        if constexpr (IS_HR_ELEM) {
            // my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like v)
            // my_bcs.push_back(vpn * inode + 2);  // v2 equiv to e22 strain-gap disp (zero like v)
            my_bcs.push_back(vpn * inode + 4);  // v3 equiv to gam13 strain-gap disp (like thy)
        }
    }
    // (+x1,+x2) corner node, add thy DOF
    int inode = nnx * (nny - 1) + nnx - 1;
    if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 4);  // set thy DOF zero here too
    if constexpr (IS_HR_ELEM) {
        // my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like v)
        // my_bcs.push_back(vpn * inode + 2);  // v2 equiv to e22 strain-gap disp (zero like v)
        my_bcs.push_back(vpn * inode + 4);  // v3 equiv to gam13 strain-gap disp (like thy)
    }
    // (-x1,+x2) corner node, top left
    inode = nnx * (nny - 1);
    if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 4);  // set thy DOF zero here too
    if constexpr (IS_HR_ELEM) {
        my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like v)
        my_bcs.push_back(vpn * inode + 4);  // v4 equiv to gam23 strain-gap disp (like thy)
    }
    // (+x1,-x2) corner node, bottom right
    inode = nnx - 1;
    if (theta_ss_bc) my_bcs.push_back(vpn * inode + offset + 3);  // corresp dof 3 for thx
    if constexpr (IS_HR_ELEM) {
        my_bcs.push_back(vpn * inode + 1);  // v1 equiv to e12 strain-gap disp (zero like v)
        my_bcs.push_back(vpn * inode + 3);  // v3 equiv to gam13 strain-gap disp (like thx)
    }

    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // printf("checkpoint 2 - post bcs\n");

    // printf("bcs: ");
    // printVec<int>(bcs.getSize(), bcs.getPtr());
    int n = order + 1;  // num local nodes

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // no sorted order like in MITC?
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = Basis::ISOGEOM ? ixe + ilx : order * ixe + ilx;
                int iy = Basis::ISOGEOM ? iye + ily : order * iye + ily;
                int inode = nnx * iy + ix;

                elem_conn[Basis::num_nodes * ielem + iloc] = inode;
            }
        }
    }
    // return;

    // printf("elem_conn with nnodes_per_elem %d: ", Basis::num_nodes);
    // printVec<int>(N, elem_conn);
    // printf("checkpoint 3 - post elem_conn\n");

    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the x-coordinates of the panel
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);
    for (int iy = 0; iy < nny; iy++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * iy + ix;
            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            if constexpr (Basis::order == 1 || Basis::ISOGEOM) {
                xpt_node[0] = dx * ix;
                xpt_node[1] = dy * iy;
                xpt_node[2] = 0.0;
            } else {
                // higher-order nodes
                int ix_corner = (ix / order) * order;
                int iy_corner = (iy / order) * order;

                // local node index inside element
                int ix_local = ix % order;
                int iy_local = iy % order;

                // get reference Gauss point [-1,1]
                T xi = Basis::getGaussPoint(ix_local);
                T eta = Basis::getGaussPoint(iy_local);

                // physical element corners
                T x0 = dx * ix_corner;
                T x1 = dx * (ix_corner + (n - 1));  // last node in this element
                T y0 = dy * iy_corner;
                T y1 = dy * (iy_corner + (n - 1));

                // map reference [-1,1] → [x0,x1] and [y0,y1]
                xpt_node[0] = 0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1;
                xpt_node[1] = 0.5 * (1.0 - eta) * y0 + 0.5 * (1.0 + eta) * y1;
                xpt_node[2] = 0.0;

                // printf("xi %.4e, eta %.4e with (x %.4e, y %.4e, z %.4e)\n", xi, eta, xpt_node[0],
                // xpt_node[1], xpt_node[2]);
            }
        }
    }

    // printf("checkpoint 4 - post xpts\n");
    HostVec<Data> physData(num_elements, Data(E, nu, thick, rho, ys));

    // printf("checkpoint 5 - create physData\n");

    // make elem_components
    int num_xcomp = nxe / nxe_per_comp;
    int num_ycomp = nye / nye_per_comp;
    int num_components = num_xcomp * num_ycomp;

    HostVec<int> elem_components(num_elements);
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = num_xcomp * iy_comp + ix_comp;

            elem_components[ielem] = icomp;
        }
    }

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData, num_components, elem_components);

    // printf("checkpoint 6 - create assembler\n");
    // printf("num_components = %d\n", num_components);
    // printf("elem_components:");
    // printVec<int>(elem_components.getSize(), elem_components.getPtr());

    return assembler;
}

template <class Assembler, bool swap_xy = false>
Assembler createPlateClampedAssembler(int nxe, int nye, double Lx, double Ly, double E, double nu,
                                      double thick, double rho = 2500, double ys = 350e6,
                                      int nxe_per_comp = 1, int nye_per_comp = 1) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;
    using Physics = typename Assembler::Phys;
    const int order = Basis::order;

    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    - In the very thin-walled regime (low thick) becomes
    CPT or Kirchoff plate theory with no transverse shear effects
    - PDE for Kirchoff plate theory, linear static analysis
        D * nabla^4 w = q(x,y)
        w = 0, w_{,n} = 0 (clamped)
    - if transverse loads q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
      [one half-wave each direction], then solution is:
        w(x,y) = A * sin(pi * x / a) * sin(pi * y / b)
        with A = Q / D / pi^4 / (1/a^4 + 1 / b^4 + 2 / a^2 b^2)

    - clamped BCs are: 123456 on each edge
    */

    assert(nxe % nxe_per_comp == 0);
    assert(nye % nye_per_comp == 0);

    // number of nodes per direction
    int nnx = Basis::ISOGEOM ? nxe + order : order * nxe + 1;
    int nny = Basis::ISOGEOM ? nye + order : order * nye + 1;
    int num_nodes = nnx * nny;
    int num_elements = nxe * nye;

    // printf("num nodes %d, num_elements %d\n", num_nodes, num_elements);

    // printf("checkpoint 1\n");

    constexpr bool IS_HR_ELEM = Physics::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Physics::vars_per_node;

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // (0,0) corner with dof 123456
    for (int idof = 0; idof < vpn; idof++) {
        my_bcs.push_back(idof);
    }
    // negative x2 (or y) edge with dof 23
    for (int ix = 1; ix < nnx; ix++) {
        int iy = 0;
        int inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
        for (int idof = 0; idof < vpn; idof++) {
            my_bcs.push_back(vpn * inode + offset + idof);
        }
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
        for (int idof = 0; idof < vpn; idof++) {
            my_bcs.push_back(vpn * inode + offset + idof);
        }

        // pos x1 edge
        ix = nnx - 1;
        inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
        for (int idof = 0; idof < vpn; idof++) {
            my_bcs.push_back(vpn * inode + offset + idof);
        }
    }
    // pos x2 edge (up to one before upper-right corner)
    for (int ix = 1; ix < nnx - 1; ix++) {
        int iy = nny - 1;
        int inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
        for (int idof = 0; idof < vpn; idof++) {
            my_bcs.push_back(vpn * inode + offset + idof);
        }
    }
    // (+x1,+x2) corner node, add thy DOF
    int ix = nnx - 1, iy = nny - 1;
    int inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
    for (int idof = 0; idof < vpn; idof++) {
        my_bcs.push_back(vpn * inode + offset + idof);
    }
    // (-x1,+x2) corner node, top left
    ix = 0, iy = nny - 1;
    inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
    for (int idof = 0; idof < vpn; idof++) {
        my_bcs.push_back(vpn * inode + offset + idof);
    }
    // (+x1,-x2) corner node, bottom right
    ix = nnx - 1, iy = 0;
    inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;
    for (int idof = 0; idof < vpn; idof++) {
        my_bcs.push_back(vpn * inode + offset + idof);
    }

    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // printf("checkpoint 2 - post bcs\n");

    // printf("bcs: ");
    // printVec<int>(bcs.getSize(), bcs.getPtr());
    int n = order + 1;  // num local nodes

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // no sorted order like in MITC?
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = Basis::ISOGEOM ? ixe + ilx : order * ixe + ilx;
                int iy = Basis::ISOGEOM ? iye + ily : order * iye + ily;
                int inode = swap_xy ? (nny * ix + iy) : nnx * iy + ix;

                elem_conn[Basis::num_nodes * ielem + iloc] = inode;
            }
        }
    }
    // return;

    // printf("elem_conn with nnodes_per_elem %d: ", Basis::num_nodes);
    // printVec<int>(N, elem_conn);
    // printf("checkpoint 3 - post elem_conn\n");

    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the x-coordinates of the panel
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);
    if constexpr (swap_xy) {
        for (int ix = 0; ix < nnx; ix++) {
            for (int iy = 0; iy < nny; iy++) {
                int inode = nny * ix + iy;
                T *xpt_node = &xpts[Geo::spatial_dim * inode];
                if constexpr (Basis::order == 1 || Basis::ISOGEOM) {
                    xpt_node[0] = dx * ix;
                    xpt_node[1] = dy * iy;
                    xpt_node[2] = 0.0;
                } else {
                    // pass
                }
            }
        }
    } else {
        // not swap xy
        for (int iy = 0; iy < nny; iy++) {
            for (int ix = 0; ix < nnx; ix++) {
                int inode = nny * iy + ix;
                T *xpt_node = &xpts[Geo::spatial_dim * inode];
                if constexpr (Basis::order == 1 || Basis::ISOGEOM) {
                    xpt_node[0] = dx * ix;
                    xpt_node[1] = dy * iy;
                    xpt_node[2] = 0.0;
                } else {
                    // pass
                }
            }
        }
    }

    // printf("checkpoint 4 - post xpts\n");
    HostVec<Data> physData(num_elements, Data(E, nu, thick, rho, ys));

    // printf("checkpoint 5 - create physData\n");

    // make elem_components
    int num_xcomp = nxe / nxe_per_comp;
    int num_ycomp = nye / nye_per_comp;
    int num_components = num_xcomp * num_ycomp;

    HostVec<int> elem_components(num_elements);
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = num_xcomp * iy_comp + ix_comp;

            elem_components[ielem] = icomp;
        }
    }

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData, num_components, elem_components);

    return assembler;
}

template <typename T, class Phys>
T *getPlatePointLoad(int nxe, int nye, double Lx, double Ly, double load_mag) {
    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    make the load set for this mesh
    q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;

    int ix = nnx / 2;
    int iy = nny / 2;
    int inode = nnx * iy + ix;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    my_loads[Phys::vars_per_node * inode + 2] = load_mag;
    return my_loads;
}

// template <typename T, class Phys>
// T *getPlateNonlinearLoads(int nxe, int nye, double Lx, double Ly, double load_mag,
//                           double in_plane_frac = 0.0) {
//     /*
//     make a rectangular plate mesh of shell elements
//     simply supported with transverse constrant distributed load

//     make the load set for this mesh
//     q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
//     */

//     // number of nodes per direction
//     int nnx = nxe + 1;
//     int nny = nye + 1;
//     int num_nodes = nnx * nny;

//     T dx = Lx / nxe;
//     T dy = Ly / nye;

//     double PI = 3.1415926535897;

//     int num_dof = Phys::vars_per_node * num_nodes;
//     T *my_loads = new T[num_dof];
//     memset(my_loads, 0.0, num_dof * sizeof(T));

//     // technically we should be integrating this somehow or distributing this
//     // among the elements somehow..
//     // the actual rhs is integral q(x,y) * phi_i(x,y) dxdy, fix later if want
//     // better error conv.
//     for (int iy = 0; iy < nny; iy++) {
//         for (int ix = 0; ix < nnx; ix++) {
//             int inode = nnx * iy + ix;
//             T x = ix * dx, y = iy * dy;
//             T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
//             my_loads[Phys::vars_per_node * inode + 2] = nodal_load * 10 / nxe / nye;
//         }

//         int ix = nnx - 1;  // pos x1 edge for in-plane
//         int inode = nnx * iy + ix;
//         T x = ix * dx, y = iy * dy;
//         my_loads[Phys::vars_per_node * inode] = -load_mag * in_plane_frac / nye;
//     }
//     return my_loads;
// }

template <typename T, class Basis, class Phys>
T *getPlateLoads(int nxe, int nye, double Lx, double Ly, double load_mag, double axial_frac = 0.0) {
    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    make the load set for this mesh
    q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
    */

    // number of nodes per direction
    int order = Basis::order;
    int nnx = order * nxe + 1;
    int nny = order * nye + 1;
    int num_nodes = nnx * nny;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);

    double PI = 3.1415926535897;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // for (int iy = 0; iy < nny; iy++) {
    //     for (int ix = 0; ix < nnx; ix++) {
    //         int inode = nnx * iy + ix;
    //         T x = ix * dx, y = iy * dy;
    //         // T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
    //         T r = sqrt(x * x + y * y);
    //         T th = atan2(y, x);
    //         T nodal_load = load_mag * sin(5.0 * PI * r) * cos(4.0 * th);

    //         if (ix % order == 0 && iy % order == 0) {
    //             // no loads on mid-side nodes
    //             // don't put loads on mid-side nodes etc of higher order (weird results)
    //             my_loads[vpn * inode + offset + 2] = nodal_load;  // * dx * dy;
    //         }
    //     }
    // }

    using Quadrature = typename Basis::Quadrature;
    const int n = Basis::nx;

    // the actual rhs is integral q(x,y) * phi_i(x,y) dxdy, fix later if want
    // better error conv. (loop over each element to add loads in)
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // no sorted order like in MITC?
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = order * ixe + ilx;
                int iy = order * iye + ily;

                // higher-order nodes
                int ix_corner = (ix / order) * order;
                int iy_corner = (iy / order) * order;

                // local node index inside element
                int ix_local = ix % order;
                int iy_local = iy % order;

                // get reference Gauss point [-1,1]
                T xi = Basis::getGaussPoint(ix_local);
                T eta = Basis::getGaussPoint(iy_local);

                // physical element corners
                T x0 = dx * ix_corner;
                T x1 = dx * (ix_corner + (n - 1));  // last node in this element
                T y0 = dy * iy_corner;
                T y1 = dy * (iy_corner + (n - 1));

                // map reference [-1,1] → [x0,x1] and [y0,y1]
                T x = 0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1;
                T y = 0.5 * (1.0 - eta) * y0 + 0.5 * (1.0 + eta) * y1;
                T z = 0.0;

                int inode = nnx * iy + ix;

                // darea and quadrature points in the element
                T J = dx * order * dy * order / 4;
                T pt[2] = {0};
                // pt[0] = xi, pt[1] = eta;
                // this multiplies both weights internally
                T weight = Quadrature::getQuadraturePoint(iloc, pt);

                T r = sqrt(x * x + y * y);
                T th = atan2(y, x);
                T nodal_load =
                    load_mag * sin(5.0 * PI * r / Lx) * cos(4.0 * th);  // + 0.1 * load_mag;
                // somehow need to scale by local dGP area though no?
                nodal_load *= weight * J;

                my_loads[vpn * inode + offset + 2] += nodal_load;

                my_loads[vpn * inode + offset + 0] += axial_frac * nodal_load; // add axial component to load
            }
        }
    }

    return my_loads;
}

template <typename T, class Basis, class Phys>
T *getPlateNonlinearLoads(int nxe, int nye, double Lx, double Ly, double load_mag,
                          double in_plane_frac = 0.1) {
    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load

    make the load set for this mesh
    q(x,y) = Q * sin(pi * x / a) * sin(pi * y / b)
    */

    // number of nodes per direction
    int order = Basis::order;
    int nnx = order * nxe + 1;
    int nny = order * nye + 1;
    int num_nodes = nnx * nny;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);

    double PI = 3.1415926535897;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // for (int iy = 0; iy < nny; iy++) {
    //     for (int ix = 0; ix < nnx; ix++) {
    //         int inode = nnx * iy + ix;
    //         T x = ix * dx, y = iy * dy;
    //         // T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
    //         T r = sqrt(x * x + y * y);
    //         T th = atan2(y, x);
    //         T nodal_load = load_mag * sin(5.0 * PI * r) * cos(4.0 * th);

    //         if (ix % order == 0 && iy % order == 0) {
    //             // no loads on mid-side nodes
    //             // don't put loads on mid-side nodes etc of higher order (weird results)
    //             my_loads[vpn * inode + offset + 2] = nodal_load;  // * dx * dy;
    //         }
    //     }
    // }

    using Quadrature = typename Basis::Quadrature;
    const int n = Basis::nx;

    // the actual rhs is integral q(x,y) * phi_i(x,y) dxdy, fix later if want
    // better error conv. (loop over each element to add loads in)
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // no sorted order like in MITC?
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = order * ixe + ilx;
                int iy = order * iye + ily;

                // higher-order nodes
                int ix_corner = (ix / order) * order;
                int iy_corner = (iy / order) * order;

                // local node index inside element
                int ix_local = ix % order;
                int iy_local = iy % order;

                // get reference Gauss point [-1,1]
                T xi = Basis::getGaussPoint(ix_local);
                T eta = Basis::getGaussPoint(iy_local);

                // physical element corners
                T x0 = dx * ix_corner;
                T x1 = dx * (ix_corner + (n - 1));  // last node in this element
                T y0 = dy * iy_corner;
                T y1 = dy * (iy_corner + (n - 1));

                // map reference [-1,1] → [x0,x1] and [y0,y1]
                T x = 0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1;
                T y = 0.5 * (1.0 - eta) * y0 + 0.5 * (1.0 + eta) * y1;
                T z = 0.0;

                int inode = nnx * iy + ix;

                // darea and quadrature points in the element
                T J = dx * order * dy * order / 4;
                T pt[2] = {0};
                // pt[0] = xi, pt[1] = eta;
                // this multiplies both weights internally
                T weight = Quadrature::getQuadraturePoint(iloc, pt);

                T r = sqrt(x * x + y * y);
                T th = atan2(y, x);
                T nodal_load =
                    load_mag * sin(5.0 * PI * r / Lx) * cos(4.0 * th);  // + 0.1 * load_mag;
                // somehow need to scale by local dGP area though no?
                nodal_load *= weight * J;

                my_loads[vpn * inode + offset + 2] += nodal_load;

                // add an obliquue in-plane load.. like shear load with striations
                T in_plane_load =
                    -1.0 *
                    sin(PI * r / Lx /
                        1.414);  // should be compressive all along diagonal (shear, radial load)
                in_plane_load *= weight * J * load_mag * in_plane_frac;
                // radially aligned in-plane load (want to make sure it's compressive though..)
                my_loads[vpn * inode + offset + 0] += in_plane_load * cos(th);
                my_loads[vpn * inode + offset + 1] += in_plane_load * sin(th);
            }
        }
    }

    return my_loads;
}

template <typename T, class Phys>
T *getPlateSineLoads(int nxe, int nye, double Lx, double Ly, int m, int n, double load_mag) {
    /*
    make a rectangular plate mesh of shell elements
    simply supported with transverse constrant distributed load
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nny = nye + 1;
    int num_nodes = nnx * nny;

    T dx = Lx / nxe;
    T dy = Ly / nye;

    double PI = 3.1415926535897;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // technically we should be integrating this somehow or distributing this
    // among the elements somehow..
    // the actual rhs is integral q(x,y) * phi_i(x,y) dxdy, fix later if want
    // better error conv.
    for (int iy = 0; iy < nny; iy++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * iy + ix;
            T x = ix * dx, y = iy * dy;
            T nodal_load = load_mag * sin(m * PI * x / Lx) * sin(n * PI * y / Ly);

            my_loads[Phys::vars_per_node * inode + 2] = nodal_load;  // * dx * dy;
        }
    }
    return my_loads;
}

template <typename T, class Assembler>
T *getPlateMeshConvLoads(Assembler &assembler, int nxe, int nye, double Lx, double Ly,
                         double load_mag, bool uniform_load = false, int mx = 1, int my = 1) {
    /* more robust version of cylinder loads */

    using Phys = typename Assembler::Phys;
    using Basis = typename Assembler::Basis;

    const int order = Basis::order;
    const int n = order + 1;
    using Quadrature = typename Basis::Quadrature;

    // number of nodes per direction
    int nnx = Basis::ISOGEOM ? order + nxe : order * nxe + 1;
    int nny = Basis::ISOGEOM ? order + nye : order * nye + 1;
    int num_nodes = nnx * nny;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // get importnat data out of assembler and move to host
    T *h_xpts = assembler.getXpts().createHostVec().getPtr();
    int *h_conn = assembler.getConn().createHostVec().getPtr();

    // loop over each element to compute load shape-function integrals
    for (int iye = 0; iye < nye; iye++) {
        // bool last_hoop = ihe == nhe - 1;
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // build element nodes and xpts from assembler
            int *elem_nodes = &h_conn[Basis::num_nodes * ielem];
            // get nodal coords of the element
            T elem_xpts[3 * Basis::num_nodes] = {0};
            for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                int inode = elem_nodes[lnode];
                for (int dir = 0; dir < 3; dir++) {
                    elem_xpts[3 * lnode + dir] = h_xpts[3 * inode + dir];
                }
            }

            // compute shell normals
            T fn[3 * Basis::num_nodes] = {0.0};
            ShellComputeNodeNormals<T, Basis>(elem_xpts, fn);

            // now do gauss quadrature integral
            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                T pt[2];
                T weight = Quadrature::getQuadraturePoint(iquad, pt);

                // interp x,y,z to the quadrature point (using element basis functions)
                T xpt[3] = {0};
                Basis::template interpFields<3, 3>(pt, elem_xpts, xpt);

                // compute J = d(x,y,z)/dxi,deta
                T J = getDetXd<T, Basis>(pt, elem_xpts, fn);

                // compute load magnitudes at the quadrature point
                // T M_PI = 3.141592653589723846;
                T mag = 0.0;
                if (uniform_load) {
                    mag = load_mag;
                } else {
                    mag = load_mag * sin(M_PI * mx * xpt[0] / Lx) * sin(M_PI * my * xpt[1] / Ly);
                }

                // compute element basis functions at the quadpt (for nodal load distribution)
                T N[Basis::num_nodes] = {0.0};
                Basis::getBasis(pt, N);

                // now loop over each node to distribute load integral among nodes
                for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                    int inode = elem_nodes[lnode];
                    T nodal_mag = mag * N[lnode] * weight * J;

                    // (temp debug) add small in-plane loads too (for testing membrane response of
                    // HR)
                    // my_loads[vpn * inode + offset] += 0.01 * nodal_mag;
                    // my_loads[vpn * inode + offset + 1] += 0.01 * nodal_mag;

                    // add to each node now using element shape functions
                    my_loads[vpn * inode + offset + 2] += nodal_mag;
                }  // end of nodal distribution loop
            }      // end of quadpt loop

        }  // end of x element loop
    }      // end of hoop element loop

    return my_loads;
}

template <class Assembler>
Assembler createPlateDistortedAssembler(int nxe, int nye, double Lx, double Ly, double E, double nu,
                                        double thick, double rho = 2500, double ys = 350e6,
                                        int nxe_per_comp = 1, int nye_per_comp = 1, int m = 1,
                                        int n = 1, double x_frac = 0.0, double y_frac = 0.0,
                                        double shear_frac = 0.0) {
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

    assert(nxe % nxe_per_comp == 0);
    assert(nye % nye_per_comp == 0);

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
    // negative x2 (or y) edge with dof 23
    for (int ix = 1; ix < nnx; ix++) {
        int iy = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode + 1);  // dof 2 for v
        my_bcs.push_back(6 * inode + 2);  // dof 3 for w
        my_bcs.push_back(6 * inode + 4);  // dof 5 for thy
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);
        my_bcs.push_back(6 * inode + 2);

        my_bcs.push_back(6 * inode + 3);  // dof 4 for thx

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
        my_bcs.push_back(6 * inode + 4);  // dof 5 for thy
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

    /* structured mesh distortion here.. */
    T X_cf = 1.0 / 3.14159 / m * x_frac;
    T Y_cf = 1.0 / 3.14159 / n * y_frac;

    for (int inode = 0; inode < num_nodes; inode++) {
        int ix = inode % nnx, iy = inode / nnx;
        T x = ix * dx, y = iy * dx;

        // spacing distortion
        T disp_x = X_cf * sin(m * 3.14159 * x);
        T disp_y = Y_cf * sin(n * 3.14159 * y);

        // shearing distortion
        T shear = shear_frac * (X_cf + Y_cf) * sin(3.14159 * x) * sin(3.14159 * y);
        disp_x += shear * y;
        disp_y += shear * x;

        // now add to it..
        xpts[3 * inode] += disp_x;
        xpts[3 * inode + 1] += disp_y;
    }

    // printf("checkpoint 4 - post xpts\n");

    HostVec<Data> physData(num_elements, Data(E, nu, thick, rho, ys));

    // printf("checkpoint 5 - create physData\n");

    // make elem_components
    int num_xcomp = nxe / nxe_per_comp;
    int num_ycomp = nye / nye_per_comp;
    int num_components = num_xcomp * num_ycomp;

    HostVec<int> elem_components(num_elements);
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = num_xcomp * iy_comp + ix_comp;

            elem_components[ielem] = icomp;
        }
    }

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData, num_components, elem_components);

    // printf("checkpoint 6 - create assembler\n");
    // printf("num_components = %d\n", num_components);
    // printf("elem_components:");
    // printVec<int>(elem_components.getSize(), elem_components.getPtr());

    return assembler;
}

template <class Assembler>
Assembler createCylinderAssembler(int nxe, int nhe, double L, double R, double E, double nu,
                                  double thick, bool imperfection = false, int imp_x = 5,
                                  int imp_hoop = 4, double rho = 2500, double ys = 350e6,
                                  int nx_comp = 1, int ny_comp = 1) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;
    using Physics = typename Assembler::Phys;

    // number of nodes per direction
    const int order = Basis::order;
    int n = order + 1;
    int nnx = order * nxe + 1;  // axial nodes
    int nnh = order * nhe;      // number of hoop nodes
    int num_nodes = nnx * nnh;
    int num_elements = nxe * nhe;

    constexpr bool IS_HR_ELEM = Physics::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Physics::vars_per_node;

    // cylinder is in x direction along its axis, circle planes are in yz plane

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // TODO : will change to use disp control BCs later
    // node 0 has dof 123456, changed now to just 123
    // for (int idof = 0; idof < 3; idof++) {
    for (int idof = 0; idof < vpn; idof++) {  // clamped
        my_bcs.push_back(idof);
    }
    // rest of nodes on xneg hoop are simply supported and with no axial disp
    for (int ih = 0; ih < nnh; ih++) {
        int inode_L = ih * nnx;           // xneg node
        int inode_R = inode_L + nnx - 1;  // xpos node
        if (inode_L != 0) {               // xneg nodes
            // for (int idof = 0; idof < 3; idof++) {  // simply supported
            for (int idof = 0; idof < 6; idof++) {  // clamped
                // constrain u,v,w disp on xneg edge
                if (inode_L != 0) my_bcs.push_back(vpn * inode_L + offset + idof);
            }
            // also constrain the thx rotation?
            // then hellinger reissner bcs
            if constexpr (IS_HR_ELEM) {
                my_bcs.push_back(vpn * inode_L + 0);  // e11 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 1);  // e12 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 2);  // e22 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 3);  // gam13 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 4);  // gam23 strain-gap disp
            }
        }
        // xpos nodes
        // for (int idof = 1; idof < 3; idof++) {  // simply supported
        for (int idof = 0; idof < 6; idof++) {  // clamped
            // only constraint v,w on xpos edge (TODO : later make disp control here)
            my_bcs.push_back(6 * inode_R + offset + idof);
        }
        if constexpr (IS_HR_ELEM) {
            my_bcs.push_back(vpn * inode_R + 0);  // e11 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 1);  // e12 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 2);  // e22 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 3);  // gam13 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 4);  // gam23 strain-gap disp
        }
    }

    // make hostvec of bcs now
    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    // all elements done with same pattern except last one before closing hoop
    for (int ihe = 0; ihe < nhe - 1; ihe++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * ihe + ixe;
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = order * ixe + ilx;
                int iy = order * ihe + ily;
                int inode = nnx * iy + ix;

                elem_conn[Basis::num_nodes * ielem + iloc] = inode;
            }
        }
    }
    // last elements to close hoop
    int ihe = nhe - 1;
    for (int ixe = 0; ixe < nxe; ixe++) {
        int ielem = nxe * ihe + ixe;
        for (int iloc = 0; iloc < n * n; iloc++) {
            int ilx = iloc % n, ily = iloc / n;
            int ix = order * ixe + ilx;
            int iy = order * ihe + ily;
            if (ily == n - 1) iy = 0;  // loops back around
            int inode = nnx * iy + ix;

            elem_conn[Basis::num_nodes * ielem + iloc] = inode;
        }
    }

    // printf("checkpoint 3 - post elem_conn\n");

    // make element connectivities now
    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the xyz-coordinates of the cylinder
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = L / (nnx - 1);
    T dth = 2 * M_PI / nnh;
    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * ih + ix;

            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            T x[1] = {0}, th[1] = {0}, R_mid[1] = {0};
            if constexpr (Basis::order == 1) {
                x[0] = dx * ix;
                th[0] = dth * ih;
                R_mid[0] = R;
            } else {
                // higher-order nodes
                int ix_corner = (ix / order) * order;
                int ih_corner = (ih / order) * order;

                // local node index inside element
                int ix_local = ix % order;
                int ih_local = ih % order;

                // get reference Gauss point [-1,1]
                T node_pt[2];
                Basis::getNodePoint(n * ih_local + ix_local, node_pt);
                T xi = node_pt[0], eta = node_pt[1];

                // physical element corners
                T x0 = dx * ix_corner;
                T x1 = dx * (ix_corner + (n - 1));  // last node in this element
                T th0 = dth * ih_corner;
                T th1 = dth * (ih_corner + (n - 1));

                x[0] = 0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1;
                th[0] = 0.5 * (1.0 - eta) * th0 + 0.5 * (1.0 + eta) * th1;
                R_mid[0] = R;
            }
            if (imperfection) {
                T x_hat = x[0] / L;
                T th_hat = th[0] / 2 / M_PI;

                // can change settings here
                T imp_mag = 0.5;
                T imp_shape = sin(x_hat * imp_x * M_PI) * sin(th_hat * imp_hoop * M_PI);
                R_mid[0] += thick * imp_mag * imp_shape;
            }

            xpt_node[0] = x[0];
            xpt_node[1] = R_mid[0] * sin(th[0]);
            xpt_node[2] = R_mid[0] * cos(th[0]);
        }
    }

    // printf("checkpoint 4 - post xpts\n");

    HostVec<Data> physData(num_elements, Data(E, nu, thick, rho, ys));

    // printf("checkpoint 5 - create physData\n");

    // make elem_components
    assert(nxe % nx_comp == 0);
    assert(nhe % ny_comp == 0);
    int num_components = nx_comp * ny_comp;
    int nxe_per_comp = nxe / nx_comp;
    int nye_per_comp = nhe / ny_comp;

    HostVec<int> elem_components(num_elements);
    for (int iye = 0; iye < nhe; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = nx_comp * iy_comp + ix_comp;

            elem_components[ielem] = icomp;
        }
    }

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData, num_components, elem_components);

    // printf("checkpoint 6 - create assembler\n");

    return assembler;
}

template <typename T, class Basis, class Phys, int load_case = 2>
T *getCylinderLoads(int nxe, int nhe, double L, double R, double load_mag) {
    /*
    make compressive loads on the xpos edge of cylinder whose axis is in the (1,0,0) or x-direction
    TODO : later we will switch from this load control to disp control
    */

    const int order = Basis::order;
    const int n = order + 1;
    using Quadrature = typename Basis::Quadrature;

    // number of nodes per direction
    int nnx = order * nxe + 1;
    int nnh = order * nhe;
    int num_nodes = nnx * nnh;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    T dx = L / (nnx - 1);
    T dth = 2 * M_PI / nnh;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // loop over each element to compute load shape-function integrals
    for (int ihe = 0; ihe < nhe; ihe++) {
        // bool last_hoop = ihe == nhe - 1;
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * ihe + ixe;

            // build element nodes
            int elem_nodes[Basis::num_nodes] = {0};
            // get nodal coords of the element
            T elem_xpts[3 * Basis::num_nodes] = {0};
            for (int lnode = 0; lnode < n * n; lnode++) {
                int lx = lnode % n, lh = lnode / n;
                int ix = order * ixe + lx, ih = order * ihe + lh;

                // get xi and eta at a node point
                T node_pt[2];
                Basis::getNodePoint(lnode, node_pt);
                T xi = node_pt[0], eta = node_pt[1];

                // higher-order nodes
                int ix_corner = (ix / order) * order, ih_corner = (ih / order) * order;
                // int ix_local = ix % order, ih_local = ih % order;
                if (lx == n - 1) ix_corner -= order;
                if (lh == n - 1) ih_corner -= order;  // so x0, x1 same for whole element, etc.
                // physical element corners
                T x0 = dx * ix_corner, x1 = dx * (ix_corner + (n - 1));
                T th0 = dth * ih_corner, th1 = dth * (ih_corner + (n - 1));
                // map reference [-1,1] → [x0,x1] and [y0,y1]
                elem_xpts[3 * lnode] = 0.5 * (1.0 - xi) * x0 + 0.5 * (1.0 + xi) * x1;
                T th = 0.5 * (1.0 - eta) * th0 + 0.5 * (1.0 + eta) * th1;

                // if (ielem < 20) {
                //     printf("node %d: x0 %.4e to x1 %.4e with x %.4e, th %.4e\n", nnx * ih + ix,
                //     x0, x1, elem_xpts[3 * lnode], th); printf("\txi %.4e, eta %.4e\n", xi, eta);
                // }
                elem_xpts[3 * lnode + 1] = R * sin(th);
                elem_xpts[3 * lnode + 2] = R * cos(th);

                if (ih == nnh)
                    ih = 0;  // hoop closes back on itself // only put here otherwise nodes not
                             // quite right for closing loop elems
                elem_nodes[lnode] = nnx * ih + ix;
            }

            // debug, compute dXdxi and dXdeta for first node
            T _pt[2];
            Basis::getNodePoint(0, _pt);
            // compute the computational coord gradients of Xpts for xi, eta
            T dXdxi[3], dXdeta[3];
            Basis::template interpFieldsGrad<3, 3>(_pt, elem_xpts, dXdxi, dXdeta);

            // compute shell normals
            T fn[3 * Basis::num_nodes] = {0.0};
            ShellComputeNodeNormals<T, Basis>(elem_xpts, fn);

            // if (ielem < 10) {
            //     printf("ielem %d, elem_nodes: ", ielem);
            //     printVec<int>(Basis::num_nodes, elem_nodes);
            //     printf("elem_xpts: ");
            //     printVec<T>(3 * Basis::num_nodes, elem_xpts);
            //     printf("elem_fn: ");
            //     printVec<T>(3 * Basis::num_nodes, fn);
            //     printf("_dXdxi: ");
            //     printVec<T>(3, dXdxi);
            //     printf("_dXdeta: ");
            //     printVec<T>(3, dXdeta);
            // }

            // now do gauss quadrature integral
            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                T pt[2];
                T weight = Quadrature::getQuadraturePoint(iquad, pt);

                // interp x,y,z to the quadrature point (using element basis functions)
                T xpt[3] = {0};
                Basis::template interpFields<3, 3>(pt, elem_xpts, xpt);

                // compute J = d(x,y,z)/dxi,deta
                T J = getDetXd<T, Basis>(pt, elem_xpts, fn);

                // compute load magnitudes at the quadrature point
                T x_hat = xpt[0] / L;
                // don't do that cause sometimes not on surface, mag may change some
                // T y_hat = xpt[1] / R;
                // T z_hat = xpt[2] / R;
                T th = atan2(xpt[1], xpt[2]);
                T th_hat = th / 2 / M_PI;
                T mag = load_mag *
                        (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                         0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                        sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);
                mag *= weight * J;

                // compute element basis functions at the quadpt (for nodal load distribution)
                T N[Basis::num_nodes] = {0.0};
                Basis::getBasis(pt, N);

                // if (ielem < 10) {
                //     printf("quadpt (%.4e, %.4e), xpt: ", pt[0], pt[1]);
                //     printVec<T>(3, xpt);
                //     printf("J %.4e, x_hat %.4e, th %.4e\n", J, x_hat, th);
                // }

                // now loop over each node to distribute load integral among nodes
                for (int lnode = 0; lnode < n * n; lnode++) {
                    int inode = elem_nodes[lnode];
                    T nodal_mag = mag * N[lnode];

                    // if (ielem < 100) {
                    //     printf("load mag %.4e and th %.4e and inode %d\n", nodal_mag, th, inode);
                    // }

                    // add to each node now using element shape functions
                    // y and z components of the load
                    my_loads[vpn * inode + offset + 1] += sin(th) * nodal_mag;
                    my_loads[vpn * inode + offset + 2] += cos(th) * nodal_mag;
                }  // end of nodal distribution loop
            }      // end of quadpt loop

        }  // end of x element loop
    }      // end of hoop element loop

    return my_loads;
}

template <typename T, class Assembler>
T *getCylinderLoadsRobust(Assembler &assembler, int nxe, int nhe, double L, double R,
                          double load_mag, double in_plane_frac = 0.0) {
    /* more robust version of cylinder loads */

    using Phys = typename Assembler::Phys;
    using Basis = typename Assembler::Basis;

    const int order = Basis::order;
    const int n = order + 1;
    using Quadrature = typename Basis::Quadrature;

    // number of nodes per direction
    int nnx = order * nxe + 1;
    int nnh = order * nhe;
    int num_nodes = nnx * nnh;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // get importnat data out of assembler and move to host
    T *h_xpts = assembler.getXpts().createHostVec().getPtr();
    int *h_conn = assembler.getConn().createHostVec().getPtr();

    // loop over each element to compute load shape-function integrals
    for (int ihe = 0; ihe < nhe; ihe++) {
        // bool last_hoop = ihe == nhe - 1;
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * ihe + ixe;

            // build element nodes and xpts from assembler
            int *elem_nodes = &h_conn[Basis::num_nodes * ielem];
            // get nodal coords of the element
            T elem_xpts[3 * Basis::num_nodes] = {0};
            for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                int inode = elem_nodes[lnode];
                for (int dir = 0; dir < 3; dir++) {
                    elem_xpts[3 * lnode + dir] = h_xpts[3 * inode + dir];
                }
            }

            // compute shell normals
            T fn[3 * Basis::num_nodes] = {0.0};
            ShellComputeNodeNormals<T, Basis>(elem_xpts, fn);

            // now do gauss quadrature integral
            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                T pt[2];
                T weight = Quadrature::getQuadraturePoint(iquad, pt);

                // interp x,y,z to the quadrature point (using element basis functions)
                T xpt[3] = {0};
                Basis::template interpFields<3, 3>(pt, elem_xpts, xpt);

                // compute J = d(x,y,z)/dxi,deta
                T J = getDetXd<T, Basis>(pt, elem_xpts, fn);

                // compute load magnitudes at the quadrature point
                T x_hat = xpt[0] / L;
                T th = atan2(xpt[1], xpt[2]);
                T th_hat = th / 2 / M_PI;
                T mag = load_mag *
                        (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                         0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                        sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);

                // compute element basis functions at the quadpt (for nodal load distribution)
                T N[Basis::num_nodes] = {0.0};
                Basis::getBasis(pt, N);

                // now loop over each node to distribute load integral among nodes
                for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                    int inode = elem_nodes[lnode];
                    T nodal_mag = mag * N[lnode] * weight * J;

                    // in plane load fraction
                    my_loads[vpn * inode + offset] += nodal_mag * -1.0 * in_plane_frac;

                    // add to each node now using element shape functions
                    // y and z components of the load
                    my_loads[vpn * inode + offset + 1] += sin(th) * nodal_mag;
                    my_loads[vpn * inode + offset + 2] += cos(th) * nodal_mag;
                }  // end of nodal distribution loop
            }      // end of quadpt loop

        }  // end of x element loop
    }      // end of hoop element loop

    return my_loads;
}

template <typename T, class Assembler>
T *getCylinderLoadsSimple(Assembler &assembler, int nxe, int nhe, double L, double R,
                          double load_mag) {
    /* more robust version of cylinder loads */

    using Phys = typename Assembler::Phys;
    using Basis = typename Assembler::Basis;

    const int order = Basis::order;
    const int n = order + 1;
    using Quadrature = typename Basis::Quadrature;

    // number of nodes per direction
    int nnx = order * nxe + 1;
    int nnh = order * nhe;
    int num_nodes = nnx * nnh;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    // get importnat data out of assembler and move to host
    T *h_xpts = assembler.getXpts().createHostVec().getPtr();
    int *h_conn = assembler.getConn().createHostVec().getPtr();

    // loop over each element to compute load shape-function integrals
    for (int ihe = 0; ihe < nhe; ihe++) {
        // bool last_hoop = ihe == nhe - 1;
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * ihe + ixe;

            // build element nodes and xpts from assembler
            int *elem_nodes = &h_conn[Basis::num_nodes * ielem];
            // get nodal coords of the element
            T elem_xpts[3 * Basis::num_nodes] = {0};
            for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                int inode = elem_nodes[lnode];
                for (int dir = 0; dir < 3; dir++) {
                    elem_xpts[3 * lnode + dir] = h_xpts[3 * inode + dir];
                }
            }

            // compute shell normals
            T fn[3 * Basis::num_nodes] = {0.0};
            ShellComputeNodeNormals<T, Basis>(elem_xpts, fn);

            // now do gauss quadrature integral
            for (int iquad = 0; iquad < Quadrature::num_quad_pts; iquad++) {
                T pt[2];
                T weight = Quadrature::getQuadraturePoint(iquad, pt);

                // interp x,y,z to the quadrature point (using element basis functions)
                T xpt[3] = {0};
                Basis::template interpFields<3, 3>(pt, elem_xpts, xpt);

                // compute J = d(x,y,z)/dxi,deta
                T J = getDetXd<T, Basis>(pt, elem_xpts, fn);

                // compute load magnitudes at the quadrature point
                T x_hat = xpt[0] / L;
                T th = atan2(xpt[1], xpt[2]);
                T th_hat = th / 2 / M_PI;
                T mag = load_mag;

                // compute element basis functions at the quadpt (for nodal load distribution)
                T N[Basis::num_nodes] = {0.0};
                Basis::getBasis(pt, N);

                // now loop over each node to distribute load integral among nodes
                for (int lnode = 0; lnode < Basis::num_nodes; lnode++) {
                    int inode = elem_nodes[lnode];
                    T nodal_mag = mag * N[lnode] * weight * J;

                    // in plane load fraction
                    // my_loads[vpn * inode + offset] += nodal_mag * -1.0 * in_plane_frac;

                    // add to each node now using element shape functions
                    // y and z components of the load
                    my_loads[vpn * inode + offset + 1] += sin(th) * nodal_mag;
                    my_loads[vpn * inode + offset + 2] += cos(th) * nodal_mag;
                }  // end of nodal distribution loop
            }      // end of quadpt loop

        }  // end of x element loop
    }      // end of hoop element loop

    return my_loads;
}

template <class Assembler>
Assembler createHemisphereAssembler(int nxe, int nhe, double phi, double R, double E, double nu,
                                    double thick, double rho = 2500, double ys = 350e6,
                                    int nx_comp = 1, int ny_comp = 1) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;
    using Physics = typename Assembler::Phys;

    // hemisphere is like a cylinder in terms of topology..

    // number of nodes per direction
    const int order = Basis::order;
    int n = order + 1;
    int nnx = order * nxe + 1;  // axial nodes
    int nnh = order * nhe;      // number of hoop nodes
    int num_nodes = nnx * nnh;
    int num_elements = nxe * nhe;

    constexpr bool IS_HR_ELEM = Physics::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Physics::vars_per_node;

    if constexpr (Basis::order > 1) {
        printf(
            "ERROR TODO, need to add different GP spacing of mid-side nodes for assembler of "
            "hemisphere\n");
        exit(0);
    }

    // hemisphere is in x direction along its axis, circle planes are in yz plane (like the
    // cylinder)

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // TODO : will change to use disp control BCs later
    // node 0 has dof 123456, changed now to just 123
    // for (int idof = 0; idof < 3; idof++) {
    for (int idof = 0; idof < vpn; idof++) {  // clamped
        my_bcs.push_back(idof);
    }
    // rest of nodes on xneg hoop are simply supported and with no axial disp
    for (int ih = 0; ih < nnh; ih++) {
        int inode_L = ih * nnx;           // xneg node
        int inode_R = inode_L + nnx - 1;  // xpos node
        if (inode_L != 0) {               // xneg nodes
            // for (int idof = 0; idof < 3; idof++) {  // simply supported
            for (int idof = 0; idof < 6; idof++) {  // clamped
                // constrain u,v,w disp on xneg edge
                if (inode_L != 0) my_bcs.push_back(vpn * inode_L + offset + idof);
            }
            // also constrain the thx rotation?
            // then hellinger reissner bcs
            if constexpr (IS_HR_ELEM) {
                my_bcs.push_back(vpn * inode_L + 0);  // e11 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 1);  // e12 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 2);  // e22 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 3);  // gam13 strain-gap disp
                my_bcs.push_back(vpn * inode_L + 4);  // gam23 strain-gap disp
            }
        }
        // xpos nodes
        // for (int idof = 1; idof < 3; idof++) {  // simply supported
        for (int idof = 0; idof < 6; idof++) {  // clamped
            // only constraint v,w on xpos edge (TODO : later make disp control here)
            my_bcs.push_back(6 * inode_R + offset + idof);
        }
        if constexpr (IS_HR_ELEM) {
            my_bcs.push_back(vpn * inode_R + 0);  // e11 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 1);  // e12 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 2);  // e22 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 3);  // gam13 strain-gap disp
            my_bcs.push_back(vpn * inode_R + 4);  // gam23 strain-gap disp
        }
    }

    // make hostvec of bcs now
    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    // all elements done with same pattern except last one before closing hoop
    for (int ihe = 0; ihe < nhe - 1; ihe++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * ihe + ixe;
            for (int iloc = 0; iloc < n * n; iloc++) {
                int ilx = iloc % n, ily = iloc / n;
                int ix = order * ixe + ilx;
                int iy = order * ihe + ily;
                int inode = nnx * iy + ix;

                elem_conn[Basis::num_nodes * ielem + iloc] = inode;
            }
        }
    }
    // last elements to close hoop
    int ihe = nhe - 1;
    for (int ixe = 0; ixe < nxe; ixe++) {
        int ielem = nxe * ihe + ixe;
        for (int iloc = 0; iloc < n * n; iloc++) {
            int ilx = iloc % n, ily = iloc / n;
            int ix = order * ixe + ilx;
            int iy = order * ihe + ily;
            if (ily == n - 1) iy = 0;  // loops back around
            int inode = nnx * iy + ix;

            elem_conn[Basis::num_nodes * ielem + iloc] = inode;
        }
    }

    // printf("checkpoint 3 - post elem_conn\n");

    // make element connectivities now
    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the xyz-coordinates of the cylinder
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx_phi = phi / (nnx - 1);  // spherical arc angle along x direction
    T dth = 2 * M_PI / nnh;
    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * ih + ix;

            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            T x_phi[1] = {0}, th[1] = {0}, R_mid[1] = {0};
            if constexpr (Basis::order == 1) {
                x_phi[0] = dx_phi * ix;
                th[0] = dth * ih;
                R_mid[0] = R;
            } else {
                int ix_corner = (ix / n) * n, ih_corner = (ih / n) * n;
                // here nx is the number of points in element (related to elem order)
                T xi = Basis::getGaussPoint(ix % n), eta = Basis::getGaussPoint(ih % n);
                x_phi[0] = dx_phi * ix_corner + 0.5 * (1 + xi) * (dx_phi * order);
                th[0] = dth * ih_corner + 0.5 * (1 + eta) * (dth * order);
                R_mid[0] = R;
            }

            // now use spherical coordinates here with radius of sphere R
            T R_yz = R_mid[0] * cos(x_phi[0]);
            T x = R_mid[0] * sin(x_phi[0]);

            xpt_node[0] = x;
            xpt_node[1] = R_yz * sin(th[0]);
            xpt_node[2] = R_yz * cos(th[0]);
        }
    }

    // printf("checkpoint 4 - post xpts\n");

    HostVec<Data> physData(num_elements, Data(E, nu, thick, rho, ys));

    // printf("checkpoint 5 - create physData\n");

    // make elem_components
    assert(nxe % nx_comp == 0);
    assert(nhe % ny_comp == 0);
    int num_components = nx_comp * ny_comp;
    int nxe_per_comp = nxe / nx_comp;
    int nye_per_comp = nhe / ny_comp;

    HostVec<int> elem_components(num_elements);
    for (int iye = 0; iye < nhe; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = nx_comp * iy_comp + ix_comp;

            elem_components[ielem] = icomp;
        }
    }

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData, num_components, elem_components);

    // printf("checkpoint 6 - create assembler\n");

    return assembler;
}

template <typename T, class Phys, int load_case = 2>
T *getHemisphereLoads(int nxe, int nhe, double phi, double R, double load_mag) {
    /*
    make compressive loads on the xpos edge of cylinder whose axis is in the (1,0,0) or x-direction
    TODO : later we will switch from this load control to disp control
    */

    // if constexpr (Basis::order > 1) {
    //     printf("ERROR TODO, need to add different GP spacing of mid-side nodes for loads on
    //     hemisphere\n"); exit(0);
    // }

    // number of nodes per direction
    int nnx = nxe + 1;
    int nnh = nhe + 1;
    int num_nodes = nnx * nnh;

    constexpr bool IS_HR_ELEM = Phys::hellingerReissner;
    int offset = IS_HR_ELEM ? 5 : 0;  // for where standard u,v,w,thx,thy,thz DOF are
    int vpn = Phys::vars_per_node;

    T dx_phi = phi / nxe;  // spherical arc angle
    T dth = 2 * M_PI / nhe;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            T c_phi = dx_phi * ix;
            T th = dth * ih;
            T R_yz = R * cos(c_phi);
            T x = R * sin(c_phi);
            T y = R_yz * sin(th);
            T z = R_yz * cos(th);
            int inode = nnx * ih + ix;
            // petal load
            // rose shape in hoop, chirp in x direction
            T x_hat = c_phi / phi;
            T th_hat = th / 2 / M_PI;
            // T mag = load_mag * (0.7 * cos(5 * th) + 0.3 * cos(10 * th + 3.14159 / 6.0)) *
            // sin(2 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);
            T mag = load_mag *
                    (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                     0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                    sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);

            // y and z transverse loads in radial direction only
            my_loads[vpn * inode + offset + 1] = sin(th) * mag;
            my_loads[vpn * inode + offset + 2] = cos(th) * mag;
        }
    }
    return my_loads;
}
