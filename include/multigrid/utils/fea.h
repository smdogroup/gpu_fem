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
                               int nxe_per_comp = 1, int nye_per_comp = 1) {
    using T = typename Assembler::T;
    using Basis = typename Assembler::Basis;
    using Geo = typename Assembler::Geo;
    using Data = typename Assembler::Data;
    int order = Basis::order;

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
    int nnx = order * nxe + 1;
    int nny = order * nye + 1;
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
                int ix = order * ixe + ilx;
                int iy = order * iye + ily;
                int inode = nnx * iy + ix;

                elem_conn[Basis::num_nodes * ielem + iloc] = inode;
            }
        }
    }

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
            xpt_node[0] = dx * ix;
            xpt_node[1] = dy * iy;
            xpt_node[2] = 0.0;
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

template <typename T, class Phys>
T *getPlateNonlinearLoads(int nxe, int nye, double Lx, double Ly, double load_mag,
                          double in_plane_frac = 0.0) {
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
            T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
            my_loads[Phys::vars_per_node * inode + 2] = nodal_load * 10 / nxe / nye;
        }

        int ix = nnx - 1;  // pos x1 edge for in-plane
        int inode = nnx * iy + ix;
        T x = ix * dx, y = iy * dy;
        my_loads[Phys::vars_per_node * inode] = -load_mag * in_plane_frac / nye;
    }
    return my_loads;
}

template <typename T, class Basis, class Phys>
T *getPlateLoads(int nxe, int nye, double Lx, double Ly, double load_mag) {
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

    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);

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
            // T nodal_load = load_mag * sin(PI * x / Lx) * sin(PI * y / Ly);
            T r = sqrt(x * x + y * y);
            T th = atan2(y, x);
            T nodal_load = load_mag * sin(5.0 * PI * r) * cos(4.0 * th);

            if (ix % order == 0 && iy % order == 0) {
                // don't put loads on mid-side nodes etc of higher order (weird results)
                my_loads[Phys::vars_per_node * inode + 2] = nodal_load;  // * dx * dy;
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

    // number of nodes per direction
    int nnx = nxe + 1;  // axial nodes
    int nnh = nhe;      // number of hoop nodes
    int num_nodes = nnx * nnh;
    int num_elements = nxe * nhe;

    // cylinder is in x direction along its axis, circle planes are in yz plane

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // TODO : will change to use disp control BCs later
    // node 0 has dof 123456, changed now to just 123
    // for (int idof = 0; idof < 3; idof++) {
    for (int idof = 0; idof < 6; idof++) {  // clamped
        my_bcs.push_back(idof);
    }
    // rest of nodes on xneg hoop are simply supported and with no axial disp
    for (int ih = 0; ih < nnh; ih++) {
        int inode_L = ih * nnx;                     // xneg node
        int inode_R = inode_L + nnx - 1;            // xpos node
        if (inode_L != 0) {                         // xneg nodes
            for (int idof = 0; idof < 3; idof++) {  // simply supported
                // for (int idof = 0; idof < 6; idof++) {  // clamped
                // constrain u,v,w disp on xneg edge
                if (inode_L != 0) my_bcs.push_back(6 * inode_L + idof);
            }
        }
        // xpos nodes
        for (int idof = 1; idof < 3; idof++) {  // simply supported
            // for (int idof = 0; idof < 6; idof++) {  // clamped
            // only constraint v,w on xpos edge (TODO : later make disp control here)
            my_bcs.push_back(6 * inode_R + idof);
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
            int inode = nnx * ihe + ixe;
            int nodes[] = {inode, inode + 1, inode + nnx, inode + nnx + 1};
            for (int inode = 0; inode < Basis::num_nodes; inode++) {
                elem_conn[Basis::num_nodes * ielem + inode] = nodes[inode];
            }
        }
    }
    // last elements to close hoop
    int ihe = nhe - 1;
    for (int ixe = 0; ixe < nxe; ixe++) {
        int ielem = nxe * ihe + ixe;
        int inode_n = nnx * ihe + ixe;  // last row before closing hoop
        int inode_p = ixe;              // closes hoop so first row of nodes
        int nodes[] = {inode_n, inode_n + 1, inode_p, inode_p + 1};
        for (int inode = 0; inode < Basis::num_nodes; inode++) {
            elem_conn[Basis::num_nodes * ielem + inode] = nodes[inode];
        }
    }

    // printf("checkpoint 3 - post elem_conn\n");

    // make element connectivities now
    HostVec<int32_t> geo_conn(N, elem_conn);
    HostVec<int32_t> vars_conn(N, elem_conn);

    // now set the xyz-coordinates of the cylinder
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    HostVec<T> xpts(num_xpts);
    T dx = L / nxe;
    T dth = 2 * M_PI / nhe;
    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * ih + ix;
            T *xpt_node = &xpts[Geo::spatial_dim * inode];
            xpt_node[0] = dx * ix;
            T th = dth * ih;

            // apply imperfection to shell midplane coords
            T R_mid = R;
            if (imperfection) {
                T x_hat = xpt_node[0] / L;
                T th_hat = th / 2 / M_PI;

                // can change settings here
                T imp_mag = 0.5;
                T imp_shape = sin(x_hat * imp_x * M_PI) * sin(th_hat * imp_hoop * M_PI);
                R_mid += thick * imp_mag * imp_shape;
            }

            xpt_node[1] = R_mid * sin(th);
            xpt_node[2] = R_mid * cos(th);
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
T *getCylinderLoads(int nxe, int nhe, double L, double R, double load_mag) {
    /*
    make compressive loads on the xpos edge of cylinder whose axis is in the (1,0,0) or x-direction
    TODO : later we will switch from this load control to disp control
    */

    // number of nodes per direction
    int nnx = nxe + 1;
    int nnh = nhe + 1;
    int num_nodes = nnx * nnh;

    T dx = L / nxe;
    T dth = 2 * M_PI / nhe;

    int num_dof = Phys::vars_per_node * num_nodes;
    T *my_loads = new T[num_dof];
    memset(my_loads, 0.0, num_dof * sizeof(T));

    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            T x = dx * ix;
            T th = dth * ih;
            T y = R * sin(th);
            T z = R * cos(th);
            int inode = nnx * ih + ix;
            if constexpr (load_case == 1) {
                if (ix == nnx - 1) {
                    // on xpos edge, make compressive loads in x-direction
                    // printf("compressive load on %d\n", inode);
                    my_loads[Phys::vars_per_node * inode] = -load_mag;
                }
            } else if (load_case == 2) {  // otherwise transverse simply sine-sine load
                // transverse sinusoidal magnitude
                T x_hat = x / L;
                T th_hat = th / 2 / M_PI;
                T mag = load_mag * sin(x_hat * 5 * M_PI) * sin(th_hat * 4 * M_PI);

                // y and z transverse loads in radial direction only
                my_loads[Phys::vars_per_node * inode + 1] = sin(th) * mag;
                my_loads[Phys::vars_per_node * inode + 2] = cos(th) * mag;
            } else if (load_case == 3) {  // petal load
                // rose shape in hoop, chirp in x direction
                T x_hat = x / L;
                T th_hat = th / 2 / M_PI;
                // T mag = load_mag * (0.7 * cos(5 * th) + 0.3 * cos(10 * th + 3.14159 / 6.0)) *
                // sin(2 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);
                T mag = load_mag *
                        (0.3 * cos(5 * th + 2.0 * M_PI * x_hat) +
                         0.7 * cos(10 * th + 3.14159 / 6.0 + 5.3 * M_PI * x_hat)) *
                        sin(5 * M_PI * x_hat + 0.5 * 2.0 * x_hat * x_hat);

                // y and z transverse loads in radial direction only
                my_loads[Phys::vars_per_node * inode + 1] = sin(th) * mag;
                my_loads[Phys::vars_per_node * inode + 2] = cos(th) * mag;
            }
        }
    }
    return my_loads;
}
