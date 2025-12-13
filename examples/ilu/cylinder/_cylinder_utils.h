#pragma once
#include <fstream>
#include <vector>

#include "assembler.h"
#include "linalg/vec.h"
#include "math.h"
#include "utils.h"

template <class Assembler>
Assembler createCylinderAssembler(int nxe, int nhe, double L, double R, double E, double nu,
                                  double thick, bool imperfection = false, int imp_x = 5, int imp_hoop = 4) {
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
    for (int idof = 0; idof < 3; idof++) {
        my_bcs.push_back(idof);
    }
    // rest of nodes on xneg hoop are simply supported and with no axial disp
    for (int ih = 0; ih < nnh; ih++) {
        int inode_L = ih * nnx;           // xneg node
        int inode_R = inode_L + nnx - 1;  // xpos node
        if (inode_L != 0) {               // xneg nodes
            for (int idof = 0; idof < 3; idof++) {
                // constrain u,v,w disp on xneg edge
                if (inode_L != 0) my_bcs.push_back(6 * inode_L + idof);
            }
        }
        // xpos nodes
        for (int idof = 1; idof < 3; idof++) {
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

    HostVec<Data> physData(num_elements, Data(E, nu, thick));

    // printf("checkpoint 5 - create physData\n");

    // make the assembler
    Assembler assembler(num_nodes, num_nodes, num_elements, geo_conn, vars_conn, xpts, bcs,
                        physData);

    // printf("checkpoint 6 - create assembler\n");

    return assembler;
}

template <typename T, class Phys, bool compressive = false>
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
            if constexpr (compressive) {
                if (ix == nnx - 1) {
                    // on xpos edge, make compressive loads in x-direction
                    // printf("compressive load on %d\n", inode);
                    my_loads[Phys::vars_per_node * inode] = -load_mag;
                }
            } else {  // otherwise transverse
                // transverse sinusoidal magnitude
                T x_hat = x / L;
                T th_hat = th / 2 / M_PI;
                T mag = load_mag * sin(x_hat * 5 * M_PI) * sin(th_hat * 4 * M_PI);

                // y and z transverse loads in radial direction only
                my_loads[Phys::vars_per_node * inode + 1] = sin(th) * mag;
                my_loads[Phys::vars_per_node * inode + 2] = cos(th) * mag;
            }
        }
    }
    return my_loads;
}
