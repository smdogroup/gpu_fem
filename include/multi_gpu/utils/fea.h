#pragma once
#include <fstream>
#include <vector>

#include "assembler.h"
#include "linalg/vec.h"
#include "math.h"
#include "utils.h"
#include "utils/multigpu_context.h"

template <class Assembler>
Assembler *createGPUCylinderAssembler(MultiGPUContext *ctx, int nxe, int nhe, double L, double R,
                                      double E, double nu, double thick, bool imperfection = false,
                                      int imp_x = 5, int imp_hoop = 4, double rho = 2500,
                                      double ys = 350e6, int nx_comp = 1, int ny_comp = 1) {
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
    auto bcs = new HostVec<int>(my_bcs.size());
    int *bcs_ptr = bcs->getPtr();
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs_ptr[ibc] = my_bcs.at(ibc);
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

    // make element connectivities now
    auto conn = new HostVec<int>(N, elem_conn);

    // now set the xyz-coordinates of the cylinder
    int32_t num_xpts = Geo::spatial_dim * num_nodes;
    auto xpts = new HostVec<T>(num_xpts);
    T *xpts_ptr = xpts->getPtr();
    T dx = L / (nnx - 1);
    T dth = 2 * M_PI / nnh;
    for (int ih = 0; ih < nnh; ih++) {
        for (int ix = 0; ix < nnx; ix++) {
            int inode = nnx * ih + ix;

            T *xpt_node = &xpts_ptr[Geo::spatial_dim * inode];
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

    auto physData = new HostVec<Data>(num_elements, Data(E, nu, thick, rho, ys));

    // make elem_components
    assert(nxe % nx_comp == 0);
    assert(nhe % ny_comp == 0);
    int num_components = nx_comp * ny_comp;
    int nxe_per_comp = nxe / nx_comp;
    int nye_per_comp = nhe / ny_comp;

    auto elem_components = new HostVec<int>(num_elements);
    int *elem_comp_ptr = elem_components->getPtr();
    for (int iye = 0; iye < nhe; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;
            int ix_comp = ixe / nxe_per_comp;
            int iy_comp = iye / nye_per_comp;

            int icomp = nx_comp * iy_comp + ix_comp;

            elem_comp_ptr[ielem] = icomp;
        }
    }

    // printf("h_elem_conn[%d] in cylinder: ", num_elements);
    // printVec<int>(4 * num_elements, elem_conn);

    // make the assembler
    // printf("create assembler\n");
    auto assembler = new Assembler(ctx, num_nodes, num_elements, conn, xpts, bcs, physData,
                                   num_components, elem_components);

    return assembler;
}

template <typename T, class Basis, class Phys, int load_case = 2>
T *getCylinderLoads2(int nxe, int nhe, double L, double R, double load_mag) {
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