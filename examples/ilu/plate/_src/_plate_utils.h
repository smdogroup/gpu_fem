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
                               int nxe_per_comp = 1, int nye_per_comp = 1, bool v_constr = false) {
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
        if (v_constr) my_bcs.push_back(6 * inode + 1);  // also v constr here..
        my_bcs.push_back(6 * inode + 2);                // corresp dof 3 for w
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
Assembler createClampedPlateAssembler(int nxe, int nye, double Lx, double Ly, double E, double nu,
                               double thick, double rho = 2500, double ys = 350e6,
                               int nxe_per_comp = 1, int nye_per_comp = 1, int order = 1) { 
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
    int nnx = order * nxe + 1;
    int nny = order * nye + 1;
    int num_nodes = nnx * nny;
    int num_elements = nxe * nye;

    // printf("checkpoint 1\n");

    // make our bcs vec (note I use 1-based terminology from nastran in
    // description above) but since this is in C++ I apply BCs here 0-based as
    // in 012345
    std::vector<int> my_bcs;
    // negative x2 (or y) edge with dof 23
    for (int ix = 0; ix < nnx; ix++) {
        int iy = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);  
        my_bcs.push_back(6 * inode + 1);  
        my_bcs.push_back(6 * inode + 2); 
        my_bcs.push_back(6 * inode + 3); 
        my_bcs.push_back(6 * inode + 4);  
        my_bcs.push_back(6 * inode + 5);
    }
    // neg and pos x1 edges with dof 13 and 3 resp.
    for (int iy = 1; iy < nny; iy++) {
        // neg x1 edge
        int ix = 0;
        int inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);  
        my_bcs.push_back(6 * inode + 1);  
        my_bcs.push_back(6 * inode + 2); 
        my_bcs.push_back(6 * inode + 3); 
        my_bcs.push_back(6 * inode + 4);  
        my_bcs.push_back(6 * inode + 5);

        // pos x1 edge
        ix = nnx - 1;
        inode = nnx * iy + ix;
        my_bcs.push_back(6 * inode);  
        my_bcs.push_back(6 * inode + 1);  
        my_bcs.push_back(6 * inode + 2); 
        my_bcs.push_back(6 * inode + 3); 
        my_bcs.push_back(6 * inode + 4);  
        my_bcs.push_back(6 * inode + 5);
    }
    // pos x2 edge
    for (int ix = 1; ix < nnx - 1; ix++) {
        int iy = nny - 1;
        int inode = nnx * iy + ix;
        // printf("new bc = %d\n", 6 * inode + 2);
        my_bcs.push_back(6 * inode);  
        my_bcs.push_back(6 * inode + 1);  
        my_bcs.push_back(6 * inode + 2); 
        my_bcs.push_back(6 * inode + 3); 
        my_bcs.push_back(6 * inode + 4);  
        my_bcs.push_back(6 * inode + 5);
    }

    HostVec<int> bcs(my_bcs.size());
    // deep copy here
    for (int ibc = 0; ibc < my_bcs.size(); ibc++) {
        bcs[ibc] = my_bcs.at(ibc);
    }

    // printf("checkpoint 2 - post bcs\n");

    // printf("bcs: ");
    // printVec<int>(my_bcs.size(), bcs.getPtr());
    int n = order + 1; // num local nodes

    // now initialize the element connectivity
    int N = Basis::num_nodes * num_elements;
    int32_t *elem_conn = new int[N];
    for (int iye = 0; iye < nye; iye++) {
        for (int ixe = 0; ixe < nxe; ixe++) {
            int ielem = nxe * iye + ixe;

            // no sorted order like in MITC?
            for (int iloc = 0; iloc < n * n ; iloc++) {
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
T *getPlateLoads(int nxe, int nye, double Lx, double Ly, double load_mag,
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
            my_loads[Phys::vars_per_node * inode + 2] = nodal_load;
        }

        int ix = nnx - 1;  // pos x1 edge for in-plane
        int inode = nnx * iy + ix;
        T x = ix * dx, y = iy * dy;
        my_loads[Phys::vars_per_node * inode] = -load_mag * in_plane_frac;
    }
    return my_loads;
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

void write_to_binary(const double *array, size_t size, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::ios_base::failure("Failed to open file for writing");
    }
    out.write(reinterpret_cast<const char *>(array), size * sizeof(double));
    out.close();
}

// template <typename T>
// void write_to_csv(const T *array, size_t size, const std::string &filename) {
//     std::ofstream out(filename);
//     if (!out) {
//         throw std::ios_base::failure("Failed to open file for writing");
//     }
//     for (size_t i = 0; i < size; ++i) {
//         out << array[i];
//         if (i != size - 1) {
//             out << ",";
//         }
//     }
//     out << "\n";
//     out.close();
// }

// for MELD:
// ---------

template <typename T>
T convertNanToZero(T value) {
    // return std::isnan(value) ? 0.0f : value;

    // temporarily keep nans
    return value;
}

template <typename T>
void printGridToVTK(int nnx, int nny, HostVec<T> &x0, HostVec<T> &u, std::string filename) {
    // NOTE : better to use F5 binary for large cases, we will handle that
    // later
    using namespace std;
    string sp = " ";
    string dataType = "double64";

    ofstream myfile;
    myfile.open(filename);
    myfile << "# vtk DataFile Version 2.0\n";
    myfile << "TACS GPU shell writer\n";
    myfile << "ASCII\n";

    // make an unstructured grid even though it is really structured
    myfile << "DATASET UNSTRUCTURED_GRID\n";
    int num_nodes = nnx * nny;
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    // print all the xpts coordinates
    double *xpts_ptr = x0.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // print all the cells
    int nelems = (nnx - 1) * (nny - 1);
    int nodes_per_elem = 4;
    int num_elem_nodes = nelems * (nodes_per_elem + 1);
    myfile << "CELLS " << nelems << " " << num_elem_nodes << "\n";

    int nxe = nnx - 1, nye = nny - 1;
    for (int iy = 0; iy < nye; iy++) {
        for (int ix = 0; ix < nxe; ix++) {
            int istart = iy * nnx + ix;
            myfile << sp << 4;
            myfile << sp << istart;
            myfile << sp << istart + 1;
            myfile << sp << istart + nnx + 1;
            myfile << sp << istart + nnx;
            myfile << "\n";
        }
    }

    // cell type 9 is for CQUAD4 basically
    myfile << "CELL_TYPES " << nelems << "\n";
    for (int ielem = 0; ielem < nelems; ielem++) {
        myfile << 9 << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << convertNanToZero(u[3 * inode]) << sp;
        myfile << convertNanToZero(u[3 * inode + 1]) << sp;
        myfile << convertNanToZero(u[3 * inode + 2]) << "\n";
    }

    myfile.close();
}

template <typename T>
HostVec<T> makeGridMesh(int nnx, int nny, T Lx, T Ly, T z0) {
    int N = nnx * nny;
    HostVec<T> x0(3 * N);
    T pi = 3.14159265358979323846;
    // printf("z0 = %.4e\n", z0);

    T dx = Lx / (nnx - 1);
    T dy = Ly / (nny - 1);
    for (int iy = 0; iy < nny; iy++) {
        T eta = iy * 1.0 / (nny - 1);
        T yfac = sin(pi * eta);
        for (int ix = 0; ix < nnx; ix++) {
            T xi = ix * 1.0 / (nnx - 1);
            T xfac = sin(pi * xi);

            int ind = iy * nnx + ix;
            x0[3 * ind] = ix * dx;
            x0[3 * ind + 1] = iy * dy;
            x0[3 * ind + 2] = z0 * xfac * yfac;
            // printf("ix %d xi %.4e, iy %d eta %.4e\n", ix, xi, iy, eta);
            // printf("zval = %.4e, xfac %.4e, yfac %.4e\n", x0[3*ind+2], xfac, yfac);
        }
    }

    return x0;
}

template <typename T>
HostVec<T> makeInPlaneShearDisp(HostVec<T> &x0, T angleDeg) {
    int N = x0.getSize() / 3;
    HostVec<T> u(3 * N);
    T angleRad = angleDeg * 3.14159265 / 180.0;

    for (int inode = 0; inode < N; inode++) {
        T *xpt = &x0[3 * inode];
        T *upt = &u[3 * inode];
        upt[0] = tan(angleRad / 2.0) * xpt[1];
        upt[1] = tan(angleRad / 2.0) * xpt[0];
        upt[2] = 0.0;
    }

    return u;
}

template <typename T>
HostVec<T> makeCustomDisp(HostVec<T> &x0, T scale) {
    int N = x0.getSize() / 3;
    HostVec<T> u(3 * N);
    T pi = 3.14159265;

    for (int inode = 0; inode < N; inode++) {
        T *xpt = &x0[3 * inode];
        T *upt = &u[3 * inode];
        upt[0] = sin(4 * pi * xpt[0]) * cos(6 * pi * xpt[1]) + 0.5 * cos(7 * pi * xpt[0] * xpt[1]);
        upt[1] = cos(5 * pi * xpt[0]) * sin(4 * pi * xpt[1]) + 0.3 * sin(6 * pi * xpt[0] * xpt[1]);
        upt[2] = sin(3 * pi * xpt[0]) * cos(5 * pi * xpt[1]) + 0.4 * cos(6 * pi * xpt[0] * xpt[1]);
        for (int i = 0; i < 3; i++) {
            upt[i] *= scale;
        }
    }

    return u;
}

// Helper function to convert string to lowercase (in-place)
void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}
