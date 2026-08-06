#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "matvec/gpuvec.h"

template <typename T, class Assembler>
void printToVTK_v2(Assembler assembler, T *soln, std::string filename) {
    using Basis = typename Assembler::Basis;
    using Phys = typename Assembler::Phys;

    using namespace std;

    const string sp = " ";
    const string dataType = "double";

    ofstream myfile(filename);
    myfile << "# vtk DataFile Version 2.0\n";
    myfile << "TACS GPU shell writer\n";
    myfile << "ASCII\n";
    myfile << "DATASET UNSTRUCTURED_GRID\n";

    const int num_nodes = assembler.get_num_nodes();
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    auto h_xpts = assembler.getXpts();
    double *xpts_ptr = h_xpts->getPtr();

    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // Shell quad elements: use geometry/basis nodes, not vars_nodes_per_elem.
    const int num_elems = assembler.get_num_elements();
    static constexpr int nodes_per_elem = Basis::num_nodes;

    static_assert(nodes_per_elem == 4, "printToVTK_v2 currently supports only 4-node shell quads.");

    const int entries_per_cell = nodes_per_elem + 1;
    myfile << "CELLS " << num_elems << sp << num_elems * entries_per_cell << "\n";

    auto h_conn = assembler.getConn();
    int *conn_ptr = h_conn->getPtr();

    // VTK_QUAD requires nodes ordered around the perimeter.
    // For structured shell order [0, 1, 2, 3] = [LL, LR, UL, UR],
    // VTK order should be [LL, LR, UR, UL].
    const int local_perm[4] = {0, 1, 3, 2};

    for (int ielem = 0; ielem < num_elems; ielem++) {
        const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];

        myfile << "4";
        for (int inode = 0; inode < 4; inode++) {
            myfile << sp << elem_conn[local_perm[inode]];
        }
        myfile << "\n";
    }

    // VTK_QUAD = 9
    myfile << "CELL_TYPES " << num_elems << "\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << "9\n";
    }

    const int vpn = Phys::vars_per_node;
    const int std_vpn = Phys::std_vpn;
    const int offset = (vpn == std_vpn) ? 0 : 5;  // Hellinger-Reissner offset

    myfile << "POINT_DATA " << num_nodes << "\n";

    myfile << "VECTORS disp " << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset + 0] << sp << soln[vpn * inode + offset + 1] << sp
               << soln[vpn * inode + offset + 2] << "\n";
    }

    myfile << "VECTORS rot " << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset + 3] << sp << soln[vpn * inode + offset + 4] << sp
               << soln[vpn * inode + offset + 5] << "\n";
    }

    myfile.close();
}

template <typename T, class Assembler>
void printToVTK_v2_elemsd(Assembler assembler, T *soln, int *elem_sd_ind, std::string field_name,
                          std::string filename) {
    using Basis = typename Assembler::Basis;
    using Phys = typename Assembler::Phys;

    using namespace std;

    const string sp = " ";
    const string dataType = "double";

    ofstream myfile(filename);
    myfile << "# vtk DataFile Version 2.0\n";
    myfile << "TACS GPU shell writer\n";
    myfile << "ASCII\n";
    myfile << "DATASET UNSTRUCTURED_GRID\n";

    const int num_nodes = assembler.get_num_nodes();
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    auto h_xpts = assembler.getXpts();
    double *xpts_ptr = h_xpts->getPtr();

    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // Shell quad elements: use geometry/basis nodes, not vars_nodes_per_elem.
    const int num_elems = assembler.get_num_elements();
    static constexpr int nodes_per_elem = Basis::num_nodes;

    static_assert(nodes_per_elem == 4, "printToVTK_v2 currently supports only 4-node shell quads.");

    const int entries_per_cell = nodes_per_elem + 1;
    myfile << "CELLS " << num_elems << sp << num_elems * entries_per_cell << "\n";

    auto h_conn = assembler.getConn();
    int *conn_ptr = h_conn->getPtr();

    // VTK_QUAD requires nodes ordered around the perimeter.
    // For structured shell order [0, 1, 2, 3] = [LL, LR, UL, UR],
    // VTK order should be [LL, LR, UR, UL].
    const int local_perm[4] = {0, 1, 3, 2};

    for (int ielem = 0; ielem < num_elems; ielem++) {
        const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];

        myfile << "4";
        for (int inode = 0; inode < 4; inode++) {
            myfile << sp << elem_conn[local_perm[inode]];
        }
        myfile << "\n";
    }

    // VTK_QUAD = 9
    myfile << "CELL_TYPES " << num_elems << "\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << "9\n";
    }

    const int vpn = Phys::vars_per_node;
    const int std_vpn = Phys::std_vpn;
    const int offset = (vpn == std_vpn) ? 0 : 5;  // Hellinger-Reissner offset

    myfile << "POINT_DATA " << num_nodes << "\n";

    myfile << "VECTORS disp " << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset + 0] << sp << soln[vpn * inode + offset + 1] << sp
               << soln[vpn * inode + offset + 2] << "\n";
    }

    myfile << "VECTORS rot " << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset + 3] << sp << soln[vpn * inode + offset + 4] << sp
               << soln[vpn * inode + offset + 5] << "\n";
    }

    myfile << "CELL_DATA " << num_elems << "\n";
    string scalarName = field_name;
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << elem_sd_ind[ielem] << "\n";
    }

    myfile.close();
}