#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "assembler.h"

template <class Assembler, class Vec>
void printToVTK(Assembler assembler, Vec soln, std::string filename) {
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
    int num_nodes = assembler.get_num_nodes();
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    // print all the xpts coordinates
    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();

    using Phys = typename Assembler::Phys;
    int vpn = Phys::vars_per_node;
    int std_vpn = Phys::std_vpn;
    int offset = (vpn == std_vpn) ? 0 : 5;  // for hellinger-reissner

    double *xpts_ptr = h_xpts.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // print all the cells
    int num_elems = assembler.get_num_elements();
    int nodes_per_elem = Assembler::vars_nodes_per_elem;
    int num_elem_nodes = num_elems * (nodes_per_elem + 1);  // repeats here
    myfile << "CELLS " << num_elems << " " << num_elem_nodes << "\n";

    auto d_vars_conn = assembler.getConn();
    auto h_vars_conn = d_vars_conn.createHostVec();
    int *conn_ptr = h_vars_conn.getPtr();

    if (nodes_per_elem == 4) {
        const int32_t local_perm[4] = {0, 1, 3, 2};
        for (int ielem = 0; ielem < num_elems; ielem++) {
            const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
            myfile << nodes_per_elem;
            for (int inode = 0; inode < nodes_per_elem; inode++) {
                myfile << sp << elem_conn[local_perm[inode]];
            }
            myfile << "\n";
        }
    } else if (nodes_per_elem == 9) {
        const int32_t local_perm[9] = {0, 2, 8, 6, 1, 5, 7, 3, 4};
        for (int ielem = 0; ielem < num_elems; ielem++) {
            const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
            myfile << nodes_per_elem;
            for (int inode = 0; inode < nodes_per_elem; inode++) {
                myfile << sp << elem_conn[local_perm[inode]];
            }
            myfile << "\n";
        }
    }

    // cell type 9 is for CQUAD4 basically
    myfile << "CELL_TYPES " << num_elems << "\n";
    int cell_type = (nodes_per_elem == 4) ? 9 : 23;  // CQUAD4 is type 9, CQUAD9 is type 23
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << cell_type << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";

    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset] << sp;
        myfile << soln[vpn * inode + offset + 1] << sp;
        myfile << soln[vpn * inode + offset + 2] << "\n";
    }

    scalarName = "rot";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[vpn * inode + offset + 3] << sp;
        myfile << soln[vpn * inode + offset + 4] << sp;
        myfile << soln[vpn * inode + offset + 5] << "\n";
    }

    // init visualization states
    int ndvs = assembler.get_num_dvs();
    DeviceVec<double> d_dvs(ndvs);
    DeviceVec<int> elem_components(num_elems);
    assembler.get_elem_components(elem_components);
    DeviceVec<double> fail_index(num_elems);
    int nstresses = assembler.get_num_vis_stresses();
    DeviceVec<double> strains(nstresses), stresses(nstresses);

    // compute visualization states
    assembler.compute_visualization_states(d_dvs, fail_index, strains, stresses);

    // write thicknesses
    double *h_dvs = d_dvs.createHostVec().getPtr();
    int *h_elem_comp = elem_components.createHostVec().getPtr();
    myfile << "CELL_DATA " << num_elems << "\n";
    scalarName = "thickness";
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        int comp_id = h_elem_comp[ielem];
        myfile << h_dvs[comp_id] << "\n";
    }

    // write failure indexes
    double *h_fail_index = fail_index.createHostVec().getPtr();
    scalarName = "fail_index";
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        myfile << h_fail_index[ielem] << "\n";
    }

    // write strains
    double *h_strains = strains.createHostVec().getPtr();
    myfile << "FIELD FieldData 4\n";
    scalarName = "midplane_strain";
    myfile << scalarName << " 3 " << num_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        myfile << h_strains[6 * ielem + 0] << " " << h_strains[6 * ielem + 1] << " "
               << h_strains[6 * ielem + 2] << "\n";
    }
    scalarName = "bending_strain";
    myfile << scalarName << " 3 " << num_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        myfile << h_strains[6 * ielem + 3] << " " << h_strains[6 * ielem + 4] << " "
               << h_strains[6 * ielem + 5] << "\n";
    }

    // write stresses
    double *h_stresses = stresses.createHostVec().getPtr();
    scalarName = "midplane_stress";
    myfile << scalarName << " 3 " << num_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        myfile << h_stresses[6 * ielem + 0] << " " << h_stresses[6 * ielem + 1] << " "
               << h_stresses[6 * ielem + 2] << "\n";
    }
    scalarName = "bending_stress";
    myfile << scalarName << " 3 " << num_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        myfile << h_stresses[6 * ielem + 3] << " " << h_stresses[6 * ielem + 4] << " "
               << h_stresses[6 * ielem + 5] << "\n";
    }

    myfile.close();
}

template <class Assembler, class Vec>
void printToVTK_points(Assembler assembler, Vec soln, std::string filename) {
    /* for point cloud data from FUN3D aero surf mesh */

    // later
    using namespace std;
    string sp = " ";
    string dataType = "double64";

    ofstream myfile;
    myfile.open(filename);
    myfile << "# vtk DataFile Version 3.0\n";
    myfile << "TACS GPU Point cloud writer\n";
    myfile << "ASCII\n";

    // make an unstructured grid even though it is really structured
    myfile << "DATASET POLYDATA\n";
    int num_nodes = assembler.get_num_nodes();
    myfile << "POINTS " << num_nodes << sp << dataType << "\n";

    // print all the xpts coordinates
    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();

    double *xpts_ptr = h_xpts.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        double *node_xpts = &xpts_ptr[3 * inode];
        myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
    }

    // list each vertex as standalong point
    myfile << "VERTICES " << num_nodes << " " << 2 * num_nodes << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << "1 " << inode << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        myfile << soln[6 * inode] << sp;
        myfile << soln[6 * inode + 1] << sp;
        myfile << soln[6 * inode + 2] << "\n";
    }

    myfile.close();
}