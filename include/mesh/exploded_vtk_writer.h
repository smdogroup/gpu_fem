#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "assembler.h"

enum EXPLODED {
    INT_STRUCT,
    UPPER_SKIN,
    LOWER_SKIN
};

template <class Assembler, class Vec, EXPLODED exploded>
void explodedPrintToVTK(Assembler assembler, Vec &soln, const std::string filename) {
    /* write an exploded view of a wing to VTK (assuming only upper and lower skin fully offset Z>0 or Z<0) trick (such as AOB wing) */

    // ================================================
    /* compute the elements and nodes that belong to this sub-comp of exploded mesh */
    // ================================================
    int num_nodes = assembler.get_num_nodes();
    int num_elements = assembler.get_num_elements();
    int num_components = assembler.get_num_components();

    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();
    double *xpts_ptr = h_xpts.getPtr();
    using Phys = typename Assembler::Phys;
    int vpn = Phys::vars_per_node;
    int std_vpn = Phys::std_vpn;
    int offset = (vpn == std_vpn) ? 0 : 5;  // for hellinger-reissner
    int nodes_per_elem = Assembler::vars_nodes_per_elem;
    auto d_vars_conn = assembler.getConn();
    auto h_vars_conn = d_vars_conn.createHostVec();
    int *conn_ptr = h_vars_conn.getPtr();
    DeviceVec<int> elem_components(num_elements);
    assembler.get_elem_components(elem_components);
    int *h_elem_comp = elem_components.createHostVec().getPtr();


    // first identify which components belong to this one of 3 exploded meshes (0,1,2) for (int struct, upper skin, lower skin)
    bool *upper_skin_comp_mask = new bool[num_components];
    bool *lower_skin_comp_mask = new bool[num_components];
    memset(upper_skin_comp_mask, true, num_components * sizeof(bool)); // assume true until proven false
    memset(lower_skin_comp_mask, true, num_components * sizeof(bool)); // assume true until proven false
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_comp[ielem];
        int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];
        for (int i = 0; i < nodes_per_elem; i++) {
            int inode = elem_nodes[i];
            double *xpt = &h_xpts[3 * inode];
            if (xpt[2] < 0.0) {
                // then this component cannot be upper skin
                upper_skin_comp_mask[icomp] = false;
            }
            if (xpt[2] > 0.0) {
                // then this component cannot be lower skin
                lower_skin_comp_mask[icomp] = false;
            }
        }
    }
    // now determine if each component is the desired EXPLODED exploded enum type from template
    bool *comp_mask = new bool[num_components];
    if constexpr (exploded == UPPER_SKIN) {
        memcpy(comp_mask, upper_skin_comp_mask, num_components * sizeof(bool));
    } else if constexpr (exploded == LOWER_SKIN) {
        memcpy(comp_mask, lower_skin_comp_mask, num_components * sizeof(bool));
    } else if constexpr (exploded == INT_STRUCT) {
        // do != upper skin and != lower skin check
        for (int icomp = 0; icomp < num_components; icomp++) {
            comp_mask[icomp] = !upper_skin_comp_mask[icomp] && !lower_skin_comp_mask[icomp];
        }
    }

    // now determine which elements and nodes are in the components of comp mask
    bool *elem_mask = new bool[num_elements];
    memset(elem_mask, false, num_elements * sizeof(bool));
    bool *nodes_mask = new bool[num_nodes];
    memset(nodes_mask, false, num_nodes * sizeof(bool));
    int num_mask_nodes = 0, num_mask_elems = 0;
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_comp[ielem];
        if (comp_mask[icomp]) {
            elem_mask[ielem] = true; // it's in an allowed component
            num_mask_elems++;
            int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];
            for (int i = 0; i < nodes_per_elem; i++) {
                int inode = elem_nodes[i];
                nodes_mask[inode] = true; // it's in an allowed component
            }
        }
    }
    // to avoid non-uniqueness, loop back through nodes mask after construction
    // compute also a map from prev total [0, num_nodes) list to [0, num_mask_nodes) list for elem connectivity
    int *red_node_map = new int[num_nodes]; // old node to reduced node (as long as reduced node in the exploded type)
    memset(red_node_map, 0, num_nodes * sizeof(int));
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            red_node_map[inode] = num_mask_nodes;
            num_mask_nodes++;
        }
    }

    // ===========================================
    /* proceed as normal to write VTK file*/
    // ===========================================

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
    myfile << "POINTS " << num_mask_nodes << sp << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            double *node_xpts = &xpts_ptr[3 * inode];
            myfile << node_xpts[0] << sp << node_xpts[1] << sp << node_xpts[2] << "\n";
        }
    }

    // print all the cells
    int num_elems = assembler.get_num_elements();
    int num_elem_nodes = num_mask_elems * (nodes_per_elem + 1);  // repeats here
    myfile << "CELLS " << num_mask_elems << " " << num_elem_nodes << "\n";

    if (nodes_per_elem == 4) {
        const int32_t local_perm[4] = {0, 1, 3, 2};
        for (int ielem = 0; ielem < num_elems; ielem++) {
            if (elem_mask[ielem]) {
                const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
                myfile << nodes_per_elem;
                for (int i = 0; i < nodes_per_elem; i++) {
                    int inode = elem_conn[local_perm[i]];
                    int red_node = red_node_map[inode]; // get reduced node since less total # nodes
                    myfile << sp << red_node;
                }
                myfile << "\n";
            }
        }
    }
    // add for higher order later (time crunch on paper rn)
    //  else if (nodes_per_elem == 9) {
    //     const int32_t local_perm[9] = {0, 2, 8, 6, 1, 5, 7, 3, 4};
    //     for (int ielem = 0; ielem < num_elems; ielem++) {
    //         const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
    //         myfile << nodes_per_elem;
    //         for (int inode = 0; inode < nodes_per_elem; inode++) {
    //             myfile << sp << elem_conn[local_perm[inode]];
    //         }
    //         myfile << "\n";
    //     }
    // } else if (nodes_per_elem == 16) {
    //     // VTK_LAGRANGE_QUADRILATERAL (order 3 → 16 nodes)
    //     const int32_t local_perm[16] = {
    //         // corners (LL, LR, UR, UL)
    //         0, 3, 15, 12,
    //         // edge internal nodes (bottom left->right, right bottom->top,
    //         //                      top right->left, left top->bottom)
    //         1, 2, 7, 11, 14, 13, 8, 4,
    //         // interior nodes (row-major for i=1..2, j=1..2, bottom->top)
    //         5, 6, 9, 10};

    //     for (int ielem = 0; ielem < num_elems; ielem++) {
    //         const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
    //         myfile << nodes_per_elem;
    //         for (int inode = 0; inode < nodes_per_elem; inode++) {
    //             myfile << sp << elem_conn[local_perm[inode]];
    //         }
    //         myfile << "\n";
    //     }
    // }

    // cell type 9 is for CQUAD4 basically
    myfile << "CELL_TYPES " << num_mask_elems << "\n";
    int cell_type = (nodes_per_elem == 4) ? 9 :  // VTK_QUAD
                        (nodes_per_elem == 9) ? 28
                                              :  // VTK_BIQUADRATIC_QUAD
                        (nodes_per_elem == 16) ? 70
                                               :  // VTK_LAGRANGE_QUADRILATERAL
                        -1;
    for (int ielem = 0; ielem < num_mask_elems; ielem++) {
        myfile << cell_type << "\n";
    }

    // disp vector field now
    myfile << "POINT_DATA " << num_mask_nodes << "\n";
    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            myfile << soln[vpn * inode + offset] << sp;
            myfile << soln[vpn * inode + offset + 1] << sp;
            myfile << soln[vpn * inode + offset + 2] << "\n";
        }
    }

    scalarName = "rot";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            myfile << soln[vpn * inode + offset + 3] << sp;
            myfile << soln[vpn * inode + offset + 4] << sp;
            myfile << soln[vpn * inode + offset + 5] << "\n";
        }
    }

    // do later.. (time crunch rn)
    // if constexpr (Phys::hellingerReissner) {
    //     scalarName = "HRmem";
    //     myfile << "VECTORS " << scalarName << " double64\n";
    //     for (int inode = 0; inode < num_mask_nodes; inode++) {
    //         myfile << soln[vpn * inode + 0] << sp;
    //         myfile << soln[vpn * inode + 1] << sp;
    //         myfile << soln[vpn * inode + 2] << "\n";
    //     }

    //     scalarName = "HRtrv";
    //     myfile << "VECTORS " << scalarName << " double64\n";
    //     for (int inode = 0; inode < num_mask_nodes; inode++) {
    //         myfile << soln[vpn * inode + 3] << sp;
    //         myfile << soln[vpn * inode + 4] << sp;
    //         myfile << 0.0 << "\n";
    //     }
    // }

    // init visualization states
    int ndvs = assembler.get_num_dvs();
    DeviceVec<double> d_dvs(ndvs);
    DeviceVec<double> fail_index(num_elems);
    int nstresses = assembler.get_num_vis_stresses();
    DeviceVec<double> strains(nstresses), stresses(nstresses);

    // compute visualization states
    assembler.compute_visualization_states(d_dvs, fail_index, strains, stresses);

    // write thicknesses
    double *h_dvs = d_dvs.createHostVec().getPtr();
    int ndvs_per_comp = d_dvs.getSize() / assembler.get_num_components();
    myfile << "CELL_DATA " << num_mask_elems << "\n";
    scalarName = "thickness";
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        if (elem_mask[ielem]) {
            int comp_id = h_elem_comp[ielem];
            myfile << h_dvs[ndvs_per_comp * comp_id] << "\n";
        }
    }

    if (ndvs_per_comp > 1) {
        // like stiffened panel, writeout additional DVs also
        scalarName = "stiffHeight";
        myfile << "SCALARS " << scalarName << " double64 1\n";
        myfile << "LOOKUP_TABLE default\n";
        for (int ielem = 0; ielem < num_elems; ielem++) {
            if (elem_mask[ielem]) {
                int comp_id = h_elem_comp[ielem];
                myfile << h_dvs[ndvs_per_comp * comp_id + 1] << "\n";
            }
        }

        scalarName = "stiffThick";
        myfile << "SCALARS " << scalarName << " double64 1\n";
        myfile << "LOOKUP_TABLE default\n";
        for (int ielem = 0; ielem < num_elems; ielem++) {
            if (elem_mask[ielem]) {
                int comp_id = h_elem_comp[ielem];
                myfile << h_dvs[ndvs_per_comp * comp_id + 2] << "\n";
            }
        }

        scalarName = "stiffPitch";
        myfile << "SCALARS " << scalarName << " double64 1\n";
        myfile << "LOOKUP_TABLE default\n";
        for (int ielem = 0; ielem < num_elems; ielem++) {
            if (elem_mask[ielem]) {
                    int comp_id = h_elem_comp[ielem];
                    myfile << h_dvs[ndvs_per_comp * comp_id + 3] << "\n";
            }
        }

    }

    // write failure indexes
    double *h_fail_index = fail_index.createHostVec().getPtr();
    scalarName = "fail_index";
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elems; ielem++) {
        if (elem_mask[ielem]) {
            myfile << h_fail_index[ielem] << "\n";
        }
    }

    // write strains
    double *h_strains = strains.createHostVec().getPtr();
    myfile << "FIELD FieldData 4\n";
    scalarName = "midplane_strain";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_strains[6 * ielem + 0] << " " << h_strains[6 * ielem + 1] << " "
                << h_strains[6 * ielem + 2] << "\n";
        }
    }
    scalarName = "bending_strain";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_strains[6 * ielem + 3] << " " << h_strains[6 * ielem + 4] << " "
                << h_strains[6 * ielem + 5] << "\n";
        }
    }

    // write stresses
    double *h_stresses = stresses.createHostVec().getPtr();
    scalarName = "midplane_stress";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_stresses[6 * ielem + 0] << " " << h_stresses[6 * ielem + 1] << " "
                << h_stresses[6 * ielem + 2] << "\n";
        }
    }
    scalarName = "bending_stress";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elems; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_stresses[6 * ielem + 3] << " " << h_stresses[6 * ielem + 4] << " "
                << h_stresses[6 * ielem + 5] << "\n";
        }
    }

    myfile.close();
}
