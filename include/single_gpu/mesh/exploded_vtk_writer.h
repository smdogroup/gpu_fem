#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "assembler.h"

enum EXPLODED { INT_STRUCT, UPPER_SKIN, LOWER_SKIN };

template <class Assembler, class Vec, EXPLODED exploded>
void explodedPrintToVTK(Assembler assembler, Vec &soln, const std::string filename,
                        int nsubdomain = -1, int *elem_sd_ind = nullptr) {
    /* write an exploded view of a wing to VTK (assuming only upper and lower skin fully offset Z>0
     * or Z<0) trick (such as AOB wing) */

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

    // first identify which components belong to this one of 3 exploded meshes (0,1,2) for (int
    // struct, upper skin, lower skin)
    bool *upper_skin_comp_mask = new bool[num_components];
    bool *lower_skin_comp_mask = new bool[num_components];
    memset(upper_skin_comp_mask, true,
           num_components * sizeof(bool));  // assume true until proven false
    memset(lower_skin_comp_mask, true,
           num_components * sizeof(bool));  // assume true until proven false
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
            elem_mask[ielem] = true;  // it's in an allowed component
            num_mask_elems++;
            int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];
            for (int i = 0; i < nodes_per_elem; i++) {
                int inode = elem_nodes[i];
                nodes_mask[inode] = true;  // it's in an allowed component
            }
        }
    }
    // to avoid non-uniqueness, loop back through nodes mask after construction
    // compute also a map from prev total [0, num_nodes) list to [0, num_mask_nodes) list for elem
    // connectivity
    int *red_node_map = new int[num_nodes];  // old node to reduced node (as long as reduced node in
                                             // the exploded type)
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
                    int red_node =
                        red_node_map[inode];  // get reduced node since less total # nodes
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

template <class Assembler, class Vec, EXPLODED exploded>
void explodedSubdomainsPrintToVTK(Assembler assembler, Vec &soln, const std::string filename,
                                  int num_subdomains, const int *elem_sd_ind) {
    int num_nodes = assembler.get_num_nodes();
    int num_elements = assembler.get_num_elements();

    auto d_xpts = assembler.getXpts();
    auto h_xpts = d_xpts.createHostVec();
    double *xpts_ptr = h_xpts.getPtr();

    using Phys = typename Assembler::Phys;
    int vpn = Phys::vars_per_node;
    int std_vpn = Phys::std_vpn;
    int offset = (vpn == std_vpn) ? 0 : 5;

    int nodes_per_elem = Assembler::vars_nodes_per_elem;

    auto d_vars_conn = assembler.getConn();
    auto h_vars_conn = d_vars_conn.createHostVec();
    int *conn_ptr = h_vars_conn.getPtr();

    // ==========================================================
    // classify each subdomain as upper / lower / internal
    // based on ALL nodes touched by that subdomain
    // ==========================================================
    bool *sd_is_upper = new bool[num_subdomains];
    bool *sd_is_lower = new bool[num_subdomains];
    bool *sd_node_seen = new bool[num_subdomains * num_nodes];

    std::memset(sd_is_upper, true, num_subdomains * sizeof(bool));
    std::memset(sd_is_lower, true, num_subdomains * sizeof(bool));
    std::memset(sd_node_seen, false, num_subdomains * num_nodes * sizeof(bool));

    for (int ielem = 0; ielem < num_elements; ielem++) {
        int isd = elem_sd_ind[ielem];
        int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];

        for (int i = 0; i < nodes_per_elem; i++) {
            int inode = elem_nodes[i];
            if (sd_node_seen[isd * num_nodes + inode]) {
                continue;
            }
            sd_node_seen[isd * num_nodes + inode] = true;

            double z = xpts_ptr[3 * inode + 2];

            if (z <= 0.0) {
                sd_is_upper[isd] = false;
            }
            if (z >= 0.0) {
                sd_is_lower[isd] = false;
            }
        }
    }

    bool *sd_mask = new bool[num_subdomains];
    for (int isd = 0; isd < num_subdomains; isd++) {
        if constexpr (exploded == UPPER_SKIN) {
            sd_mask[isd] = sd_is_upper[isd];
        } else if constexpr (exploded == LOWER_SKIN) {
            sd_mask[isd] = sd_is_lower[isd];
        } else if constexpr (exploded == INT_STRUCT) {
            sd_mask[isd] = !sd_is_upper[isd] && !sd_is_lower[isd];
        }
    }

    // ==========================================================
    // filter elements/nodes by SUBDOMAIN classification
    // ==========================================================
    bool *elem_mask = new bool[num_elements];
    bool *nodes_mask = new bool[num_nodes];
    std::memset(elem_mask, false, num_elements * sizeof(bool));
    std::memset(nodes_mask, false, num_nodes * sizeof(bool));

    int num_mask_nodes = 0, num_mask_elems = 0;

    for (int ielem = 0; ielem < num_elements; ielem++) {
        int isd = elem_sd_ind[ielem];
        if (!sd_mask[isd]) {
            continue;
        }

        elem_mask[ielem] = true;
        num_mask_elems++;

        int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];
        for (int i = 0; i < nodes_per_elem; i++) {
            nodes_mask[elem_nodes[i]] = true;
        }
    }

    int *red_node_map = new int[num_nodes];
    std::memset(red_node_map, -1, num_nodes * sizeof(int));
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            red_node_map[inode] = num_mask_nodes;
            num_mask_nodes++;
        }
    }

    // ==========================================================
    // global centroid over retained nodes
    // ==========================================================
    double global_centroid[3] = {0.0, 0.0, 0.0};
    int global_count = 0;
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            global_centroid[0] += xpts_ptr[3 * inode + 0];
            global_centroid[1] += xpts_ptr[3 * inode + 1];
            global_centroid[2] += xpts_ptr[3 * inode + 2];
            global_count++;
        }
    }
    if (global_count > 0) {
        global_centroid[0] /= global_count;
        global_centroid[1] /= global_count;
        global_centroid[2] /= global_count;
    }

    // ==========================================================
    // centroid of each retained subdomain
    // ==========================================================
    double *sd_centroid = new double[3 * num_subdomains];
    int *sd_node_count = new int[num_subdomains];
    bool *sd_node_seen2 = new bool[num_subdomains * num_nodes];

    std::memset(sd_centroid, 0, 3 * num_subdomains * sizeof(double));
    std::memset(sd_node_count, 0, num_subdomains * sizeof(int));
    std::memset(sd_node_seen2, false, num_subdomains * num_nodes * sizeof(bool));

    for (int ielem = 0; ielem < num_elements; ielem++) {
        if (!elem_mask[ielem]) {
            continue;
        }

        int isd = elem_sd_ind[ielem];
        int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];

        for (int i = 0; i < nodes_per_elem; i++) {
            int inode = elem_nodes[i];
            if (sd_node_seen2[isd * num_nodes + inode]) {
                continue;
            }
            sd_node_seen2[isd * num_nodes + inode] = true;

            sd_centroid[3 * isd + 0] += xpts_ptr[3 * inode + 0];
            sd_centroid[3 * isd + 1] += xpts_ptr[3 * inode + 1];
            sd_centroid[3 * isd + 2] += xpts_ptr[3 * inode + 2];
            sd_node_count[isd]++;
        }
    }

    for (int isd = 0; isd < num_subdomains; isd++) {
        if (sd_node_count[isd] > 0) {
            sd_centroid[3 * isd + 0] /= sd_node_count[isd];
            sd_centroid[3 * isd + 1] /= sd_node_count[isd];
            sd_centroid[3 * isd + 2] /= sd_node_count[isd];
        }
    }

    // ==========================================================
    // explode scale from retained bounding box
    // ==========================================================
    double xmin = 1e20, xmax = -1e20;
    double ymin = 1e20, ymax = -1e20;
    double zmin = 1e20, zmax = -1e20;
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            double x = xpts_ptr[3 * inode + 0];
            double y = xpts_ptr[3 * inode + 1];
            double z = xpts_ptr[3 * inode + 2];
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
    }
    double diag = std::sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin) +
                            (zmax - zmin) * (zmax - zmin));
    double explode_scale = 0.15 * diag;

    // representative node -> subdomain map over retained elems
    int *node_sd = new int[num_nodes];
    std::memset(node_sd, -1, num_nodes * sizeof(int));
    for (int ielem = 0; ielem < num_elements; ielem++) {
        if (!elem_mask[ielem]) {
            continue;
        }

        int isd = elem_sd_ind[ielem];
        int *elem_nodes = &conn_ptr[nodes_per_elem * ielem];
        for (int i = 0; i < nodes_per_elem; i++) {
            int inode = elem_nodes[i];
            if (node_sd[inode] < 0) {
                node_sd[inode] = isd;
            }
        }
    }

    // ==========================================================
    // write vtk
    // ==========================================================
    using namespace std;
    string sp = " ";
    string dataType = "double64";

    ofstream myfile;
    myfile.open(filename);
    myfile << "# vtk DataFile Version 2.0\n";
    myfile << "TACS GPU shell writer exploded by subdomain\n";
    myfile << "ASCII\n";
    myfile << "DATASET UNSTRUCTURED_GRID\n";

    myfile << "POINTS " << num_mask_nodes << sp << dataType << "\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (!nodes_mask[inode]) {
            continue;
        }

        double x = xpts_ptr[3 * inode + 0];
        double y = xpts_ptr[3 * inode + 1];
        double z = xpts_ptr[3 * inode + 2];

        int isd = node_sd[inode];
        double dx = 0.0, dy = 0.0, dz = 0.0;

        // if (isd >= 0 && sd_node_count[isd] > 0) {
        //     dx = sd_centroid[3 * isd + 0] - global_centroid[0];
        //     dy = sd_centroid[3 * isd + 1] - global_centroid[1];
        //     dz = sd_centroid[3 * isd + 2] - global_centroid[2];

        //     double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        //     if (norm > 1e-30) {
        //         dx *= explode_scale / norm;
        //         dy *= explode_scale / norm;
        //         dz *= explode_scale / norm;
        //     } else {
        //         dx = dy = dz = 0.0;
        //     }
        // }
        dx = dy = dz = 0.0;  // don't shift subdomains at all..

        myfile << x + dx << sp << y + dy << sp << z + dz << "\n";
    }

    int num_elem_nodes = num_mask_elems * (nodes_per_elem + 1);
    myfile << "CELLS " << num_mask_elems << " " << num_elem_nodes << "\n";

    if (nodes_per_elem == 4) {
        const int32_t local_perm[4] = {0, 1, 3, 2};
        for (int ielem = 0; ielem < num_elements; ielem++) {
            if (!elem_mask[ielem]) {
                continue;
            }

            const int *elem_conn = &conn_ptr[nodes_per_elem * ielem];
            myfile << nodes_per_elem;
            for (int i = 0; i < nodes_per_elem; i++) {
                int inode = elem_conn[local_perm[i]];
                myfile << sp << red_node_map[inode];
            }
            myfile << "\n";
        }
    }

    myfile << "CELL_TYPES " << num_mask_elems << "\n";
    int cell_type = (nodes_per_elem == 4)    ? 9
                    : (nodes_per_elem == 9)  ? 28
                    : (nodes_per_elem == 16) ? 70
                                             : -1;
    for (int ielem = 0; ielem < num_mask_elems; ielem++) {
        myfile << cell_type << "\n";
    }

    myfile << "POINT_DATA " << num_mask_nodes << "\n";

    string scalarName = "disp";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            myfile << soln[vpn * inode + offset] << sp << soln[vpn * inode + offset + 1] << sp
                   << soln[vpn * inode + offset + 2] << "\n";
        }
    }

    scalarName = "rot";
    myfile << "VECTORS " << scalarName << " double64\n";
    for (int inode = 0; inode < num_nodes; inode++) {
        if (nodes_mask[inode]) {
            myfile << soln[vpn * inode + offset + 3] << sp << soln[vpn * inode + offset + 4] << sp
                   << soln[vpn * inode + offset + 5] << "\n";
        }
    }

    int nstresses = assembler.get_num_vis_stresses();
    DeviceVec<double> fail_index(num_elements);
    DeviceVec<double> strains(nstresses), stresses(nstresses);
    DeviceVec<double> d_dvs(assembler.get_num_dvs());
    assembler.compute_visualization_states(d_dvs, fail_index, strains, stresses);

    myfile << "CELL_DATA " << num_mask_elems << "\n";

    scalarName = "subdomains";
    myfile << "SCALARS " << scalarName << " int 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elements; ielem++) {
        if (elem_mask[ielem]) {
            myfile << elem_sd_ind[ielem] << "\n";
        }
    }

    double *h_fail_index = fail_index.createHostVec().getPtr();
    scalarName = "fail_index";
    myfile << "SCALARS " << scalarName << " double64 1\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int ielem = 0; ielem < num_elements; ielem++) {
        if (elem_mask[ielem]) {
            myfile << h_fail_index[ielem] << "\n";
        }
    }

    double *h_strains = strains.createHostVec().getPtr();
    myfile << "FIELD FieldData 4\n";

    scalarName = "midplane_strain";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elements; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_strains[6 * ielem + 0] << " " << h_strains[6 * ielem + 1] << " "
                   << h_strains[6 * ielem + 2] << "\n";
        }
    }

    scalarName = "bending_strain";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elements; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_strains[6 * ielem + 3] << " " << h_strains[6 * ielem + 4] << " "
                   << h_strains[6 * ielem + 5] << "\n";
        }
    }

    double *h_stresses = stresses.createHostVec().getPtr();
    scalarName = "midplane_stress";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elements; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_stresses[6 * ielem + 0] << " " << h_stresses[6 * ielem + 1] << " "
                   << h_stresses[6 * ielem + 2] << "\n";
        }
    }

    scalarName = "bending_stress";
    myfile << scalarName << " 3 " << num_mask_elems << " double\n";
    for (int ielem = 0; ielem < num_elements; ++ielem) {
        if (elem_mask[ielem]) {
            myfile << h_stresses[6 * ielem + 3] << " " << h_stresses[6 * ielem + 4] << " "
                   << h_stresses[6 * ielem + 5] << "\n";
        }
    }

    myfile.close();

    delete[] sd_is_upper;
    delete[] sd_is_lower;
    delete[] sd_node_seen;
    delete[] sd_mask;
    delete[] elem_mask;
    delete[] nodes_mask;
    delete[] red_node_map;
    delete[] sd_centroid;
    delete[] sd_node_count;
    delete[] sd_node_seen2;
    delete[] node_sd;
}