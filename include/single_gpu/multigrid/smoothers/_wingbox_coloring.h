#pragma once

template <class Assembler>
class WingboxMultiColoring {
    /* some utils for wingbox multicoloring (stabilizes colors in interior, junction + corners)
    aka uses geom hierarchy better */
  public:

    static void apply_coloring(Assembler &assembler, BsrData &bsr_data, 
        int &num_colors, int *&_color_rowp) {

        // V2 more stable one with one less color
        int *nodal_num_comps, *node_geom_ind;
        get_nodal_geom_indices(assembler, nodal_num_comps, node_geom_ind);
        bsr_data.multicolor_junction_reordering_v2(node_geom_ind, num_colors, _color_rowp);

        // free or delete this stuff..
        delete[] nodal_num_comps;
        delete[] node_geom_ind;
    }

    static void get_interior_node_flags(Assembler &assembler, bool *&is_interior_node) {
        // get for each node whether it's an interior node (not touching junction or multiple
        // components) or whether it's on a junction (connected to multiple comopnents)

        int _nnodes = assembler.get_num_nodes();
        is_interior_node = new bool[_nnodes];
        memset(is_interior_node, false, _nnodes * sizeof(bool));
        int nelems = assembler.get_num_elements();
        int *h_elem_comps = assembler.getElemComponents().createHostVec().getPtr();
        int *h_elem_conn = assembler.getConn().createHostVec().getPtr();

        // first compute the components (with repeats) each node touches (sparse data structure)
        int *node_comp_cts = new int[_nnodes];
        memset(node_comp_cts, 0, _nnodes * sizeof(int));
        for (int ielem = 0; ielem < nelems; ielem++) {
            // generalize this later to more than 4-node elems
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                node_comp_cts[inode]++;
            }
        }
        int *node_comp_ptr = new int[_nnodes + 1];
        node_comp_ptr[0] = 0;
        for (int inode = 0; inode < _nnodes; inode++) {
            node_comp_ptr[inode + 1] = node_comp_ptr[inode] + node_comp_cts[inode];
        }
        int _nnz = node_comp_ptr[_nnodes];
        int *insert_helper = new int[_nnodes];
        memcpy(insert_helper, node_comp_ptr, _nnodes * sizeof(int));
        int *node_comp_vals = new int[_nnz];
        for (int ielem = 0; ielem < nelems; ielem++) {
            int icomp = h_elem_comps[ielem];
            // generalize this later to more than 4-node elems
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                node_comp_vals[insert_helper[inode]++] = icomp;
            }
        }

        // check if all components on the node are the same (then it's interior), don't actually
        // need to know how many unique
        for (int inode = 0; inode < _nnodes; inode++) {
            bool interior = true;
            int first_comp_on_node = node_comp_vals[node_comp_ptr[inode]];
            for (int jp = node_comp_ptr[inode]; jp < node_comp_ptr[inode + 1]; jp++) {
                int icomp = node_comp_vals[jp];
                if (icomp != first_comp_on_node) {
                    interior = false;
                    break;
                }
            }
            is_interior_node[inode] = interior;
            // printf("\tnode %d interior = %d\n", inode, is_interior_node[inode]);
        }

        // TODO : free up temp arrays here..
    }

    static void get_nodal_geom_indices(Assembler &assembler, int *&nodal_num_comps, int *&node_geom_ind) {
        // get for each node whether it's an interior node (not touching junction or multiple
        // components) or whether it's on a junction (connected to multiple comopnents)

        int _nnodes = assembler.get_num_nodes();
        int nelems = assembler.get_num_elements();
        int *h_elem_comps = assembler.getElemComponents().createHostVec().getPtr();
        int *h_elem_conn = assembler.getConn().createHostVec().getPtr();

        // first compute the components (with repeats) each node touches (sparse data structure)
        int *node_comp_cts = new int[_nnodes];
        memset(node_comp_cts, 0, _nnodes * sizeof(int));
        for (int ielem = 0; ielem < nelems; ielem++) {
            // generalize this later to more than 4-node elems
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                node_comp_cts[inode]++;
            }
        }
        int *node_comp_ptr = new int[_nnodes + 1];
        node_comp_ptr[0] = 0;
        for (int inode = 0; inode < _nnodes; inode++) {
            node_comp_ptr[inode + 1] = node_comp_ptr[inode] + node_comp_cts[inode];
        }
        int _nnz = node_comp_ptr[_nnodes];
        int *insert_helper = new int[_nnodes];
        memcpy(insert_helper, node_comp_ptr, _nnodes * sizeof(int));
        int *node_comp_vals = new int[_nnz];
        for (int ielem = 0; ielem < nelems; ielem++) {
            int icomp = h_elem_comps[ielem];
            // generalize this later to more than 4-node elems
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                node_comp_vals[insert_helper[inode]++] = icomp;
            }
        }

        // get number of unique components on each node
        nodal_num_comps = new int[_nnodes];
        for (int inode = 0; inode < _nnodes; inode++) {
            int start = node_comp_ptr[inode], end = node_comp_ptr[inode + 1];
            std::set<int> unique(&node_comp_vals[start], &node_comp_vals[end]);
            nodal_num_comps[inode] = unique.size();
        }

        // now for node hierarchies..
        // value 0 - interior/face nodes which belong to only 1 component
        // value 1 - edge nodes belong to two or more components
        // value 2 - corner nodes belong on an edge and connect two or more edges
        //     and belong to more components than their neighboring edge nodes in same element (if ties then there is no corner)
        node_geom_ind = new int[_nnodes];
        memset(node_geom_ind, 0, _nnodes * sizeof(int));
        // first label some nodes as edge nodes from nodal_num_comps (if num comps on the node > 1)
        for (int inode = 0; inode < _nnodes; inode++) {
            node_geom_ind[inode] = nodal_num_comps[inode] > 1 ? 1 : 0;
        }

        for (int ielem = 0; ielem < nelems; ielem++) {
            // check if any nodes in element are edge nodes
            bool any_edge_nodes = false;
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                any_edge_nodes += node_geom_ind[inode] == 1;
            }

            if (!any_edge_nodes) continue; // go to next element, no edge nodes here

            // then if edge nodes, check for corner nodes
            std::set<int> edge_node_comp_nums;
            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                // in case previous corner node was flagged, include it too so we don't mess up and confuse another edge node as corner node
                if (node_geom_ind[inode] >= 1) edge_node_comp_nums.insert(nodal_num_comps[inode]);
            }
            // if more than one unique number of edge comp nums (there will be a corner node, which is the node with max # comps it belongs to)
            int n_edge_set = edge_node_comp_nums.size();
            if (n_edge_set == 1) continue;
            // if didn't continue, then we have a corner node, let's find it
            int max_ncomp = *edge_node_comp_nums.rbegin();
            // now let's get the inode for the max value and mark it as a corner node

            for (int lnode = 0; lnode < 4; lnode++) {
                int inode = h_elem_conn[4 * ielem + lnode];
                if (nodal_num_comps[inode] == max_ncomp) {
                    node_geom_ind[inode] = 2; // corner node!
                }
            }
        } // end of element corner node labels

        // TODO : free up temp arrays here..
    }

};

