#pragma once

template <typename T, class Basis, class Assembler>
void addSkinLoadsToWing(Assembler &assembler, T *wing_loads, T pressure) {
    // just apply pressure load evenly on the lower skin compoenents
    // assume wing loads pointer is already defined / initialized (on host) and in VIS order not SOLVE order

    /* 1) get bool of which components are lower skin components (based on centroid instead of name, more convenient) */
    // ------------------------------------------------------------------------
    auto h_elem_components = assembler.getElemComponents().createHostVec();
    auto h_elem_conn = assembler.getConn().createHostVec();
    auto h_xpts = assembler.getXpts().createHostVec();
    
    int num_components = assembler.get_num_components();
    int num_elements = assembler.get_num_elements();
    int num_nodes = assembler.get_num_nodes();

    // loop over each component and the elements in that component, if all nodes have y < 0 then it's lower skin component
    // this is easy trick since ribs, spars have some y<0 and some y>0 and upper skin all y > 0 coords
    bool *is_comp_lower_skin = new bool[num_components];
    for (int icomp = 0; icomp < num_components; icomp++) {
        is_comp_lower_skin[icomp] = true; // assume true until proven otherwise (innocent until proven guilty, then exit loop)
    }
    // loop through all elements now, computing which comp it is
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        // get nodes for it (only need one node, not all four or something)
        int inode = h_elem_conn[Basis::num_nodes * ielem];
        T *xpt = &h_xpts[3 * inode];
        if (xpt[1] > 0.0) {
            is_comp_lower_skin[icomp] = false; // it's not a lower skin panel anymore
        }
    }

    // get the number of elements in the lower skin (instead of num nodes, since don't want to worry about unique node)
    int num_lower_skin_elems = 0;
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        if (is_comp_lower_skin[icomp]) {
            num_lower_skin_elems++;
        }
    }

    /* 2) apply uniform pressure load to all lower skin elems, evenly distributed among their nodes */
    T nodal_load_scale = 1.0 / Basis::num_nodes / num_lower_skin_elems;
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        if (is_comp_lower_skin[icomp]) {
            int *elem_nodes = &h_elem_conn[Basis::num_nodes * ielem];
            for (int i = 0; i < Basis::num_nodes; i++) {
                int inode = elem_nodes[i];
                wing_loads[3 * inode + 2] += nodal_load_scale;
            }
        }
    }

    // DONE
    // free up new pointers
    // h_elem_components.free();
    // h_elem_conn.free();
    // h_xpts.free();
}