#pragma once

template <typename T, class Basis, class Assembler>
void addSkinLoadsToWing(Assembler &assembler, T *&wing_loads, T force) {
    // just apply pressure load evenly on the lower skin compoenents
    // assume wing loads pointer is already defined / initialized (on host) and in VIS order not SOLVE order

    /* 1) get bool of which components are lower skin components (based on centroid instead of name, more convenient) */
    // ------------------------------------------------------------------------
    int num_components = assembler.get_num_components();
    int num_elements = assembler.get_num_elements();
    int num_nodes = assembler.get_num_nodes();
    int num_vars = assembler.get_num_vars();

    wing_loads = new T[num_vars];
    memset(wing_loads, 0, num_vars * sizeof(T));

    DeviceVec<int> elem_components(num_elements);
    assembler.get_elem_components(elem_components);
    int *h_elem_components = elem_components.createHostVec().getPtr();
    int *h_elem_conn = assembler.getConn().createHostVec().getPtr();
    T *h_xpts = assembler.getXpts().createHostVec().getPtr();

    // loop over each component and the elements in that component, if all nodes have y < 0 then it's lower skin component
    // this is easy trick since ribs, spars have some y<0 and some y>0 and upper skin all y > 0 coords
    bool *is_comp_lower_skin = new bool[num_components];
    for (int icomp = 0; icomp < num_components; icomp++) {
        is_comp_lower_skin[icomp] = true; // assume true until proven otherwise (innocent until proven guilty, then exit loop)
    }
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        int *elem_nodes = &h_elem_conn[Basis::num_nodes * ielem];

        for (int i = 0; i < Basis::num_nodes; i++) {
            int inode = elem_nodes[i];
            T* xpt = &h_xpts[3 * inode];
            if (xpt[2] > 0.0) {
                is_comp_lower_skin[icomp] = false;
                break;  // no need to inspect the other nodes for this component
            }
        }
    }


    // then check how many components and how many of them are lower skin
    int num_lower_skin_comp = 0;
    for (int icomp = 0; icomp < num_components; icomp++) {
        num_lower_skin_comp += is_comp_lower_skin[icomp];
    }
    // printf("%d # components and %d # comps in lower skin\n", num_components, num_lower_skin_comp);

    // get the number of elements in the lower skin (instead of num nodes, since don't want to worry about unique node)
    int num_lower_skin_elems = 0;
    int *num_comp_elems = new int[num_components];
    memset(num_comp_elems, 0, num_components * sizeof(int));
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        if (is_comp_lower_skin[icomp]) {
            num_lower_skin_elems++;
            num_comp_elems[icomp]++;
        }
    }
    // DEBUG
    // printf("add skin loads to wing with %d lower skin elems\n", num_lower_skin_elems);
    // for (int icomp = 0; icomp < num_components; icomp++) {
    //     if (is_comp_lower_skin[icomp]) {
    //         printf("comp %d in lower skin with %d elems\n", icomp, num_comp_elems[icomp]);
    //     }
    // }

    // compute the load scales based on # elems in each component (so rough even pressure distributions)
    T *comp_load_scales = new T[num_components];
    memset(comp_load_scales, 0, num_components * sizeof(T));
    for (int icomp = 0; icomp < num_components; icomp++) {
        if (is_comp_lower_skin[icomp]) {
            comp_load_scales[icomp] = force / num_lower_skin_comp / Basis::num_nodes / num_comp_elems[icomp];
        }
    }

    /* 2) apply uniform pressure load to all lower skin elems, evenly distributed among their nodes */
    int block_dim = assembler.getBsrData().block_dim;
    for (int ielem = 0; ielem < num_elements; ielem++) {
        int icomp = h_elem_components[ielem];
        if (is_comp_lower_skin[icomp]) {
            int *elem_nodes = &h_elem_conn[Basis::num_nodes * ielem];
            for (int i = 0; i < Basis::num_nodes; i++) {
                int inode = elem_nodes[i];
                wing_loads[block_dim * inode + 2] += comp_load_scales[icomp];
            }
        }
    }

    // DEBUG
    // writeout the element and nodes in each component
    // for (int ielem = 0; ielem < num_elements; ielem++) {
    //     int icomp = h_elem_components[ielem];
    //     if (is_comp_lower_skin[icomp]) {
    //         int *elem_nodes = &h_elem_conn[Basis::num_nodes * ielem];
    //         printf("ielem %d in icomp %d with nodes: ");
    //         printVec<int>(4, elem_nodes);
    //     }
    // }
    // auto h_loads = HostVec<T>(num_vars, wing_loads);
    // printToVTK<Assembler,HostVec<T>>(assembler, h_loads, "out/wing_loads_in_method.vtk");

    // DONE
    // free up new pointers
    // h_elem_components.free();
    // h_elem_conn.free();
    // h_xpts.free();
}