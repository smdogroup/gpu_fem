# you have to declare methods here for numba to actually compile them in relative import to other file
__all__ = ["_compute_CSR_pattern_numba",
           "_compute_elem_to_CSR_assembly_map_numba",
           "_compute_colbc_row_counts_numba",
           "_compute_bc_cols_csr_map_numba",
           "_helper_numba_assembly_serial",
           "_get_elem_fails_numba",
           "_get_elem_fails_SVsens_numba",
           "_compute_adj_dRdx_numba"]

import numpy as np
from numba import njit #, prange

"""
* these numba methods drastically speedup the FEA assembly and other elem-wise computations for struct opt of the plate.
I don't want to go too crazy with this, but do need to numba every method otherwise can't try high number of elements and DVs (code would be too slow).
* numba can also do multithreading with parallel=True, but that is actually slower than parallel=False right now, so I don't mess with it (on my linux machine)
"""

@njit(parallel=False)
def _compute_CSR_pattern_numba(num_nodes, num_elements, elem_conn):
    N = 3 * num_nodes

    # Estimate worst-case number of non-zeros: fully connected nodes
    max_nnz = N * N  # Very conservative upper bound

    rowp = np.zeros(N + 1, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    rows_scipy = np.empty(max_nnz, dtype=np.int32)

    nnz = 0

    for row in range(N):
        inode = row // 3

        # Build attached nodes using a boolean mask (simulate a set)
        attached_nodes_mask = np.zeros(num_nodes, dtype=np.uint8)

        for e in range(num_elements):
            elem = elem_conn[e]
            for j in range(4):
                if elem[j] == inode:
                    for k in range(4):
                        attached_nodes_mask[elem[k]] = 1

        # Extract attached nodes from mask
        count_attached = 0
        for node_id in range(num_nodes):
            if attached_nodes_mask[node_id]:
                for idof in range(3):
                    cols[nnz] = 3 * node_id + idof
                    rows_scipy[nnz] = row
                    nnz += 1
                count_attached += 1

        rowp[row + 1] = nnz

    # Trim arrays to actual size
    cols = cols[:nnz]
    rows_scipy = rows_scipy[:nnz]

    return nnz, rowp, cols, rows_scipy

@njit(parallel=False) # parallel=True is slower (bc multithreading issue on my machine? so stick with False)
def _compute_elem_to_CSR_assembly_map_numba(
    num_elements, dof_per_elem, dof_conn, rowp, cols):
    """helper function for speeding up the elem to CSR assembly map comp (very slow for high num elements in regular serial case)"""
    dof_per_elem2 = dof_per_elem ** 2
    elem_to_csr_map = np.zeros((num_elements, dof_per_elem2), dtype=np.int32)

    # prange (parallel range) could also be used, but not working well bc multithreading issue..
    for ielem in range(num_elements): 
        local_conn = dof_conn[ielem]

        for idof_in_elem in range(dof_per_elem2):
            elem_row = idof_in_elem // dof_per_elem
            elem_col = idof_in_elem % dof_per_elem
            global_row = local_conn[elem_row]
            global_col = local_conn[elem_col]

            for csr_ind in range(rowp[global_row], rowp[global_row + 1]):
                if cols[csr_ind] == global_col:
                    elem_to_csr_map[ielem, idof_in_elem] = csr_ind
                    break

    return elem_to_csr_map

@njit(parallel=False)
def _compute_colbc_row_counts_numba(bcs, num_dof, rowp, cols):
    """helper numba function for bc col map, pre-computes how many rows per bc col"""
    nbcs = bcs.shape[0]
    bc_counts = np.zeros(nbcs, dtype=np.int32)

    for i in range(nbcs):
        bc = bcs[i]
        for glob_row in range(num_dof):
            row_start = rowp[glob_row]
            row_end = rowp[glob_row + 1]
            for j in range(row_start, row_end):
                if cols[j] == bc:
                    bc_counts[i] += 1

    return bc_counts

@njit(parallel=False)
def _compute_bc_cols_csr_map_numba(bcs, num_dof, rowp, cols, bc_counts):
    """helper numba function for bc col map"""
    total_entries = np.sum(bc_counts)
    bc_row_indices = np.zeros(total_entries, dtype=np.int32)
    nbcs = bcs.shape[0]
    bc_offsets = np.zeros(nbcs + 1, dtype=np.int32)

    # Build the offsets array (like CSR row pointers)
    bc_offsets[1:] = np.cumsum(bc_counts)

    # Fill the index array
    current_positions = bc_offsets[:-1].copy()

    for i in range(nbcs):
        bc = bcs[i]
        for glob_row in range(num_dof):
            row_start = rowp[glob_row]
            row_end = rowp[glob_row + 1]
            for j in range(row_start, row_end):
                if cols[j] == bc:
                    insert_pos = current_positions[i]
                    bc_row_indices[insert_pos] = glob_row
                    current_positions[i] += 1

    return bc_offsets, bc_row_indices

@njit(parallel=False) # parallel = True didn't work (faster with parallel=False) bc it throws error every time with parallel=True..
def _helper_numba_assembly_serial(num_elements, elem_to_csr_map, dof_conn, 
                            csr_data, kelem_local, kelem_scales,
                            global_force, force_local, felem_scales):
    # for ielem in prange(num_elements): # can also do reg range
    for ielem in range(num_elements):
        loc_elem_to_csr_map = elem_to_csr_map[ielem]
        csr_data[loc_elem_to_csr_map] += kelem_local * kelem_scales[ielem]

        local_conn = dof_conn[ielem]
        global_force[local_conn] += force_local * felem_scales[ielem]

@njit(parallel=False)
def _get_elem_fails_numba(num_elements, Dvec, dof_conn, u, hess_basis, nu, helem_vec, ys):
    elem_fails = np.zeros((num_elements,))
    for ielem in range(num_elements):
        D = Dvec[ielem]
        local_conn = dof_conn[ielem]
        local_disp = u[local_conn]
        dx2 = dy2 = dxy = 0
        for ibasis in range(12):
            dx2 += hess_basis[ibasis,0] * local_disp[ibasis]
            dy2 += hess_basis[ibasis,1] * local_disp[ibasis]
            dxy += hess_basis[ibasis,2] * local_disp[ibasis]
        # compute bending moments
        Mxx = -D * (dx2 + nu * dy2)
        Myy = -D * (dy2 + nu * dx2)
        Mxy = -D * (1-nu) * dxy

        I = helem_vec[ielem]**3/12.0
        z = helem_vec[ielem]/2

        # compute the 2d stresses
        sxx = Mxx * z / I
        syy = Myy * z / I
        sxy = Mxy * z / I

        # now compute von mises stress
        vm_stress = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3 * sxy**2)
        nd_stress = vm_stress / ys
        elem_fails[ielem] = nd_stress
    return elem_fails

@njit(parallel=False)
def _get_elem_fails_SVsens_numba(num_dof, num_elements, Dvec, dof_conn, u, hess_basis, nu, helem_vec, ys, delem_fails):
    du_global = np.zeros((num_dof,))
    for ielem in range(num_elements):
        D = Dvec[ielem]
        local_conn = dof_conn[ielem]
        local_disp = u[local_conn]
        dx2 = dy2 = dxy = 0
        for ibasis in range(12):
            dx2 += hess_basis[ibasis,0] * local_disp[ibasis]
            dy2 += hess_basis[ibasis,1] * local_disp[ibasis]
            dxy += hess_basis[ibasis,2] * local_disp[ibasis]
        
        # compute bending moments
        Mxx = -D * (dx2 + nu * dy2)
        Myy = -D * (dy2 + nu * dx2)
        Mxy = -D * (1-nu) * dxy

        I = helem_vec[ielem]**3/12.0
        z = helem_vec[ielem]/2

        # compute the 2d stresses
        sxx = Mxx * z / I
        syy = Myy * z / I
        sxy = Mxy * z / I

        # now compute von mises stress
        vm_stress = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3 * sxy**2)
        # nd_stress = vm_stress / self.ys

        # now backprop the stress derivatives here
        dvm_dsxx = (2 * sxx - syy) / 2.0 / vm_stress / ys
        dvm_dsyy = (2 * syy - sxx) / 2.0 / vm_stress / ys
        dvm_dsxy = (6 * sxy) / 2.0 / vm_stress / ys

        # backprop through the bending moments
        dvm_dMxx = dvm_dsxx * z / I
        dvm_dMyy = dvm_dsyy * z / I
        dvm_dMxy = dvm_dsxy * z / I

        # backprop through to the hessian derivatives
        dvm_dx2 = -D * (dvm_dMxx + nu * dvm_dMyy)
        dvm_dy2 = -D * (dvm_dMyy + nu * dvm_dMxx)
        dvm_dxy = -D * (1-nu) * dvm_dMxy

        du_vec = dvm_dx2 * hess_basis[:,0] + \
                dvm_dy2 * hess_basis[:,1] + \
                dvm_dxy * hess_basis[:,2]
        du_global[local_conn] += du_vec * delem_fails[ielem]
    return du_global

@njit(parallel=False)
def _compute_adj_dRdx_numba(num_elements, Kelem_nom, dof_conn, psi, u, Dvec, helem_vec, bcs):
    dthick_elem = np.zeros((num_elements,))

    # compute the gradient at the element level so more efficient
    for ielem in range(num_elements):
        
        # get local element vectors or data
        local_conn = dof_conn[ielem]
        psi_local = psi[local_conn]
        u_local = u[local_conn]
        D = Dvec[ielem]

        # compute local Kelem thickness derivative
        dKelem = Kelem_nom * D * 3 / helem_vec[ielem]

        # apply local bcs to dKelem matrix
        # start_node = local_node_conn[0]
        for local_dof,global_dof in enumerate(local_conn):
            if global_dof in bcs:
                dKelem[local_dof,:] = 0.0
                dKelem[:,local_dof] = 0.0

        dthick_elem[ielem] = np.dot(psi_local, np.dot(dKelem, u_local))
    
    return dthick_elem