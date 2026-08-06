// void build_ghost_maps() {
//     for (int dst = 0; dst < ngpus; dst++) {
//         int *dst_conn = A->getHostLocalElemConn(dst);

//         for (int e = 0; e < batch_size[dst]; e++) {
//             for (int ij = 0; ij < size4; ij++) {
//                 int i = ij % size2;
//                 int j = ij / size2;

//                 int dst_row_node = dst_conn[e * size2 + i];
//                 int dst_col_node = dst_conn[e * size2 + j];

//                 if (dst_row_node < 0 || dst_col_node < 0) continue;

//                 // Only patch ghost x ghost blocks on dst
//                 if (dst_row_node < part->owned_nnodes[dst]) continue;
//                 if (dst_col_node < part->owned_nnodes[dst]) continue;

//                 int glob_row = part->h_local_nodes[dst][dst_row_node];
//                 int glob_col = part->h_local_nodes[dst][dst_col_node];

//                 for (int src = 0; src < ngpus; src++) {
//                     if (src == dst) continue;

//                     int *src_rowp = A->getHostLocalRowp(src);
//                     int *src_cols = A->getHostLocalCols(src);

//                     int src_row_node = -1;
//                     int src_col_node = -1;

//                     for (int a = 0; a < part->local_nnodes[src]; a++) {
//                         int gnode = part->h_local_nodes[src][a];
//                         if (gnode == glob_row) src_row_node = a;
//                         if (gnode == glob_col) src_col_node = a;
//                     }

//                     if (src_row_node < 0 || src_col_node < 0) continue;

//                     int jp_found = -1;
//                     for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
//                          jp++) {
//                         if (src_cols[jp] == src_col_node) {
//                             jp_found = jp;
//                             break;
//                         }
//                     }

//                     if (jp_found >= 0) {
//                         int idx = pair_index(dst, src);
//                         h_ghost_asw_blocks[idx].push_back(e * size4 + ij);
//                         h_ghost_kmat_blocks[idx].push_back(jp_found);
//                         break;
//                     }
//                 }
//             }
//         }
//     }

//     for (int dst = 0; dst < ngpus; dst++) {
//         for (int src = 0; src < ngpus; src++) {
//             if (src == dst) continue;
//             int idx = pair_index(dst, src);
//             ghost_pair_nblocks[idx] = (int)h_ghost_asw_blocks[idx].size();
//         }
//     }
// }

// void build_ghost_maps(bool debug_print = true) {
//     int total_candidates = 0;
//     int total_already_local = 0;
//     int total_src_nodes_found = 0;
//     int total_src_block_found = 0;
//     int total_pushed = 0;

//     for (int dst = 0; dst < ngpus; dst++) {
//         int *dst_conn = A->getHostLocalElemConn(dst);

//         for (int e = 0; e < batch_size[dst]; e++) {
//             for (int ij = 0; ij < size4; ij++) {
//                 int i = ij % size2;
//                 int j = ij / size2;

//                 int dst_row_node = dst_conn[e * size2 + i];
//                 int dst_col_node = dst_conn[e * size2 + j];

//                 if (dst_row_node < 0 || dst_col_node < 0) continue;

//                 int glob_row = part->h_local_nodes[dst][dst_row_node];
//                 int glob_col = part->h_local_nodes[dst][dst_col_node];

//                 // bool dst_row_is_ghost = (part->h_node_gpu_ind[glob_row] != dst);
//                 // bool dst_col_is_ghost = (part->h_node_gpu_ind[glob_col] != dst);

//                 // not if it's owned, but if this node also appears on other proc is ghost
//                 here bool dst_row_is_ghost = part->h_is_local_ghost[dst][dst_row_node]; bool
//                 dst_col_is_ghost = part->h_is_local_ghost[dst][dst_col_node];

//                 // Only destination ghost x ghost blocks
//                 if (!dst_row_is_ghost || !dst_col_is_ghost) continue;

//                 total_candidates++;

//                 int dst_batch_block = e * size4 + ij;
//                 bool already_local = (h_block_inds[dst][dst_batch_block] >= 0);
//                 if (already_local) total_already_local++;

//                 bool found_src_nodes = false;
//                 bool found_src_block = false;

//                 for (int src = 0; src < ngpus; src++) {
//                     if (src == dst) continue;

//                     int *src_rowp = A->getHostLocalRowp(src);
//                     int *src_cols = A->getHostLocalCols(src);

//                     int src_row_node = -1;
//                     int src_col_node = -1;

//                     for (int a = 0; a < part->local_nnodes[src]; a++) {
//                         int gnode = part->h_local_nodes[src][a];

//                         if (gnode == glob_row) src_row_node = a;
//                         if (gnode == glob_col) src_col_node = a;

//                         if (src_row_node >= 0 && src_col_node >= 0) break;
//                     }

//                     if (src_row_node < 0 || src_col_node < 0) continue;
//                     found_src_nodes = true;

//                     int jp_found = -1;
//                     for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
//                          jp++) {
//                         if (src_cols[jp] == src_col_node) {
//                             jp_found = jp;
//                             break;
//                         }
//                     }

//                     if (jp_found < 0) continue;
//                     found_src_block = true;

//                     int idx = pair_index(dst, src);
//                     h_ghost_asw_blocks[idx].push_back(dst_batch_block);
//                     h_ghost_kmat_blocks[idx].push_back(jp_found);
//                     total_pushed++;

//                     break;
//                 }

//                 if (found_src_nodes) total_src_nodes_found++;
//                 if (found_src_block) total_src_block_found++;
//             }
//         }
//     }

//     for (int dst = 0; dst < ngpus; dst++) {
//         for (int src = 0; src < ngpus; src++) {
//             if (src == dst) continue;

//             int idx = pair_index(dst, src);
//             ghost_pair_nblocks[idx] = (int)h_ghost_asw_blocks[idx].size();

//             if (debug_print && ghost_pair_nblocks[idx] > 0) {
//                 printf("[ghost-map] dst %d <- src %d : %d blocks\n", dst, src,
//                        ghost_pair_nblocks[idx]);
//             }
//         }
//     }

//     if (debug_print) {
//         printf(
//             "[ghost-map] candidates=%d already_local=%d "
//             "src_nodes_found=%d src_block_found=%d pushed=%d\n",
//             total_candidates, total_already_local, total_src_nodes_found,
//             total_src_block_found, total_pushed);
//     }
// }