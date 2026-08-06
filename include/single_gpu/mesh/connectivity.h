#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include "basic_connectivity.h"

enum ElementEntity { VOLUME, FACE, EDGE, VERTEX };

class ElementOrdering {
 public:
  int get_num_dof(ElementType etype) const {}
  int get_num_dof(ElementType etype, ElementEntity entity) const {}
  void set_dof(ElementType etype, ElementEntity entity, int dof[]) const {}
  void copy_dof(ElementType src_type, ElementEntity entity, const int src_dof[],
                ElementType dest_type, int dest_dof[]) const {}
};

class Connectivity3D {
 public:
  Connectivity3D(BasicConnectivity3D &conn, ElementOrdering &elem_conv)
      : tet_dof_per_elem(props.get_num_dof(TETRAHEDRON)),
        hex_dof_per_elem(props.get_num_dof(HEXAHEDRON)),
        pyrd_dof_per_elem(props.get_num_dof(PYRAMID)),
        wedge_dof_per_elem(props.get_num_dof(WEDGE)) {
    num_local_tets = conn.get_tets();
    num_local_hexs = conn.get_hexs();
    num_local_pyrds = conn.get_pyrds();
    num_local_wedges = conn.get_wedges();

    num_local_and_halo_tets = num_local_tets;
    num_local_and_halo_hexs = num_local_hexs;
    num_local_and_halo_pyrds = num_local_pyrds;
    num_local_and_halo_wedges = num_local_wedges;

    tets = new int[num_local_tets * tet_dof_per_elem];
    hexs = new int[num_local_hexs * hex_dof_per_elem];
    pyrds = new int[num_local_pyrds * pyrd_dof_per_elem];
    wedges = new int[num_local_wedges * wedge_dof_per_elem];

    // Count up the number of owned and halo nodes
    num_owned_nodes = 0;

    int *vert_flags = new int[conn.get_num_verts()];
    int *edge_flags = new int[conn.get_num_edges()];
    int *face_flags = new int[conn.get_num_faces()];
    int *vol_flags = new int[conn.get_num_elements()];

    // Loop over all of the elements and count up the number of degrees of
    // freeom

    for (int i = 0; i) }

  const int tet_dof_per_elem;
  const int hex_dof_per_elem;
  const int pyrd_dof_per_elem;
  const int wedge_dof_per_elem;

  // Input information about the mesh
  int num_owned_nodes;  // Number of nodes that belong to this processor
  int num_owned_and_halo_nodes;  // Number of nodes + number of halo nodes

  // Global offset for the owned nodes
  // The global index of an owned node = global_node_offset + local index
  int global_node_offset;

  // Sorted array of the global node numbers for the halo nodes
  // The global node number of a halo node =
  // halo_node_number[local node number - num_owned_nodes]
  int *halo_node_numbers;

  // Element information
  // Number of local elements = elements in the mesh that touch an owned node
  int num_local_tets;
  int num_local_hexs;
  int num_local_pyrds;
  int num_local_wedges;

  // Number of local + halo elements = elements that touch an owned node
  // and the elements that share a node with the local elements
  int num_local_and_halo_tets;
  int num_local_and_halo_hexs;
  int num_local_and_halo_pyrds;
  int num_local_and_halo_wedges;

  // Boundary connectivity information
  int num_boundaries;               // Number of boundaries in the mesh
  BoundaryConnectivity **boundary;  // Boundary objects

  // Connectivity<ndim> *create_global_partitions(
  //     int mpi_size, int **owner_range0, int **global_node_numbers0) const {
  //   Connectivity<ndim> *conn =
  //       new Connectivity<ndim>(num_owned_nodes, num_boundaries);

  //   // Allocate space to store the partition
  //   int nnodes = num_owned_nodes;
  //   int *partition = new int[nnodes];

  //   if (mpi_size > 1) {
  //     // Create the CSR structure
  //     int *rowp, *cols;
  //     bool include_diag = false;
  //     create_csr(&rowp, &cols, include_diag);

  //     // Partition via METIS
  //     int ncon = 1;  // "It should be at least 1"??

  //     // Set the default options
  //     int options[METIS_NOPTIONS];
  //     METIS_SetDefaultOptions(options);

  //     // Use 0-based numbering
  //     options[METIS_OPTION_NUMBERING] = 0;

  //     // The objective value in METIS
  //     int objval = 0;

  //     if (mpi_size < 8) {
  //       METIS_PartGraphRecursive(&nnodes, &ncon, rowp, cols, NULL, NULL,
  //       NULL,
  //                                &mpi_size, NULL, NULL, options, &objval,
  //                                partition);
  //     } else {
  //       METIS_PartGraphKway(&nnodes, &ncon, rowp, cols, NULL, NULL, NULL,
  //                           &mpi_size, NULL, NULL, options, &objval,
  //                           partition);
  //     }

  //     delete[] rowp;
  //     delete[] cols;

  //   } else {
  //     for (int i = 0; i < nnodes; i++) {
  //       partition[i] = 0;
  //     }
  //   }

  //   // Set the global node numbers
  //   int *owner_range = new int[mpi_size + 1];
  //   std::fill(owner_range, owner_range + mpi_size + 1, 0);
  //   for (int i = 0; i < num_owned_nodes; i++) {
  //     owner_range[partition[i] + 1]++;
  //   }

  //   // Set the ownership range
  //   for (int i = 0; i < mpi_size; i++) {
  //     owner_range[i + 1] += owner_range[i];
  //   }

  //   // Compute the global node numbers
  //   int *global_node_numbers = new int[num_owned_nodes];
  //   for (int i = 0; i < num_owned_nodes; i++) {
  //     global_node_numbers[i] = owner_range[partition[i]];
  //     owner_range[partition[i]]++;
  //   }

  //   // Reset the owner range
  //   for (int i = 0, next = 0; i < mpi_size; i++) {
  //     int tmp = owner_range[i];
  //     owner_range[i] = next;
  //     next = tmp;
  //   }

  //   // Set the local number of elements
  //   conn->num_local_tris = num_local_tris;
  //   conn->num_local_quads = num_local_quads;
  //   conn->num_local_tets = num_local_tets;
  //   conn->num_local_hexs = num_local_hexs;
  //   conn->num_local_pyrds = num_local_pyrds;
  //   conn->num_local_wedges = num_local_wedges;

  //   // Since we're partitioning from a serial connectivity, these values
  //   // should be the same as above - i.e. no halo nodes
  //   conn->num_local_and_halo_tris = num_local_tris;
  //   conn->num_local_and_halo_quads = num_local_quads;
  //   conn->num_local_and_halo_tets = num_local_tets;
  //   conn->num_local_and_halo_hexs = num_local_hexs;
  //   conn->num_local_and_halo_pyrds = num_local_pyrds;
  //   conn->num_local_and_halo_wedges = num_local_wedges;

  //   // Allocate new space for the data
  //   conn->tris = new Triangle[num_local_and_halo_tris];
  //   conn->quads = new Quadrilateral[num_local_and_halo_quads];
  //   conn->tets = new Tetrahedron[num_local_and_halo_tets];
  //   conn->hexs = new Hexahedron[num_local_and_halo_hexs];
  //   conn->pyrds = new Pyramid[num_local_and_halo_pyrds];
  //   conn->wedges = new Wedge[num_local_and_halo_wedges];

  //   // Set the node numbers
  //   for (int i = 0; i < num_local_and_halo_tris; i++) {
  //     for (int j = 0; j < TriInfo::NNODES; j++) {
  //       conn->tris[i].nodes[j] = global_node_numbers[tris[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < num_local_and_halo_quads; i++) {
  //     for (int j = 0; j < QuadInfo::NNODES; j++) {
  //       conn->quads[i].nodes[j] = global_node_numbers[quads[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < num_local_and_halo_tets; i++) {
  //     for (int j = 0; j < TetInfo::NNODES; j++) {
  //       conn->tets[i].nodes[j] = global_node_numbers[tets[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < num_local_and_halo_hexs; i++) {
  //     for (int j = 0; j < HexInfo::NNODES; j++) {
  //       conn->hexs[i].nodes[j] = global_node_numbers[hexs[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < num_local_and_halo_pyrds; i++) {
  //     for (int j = 0; j < PyramidInfo::NNODES; j++) {
  //       conn->pyrds[i].nodes[j] = global_node_numbers[pyrds[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < num_local_and_halo_wedges; i++) {
  //     for (int j = 0; j < WedgeInfo::NNODES; j++) {
  //       conn->wedges[i].nodes[j] = global_node_numbers[wedges[i].nodes[j]];
  //     }
  //   }

  //   // Set the boundary numbers
  //   for (int index = 0; index < num_boundaries; index++) {
  //     const BoundaryConnectivity *b0 = boundary[index];
  //     BoundaryConnectivity *b = new BoundaryConnectivity();

  //     b->num_owned_nodes = b0->num_owned_nodes;
  //     b->num_owned_and_halo_nodes = b0->num_owned_and_halo_nodes;
  //     b->nodes = new int[b0->num_owned_and_halo_nodes];
  //     for (int i = 0; i < b0->num_owned_and_halo_nodes; i++) {
  //       b->nodes[i] = global_node_numbers[b0->nodes[i]];
  //     }

  //     b->num_local_edges = b0->num_local_edges;
  //     b->num_local_and_halo_edges = b0->num_local_and_halo_edges;
  //     b->edges = new Edge[b0->num_local_and_halo_edges];
  //     for (int i = 0; i < b0->num_local_and_halo_edges; i++) {
  //       for (int j = 0; j < 2; j++) {
  //         b->edges[i].nodes[j] = global_node_numbers[b0->edges[i].nodes[j]];
  //       }
  //     }

  //     b->num_local_tris = b0->num_local_tris;
  //     b->num_local_and_halo_tris = b0->num_local_and_halo_tris;
  //     b->tris = new Triangle[b0->num_local_and_halo_tris];
  //     for (int i = 0; i < b0->num_local_and_halo_tris; i++) {
  //       for (int j = 0; j < TriInfo::NNODES; j++) {
  //         b->tris[i].nodes[j] = global_node_numbers[b0->tris[i].nodes[j]];
  //       }
  //     }

  //     b->num_local_quads = b0->num_local_quads;
  //     b->num_local_and_halo_quads = b0->num_local_and_halo_quads;
  //     b->quads = new Quadrilateral[b0->num_local_and_halo_quads];
  //     for (int i = 0; i < b0->num_local_and_halo_quads; i++) {
  //       for (int j = 0; j < QuadInfo::NNODES; j++) {
  //         b->quads[i].nodes[j] = global_node_numbers[b0->quads[i].nodes[j]];
  //       }
  //     }

  //     conn->boundary[index] = b;
  //   }

  //   delete[] partition;

  //   *owner_range0 = owner_range;
  //   *global_node_numbers0 = global_node_numbers;

  //   return conn;
  // }

  // Connectivity<ndim> *create_local_partition(int start, int end,
  //                                            int **node_to_elem_ptr,
  //                                            int **node_to_elems) const {
  //   int *node_ptr, *node_elems;
  //   if (*node_to_elem_ptr && *node_to_elems) {
  //     node_ptr = *node_to_elem_ptr;
  //     node_elems = *node_to_elems;
  //   } else {
  //     init_node_element_data(&node_ptr, &node_elems);
  //     *node_to_elem_ptr = node_ptr;
  //     *node_to_elems = node_elems;
  //   }

  //   int nowned = end - start;

  //   Connectivity<ndim> *conn = new Connectivity<ndim>(nowned,
  //   num_boundaries);

  //   // The local elements - this is serial so no halos for this
  //   int nelems = get_num_local_elements();

  //   // Keep track of the elements, halo nodes and local node numbers
  //   int nhalo = 0;
  //   int *elem_marker = new int[nelems];
  //   int *halo_nodes = new int[num_owned_nodes];
  //   int *local_node_numbers = new int[num_owned_nodes];

  //   // Set the markers for the 0th halo and 1st halo
  //   const int HALO_ZERO_LABEL = 1;
  //   const int HALO_ONE_LABEL = 2;

  //   // Set the markers to NO_LABEL for the element marker and nodes. Note
  //   // that local_node_numbers is used as a temporary marker array and later
  //   // to store the local node numbers
  //   std::fill(elem_marker, elem_marker + nelems, NO_LABEL);
  //   std::fill(local_node_numbers, local_node_numbers + num_owned_nodes,
  //             NO_LABEL);

  //   for (int node = start; node < end; node++) {
  //     local_node_numbers[node] = HALO_ZERO_LABEL;
  //   }

  //   // Look for adjacent elements
  //   for (int node = start; node < end; node++) {
  //     for (int ptr = node_ptr[node]; ptr < node_ptr[node + 1]; ptr++) {
  //       int elem = node_elems[ptr];

  //       if (elem_marker[elem] == NO_LABEL) {
  //         const int *nodes;
  //         int nelem_nodes = get_element_nodes(elem, &nodes);

  //         for (int i = 0; i < nelem_nodes; i++) {
  //           // Check if this node should be added as a halo node
  //           if (local_node_numbers[nodes[i]] == NO_LABEL) {
  //             if (nodes[i] < start || nodes[i] >= end) {
  //               halo_nodes[nhalo] = nodes[i];
  //               nhalo++;
  //             }
  //             local_node_numbers[nodes[i]] = HALO_ZERO_LABEL;
  //           }
  //         }

  //         // This is a local element
  //         elem_marker[elem] = HALO_ZERO_LABEL;
  //       }
  //     }
  //   }

  //   // Now, go through the halo nodes to add the second layer of nodes...
  //   int num_halo_zero = nhalo;
  //   for (int k = 0; k < num_halo_zero; k++) {
  //     int node = halo_nodes[k];
  //     for (int ptr = node_ptr[node]; ptr < node_ptr[node + 1]; ptr++) {
  //       int elem = node_elems[ptr];

  //       if (elem_marker[elem] == NO_LABEL) {
  //         const int *nodes;
  //         int nelem_nodes = get_element_nodes(elem, &nodes);

  //         for (int i = 0; i < nelem_nodes; i++) {
  //           // Check if this node should be added as a halo node
  //           if (local_node_numbers[nodes[i]] == NO_LABEL) {
  //             if (nodes[i] < start || nodes[i] >= end) {
  //               halo_nodes[nhalo] = nodes[i];
  //               nhalo++;
  //             }
  //             local_node_numbers[nodes[i]] = HALO_ONE_LABEL;
  //           }
  //         }

  //         // This is a local element
  //         elem_marker[elem] = HALO_ONE_LABEL;
  //       }
  //     }
  //   }

  //   // Count up the elements
  //   conn->num_local_tris = 0;
  //   conn->num_local_quads = 0;
  //   conn->num_local_tets = 0;
  //   conn->num_local_hexs = 0;
  //   conn->num_local_pyrds = 0;
  //   conn->num_local_wedges = 0;

  //   conn->num_local_and_halo_tris = 0;
  //   conn->num_local_and_halo_quads = 0;
  //   conn->num_local_and_halo_tets = 0;
  //   conn->num_local_and_halo_hexs = 0;
  //   conn->num_local_and_halo_pyrds = 0;
  //   conn->num_local_and_halo_wedges = 0;

  //   int index = 0;
  //   for (int i = 0; i < num_local_tris; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_tris++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_tris++;
  //     }
  //   }
  //   for (int i = 0; i < num_local_quads; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_quads++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_quads++;
  //     }
  //   }
  //   for (int i = 0; i < num_local_tets; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_tets++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_tets++;
  //     }
  //   }
  //   for (int i = 0; i < num_local_hexs; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_hexs++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_hexs++;
  //     }
  //   }
  //   for (int i = 0; i < num_local_pyrds; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_pyrds++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_pyrds++;
  //     }
  //   }
  //   for (int i = 0; i < num_local_wedges; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->num_local_wedges++;
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->num_local_and_halo_wedges++;
  //     }
  //   }

  //   conn->num_local_and_halo_tris += conn->num_local_tris;
  //   conn->num_local_and_halo_quads += conn->num_local_quads;
  //   conn->num_local_and_halo_tets += conn->num_local_tets;
  //   conn->num_local_and_halo_hexs += conn->num_local_hexs;
  //   conn->num_local_and_halo_pyrds += conn->num_local_pyrds;
  //   conn->num_local_and_halo_wedges += conn->num_local_wedges;

  //   conn->tris = new Triangle[conn->num_local_and_halo_tris];
  //   conn->quads = new Quadrilateral[conn->num_local_and_halo_quads];
  //   conn->tets = new Tetrahedron[conn->num_local_and_halo_tets];
  //   conn->hexs = new Hexahedron[conn->num_local_and_halo_hexs];
  //   conn->pyrds = new Pyramid[conn->num_local_and_halo_pyrds];
  //   conn->wedges = new Wedge[conn->num_local_and_halo_wedges];

  //   // Copy over the elements (these still have the global ordering)
  //   index = 0;
  //   for (int i = 0, local = 0, halo = conn->num_local_tris; i <
  //   num_local_tris;
  //        i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->tris[local++] = tris[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->tris[halo++] = tris[index];
  //     }
  //   }
  //   for (int i = 0, local = 0, halo = conn->num_local_quads;
  //        i < num_local_quads; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->quads[local++] = quads[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->quads[halo++] = quads[index];
  //     }
  //   }
  //   for (int i = 0, local = 0, halo = conn->num_local_tets; i <
  //   num_local_tets;
  //        i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->tets[local++] = tets[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->tets[halo++] = tets[index];
  //     }
  //   }
  //   for (int i = 0, local = 0, halo = conn->num_local_hexs; i <
  //   num_local_hexs;
  //        i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->hexs[local++] = hexs[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->hexs[halo++] = hexs[index];
  //     }
  //   }
  //   for (int i = 0, local = 0, halo = conn->num_local_pyrds;
  //        i < num_local_pyrds; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->pyrds[local++] = pyrds[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->pyrds[halo++] = pyrds[index];
  //     }
  //   }
  //   for (int i = 0, local = 0, halo = conn->num_local_wedges;
  //        i < num_local_wedges; i++, index++) {
  //     if (elem_marker[index] == HALO_ZERO_LABEL) {
  //       conn->wedges[local++] = wedges[index];
  //     } else if (elem_marker[index] == HALO_ONE_LABEL) {
  //       conn->wedges[halo++] = wedges[index];
  //     }
  //   }

  //   // Sort the halo nodes
  //   std::sort(halo_nodes, halo_nodes + nhalo);

  //   conn->num_owned_and_halo_nodes = conn->num_owned_nodes + nhalo;
  //   conn->halo_node_numbers = new int[nhalo];
  //   for (int i = 0; i < nhalo; i++) {
  //     conn->halo_node_numbers[i] = halo_nodes[i];
  //   }

  //   // Now assign the local node numbers. Order the owned nodes first as is
  //   // required
  //   int local_node_num = 0;
  //   for (int node = start; node < end; node++, local_node_num++) {
  //     local_node_numbers[node] = local_node_num;
  //   }

  //   // The halo node numbers are sorted based on their global index
  //   for (int i = 0; i < nhalo; i++, local_node_num++) {
  //     local_node_numbers[halo_nodes[i]] = local_node_num;
  //   }

  //   // Set the node numbers
  //   for (int i = 0; i < conn->num_local_and_halo_tris; i++) {
  //     for (int j = 0; j < TriInfo::NNODES; j++) {
  //       conn->tris[i].nodes[j] = local_node_numbers[conn->tris[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < conn->num_local_and_halo_quads; i++) {
  //     for (int j = 0; j < QuadInfo::NNODES; j++) {
  //       conn->quads[i].nodes[j] =
  //       local_node_numbers[conn->quads[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < conn->num_local_and_halo_tets; i++) {
  //     for (int j = 0; j < TetInfo::NNODES; j++) {
  //       conn->tets[i].nodes[j] = local_node_numbers[conn->tets[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < conn->num_local_and_halo_hexs; i++) {
  //     for (int j = 0; j < HexInfo::NNODES; j++) {
  //       conn->hexs[i].nodes[j] = local_node_numbers[conn->hexs[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < conn->num_local_and_halo_pyrds; i++) {
  //     for (int j = 0; j < PyramidInfo::NNODES; j++) {
  //       conn->pyrds[i].nodes[j] =
  //       local_node_numbers[conn->pyrds[i].nodes[j]];
  //     }
  //   }
  //   for (int i = 0; i < conn->num_local_and_halo_wedges; i++) {
  //     for (int j = 0; j < WedgeInfo::NNODES; j++) {
  //       conn->wedges[i].nodes[j] =
  //       local_node_numbers[conn->wedges[i].nodes[j]];
  //     }
  //   }

  //   // Handle the boundaries
  //   for (int index = 0; index < conn->num_boundaries; index++) {
  //     const BoundaryConnectivity *b0 = boundary[index];
  //     BoundaryConnectivity *b = new BoundaryConnectivity();

  //     b->num_owned_nodes = 0;
  //     b->num_owned_and_halo_nodes = 0;
  //     b->num_local_edges = 0;
  //     b->num_local_and_halo_edges = 0;
  //     b->num_local_tris = 0;
  //     b->num_local_and_halo_tris = 0;
  //     b->num_local_quads = 0;
  //     b->num_local_and_halo_quads = 0;

  //     // Count up the number of owned and halo nodes and local and halo
  //     elements
  //     // on the boundaries
  //     for (int i = 0; i < b0->num_owned_nodes; i++) {
  //       // Check the status
  //       int n0 = local_node_numbers[b0->nodes[i]];
  //       if (n0 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes) {
  //           b->num_owned_nodes++;
  //         } else {
  //           b->num_owned_and_halo_nodes++;
  //         }
  //       }
  //     }

  //     b->num_owned_and_halo_nodes += b->num_owned_nodes;

  //     // Allocate space to store the node numbers
  //     b->nodes = new int[b->num_owned_and_halo_nodes];
  //     for (int i = 0, local = 0, halo = 0; i < b0->num_owned_nodes; i++) {
  //       // Check the status
  //       int n0 = local_node_numbers[b0->nodes[i]];
  //       if (n0 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes) {
  //           b->nodes[local] = n0;
  //           local++;
  //         } else {
  //           b->nodes[halo + b->num_owned_nodes] = n0;
  //           halo++;
  //         }
  //       }
  //     }

  //     // Count up the number of edges
  //     for (int i = 0; i < b0->num_local_and_halo_edges; i++) {
  //       int n0 = local_node_numbers[b0->edges[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->edges[i].nodes[1]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes) {
  //           b->num_local_edges++;
  //         } else {
  //           b->num_local_and_halo_edges++;
  //         }
  //       }
  //     }

  //     b->num_local_and_halo_edges += b->num_local_edges;

  //     // Allocate space to store the edges
  //     b->edges = new Edge[b->num_local_and_halo_edges];
  //     for (int i = 0, local = 0, halo = 0; i < b0->num_local_and_halo_edges;
  //          i++) {
  //       // Check the status
  //       int n0 = local_node_numbers[b0->edges[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->edges[i].nodes[1]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes) {
  //           b->edges[local].nodes[0] = n0;
  //           b->edges[local].nodes[1] = n1;
  //           local++;
  //         } else {
  //           b->edges[halo + b->num_local_edges].nodes[0] = n0;
  //           b->edges[halo + b->num_local_edges].nodes[1] = n1;
  //           halo++;
  //         }
  //       }
  //     }

  //     // Count up the number of triangles
  //     for (int i = 0; i < b0->num_local_and_halo_tris; i++) {
  //       int n0 = local_node_numbers[b0->tris[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->tris[i].nodes[1]];
  //       int n2 = local_node_numbers[b0->tris[i].nodes[2]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL && n2 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes ||
  //             n2 < conn->num_owned_nodes) {
  //           b->num_local_tris++;
  //         } else {
  //           b->num_local_and_halo_tris++;
  //         }
  //       }
  //     }

  //     b->num_local_and_halo_tris += b->num_local_tris;

  //     // Allocate space to store the triangles
  //     b->tris = new Triangle[b->num_local_and_halo_tris];
  //     for (int i = 0, local = 0, halo = 0; i < b0->num_local_and_halo_tris;
  //          i++) {
  //       int n0 = local_node_numbers[b0->tris[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->tris[i].nodes[1]];
  //       int n2 = local_node_numbers[b0->tris[i].nodes[2]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL && n2 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes ||
  //             n2 < conn->num_owned_nodes) {
  //           b->tris[local].nodes[0] = n0;
  //           b->tris[local].nodes[1] = n1;
  //           b->tris[local].nodes[2] = n2;
  //           local++;
  //         } else {
  //           b->tris[halo + b->num_local_tris].nodes[0] = n0;
  //           b->tris[halo + b->num_local_tris].nodes[1] = n1;
  //           b->tris[halo + b->num_local_tris].nodes[2] = n2;
  //           halo++;
  //         }
  //       }
  //     }

  //     // Count up the number of quads
  //     for (int i = 0; i < b0->num_local_and_halo_quads; i++) {
  //       int n0 = local_node_numbers[b0->quads[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->quads[i].nodes[1]];
  //       int n2 = local_node_numbers[b0->quads[i].nodes[2]];
  //       int n3 = local_node_numbers[b0->quads[i].nodes[3]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL && n2 != NO_LABEL &&
  //           n3 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes ||
  //             n2 < conn->num_owned_nodes || n3 < conn->num_owned_nodes) {
  //           b->num_local_quads++;
  //         } else {
  //           b->num_local_and_halo_quads++;
  //         }
  //       }
  //     }

  //     b->num_local_and_halo_quads += b->num_local_quads;

  //     // Allocate space to store the triangles
  //     b->quads = new Quadrilateral[b->num_local_and_halo_quads];
  //     for (int i = 0, local = 0, halo = 0; i < b0->num_local_and_halo_quads;
  //          i++) {
  //       int n0 = local_node_numbers[b0->quads[i].nodes[0]];
  //       int n1 = local_node_numbers[b0->quads[i].nodes[1]];
  //       int n2 = local_node_numbers[b0->quads[i].nodes[2]];
  //       int n3 = local_node_numbers[b0->quads[i].nodes[3]];
  //       if (n0 != NO_LABEL && n1 != NO_LABEL && n2 != NO_LABEL &&
  //           n3 != NO_LABEL) {
  //         if (n0 < conn->num_owned_nodes || n1 < conn->num_owned_nodes ||
  //             n2 < conn->num_owned_nodes || n3 < conn->num_owned_nodes) {
  //           b->quads[local].nodes[0] = n0;
  //           b->quads[local].nodes[1] = n1;
  //           b->quads[local].nodes[2] = n2;
  //           b->quads[local].nodes[3] = n3;
  //           local++;
  //         } else {
  //           b->quads[halo + b->num_local_quads].nodes[0] = n0;
  //           b->quads[halo + b->num_local_quads].nodes[1] = n1;
  //           b->quads[halo + b->num_local_quads].nodes[2] = n2;
  //           b->quads[halo + b->num_local_quads].nodes[3] = n3;
  //           halo++;
  //         }
  //       }
  //     }

  //     conn->boundary[index] = b;
  //   }

  //   // Deallocate the space
  //   delete[] local_node_numbers;
  //   delete[] elem_marker;
  //   delete[] halo_nodes;

  //   return conn;
  // }

  // /**
  //  * @brief Send the data from the connectivity on this processor to the
  //  * connectivity on another processor
  //  *
  //  * @param comm MPI communicator
  //  * @param dest_rank Destination rank for the data
  //  * @param tag Tag associated with the communication
  //  */
  // void send_to_proc(MPI_Comm comm, int dest_rank, int tag) {
  //   const int buff_size = 16;
  //   int buff[buff_size];
  //   buff[0] = num_owned_nodes;
  //   buff[1] = num_owned_and_halo_nodes;
  //   buff[2] = global_node_offset;
  //   buff[3] = num_local_tris;
  //   buff[4] = num_local_quads;
  //   buff[5] = num_local_tets;
  //   buff[6] = num_local_hexs;
  //   buff[7] = num_local_pyrds;
  //   buff[8] = num_local_wedges;
  //   buff[9] = num_local_and_halo_tris;
  //   buff[10] = num_local_and_halo_quads;
  //   buff[11] = num_local_and_halo_tets;
  //   buff[12] = num_local_and_halo_hexs;
  //   buff[13] = num_local_and_halo_pyrds;
  //   buff[14] = num_local_and_halo_wedges;
  //   buff[15] = num_boundaries;

  //   MPI_Send(buff, buff_size, MPI_INT, dest_rank, tag, comm);

  //   int num_halo = num_owned_and_halo_nodes - num_owned_nodes;
  //   if (num_halo > 0) {
  //     MPI_Send(halo_node_numbers, num_halo, MPI_INT, dest_rank, tag, comm);
  //   }

  //   if (num_local_and_halo_tris > 0) {
  //     MPI_Send(tris, num_local_and_halo_tris, Triangle_MPI_type, dest_rank,
  //     tag,
  //              comm);
  //   }
  //   if (num_local_and_halo_quads > 0) {
  //     MPI_Send(quads, num_local_and_halo_quads, Quadrilateral_MPI_type,
  //              dest_rank, tag, comm);
  //   }
  //   if (num_local_and_halo_tets > 0) {
  //     MPI_Send(tets, num_local_and_halo_tets, Tetrahedron_MPI_type,
  //     dest_rank,
  //              tag, comm);
  //   }
  //   if (num_local_and_halo_hexs > 0) {
  //     MPI_Send(hexs, num_local_and_halo_hexs, Hexahedron_MPI_type, dest_rank,
  //              tag, comm);
  //   }
  //   if (num_local_and_halo_pyrds > 0) {
  //     MPI_Send(pyrds, num_local_and_halo_pyrds, Pyramid_MPI_type, dest_rank,
  //              tag, comm);
  //   }
  //   if (num_local_and_halo_wedges > 0) {
  //     MPI_Send(wedges, num_local_and_halo_wedges, Wedge_MPI_type, dest_rank,
  //              tag, comm);
  //   }

  //   for (int i = 0; i < num_boundaries; i++) {
  //     boundary[i]->send_to_proc(comm, dest_rank, tag);
  //   }
  // }

  // /**
  //  * @brief Receive the data for the connectivity from a processor
  //  *
  //  * @param comm MPI communicator
  //  * @param src_rank Source rank for the data
  //  * @param tag Tag associated with the communication
  //  */
  // void recv_from_proc(MPI_Comm comm, int src_rank, int tag) {
  //   const int buff_size = 16;
  //   int buff[buff_size];

  //   MPI_Recv(buff, buff_size, MPI_INT, src_rank, tag, comm,
  //   MPI_STATUS_IGNORE);

  //   num_owned_nodes = buff[0];
  //   num_owned_and_halo_nodes = buff[1];
  //   global_node_offset = buff[2];
  //   num_local_tris = buff[3];
  //   num_local_quads = buff[4];
  //   num_local_tets = buff[5];
  //   num_local_hexs = buff[6];
  //   num_local_pyrds = buff[7];
  //   num_local_wedges = buff[8];
  //   num_local_and_halo_tris = buff[9];
  //   num_local_and_halo_quads = buff[10];
  //   num_local_and_halo_tets = buff[11];
  //   num_local_and_halo_hexs = buff[12];
  //   num_local_and_halo_pyrds = buff[13];
  //   num_local_and_halo_wedges = buff[14];
  //   num_boundaries = buff[15];

  //   int num_halo = num_owned_and_halo_nodes - num_owned_nodes;
  //   if (num_halo > 0) {
  //     halo_node_numbers = new int[num_halo];
  //     MPI_Recv(halo_node_numbers, num_halo, MPI_INT, src_rank, tag, comm,
  //              MPI_STATUS_IGNORE);
  //   }

  //   if (num_local_and_halo_tris > 0) {
  //     tris = new Triangle[num_local_and_halo_tris];
  //     MPI_Recv(tris, num_local_and_halo_tris, Triangle_MPI_type, src_rank,
  //     tag,
  //              comm, MPI_STATUS_IGNORE);
  //   }
  //   if (num_local_and_halo_quads > 0) {
  //     quads = new Quadrilateral[num_local_and_halo_quads];
  //     MPI_Recv(quads, num_local_and_halo_quads, Quadrilateral_MPI_type,
  //              src_rank, tag, comm, MPI_STATUS_IGNORE);
  //   }
  //   if (num_local_and_halo_tets > 0) {
  //     tets = new Tetrahedron[num_local_and_halo_tets];
  //     MPI_Recv(tets, num_local_and_halo_tets, Tetrahedron_MPI_type, src_rank,
  //              tag, comm, MPI_STATUS_IGNORE);
  //   }
  //   if (num_local_and_halo_hexs > 0) {
  //     hexs = new Hexahedron[num_local_and_halo_hexs];
  //     MPI_Recv(hexs, num_local_and_halo_hexs, Hexahedron_MPI_type, src_rank,
  //              tag, comm, MPI_STATUS_IGNORE);
  //   }
  //   if (num_local_and_halo_pyrds > 0) {
  //     pyrds = new Pyramid[num_local_and_halo_pyrds];
  //     MPI_Recv(pyrds, num_local_and_halo_pyrds, Pyramid_MPI_type, src_rank,
  //     tag,
  //              comm, MPI_STATUS_IGNORE);
  //   }
  //   if (num_local_and_halo_wedges > 0) {
  //     wedges = new Wedge[num_local_and_halo_wedges];
  //     MPI_Recv(wedges, num_local_and_halo_wedges, Wedge_MPI_type, src_rank,
  //     tag,
  //              comm, MPI_STATUS_IGNORE);
  //   }

  //   boundary = new BoundaryConnectivity *[num_boundaries];
  //   for (int i = 0; i < num_boundaries; i++) {
  //     boundary[i] = new BoundaryConnectivity();
  //     boundary[i]->recv_from_proc(comm, src_rank, tag);
  //   }
  // }
};

#endif  // CONNECTIVITY_CONVERTER_H