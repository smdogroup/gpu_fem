#ifndef BASIC_CONNECTIVITY_H
#define BASIC_CONNECTIVITY_H

#include <algorithm>
#include <cstdint>

enum ElementType {
  EDGE,
  TRIANGLE,
  QUADRILATERAL,
  TETRAHEDRON,
  HEXAHEDRON,
  PYRAMID,
  WEDGE
};

/**
 * @brief Static class for Triangle information
 *
 *   N2
 *   *  *
 *   *      *
 *   E1        E0
 *   *             *
 *   *                *
 *   N0 * * * E2 * * * N1
 *
 * Connectivities:
 *
 * Edges:
 *
 * E0: (1, 2), E1: (2, 0), E2: (0, 1)
 */
class Triangle {
public:
  static const int ndim = 2;
  static const int NVERTS = 3;
  static const int NEDGES = 3;

  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];

  // Edge -> Vert info
  static const int EDGE0_VERT0 = 1;
  static const int EDGE0_VERT1 = 2;

  static const int EDGE1_VERT0 = 2;
  static const int EDGE1_VERT1 = 0;

  static const int EDGE2_VERT0 = 0;
  static const int EDGE2_VERT1 = 1;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      *n0 = verts[EDGE0_VERT0];
      *n1 = verts[EDGE0_VERT1];
    } else if (edge == 1) {
      *n0 = verts[EDGE1_VERT0];
      *n1 = verts[EDGE1_VERT1];
    } else if (edge == 2) {
      *n0 = verts[EDGE2_VERT0];
      *n1 = verts[EDGE2_VERT1];
    }
  }

  /**
   * @brief Return if the two lists of face vertices are flipped
   *
   * @param v1
   * @param v2
   */
  template <typename I>
  inline static bool is_flipped(const I v1[], const I v2[]) {
    if (v1[0] == v2[0] and v1[1] == v2[2] and v1[2] == v2[1]) {
      return true; // Same orientation
    }

    else if (v1[0] == v2[1] and v1[1] == v2[0] and v1[2] == v2[2]) {
      return true; // Rotated
    }

    else if (v1[0] == v2[2] and v1[1] == v2[1] and v1[2] == v2[0]) {
      return true; // Rotated
    }

    return false; // Flipped
  }
};

/**
 * @brief Quadrilateral information
 *
 *  N3 * * * E2 * * * N2
 *  *                 *
 *  *                 *
 *  E3                E1
 *  *                 *
 *  *                 *
 *  N0 * * * E0 * * * N1
 */
class Quadrilateral {
public:
  static const int ndim = 2;
  static const int NVERTS = 4;
  static const int NEDGES = 4;

  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];

  // Edge -> Vert info
  static const int EDGE0_VERT0 = 0;
  static const int EDGE0_VERT1 = 1;

  static const int EDGE1_VERT0 = 1;
  static const int EDGE1_VERT1 = 2;

  static const int EDGE2_VERT0 = 2;
  static const int EDGE2_VERT1 = 3;

  static const int EDGE3_VERT0 = 3;
  static const int EDGE3_VERT1 = 0;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      *n0 = verts[EDGE0_VERT0];
      *n1 = verts[EDGE0_VERT1];
    } else if (edge == 1) {
      *n0 = verts[EDGE1_VERT0];
      *n1 = verts[EDGE1_VERT1];
    } else if (edge == 2) {
      *n0 = verts[EDGE2_VERT0];
      *n1 = verts[EDGE2_VERT1];
    } else if (edge == 3) {
      *n0 = verts[EDGE3_VERT0];
      *n1 = verts[EDGE3_VERT1];
    }
  }

  /**
   * @brief Return if the two lists of vertices have flipped orientation
   *
   * @param v1
   * @param v2
   */
  template <typename I>
  inline static bool is_flipped(const I v1[], const I v2[]) {
    if (v1[0] == v2[0] and v1[1] == v2[3] and v1[2] == v2[2] and
        v1[3] == v2[1]) {
      return true;
    } else if (v1[0] == v2[1] and v1[1] == v2[0] and v1[2] == v2[3] and
               v1[3] == v2[2]) {
      return true;
    } else if (v1[0] == v2[2] and v1[1] == v2[1] and v1[2] == v2[0] and
               v1[3] == v2[3]) {
      return true;
    } else if (v1[0] == v2[3] and v1[1] == v2[2] and v1[2] == v2[1] and
               v1[3] == v2[0]) {
      return true;
    }
    return false;
  }
};

/**
 * @brief Static class for the tetrahedral information
 *
 *         V3
 *         ..  .
 *        .  .    .
 *       .    .     E5
 *      E3     .        .
 *     .       E4           V2
 *    .          .   *      *
 *   .     * E1   .        *
 *  V0             .      E0
 *      *           .    *
 *          E2       .  *
 *              *     .*
 *                    V1
 *
 * Connectivities:
 *
 * Edges:
 *
 * E0: (1, 2), E1: (2, 0), E2: (0, 1), E3: (0, 3), E4: (1, 3), E5: (2, 3)
 *
 * Oriented for outward-facing normals (faces ordered from opposite verts)
 *
 * Faces: F0: (1, 2, 3), F1: (0, 3, 2), F2: (0, 1, 3), F3: (0, 2, 1)
 */
class Tetrahedron {
public:
  static const int ndim = 3;
  static const int NVERTS = 4;
  static const int NEDGES = 6;
  static const int NFACES = 4;

  // Local information
  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];
  int32_t faces[NFACES];

  // Edge -> Vert info
  static const int EDGE0_VERT0 = 1;
  static const int EDGE0_VERT1 = 2;

  static const int EDGE1_VERT0 = 2;
  static const int EDGE1_VERT1 = 0;

  static const int EDGE2_VERT0 = 0;
  static const int EDGE2_VERT1 = 1;

  static const int EDGE3_VERT0 = 0;
  static const int EDGE3_VERT1 = 3;

  static const int EDGE4_VERT0 = 1;
  static const int EDGE4_VERT1 = 3;

  static const int EDGE5_VERT0 = 2;
  static const int EDGE5_VERT1 = 3;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      // E0: (1, 2)
      *n0 = verts[1];
      *n1 = verts[2];
    } else if (edge == 1) {
      // E1: (2, 0)
      *n0 = verts[2];
      *n1 = verts[0];
    } else if (edge == 2) {
      // E2: (0, 1)
      *n0 = verts[0];
      *n1 = verts[1];
    } else if (edge == 3) {
      // E3: (0, 3)
      *n0 = verts[0];
      *n1 = verts[3];
    } else if (edge == 4) {
      // E4: (1, 3)
      *n0 = verts[1];
      *n1 = verts[3];
    } else if (edge == 5) {
      // E5: (2, 3)
      *n0 = verts[2];
      *n1 = verts[3];
    }
  }

  // Get the face verts with an outward facing orientation
  template <typename I> inline int get_face_verts(const int face, I f[]) {
    if (face == 0) {
      // F0: (1, 2, 3)
      f[0] = verts[1];
      f[1] = verts[2];
      f[2] = verts[3];
    } else if (face == 1) {
      // F1: (0, 3, 2)
      f[0] = verts[0];
      f[1] = verts[3];
      f[2] = verts[2];
    } else if (face == 2) {
      // F2: (0, 1, 3)
      f[0] = verts[0];
      f[1] = verts[1];
      f[2] = verts[3];
    } else if (face == 3) {
      // F3: (0, 2, 1)
      f[0] = verts[0];
      f[1] = verts[2];
      f[2] = verts[1];
    }
    return Triangle::NVERTS; // Verts on each face
  }
};

/**
 * @brief Info to be filled in later
 *
 *         N7 * * * * E6 * * * * N6
 *        *|                   * |
 *     E7  |                 E5  |
 *    *   E11              *     |
 *  N4 * * * * E4  * * * N5     E10
 *   |     |             |       |
 *   |     |             |       |
 *   |     N3 * * * E2  *|* * * N2
 *   E8  *              E9    *
 *   |  E3               |   E1
 *   |*                  | *
 *  N0 * * * E0  * * * * N1
 */

class Hexahedron {
public:
  static const int ndim = 3;
  static const int NVERTS = 8;
  static const int NFACES = 6;
  static const int NEDGES = 12;

  // Local information
  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];
  int32_t faces[NFACES];

  // Edge -> Vert info
  static const int EDGE0_VERT0 = 0;
  static const int EDGE0_VERT1 = 1;

  static const int EDGE1_VERT0 = 1;
  static const int EDGE1_VERT1 = 2;

  static const int EDGE2_VERT0 = 2;
  static const int EDGE2_VERT1 = 3;

  static const int EDGE3_VERT0 = 0;
  static const int EDGE3_VERT1 = 3;

  static const int EDGE4_VERT0 = 4;
  static const int EDGE4_VERT1 = 5;

  static const int EDGE5_VERT0 = 5;
  static const int EDGE5_VERT1 = 6;

  static const int EDGE6_VERT0 = 6;
  static const int EDGE6_VERT1 = 7;

  static const int EDGE7_VERT0 = 7;
  static const int EDGE7_VERT1 = 4;

  static const int EDGE8_VERT0 = 0;
  static const int EDGE8_VERT1 = 4;

  static const int EDGE9_VERT0 = 1;
  static const int EDGE9_VERT1 = 5;

  static const int EDGE10_VERT0 = 2;
  static const int EDGE10_VERT1 = 6;

  static const int EDGE11_VERT0 = 3;
  static const int EDGE11_VERT1 = 7;

  // Face -> Verts
  static const int FACE0_VERT0 = 0;
  static const int FACE0_VERT1 = 4;
  static const int FACE0_VERT2 = 7;
  static const int FACE0_VERT3 = 3;

  static const int FACE1_VERT0 = 1;
  static const int FACE1_VERT1 = 2;
  static const int FACE1_VERT2 = 6;
  static const int FACE1_VERT3 = 5;

  static const int FACE2_VERT0 = 0;
  static const int FACE2_VERT1 = 1;
  static const int FACE2_VERT2 = 5;
  static const int FACE2_VERT3 = 4;

  static const int FACE3_VERT0 = 3;
  static const int FACE3_VERT1 = 7;
  static const int FACE3_VERT2 = 6;
  static const int FACE3_VERT3 = 2;

  static const int FACE4_VERT0 = 0;
  static const int FACE4_VERT1 = 3;
  static const int FACE4_VERT2 = 2;
  static const int FACE4_VERT3 = 1;

  static const int FACE5_VERT0 = 4;
  static const int FACE5_VERT1 = 5;
  static const int FACE5_VERT2 = 6;
  static const int FACE5_VERT3 = 7;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      *n0 = verts[EDGE0_VERT0];
      *n1 = verts[EDGE0_VERT1];
    } else if (edge == 1) {
      *n0 = verts[EDGE1_VERT0];
      *n1 = verts[EDGE1_VERT1];
    } else if (edge == 2) {
      *n0 = verts[EDGE2_VERT0];
      *n1 = verts[EDGE2_VERT1];
    } else if (edge == 3) {
      *n0 = verts[EDGE3_VERT0];
      *n1 = verts[EDGE3_VERT1];
    } else if (edge == 4) {
      *n0 = verts[EDGE4_VERT0];
      *n1 = verts[EDGE4_VERT1];
    } else if (edge == 5) {
      *n0 = verts[EDGE5_VERT0];
      *n1 = verts[EDGE5_VERT1];
    } else if (edge == 6) {
      *n0 = verts[EDGE6_VERT0];
      *n1 = verts[EDGE6_VERT1];
    } else if (edge == 7) {
      *n0 = verts[EDGE7_VERT0];
      *n1 = verts[EDGE7_VERT1];
    } else if (edge == 8) {
      *n0 = verts[EDGE8_VERT0];
      *n1 = verts[EDGE8_VERT1];
    } else if (edge == 9) {
      *n0 = verts[EDGE9_VERT0];
      *n1 = verts[EDGE9_VERT1];
    } else if (edge == 10) {
      *n0 = verts[EDGE10_VERT0];
      *n1 = verts[EDGE10_VERT1];
    } else if (edge == 11) {
      *n0 = verts[EDGE11_VERT0];
      *n1 = verts[EDGE11_VERT1];
    }
  }

  template <typename I> inline int get_face_verts(const int face, I f[]) {
    if (face == 0) {
      f[0] = verts[FACE0_VERT0];
      f[1] = verts[FACE0_VERT1];
      f[2] = verts[FACE0_VERT2];
      f[3] = verts[FACE0_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 1) {
      f[0] = verts[FACE1_VERT0];
      f[1] = verts[FACE1_VERT1];
      f[2] = verts[FACE1_VERT2];
      f[3] = verts[FACE1_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 2) {
      f[0] = verts[FACE2_VERT0];
      f[1] = verts[FACE2_VERT1];
      f[2] = verts[FACE2_VERT2];
      f[3] = verts[FACE2_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 3) {
      f[0] = verts[FACE3_VERT0];
      f[1] = verts[FACE3_VERT1];
      f[2] = verts[FACE3_VERT2];
      f[3] = verts[FACE3_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 4) {
      f[0] = verts[FACE4_VERT0];
      f[1] = verts[FACE4_VERT1];
      f[2] = verts[FACE4_VERT2];
      f[3] = verts[FACE4_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 5) {
      f[0] = verts[FACE5_VERT0];
      f[1] = verts[FACE5_VERT1];
      f[2] = verts[FACE5_VERT2];
      f[3] = verts[FACE5_VERT3];
      return Quadrilateral::NVERTS;
    }
    return 0;
  }
};

// Geometry information
/**
 * @brief Pyramid properties
 *
 * Pyramid is a 3D element, its entities include:
 * - domain
 * - bound
 * - edge
 * - vertex
 *
 *           4
 *        . / \ .
 *     .   /   \   .
 *   0 ---/-----\--- 3
 *   |   /       \   |
 *   |  /         \  |
 *   | /           \ |
 *   |/             \|
 *   1 --------------2
 *
 * The edges are
 * Idx    Edge
 * (0)    0 -> 1
 * (1)    1 -> 2
 * (2)    2 -> 3
 * (3)    3 -> 0
 * (4)    0 -> 4
 * (5)    1 -> 4
 * (6)    2 -> 4
 * (7)    3 -> 4
 *
 * The bounds are
 * Idx    Bound               Edges
 * (0)    0 -> 1 -> 4         0, 5, -4
 * (1)    1 -> 2 -> 4         1, 6, -5
 * (2)    2 -> 3 -> 4         2, 7, -6
 * (3)    0 -> 4 -> 3         4, -7, 3
 * (4)    0 -> 3 -> 2 -> 1    -3, -2, -1, -0
 */
class Pyramid {
public:
  static const int ndim = 3;
  static const int NVERTS = 5;
  static const int NFACES = 5;
  static const int NEDGES = 8;

  // Local information
  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];
  int32_t faces[NFACES];

  // Edge -> vert info
  static const int EDGE0_VERT0 = 0;
  static const int EDGE0_VERT1 = 1;

  static const int EDGE1_VERT0 = 1;
  static const int EDGE1_VERT1 = 2;

  static const int EDGE2_VERT0 = 2;
  static const int EDGE2_VERT1 = 3;

  static const int EDGE3_VERT0 = 3;
  static const int EDGE3_VERT1 = 0;

  static const int EDGE4_VERT0 = 0;
  static const int EDGE4_VERT1 = 4;

  static const int EDGE5_VERT0 = 1;
  static const int EDGE5_VERT1 = 4;

  static const int EDGE6_VERT0 = 2;
  static const int EDGE6_VERT1 = 4;

  static const int EDGE7_VERT0 = 3;
  static const int EDGE7_VERT1 = 4;

  // Face -> vert info
  static const int FACE0_VERT0 = 0;
  static const int FACE0_VERT1 = 1;
  static const int FACE0_VERT2 = 4;

  static const int FACE1_VERT0 = 1;
  static const int FACE1_VERT1 = 2;
  static const int FACE1_VERT2 = 4;

  static const int FACE2_VERT0 = 2;
  static const int FACE2_VERT1 = 3;
  static const int FACE2_VERT2 = 4;

  static const int FACE3_VERT0 = 0;
  static const int FACE3_VERT1 = 4;
  static const int FACE3_VERT2 = 3;

  static const int FACE4_VERT0 = 0;
  static const int FACE4_VERT1 = 3;
  static const int FACE4_VERT2 = 2;
  static const int FACE4_VERT3 = 1;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      *n0 = verts[EDGE0_VERT0];
      *n1 = verts[EDGE0_VERT1];
    } else if (edge == 1) {
      *n0 = verts[EDGE1_VERT0];
      *n1 = verts[EDGE1_VERT1];
    } else if (edge == 2) {
      *n0 = verts[EDGE2_VERT0];
      *n1 = verts[EDGE2_VERT1];
    } else if (edge == 3) {
      *n0 = verts[EDGE3_VERT0];
      *n1 = verts[EDGE3_VERT1];
    } else if (edge == 4) {
      *n0 = verts[EDGE4_VERT0];
      *n1 = verts[EDGE4_VERT1];
    } else if (edge == 5) {
      *n0 = verts[EDGE5_VERT0];
      *n1 = verts[EDGE5_VERT1];
    } else if (edge == 6) {
      *n0 = verts[EDGE6_VERT0];
      *n1 = verts[EDGE6_VERT1];
    } else if (edge == 7) {
      *n0 = verts[EDGE7_VERT0];
      *n1 = verts[EDGE7_VERT1];
    }
  }

  template <typename I> inline int get_face_verts(const int face, I f[]) {
    if (face == 0) {
      f[0] = verts[FACE0_VERT0];
      f[1] = verts[FACE0_VERT1];
      f[2] = verts[FACE0_VERT2];
      return Triangle::NVERTS;
    } else if (face == 1) {
      f[0] = verts[FACE1_VERT0];
      f[1] = verts[FACE1_VERT1];
      f[2] = verts[FACE1_VERT2];
      return Triangle::NVERTS;
    } else if (face == 2) {
      f[0] = verts[FACE2_VERT0];
      f[1] = verts[FACE2_VERT1];
      f[2] = verts[FACE2_VERT2];
      return Triangle::NVERTS;
    } else if (face == 3) {
      f[0] = verts[FACE3_VERT0];
      f[1] = verts[FACE3_VERT1];
      f[2] = verts[FACE3_VERT2];
      return Triangle::NVERTS;
    } else if (face == 4) {
      f[0] = verts[FACE4_VERT0];
      f[1] = verts[FACE4_VERT1];
      f[2] = verts[FACE4_VERT2];
      f[3] = verts[FACE4_VERT3];
      return Quadrilateral::NVERTS;
    }
    return 0;
  }
};
/**
 * @brief Wedge properties
 *
 * Wedge is a 3D element, its entities include:
 * - domain
 * - bound
 * - edge
 * - vertex
 *
 *        2 ------E8------ 5
 *      / |             / |
 *    E2  |          E5   |
 *    /   |           /   |
 *   0 ---|---E6------3   E4
 *    \   |           \   |
 *    E0  E1          E3  |
 *      \ |             \ |
 *        1 -----E7------- 4
 *
 * The edges are
 * Idx    Edge
 * (0)    0 -> 1
 * (1)    1 -> 2
 * (2)    2 -> 0
 * (3)    3 -> 4
 * (4)    4 -> 5
 * (5)    5 -> 3
 * (6)    0 -> 3
 * (7)    1 -> 4
 * (8)    2 -> 5
 *
 * The bounds are
 * Idx    Bound               Edges
 * (0)    0 -> 1 -> 2         0, 1, 2
 * (1)    3 -> 4 -> 5         3, 4, 5
 * (2)    0 -> 3 -> 4 -> 1    6, 3, -7, -0
 * (3)    1 -> 4 -> 5 -> 2    7, 4, -8, -1
 * (4)    0 -> 2 -> 5 -> 3    -2, 8, 5, -6
 */
class Wedge {
public:
  static const int ndim = 3;
  static const int NVERTS = 6;
  static const int NFACES = 5;
  static const int NEDGES = 9;

  // Local information
  int32_t index;
  int32_t verts[NVERTS];
  int32_t edges[NEDGES];
  int32_t faces[NFACES];

  // Edge -> vert info
  static const int EDGE0_VERT0 = 0;
  static const int EDGE0_VERT1 = 1;

  static const int EDGE1_VERT0 = 1;
  static const int EDGE1_VERT1 = 2;

  static const int EDGE2_VERT0 = 2;
  static const int EDGE2_VERT1 = 0;

  static const int EDGE3_VERT0 = 3;
  static const int EDGE3_VERT1 = 4;

  static const int EDGE4_VERT0 = 4;
  static const int EDGE4_VERT1 = 5;

  static const int EDGE5_VERT0 = 5;
  static const int EDGE5_VERT1 = 3;

  static const int EDGE6_VERT0 = 0;
  static const int EDGE6_VERT1 = 3;

  static const int EDGE7_VERT0 = 1;
  static const int EDGE7_VERT1 = 4;

  static const int EDGE8_VERT0 = 2;
  static const int EDGE8_VERT1 = 5;

  // Face -> verts
  static const int FACE0_VERT0 = 0;
  static const int FACE0_VERT1 = 1;
  static const int FACE0_VERT2 = 2;

  static const int FACE1_VERT0 = 3;
  static const int FACE1_VERT1 = 4;
  static const int FACE1_VERT2 = 5;

  static const int FACE2_VERT0 = 0;
  static const int FACE2_VERT1 = 3;
  static const int FACE2_VERT2 = 4;
  static const int FACE2_VERT3 = 1;

  static const int FACE3_VERT0 = 1;
  static const int FACE3_VERT1 = 4;
  static const int FACE3_VERT2 = 5;
  static const int FACE3_VERT3 = 2;

  static const int FACE4_VERT0 = 0;
  static const int FACE4_VERT1 = 2;
  static const int FACE4_VERT2 = 5;
  static const int FACE4_VERT3 = 3;

  template <typename I>
  inline void get_edge_verts(const int edge, I *n0, I *n1) {
    if (edge == 0) {
      *n0 = verts[EDGE0_VERT0];
      *n1 = verts[EDGE0_VERT1];
    } else if (edge == 1) {
      *n0 = verts[EDGE1_VERT0];
      *n1 = verts[EDGE1_VERT1];
    } else if (edge == 2) {
      *n0 = verts[EDGE2_VERT0];
      *n1 = verts[EDGE2_VERT1];
    } else if (edge == 3) {
      *n0 = verts[EDGE3_VERT0];
      *n1 = verts[EDGE3_VERT1];
    } else if (edge == 4) {
      *n0 = verts[EDGE4_VERT0];
      *n1 = verts[EDGE4_VERT1];
    } else if (edge == 5) {
      *n0 = verts[EDGE5_VERT0];
      *n1 = verts[EDGE5_VERT1];
    } else if (edge == 6) {
      *n0 = verts[EDGE6_VERT0];
      *n1 = verts[EDGE6_VERT1];
    } else if (edge == 7) {
      *n0 = verts[EDGE7_VERT0];
      *n1 = verts[EDGE7_VERT1];
    } else if (edge == 8) {
      *n0 = verts[EDGE8_VERT0];
      *n1 = verts[EDGE8_VERT1];
    }
  }

  template <typename I> inline int get_face_verts(const int face, I f[]) {
    if (face == 0) {
      f[0] = verts[FACE0_VERT0];
      f[1] = verts[FACE0_VERT1];
      f[2] = verts[FACE0_VERT2];
      return Triangle::NVERTS;
    } else if (face == 1) {
      f[0] = verts[FACE1_VERT0];
      f[1] = verts[FACE1_VERT1];
      f[2] = verts[FACE1_VERT2];
      return Triangle::NVERTS;
    } else if (face == 2) {
      f[0] = verts[FACE2_VERT0];
      f[1] = verts[FACE2_VERT1];
      f[2] = verts[FACE2_VERT2];
      f[3] = verts[FACE2_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 3) {
      f[0] = verts[FACE3_VERT0];
      f[1] = verts[FACE3_VERT1];
      f[2] = verts[FACE3_VERT2];
      f[3] = verts[FACE3_VERT3];
      return Quadrilateral::NVERTS;
    } else if (face == 4) {
      f[0] = verts[FACE4_VERT0];
      f[1] = verts[FACE4_VERT1];
      f[2] = verts[FACE4_VERT2];
      f[3] = verts[FACE4_VERT3];
      return Quadrilateral::NVERTS;
    }
    return 0;
  }
};

/**
 * @brief Connectivity class
 *
 */
class BasicConnectivity3D {
public:
  static constexpr int NO_LABEL = -1;

  // Structure to store the boundary information
  class BoundaryConnectivity {
  public:
    BoundaryConnectivity() {
      num_verts = 0;
      verts = nullptr;

      // 3D mesh information
      num_tris = 0;
      tris = nullptr;

      num_quads = 0;
      quads = nullptr;

      // Elements numbers and local element edge indices for elements on the
      // boundary for the triangles and quads, respectively
      tri_elements = nullptr;
      tri_face_indices = nullptr;

      quad_elements = nullptr;
      quad_face_indices = nullptr;
    }
    ~BoundaryConnectivity() {
      if (verts) {
        delete[] verts;
      }
      if (tris) {
        delete[] tris;
      }
      if (quads) {
        delete[] quads;
      }
      if (tri_elements) {
        delete[] tri_elements;
        delete[] tri_face_indices;
      }
      if (quad_elements) {
        delete[] quad_elements;
        delete[] quad_face_indices;
      }
    }

    // Initialize the boundary information based on what's stored locally
    void initialize_boundary(const BasicConnectivity3D *conn,
                             const int *vert_ptr, const int *vert_elems) {
      if (tri_elements) {
        delete[] tri_elements;
        delete[] tri_face_indices;
      }
      if (quad_elements) {
        delete[] quad_elements;
        delete[] quad_face_indices;
      }

      tri_elements = new int[num_tris];
      tri_face_indices = new int[num_tris];

      quad_elements = new int[num_quads];
      quad_face_indices = new int[num_quads];

      // Make sure the boundary tris are a match with the interior faces
      for (int j = 0; j < num_tris; j++) {
        bool tri_found = false;

        // Find elements connected via the first vert
        int n0 = tris[j].verts[0];
        for (int k = vert_ptr[n0]; k < vert_ptr[n0 + 1]; k++) {
          int elem = vert_elems[k];

          int nfaces = conn->get_element_num_faces(elem);
          for (int face = 0; face < nfaces; face++) {
            int face_verts[Quadrilateral::NVERTS];
            int nfv = conn->get_element_face_verts(elem, face, face_verts);

            if (nfv == Triangle::NVERTS) {
              // Check if all the verts on the boundary mach
              bool match = true;
              for (int ii = 0; ii < Triangle::NVERTS; ii++) {
                bool found = false;
                for (int jj = 0; jj < Triangle::NVERTS; jj++) {
                  if (tris[j].verts[ii] == face_verts[jj]) {
                    found = true;
                  }
                }
                if (!found) {
                  match = false;
                  break;
                }
              }

              // We've found a match for the boundary, set the triangle so
              // it shares the connectivity with the element
              if (match) {
                tri_elements[j] = elem;
                tri_face_indices[j] = face;
                tris[j].index = j;
                tris[j].verts[0] = face_verts[0];
                tris[j].verts[1] = face_verts[1];
                tris[j].verts[2] = face_verts[2];
                tri_found = true;
                break;
              }
            }
          }

          if (tri_found) {
            break;
          }
        }
      }

      // Find the quadrilaterals and match them
      // Make sure the boundary quads are a match with the interior faces
      for (int j = 0; j < num_quads; j++) {
        bool quad_found = false;

        // Find elements connected via the first vert
        int n0 = quads[j].verts[0];
        for (int k = vert_ptr[n0]; k < vert_ptr[n0 + 1]; k++) {
          int elem = vert_elems[k];

          int nfaces = conn->get_element_num_faces(elem);
          for (int face = 0; face < nfaces; face++) {
            int face_verts[Quadrilateral::NVERTS];
            int nfv = conn->get_element_face_verts(elem, face, face_verts);

            if (nfv == Quadrilateral::NVERTS) {
              // Check if all the verts on the boundary mach
              bool match = true;
              for (int ii = 0; ii < Quadrilateral::NVERTS; ii++) {
                bool found = false;
                for (int jj = 0; jj < Quadrilateral::NVERTS; jj++) {
                  if (quads[j].verts[ii] == face_verts[jj]) {
                    found = true;
                  }
                }
                if (!found) {
                  match = false;
                  break;
                }
              }

              // We've found a match for the boundary, set the triangle so
              // it shares the connectivity with the element
              if (match) {
                quad_elements[j] = elem;
                quad_face_indices[j] = face;
                quads[j].index = j;
                quads[j].verts[0] = face_verts[0];
                quads[j].verts[1] = face_verts[1];
                quads[j].verts[2] = face_verts[2];
                quads[j].verts[3] = face_verts[3];
                quad_found = true;
                break;
              }
            }
          }

          if (quad_found) {
            break;
          }
        }
      }
    }

    // Entities that are on the boundary
    int num_verts; // Number of vertices on the boundary
    int *verts;    // Vertex numbers

    // 3D mesh boundary information
    int num_tris;   // Triangles
    Triangle *tris; // Triangles on the boundary

    int num_quads;        // Quads
    Quadrilateral *quads; // Quadrilaterals on the boundary

    // Information required for viscous computations
    int *tri_elements;
    int *tri_face_indices;

    int *quad_elements;
    int *quad_face_indices;
  };

  /**
   * @brief Construct a basic connectivity object on a single processor.
   *
   * @param num_verts Number of vertices in the mesh
   * @param num_boundaries Number of boundaries in the mesh
   */
  BasicConnectivity3D(int num_verts = 0, int num_boundaries = 0)
      : num_verts(num_verts), num_boundaries(num_boundaries) {
    // Set the number of local elements
    num_tets = 0;
    num_hexs = 0;
    num_pyrds = 0;
    num_wedges = 0;

    // Space for the element information
    tets = nullptr;
    hexs = nullptr;
    pyrds = nullptr;
    wedges = nullptr;

    // Set the boundary information
    if (num_boundaries > 0) {
      boundary = new BoundaryConnectivity *[num_boundaries];
      for (int i = 0; i < num_boundaries; i++) {
        boundary[i] = nullptr;
      }
    } else {
      boundary = nullptr;
    }
  }
  ~BasicConnectivity3D() {
    if (tets) {
      delete[] tets;
    }
    if (hexs) {
      delete[] hexs;
    }
    if (pyrds) {
      delete[] pyrds;
    }
    if (wedges) {
      delete[] wedges;
    }

    for (int i = 0; i < num_boundaries; i++) {
      if (boundary[i]) {
        delete boundary[i];
      }
    }
  }

  /**
   * @brief Add a serial 3D mesh
   *
   * @param ntets Number of tetrahedron
   * @param tet_verts Verts for each tetrahedron
   * @param nhex Number of hexahedron
   * @param hex_verts Verts for each hexahedron
   * @param npyrd Number of pyramids
   * @param pyrd_verts Verts for each pyramid
   * @param nwedges Number of wedges
   * @param wedge_verts Verts for each wedge
   */
  template <typename I1, typename I2, typename I3, typename I4>
  void add_mesh(int ntets, const I1 tet_verts, int nhex, const I2 hex_verts,
                int npyrds, const I3 pyrd_verts, int nwedges,
                const I4 wedge_verts) {
    num_tets = ntets;
    num_hexs = nhex;
    num_pyrds = npyrds;
    num_wedges = nwedges;

    // Allocate space for the element -> vert connectivity
    tets = new Tetrahedron[num_tets];
    hexs = new Hexahedron[num_hexs];
    pyrds = new Pyramid[num_pyrds];
    wedges = new Wedge[num_wedges];

    // Set the connectivity: element -> verts
    int index = 0;
    for (int i = 0; i < num_tets; i++, index++) {
      tets[i].index = index;
      for (int j = 0; j < Tetrahedron::NVERTS; j++) {
        tets[i].verts[j] = tet_verts[Tetrahedron::NVERTS * i + j];
      }
    }
    for (int i = 0; i < num_hexs; i++, index++) {
      hexs[i].index = index;
      for (int j = 0; j < Hexahedron::NVERTS; j++) {
        hexs[i].verts[j] = hex_verts[Hexahedron::NVERTS * i + j];
      }
    }
    for (int i = 0; i < num_pyrds; i++, index++) {
      pyrds[i].index = index;
      for (int j = 0; j < Pyramid::NVERTS; j++) {
        pyrds[i].verts[j] = pyrd_verts[Pyramid::NVERTS * i + j];
      }
    }
    for (int i = 0; i < num_wedges; i++, index++) {
      wedges[i].index = index;
      for (int j = 0; j < Wedge::NVERTS; j++) {
        wedges[i].verts[j] = wedge_verts[Wedge::NVERTS * i + j];
      }
    }
  }

  /**
   * @brief Add a 2D mesh boundary
   *
   * Note that the boundary orientation is outward facing normals
   *
   * @param index Index of the boundary
   * @param nverts Number of vertices on the boundary
   * @param verts Array of length nverts of the vert numbers on the boundary
   * @param ntris Number of triangles on the boundary
   * @param tris Triangle verts for each triangle on the boundary
   * @param nquads Number of quads on the boundary
   * @param quads Quad verts for each quad on the boundary
   */
  template <typename I>
  void add_boundary(int index, int nverts, const I verts, int ntris,
                    const I tris, int nquads, const I quads) {
    if (index >= 0 && index < num_boundaries) {
      BoundaryConnectivity *b = new BoundaryConnectivity();

      // Set the data for the boundary
      b->num_verts = nverts;
      b->verts = new int[nverts];
      b->num_tris = ntris;
      b->tris = new Triangle[ntris];
      b->num_quads = nquads;
      b->quads = new Quadrilateral[nquads];

      for (int i = 0; i < nverts; i++) {
        b->verts[i] = verts[i];
      }
      for (int i = 0; i < ntris; i++) {
        for (int j = 0; j < Triangle::NVERTS; j++) {
          b->tris[i].verts[j] = tris[Triangle::NVERTS * i + j];
        }
      }
      for (int i = 0; i < nquads; i++) {
        for (int j = 0; j < Quadrilateral::NVERTS; j++) {
          b->quads[i].verts[j] = quads[Quadrilateral::NVERTS * i + j];
        }
      }

      boundary[index] = b;
    }
  }

  /**
   * @brief Initialize the edge and face numbers
   */
  void initialize() {
    // Initialize all of the edge info
    for (int i = 0; i < num_tets; i++) {
      for (int j = 0; j < Tetrahedron::NEDGES; j++) {
        tets[i].edges[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_hexs; i++) {
      for (int j = 0; j < Hexahedron::NEDGES; j++) {
        hexs[i].edges[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_pyrds; i++) {
      for (int j = 0; j < Pyramid::NEDGES; j++) {
        pyrds[i].edges[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_wedges; i++) {
      for (int j = 0; j < Wedge::NEDGES; j++) {
        wedges[i].edges[j] = NO_LABEL;
      }
    }

    // Initialize all of the face info
    for (int i = 0; i < num_tets; i++) {
      for (int j = 0; j < Tetrahedron::NFACES; j++) {
        tets[i].faces[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_hexs; i++) {
      for (int j = 0; j < Hexahedron::NFACES; j++) {
        hexs[i].faces[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_pyrds; i++) {
      for (int j = 0; j < Pyramid::NFACES; j++) {
        pyrds[i].faces[j] = NO_LABEL;
      }
    }
    for (int i = 0; i < num_wedges; i++) {
      for (int j = 0; j < Wedge::NFACES; j++) {
        wedges[i].faces[j] = NO_LABEL;
      }
    }

    int *vert_ptr, *vert_elems;
    init_vert_element_data(&vert_ptr, &vert_elems);

    init_edge_data(vert_ptr, vert_elems);
    init_face_data(vert_ptr, vert_elems);

    for (int i = 0; i < num_boundaries; i++) {
      boundary[i]->initialize_boundary(this, vert_ptr, vert_elems);
    }

    delete[] vert_ptr;
    delete[] vert_elems;
  }

  // Get basic information
  int get_num_vertices() const { return num_verts; }
  int get_num_boundaries() { return num_boundaries; }
  int get_num_edges() { return num_edges; }
  int get_num_faces() { return num_faces; }

  // Get the number of local elements
  int get_num_elements() const {
    return (num_tets + num_hexs + num_pyrds + num_wedges);
  }

  // Get the elements broken down by type
  int get_tets(const Tetrahedron **ts = nullptr) const {
    if (ts) {
      *ts = tets;
    }
    return num_tets;
  }
  int get_hexs(const Hexahedron **hx = nullptr) const {
    if (hx) {
      *hx = hexs;
    }
    return num_hexs;
  }
  int get_pyrds(const Pyramid **pys = nullptr) const {
    if (pys) {
      *pys = pyrds;
    }
    return num_pyrds;
  }
  int get_wedges(const Wedge **wdgs = nullptr) const {
    if (wdgs) {
      *wdgs = wedges;
    }
    return num_wedges;
  }

  // Get the boundary information
  int get_boundary_verts(int index, const int **nds) const {
    if (nds) {
      *nds = boundary[index]->verts;
    }
    return boundary[index]->num_verts;
  }
  int get_boundary_tris(int index, const Triangle **tris) const {
    if (tris) {
      *tris = boundary[index]->tris;
    }
    return boundary[index]->num_tris;
  }
  int get_boundary_quads(int index, const Quadrilateral **quads) const {
    if (quads) {
      *quads = boundary[index]->quads;
    }
    return boundary[index]->num_quads;
  }
  void get_boundary_tri_elements(int index, const int **elems,
                                 const int **indices) const {
    if (elems) {
      *elems = boundary[index]->tri_elements;
    }
    if (indices) {
      *indices = boundary[index]->tri_face_indices;
    }
  }
  void get_boundary_quad_elements(int index, const int **elems,
                                  const int **indices) const {
    if (elems) {
      *elems = boundary[index]->quad_elements;
    }
    if (indices) {
      *indices = boundary[index]->quad_face_indices;
    }
  }

  // Get the element type
  ElementType get_element_type(const int local_elem) const {
    int index = local_elem;
    if (index < num_tets) {
      return TETRAHEDRON;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      return HEXAHEDRON;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      return PYRAMID;
    }

    return WEDGE;
  }

  // Get the element verts based on a number of the local elements
  int get_element_verts(const int local_elem, const int **verts) const {
    int index = local_elem;
    if (index < num_tets) {
      *verts = tets[index].verts;
      return Tetrahedron::NVERTS;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      *verts = hexs[index].verts;
      return Hexahedron::NVERTS;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      *verts = pyrds[index].verts;
      return Pyramid::NVERTS;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      *verts = wedges[index].verts;
      return Wedge::NVERTS;
    }

    *verts = nullptr;
    return 0;
  }

  int get_element_edges(const int local_elem, const int **edgs) const {
    int index = local_elem;
    if (index < num_tets) {
      *edgs = tets[index].edges;
      return Tetrahedron::NEDGES;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      *edgs = hexs[index].edges;
      return Hexahedron::NEDGES;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      *edgs = pyrds[index].edges;
      return Pyramid::NEDGES;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      *edgs = wedges[index].edges;
      return Wedge::NEDGES;
    }

    *edgs = nullptr;
    return 0;
  }

  int get_element_faces(const int local_elem, const int **faces) const {
    int index = local_elem;
    if (index < num_tets) {
      *faces = tets[index].faces;
      return Tetrahedron::NFACES;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      *faces = hexs[index].faces;
      return Hexahedron::NFACES;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      *faces = pyrds[index].faces;
      return Pyramid::NFACES;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      *faces = wedges[index].faces;
      return Wedge::NFACES;
    }

    *faces = nullptr;
    return 0;
  }

  void get_element_edge_verts(const int local_elem, const int edge, int *n0,
                              int *n1) const {
    *n0 = NO_LABEL;
    *n1 = NO_LABEL;

    int index = local_elem;
    if (index < num_tets) {
      tets[index].get_edge_verts(edge, n0, n1);
      return;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      hexs[index].get_edge_verts(edge, n0, n1);
      return;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      pyrds[index].get_edge_verts(edge, n0, n1);
      return;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      wedges[index].get_edge_verts(edge, n0, n1);
      return;
    }
  }

  // Get the number of faces or edges associated with an element
  int get_element_num_faces(const int elem) const {
    int index = elem;
    if (index < num_tets) {
      return Tetrahedron::NFACES;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      return Hexahedron::NFACES;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      return Pyramid::NFACES;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      return Wedge::NFACES;
    }

    return 0;
  }

  int get_element_face_verts(const int elem, int face, int *verts) const {
    int index = elem;
    if (index < num_tets) {
      return tets[index].get_face_verts(face, verts);
    }

    index = index - num_tets;
    if (index < num_hexs) {
      return hexs[index].get_face_verts(face, verts);
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      return pyrds[index].get_face_verts(face, verts);
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      return wedges[index].get_face_verts(face, verts);
    }

    return 0;
  }

private:
  /**
   * @brief Compute an element to element connectivity
   *
   * This is used to partition and order the mesh
   *
   * @param rowp0 Row pointer
   * @param cols0 Column indices
   * @param include_diag Flag to indicate whether to include the diagonal
   */
  void create_element_csr(int **rowp0, int **cols0,
                          bool include_diag = true) const {
    int *vert_ptr, *vert_elems;
    init_vert_element_data(&vert_ptr, &vert_elems);

    int *rowp = new int[num_verts + 1];
    std::fill(rowp, rowp + num_verts + 1, 0);

    int *marker = new int[num_verts];
    std::fill(marker, marker + num_verts, -1);

    for (int i = 0; i < num_verts; i++) {
      // Always include the diagonal element (this would get added anyway)
      if (include_diag) {
        marker[i] = i;
        rowp[i + 1] = 1;
      } else {
        rowp[i + 1] = 0;
      }

      for (int jp = vert_ptr[i]; jp < vert_ptr[i + 1]; jp++) {
        int elem = vert_elems[jp];

        const int *verts;
        int nverts = get_element_verts(elem, &verts);

        for (int k = 0; k < nverts; k++) {
          if (verts[k] != i && marker[verts[k]] != i) {
            marker[verts[k]] = i;
            rowp[i + 1]++;
          }
        }
      }
    }

    for (int i = 0; i < num_verts; i++) {
      rowp[i + 1] += rowp[i];
    }
    int *cols = new int[rowp[num_verts]];

    std::fill(marker, marker + num_verts, -1);
    for (int i = 0; i < num_verts; i++) {
      int *col_index = &cols[rowp[i]];
      if (include_diag) {
        marker[i] = i;
        col_index[0] = i;
        col_index++;
      }

      for (int jp = vert_ptr[i]; jp < vert_ptr[i + 1]; jp++) {
        int elem = vert_elems[jp];

        const int *verts;
        int nverts = get_element_verts(elem, &verts);

        for (int k = 0; k < nverts; k++) {
          if (verts[k] != i && marker[verts[k]] != i) {
            marker[verts[k]] = i;
            col_index[0] = verts[k];
            col_index++;
          }
        }
      }
    }

    // Free the data that's not needed anymore
    delete[] marker;
    delete[] vert_ptr;
    delete[] vert_elems;

    *rowp0 = rowp;
    *cols0 = cols;
  }

  // Convenience private non-const-qualified version
  int get_element_edges(const int local_elem, int **edgs) {
    int index = local_elem;
    if (index < num_tets) {
      *edgs = tets[index].edges;
      return Tetrahedron::NEDGES;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      *edgs = hexs[index].edges;
      return Hexahedron::NEDGES;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      *edgs = pyrds[index].edges;
      return Pyramid::NEDGES;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      *edgs = wedges[index].edges;
      return Wedge::NEDGES;
    }

    *edgs = nullptr;
    return 0;
  }

  // Convenience private non-const-qualified version
  int get_element_faces(const int local_elem, int **faces) {
    int index = local_elem;
    if (index < num_tets) {
      *faces = tets[index].faces;
      return Tetrahedron::NFACES;
    }

    index = index - num_tets;
    if (index < num_hexs) {
      *faces = hexs[index].faces;
      return Hexahedron::NFACES;
    }

    index = index - num_hexs;
    if (index < num_pyrds) {
      *faces = pyrds[index].faces;
      return Pyramid::NFACES;
    }

    index = index - num_pyrds;
    if (index < num_wedges) {
      *faces = wedges[index].faces;
      return Wedge::NFACES;
    }

    *faces = nullptr;
    return 0;
  }

  /**
   * @brief Order the edges uniquely across the entire mesh
   *
   * This code loops over all the elements and their edges. Edges that have not
   * been ordered are given a edge number. All adjacent edges are located and
   * given the same edge number.
   *
   * @param vert_ptr The CSR pointer created by init_vert_element_data()
   * @param vert_elems The CSR element indicies created by
   * init_vert_element_data()
   */
  void init_edge_data(const int *vert_ptr, const int *vert_elems) {
    num_edges = 0;

    int num_elems = get_num_elements();
    for (int elem = 0; elem < num_elems; elem++) {
      // Loop over the element edges
      int *elem_edges;
      int ne = get_element_edges(elem, &elem_edges);

      // Loop over the element edges and check if any are undefined
      for (int edge = 0; edge < ne; edge++) {
        // Edge j is not ordered
        if (elem_edges[edge] == NO_LABEL) {
          // Set the element edge number
          elem_edges[edge] = num_edges;
          num_edges++;

          // Find the edges in all of the adjacent elements
          int n0, n1; // Get the verts for the element
          get_element_edge_verts(elem, edge, &n0, &n1);

          // Find the elements that are adjacent to nj0
          for (int k = vert_ptr[n0]; k < vert_ptr[n0 + 1]; k++) {
            // Element p may share an edge with element i
            int adj_elem = vert_elems[k];
            if (adj_elem != elem) {
              int *adj_elem_edges;
              int adj_ne = get_element_edges(adj_elem, &adj_elem_edges);

              // Loop to find the matching edge
              for (int adj_edge = 0; adj_edge < adj_ne; adj_edge++) {
                int adj_n0, adj_n1;
                get_element_edge_verts(adj_elem, adj_edge, &adj_n0, &adj_n1);

                // Adjacent edge matches
                if ((n0 == adj_n0 && n1 == adj_n1) ||
                    (n1 == adj_n0 && n0 == adj_n1)) {
                  adj_elem_edges[adj_edge] = elem_edges[edge];
                }
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief Order the faces uniquely across the entire mesh
   *
   * This code loops over all the elements and faces. Faces that have not been
   * ordered are given a face number and any matching faces in adjacent elements
   * are located and given the same number.
   *
   * @param vert_ptr The CSR pointer created by init_vert_element_data()
   * @param vert_elems The CSR element indicies created by
   * init_vert_element_data()
   */
  void init_face_data(const int *vert_ptr, const int *vert_elems) {
    num_faces = 0;

    int num_elems = get_num_elements();
    for (int elem = 0; elem < num_elems; elem++) {
      // Loop over the element faces
      int *elem_faces;
      int nf = get_element_faces(elem, &elem_faces);

      // Loop over the element faces and check if any are undefined
      for (int face = 0; face < nf; face++) {
        // Edge j is not ordered
        if (elem_faces[face] == NO_LABEL) {
          // Set the element face number
          elem_faces[face] = num_faces;
          num_faces++;

          // Find the faces in all of the adjacent elements
          int face_verts[Quadrilateral::NVERTS];
          int nfv = get_element_face_verts(elem, face, face_verts);

          // Find the elements that are adjacent to nj0
          int n0 = face_verts[0];
          for (int k = vert_ptr[n0]; k < vert_ptr[n0 + 1]; k++) {
            // Element may share a face with the element
            int adj_elem = vert_elems[k];
            if (adj_elem != elem) {
              int *adj_elem_faces;
              int adj_nf = get_element_faces(adj_elem, &adj_elem_faces);

              // Loop to find the matching face
              for (int adj_face = 0; adj_face < adj_nf; adj_face++) {
                int adj_face_verts[Quadrilateral::NVERTS];
                int adj_nfv =
                    get_element_face_verts(adj_elem, adj_face, adj_face_verts);

                // Adjacent faces match
                if (nfv == adj_nfv) {
                  if (nfv == Triangle::NVERTS &&
                      Triangle::is_flipped(face_verts, adj_face_verts)) {
                    adj_elem_faces[adj_face] = elem_faces[face];
                  } else if (nfv == Quadrilateral::NVERTS &&
                             Quadrilateral::is_flipped(face_verts,
                                                       adj_face_verts)) {
                    adj_elem_faces[adj_face] = elem_faces[face];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief Initialize a CSR-type data structure that points from vertices to
   * elements
   *
   * The array vert_elems[j] from j = vert_ptr[i] to j = vert_ptr[i + 1] - 1
   * stores the element indices j that reference vertex i.
   *
   * vert_ptr has length num_verts + 1.
   *
   * @param vert_ptr Pointer into the vert_elems array of length num_verts + 1
   * @param vert_elems Elements that contain the specified vertex
   */
  void init_vert_element_data(int **vert_ptr, int **vert_elems) const {
    int *vert_element_ptr = new int[num_verts + 1];
    std::fill(vert_element_ptr, vert_element_ptr + num_verts + 1, 0);

    int num_elems = get_num_elements();
    for (int elem = 0; elem < num_elems; elem++) {
      const int *verts;
      int n = get_element_verts(elem, &verts);
      for (int i = 0; i < n; i++) {
        vert_element_ptr[verts[i] + 1]++;
      }
    }

    for (int i = 0; i < num_verts; i++) {
      vert_element_ptr[i + 1] += vert_element_ptr[i];
    }

    int *vert_elements = new int[vert_element_ptr[num_verts]];

    for (int elem = 0; elem < num_elems; elem++) {
      const int *verts;
      int n = get_element_verts(elem, &verts);
      for (int i = 0; i < n; i++) {
        vert_elements[vert_element_ptr[verts[i]]] = elem;
        vert_element_ptr[verts[i]]++;
      }
    }

    for (int i = num_verts; i > 0; i--) {
      vert_element_ptr[i] = vert_element_ptr[i - 1];
    }
    vert_element_ptr[0] = 0;

    *vert_ptr = vert_element_ptr;
    *vert_elems = vert_elements;
  }

  // Input information about the mesh
  int num_verts; // Number of verts

  // Number of each type of element
  int num_tets;
  int num_hexs;
  int num_pyrds;
  int num_wedges;

  // Element data for each element type.
  Tetrahedron *tets;
  Hexahedron *hexs;
  Pyramid *pyrds;
  Wedge *wedges;

  // Number of edges and number of faces
  int num_edges;
  int num_faces;

  // Boundary connectivity information
  int num_boundaries;              // Number of boundaries in the mesh
  BoundaryConnectivity **boundary; // Boundary objects
};

#endif // BASIC_CONNECTIVITY_H
