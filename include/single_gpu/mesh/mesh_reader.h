#ifndef MESH_READER_H
#define MESH_READER_H

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "basic_connectivity.h"
#include "utils.h"

template <typename T>
class SU2MeshReader {
 public:
  /**
   * @brief Parse the SU2 file
   *
   * Parse and store the data contained within an SU2 file. The elements, points
   * and boundary markers are stored in the object. This object is not MPI aware
   * so this should be created on a single proc.
   *
   * @param filename The filename to parse
   */
  SU2MeshReader(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open mesh file");
    }

    std::string line;
    while (getline(file, line)) {
      if (line.find("NDIME=") != std::string::npos) {
        dimension = std::stoi(line.substr(line.find('=') + 1));
      } else if (line.find("NPOIN=") != std::string::npos) {
        parse_points(file, std::stoi(line.substr(line.find('=') + 1)));
      } else if (line.find("NELEM=") != std::string::npos) {
        parse_elements(file, std::stoi(line.substr(line.find('=') + 1)));
      } else if (line.find("NMARK=") != std::string::npos) {
        parse_boundaries(file, std::stoi(line.substr(line.find('=') + 1)));
      }
    }
  }

  /**
   * @brief Write a summary of the file contents, including boundary names.
   *
   * @param out Output stream
   */
  void write_summary(std::ostream &out = std::cout) const {
    out << "Dimension        " << dimension << std::endl;
    out << "Number of nodes  " << points.size() / dimension << std::endl;

    if (dimension == 2) {
      out << "Number of tris   " << tris.size() / Triangle::NVERTS << std::endl;
      out << "Number of quads  " << quads.size() / Quadrilateral::NVERTS
          << std::endl;
    } else if (dimension == 3) {
      out << "Number of tets   " << tets.size() / Tetrahedron::NVERTS
          << std::endl;
      out << "Number of hexs   " << hexs.size() / Hexahedron::NVERTS
          << std::endl;
      out << "Number of pyrds  " << pyrmids.size() / Pyramid::NVERTS
          << std::endl;
      out << "Number of wedges " << wedges.size() / Wedge::NVERTS << std::endl;
    }

    for (int index = 0; index < boundaries.size(); index++) {
      BoundaryInfo info = boundaries[index];
      out << "Boundary number " << info.number << std::endl;
      out << "Boundary tag    " << info.tag << std::endl;

      if (info.dimension == 2) {
        out << "Number of lines " << info.lines.size() / 2 << std::endl;
      } else if (info.dimension == 3) {
        out << "Number of tris  " << info.tris.size() / Triangle::NVERTS
            << std::endl;
        out << "Number of quads " << info.quads.size() / Quadrilateral::NVERTS
            << std::endl;
      }
    }
  }

  /**
   * @brief Get the dimension of the problem (2D or 3D)
   *
   * @return int Problem dimension
   */
  int get_dimension() const { return dimension; }

  /**
   * @brief Get the number of points
   *
   * @return int Number of points
   */
  int get_num_points() const { return points.size() / dimension; }

  /**
   * @brief Get the x, y, z points of each node as a raw pointer
   *
   * @return const T* Pointer to the x, y, z coordinate of each point
   */
  const T *get_points() const { return points.data(); }

  /**
   * @brief Create a 3D connectivity object
   *
   * For this function to work problem dimension == 3
   *
   * @return Connectivity3D* The connectivity object
   */
  BasicConnectivity3D *create_connectivity() {
    int num_nodes = points.size() / dimension;
    int num_boundaries = boundaries.size();
    BasicConnectivity3D *conn =
        new BasicConnectivity3D(num_nodes, num_boundaries);

    conn->add_mesh(tets.size() / Tetrahedron::NVERTS, tets,
                   hexs.size() / Hexahedron::NVERTS, hexs,
                   pyrmids.size() / Pyramid::NVERTS, pyrmids,
                   wedges.size() / Wedge::NVERTS, wedges);

    for (int i = 0; i < num_boundaries; i++) {
      conn->add_boundary(i, boundaries[i].nodes.size(), boundaries[i].nodes,
                         boundaries[i].tris.size() / Triangle::NVERTS,
                         boundaries[i].tris,
                         boundaries[i].quads.size() / Quadrilateral::NVERTS,
                         boundaries[i].quads);
    }

    conn->initialize();

    return conn;
  }

  /**
   * @brief Get the boundary names
   *
   * @param boundary_names Vector of the boundary names
   */
  void get_boundary_names(std::vector<std::string> &boundary_names) const {
    for (int i = 0; i < boundaries.size(); i++) {
      boundary_names.push_back(boundaries[i].tag);
    }
  }

 private:
  // The dimension of the problem
  int dimension;

  // Points
  std::vector<T> points;

  // Only for 2D meshes
  std::vector<int> tris;
  std::vector<int> quads;

  // Only for 3D meshes
  std::vector<int> tets;
  std::vector<int> hexs;
  std::vector<int> pyrmids;
  std::vector<int> wedges;

  struct BoundaryInfo {
    int dimension;
    int number;
    std::string tag;
    std::vector<int> nodes;
    std::vector<int> lines;
    std::vector<int> tris;
    std::vector<int> quads;
  };

  // Boundary information
  std::vector<BoundaryInfo> boundaries;

  // Parse the points
  void parse_points(std::ifstream &file, int num_points) {
    points.reserve(dimension * num_points);

    if (dimension == 2) {
      std::string line;
      for (int i = 0; i < num_points; i++) {
        if (getline(file, line)) {
          T x, y;
          std::istringstream input(line);
          input >> x >> y;
          points.push_back(x);
          points.push_back(y);
        }
      }
    } else if (dimension == 3) {
      std::string line;
      for (int i = 0; i < num_points; i++) {
        if (getline(file, line)) {
          T x, y, z;
          std::istringstream input(line);
          input >> x >> y >> z;
          points.push_back(x);
          points.push_back(y);
          points.push_back(z);
        }
      }
    }
  }

  // Parse the elements
  void parse_elements(std::ifstream &file, int num_elements) {
    if (dimension == 2) {
      tris.reserve(Triangle::NVERTS * num_elements);
      quads.reserve(Quadrilateral::NVERTS * num_elements);
    } else {
      tets.reserve(Tetrahedron::NVERTS * num_elements);
      hexs.reserve(Hexahedron::NVERTS * num_elements);
      pyrmids.reserve(Pyramid::NVERTS * num_elements);
      wedges.reserve(Wedge::NVERTS * num_elements);
    }

    for (int i = 0; i < num_elements; i++) {
      std::string line;
      if (getline(file, line)) {
        int vtk_type;
        std::istringstream input(line);
        input >> vtk_type;

        if (vtk_type == VTKInfo::VTK_TRIANGLE) {
          for (int j = 0; j < Triangle::NVERTS; j++) {
            int index;
            input >> index;
            tris.push_back(index);
          }
        } else if (vtk_type == VTKInfo::VTK_QUADRILATERAL) {
          for (int j = 0; j < Quadrilateral::NVERTS; j++) {
            int index;
            input >> index;
            quads.push_back(index);
          }
        } else if (vtk_type == VTKInfo::VTK_TETRAHEDRON) {
          for (int j = 0; j < Tetrahedron::NVERTS; j++) {
            int index;
            input >> index;
            tets.push_back(index);
          }
        } else if (vtk_type == VTKInfo::VTK_HEXAHEDRAL) {
          for (int j = 0; j < Hexahedron::NVERTS; j++) {
            int index;
            input >> index;
            hexs.push_back(index);
          }
        } else if (vtk_type == VTKInfo::VTK_PRISM) {
          for (int j = 0; j < Wedge::NVERTS; j++) {
            int index;
            input >> index;
            wedges.push_back(index);
          }
        } else if (vtk_type == VTKInfo::VTK_PYRAMID) {
          for (int j = 0; j < Pyramid::NVERTS; j++) {
            int index;
            input >> index;
            pyrmids.push_back(index);
          }
        }
      }
    }
  }

  // Function to trim whitespace from both ends of a string
  std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    if (first == std::string::npos || last == std::string::npos)
      return "";  // String is either empty or all whitespace
    return str.substr(first, (last - first + 1));
  }

  // Parse the boundaries
  void parse_boundaries(std::ifstream &file, int num_boundaries) {
    for (int index = 0; index < num_boundaries; index++) {
      BoundaryInfo info;
      info.dimension = dimension;
      info.number = index;

      std::string line;
      getline(file, line);  // Skip MARKER_TAG line
      info.tag = trim(line.substr(line.find('=') + 1));

      getline(file, line);  // Skip MARKER_ELEMS line
      int num_elements = std::stoi(line.substr(line.find('=') + 1));

      std::unordered_set<int> boundary_nodes;

      for (int k = 0; k < num_elements; k++) {
        if (getline(file, line)) {
          int vtk_type;
          std::istringstream input(line);
          input >> vtk_type;

          if (vtk_type == VTKInfo::VTK_LINE) {
            for (int j = 0; j < 2; j++) {
              int index;
              input >> index;
              info.lines.push_back(index);
              boundary_nodes.insert(index);
            }
          } else if (vtk_type == VTKInfo::VTK_TRIANGLE) {
            for (int j = 0; j < Triangle::NVERTS; j++) {
              int index;
              input >> index;
              info.tris.push_back(index);
              boundary_nodes.insert(index);
            }
          } else if (vtk_type == VTKInfo::VTK_QUADRILATERAL) {
            for (int j = 0; j < Quadrilateral::NVERTS; j++) {
              int index;
              input >> index;
              info.quads.push_back(index);
              boundary_nodes.insert(index);
            }
          }
        }
      }

      // Create the list of unique boundary nodes
      info.nodes.insert(info.nodes.end(), boundary_nodes.begin(),
                        boundary_nodes.end());

      boundaries.push_back(info);
    }
  }
};

#endif  // MESH_READER_H
