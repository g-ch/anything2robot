#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include "igl/readSTL.h"
// #include "igl/writeOFF.h"
// #include <igl/signed_distance.h>
// #include <igl/per_vertex_normals.h>
#include <vector>

#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>


bool is_watertight(const std::string& file_path) {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
    typedef CGAL::Surface_mesh<K::Point_3> Mesh;
    Mesh mesh;

    std::ifstream input(file_path);
    if (!input || !(input >> mesh)) {
        std::cerr << "Cannot open " << file_path << std::endl;
        return false;
    }

    bool is_closed = CGAL::is_closed(mesh);
    bool is_valid = CGAL::is_valid_polygon_mesh(mesh);

    return is_closed && is_valid;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> " << std::endl;
        return 1;
    }

    std::string path = argv[1];

    // Check the surffix of the input file.
    std::string suffix = path.substr(path.find_last_of(".") + 1);
    std::cout << "The suffix of the input file is: " << suffix << std::endl;
    std::cout << "The accepted suffix is: off" << std::endl;

    if (suffix != "off") {
        std::cerr << "The input file should be in STL or OFF format." << std::endl;
        return 1;
    }
    
    // Check if the output mesh is watertight
    if (is_watertight(path)) {
        std::cout << "The mesh is watertight." << std::endl;
    } else {
        std::cout << "The mesh is not watertight." << std::endl;
    }
    
    return 0;
}



