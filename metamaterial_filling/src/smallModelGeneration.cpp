#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "igl/readSTL.h"
#include "igl/writeSTL.h"
#include <igl/signed_distance.h>
#include <igl/per_vertex_normals.h>
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
// #include <CGAL/IO/OFF_reader.h>

// default triangulation for Surface_mesher
typedef CGAL::Surface_mesh_default_triangulation_3 Tr;

// c2t3
typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
typedef Tr::Geom_traits GT;
typedef GT::Sphere_3 Sphere_3;
typedef GT::Point_3 Point_3;
typedef GT::FT FT;
typedef FT (*Function)(Point_3);
typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;


// Define the mesh variables
Eigen::MatrixXd V;  // Vertex coordinates
Eigen::MatrixXi F;  // Face indices

double x_min, x_max, y_min, y_max, z_min, z_max;

double thickness = 0.1;

// Define the grid for the SDF approximation
Eigen::VectorXd sdfGrid;
Eigen::MatrixXd grid;
int gridResolution = 200; //400


/// @brief Calculate the signed distances of a set of points to a mesh
/// @param[in] points The points to calculate the signed distances for
/// @param[out] distances The signed distances of the points
void calculateSignedDistances(const Eigen::MatrixXd& points, Eigen::VectorXd &distances)
{
    Eigen::VectorXi I;
    Eigen::MatrixXd C, N;
    // Choose type of signing to use
    igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
    igl::signed_distance(points, V, F, type, distances, I,C,N);
}


/// @brief Create a voxel grid and calculate the signed distances for all points
/// @param[in] gridResolution The resolution of the grid in each dimension
void createVoxelGridSDF(int gridResolution) {
    // Compute the step size in each dimension
    double x_step = (x_max - x_min) / (gridResolution - 1);
    double y_step = (y_max - y_min) / (gridResolution - 1);
    double z_step = (z_max - z_min) / (gridResolution - 1);
    
    // Resize the grid matrix and the SDF grid vector
    grid.resize(gridResolution * gridResolution * gridResolution, 3);
    sdfGrid.resize(gridResolution * gridResolution * gridResolution);
    
    // Fill the grid with points and calculate SDF values
    int idx = 0;
    for (int i = 0; i < gridResolution; ++i) {
        for (int j = 0; j < gridResolution; ++j) {
            for (int k = 0; k < gridResolution; ++k) {
                grid(idx, 0) = x_min + i * x_step;
                grid(idx, 1) = y_min + j * y_step;
                grid(idx, 2) = z_min + k * z_step;
                ++idx;
            }
        }
    }
    
    // Calculate SDF for all grid points
    calculateSignedDistances(grid, sdfGrid);
}


/// @brief Perform trilinear interpolation to approximate the SDF at a query point
/// @param[in] queryPoint The point to query the SDF at
double trilinearInterpolation(const Eigen::Vector3d &queryPoint) {
    static const double x_step = (x_max - x_min) / (gridResolution - 1);
    static const double y_step = (y_max - y_min) / (gridResolution - 1);
    static const double z_step = (z_max - z_min) / (gridResolution - 1);
    
    // Compute the voxel coordinates
    Eigen::Vector3d voxelCoord = (queryPoint.array() - Eigen::Vector3d(x_min, y_min, z_min).array()) / Eigen::Vector3d(x_step, y_step, z_step).array();
    int x0 = floor(voxelCoord.x());
    int y0 = floor(voxelCoord.y());
    int z0 = floor(voxelCoord.z());
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    
    // Ensure the coordinates are within the grid bounds
    x0 = std::max(0, std::min(x0, gridResolution - 1));
    y0 = std::max(0, std::min(y0, gridResolution - 1));
    z0 = std::max(0, std::min(z0, gridResolution - 1));
    x1 = std::max(0, std::min(x1, gridResolution - 1));
    y1 = std::max(0, std::min(y1, gridResolution - 1));
    z1 = std::max(0, std::min(z1, gridResolution - 1));
    
    // Compute interpolation weights
    double xd = (queryPoint.x() - (x_min + x0 * x_step)) / x_step;
    double yd = (queryPoint.y() - (y_min + y0 * y_step)) / y_step;
    double zd = (queryPoint.z() - (z_min + z0 * z_step)) / z_step;
    
    // Get the SDF values at the corners of the voxel
    int idx000 = x0 * gridResolution * gridResolution + y0 * gridResolution + z0;
    int idx100 = x1 * gridResolution * gridResolution + y0 * gridResolution + z0;
    int idx010 = x0 * gridResolution * gridResolution + y1 * gridResolution + z0;
    int idx110 = x1 * gridResolution * gridResolution + y1 * gridResolution + z0;
    int idx001 = x0 * gridResolution * gridResolution + y0 * gridResolution + z1;
    int idx101 = x1 * gridResolution * gridResolution + y0 * gridResolution + z1;
    int idx011 = x0 * gridResolution * gridResolution + y1 * gridResolution + z1;
    int idx111 = x1 * gridResolution * gridResolution + y1 * gridResolution + z1;
    
    double c000 = sdfGrid(idx000);
    double c100 = sdfGrid(idx100);
    double c010 = sdfGrid(idx010);
    double c110 = sdfGrid(idx110);
    double c001 = sdfGrid(idx001);
    double c101 = sdfGrid(idx101);
    double c011 = sdfGrid(idx011);
    double c111 = sdfGrid(idx111);
    
    // Perform trilinear interpolation
    double c00 = c000 * (1 - xd) + c100 * xd;
    double c10 = c010 * (1 - xd) + c110 * xd;
    double c01 = c001 * (1 - xd) + c101 * xd;
    double c11 = c011 * (1 - xd) + c111 * xd;
    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;
    double c = c0 * (1 - zd) + c1 * zd;
    
    return c;
}

/// @brief Compute the signed distance function for the shell
/// @param[in] p The point to compute the SDF for
FT sdfFunction(Point_3 p) {
    
    Eigen::MatrixXd points(1, 3);
    points << p.x(), p.y(), p.z();

    Eigen::VectorXd distances;

    calculateSignedDistances(points, distances);

    double distance = distances(0) + thickness; // Thickness of the shell used to bias the SDF

    return distance;
}


/// @brief Compute the signed distance function for the shell using the trilinear interpolation. Faster than the previous method but not as accurate
/// @param[in] p The point to compute the SDF for
FT sdfFunctionTrilinear(Point_3 p) {
    Eigen::Vector3d queryPoint(p.x(), p.y(), p.z());
    double distance = trilinearInterpolation(queryPoint) + thickness; // Thickness of the shell used to bias the SDF
    return distance;
}

/// @brief Mesh the implicit surface using the given SDF function
/// @param[in] bbox_out_shpere_radius The radius of the bounding sphere
/// @param[in] error_bound The error bound for the meshing
/// @param[in] out_path The path to save the output mesh
void meshWithVoxels(double bbox_out_shpere_radius = 0.5, double error_bound = 1e-3, std::string out_path = "output.off") {
    Tr tr;            // 3D-Delaunay triangulation
    C2t3 c2t3 (tr);   // 2D-complex in 3D-Delaunay triangulation

    // Surface_3 surface(sdfFunction, Sphere_3(CGAL::ORIGIN, bbox_out_shpere_radius * bbox_out_shpere_radius), error_bound);
    Surface_3 surface(sdfFunctionTrilinear, Sphere_3(CGAL::ORIGIN, bbox_out_shpere_radius * bbox_out_shpere_radius), error_bound);

    // defining meshing criteria
    CGAL::Surface_mesh_default_criteria_3<Tr> criteria(30.0,  // angular bound
                                                        error_bound,  // radius bound. Use the same as the error bound for simplicity
                                                        error_bound); // distance bound. Use the same as the error bound for simplicity
    // meshing surface
    CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Manifold_tag()); // Manifold_tag() is used to ensure that the output is a closed surface

    Surface_mesh sm;
    CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, sm);

    // Write the modified mesh to a new off file

    // Check the surfix of the output file. If it is not .off, change it to .off
    std::string surfix = out_path.substr(out_path.size() - 4, 4);
    if (surfix != ".off") {
        out_path = out_path.substr(0, out_path.size() - 4) + ".off";
    }

    std::ofstream out_file(out_path);
    out_file << sm << std::endl;

    std::cout << "Mesh processed and saved to: " << out_path << std::endl;
    std::cout << "Final number of points: " << sm.number_of_vertices() << std::endl;
}



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
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> <output-off-file> <shell thickness> <error_bound_for_marching_cubes>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::string out_path = argv[2];
    thickness = std::stod(argv[3]);
    double error_bound_for_marching_cubes = std::stod(argv[4]);

    // Show the input arguments
    std::cout << "Input STL file: " << path << std::endl;
    std::cout << "Output OFF file: " << out_path << std::endl;
    std::cout << "Shell thickness: " << thickness << std::endl;
    std::cout << "error_bound_for_marching_cubes: " << error_bound_for_marching_cubes << std::endl;

    // Load the mesh first with igl
    Eigen::MatrixXd norms;  // Face normals

    std::ifstream in_file(path);
    // Read the STL file
    if (!igl::readSTL(in_file, V, F, norms)) {
        std::cerr << "Failed to load STL file: " << path << std::endl;
        return -1;
    }

    // Find min, max coordinates
    Eigen::RowVector3d minCoords = V.colwise().minCoeff();
    Eigen::RowVector3d maxCoords = V.colwise().maxCoeff();

    x_min = minCoords(0), x_max = maxCoords(0);
    y_min = minCoords(1), y_max = maxCoords(1);
    z_min = minCoords(2), z_max = maxCoords(2);

    // Create a grid of points to sample the signed distance field and approximate the SDF for acceleration
    std::cout << "Creating voxel grid and calculating SDF..." << std::endl;
    createVoxelGridSDF(gridResolution);

    // Now do the meshing and save the result
    double bbox_out_shpere_radius = std::max({x_max - x_min, y_max - y_min, z_max - z_min}) / 2.0;

    std::cout << "Bounding box radius: " << bbox_out_shpere_radius << std::endl;
    std::cout << "Doing meshing with resolution: " << error_bound_for_marching_cubes << std::endl;

    meshWithVoxels(bbox_out_shpere_radius, error_bound_for_marching_cubes, out_path);

    // Check if the output mesh is watertight
    // if (is_watertight(out_path)) {
    //     std::cout << "The mesh is watertight." << std::endl;
    // } else {
    //     std::cout << "The mesh is not watertight." << std::endl;
    // }


    return 0;

}



