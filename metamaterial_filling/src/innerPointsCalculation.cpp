#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "igl/readSTL.h"
#include "igl/writeSTL.h"
#include <igl/signed_distance.h>
#include <igl/per_vertex_normals.h>
#include <vector>


// Define the mesh variables
Eigen::MatrixXd V;  // Vertex coordinates
Eigen::MatrixXi F;  // Face indices

double x_min, x_max, y_min, y_max, z_min, z_max;

double thickness = 0.1;

// Define the grid for the SDF approximation
double gridStep = 0.01;

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
/// @param[in] grid_size_mm The resolution of the grid in each dimension
/// @param[out] sdfGrid The signed distance field values for the grid points
/// @param[out] grid The grid points
void createVoxelGridSDF(double grid_size_mm, Eigen::VectorXd &sdfGrid, Eigen::MatrixXd &grid)
{
    // Compute the step size in each dimension
    // double x_step = (x_max - x_min) / (maxGridResolution - 1);
    // double y_step = (y_max - y_min) / (maxGridResolution - 1);
    // double z_step = (z_max - z_min) / (maxGridResolution - 1);

    double x_step = grid_size_mm, y_step = grid_size_mm, z_step = grid_size_mm;
    int x_dim = ceil((x_max - x_min) / x_step);
    int y_dim = ceil((y_max - y_min) / y_step);
    int z_dim = ceil((z_max - z_min) / z_step);
    
    // Resize the grid matrix and the SDF grid vector
    grid.resize(x_dim * y_dim * z_dim, 3);
    sdfGrid.resize(x_dim * y_dim * z_dim);
    
    // Fill the grid with points and calculate SDF values
    int idx = 0;
    for (int i = 0; i < x_dim; ++i) {
        for (int j = 0; j < y_dim; ++j) {
            for (int k = 0; k < z_dim; ++k) {
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


int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> <output-bin-file> <shell_thickness> <grid_size_mm>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::string out_path = argv[2];
    thickness = std::stod(argv[3]);
    double grid_size_mm = std::stod(argv[4]);

    // Show the input arguments
    std::cout << "Input STL file: " << path << std::endl;
    std::cout << "Output bin file: " << out_path << std::endl;
    std::cout << "Shell thickness: " << thickness << std::endl;

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
    Eigen::VectorXd sdfGrid;
    Eigen::MatrixXd grid;
    createVoxelGridSDF(grid_size_mm, sdfGrid, grid);

    std::vector<Eigen::Vector3d> innerPoints;

    for(int i = 0; i < sdfGrid.size(); i++) {
        if(sdfGrid(i) < -thickness) {
            // The point is inside the shell. Add it to the list of inner points
            innerPoints.push_back(grid.row(i));
        }
    }

    // Print the first 10 inner points
    std::cout << "First 10 inner points: " << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << innerPoints[i].transpose() << std::endl;
    }

    // Write the inner points to a binary file
    std::ofstream out_file(out_path, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open output file: " << out_path << std::endl;
        return -1;
    }

    // Write each point to the file
    for (const auto& point : innerPoints) {
        out_file.write(reinterpret_cast<const char*>(point.data()), sizeof(Eigen::Vector3d));
    }
    
    out_file.close();

    return 0;
}



