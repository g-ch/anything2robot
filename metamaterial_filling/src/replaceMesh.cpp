#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// Function to move center to origin and align largest side to the bottom
void processMesh(const std::string& inputFilename, const std::string& outputFilename, double scale=1.0) {
    Eigen::MatrixXd V;  // Vertex coordinates
    Eigen::MatrixXi F;  // Face indices
    Eigen::MatrixXd norms;  // Face normals

    std::ifstream in_file(inputFilename);
    // Read the STL file
    if (!igl::readSTL(in_file, V, F, norms)) {
        std::cerr << "Failed to load STL file: " << inputFilename << std::endl;
        return;
    }

    std::cout << "Use the scale: " << scale << std::endl;

    // Scale the mesh
    V *= scale;

    // Find min, max coordinates
    Eigen::RowVector3d minCoords = V.colwise().minCoeff();
    Eigen::RowVector3d maxCoords = V.colwise().maxCoeff();

    // Calculate the center point
    Eigen::RowVector3d centerPoint = (minCoords + maxCoords) / 2.0;

    // Translate vertices to move center to origin
    V.rowwise() -= centerPoint;

    // Calculate bounding box dimensions
    Eigen::RowVector3d dimensions = maxCoords - minCoords;

    // Determine the axis of the largest dimension
    int largestAxis = 0;
    if (dimensions[1] > dimensions[0] && dimensions[1] > dimensions[2]) {
        largestAxis = 1;
    } else if (dimensions[2] > dimensions[0]) {
        largestAxis = 2;
    }

    // Rotation matrix to align the largest side to the bottom (z = 0)
    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
    if (largestAxis == 1) {
        // Rotate around x-axis to bring y to z
        rotationMatrix << 1, 0, 0,
                          0, 0, -1,
                          0, 1, 0;
    } else if (largestAxis == 2) {
        // Rotate around x-axis to bring z to y
        rotationMatrix << 1, 0, 0,
                          0, 0, 1,
                          0, -1, 0;
    }

    // Apply rotation
    V = (rotationMatrix * V.transpose()).transpose();

    // Write the modified mesh to a new STL file
    if (!igl::writeSTL(outputFilename, V, F)) {
        std::cerr << "Failed to save STL file: " << outputFilename << std::endl;
        return;
    }

    // Output the results
    std::cout << "Mesh processed and saved to: " << outputFilename << std::endl;

    // Find min, max coordinates of the new mesh
    Eigen::RowVector3d minCoords_new = V.colwise().minCoeff();
    Eigen::RowVector3d maxCoords_new = V.colwise().maxCoeff();

    // Write coordinates to a csv file
    std::string outputFilename_csv = outputFilename.substr(0, outputFilename.find_last_of(".")) + ".csv";
    std::ofstream out_file(outputFilename_csv);
    out_file << minCoords_new[0] << "," << minCoords_new[1] << "," << minCoords_new[2] << std::endl;
    out_file << maxCoords_new[0] << "," << maxCoords_new[1] << "," << maxCoords_new[2] << std::endl;
    out_file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> <output-stl-file> <scale>[optional]" << std::endl;
        return 1;
    }

    if(argc == 4) {
        // Call the function with the STL file paths and scale
        processMesh(argv[1], argv[2], std::stod(argv[3]));
    }else{
        // Call the function with the STL file paths
        processMesh(argv[1], argv[2]);
    }

    return 0;
}
