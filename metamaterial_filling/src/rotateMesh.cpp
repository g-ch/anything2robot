#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// Function to rotate the mesh with a given angle around a given axis
void rotateMesh(const std::string& inputFilename, const std::string& outputFilename, double angle, Eigen::Vector3d axis) {
    Eigen::MatrixXd V;  // Vertex coordinates
    Eigen::MatrixXi F;  // Face indices
    Eigen::MatrixXd norms;  // Face normals

    std::ifstream in_file(inputFilename);
    // Read the STL file
    if (!igl::readSTL(in_file, V, F, norms)) {
        std::cerr << "Failed to load STL file: " << inputFilename << std::endl;
        return;
    }

    // Rotate the mesh
    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
    rotationMatrix.block<3, 3>(0, 0) = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    V = (rotationMatrix * V.transpose()).transpose();

    // Write the modified mesh to a new STL file
    if (!igl::writeSTL(outputFilename, V, F)) {
        std::cerr << "Failed to save STL file: " << outputFilename << std::endl;
        return;
    }

    // Output the results
    std::cout << "Mesh processed and saved to: " << outputFilename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> <output-stl-file> <angle> <axis-x> <axis-y> <axis-z>" << std::endl;
        return 1;
    }

    rotateMesh(argv[1], argv[2], std::stod(argv[3]), Eigen::Vector3d(std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6])));

    return 0;
}
