#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// Function to move center to origin and align largest side to the bottom
void rotateMesh(const std::string& inputFilename, const std::string& outputFilename, double scale=1.0) {
    Eigen::MatrixXd V;  // Vertex coordinates
    Eigen::MatrixXi F;  // Face indices
    Eigen::MatrixXd norms;  // Face normals

    std::ifstream in_file(inputFilename);
    // Read the STL file
    if (!igl::readSTL(in_file, V, F, norms)) {
        std::cerr << "Failed to load STL file: " << inputFilename << std::endl;
        return;
    }

    // Scale the mesh
    V *= scale;

    // Write the modified mesh to a new STL file
    if (!igl::writeSTL(outputFilename, V, F)) {
        std::cerr << "Failed to save STL file: " << outputFilename << std::endl;
        return;
    }

    // Output the results
    std::cout << "Mesh processed and saved to: " << outputFilename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input-stl-file> <output-stl-file> <scale>" << std::endl;
        return 1;
    }

    rotateMesh(argv[1], argv[2], std::stod(argv[3]));

    return 0;
}
