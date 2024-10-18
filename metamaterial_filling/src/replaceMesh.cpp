#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

// Function to move center to origin and align largest side to the bottom
void processMesh(const std::string& inputFilename, const std::string& outputFilename, double scale=1.0) 
{
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

    // Determine the largest face of the bounding box, and align the normal of the largest face to the z-axis
    float xy_area = dimensions[0] * dimensions[1];
    float xz_area = dimensions[0] * dimensions[2];
    float yz_area = dimensions[1] * dimensions[2];

    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
    if(xy_area > xz_area && xy_area > yz_area){
        // Align the normal of the xy plane to the z-axis, which means do nothing

    }else if(xz_area > xy_area && xz_area > yz_area){
        // Align the normal of the xz plane, namely y to the z axis
        rotationMatrix << 1, 0, 0,
                          0, 0, -1,
                          0, 1, 0;
    }else{
        // Align the normal of the yz plane, namely x to the z axis
        rotationMatrix << 0, 0, 1,
                          0, 1, 0,
                          -1, 0, 0;
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

    Eigen::RowVector3d moved_vec = -centerPoint;

    // Write coordinates to a csv file
    std::string outputFilename_csv = outputFilename.substr(0, outputFilename.find_last_of(".")) + ".csv";
    std::ofstream out_file(outputFilename_csv);
    out_file << minCoords_new[0] << "," << minCoords_new[1] << "," << minCoords_new[2] << std::endl;
    out_file << maxCoords_new[0] << "," << maxCoords_new[1] << "," << maxCoords_new[2] << std::endl;
    out_file << scale << "," << 0 << "," << 0 << std::endl;
    out_file << moved_vec[0] << "," << moved_vec[1] << "," << moved_vec[2] << std::endl;
    out_file << rotationMatrix(0, 0) << "," << rotationMatrix(0, 1) << "," << rotationMatrix(0, 2) << std::endl;
    out_file << rotationMatrix(1, 0) << "," << rotationMatrix(1, 1) << "," << rotationMatrix(1, 2) << std::endl;
    out_file << rotationMatrix(2, 0) << "," << rotationMatrix(2, 1) << "," << rotationMatrix(2, 2) << std::endl;
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
