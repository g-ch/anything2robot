/**
 * @file meshFilling.h
 * @author Gang Chen (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "igl/readSTL.h"
#include "igl/writeSTL.h"
#include <igl/signed_distance.h>
#include <type_traits>
#include "CellStructure/cubicFoam.h"
#include <unordered_map>
#include <functional>  



// 定义Eigen::Vector3d的哈希函数  
struct EigenVector3dHash {  
    std::size_t operator()(const Eigen::Vector3d& v) const {  
        std::size_t h1 = std::hash<double>()(v(0));
        std::size_t h2 = std::hash<double>()(v(1));
        std::size_t h3 = std::hash<double>()(v(2));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }  
};  
  
// 定义Eigen::Vector3d的比较函数  
struct EigenVector3dEqual {  
    bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {  
        return a.isApprox(b);  
    }  
}; 


// int PRECISION = 1000;
// int BITS_PER_NUMBER = 21;

// inline int64_t encodeVector3d(Eigen::Vector3d &n) {  
//     int64_t intA = static_cast<int64_t>(std::round(n(0) * PRECISION));
//     int64_t intB = static_cast<int64_t>(std::round(n(1) * PRECISION));  
//     int64_t intC = static_cast<int64_t>(std::round(n(2) * PRECISION));  

//     int64_t encoded = (intA << (2 * BITS_PER_NUMBER)) | (intB << BITS_PER_NUMBER) | intC;  
//     return encoded;  
// }  

// inline Eigen::Vector3d decodeVector3d(int64_t encoded) {  
//     int64_t intA = encoded >> (2 * BITS_PER_NUMBER);  
//     int64_t intB = (encoded >> BITS_PER_NUMBER) & ((1 << BITS_PER_NUMBER) - 1);  
//     int64_t intC = encoded & ((1 << BITS_PER_NUMBER) - 1);  

//     Eigen::Vector3d n;  
//     n(0) = static_cast<double>(intA) / PRECISION;  
//     n(1) = static_cast<double>(intB) / PRECISION;  
//     n(2) = static_cast<double>(intC) / PRECISION;  
//     return n;  
// }


class MeshFilling
{
public:

    MeshFilling() : mesh_x_min_(0), mesh_x_max_(0), mesh_y_min_(0), mesh_y_max_(0), mesh_z_min_(0), mesh_z_max_(0)
    {
       
    }

    ~MeshFilling()
    {
    }

    /// @brief Load the mesh from a file
    /// @param path The path of the mesh file
    /// @return bool If the mesh is loaded successfully
    bool loadMesh(std::string path)
    {
        std::ifstream in_path(path);
        bool success = igl::readSTL(in_path, ori_vertices_, ori_faces_, ori_norms_);
        
        // std::cout << "Vertices: " << std::endl << ori_vertices_ << std::endl;
        // std::cout << "Faces:    " << std::endl << ori_faces_ << std::endl;

        // Find the min and max of the mesh
        mesh_x_min_ = ori_vertices_.col(0).minCoeff();
        mesh_x_max_ = ori_vertices_.col(0).maxCoeff();
        mesh_y_min_ = ori_vertices_.col(1).minCoeff();
        mesh_y_max_ = ori_vertices_.col(1).maxCoeff();
        mesh_z_min_ = ori_vertices_.col(2).minCoeff();
        mesh_z_max_ = ori_vertices_.col(2).maxCoeff();

        std::cout << "Mesh x min: " << mesh_x_min_ << ", x max: " << mesh_x_max_ << std::endl;
        std::cout << "Mesh y min: " << mesh_y_min_ << ", y max: " << mesh_y_max_ << std::endl;
        std::cout << "Mesh z min: " << mesh_z_min_ << ", z max: " << mesh_z_max_ << std::endl;

        return success;
    }

    
    /// @brief Fill the mesh with a cell structure
    /// @param cell The cell structure
    /// @param cell_start_point The start point of the cell. Minimum x, y, z of the cells
    /// @param cell_inside_flags The flags to indicate if the cell is inside the mesh
    /// @param cell_size The size of the cell
    /// @param do_triangulization If true, the faces in noncontact_polygons and contact_polygons will be triangulized. If false, we consider them already triangulized
    template<typename T>
    bool fillMeshWithCell(const T& cell, const Eigen::Vector3d &cell_start_point, const std::vector<std::vector<std::vector<bool>>> &cell_inside_flags, double cell_size, bool do_triangulization = true) {
        // Clear the inner_triangles_to_add
        inner_triangles_to_add.clear();

        std::cout << "cell_start_point: " << cell_start_point << std::endl;

        // Triangulize the faces in noncontact_polygons and contact_polygons if necessary
        std::vector<std::vector<Eigen::Vector3d>> noncontact_triangles;
        std::vector<std::vector<Eigen::Vector3d>> contact_triangles;
        std::vector<int> contact_triangles_directions;

        if(do_triangulization)
        {
            for(auto &polygon : cell.noncontact_polygons)
            {
                std::vector<std::vector<Eigen::Vector3d>> triangles;
                triangularizePolygonFace(polygon, triangles);
                noncontact_triangles.insert(noncontact_triangles.end(), triangles.begin(), triangles.end());
            }

            for(int i=0; i<cell.contact_polygons.size(); i++)
            {
                std::vector<std::vector<Eigen::Vector3d>> triangles;
                triangularizePolygonFace(cell.contact_polygons[i], triangles);
                contact_triangles.insert(contact_triangles.end(), triangles.begin(), triangles.end());
                for(int j=0; j<triangles.size(); j++)
                {
                    contact_triangles_directions.push_back(cell.contact_polygons_directions[i]);
                }
            }

        }else{
            noncontact_triangles = cell.noncontact_polygons;
            contact_triangles = cell.contact_polygons;
            contact_triangles_directions = cell.contact_polygons_directions;
        }

        std::cout << "contact_triangles size = " << contact_triangles.size() << std::endl;
        std::cout << "contact_triangles_directions size = " << contact_triangles_directions.size() << std::endl;

        // Now iterate each cell and add the triangles to the mesh
        Eigen::Vector3d correction(0.5, 0.5, 0.5);

        for(int i=0; i<cell_inside_flags.size(); i++)
        {
            for(int j=0; j<cell_inside_flags[i].size(); j++)
            {
                for(int k=0; k<cell_inside_flags[i][j].size(); k++)
                {
                    if(cell_inside_flags[i][j][k])
                    {
                        // Add the noncontact_triangles to the mesh
                        for(auto &triangle : noncontact_triangles)
                        {
                            std::vector<Eigen::Vector3d> new_triangle;
                            for(auto &vertex : triangle)
                            {
                                Eigen::Vector3d new_vertex = (vertex + correction) * cell_size + cell_start_point + Eigen::Vector3d(i, j, k) * cell_size;
                                new_triangle.push_back(new_vertex);
                            }
                            inner_triangles_to_add.push_back(new_triangle);
                        }

                        // Now consider the contact_triangles. We need to check the direction of the contact_triangles. If the cell_inside_flags of the cell in the direction of the contact_triangles is false, we add the contact_triangles to the mesh
                        for(int m=0; m<contact_triangles.size(); m++)
                        {
                            int direction = contact_triangles_directions[m];
                            Eigen::Vector3i cell_index(i, j, k);
                            Eigen::Vector3i cell_index_in_direction = cell_index;
                            switch (direction)
                            {
                            case 0:
                                cell_index_in_direction(0) += 1;
                                break;
                            case 1:
                                cell_index_in_direction(0) -= 1;
                                break;
                            case 2:
                                cell_index_in_direction(1) += 1;
                                break;
                            case 3:
                                cell_index_in_direction(1) -= 1;
                                break;
                            case 4:
                                cell_index_in_direction(2) += 1;
                                break;
                            case 5:
                                cell_index_in_direction(2) -= 1;
                                break;
                            default:
                                break;
                            }

                            if(cell_index_in_direction(0) >= 0 && cell_index_in_direction(0) < cell_inside_flags.size() &&
                               cell_index_in_direction(1) >= 0 && cell_index_in_direction(1) < cell_inside_flags[0].size() &&
                               cell_index_in_direction(2) >= 0 && cell_index_in_direction(2) < cell_inside_flags[0][0].size() &&
                               !cell_inside_flags[cell_index_in_direction(0)][cell_index_in_direction(1)][cell_index_in_direction(2)])
                            {
                                std::vector<Eigen::Vector3d> new_triangle;
                                for(auto &vertex : contact_triangles[m])
                                {
                                    Eigen::Vector3d new_vertex = (vertex + correction) * cell_size + cell_start_point + Eigen::Vector3d(i, j, k) * cell_size;
                                    new_triangle.push_back(new_vertex);
                                }
                                inner_triangles_to_add.push_back(new_triangle);
                            }
                        }
                        
                    }
                }
            }
        }

        return true;

    }
                  
       

    /// @brief Divide the mesh into cells.
    /// @param cell_size The size of the cell
    /// @param bias The bias of the cell start point
    /// @param minimum_shell_thickness The minimum shell thickness for the mesh
    /// @param cell_start_point The start point of the cell. Minimum x, y, z of the cells
    /// @param cell_num_each_axis The number of cells in each axis
    /// @param cell_inside_flags The flags to indicate if the cell is inside the mesh
    /// @return int The number of cells inside the mesh
    int divideMeshIntoCells(double cell_size, Eigen::Vector3d bias, double minimum_shell_thickness, Eigen::Vector3d &cell_start_point, Eigen::Vector3i &cell_num_each_axis, std::vector<std::vector<std::vector<bool>>> &cell_inside_flags) {
        // Divide the mesh into cells. Consider the bias. Expand the cells by 2, which means each side one more cell.
        int mesh_x_cells = (int) ((mesh_x_max_ - mesh_x_min_) / cell_size);
        int mesh_y_cells = (int) ((mesh_y_max_ - mesh_y_min_) / cell_size);
        int mesh_z_cells = (int) ((mesh_z_max_ - mesh_z_min_) / cell_size);


        if(mesh_x_cells < 1 || mesh_y_cells < 1 || mesh_z_cells < 1) {
            std::cout << "The mesh is too small to be divided into cells with the current cell size" << std::endl;
            return 0;
        }

        // Expand the cells by 2, which means each side one more cell.
        mesh_x_cells += 2;
        mesh_y_cells += 2;
        mesh_z_cells += 2;


        // Define a MatrixXd to store the vertices of all the cells. There should be (mesh_x_cells+1)*(mesh_y_cells+1)*(mesh_z_cells+1) vertices in total.
        // We store the vertices of each cell in a row with the order of 
        int total_vertices = (mesh_x_cells+1)*(mesh_y_cells+1)*(mesh_z_cells+1);
        Eigen::MatrixXd cell_vertices(total_vertices, 3);

        double cell_vertex_start_x = mesh_x_min_ + bias(0) - cell_size;
        double cell_vertex_start_y = mesh_y_min_ + bias(1) - cell_size;
        double cell_vertex_start_z = mesh_z_min_ + bias(2) - cell_size;

        cell_start_point << cell_vertex_start_x, cell_vertex_start_y, cell_vertex_start_z;

        for(int i=0; i<mesh_x_cells+1; i++) {
            for(int j=0; j<mesh_y_cells+1; j++) {
                for(int k=0; k<mesh_z_cells+1; k++) {
                    // Store the vertices of each cell in a row with the order of
                    cell_vertices.row(i*(mesh_y_cells+1)*(mesh_z_cells+1) + j*(mesh_z_cells+1) + k) << cell_vertex_start_x + i*cell_size, cell_vertex_start_y + j*cell_size, cell_vertex_start_z + k*cell_size;
                }
            }
        }

        // Calculate the signed distance of the vertices
        Eigen::VectorXd distances;
        calculateSignedDistances(cell_vertices, distances);


        // Check if each cell is in the mesh by checkin the SDF
        cell_num_each_axis << mesh_x_cells, mesh_y_cells, mesh_z_cells;
        cell_inside_flags  = std::vector<std::vector<std::vector<bool>>>(mesh_x_cells, std::vector<std::vector<bool>>(mesh_y_cells, std::vector<bool>(mesh_z_cells, false)));

        int cell_num = 0;
        for(int i=0; i<mesh_x_cells; ++i){
            for(int j=0; j<mesh_y_cells; ++j){
                for(int k=0; k<mesh_z_cells; ++k){
                    Eigen::Vector3i cell_index(i,j,k);

                    std::vector<int> vertex_indices;
                    findCellVertexIndex(cell_index, cell_num_each_axis, vertex_indices);

                    // Check if the SDF of the vertex_indices are negtive.
                    bool inside_flag = true;
                    for(auto &index : vertex_indices)
                    {
                        if(distances(index) > -minimum_shell_thickness){
                            inside_flag = false;
                            break;
                        }
                    }

                    cell_inside_flags[i][j][k] = inside_flag;

                    if(inside_flag)
                    {
                        cell_num++;
                    }
                }
            }
        }


        return cell_num;
    }


    void addNewTrianglesToOldMesh()
    {
        std::unordered_map<Eigen::Vector3d, int, EigenVector3dHash, EigenVector3dEqual> vertexIndexMap;

        // std::unordered_map<uint64_t, int> vertexIndexMap;

        // Copy old vertices and faces to new_vertices and new_faces to keep the original mesh
        new_vertices_ = ori_vertices_;
        new_faces_ = ori_faces_;

        int current_vertex_index = ori_vertices_.rows();
        int current_face_index = ori_faces_.rows();

        std::cout << "Before vertexes merge, vertices num: " << new_vertices_.rows() << " Faces num: " << new_faces_.rows() << std::endl;
        std::cout << "Triangle vertices to add num: " << inner_triangles_to_add.size()*3 << " Faces num: " << inner_triangles_to_add.size() << std::endl;


        // Loop through the inner_triangles_to_add and add the vertices to new_vertices_ and add the faces to new_faces_
        for(auto &triangle : inner_triangles_to_add)
        {
            // Add the vertices to new_vertices_
            for(auto &vertex : triangle)
            {
                // Check if the vertex is already in the new_vertices_

                // uint64_t vertex_encoded = encodeVector3d(vertex);
                // if(vertexIndexMap.find(vertex_encoded) == vertexIndexMap.end())
                // {
                //     vertexIndexMap[vertex_encoded] = new_vertices_.rows();
                //     new_vertices_.conservativeResize(new_vertices_.rows()+1, 3);
                //     new_vertices_.row(new_vertices_.rows()-1) = vertex;
                // }

                if(vertexIndexMap.find(vertex) == vertexIndexMap.end())
                {
                    /// TODO: Try to improve the efficiency of the following code. One way, use csv file.
                    vertexIndexMap[vertex] = new_vertices_.rows();
                    new_vertices_.conservativeResize(new_vertices_.rows()+1, 3);
                    new_vertices_.row(new_vertices_.rows()-1) = vertex;
                }
            }

            Eigen::RowVector3i new_face;
            for(int i=0; i<3; i++)
            {
                // uint64_t vertex_encoded = encodeVector3d(triangle[i]);
                // new_face(i) = vertexIndexMap[vertex_encoded];

                new_face(i) = vertexIndexMap[triangle[i]];
            }
            new_faces_.conservativeResize(new_faces_.rows()+1, 3);
            new_faces_.row(new_faces_.rows()-1) = new_face;
        }

        
        std::cout << "After vertexes merge, vertices num: " << new_vertices_.rows() << " Faces num: " << new_faces_.rows() << std::endl;

    }

    void saveNewMesh(std::string path)
    {
        igl::writeSTL(path, new_vertices_, new_faces_, igl::FileEncoding::Binary);
        std::cout << "New mesh saved to " << path << std::endl;
    }


public:
    std::vector<std::vector<Eigen::Vector3d>> inner_triangles_to_add;


private:
    Eigen::MatrixXd ori_vertices_;
    Eigen::MatrixXd ori_norms_;
    Eigen::MatrixXi ori_faces_;

    Eigen::MatrixXd new_vertices_;
    Eigen::MatrixXi new_faces_;
 
    double mesh_x_min_, mesh_x_max_;
    double mesh_y_min_, mesh_y_max_;
    double mesh_z_min_, mesh_z_max_;


    inline void calculateSignedDistances(const Eigen::MatrixXd& points, Eigen::VectorXd &distances)
    {
        Eigen::VectorXi I;
        Eigen::MatrixXd C, N;
        // Choose type of signing to use
        igl::SignedDistanceType type = igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
        igl::signed_distance(points, ori_vertices_, ori_faces_, type, distances, I,C,N);
    }

    inline void findCellVertexIndex(const Eigen::Vector3i &cell_index, const Eigen::Vector3i &cell_num_each_axis, std::vector<int> &vertex_indices)
    {
        int x_min_index_vertex = cell_index(0);
        int y_min_index_vertex = cell_index(1);
        int z_min_index_vertex = cell_index(2);
        int x_max_index_vertex = cell_index(0) + 1;
        int y_max_index_vertex = cell_index(1) + 1;
        int z_max_index_vertex = cell_index(2) + 1;

        Eigen::Vector3i vertex_num_each_axis = cell_num_each_axis + Eigen::Vector3i::Ones();

        // Point 0
        vertex_indices.push_back(x_min_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_min_index_vertex * vertex_num_each_axis(2) + z_min_index_vertex);
        // Point 1
        vertex_indices.push_back(x_max_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_min_index_vertex * vertex_num_each_axis(2) + z_min_index_vertex);
        // Point 2
        vertex_indices.push_back(x_max_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_max_index_vertex * vertex_num_each_axis(2) + z_min_index_vertex);
        // Point 3
        vertex_indices.push_back(x_min_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_max_index_vertex * vertex_num_each_axis(2) + z_min_index_vertex);
        // Point 4
        vertex_indices.push_back(x_min_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_min_index_vertex * vertex_num_each_axis(2) + z_max_index_vertex);
        // Point 5
        vertex_indices.push_back(x_max_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_min_index_vertex * vertex_num_each_axis(2) + z_max_index_vertex);
        // Point 6
        vertex_indices.push_back(x_max_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_max_index_vertex * vertex_num_each_axis(2) + z_max_index_vertex);
        // Point 7
        vertex_indices.push_back(x_min_index_vertex * vertex_num_each_axis(1) * vertex_num_each_axis(2) + y_max_index_vertex * vertex_num_each_axis(2) + z_max_index_vertex);
    } 


    inline void triangularizePolygonFace(const std::vector<Eigen::Vector3d> &polygon, std::vector<std::vector<Eigen::Vector3d>> &triangles)
    {
        // Triangulize the polygon face
        for(int i=0; i<polygon.size()-2; i++)
        {
            std::vector<Eigen::Vector3d> triangle;
            triangle.push_back(polygon[0]);
            triangle.push_back(polygon[i+1]);
            triangle.push_back(polygon[i+2]);
            triangles.push_back(triangle);
        }
    }


};

