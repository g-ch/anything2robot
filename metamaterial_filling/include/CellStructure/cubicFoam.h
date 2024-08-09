/**
 * @file cubicFoam.h
 * @author Gang Chen (you@domain.com)
 * @brief This head file is to define the cubic foam structure. 
 * IMPORTANT: There public parameters must be included in a cellStructure class to represent the faces of the "Removed" volumes.
 * 1. std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons: representing the convex polygon faces in the cell that will not contact another cell.
 * 2. std::vector<std::vector<Eigen::Vector3d>> contact_polygons: representing the convex polygon faces in the cell that may contact another cell.
 * 3. std::vector<int> contact_polygons_directions: The (normal) directions of the contact_polygons. Describe in which face the polygon may contact another cell. 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
 * The coordinates of the cell should follow: the center is (0,0,0), the length of the cell is 1 (thus one vertice is 0.5,0.5,0.5) and the cell is a unit cube. No orientation is considered.
 * The rest of the cell structure and the class variables/functions can be defined by the user.
 * @version 0.1
 * @date 2024-03-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <vector>
#include "../Eigen/Dense"

class CubicFoam{

public:
    CubicFoam(){
        calculateSurfaces();
    };

    ~CubicFoam(){};

    /// @brief Return the thickness_ of the cubic foam. The faces of the cubic foam are defined by the thickness_ and will be calculated automatically.
    /// @param thickness 
    void setThickness(double thickness_to_set)
    {
        thickness_ = thickness_to_set;
        calculateSurfaces();
    }


public:
    /// @brief Noncontact_polygons represents the convex polygon faces in the cell that will not contact another cell.
    std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons;

    /// @brief Contact_polygons represents  the convex polygon faces in the cell that may contact another cell.
    std::vector<std::vector<Eigen::Vector3d>> contact_polygons;

    ///  @brief The (normal) directions of the contact_polygons. Describe in which face the polygon may contact another cell. 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
    std::vector<int> contact_polygons_directions;

private:
    double thickness_ = 0.2;

private:

    // void calculateSurfaces()
    // {
    //     noncontact_polygons.clear();

    //     double size = 1.0 - thickness_;  
    //     double half_size = size / 2.0;  
        
    //     Eigen::Vector3d p1, p2, p3, p4, p5, p6, p7, p8;  
    //     p1 << half_size, half_size, half_size;  
    //     p2 << -half_size, half_size, half_size;  
    //     p3 << -half_size, -half_size, half_size;  
    //     p4 << half_size, -half_size, half_size;  
    //     p5 << half_size, half_size, -half_size;  
    //     p6 << -half_size, half_size, -half_size;  
    //     p7 << -half_size, -half_size, -half_size;  
    //     p8 << half_size, -half_size, -half_size;

    //     std::vector<Eigen::Vector3d> face1{p5, p6, p2, p1}; 
    //     std::vector<Eigen::Vector3d> face2{p1, p2, p3, p4}; 
    //     std::vector<Eigen::Vector3d> face3{p4, p3, p7, p8};   
    //     std::vector<Eigen::Vector3d> face4{p8, p7, p6, p5}; 
    //     std::vector<Eigen::Vector3d> face5{p5, p1, p4, p8};  
    //     std::vector<Eigen::Vector3d> face6{p6, p7, p3, p2}; 

    //     noncontact_polygons.push_back(face1);
    //     noncontact_polygons.push_back(face2);
    //     noncontact_polygons.push_back(face3);
    //     noncontact_polygons.push_back(face4);
    //     noncontact_polygons.push_back(face5);
    //     noncontact_polygons.push_back(face6);
    // }

    void calculateSurfaces()
    {
        noncontact_polygons.clear();
        contact_polygons.clear();

        // Define the vertices of the inner faces for a cube whose all coordinates are positive. Then we use the symmetry to get the rest of the faces.
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_full_positive_side;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_full_positive_side;
        std::vector<int> contact_directions_full_positive_side;

        double half_thickness = thickness_ / 2.0;
        double sequare_face_size = 0.5 - half_thickness;

        std::vector<Eigen::Vector3d> inner_face_1; //noncontact
        Eigen::Vector3d p1(half_thickness, half_thickness, half_thickness);
        Eigen::Vector3d p2(half_thickness, half_thickness+sequare_face_size, half_thickness);
        Eigen::Vector3d p3(half_thickness, half_thickness+sequare_face_size, half_thickness+sequare_face_size);
        Eigen::Vector3d p4(half_thickness, half_thickness, half_thickness+sequare_face_size);

        inner_face_1.push_back(p1);
        inner_face_1.push_back(p2);
        inner_face_1.push_back(p3);
        inner_face_1.push_back(p4);

        std::vector<Eigen::Vector3d> inner_face_2; //noncontact
        p1 << half_thickness, half_thickness, half_thickness;
        p2 << half_thickness+sequare_face_size, half_thickness, half_thickness;
        p3 << half_thickness+sequare_face_size, half_thickness, half_thickness+sequare_face_size;
        p4 << half_thickness, half_thickness, half_thickness+sequare_face_size;

        inner_face_2.push_back(p1);
        inner_face_2.push_back(p2);
        inner_face_2.push_back(p3);
        inner_face_2.push_back(p4);

        std::vector<Eigen::Vector3d> inner_face_3; //noncontact
        p1 << half_thickness, half_thickness, half_thickness;
        p2 << half_thickness, half_thickness+sequare_face_size, half_thickness;
        p3 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness;
        p4 << half_thickness+sequare_face_size, half_thickness, half_thickness;

        inner_face_3.push_back(p1);
        inner_face_3.push_back(p2);
        inner_face_3.push_back(p3);
        inner_face_3.push_back(p4);
    

        std::vector<Eigen::Vector3d> inner_face_4; //contact
        p1 << half_thickness, half_thickness, half_thickness+sequare_face_size;
        p2 << half_thickness, half_thickness+sequare_face_size, half_thickness+sequare_face_size;
        p3 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness+sequare_face_size;
        p4 << half_thickness+sequare_face_size, half_thickness, half_thickness+sequare_face_size;

        inner_face_4.push_back(p1);
        inner_face_4.push_back(p2);
        inner_face_4.push_back(p3);
        inner_face_4.push_back(p4);

        std::vector<Eigen::Vector3d> inner_face_5; //contact
        p1 << half_thickness+sequare_face_size, half_thickness, half_thickness;
        p2 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness;
        p3 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness+sequare_face_size;
        p4 << half_thickness+sequare_face_size, half_thickness, half_thickness+sequare_face_size;

        inner_face_5.push_back(p1);
        inner_face_5.push_back(p2);
        inner_face_5.push_back(p3);
        inner_face_5.push_back(p4);

        std::vector<Eigen::Vector3d> inner_face_6; //contact
        p1 << half_thickness, half_thickness+sequare_face_size, half_thickness;
        p2 << half_thickness, half_thickness+sequare_face_size, half_thickness+sequare_face_size;
        p3 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness+sequare_face_size;
        p4 << half_thickness+sequare_face_size, half_thickness+sequare_face_size, half_thickness;

        inner_face_6.push_back(p1);
        inner_face_6.push_back(p2);
        inner_face_6.push_back(p3);
        inner_face_6.push_back(p4);
        
        noncontact_polygons_full_positive_side.push_back(inner_face_1);
        noncontact_polygons_full_positive_side.push_back(inner_face_2);
        noncontact_polygons_full_positive_side.push_back(inner_face_3);

        contact_polygons_full_positive_side.push_back(inner_face_4);
        contact_polygons_full_positive_side.push_back(inner_face_5);
        contact_polygons_full_positive_side.push_back(inner_face_6);

        contact_directions_full_positive_side.push_back(4); // +z
        contact_directions_full_positive_side.push_back(0); // +x
        contact_directions_full_positive_side.push_back(2); // +y


        // Now consdier Cube 1 who is symmetric to Cube 0 about the x-y plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube1;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube1;
        std::vector<int> contact_directions_cube1;


        for(int i = 0; i < noncontact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = noncontact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(2) = -point(2);
                face.push_back(new_point);
            }
            noncontact_polygons_cube1.push_back(face);
        }

        for(int i = 0; i < contact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = contact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(2) = -point(2);
                face.push_back(new_point);
            }
            contact_polygons_cube1.push_back(face);
            
            if(contact_directions_full_positive_side[i] == 4) // +z
            {
                contact_directions_cube1.push_back(5); // -z
            }else{
                contact_directions_cube1.push_back(contact_directions_full_positive_side[i]);
            }
        }

        // Now consdier Cube 2 who is symmetric to Cube 0 about the y-z plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube2;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube2;
        std::vector<int> contact_directions_cube2;


        for(int i = 0; i < noncontact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = noncontact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(0) = -point(0);
                face.push_back(new_point);
            }
            noncontact_polygons_cube2.push_back(face);
        }

        for(int i = 0; i < contact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = contact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(0) = -point(0);
                face.push_back(new_point);
            }
            contact_polygons_cube2.push_back(face);

            if(contact_directions_full_positive_side[i] == 0) // +x
            {
                contact_directions_cube2.push_back(1); // -x
            }else{
                contact_directions_cube2.push_back(contact_directions_full_positive_side[i]);
            }
        }

        // Now consdier Cube 3 who is symmetric to Cube 0 about the z-x plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube3;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube3;
        std::vector<int> contact_directions_cube3;

        for(int i = 0; i < noncontact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = noncontact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(1) = -point(1);
                face.push_back(new_point);
            }
            noncontact_polygons_cube3.push_back(face);
        }

        for(int i = 0; i < contact_polygons_full_positive_side.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_full_positive_side[i].size(); j++)
            {
                Eigen::Vector3d point = contact_polygons_full_positive_side[i][j];
                Eigen::Vector3d new_point = point;
                new_point(1) = -point(1);
                face.push_back(new_point);
            }
            contact_polygons_cube3.push_back(face);

            if(contact_directions_full_positive_side[i] == 2) // +y
            {
                contact_directions_cube3.push_back(3); // -y
            }else{
                contact_directions_cube3.push_back(contact_directions_full_positive_side[i]);
            }
        }

        // Now consdier Cube 4 who is symmetric to Cube 1 about the z-x plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube4;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube4;
        std::vector<int> contact_directions_cube4;

        for(int i = 0; i < noncontact_polygons_cube1.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_cube1[i].size(); j++)
            {
                 Eigen::Vector3d point = noncontact_polygons_cube1[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(1) = -point(1);
                 face.push_back(new_point);
            }
            noncontact_polygons_cube4.push_back(face);
        }


        for(int i = 0; i < contact_polygons_cube1.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_cube1[i].size(); j++)
            {
                 Eigen::Vector3d point = contact_polygons_cube1[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(1) = -point(1);
                 face.push_back(new_point);
            }
            contact_polygons_cube4.push_back(face);

            if(contact_directions_cube1[i] == 2) // +y
            {
                contact_directions_cube4.push_back(3); // -y
            }else{
                contact_directions_cube4.push_back(contact_directions_cube1[i]);
            }
        }

        // Now consdier Cube 5 who is symmetric to Cube 1 about the y-z plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube5;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube5;
        std::vector<int> contact_directions_cube5;


        for(int i = 0; i < noncontact_polygons_cube1.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_cube1[i].size(); j++)
            {
                 Eigen::Vector3d point = noncontact_polygons_cube1[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            noncontact_polygons_cube5.push_back(face);
        }


        for(int i = 0; i < contact_polygons_cube1.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_cube1[i].size(); j++)
            {
                 Eigen::Vector3d point = contact_polygons_cube1[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            contact_polygons_cube5.push_back(face);

            if(contact_directions_cube1[i] == 0) // +x
            {
                contact_directions_cube5.push_back(1); // -x
            }else{
                contact_directions_cube5.push_back(contact_directions_cube1[i]);
            }
        }

        // Now consdier Cube 6 who is symmetric to Cube 3 about the y-z plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube6;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube6;
        std::vector<int> contact_directions_cube6;

        for(int i = 0; i < noncontact_polygons_cube3.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_cube3[i].size(); j++)
            {
                 Eigen::Vector3d point = noncontact_polygons_cube3[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            noncontact_polygons_cube6.push_back(face);
        }


        for(int i = 0; i < contact_polygons_cube3.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_cube3[i].size(); j++)
            {
                 Eigen::Vector3d point = contact_polygons_cube3[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            contact_polygons_cube6.push_back(face);

            if(contact_directions_cube3[i] == 0) // +x
            {
                contact_directions_cube6.push_back(1); // -x
            }else{
                contact_directions_cube6.push_back(contact_directions_cube3[i]);
            }
        }


        // Now consdier Cube 7 who is symmetric to Cube 4 about the y-z plane
        std::vector<std::vector<Eigen::Vector3d>> noncontact_polygons_cube7;
        std::vector<std::vector<Eigen::Vector3d>> contact_polygons_cube7;
        std::vector<int> contact_directions_cube7;

        for(int i = 0; i < noncontact_polygons_cube4.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < noncontact_polygons_cube4[i].size(); j++)
            {
                 Eigen::Vector3d point = noncontact_polygons_cube4[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            noncontact_polygons_cube7.push_back(face);
        }

        for(int i = 0; i < contact_polygons_cube4.size(); i++)
        {
            std::vector<Eigen::Vector3d> face;
            for(int j = 0; j < contact_polygons_cube4[i].size(); j++)
            {
                 Eigen::Vector3d point = contact_polygons_cube4[i][j];
                 Eigen::Vector3d new_point = point;
                 new_point(0) = -point(0);
                 face.push_back(new_point);
            }
            contact_polygons_cube7.push_back(face);

            if(contact_directions_cube4[i] == 0) // +x
            {
                contact_directions_cube7.push_back(1); // -x
            }else{
                contact_directions_cube7.push_back(contact_directions_cube4[i]);
            }
        }


        // Now add all the faces to the noncontact_polygons and contact_polygons
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_full_positive_side.begin(), noncontact_polygons_full_positive_side.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube1.begin(), noncontact_polygons_cube1.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube2.begin(), noncontact_polygons_cube2.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube3.begin(), noncontact_polygons_cube3.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube4.begin(), noncontact_polygons_cube4.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube5.begin(), noncontact_polygons_cube5.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube6.begin(), noncontact_polygons_cube6.end());
        noncontact_polygons.insert(noncontact_polygons.end(), noncontact_polygons_cube7.begin(), noncontact_polygons_cube7.end());

        contact_polygons.insert(contact_polygons.end(), contact_polygons_full_positive_side.begin(), contact_polygons_full_positive_side.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube1.begin(), contact_polygons_cube1.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube2.begin(), contact_polygons_cube2.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube3.begin(), contact_polygons_cube3.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube4.begin(), contact_polygons_cube4.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube5.begin(), contact_polygons_cube5.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube6.begin(), contact_polygons_cube6.end());
        contact_polygons.insert(contact_polygons.end(), contact_polygons_cube7.begin(), contact_polygons_cube7.end());

        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_full_positive_side.begin(), contact_directions_full_positive_side.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube1.begin(), contact_directions_cube1.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube2.begin(), contact_directions_cube2.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube3.begin(), contact_directions_cube3.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube4.begin(), contact_directions_cube4.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube5.begin(), contact_directions_cube5.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube6.begin(), contact_directions_cube6.end());
        contact_polygons_directions.insert(contact_polygons_directions.end(), contact_directions_cube7.begin(), contact_directions_cube7.end());
    }
    
};