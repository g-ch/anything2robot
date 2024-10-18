/**
 * @file metamaterial_filling.cpp
 * @author Clarence Chen (g-ch@github.com)
 * @brief An exampe of using the SemanticDSPMap in a ROS node
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include "meshFilling.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>  
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "filling");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("mesh_cells", 1);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>("triangle_edges", 1);

    MeshFilling meshFilling;

    std::string path = "/home/clarence/ros_ws/quad_ws/src/robots/BullDog_description/meshes/body.stl";

    // std::string path = "/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/ASCII.stl";

    meshFilling.loadMesh(path);

    Eigen::Vector3d bias(0,0,0);
    Eigen::Vector3d cell_start_point;
    Eigen::Vector3i cell_num_each_axis;
    std::vector<std::vector<std::vector<bool>>> cell_inside_flags;

    double cell_size = 0.05;
    double shell_min_thickness = 0.02;

    int cell_num = meshFilling.divideMeshIntoCells(cell_size, bias, shell_min_thickness, cell_start_point, cell_num_each_axis, cell_inside_flags);
    std::cout << "cell_num = " << cell_num << std::endl;

    // // Define a PCL point cloud rgb to show the result cells
    // pcl::PointCloud<pcl::PointXYZRGB> cloud;
    // int point_num = 0;

    // for(int i = 0; i < cell_num_each_axis(0); i++)
    // {
    //     for(int j = 0; j < cell_num_each_axis(1); j++)
    //     {
    //         for(int k = 0; k < cell_num_each_axis(2); k++)
    //         {
    //             if(cell_inside_flags[i][j][k])
    //             {
    //                 pcl::PointXYZRGB point;
    //                 point.x = cell_start_point(0) + i * cell_size;
    //                 point.y = cell_start_point(1) + j * cell_size;
    //                 point.z = cell_start_point(2) + k * cell_size;
    //                 point.r = 255;
    //                 point.g = 0;
    //                 point.b = 0;
    //                 cloud.points.push_back(point);
    //                 point_num++;
    //             }
    //         }
    //     }
    // }
    // cloud.width = point_num;
    // cloud.height = 1;
    // sensor_msgs::PointCloud2 output;
    // pcl::toROSMsg(cloud, output);
    // output.header.frame_id = "map";
    // pub.publish(output);


    double time0 = ros::Time::now().toSec();

    CubicFoam cubicFoam;

    cubicFoam.setThickness(0.2);

    // bool fillMeshWithCell(const T& cell, const Eigen::Vector3d &cell_start_point, const std::vector<std::vector<std::vector<bool>>> &cell_inside_flags, double cell_size, bool do_triangulization = true) {


    meshFilling.fillMeshWithCell(cubicFoam, cell_start_point, cell_inside_flags, cell_size, true);

    meshFilling.addNewTrianglesToOldMesh();

    meshFilling.saveNewMesh("/home/clarence/filled_body.stl");

    double time1 = ros::Time::now().toSec();

    std::cout << "Time cost: " << time1 - time0 << " s" << std::endl;


    //// The rest if for visualization

    // Publish cubicFoam.noncontact_polygons and cubicFoam.contact_polygons as point clouds with different colors
    pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
    int point_num = 0;
    // for(int i = 0; i < cubicFoam.noncontact_polygons.size(); i++)
    // {
    //     for(int j = 0; j < cubicFoam.noncontact_polygons[i].size(); j++)
    //     {
    //         pcl::PointXYZRGB point;
    //         point.x = cubicFoam.noncontact_polygons[i][j](0);
    //         point.y = cubicFoam.noncontact_polygons[i][j](1);
    //         point.z = cubicFoam.noncontact_polygons[i][j](2);
    //         point.r = 255;
    //         point.g = 0;
    //         point.b = 0;
    //         point_cloud.points.push_back(point);
    //         point_num++;
    //     }
    // }
    

    for(int i = 0; i < cubicFoam.contact_polygons.size(); i++)
    {
        for(int j = 0; j < cubicFoam.contact_polygons[i].size(); j++)
        {
            pcl::PointXYZRGB point;
            point.x = cubicFoam.contact_polygons[i][j](0);
            point.y = cubicFoam.contact_polygons[i][j](1);
            point.z = cubicFoam.contact_polygons[i][j](2);
            point.r = 0;
            point.g = 255;
            point.b = 0;
            point_cloud.points.push_back(point);
            point_num++;
        }
        
        // Add a point to show the norm direction of each face
        pcl::PointXYZRGB point;
        point.x = point.y = point.z = 0;
        for(int j = 0; j < cubicFoam.contact_polygons[i].size(); j++)
        {
            point.x += cubicFoam.contact_polygons[i][j](0);
            point.y += cubicFoam.contact_polygons[i][j](1);
            point.z += cubicFoam.contact_polygons[i][j](2);
        }
        point.x /= cubicFoam.contact_polygons[i].size();
        point.y /= cubicFoam.contact_polygons[i].size();
        point.z /= cubicFoam.contact_polygons[i].size();

        point.r = 0;
        point.g = 0;
        point.b = 255;

        int direction = cubicFoam.contact_polygons_directions[i];
        switch (direction)
        {
        case 0:
            point.x += 0.1;
            break;
        case 1:
            point.x -= 0.1;
            break;
        case 2:
            point.y += 0.1;
            break;
        case 3:
            point.y -= 0.1;
            break;
        case 4:
            point.z += 0.1;
            break;
        case 5:
            point.z -= 0.1;
            break;
        
        default:
            std::cout << "Error: direction out of range" << std::endl;
            break;
        }

        point_cloud.points.push_back(point);
        point_num++;
    }



    point_cloud.width = point_num;
    point_cloud.height = 1;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(point_cloud, output);
    output.header.frame_id = "map";


    // Use marker_array to show the lines of the faces in meshFilling.inner_triangles_to_add
    std::cout << "meshFilling.inner_triangles_to_add.size() = " << meshFilling.inner_triangles_to_add.size() << std::endl;

    visualization_msgs::MarkerArray marker_array;

    
    for(int i = 0; i < meshFilling.inner_triangles_to_add.size(); i++)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time();
        marker.ns = "triangle_edges";
        marker.id = i;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.0001;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        for(int j = 0; j < 3; j++)
        {
            geometry_msgs::Point point;
            point.x = meshFilling.inner_triangles_to_add[i][j](0);
            point.y = meshFilling.inner_triangles_to_add[i][j](1);
            point.z = meshFilling.inner_triangles_to_add[i][j](2);
            marker.points.push_back(point);
        }

        // Connect the last point with the first point to form a closed loop
        geometry_msgs::Point point;
        point.x = meshFilling.inner_triangles_to_add[i][0](0);
        point.y = meshFilling.inner_triangles_to_add[i][0](1);
        point.z = meshFilling.inner_triangles_to_add[i][0](2);
        marker.points.push_back(point);

        marker_array.markers.push_back(marker);
    }



    // for(int i=0; i <  cubicFoam.noncontact_polygons.size(); i++)
    // {
    //     visualization_msgs::Marker marker;
    //     marker.header.frame_id = "map";
    //     marker.header.stamp = ros::Time();
    //     marker.ns = "triangle_edges";
    //     marker.id = i;
    //     marker.type = visualization_msgs::Marker::LINE_STRIP;
    //     marker.action = visualization_msgs::Marker::ADD;
    //     marker.pose.orientation.w = 1.0;
    //     marker.scale.x = 0.0005;
    //     marker.color.r = 1.0;
    //     marker.color.g = 1.0;
    //     marker.color.b = 1.0;
    //     marker.color.a = 1.0;

    //     std::cout << "cubicFoam.noncontact_polygons[i].size() = " << cubicFoam.noncontact_polygons[i].size() << std::endl;

    //     for(int j = 0; j < cubicFoam.noncontact_polygons[i].size(); j++)
    //     {
    //         geometry_msgs::Point point;
    //         point.x = cubicFoam.noncontact_polygons[i][j](0);
    //         point.y = cubicFoam.noncontact_polygons[i][j](1);
    //         point.z = cubicFoam.noncontact_polygons[i][j](2);
    //         marker.points.push_back(point);
    //     }

    //         // Connect the last point with the first point to form a closed loop
    //     geometry_msgs::Point point;
    //     point.x = cubicFoam.noncontact_polygons[i][0](0);
    //     point.y = cubicFoam.noncontact_polygons[i][0](1);
    //     point.z = cubicFoam.noncontact_polygons[i][0](2);
    //     marker.points.push_back(point);

    //     marker_array.markers.push_back(marker);
    // }

    



    // Publish the point cloud with a fixed rate of 1 Hz
    ros::Rate loop_rate(1);
    while(ros::ok())
    {
        pub.publish(output);

        marker_pub.publish(marker_array);

        loop_rate.sleep();
    }

    


    // Eigen::MatrixXd points(4,3);
    // points << 0,0,0,
    //           0,0,0.2,
    //           0,0,10,
    //           0,0,1;

    // meshFilling.calculateSignedDistances(points);
    
    return 0;
}
