'''
@Author: Clarence
@Date: 2024-7-5
@Description: This script is used to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''

import os
import argparse
import subprocess
import sys
import pickle as pkl

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir + '/metamaterial_filling/script')
sys.path.append(os.path.join(project_dir, '/auto_design/modules'))
sys.path.append(os.path.join(project_dir, '/auto_design'))

from metamaterial.sixFoldPlatesFillingWithShellTenon import SixFoldPlatesFillingWithShellTenon
from metamaterial.generateMeshFromPoints import generate_mesh_from_points
import time
import numpy as np
import math
from format_transform.off_to_stl import off_to_stl

from visualization.assemble_vis import get_rotation_matrix, visualize_meshes, transform_trimesh, get_rotation_matrix_from_angle


from skimage import measure
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import time

'''
@Description: Voxelize a trimesh mesh with a given voxel size
@Input:
    trimesh_mesh: The trimesh mesh to voxelize
    voxel_size: The size of the voxels
@Output:
    voxels: The 3D array of voxels
    min_bound: The minimum bounds of the voxel grid
    max_bound: The maximum bounds of the voxel grid
    voxel_size: The size of the voxels
'''
def voxelize_mesh(trimesh_mesh, voxel_size):
    # Convert trimesh mesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces)
    )
    
    # Create a voxel grid from the Open3D mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size=voxel_size)
    
    # Extract voxel coordinates
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    
    # Calculate the bounds of the voxel grid
    min_bound = voxel_indices.min(axis=0) * voxel_size + voxel_grid.origin
    max_bound = voxel_indices.max(axis=0) * voxel_size + voxel_grid.origin

    # Create a 3D array to hold the voxels
    shape = ((max_bound - min_bound) / voxel_size).astype(int) + 1
    voxels = np.zeros(shape, dtype=bool)

    # Set the corresponding voxels to True, clamping to avoid out-of-bounds errors
    for idx in voxel_indices:
        grid_idx = ((idx * voxel_size + voxel_grid.origin) - min_bound) / voxel_size
        grid_idx = np.clip(grid_idx.astype(int), 0, np.array(shape) - 1)
        voxels[tuple(grid_idx)] = True

    return voxels, min_bound, max_bound, voxel_size


'''
@Description: Check if a point is occupied in the voxel grid
@Input:
    point: The point to check
    voxels: The 3D array of voxels
    min_bound: The minimum bounds of the voxel grid
    max_bound: The maximum bounds of the voxel grid
    voxel_size: The size of the voxels
@Output:
    Whether the point is occupied

'''
def is_point_occupied(point, voxels, min_bound, max_bound, voxel_size):
    # Check if the point is outside the mesh bounds
    if np.any(point < min_bound) or np.any(point > max_bound):
        return False

    # Convert the point to voxel grid indices
    index = ((point - min_bound) / voxel_size).astype(int)

    # Check if the indices are within the voxel grid shape
    if np.any(index < 0) or np.any(index >= np.array(voxels.shape)):
        return False

    # Return the occupancy of the voxel at the given index
    return voxels[tuple(index)]


'''
@Description: Generate a set of perpendicular vectors to a given direction
@Input:
    direction: The direction vector
    angle_interval: The interval between the angles of the perpendicular vectors
@Output:
    vectors: The list of perpendicular vectors
    angles: The angles of the perpendicular vectors
'''
def generate_perpendicular_vectors(direction, angle_interval):
    # Ensure direction is a unit vector
    direction = direction / np.linalg.norm(direction)
    
    # Find two vectors perpendicular to the direction
    if np.allclose(direction, [1, 0, 0]) or np.allclose(direction, [-1, 0, 0]):
        v1 = np.array([0, 1, 0])
        angles = np.arange(0, 2 * np.pi, angle_interval) - np.pi / 2
    else:
        v1 = np.cross(direction, [1, 0, 0])
        angles = np.arange(0, 2 * np.pi, angle_interval)
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(direction, v1)
    
    # Sample angles
    vectors = [np.cos(angle) * v1 + np.sin(angle) * v2 for angle in angles]

    normalized_vectors = []
    for vec in vectors:
        normalized_vectors.append(vec / np.linalg.norm(vec))

    # print(f"Direction: {direction}")
    # print(f"Perpendicular vectors: {vectors}")
    
    return vectors, angles


'''
@Description: Check the occupancy of the voxels along perpendicular rays from a point
@Input:
    point: The point to check from
    direction: The direction of the rays
    voxel_size: The size of the voxels
    checking_distance: The distance to check along the rays
    angle_interval: The interval between the angles of the perpendicular vectors
    voxels: The 3D array of voxels
    min_bound: The minimum bounds of the voxel grid
    max_bound: The maximum bounds of the voxel grid
@Output:
    results: The list of whether the rays hit an occupied voxel
    vectors: The list of vectors along the rays
'''
def check_perpendicular_rays_occupancy(point, direction, voxel_size, checking_distance, angle_interval, voxels, min_bound, max_bound):
    vectors, angles = generate_perpendicular_vectors(direction, angle_interval)
    
    results = []    
    for vec in vectors:
        hit = False
        normalized_vec = vec / np.linalg.norm(vec)
        for i in range(1, int(checking_distance / voxel_size) + 1):
            # Calculate the point along the ray
            ray_point = point + normalized_vec * i * voxel_size
            
            # Check if the point is occupied
            if is_point_occupied(ray_point, voxels, min_bound, max_bound, voxel_size):
                hit = True
                break
        
        results.append(hit)
    
    return results, vectors, angles

'''
@Description: Visualize the mesh, occupied voxels, and direction vectors
@Input:
    mesh: The trimesh mesh to visualize
    voxels: The 3D array of voxels
    voxel_size: The size of the voxels
    min_bound: The minimum bounds of the voxel grid
    start_point: The starting point of the direction vectors
    direction_vectors: The list of direction vectors
    results: The list of whether the rays hit an occupied voxel
'''
def visualize_mesh_voxels_vectors(mesh, voxels, voxel_size, min_bound, start_point, direction_vectors, results):
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.2, color='gray'))
    
    # Plot the occupied voxels
    occupied_voxel_coords = np.array(np.nonzero(voxels)).T * voxel_size + min_bound
    for voxel in occupied_voxel_coords:
        # Plot the voxel as a small cube
        ax.bar3d(voxel[0], voxel[1], voxel[2], voxel_size, voxel_size, voxel_size, color='blue', alpha=0.5)

    # Plot the direction vectors
    for vec, hit in zip(direction_vectors, results):
        end_point = start_point + vec  # Calculate the end point of the vector
        
        # Choose color based on whether the direction is occupied
        color = 'red' if hit else 'green'
        
        # Plot the vector
        ax.quiver(
            start_point[0], start_point[1], start_point[2], 
            vec[0], vec[1], vec[2], 
            color=color, length=1.0, normalize=True
        )
    
    # Set the limits for better visualization
    ax.set_xlim(min_bound[0], min_bound[0] + voxel_size * voxels.shape[0])
    ax.set_ylim(min_bound[1], min_bound[1] + voxel_size * voxels.shape[1])
    ax.set_zlim(min_bound[2], min_bound[2] + voxel_size * voxels.shape[2])
    
    plt.show()


'''
@Description: Transform the tenon and save the transformed tenon file
@Input:
    link: LinkResult object
    tenon_id: The id of the tenon
    tenon_file_folder: The folder for the tenon files
@Output:
    transformed_file_save_path: The path of the saved transformed tenon file
'''
def transform_tenon_and_save(link, tenon_mesh, tenon_id=0, unit='m', save_path=None, biased_tenon_length=0, tenon_orientation_vector=None):
    # We suppose the tenon is always along the +z-axis in the stl file and the tenon direction is always pointing to the +x-axis
    tenon_ori_direction_vector = np.array([0, 0, 1])
    print(f"tenon_orientation_vector: {tenon_orientation_vector}")

    tenon_direction_vector = np.array([link.tenon_pos[tenon_id][3], link.tenon_pos[tenon_id][4], link.tenon_pos[tenon_id][5]])
    tenon_direction_vector = tenon_direction_vector / np.linalg.norm(tenon_direction_vector)

    rotation_matrix = get_rotation_matrix(tenon_ori_direction_vector, tenon_direction_vector)

    # Add the biased length to the tenon position in case the tenon is too short for mesh
    biased_tenon_pos = np.array([link.tenon_pos[tenon_id][0], link.tenon_pos[tenon_id][1], link.tenon_pos[tenon_id][2]])
    biased_tenon_pos = biased_tenon_pos + tenon_direction_vector * biased_tenon_length

    # First transformation for the tenon to rotate to direction aligns with the link
    transformation1 = np.array([[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], 0],
                                [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0],
                                [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0],
                                [0, 0, 0, 1]])
    
    # Second transformation for the tenon along the tenon_direction_vector by tenon_orientation_angle
    tenon_orientation_ori_vector = np.array([1, 0, 0, 0])
    tenon_orientation_ori_vector_transformed1 = transformation1 @ tenon_orientation_ori_vector
    tenon_orientation_ori_vector_transformed1_cut = tenon_orientation_ori_vector_transformed1[:3]

    # Get transformation matrix for the tenon to rotate 
    if tenon_orientation_vector is not None:
        # Get rotation angle from tenon_orientation_ori_vector_transformed1_cut to tenon_orientation_vector
        angle = np.arccos(np.dot(tenon_orientation_ori_vector_transformed1_cut, tenon_orientation_vector))

        rotation_matrix2 = get_rotation_matrix_from_angle(tenon_direction_vector, angle)
        transformation2 = np.array([[rotation_matrix2[0, 0], rotation_matrix2[0, 1], rotation_matrix2[0, 2], 0],
                                    [rotation_matrix2[1, 0], rotation_matrix2[1, 1], rotation_matrix2[1, 2], 0],
                                    [rotation_matrix2[2, 0], rotation_matrix2[2, 1], rotation_matrix2[2, 2], 0],
                                    [0, 0, 0, 1]])
    else:
        transformation2 = np.eye(4)

    # Third transformation for the tenon to translate to the biased position
    transformation3 = np.array([[1, 0, 0, biased_tenon_pos[0]],
                                [0, 1, 0, biased_tenon_pos[1]],
                                [0, 0, 1, biased_tenon_pos[2]],
                                [0, 0, 0, 1]])
    if unit == 'm':
        print("Unit is in meter, scale the transformation matrix to mm")
        transformation3[:, 3] = transformation3[:, 3] * 1000
        transformation3[3, 3] = 1

    transformation = transformation3 @ transformation2 @ transformation1

    # transformation = transformation3 @ transformation1

    tenon_mesh = transform_trimesh(tenon_mesh, transformation, save_path=save_path)


'''
@Description: Run the metamaterial filling for a given stl file
@Input:
    input_stl_path: The path of the input stl file
    unit: The unit of the input stl file
    relative_density: The relative density of the metamaterial
    shell_thickness: The thickness of the shell
    shell_generation_voxel_resolution: The voxel resolution for shell generation
    plate_interval: The interval between the plates
    biased_tenon_length: The biased length of the tenon
    output_stl_name: The name of the output stl file
    use_existing_shell: Whether to use the existing shell
    pkl_result_path: The path of the pkl result file
    tenon_file_folder: The folder of the tenon files
    preview: Whether to preview the result
'''
def run_metamaterial_filling_for_stl_file(input_stl_path, unit, relative_density, shell_thickness, shell_generation_voxel_resolution, plate_interval, biased_tenon_length, output_stl_name, use_existing_shell, pkl_result_path, tenon_file_folder, preview):

    # Check if the input STL file exists
    if not os.path.exists(input_stl_path):
        raise FileNotFoundError(f'Input STL file not found at {input_stl_path}')
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cmake_build_dir = os.path.join(current_dir, '../build')
    output_folder = os.path.join(current_dir, '../data/output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    relative_density = relative_density
    if relative_density < 0.01:
        relative_density = 0.01
        print('Relative density should be larger than 0.01, set to 0.01')
    elif relative_density >= 1.0:
        raise ValueError('Relative density should be less than 1.0')
    
    # Check and read pkl file
    robot_result = pkl.load(open(pkl_result_path, 'rb'))

    # for link_name in robot_result.link_dict:
    #     print("Link name: ", link_name)
    #     print("Link Tenon Positions: ", robot_result.link_dict[link_name].tenon_pos)
    #     print("Link Torques: ", robot_result.link_dict[link_name].applied_torque)
    #     print("Link tenon_type: ", robot_result.link_dict[link_name].tenon_type)
    #     print("Link tenon_idx: ", robot_result.link_dict[link_name].tenon_idx)


    # Get the link name from the input stl path and get the corresponding link class from the robot result
    link_name = input_stl_path.split('/')[-1].split('.')[0]
    link = robot_result.link_dict[link_name]

    time_stamp_sec = time.time()

    #######  Find good orientation for tenon  ########
    # Load mesh and voxelize
    mesh = trimesh.load_mesh(input_stl_path)
    if unit == 'm':
        voxel_size = 0.005
        checking_distance = 0.1
    else:
        voxel_size = 5
        checking_distance = 100

    voxels, min_bound, max_bound, voxel_size = voxelize_mesh(mesh, voxel_size)
    tenon_center_top_bias = 15 # mm
    checking_angle_interval = np.pi / 24
    safe_angle_range = np.pi # 180 degrees

    # Find the best orientation for each tenon
    tenon_best_orientation_angles = []
    tenon_best_orientation_vectors = []
    for i in range(len(link.tenon_pos)): # For each tenon
        tenon_root_point = link.tenon_pos[i][:3]
        tenon_root_direction = link.tenon_pos[i][3:6]
        tenon_root_direction_norm = np.linalg.norm(tenon_root_direction)

        print(f"Tenon {i} root point: {tenon_root_point}")
        print(f"Tenon {i} root direction: {tenon_root_direction}")

        if unit == 'm':
            tenon_root_point_biased = tenon_root_point + tenon_root_direction_norm * 0.001 * tenon_center_top_bias
        else:
            tenon_root_point_biased = tenon_root_point + tenon_root_direction_norm * tenon_center_top_bias

        hit_results, vectors, angles = check_perpendicular_rays_occupancy(tenon_root_point_biased, tenon_root_direction, voxel_size, checking_distance, checking_angle_interval, voxels, min_bound, max_bound)


        # check if results and vectors have the same length
        if len(hit_results) != len(vectors):
            raise ValueError('Results and vectors have different lengths')

        # if preview:
        #     visualize_mesh_voxels_vectors(mesh, voxels, voxel_size, min_bound, tenon_root_point, vectors, hit_results)

        # Find the best direction. The best direction is the one whose adjacent directions are all free.
        best_direction = None
        adjacent_free_score_list = []
        range_min = int(- safe_angle_range / checking_angle_interval / 2)
        range_max = - range_min
        check_num = range_max - range_min + 1
        
        for j, hit in enumerate(hit_results):
            adjacent_free_score = 0    

            # Check the adjacent directions
            for k in range(range_min, range_max + 1):
                seq = j+k
                if seq < 0:
                    seq = len(hit_results) + seq
                elif seq >= len(hit_results):
                    seq = seq - len(hit_results)
                
                if hit_results[seq] == False:
                    # Use a V shape to calculate the score
                    adjacent_free_score += range_max + 1 - abs(k)

            adjacent_free_score_list.append(adjacent_free_score)

        # Find the index of the best direction whose adjacent directions are all free. 
        adjacent_free_num_array = np.array(adjacent_free_score_list)
        best_direction_index = np.argmax(adjacent_free_num_array)

        tenon_best_orientation_angles.append(angles[best_direction_index])
        tenon_best_orientation_vectors.append(vectors[best_direction_index])

        # Print the best direction and free adjacent direction number
        print(f"Tenon {i} best direction: {vectors[best_direction_index]}")
        print(f"Tenon {i} best direction angle: {angles[best_direction_index]}")
        print(f"Tenon {i} free adjacent direction number: {adjacent_free_num_array[best_direction_index]}")
        

    ###### Replace and scale the stl ######
    replaced_stl_name = input_stl_path.split('/')[-1].split('.')[0] + '_replaced.stl'
    replaced_stl_save_path = os.path.join(output_folder, replaced_stl_name)
    replaced_stl_size_csv = replaced_stl_save_path.replace('.stl', '.csv')

    input_stl_path_full_path = os.path.join(current_dir, '../', input_stl_path)

    print(f"input_stl_path_full_path: {input_stl_path_full_path}")
    print(f"replaced_stl_save_path: {replaced_stl_save_path}")

    # Remove the replaced file if it exists
    if os.path.exists(replaced_stl_save_path):
        os.remove(replaced_stl_save_path)
    if os.path.exists(replaced_stl_size_csv):
        os.remove(replaced_stl_size_csv)

    print("Generating the replaced model...")
    if unit == 'm':
        # Scale the model to mm
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./replaceMesh {input_stl_path_full_path} {replaced_stl_save_path} 1000']) #; exec bash
    else:
        # Do only the replacement
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./replaceMesh {input_stl_path_full_path} {replaced_stl_save_path}']) #; exec bash

    # Wait for the process to finish
    while not os.path.exists(replaced_stl_save_path):
        time.sleep(1)
    
    time.sleep(3)

    # Read the size of the replaced STL file from the CSV file. Two lines, the first line is min_x,min_y,min_z, the second line is max_x,max_y,max_z
    applied_scale = 1.0
    applied_placement = np.array([0, 0, 0])
    applied_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    with open(replaced_stl_size_csv, 'r') as f:
        lines = f.readlines()
        min_x, min_y, min_z = lines[0].split(',')
        max_x, max_y, max_z = lines[1].split(',')
        min_x, min_y, min_z = float(min_x), float(min_y), float(min_z)
        max_x, max_y, max_z = float(max_x), float(max_y), float(max_z)

        applied_scale = float(lines[2].split(',')[0])
        applied_placement = np.array([float(x) for x in lines[3].split(',')])
        applied_rotation = np.array([[float(x) for x in line.split(',')] for line in lines[4:7]])

        # print(f"min_x: {min_x} mm, min_y: {min_y} mm, min_z: {min_z} mm")
        # print(f"max_x: {max_x} mm, max_y: {max_y} mm, max_z: {max_z} mm")
        # print(f"applied_scale: {applied_scale}")
        # print(f"applied_placement: {applied_placement}")
        # print(f"applied_rotation: {applied_rotation}")

    
    ##### Transform tenons and save as stl files for later use in openscad #####
    print("Transforming tenons...")

    # Transform the tenons and save the transformed tenon files
    transformed_tenon_files = []                                        
    for i in range(len(link.tenon_pos)):
        #tenon_file_name = 'connection_' + link.tenon_type[i] + '.stl'

        tenon_file_name = 'motor_' + str(link.tenon_idx[i]) + '_' + link.tenon_type[i] + '.stl'

        tenon_file_path = os.path.join(tenon_file_folder, tenon_file_name)
        tenon_mesh = trimesh.load(tenon_file_path)

        # Add transformation for tenon pair alignment without interference
        #tenon_mesh = transform_trimesh(tenon_mesh, tenon_pre_transformation_matrix_list[i])

        # Transform to the right position in the stl file using the "link" result from Pickle file
        tenon_file_name_transformed = tenon_file_name.replace('.stl', '_' + str(i) + '_transformed.stl')
        file_save_path = os.path.join(output_folder, tenon_file_name_transformed)

        biased_tenon_length = biased_tenon_length
        if unit == 'm':
            biased_tenon_length = biased_tenon_length * 0.001
        
        # Transform the tenon and save the transformed tenon file considering the best orientation
        transform_tenon_and_save(link, tenon_mesh, i, unit=unit, save_path=file_save_path, biased_tenon_length=biased_tenon_length, tenon_orientation_vector=tenon_best_orientation_vectors[i])

        # Use the following to test the tenon without considering the best orientation
        # transform_tenon_and_save(link, tenon_mesh, i, unit=unit, save_path=file_save_path, biased_tenon_length=biased_tenon_length, tenon_orientation_vector=None)

        transformed_tenon_files.append(file_save_path)
        print(f"Transformed tenon file saved at {file_save_path}")

    # Transform again based on the placement and rotation for the replaced model
    second_transformation_matrix = np.eye(4)
    third_transformation_matrix = np.eye(4)
    second_transformation_matrix[:3, 3] = applied_placement
    third_transformation_matrix[:3, :3] = applied_rotation
    print(f"Second transformation: {second_transformation_matrix}")
    print(f"Third transformation: {third_transformation_matrix}")

    final_transformed_tenon_files = []
    for i in range(len(transformed_tenon_files)):
        transformed_tenon_file_path = transformed_tenon_files[i]
        transformed_tenon_mesh = trimesh.load(transformed_tenon_file_path)

        transformed_file_save_path = transformed_tenon_file_path.replace('.stl', '_second_transformed.stl')

        transformed_tenon_mesh = transform_trimesh(transformed_tenon_mesh, second_transformation_matrix)
        transform_trimesh(transformed_tenon_mesh, third_transformation_matrix, save_path=transformed_file_save_path)

        final_transformed_tenon_files.append(transformed_file_save_path)

    print(f"Final transformed tenon files: {final_transformed_tenon_files}")

    ##### Preview the transformed tenons and the link  #####
    if preview:
        stl_to_visualize = [replaced_stl_save_path] + final_transformed_tenon_files
        eye_transformation_matrix = np.eye(4)
        transformation_matrices_vis = [eye_transformation_matrix]
        scales_vis = [1.0]
        for i in range (len(final_transformed_tenon_files)):
            transformation_matrices_vis.append(eye_transformation_matrix)
            scales_vis.append(1.0)

        visualize_meshes(stl_to_visualize, transformation_matrices_vis, scales_vis)

    ###### Generate the shell ######
    # smaller_model_stl_name = input_stl_path.split('/')[-1].split('.')[0] + '_smaller.stl'
    smaller_model_stl_name = replaced_stl_name.replace('.stl', '_smaller.stl')
    smaller_stl_save_path = os.path.join(output_folder, smaller_model_stl_name)
    smaller_stl_points_bin = smaller_stl_save_path.replace('.stl', '_points.bin')

    # Remove the smaller_model for shell file if it exists and we are not using the existing model. For quick testing.
    do_smaller_generation = True
    if os.path.exists(smaller_stl_save_path):
        if use_existing_shell:
            print(f"Using existing shell file at {smaller_stl_save_path}")
            do_smaller_generation = False
        else:
            os.remove(smaller_stl_points_bin)
            os.remove(smaller_stl_save_path)

    if do_smaller_generation:
        print("Generating the smaller model for shell...")
        # Use gnome-terminal to run the command
        shell_thickness = shell_thickness
        shell_generation_voxel_resolution = shell_generation_voxel_resolution
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./innerPointsCalculation {replaced_stl_save_path} {smaller_stl_points_bin} {shell_thickness} {shell_generation_voxel_resolution}']) #; exec bash

        # Wait for the process to finish
        while not os.path.exists(smaller_stl_points_bin):
            time.sleep(1)

        time.sleep(3)
        
        print(f"smaller model point bin file generated at {smaller_stl_points_bin}")

        # Do marching cubes to generate the smaller model
        generate_mesh_from_points(smaller_stl_points_bin, shell_generation_voxel_resolution, smaller_stl_save_path)

        print(f"smaller model generated at {smaller_stl_save_path}")

    # Downsample the smaller model to reduce the number of faces. Otherwise the final model will be too dense and hard to do metamaterial filling.
    
    mesh_to_downsample = trimesh.load(smaller_stl_save_path)

    #### FOR OLD VERSION OF TRIMESH
    # down_sample_ratio = 10
    # ori_face_num = len(mesh_to_downsample.faces)
    # print(f"Original face number: {ori_face_num}")
    # down_sampled_face_num = int(ori_face_num / down_sample_ratio)
    # down_sampled_face_num = max(down_sampled_face_num, 1000) # At least 1000 points
    # print(f"Expected Downsampled point number: {down_sampled_face_num}")
    # down_sampled_mesh = mesh_to_downsample.simplify_quadratic_decimation(down_sampled_face_num)
    # down_sampled_mesh.export(smaller_stl_save_path)
    

    ### FOR NEW VERSION OF TRIMESH
    down_sample_ratio = 0.1
    down_sampled_mesh = mesh_to_downsample.simplify_quadric_decimation(down_sample_ratio)
    down_sampled_mesh.export(smaller_stl_save_path)

    ########## Generate the final model ##########
    tilt_angle = 30 # degrees
    out_stl_name = output_stl_name

    final_output_stl_path = os.path.join(output_folder, out_stl_name)

    # Define the dimensions of the board
    safe_scale = 1.5 # Safe scale to ensure the rotated plates can cover the whole model
    height = (max_z - min_z) * safe_scale
    delf_x = max_x - min_x
    delf_y = max_y - min_y
    width = max(delf_x, delf_y) * safe_scale

    interval = plate_interval # mm. The interval between plates
    thickness = relative_density / 6.0 * interval

    # Check if thickness is smaller than 0.2
    if thickness < 0.2:
        thickness = 0.2
        print('Thickness should be smaller than 0.2, set to 0.2')
        # Correct the interval
        # interval = thickness * 6.0 / relative_density  # No change to the interval for now

        print(f"Corrected interval: {interval}")
    
    # Make thickness a int multiple of 0.1
    thickness = round(thickness / 0.1) * 0.1

    plates_num = int(width / (thickness + interval) / 2)


    # Generate the final model
    print("Generating the final model with Openscad... Parameters:")
    print(f"height: {height}, width: {width}, thickness: {thickness}, interval: {interval}, plates_num: {plates_num}, tilt_angle: {tilt_angle}")
    print("This step usually takes more than 15 minutes. Please be patient.")

    sixFoldPlatesFilling = SixFoldPlatesFillingWithShellTenon(height, width, thickness, interval, plates_num, tilt_angle, replaced_stl_save_path, smaller_stl_save_path, final_transformed_tenon_files, final_output_stl_path)
    sixFoldPlatesFilling.generate_model()

    print(f"Final model generated at {final_output_stl_path}")

    print("All done! When you do 3d FDM printing, please make sure the model is placed in the generated orientation.")

    # Print the time used
    time_stamp_sec_end = time.time()
    time_used = time_stamp_sec_end - time_stamp_sec
    print(f"Time used: {time_used} seconds, which is {time_used / 60} minutes.")



'''
@Description: Main function to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stl_path', type=str, default=project_dir + '/urdf/gold_lynel20241008-182954/BODY.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--relative_density', type=float, default=0.1, help='Relative density of the metamaterial given by FEA results')
    
    parser.add_argument('--shell_thickness', type=float, default=1.5, help='Thickness of the shell. mm')
    parser.add_argument('--shell_generation_voxel_resolution', type=float, default=0.5, help='Voxel resolution for shell generation. mm')
    
    parser.add_argument('--plate_interval', type=float, default=8, help='Interval between plates. mm')
    parser.add_argument('--biased_tenon_length', type=float, default=0, help='Biased length for the tenon. mm')

    parser.add_argument('--output_stl_name', type=str, default='20241008-163714_BODY_final_output_with_shell.stl', help='Output STL file path')
    parser.add_argument('--use_existing_shell', type=bool, default=False, help='Whether to use the existing shell file')
    
    parser.add_argument('--pkl_result_path', type=str, default=project_dir+'/auto_design/results/gold_lynel20241008-182958_robot_result.pkl', help='Pickle file path for the tenon position results')
    parser.add_argument('--tenon_file_folder', type=str, default=project_dir+'/metamaterial_filling/tenon', help='Folder for the tenon files')
                        
    parser.add_argument('--preview', type=bool, default=True, help='Whether to visualize the transformed tenons and the link')

    args = parser.parse_args()

    run_metamaterial_filling_for_stl_file(args.input_stl_path, args.unit, args.relative_density, args.shell_thickness, args.shell_generation_voxel_resolution, args.plate_interval, args.biased_tenon_length, args.output_stl_name, args.use_existing_shell, args.pkl_result_path, args.tenon_file_folder, args.preview)

