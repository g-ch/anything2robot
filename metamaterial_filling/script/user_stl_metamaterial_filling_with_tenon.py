'''
@Author: Clarence
@Date: 2024-7-5
@Description: This script is used to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''

import os
import argparse
import subprocess
import sys
import shutil
import shlex

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir + '/metamaterial_filling/script')
sys.path.append(project_dir + '/auto_design/modules')
sys.path.append(project_dir + '/auto_design')
sys.path.append(project_dir)


from metamaterial.sixFoldPlatesFillingWithShellTenon import SixFoldPlatesFillingWithShellTenon
from io_interface.robot_result_compat import load_robot_result
from mesh_operations.mesh_difference import mesh_difference
from mesh_operations.create_box import create_box
from mesh_operations.create_cylinder import create_cylinder
from mesh_operations.repair_mesh import repair_mesh

import time
import numpy as np

from visualization.assemble_vis import get_rotation_matrix, visualize_meshes, transform_trimesh, get_rotation_matrix_from_angle

import trimesh

from script.motor_param_lib import MotorParameterLib

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
    voxel_grid = trimesh_mesh.voxelized(pitch=voxel_size)
    points = voxel_grid.points
    if len(points) == 0:
        raise ValueError("Voxelization produced no occupied cells")

    min_bound = points.min(axis=0) - voxel_size / 2.0
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    shape = voxel_indices.max(axis=0) + 1
    max_bound = min_bound + shape * voxel_size

    voxels = np.zeros(shape, dtype=bool)
    voxels[tuple(voxel_indices.T)] = True

    return voxels, min_bound, max_bound, voxel_size


def run_build_tool(cmake_build_dir, executable_name, *args):
    executable_path = os.path.join(cmake_build_dir, executable_name)
    if not os.path.exists(executable_path):
        raise FileNotFoundError(f"Required build tool not found: {executable_path}")

    command = [executable_path, *[str(arg) for arg in args]]
    print("Running build tool:", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


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
def generate_perpendicular_vectors(direction, angle_interval, angle_range=np.pi*2):
    # Ensure direction is a unit vector
    direction = direction / np.linalg.norm(direction)
    angle_half_range = angle_range / 2
    
    # Find two vectors perpendicular to the direction
    if np.allclose(direction, [1, 0, 0]) or np.allclose(direction, [-1, 0, 0]):
        v1 = np.array([0, 1, 0])
        angles = np.arange(-angle_half_range, angle_half_range, angle_interval) - np.pi / 2
    else:
        v1 = np.cross(direction, [1, 0, 0])
        angles = np.arange(-angle_half_range, angle_half_range, angle_interval)
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(direction, v1)
    
    # Sample angles
    vectors = [np.cos(angle) * v1 + np.sin(angle) * v2 for angle in angles]

    normalized_vectors = []
    for vec in vectors:
        normalized_vectors.append(vec / np.linalg.norm(vec))

    # print(f"Direction: {direction}")
    # print(f"Perpendicular vectors: {vectors}")
    
    return normalized_vectors, angles


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
    results: The list of the number of occupied voxels along the rays
    vectors: The list of vectors along the rays
'''
def check_perpendicular_rays_occupancy(point, direction, voxel_size, checking_distance, angle_interval, voxels, min_bound, max_bound):
    vectors, angles = generate_perpendicular_vectors(direction, angle_interval, angle_range=np.pi*2)
    results = []    
    for vec in vectors:
        hit_num = 0
        normalized_vec = vec / np.linalg.norm(vec)
        for i in range(1, int(checking_distance / voxel_size) + 1):
            # Calculate the point along the ray
            ray_point = point + normalized_vec * i * voxel_size
            
            # Check if the point is occupied
            if is_point_occupied(ray_point, voxels, min_bound, max_bound, voxel_size):
                hit_num += 1
        
        results.append(hit_num)
    
    return results, vectors, angles



'''
@Description: Transform the tenon and save the transformed tenon file
@Input:
    link: LinkResult object
    tenon_id: The id of the tenon
    tenon_file_folder: The folder for the tenon files
@Output:
    transformed_file_save_path: The path of the saved transformed tenon file
'''
def transform_tenon_and_save(link, tenon_mesh, tenon_id=0, unit='m', save_path=None, biased_tenon_distance=0, tenon_orientation_vector=None):
    # We suppose the tenon is always along the +z-axis in the stl file and the tenon direction is always pointing to the +x-axis
    tenon_ori_direction_vector = np.array([0, 0, 1])
    print(f"tenon_orientation_vector: {tenon_orientation_vector}")

    tenon_direction_vector = np.array([link.tenon_pos[tenon_id][3], link.tenon_pos[tenon_id][4], link.tenon_pos[tenon_id][5]])
    tenon_direction_vector = tenon_direction_vector / np.linalg.norm(tenon_direction_vector)

    rotation_matrix = get_rotation_matrix(tenon_ori_direction_vector, tenon_direction_vector)

    # Add the biased length to the tenon position in case the tenon is too short for mesh
    biased_tenon_pos = np.array([link.tenon_pos[tenon_id][0], link.tenon_pos[tenon_id][1], link.tenon_pos[tenon_id][2]])
    biased_tenon_pos = biased_tenon_pos + tenon_direction_vector * biased_tenon_distance

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
        dot_product = np.clip(np.dot(tenon_orientation_ori_vector_transformed1_cut, tenon_orientation_vector), -1.0, 1.0)
        angle = np.arccos(dot_product)

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


def ensure_motor_tenon_template(tenon_file_folder, tenon_idx, tenon_type):
    tenon_file_name = f'motor_{tenon_idx}_{tenon_type}.stl'
    tenon_file_path = os.path.join(tenon_file_folder, tenon_file_name)
    if os.path.exists(tenon_file_path):
        return tenon_file_path

    motor_param = MotorParameterLib()
    target_radius_mm = motor_param.motor_lib[tenon_idx][1] * 10.0

    available_templates = []
    for base_idx in range(len(motor_param.motor_lib)):
        base_path = os.path.join(tenon_file_folder, f'motor_{base_idx}_{tenon_type}.stl')
        if os.path.exists(base_path):
            base_radius_mm = motor_param.motor_lib[base_idx][1] * 10.0
            available_templates.append((abs(base_radius_mm - target_radius_mm), base_idx, base_path, base_radius_mm))

    if not available_templates:
        raise FileNotFoundError(f'No template found for missing tenon file {tenon_file_path}')

    _, base_idx, base_path, base_radius_mm = min(available_templates, key=lambda item: item[0])
    scale_xy = target_radius_mm / base_radius_mm

    print(
        f'Missing tenon template {tenon_file_name}. '
        f'Generating it by scaling motor_{base_idx}_{tenon_type}.stl with xy scale {scale_xy:.6f}.'
    )

    tenon_mesh = trimesh.load(base_path, force='mesh')
    tenon_mesh.apply_scale([scale_xy, scale_xy, 1.0])
    tenon_mesh.export(tenon_file_path)
    return tenon_file_path


'''
@Description: Run the metamaterial filling for a given stl file
@Input:
    input_stl_path: The path of the input stl file
    unit: The unit of the input stl file
    relative_density: The relative density of the metamaterial
    shell_thickness: The thickness of the shell
    shell_generation_voxel_resolution: The voxel resolution for shell generation
    plate_interval: The interval between the plates
    biased_tenon_distance: The biased length of the tenon
    output_stl_name: The name of the output stl file
    use_existing_shell: Whether to use the existing shell
    pkl_result_path: The path of the pkl result file
    tenon_file_folder: The folder of the tenon files
    preview: Whether to preview the result
'''
def run_metamaterial_filling_for_stl_file(input_stl_path, unit, relative_density, shell_thickness, shell_generation_voxel_resolution, plate_interval, biased_tenon_distance, output_stl_name, use_existing_shell, pkl_result_path, tenon_file_folder, preview, output_dir=None):

    # Check if the input STL file exists
    if not os.path.exists(input_stl_path):
        raise FileNotFoundError(f'Input STL file not found at {input_stl_path}')
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cmake_build_dir = os.path.join(current_dir, '../build')
    output_folder = output_dir or os.path.join(current_dir, '../data/output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    relative_density = relative_density
    if relative_density < 0.01:
        relative_density = 0.01
        print('Relative density should be larger than 0.01, set to 0.01')
    elif relative_density >= 1.0:
        raise ValueError('Relative density should be less than 1.0')
    
    # Check and read pkl file
    robot_result = load_robot_result(pkl_result_path)

    # for link_name in robot_result.link_dict:
    #     print("Link name: ", link_name)
    #     print("Link Tenon Positions: ", robot_result.link_dict[link_name].tenon_pos)
    #     print("Link Torques: ", robot_result.link_dict[link_name].applied_torque)
    #     print("Link tenon_type: ", robot_result.link_dict[link_name].tenon_type)
    #     print("Link tenon_idx: ", robot_result.link_dict[link_name].tenon_idx)

    # Remove the last element of the tenon_pos list for the BODY link because it exceeds our Bamboolab printer's workspace
    # robot_result.link_dict["BODY"].tenon_pos.pop()
    # robot_result.link_dict["BODY"].applied_torque.pop()
    # robot_result.link_dict["BODY"].tenon_type.pop()
    # robot_result.link_dict["BODY"].tenon_idx.pop()

    # Get the link name from the input stl path and get the corresponding link class from the robot result
    link_name = input_stl_path.split('/')[-1].split('.')[0]
    link = robot_result.link_dict[link_name]

    time_stamp_sec = time.time()

    #######  Find good orientation for tenon  ########
    # Load mesh and voxelize
    mesh = trimesh.load_mesh(input_stl_path)    

    # CHECK if mesh is water tight
    if not mesh.is_watertight:
        raise ValueError('Input mesh is not watertight')
    else:
        print("Input mesh is watertight. Conitnue...")

    if unit == 'm':
        voxel_size = 0.002
        checking_distance = 0.2
    else:
        voxel_size = 2
        checking_distance = 200

    voxels, min_bound, max_bound, voxel_size = voxelize_mesh(mesh, voxel_size)

    tenon_center_top_bias = [5, 10, 20, 30]  # Check different heights. mm
    checking_angle_interval = np.pi / 24
    safe_angle_range = np.pi 

    # Find the best orientation for each tenon
    tenon_best_orientation_angles = []
    tenon_best_orientation_vectors = []
    for i in range(len(link.tenon_pos)): # For each tenon
        print(f"Finding the best orientation for tenon {i}...")
        tenon_root_point = link.tenon_pos[i][:3]
        tenon_root_direction = link.tenon_pos[i][3:6]
        tenon_root_direction = tenon_root_direction / np.linalg.norm(tenon_root_direction)        

        # Calculate the hit numbers for different heights
        hit_nums = []
        for k in range(len(tenon_center_top_bias)):
            if unit == 'm':
                tenon_root_point_biased = tenon_root_point + tenon_root_direction * tenon_center_top_bias[k] * 0.001
            else:
                tenon_root_point_biased = tenon_root_point + tenon_root_direction * tenon_center_top_bias[k]

            hit_nums_this, vectors, angles = check_perpendicular_rays_occupancy(tenon_root_point_biased, tenon_root_direction, voxel_size, checking_distance, checking_angle_interval, voxels, min_bound, max_bound)

            if len(hit_nums) == 0:
                hit_nums = hit_nums_this
            else:
                for l in range (len(hit_nums)):
                    hit_nums[l] += hit_nums_this[l]

        # check if results and vectors have the same length
        if len(hit_nums) != len(vectors):
            raise ValueError('Results and vectors have different lengths')

        # if preview:
        #     visualize_mesh_voxels_vectors(mesh, voxels, voxel_size, min_bound, tenon_root_point, vectors, hit_nums)

        # Find the best direction. 
        adjacent_free_score_list = []
        range_min = int(- safe_angle_range / checking_angle_interval / 2)
        range_max = -range_min
        
        for j, hit_num in enumerate(hit_nums):
            adjacent_free_score = 0    

            # Check the adjacent directions
            for k in range(range_min, range_max + 1):
                seq = j+k
                if seq < 0:
                    seq = len(hit_nums) + seq
                elif seq >= len(hit_nums):
                    seq = seq - len(hit_nums)
                
                if hit_nums[seq] > 0:
                    # Use a V shape to calculate the score
                    adjacent_free_score += (range_max + 1 - abs(k)) * hit_nums[seq]  

            adjacent_free_score_list.append(adjacent_free_score)

        # Find the index of the best direction whose adjacent directions are all free. 
        adjacent_free_num_array = np.array(adjacent_free_score_list)
        best_direction_index = np.argmin(adjacent_free_num_array)

        tenon_best_orientation_angles.append(angles[best_direction_index])
        tenon_best_orientation_vectors.append(vectors[best_direction_index])

        # Visulize the best direction vectors using pyvista
        # if preview:
        #     root_points_array =np.tile(tenon_root_point_biased, (len(vectors)+1, 1))
        #     vectors.append(tenon_root_direction)
        #     vectors_array = np.array(vectors) * checking_distance
        #     end_points_array = root_points_array + vectors_array
        #     adjacent_free_score_list.append(0)
        #     adjacent_free_score_array = np.array(adjacent_free_score_list)
        #     plot_mesh_with_arrows_and_colorbar(mesh, root_points_array, end_points_array, adjacent_free_score_array, colormap_name='viridis')

        #     p = pv.Plotter()
        #     p.add_mesh(mesh, color='grey')
        #     p.add_arrows(tenon_root_point, tenon_root_point + vectors[best_direction_index], mag=1, color='red')
        #     p.show()

    # Free the memory for the voxels
    voxels = None

    ###### Replace and scale the stl ######
    replaced_stl_name = input_stl_path.split('/')[-1].split('.')[0] + '_replaced.stl'
    replaced_stl_save_path = os.path.join(output_folder, replaced_stl_name)
    replaced_stl_size_csv = replaced_stl_save_path.replace('.stl', '.csv')

    input_stl_path_full_path = os.path.abspath(input_stl_path)

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
        run_build_tool(cmake_build_dir, 'replaceMesh', input_stl_path_full_path, replaced_stl_save_path, 1000)
    else:
        # Do only the replacement
        run_build_tool(cmake_build_dir, 'replaceMesh', input_stl_path_full_path, replaced_stl_save_path)

    if not os.path.exists(replaced_stl_save_path):
        raise FileNotFoundError(f"replaceMesh did not create {replaced_stl_save_path}")

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

    
    ##### Transform tenons and save as stl files for later use in openscad #####
    print("Transforming tenons...")
    
    # Transform the tenons and save the transformed tenon files
    transformed_tenon_files = []
    assembling_interference_removal_box_files = [] # Use box to remove the interference during assembling
    assembling_interference_removal_cylinder_files = [] # use cylinder to refine the space for the motor                                  
    for i in range(len(link.tenon_pos)):

        ##### Transform the tenon mesh #####
        tenon_file_name = 'motor_' + str(link.tenon_idx[i]) + '_' + link.tenon_type[i] + '.stl'
        tenon_file_path = ensure_motor_tenon_template(tenon_file_folder, link.tenon_idx[i], link.tenon_type[i])
        tenon_mesh = trimesh.load(tenon_file_path)

        # Transform to the right position in the stl file using the "link" result from Pickle file
        tenon_file_name_transformed = tenon_file_name.replace('.stl', '_' + str(i) + '_transformed.stl')
        file_save_path = os.path.join(output_folder, tenon_file_name_transformed)

        biased_tenon_distance_this = biased_tenon_distance
        if unit == 'm':
            biased_tenon_distance_this = biased_tenon_distance * 0.001
        
        # Transform the tenon and save the transformed tenon file considering the best orientation
        transform_tenon_and_save(link, tenon_mesh, i, unit=unit, save_path=file_save_path, biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=tenon_best_orientation_vectors[i])

        # # Use the following to test the tenon without considering the best orientation
        # transform_tenon_and_save(link, tenon_mesh, i, unit=unit, save_path=file_save_path, biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=None)

        transformed_tenon_files.append(file_save_path)
        print(f"Transformed tenon file saved at {file_save_path}")

        ##### Transform the box mesh for assembling interference removal  #####
        # Define the box parameters for assembling interference removal
        box_normal_vector = [1, 0, 0] # initial direction of the box before transformation
        box_width_direction = [0, 0, 1]
        box_height = 150

        motor_param = MotorParameterLib() # Get the motor parameters
        box_face_length = motor_param.motor_lib[link.tenon_idx[i]][1] * 20 + 2 # from cm to mm and radius to diameter
        box_face_width = motor_param.motor_lib[link.tenon_idx[i]][0] * 10  + motor_param.tenon_height * 10 + 10 # from cm to mm. Add margin for tenon
        box_face_center = [0,0,box_face_width/2]

        box_save_path = file_save_path.replace('.stl', '_box.stl')
        box_mesh = create_box(box_face_center, box_face_length, box_face_width, box_normal_vector, box_width_direction, box_height)

        # define a cylinder to refine the space for the motor
        cylinder_face_center = [0, 0, biased_tenon_distance_this]
        cylinder_radius = motor_param.motor_lib[link.tenon_idx[i]][1] * 10  # from cm to mm
        cylinder_normal_vector = [0, 0, 1]
        cylinder_height = motor_param.motor_lib[link.tenon_idx[i]][0] * 10  + motor_param.tenon_height * 10
        cylinder_mesh = create_cylinder(cylinder_face_center, cylinder_radius, cylinder_normal_vector, cylinder_height)
        
        # Transform the box and cylinder meshes to the right position in the stl file using the "link" result from Pickle file
        transform_tenon_and_save(link, box_mesh, i, unit=unit, save_path=box_save_path, biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=tenon_best_orientation_vectors[i])
        transform_tenon_and_save(link, cylinder_mesh, i, unit=unit, save_path=box_save_path.replace('.stl', '_cylinder.stl'), biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=tenon_best_orientation_vectors[i])

        # # Use the following to test the tenon without considering the best orientation
        # transform_tenon_and_save(link, box_mesh, i, unit=unit, save_path=box_save_path, biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=None)
        # transform_tenon_and_save(link, cylinder_mesh, i, unit=unit, save_path=box_save_path.replace('.stl', '_cylinder.stl'), biased_tenon_distance=biased_tenon_distance_this, tenon_orientation_vector=None)

        assembling_interference_removal_box_files.append(box_save_path)
        assembling_interference_removal_cylinder_files.append(box_save_path.replace('.stl', '_cylinder.stl'))

    # Transform again based on the placement and rotation for the replaced model
    second_transformation_matrix = np.eye(4)
    third_transformation_matrix = np.eye(4)
    second_transformation_matrix[:3, 3] = applied_placement
    third_transformation_matrix[:3, :3] = applied_rotation
    print(f"Second transformation: {second_transformation_matrix}")
    print(f"Third transformation: {third_transformation_matrix}")

    final_transformed_tenon_files = []
    final_transformed_box_files = []
    final_transformed_cylinder_files = []
    for i in range(len(transformed_tenon_files)):
        # Transform the tenon
        transformed_tenon_file_path = transformed_tenon_files[i]
        transformed_tenon_mesh = trimesh.load(transformed_tenon_file_path)

        transformed_file_save_path = transformed_tenon_file_path.replace('.stl', '_second_transformed.stl')
        transformed_tenon_mesh = transform_trimesh(transformed_tenon_mesh, second_transformation_matrix)
        transform_trimesh(transformed_tenon_mesh, third_transformation_matrix, save_path=transformed_file_save_path)

        final_transformed_tenon_files.append(transformed_file_save_path)

        # Transform the box mesh for assembling interference removal
        transformed_box_file_path = assembling_interference_removal_box_files[i]
        transformed_box_mesh = trimesh.load(transformed_box_file_path)

        transformed_box_file_save_path = transformed_box_file_path.replace('.stl', '_second_transformed.stl')
        transformed_box_mesh = transform_trimesh(transformed_box_mesh, second_transformation_matrix)
        transform_trimesh(transformed_box_mesh, third_transformation_matrix, save_path=transformed_box_file_save_path)

        final_transformed_box_files.append(transformed_box_file_save_path)

        # Transform the cylinder mesh for assembling interference removal
        transformed_cylinder_file_path = assembling_interference_removal_cylinder_files[i]
        transformed_cylinder_mesh = trimesh.load(transformed_cylinder_file_path)

        transformed_cylinder_file_save_path = transformed_cylinder_file_path.replace('.stl', '_second_transformed.stl')
        transformed_cylinder_mesh = transform_trimesh(transformed_cylinder_mesh, second_transformation_matrix)
        transform_trimesh(transformed_cylinder_mesh, third_transformation_matrix, save_path=transformed_cylinder_file_save_path)

        final_transformed_cylinder_files.append(transformed_cylinder_file_save_path)

    # print(f"Final transformed tenon files: {final_transformed_tenon_files}")
    # print(f"Final transformed box files: {final_transformed_box_files}")
    # print(f"Final transformed cylinder files: {final_transformed_cylinder_files}")

    ##### Preview the transformed tenons and the link  #####
    if preview:
        eye_transformation_matrix = np.eye(4)
        transformation_matrices_vis = [eye_transformation_matrix]
        scales_vis = [1.0]
        for i in range (len(final_transformed_tenon_files)):
            transformation_matrices_vis.append(eye_transformation_matrix)
            scales_vis.append(1.0)
        # Preview the boxes and cylinders
        for i in range (len(final_transformed_box_files) + len(final_transformed_cylinder_files)):
            transformation_matrices_vis.append(eye_transformation_matrix)
            scales_vis.append(1.0)
        stls_to_visualize = [replaced_stl_save_path] + final_transformed_tenon_files + final_transformed_box_files + final_transformed_cylinder_files
        # stls_to_visualize = [replaced_stl_save_path] + final_transformed_tenon_files

        visualize_meshes(stls_to_visualize, transformation_matrices_vis, scales_vis)

    ###### Do mesh based assembling interference removal to clear place for motor insertion ######
    print("Doing mesh based assembling interference removal with cylinders...")
    for file in final_transformed_cylinder_files:
        # Back up the replaced stl file
        shutil.copy(replaced_stl_save_path, replaced_stl_save_path.replace('.stl', '_backup.stl'))
        # Do mesh difference
        success = mesh_difference(replaced_stl_save_path, file, replaced_stl_save_path)
        if not success:
            # Restore the replaced stl file
            print("Restoring the replaced stl file...")
            shutil.copy(replaced_stl_save_path.replace('.stl', '_backup.stl'), replaced_stl_save_path)
            continue

    print("Doing mesh based assembling interference removal with boxes...")
    for file in final_transformed_box_files:
        # Back up the replaced stl file
        shutil.copy(replaced_stl_save_path, replaced_stl_save_path.replace('.stl', '_backup.stl'))
        # Do mesh difference
        success = mesh_difference(replaced_stl_save_path, file, replaced_stl_save_path)
        if not success:
            # Restore the replaced stl file
            print("Restoring the replaced stl file...")
            shutil.copy(replaced_stl_save_path.replace('.stl', '_backup.stl'), replaced_stl_save_path)
            continue
    
    print("Mesh based assembling interference removal done.")

    # Repair the mesh caused by the mesh difference
    print("Repairing the mesh...")
    repair_mesh(replaced_stl_save_path, replaced_stl_save_path)

    ###### Generate a smaller model for the shell ######
    # smaller_model_stl_name = input_stl_path.split('/')[-1].split('.')[0] + '_smaller.stl'
    smaller_model_stl_name = replaced_stl_name.replace('.stl', '_smaller.stl')
    smaller_stl_save_path = os.path.join(output_folder, smaller_model_stl_name)
    smaller_stl_points_bin = smaller_stl_save_path.replace('.stl', '_points.bin')

    # In Test mode, shell and metamaterial filling are not generated but tenon is added for quick testing
    TEST_MODE_NO_SHELL_NO_INNER = False
    if shell_thickness is None or shell_generation_voxel_resolution is None:
        TEST_MODE_NO_SHELL_NO_INNER = True

    if not TEST_MODE_NO_SHELL_NO_INNER:
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
            run_build_tool(
                cmake_build_dir,
                'innerPointsCalculation',
                replaced_stl_save_path,
                smaller_stl_points_bin,
                shell_thickness,
                shell_generation_voxel_resolution,
            )

            if not os.path.exists(smaller_stl_points_bin):
                raise FileNotFoundError(f"innerPointsCalculation did not create {smaller_stl_points_bin}")
            
            print(f"smaller model point bin file generated at {smaller_stl_points_bin}")

            # Do marching cubes to generate the smaller model
            from metamaterial.generateMeshFromPoints import generate_mesh_from_points

            generate_mesh_from_points(smaller_stl_points_bin, shell_generation_voxel_resolution, smaller_stl_save_path)

            print(f"smaller model generated at {smaller_stl_save_path}")

        # Check if the smaller model has only one piece and is water-tight using open3d
        mesh_to_check = trimesh.load(smaller_stl_save_path)
        
        # Check if the mesh is watertight
        is_watertight = mesh_to_check.is_watertight
        print(f"Is the smaller model watertight: {is_watertight}")

        # Check if the mesh has only one piece
        mesh_pieces = mesh_to_check.split()
        print(f"Number of pieces in the smaller model: {len(mesh_pieces)}")

        # If not watertight or has more than one piece, do the following to fix the mesh
        if not is_watertight or len(mesh_pieces) > 1:
            print("Fixing the mesh...")
            trimesh.repair.fill_holes(mesh_to_check)
            trimesh.repair.fix_inversion(mesh_to_check)
            trimesh.repair.fix_winding(mesh_to_check)
            trimesh.repair.fix_normals(mesh_to_check)

            # Check if the mesh is watertight
            is_watertight = mesh_to_check.is_watertight
            print(f"Is the smaller model watertight after fixing: {is_watertight}")

            # Check if the mesh has only one piece
            mesh_pieces = mesh_to_check.split()
            print(f"Number of pieces in the smaller model after fixing: {len(mesh_pieces)}")

            if not is_watertight or len(mesh_pieces) > 1:
                raise ValueError('The smaller model is not watertight or has more than one piece after fixing')
        

    ########## Generate the final model ##########
    tilt_angle = 30 # degrees
    out_stl_name = output_stl_name

    final_output_stl_path = os.path.join(output_folder, out_stl_name)

    # Define the dimensions of the board
    safe_scale = 1.5
    height = (max_z - min_z) * safe_scale
    delf_x = max_x - min_x
    delf_y = max_y - min_y 
    width = max(delf_x, delf_y) * safe_scale

    interval = plate_interval # mm. The interval between plates
    if not TEST_MODE_NO_SHELL_NO_INNER:
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
    else:
        plates_num = None
        thickness = None
        interval = None


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

    parser.add_argument('--input_stl_path', type=str, default=project_dir + '/result/gold_lynel_20241201-134522_good/result_round1/urdf/RL_UP.stl', help='Input STL file path')
    #parser.add_argument('--input_stl_path', type=str, default='/home/cc/Downloads/BODY_UP.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--relative_density', type=float, default=0.15, help='Relative density of the metamaterial given by FEA results')

    parser.add_argument('--shell_thickness', type=float, default=0, help='Thickness of the shell. mm')
    parser.add_argument('--shell_generation_voxel_resolution', type=float, default=1.5, help='Voxel resolution for shell generation. mm')
    
    parser.add_argument('--plate_interval', type=float, default=10, help='Interval between plates. mm')
    parser.add_argument('--biased_tenon_distance', type=float, default=2.5, help='Biased length for the tenon. mm')

    parser.add_argument('--output_stl_name', type=str, default='FR_UP_final_output_no_shell.stl', help='Output STL file path')
    parser.add_argument('--use_existing_shell', type=bool, default=False, help='Whether to use the existing shell file')
    
    parser.add_argument('--pkl_result_path', type=str, default=project_dir+'/result/gold_lynel_20241201-134522_good/result_round1/robot_result.pkl', help='Pickle file path for the tenon position results')
    parser.add_argument('--tenon_file_folder', type=str, default=project_dir+'/metamaterial_filling/tenon', help='Folder for the tenon files')
                        
    parser.add_argument('--preview', type=bool, default=True, help='Whether to visualize the transformed tenons and the link')

    args = parser.parse_args()

    run_metamaterial_filling_for_stl_file(args.input_stl_path, args.unit, args.relative_density, args.shell_thickness, args.shell_generation_voxel_resolution, args.plate_interval, args.biased_tenon_distance, args.output_stl_name, args.use_existing_shell, args.pkl_result_path, args.tenon_file_folder, args.preview)
