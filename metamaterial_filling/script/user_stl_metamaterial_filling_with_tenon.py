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

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir + '/script/metamaterial')
sys.path.append(os.path.join(project_dir, '../auto_design/modules'))
sys.path.append(os.path.join(project_dir, '../auto_design'))

from metamaterial.sixFoldPlatesFillingWithShellTenon import SixFoldPlatesFillingWithShellTenon
from metamaterial.generateMeshFromPoints import generate_mesh_from_points
import time
import numpy as np
import math
from format_transform.off_to_stl import off_to_stl

from visualization.assemble_vis import load_and_transform_stl, get_rotation_matrix, visualize_meshes


'''
@Description: Transform the tenon and save the transformed tenon file
@Input:
    link: LinkResult object
    tenon_id: The id of the tenon
    tenon_file_folder: The folder for the tenon files
@Output:
    transformed_file_save_path: The path of the saved transformed tenon file
'''
def transform_tenon_and_save(link, tenon_id=0, unit='m', tenon_file_folder=project_dir+'/tenon'):
    # Set the transformation matrix for the tenon according to the tenon position
    tenon_basic_transformation_matrix = np.array([[1, 0, 0, 0.0],
                                            [0, 0, -1, 0.0],
                                            [0, 1, 0, 0.0],
                                            [0, 0, 0, 1]])

    tenon_ori_direction_vector = np.array([0, -1, 0])
    tenon_direction_vector = np.array([link.tenon_pos[tenon_id][3], link.tenon_pos[tenon_id][4], link.tenon_pos[tenon_id][5]])

    rotation_matrix = get_rotation_matrix(tenon_ori_direction_vector, tenon_direction_vector)

    transformation = np.array([[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], link.tenon_pos[tenon_id][0]],
                                            [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], link.tenon_pos[tenon_id][1]],
                                            [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], link.tenon_pos[tenon_id][2]],
                                            [0, 0, 0, 1]])

    if unit == 'm':
        print("Unit is in meter, scale the transformation matrix to mm")
        transformation[:, 3] = transformation[:, 3] * 1000
        transformation[3, 3] = 1


    tenon_transformation_matrix = np.dot(transformation, tenon_basic_transformation_matrix)

    # Load the tenon file
    tenon_file_name = 'connection_' + link.tenon_type[tenon_id] + '.stl'
    transformed_file_save_folder = os.path.join(project_dir, 'data/transformed_tenon')
    if not os.path.exists(transformed_file_save_folder):
        os.makedirs(transformed_file_save_folder)

    transformed_file_save_path = os.path.join(transformed_file_save_folder, tenon_file_name)

    load_and_transform_stl(os.path.join(tenon_file_folder, tenon_file_name), tenon_transformation_matrix, scale=1.0, save_path=transformed_file_save_path)

    return transformed_file_save_path


'''
@Description: Main function to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stl_path', type=str, default='data/../../urdf/lynel20240816-111233/FR_UP.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--relative_density', type=float, default=0.05, help='Relative density of the metamaterial given by FEA results')
    parser.add_argument('--shell_thickness', type=float, default=1, help='Thickness of the shell. mm')
    parser.add_argument('--shell_generation_voxel_resolution', type=float, default=1, help='Voxel resolution for shell generation. mm')
    parser.add_argument('--output_stl_name', type=str, default='FR_UP_final_output_with_shell.stl', help='Output STL file path')
    parser.add_argument('--use_existing_shell', type=bool, default=False, help='Whether to use the existing shell file')
    
    parser.add_argument('--pkl_result_path', type=str, default=project_dir+'/../auto_design/results/lynel_robot_result.pkl', help='Pickle file path for the tenon position results')
    parser.add_argument('--tenon_file_folder', type=str, default=project_dir+'/tenon', help='Folder for the tenon files')
                        
    parser.add_argument('--preview', type=bool, default=True, help='Whether to visualize the transformed tenons and the link')

    args = parser.parse_args()

    # Check if the input STL file exists
    if not os.path.exists(args.input_stl_path):
        raise FileNotFoundError(f'Input STL file not found at {args.input_stl_path}')
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cmake_build_dir = os.path.join(current_dir, '../build')
    output_folder = os.path.join(current_dir, '../data/output')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    relative_density = args.relative_density
    if relative_density < 0.01:
        relative_density = 0.01
        print('Relative density should be larger than 0.01, set to 0.01')
    elif relative_density >= 1.0:
        raise ValueError('Relative density should be less than 1.0')
    
    # Check and read pkl file
    robot_result = pkl.load(open(args.pkl_result_path, 'rb'))

    # for link_name in robot_result.link_dict:
    #     print("Link name: ", link_name)
    #     print("Link Tenon Positions: ", robot_result.link_dict[link_name].tenon_pos)
    #     print("Link Torques: ", robot_result.link_dict[link_name].applied_torque)
    #     print("Link tenon_type: ", robot_result.link_dict[link_name].tenon_type)

    link_name = args.input_stl_path.split('/')[-1].split('.')[0]
    link = robot_result.link_dict[link_name]

    

    ###### Replace and scale the stl ######
    replaced_stl_name = args.input_stl_path.split('/')[-1].split('.')[0] + '_replaced.stl'
    replaced_stl_save_path = os.path.join(output_folder, replaced_stl_name)
    replaced_stl_size_csv = replaced_stl_save_path.replace('.stl', '.csv')

    input_stl_path_full_path = os.path.join(current_dir, '../', args.input_stl_path)

    print(f"input_stl_path_full_path: {input_stl_path_full_path}")
    print(f"replaced_stl_save_path: {replaced_stl_save_path}")

    # Remove the replaced file if it exists
    if os.path.exists(replaced_stl_save_path):
        os.remove(replaced_stl_save_path)
    if os.path.exists(replaced_stl_size_csv):
        os.remove(replaced_stl_size_csv)

    print("Generating the replaced model...")
    if args.unit == 'm':
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

        print(f"min_x: {min_x} mm, min_y: {min_y} mm, min_z: {min_z} mm")
        print(f"max_x: {max_x} mm, max_y: {max_y} mm, max_z: {max_z} mm")
        print(f"applied_scale: {applied_scale}")
        print(f"applied_placement: {applied_placement}")
        print(f"applied_rotation: {applied_rotation}")

    
    ##### Transform tenons and save as stl files for later use #####
    print("Transforming tenons...")
    transformed_tenon_files = []                                        
    for i in range(len(link.tenon_pos)):
        file = transform_tenon_and_save(link, i, unit=args.unit, tenon_file_folder=args.tenon_file_folder)
        transformed_tenon_files.append(file)
        print(f"Transformed tenon file saved at {file}")

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
        transformed_file_save_path = transformed_tenon_file_path.replace('.stl', '_second_transformed.stl')

        load_and_transform_stl(transformed_tenon_file_path, second_transformation_matrix, scale=1.0, save_path=transformed_file_save_path)
        load_and_transform_stl(transformed_file_save_path, third_transformation_matrix, scale=1.0, save_path=transformed_file_save_path)
        final_transformed_tenon_files.append(transformed_file_save_path)

    ##### Preview the transformed tenons and the link  #####
    if args.preview:
        stl_to_visualize = [replaced_stl_save_path] + final_transformed_tenon_files
        eye_transformation_matrix = np.eye(4)
        transformation_matrices_vis = [eye_transformation_matrix]
        scales_vis = [1.0]
        for i in range (len(final_transformed_tenon_files)):
            transformation_matrices_vis.append(eye_transformation_matrix)
            scales_vis.append(1.0)

        visualize_meshes(stl_to_visualize, transformation_matrices_vis, scales_vis)


    ###### Generate the shell ######
    # smaller_model_stl_name = args.input_stl_path.split('/')[-1].split('.')[0] + '_smaller.stl'
    smaller_model_stl_name = replaced_stl_name.replace('.stl', '_smaller.stl')
    smaller_stl_save_path = os.path.join(output_folder, smaller_model_stl_name)
    smaller_stl_points_bin = smaller_stl_save_path.replace('.stl', '_points.bin')

    # Remove the smaller_model for shell file if it exists and we are not using the existing model. For quick testing.
    do_smaller_generation = True
    if os.path.exists(smaller_stl_save_path):
        if args.use_existing_shell:
            print(f"Using existing shell file at {smaller_stl_save_path}")
            do_smaller_generation = False
        else:
            os.remove(smaller_stl_points_bin)
            os.remove(smaller_stl_save_path)

    if do_smaller_generation:
        print("Generating the smaller model for shell...")
        # Use gnome-terminal to run the command
        shell_thickness = args.shell_thickness
        shell_generation_voxel_resolution = args.shell_generation_voxel_resolution
        subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./innerPointsCalculation {replaced_stl_save_path} {smaller_stl_points_bin} {shell_thickness} {shell_generation_voxel_resolution}']) #; exec bash

        # Wait for the process to finish
        while not os.path.exists(smaller_stl_points_bin):
            time.sleep(1)

        time.sleep(3)
        
        print(f"smaller model point bin file generated at {smaller_stl_points_bin}")

        # Do marching cubes to generate the smaller model
        generate_mesh_from_points(smaller_stl_points_bin, shell_generation_voxel_resolution, smaller_stl_save_path)

        print(f"smaller model generated at {smaller_stl_save_path}")


    ########## Generate the final model ##########
    tilt_angle = 30 # degrees
    out_stl_name = args.output_stl_name.split('.')[0] + '_' + str(args.shell_thickness) + 'mm_density_' + str(relative_density) + '' + str(tilt_angle) + '.stl'

    final_output_stl_path = os.path.join(output_folder, out_stl_name)

    # Define the dimensions of the board
    safe_scale = 1.5 # Safe scale to ensure the rotated plates can cover the whole model
    height = (max_z - min_z) * safe_scale
    delf_x = max_x - min_x
    delf_y = max_y - min_y
    width = max(delf_x, delf_y) * safe_scale

    interval = 10 # mm. The interval between plates
    thickness = relative_density / 6.0 * interval

    # Check if thickness is smaller than 0.2
    if thickness < 0.2:
        thickness = 0.2
        print('Thickness should be larger than 0.2, set to 0.2')
        # Correct the interval
        interval = thickness * 6.0 / relative_density
        print(f"Corrected interval: {interval}")
    
    plates_num = int(width / (thickness + interval) / 2)


    # Generate the final model
    print("Generating the final model with Openscad... Parameters:")
    print(f"height: {height}, width: {width}, thickness: {thickness}, interval: {interval}, plates_num: {plates_num}, tilt_angle: {tilt_angle}")
    print("This step usually takes more than 15 minutes. Please be patient.")

    sixFoldPlatesFilling = SixFoldPlatesFillingWithShellTenon(height, width, thickness, interval, plates_num, tilt_angle, replaced_stl_save_path, smaller_stl_save_path, final_transformed_tenon_files, final_output_stl_path)
    sixFoldPlatesFilling.generate_model()

    print(f"Final model generated at {final_output_stl_path}")

    print("All done! When you do 3d FDM printing, please make sure the model is placed in the generated orientation.")

