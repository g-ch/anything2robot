'''
@Author: Clarence
@Date: 2024-7-5
@Description: This script is used to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''

import os
import argparse
import subprocess
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir + '/script/metamaterial')

from metamaterial.sixFoldPlatesFillingWithShell import SixFoldPlatesFillingWithShell
from metamaterial.generateMeshFromPoints import generate_mesh_from_points
import time
import numpy as np
import math
from format_transform.off_to_stl import off_to_stl


'''
@Description: Main function to generate the final metamaterial model with shell for the input STL file given the relative density from FEA results.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stl_path', type=str, default='data/lynel/20240721/tmp/FR_UP.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--relative_density', type=float, default=0.05, help='Relative density of the metamaterial given by FEA results')
    parser.add_argument('--shell_thickness', type=float, default=1, help='Thickness of the shell. mm')
    parser.add_argument('--shell_generation_voxel_resolution', type=float, default=1, help='Voxel resolution for shell generation. mm')
    parser.add_argument('--output_stl_name', type=str, default='FR_UP_final_output_with_shell.stl', help='Output STL file path')
    parser.add_argument('--use_existing_shell', type=bool, default=False, help='Whether to use the existing shell file')

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

    ###### Replace and scale the stl if necessary ######
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
    with open(replaced_stl_size_csv, 'r') as f:
        lines = f.readlines()
        min_x, min_y, min_z = lines[0].split(',')
        max_x, max_y, max_z = lines[1].split(',')
        min_x, min_y, min_z = float(min_x), float(min_y), float(min_z)
        max_x, max_y, max_z = float(max_x), float(max_y), float(max_z)

        print(f"min_x: {min_x} mm, min_y: {min_y} mm, min_z: {min_z} mm")
        print(f"max_x: {max_x} mm, max_y: {max_y} mm, max_z: {max_z} mm")
    

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

    print("Generating the final model with Openscad... Parameters:")
    print(f"height: {height}, width: {width}, thickness: {thickness}, interval: {interval}, plates_num: {plates_num}, tilt_angle: {tilt_angle}")
    print("This step usually takes more than 15 minutes. Please be patient.")

    sixFoldPlatesFilling = SixFoldPlatesFillingWithShell(height, width, thickness, interval, plates_num, tilt_angle, replaced_stl_save_path, smaller_stl_save_path, final_output_stl_path)
    sixFoldPlatesFilling.generate_model()

    print(f"Final model generated at {final_output_stl_path}")

    print("All done! When you do 3d FDM printing, please make sure the model is placed in the generated orientation.")

